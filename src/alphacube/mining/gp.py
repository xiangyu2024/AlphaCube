"""Genetic-programming miner over the three-axis vocabulary.

Per generation the whole population is evaluated as ONE engine call (CSE
across trees, one pass over the cube per chunk phase), scored by |IC|
against the label, then bred: elitism, tournament selection, type-safe
subtree crossover, subtree/window/constant mutation. Panels of surviving
trees are cached by structural key so elites and unchanged parents cost
nothing next generation. At the end the archive's top-k is offered to the
pool (which re-derives IC on the same panels, so the fitness gate and the
pool gate cannot disagree).

Selection is WITHIN structural clusters (``clustering.cluster_population``
+ ``cull_clusters``): a parent pick draws a surviving cluster uniformly, then
runs a tournament inside it, so every structural family that survived
culling keeps a chance to reproduce. There is no numpy fallback path (one
engine).
"""

from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..catalog import Registry
from ..engine.engine import CubeEngine
from ..expression import Expression, is_valid_root
from ..metrics import batched_ic
from ..store import CubeView
from .clustering import cluster_population, cull_clusters
from .generator import TypedTreeGenerator
from .pool import LinearPool

ProgressFn = Callable[[int, int, dict], None]


@dataclass
class MiningReport:
    generations: int
    evaluated: int
    archive_size: int
    added: List[Expression]
    commit_reasons: Dict[str, int] = field(default_factory=dict)
    commit_examples: Dict[str, str] = field(default_factory=dict)
    best_fitness: float = float("-inf")
    seconds: float = 0.0
    engine_seconds: float = 0.0


class GPMiner:
    def __init__(self, registry: Registry, view: CubeView, target: np.ndarray, pool: LinearPool,
                 population_size: int = 24, tournament_size: int = 3, crossover_rate: float = 0.7,
                 mutation_rate: float = 0.3, elitism: int = 1, min_depth: int = 2, max_depth: int = 4,
                 top_k: Optional[int] = None, fitness_signed: bool = False, min_coverage: float = 0.5,
                 cull_rel_threshold: float = 0.5, merge_threshold: float = 0.8, panel_cache_items: int = 512,
                 device: str = "cpu", chunk_symbols: Optional[int] = None, chunk_days: Optional[int] = None,
                 fields: Optional[Sequence[str]] = None, allow_boolean_ops: bool = False,
                 allow_resource_risk: bool = False, skip_kernels: "frozenset[str]" = frozenset(),
                 checkpoint_dir: Optional[os.PathLike] = None,
                 checkpoint_every: int = 0,
                 workdir: Optional[os.PathLike] = None, seed: int = 0) -> None:
        self.registry = registry
        self.view = view
        self.pool = pool
        self.population_size = max(2, int(population_size))
        self.tournament_size = max(1, int(tournament_size))
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = float(mutation_rate)
        self.elitism = max(0, int(elitism))
        self.top_k = top_k
        self.cull_rel_threshold, self.merge_threshold = float(cull_rel_threshold), float(merge_threshold)
        self.fitness_signed = bool(fitness_signed)
        # A factor finite on few stock-days gets a high |IC| from a handful of lucky days;
        # score it -inf below this share of the label's finite cells (the pool applies the
        # same threshold, so the fitness ranking and the commit gate agree).
        self.min_coverage = float(min_coverage)
        self._rng = random.Random(seed)
        # The archive lives in memory and only reaches disk in _commit(), i.e. when the
        # whole budget is spent. On a full-market run that is many hours, and a run
        # interrupted before then leaves nothing at all -- no expressions, no scores.
        # checkpoint_every writes the ranked archive after every Nth generation so the
        # work so far is always recoverable.
        # Kernels the search may BREED but must never EVALUATE. This is not the same
        # as leaving them out of the vocabulary, and the difference matters when you
        # want to reproduce an earlier run: the generator draws from the vocabulary
        # with the miner's RNG, so removing an operator shifts every later draw and
        # the whole trajectory diverges from generation 1. Keeping the vocabulary
        # intact and refusing the tree at evaluation time instead leaves the draws
        # untouched, so every generation up to the first refusal is reproduced
        # exactly, and only from there on does selection differ.
        self.skip_kernels = frozenset(skip_kernels)
        self.skipped = 0
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.checkpoint_every = max(0, int(checkpoint_every))
        if self.checkpoint_dir and self.checkpoint_every:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.generator = TypedTreeGenerator(registry, fields=fields, n_minutes=view.n_minutes,
                                            min_depth=min_depth, max_depth=max_depth,
                                            allow_boolean_ops=allow_boolean_ops,
                                            allow_resource_risk=allow_resource_risk, rng=self._rng)
        self.engine = CubeEngine(view, device=device, chunk_symbols=chunk_symbols, chunk_days=chunk_days,
                                 workdir=workdir)
        self.target = np.asarray(target, dtype=np.float64)
        self._target_t = torch.from_numpy(np.ascontiguousarray(self.target)).float()
        self._target_finite = np.isfinite(self.target)
        self._target_cells = max(int(self._target_finite.sum()), 1)
        self._panel_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._panel_cache_items = int(panel_cache_items)
        self._archive: Dict[str, Tuple[Expression, float]] = {}
        self.engine_seconds = 0.0
        self.evaluated = 0

    # -- fitness ---------------------------------------------------------------------
    def _refused(self, expr: Expression) -> bool:
        if not self.skip_kernels:
            return False
        return any(n.spec is not None and n.spec.name in self.skip_kernels for n in expr.nodes())

    def _panels(self, population: Sequence[Expression]) -> List[Optional[np.ndarray]]:
        out: List[Optional[np.ndarray]] = [None] * len(population)
        todo: Dict[str, List[int]] = {}
        for i, e in enumerate(population):
            if self._refused(e):          # bred, but never handed to the engine
                self.skipped += 1
                continue
            k = e.key()
            if k in self._panel_cache:
                out[i] = self._panel_cache[k]
                self._panel_cache.move_to_end(k)
            else:
                todo.setdefault(k, []).append(i)
        if todo:
            exprs = [population[idx[0]] for idx in todo.values()]
            res = self.engine.evaluate(exprs, tolerant=True)
            self.engine_seconds += res.seconds
            self.evaluated += len(exprs)
            for (k, idx), p in zip(todo.items(), res.panels):
                if p is None:
                    continue
                self._panel_cache[k] = p
                for i in idx:
                    out[i] = p
            while len(self._panel_cache) > self._panel_cache_items:
                self._panel_cache.popitem(last=False)
        return out

    def fitness(self, population: Sequence[Expression]) -> np.ndarray:
        panels = self._panels(population)
        fit = np.full(len(population), -np.inf)
        live = [i for i, p in enumerate(panels) if p is not None]
        if live:
            tensors = [torch.from_numpy(panels[i]).float() for i in live]
            ics = batched_ic(tensors, self._target_t).numpy().astype(float)
            for i, ic in zip(live, ics):
                if not np.isfinite(ic):
                    continue
                cov = float((np.isfinite(panels[i]) & self._target_finite).sum() / self._target_cells)
                if cov < self.min_coverage:
                    continue
                fit[i] = ic if self.fitness_signed else abs(ic)
        return fit

    def _archive_update(self, population, fit) -> None:
        for e, f in zip(population, fit):
            if not np.isfinite(f):
                continue
            k = e.key()
            prev = self._archive.get(k)
            if prev is None or f > prev[1]:
                self._archive[k] = (e, float(f))

    # -- evolution -------------------------------------------------------------------
    def _select(self, population, fit, clusters) -> Expression:
        """Within-cluster tournament: cluster drawn uniformly, best-of-k inside it."""
        cluster = clusters[self._rng.randrange(len(clusters))]
        idx = self._rng.sample(list(cluster), min(self.tournament_size, len(cluster)))
        return population[max(idx, key=lambda i: fit[i])]

    def _mutate(self, expr: Expression) -> Expression:
        kinds = ["subtree", "window", "constant"]
        self._rng.shuffle(kinds)
        for k in kinds:
            child = {"subtree": self.generator.mutate_subtree, "window": self.generator.mutate_window,
                     "constant": self.generator.mutate_constant}[k](expr)
            if child is not None:
                return child
        return self.generator.generate()

    def _next_generation(self, population: List[Expression], fit: np.ndarray) -> List[Expression]:
        nxt: List[Expression] = []
        if self.elitism:
            for i in np.argsort(fit)[::-1][: self.elitism]:
                if np.isfinite(fit[i]):
                    nxt.append(population[int(i)])
        clusters = cull_clusters(cluster_population(population, self.merge_threshold), fit, self.cull_rel_threshold)
        survivors = [[i for i in c if np.isfinite(fit[i])] for c in clusters]
        survivors = [c for c in survivors if c]
        if not survivors:
            return self.generator.generate_population(self.population_size)
        while len(nxt) < self.population_size:
            if self._rng.random() < self.crossover_rate:
                a, b = self._select(population, fit, survivors), self._select(population, fit, survivors)
                child = self.generator.crossover(a, b) or a
            else:
                child = self._select(population, fit, survivors)
            if self._rng.random() < self.mutation_rate:
                child = self._mutate(child)
            if not is_valid_root(child):
                child = self.generator.generate()
            nxt.append(child)
        return nxt

    # -- entry -----------------------------------------------------------------------
    def mine(self, budget: int, on_generation: Optional[ProgressFn] = None) -> MiningReport:
        t0 = time.perf_counter()
        n_gen = max(1, math.ceil(budget / self.population_size))
        population = self.generator.generate_population(self.population_size)
        fit = self.fitness(population)
        self._archive_update(population, fit)
        best = float(np.max(fit)) if np.isfinite(fit).any() else float("-inf")
        for g in range(n_gen):
            population = self._next_generation(population, fit)
            fit = self.fitness(population)
            self._archive_update(population, fit)
            finite = fit[np.isfinite(fit)]
            if finite.size:
                best = max(best, float(finite.max()))
            if on_generation is not None:
                on_generation(g + 1, n_gen, {"best": best, "gen_best": float(finite.max()) if finite.size else float("nan"),
                                             "gen_mean": float(finite.mean()) if finite.size else float("nan"),
                                             "archive": len(self._archive), "valid": int(finite.size)})
            if self.checkpoint_every and self.checkpoint_dir and (g + 1) % self.checkpoint_every == 0:
                self.write_checkpoint(g + 1)
        report = self._commit()
        report.generations = n_gen
        report.evaluated = self.evaluated
        report.archive_size = len(self._archive)
        report.best_fitness = best
        report.seconds = time.perf_counter() - t0
        report.engine_seconds = self.engine_seconds
        return report

    def _commit(self) -> MiningReport:
        top_k = self.top_k if self.top_k is not None else 2 * self.pool.capacity
        ranked = sorted(self._archive.values(), key=lambda t: t[1], reverse=True)[: max(0, int(top_k))]
        reasons: Dict[str, int] = {}
        examples: Dict[str, str] = {}
        added: List[Expression] = []
        exprs = [e for e, _ in ranked]
        panels = self._panels(exprs) if exprs else []
        for e, p in zip(exprs, panels):
            if p is None:
                key = "panel unavailable"
            else:
                try:
                    self.pool.try_add(e, p.astype(np.float64))
                    added.append(e)
                    continue
                except Exception as exc:  # noqa: BLE001
                    key = f"{type(exc).__name__}: {str(exc)[:60]}"
            reasons[key] = reasons.get(key, 0) + 1
            examples.setdefault(key, repr(e)[:100])
        return MiningReport(generations=0, evaluated=0, archive_size=0, added=added,
                            commit_reasons=reasons, commit_examples=examples)

    def archive(self) -> List[Tuple[Expression, float]]:
        return sorted(self._archive.values(), key=lambda t: t[1], reverse=True)

    def write_checkpoint(self, generation: int) -> Optional[Path]:
        """Ranked archive -> ``archive_gen<N>.csv``, written atomically so a kill
        during the write cannot leave a half file. Fitness only; the pool metrics
        in members.csv need the panels, which is what _commit() is for."""
        if self.checkpoint_dir is None:
            return None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"archive_gen{generation:03d}.csv"
        tmp = path.with_suffix(".csv.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write("rank,fitness,expression\n")
            for i, (expr, fit) in enumerate(self.archive()):
                text = repr(expr).replace('"', '""')
                fh.write(f'{i},{fit:.6f},"{text}"\n')
        os.replace(tmp, path)
        return path


__all__ = ["GPMiner", "MiningReport"]
