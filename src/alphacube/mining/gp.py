"""Genetic-programming miner over the three-axis vocabulary.

Per generation the whole population is evaluated as ONE engine call (CSE
across trees, one pass over the cube per chunk phase), scored against the label
-- |IC| by default, or |rank IC| (or |IC|) minus a charge for the turnover of the
decile book, see ``fitness_kind`` / ``turnover_penalty`` -- then bred: elitism,
tournament selection, type-safe
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

import re

import math
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..catalog import Registry
from ..engine.engine import CubeEngine
from ..expression import Expression, is_valid_root
from ..metrics import (MIN_PAIRS, _cs_corr_days_t, batched_ic, cs_pearson, nanmean_quiet, rank_ic_turnover,
                       torch_rank_rows)
from ..store import CubeView
from .clustering import cluster_population, cull_clusters
from .generator import TypedTreeGenerator, one_layer_keys
from .rankban import CompactRankIndex, ExemplarCache, RankSimilarityBan, view_cache_key
from .scoring import BOOK_TOP_FRAC as _SHARED_TOP_FRAC, PanelScorer
from .pool import LinearPool

ProgressFn = Callable[[int, int, dict], None]

FITNESS_KINDS = ("ic", "rank_ic")
# The book every report trades -- top/bottom decile, equal weight, gross 1
# (backtest_metrics' default) -- and so the one whose turnover the fitness charges.
BOOK_TOP_FRAC = _SHARED_TOP_FRAC
# Rank-ban exemplars per engine call (see GPMiner._add_exemplars).
EXEMPLAR_BATCH = 150


def cpu_quota(path: str = "/sys/fs/cgroup/cpu.max") -> Optional[int]:
    """CPUs this process may actually use: the cgroup v2 quota when one is set,
    else None. torch sizes its CPU pool from the host -- 104 threads on the
    208-core server -- while the container is granted 25 CPUs, and oversubscribed
    the fitness sort ran 4x slower (measured there: 1.75 s a panel at 104
    threads, 0.43 s at 24).

    ``ALPHACUBE_CPU_THREADS`` caps the answer further: two mining runs sharing the
    container each sized themselves to the full quota and fought over it."""
    cap = os.environ.get("ALPHACUBE_CPU_THREADS")
    cap_n = int(cap) if cap and cap.strip().isdigit() and int(cap) > 0 else None
    try:
        quota, period = open(path).read().split()[:2]
        if quota == "max":
            return cap_n
        n = max(1, min(os.cpu_count() or 1, int(quota) // int(period)))
        return min(n, cap_n) if cap_n else n
    except (OSError, ValueError):
        return cap_n


def rss_anon(path: str = "/proc/self/status") -> float:
    """Anonymous RSS in bytes. Plain RSS also counts the mmap'd cube's page cache,
    which is reclaimable and not what runs us out of room; anon is."""
    try:
        with open(path) as fh:
            for line in fh:
                if line.startswith("RssAnon:"):
                    return float(line.split()[1]) * 1024.0
    except OSError:
        pass
    return float("nan")


def cgroup_memory(root: str = "/sys/fs/cgroup") -> Tuple[float, float]:
    """(current, max) bytes of the container's memory cgroup; NaN/inf when unset.
    The limit the container is actually killed at -- 90 GiB here -- lives here,
    not in ``free``, which reports the 754 GB host."""
    def read(name: str) -> float:
        try:
            with open(f"{root}/{name}") as fh:
                raw = fh.read().strip()
            return float("inf") if raw == "max" else float(raw)
        except (OSError, ValueError):
            return float("nan")
    return read("memory.current"), read("memory.max")


@contextmanager
def torch_threads(n: Optional[int]):
    """Run the block with ``n`` torch CPU threads, then restore the pool size."""
    prev = torch.get_num_threads()
    if n and n != prev:
        torch.set_num_threads(n)
    try:
        yield
    finally:
        if torch.get_num_threads() != prev:
            torch.set_num_threads(prev)


@dataclass
class MiningReport:
    generations: int
    evaluated: int
    archive_size: int
    added: List[Expression]
    commit_reasons: Dict[str, int] = field(default_factory=dict)
    commit_examples: Dict[str, str] = field(default_factory=dict)
    best_fitness: float = float("-inf")
    similar_refused: int = 0
    dup_demoted: int = 0
    seconds: float = 0.0
    engine_seconds: float = 0.0


_BAN_KEY = re.compile(r"^O:([A-Za-z0-9_]+)\((.*)\)(\|w\d+)?$")


def check_ban_key(key: str, registry: Registry, fields: Sequence[str]) -> None:
    """``key`` must be a depth-1 structure as ``one_layer_keys`` writes it --
    ``O:<operator>(<leaf>,...)[|w<window>]`` over leaves ``F:<field>`` / ``C:<value>``,
    e.g. ``O:log(F:Amt)`` or ``O:nt_ts_argmax(F:High)|w3`` -- naming a registered
    operator and existing fields."""
    m = _BAN_KEY.match(key)
    if not m:
        raise ValueError(f"ban key {key!r} is not a depth-1 structure like 'O:log(F:Amt)'")
    name, args = m.group(1), m.group(2)
    if name not in registry:
        raise ValueError(f"ban key {key!r}: no operator named {name!r}")
    for a in args.split(","):
        if a.startswith("F:"):
            if a[2:] not in fields:
                raise ValueError(f"ban key {key!r}: no field named {a[2:]!r}")
        elif not a.startswith("C:"):
            raise ValueError(f"ban key {key!r}: operand {a!r} is not a leaf (F:<field> or C:<value>)")


class GPMiner:
    def __init__(self, registry: Registry, view: CubeView, target: np.ndarray, pool: LinearPool,
                 population_size: int = 24, tournament_size: int = 3, crossover_rate: float = 0.7,
                 mutation_rate: float = 0.3, elitism: int = 1, min_depth: int = 2, max_depth: int = 4,
                 top_k: Optional[int] = None, fitness_signed: bool = False, min_coverage: float = 0.5,
                 min_scored_days: float = 0.5,
                 cull_rel_threshold: float = 0.5, merge_threshold: float = 0.8, panel_cache_items: int = 512,
                 exemplar_cache: Optional[str] = None, exemplar_cache_gb: float = 12.0,
                 panel_cache_bytes: Optional[int] = 12 * 1024 ** 3,
                 device: str = "cpu", chunk_symbols: Optional[int] = None, chunk_days: Optional[int] = None,
                 fields: Optional[Sequence[str]] = None, allow_boolean_ops: bool = False,
                 allow_resource_risk: bool = False, skip_kernels: "frozenset[str]" = frozenset(),
                 rolling_block: int = 0, ban_top_n: int = 0, banned_substructures: Sequence[str] = (),
                 ban_rank_corr: float = 0.0, ban_exemplars: Sequence[Expression] = (),
                 fitness_kind: str = "ic", turnover_penalty: float = 0.0, archive_dedup: float = 0.0,
                 cost_per_side: float = 0.0,
                 on_block: Optional[Callable[[dict], None]] = None,
                 on_engine_progress: Optional[Callable[[dict], None]] = None,
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
        # Same gate as the pool: coverage says the cell is finite, this says the day
        # actually produced an IC. A panel that is identically zero is finite
        # everywhere and separates nothing, and its |IC| comes from the few days that
        # happened to vary -- which is how it climbs to rank 1. See LinearPool.
        self.min_scored_days = float(min_scored_days)
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
        # A generation is ONE engine call, so without this the log goes silent for
        # however long that call takes -- 58 minutes for a cold population on the
        # production cube, and there was no way to tell progress from a hang.
        # Rolling restart: every ``rolling_block`` generations, throw the population
        # away, ban the ``ban_top_n`` depth-1 primitives the block leaned on hardest,
        # and start again. A single long run converges onto one primitive and spends
        # the rest of its budget re-wrapping it -- 179 of 364 archived expressions
        # shared one. Banning what the last block exhausted is what makes the next
        # block search somewhere else instead of rediscovering the same thing.
        # 0 disables it and the miner behaves exactly as before.
        self.rolling_block = max(0, int(rolling_block))
        self.ban_top_n = max(0, int(ban_top_n))
        self.blocks: List[dict] = []
        # Called at each rolling boundary with the block's own findings AND
        # everything found so far, so the caller can score both. A block searched
        # under a different ban set than its predecessors, so "what did this block
        # add" and "what do all blocks together give" are genuinely two questions.
        self.on_block = on_block
        self._block_seen: set = set()
        self.on_engine_progress = on_engine_progress
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.checkpoint_every = max(0, int(checkpoint_every))
        if self.checkpoint_dir and self.checkpoint_every:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.generator = TypedTreeGenerator(registry, fields=fields, n_minutes=view.n_minutes,
                                            min_depth=min_depth, max_depth=max_depth,
                                            allow_boolean_ops=allow_boolean_ops,
                                            allow_resource_risk=allow_resource_risk, rng=self._rng)
        # Structures banned from the first generation on -- e.g. what an earlier run's
        # rolling blocks already exhausted -- on top of whatever this run bans itself.
        # A key that names nothing would ban nothing, silently, so each is checked.
        if banned_substructures:
            for key in banned_substructures:
                check_ban_key(key, registry, self.generator.fields)
            self.generator.ban(banned_substructures)
        self.engine = CubeEngine(view, device=device, chunk_symbols=chunk_symbols, chunk_days=chunk_days,
                                 workdir=workdir)
        self.target = np.asarray(target, dtype=np.float64)
        # Rank-similarity bans (0 = off): panels of banned families; a candidate
        # whose daily ranks correlate with one to ban_rank_corr is refused, however
        # it is written. See mining/rankban.py. Exemplars from an earlier run are
        # evaluated when mining starts; each rolling block adds its own.
        self._rank_ban = RankSimilarityBan(self.target.shape, ban_rank_corr) if ban_rank_corr else None
        self._pending_exemplars = list(ban_exemplars) if self._rank_ban is not None else []
        self._exemplar_cache = None
        if self._rank_ban is not None and exemplar_cache:
            self._exemplar_cache = ExemplarCache(
                exemplar_cache, view_cache_key(view, self._rank_ban.index.days.size,
                                               self._rank_ban.index.cols.size),
                max_bytes=int(exemplar_cache_gb * 1024 ** 3))
        self._rank_verdict: Dict[str, Tuple[int, bool]] = {}
        self._gen_similar = 0
        self.similar_refused = 0
        self.initial_exemplars = 0
        self._target_t = torch.from_numpy(np.ascontiguousarray(self.target)).float()
        self._target_finite = np.isfinite(self.target)
        self._target_cells = max(int(self._target_finite.sum()), 1)
        self._usable_days = max(int((self._target_finite.sum(axis=1) >= MIN_PAIRS).sum()), 1)
        # What the fitness is made of. "ic" without a turnover penalty is the original
        # |mean daily Pearson IC| path, untouched. Otherwise each new panel is sorted
        # once per day (metrics.rank_ic_turnover) for its daily rank IC and for the
        # turnover of the decile book the reports trade, and the fitness is
        #     |IC or rank IC| - turnover_penalty * turnover.
        # Why: on the 51 archive leaders with after-cost backtests (gpu4-gpu8, 15 bp a
        # side), |IC| ranked them against net Sharpe at Spearman 0.02 and turnover at
        # -0.85; |rank IC| - 0.14 x turnover at 0.96. 0.14 is the cost in rank-IC
        # units: gross annual return was ~5.5 x |rank IC| there, and 15 bp a side on a
        # daily turnover T costs 2 x 0.0015 x 252 x T = 0.756 T a year.
        # Rank IC because the book trades ranks: gpu9's block-2 leaders were
        # pow_2(kurtosis) factors with IC +0.028 and rank IC -0.015 -- a Pearson IC
        # carried by a few extreme values, and a decile book that lost 2.4 Sharpe.
        if fitness_kind not in FITNESS_KINDS:
            raise ValueError(f"fitness_kind must be one of {FITNESS_KINDS}, not {fitness_kind!r}")
        if turnover_penalty < 0:
            raise ValueError("turnover_penalty must be >= 0")
        self.fitness_kind = fitness_kind
        self.turnover_penalty = float(turnover_penalty)
        self._rich = fitness_kind != "ic" or self.turnover_penalty > 0
        self._scores: Dict[str, Dict[str, float]] = {}     # key -> ic, rank_ic, turnover, fit
        self._score_threads = cpu_quota()
        self._scorer: Optional[PanelScorer] = None
        if self._rich:
            # GP 和 RL 用同一个打分器(mining/scoring.py),免得两条搜索线各算各的目标
            self._scorer = PanelScorer(self.target, kind=fitness_kind,
                                       turnover_penalty=self.turnover_penalty,
                                       signed=self.fitness_signed, top_frac=BOOK_TOP_FRAC,
                                       min_coverage=self.min_coverage,
                                       min_scored_days=self.min_scored_days)
            self._target_t64 = self._scorer.target_t64
            self._target_ranks_t = self._scorer.target_ranks_t
        # Only for the checkpoint's after-cost column; the fitness charges costs
        # through turnover_penalty.
        self.cost_per_side = float(cost_per_side)
        # Archive dedup (0 = off): factors whose daily ranks correlate >= archive_dedup
        # are one family. The archive keeps a family's best (fitness, then the smaller
        # tree), and within a generation only a family's best keeps its fitness -- the
        # rest do not breed. gpu9's first block had its top 12 archive places taken by
        # rewrites of day_max(Ret); under a rank fitness a monotone rewrite scores
        # EXACTLY what the original does, so they would not even be ordered.
        if archive_dedup < 0 or archive_dedup > 1:
            raise ValueError("archive_dedup must be in [0, 1]")
        self.archive_dedup = float(archive_dedup)
        self._families = CompactRankIndex(self.target.shape) if self.archive_dedup > 0 else None
        self._slot_of: Dict[str, int] = {}          # archived key -> its row in _families
        self._absorbed: Dict[str, set] = {}         # archived key -> lesser family members seen
        self._member_layers: Dict[str, set] = {}    # archived key -> one_layer_keys of those members
        self._slot_key: List[str] = []              # row in _families -> archived key
        self._gen_dup = 0
        self.dup_demoted = 0
        self._panel_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._panel_cache_items = int(panel_cache_items)
        # An item cap does not know how big a panel is. One panel is D x S x 4 =
        # 40 MB on the production cube, so gpu14's --panel-cache 1024 was a 41 GB
        # cache inside a 90 GiB container, next to a 23 GB ban index; it died at a
        # block boundary, where the pool fit and four tree models land on top.
        # Whichever cap binds first wins.
        self._panel_cache_bytes = int(panel_cache_bytes) if panel_cache_bytes else 0
        self._cache_bytes = 0
        self._archive: Dict[str, Tuple[Expression, float]] = {}
        self.engine_seconds = 0.0
        self.evaluated = 0

    # -- fitness ---------------------------------------------------------------------
    def _refused(self, expr: Expression) -> bool:
        if not self.skip_kernels:
            return False
        return any(n.spec is not None and n.spec.name in self.skip_kernels for n in expr.nodes())

    def _panels(self, population: Sequence[Expression], cache: bool = True) -> List[Optional[np.ndarray]]:
        """Panels of ``population``, from the cache where resident. ``cache=False``
        leaves newly evaluated panels out of the cache (panels needed once)."""
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
            cb = None
            if self.on_engine_progress is not None:
                n_new, n_cached = len(exprs), len(population) - len(exprs)
                started = time.perf_counter()

                def cb(r_idx, n_rounds, done, total, _n=n_new, _c=n_cached, _t=started):
                    frac = (done / total) if total else 0.0
                    el = time.perf_counter() - _t
                    self.on_engine_progress({"trees": _n, "cached": _c, "round": r_idx + 1,
                                             "rounds": n_rounds, "done": done, "total": total,
                                             "frac": frac, "elapsed": el,
                                             "eta": (el / frac - el) if frac > 0.01 else float("nan")})
            res = self.engine.evaluate(exprs, tolerant=True, on_progress=cb)
            self.engine_seconds += res.seconds
            self.evaluated += len(exprs)
            for (k, idx), p in zip(todo.items(), res.panels):
                if p is None:
                    continue
                if cache:
                    self._cache_panel(k, p)
                for i in idx:
                    out[i] = p
            self._evict_panels()
        return out

    def _cache_panel(self, key: str, panel: np.ndarray) -> None:
        old = self._panel_cache.pop(key, None)
        if old is not None:
            self._cache_bytes -= old.nbytes
        self._panel_cache[key] = panel
        self._cache_bytes += panel.nbytes

    def _evict_panels(self) -> None:
        """LRU down to BOTH caps. Evicting a panel this generation still holds is
        harmless -- the list returned by ``_panels`` keeps it alive until the
        generation is scored, which is exactly how long it is needed."""
        while self._panel_cache and (
                len(self._panel_cache) > self._panel_cache_items
                or (self._panel_cache_bytes and self._cache_bytes > self._panel_cache_bytes)):
            _key, panel = self._panel_cache.popitem(last=False)
            self._cache_bytes -= panel.nbytes

    def _score(self, key: str, panel: np.ndarray) -> Dict[str, float]:
        """Fitness and what it is made of, for one panel -- cached by structural key:
        a panel is a pure function of its key, and the sort behind rank IC and
        turnover costs ~0.5 s on the production cube. Same gates as the IC path
        (and the pool): finite score, coverage, and the scored-days share taken from
        the Pearson IC series exactly as LinearPool takes it."""
        s = self._scores.get(key)
        if s is not None:
            return s
        assert self._scorer is not None, "_score 只在带换手惩罚/rank IC 的路径上调用"
        s = self._scores[key] = self._scorer.score(panel)
        return s

    def fitness(self, population: Sequence[Expression]) -> np.ndarray:
        panels = self._panels(population)
        fit = np.full(len(population), -np.inf)
        live = [i for i, p in enumerate(panels) if p is not None]
        if live and self._rich:
            with torch_threads(self._score_threads):
                for i in live:
                    fit[i] = self._score(population[i].key(), panels[i])["fit"]
        elif live:
            tensors = [torch.from_numpy(panels[i]).float() for i in live]
            ics = batched_ic(tensors, self._target_t).numpy().astype(float)
            for i, ic in zip(live, ics):
                if not np.isfinite(ic):
                    continue
                cov = float((np.isfinite(panels[i]) & self._target_finite).sum() / self._target_cells)
                if cov < self.min_coverage:
                    continue
                if self.min_scored_days > 0:
                    scored = np.isfinite(cs_pearson(panels[i].astype(float), self.target)).sum()
                    if scored / self._usable_days < self.min_scored_days:
                        continue
                fit[i] = ic if self.fitness_signed else abs(ic)
        if self._rank_ban is not None and len(self._rank_ban):
            for i in live:
                if not np.isfinite(fit[i]):
                    continue
                k = population[i].key()
                v = self._rank_verdict.get(k)
                if v is None or v[0] != self._rank_ban.version:
                    v = self._rank_verdict[k] = (self._rank_ban.version,
                                                 self._rank_ban.match(panels[i]) is not None)
                if v[1]:
                    fit[i] = -np.inf
                    self._gen_similar += 1
                    self.similar_refused += 1
        return fit

    def _add_exemplars(self, exprs: Sequence[Expression], batch: int = EXEMPLAR_BATCH) -> int:
        """Store the panels of ``exprs`` as rank-ban exemplars; how many were new.

        In batches, and without keeping the panels: an exemplar is needed once, to
        compute the rows the ban index keeps. gpu12 evaluated its 462 start-up
        exemplars in ONE engine call -- 31.9 of the GPU's 32.6 GB, where a CUDA OOM
        in the tolerant engine drops trees silently, here exemplars -- and then
        held all 462 panels (~18 GB) in the panel cache, where nothing read them
        again. Same exemplars in the same order either way."""
        if self._rank_ban is None or not exprs:
            return 0
        added = 0
        exprs = list(exprs)
        cache = self._exemplar_cache
        for lo in range(0, len(exprs), max(1, int(batch))):
            part = exprs[lo:lo + batch]
            # 先查缓存,只把没命中的送进引擎;插入仍按原顺序,所以去重留下的是同一批
            rows = [cache.get(repr(e)) if cache is not None else None for e in part]
            todo = [i for i, r in enumerate(rows) if r is None]
            if todo:
                panels = self._panels([part[i] for i in todo], cache=False)
                for i, panel in zip(todo, panels):
                    if panel is None:
                        continue
                    rows[i] = self._rank_ban.index.rows(panel)
                    if cache is not None:
                        cache.put(repr(part[i]), rows[i])
            for e, r in zip(part, rows):
                if r is not None and self._rank_ban.add_rows(e, r):
                    added += 1
        if cache is not None:
            cache.prune()
        return added

    def _roll(self, population, fit, generation: int) -> List[str]:
        """End of a block: score what it found, ban what it leaned on, start over."""
        from collections import Counter
        from .generator import one_layer_keys

        # What THIS block added, as opposed to what earlier blocks had already found.
        block = [(e, f) for k, (e, f) in self._archive.items() if k not in self._block_seen]
        block.sort(key=lambda t: t[1], reverse=True)
        cumulative = self.archive()
        self._block_seen = set(self._archive)

        counts: "Counter[str]" = Counter()
        for e, f in zip(population, fit):
            if np.isfinite(f):
                counts.update(one_layer_keys(e))
        banned = [k for k, _ in counts.most_common(self.ban_top_n)] if self.ban_top_n else []
        if banned:
            self.generator.ban(banned)
        # Every archived factor that leans on a newly banned structure becomes a
        # rank exemplar, so rewriting it some other way does not bring it back. With
        # families, "leans on" covers the family's members too: a family whose
        # representative is day_max(log(Amt)) and which also met
        # pow_05(pow_05(day_max(Amt))) is the family of day_max(Amt).
        exemplars = 0
        if banned and self._rank_ban is not None:
            hit = set(banned)
            exemplars = self._add_exemplars([e for e, _ in cumulative
                                             if (one_layer_keys(e) | self._member_layers.get(e.key(), set())) & hit])

        info = {"index": len(self.blocks) + 1, "generation": generation, "banned": banned,
                "counts": counts.most_common(10),
                "banned_total": len(self.generator.banned_substructures),
                "exemplars_added": exemplars,
                "exemplars_total": len(self._rank_ban) if self._rank_ban is not None else 0,
                "block_archive": block, "cumulative_archive": cumulative}
        self.blocks.append({k: v for k, v in info.items()
                            if k not in ("block_archive", "cumulative_archive")})
        if self.on_block is not None:
            self.on_block(info)
        return banned

    def _archive_update(self, population, fit) -> np.ndarray:
        """Archive this generation's scored trees and return the fitness selection
        should use -- the same array unless archive_dedup is on, when every family
        member but the family's best in this generation is demoted to -inf."""
        if self._families is not None:
            return self._archive_update_families(population, fit)
        for e, f in zip(population, fit):
            if not np.isfinite(f):
                continue
            k = e.key()
            prev = self._archive.get(k)
            if prev is None or f > prev[1]:
                self._archive[k] = (e, float(f))
        return fit

    def _family_rows(self, expr: Expression, key: str) -> np.ndarray:
        slot = self._slot_of.get(key)
        if slot is not None:
            return self._families._y[:, slot]
        p = self._panel_cache.get(key)
        if p is None:                              # evicted -- not expected right after fitness()
            p = self._panels([expr])[0]
        return self._families.rows(p)

    def _archive_update_families(self, population, fit) -> np.ndarray:
        """``_archive_update`` with factor families (see ``archive_dedup``).

        1. Within the generation, best first (fitness, then the smaller tree), each
           tree either starts a family or joins the first kept tree it ranks like;
           joiners drop to -inf so a family breeds through one member.
        2. Each kept tree meets the archive: a tree that ranks like an archived
           family replaces its representative if it is better, and is only counted
           as a member otherwise; a tree like none starts a new archived family.
        A kept tree that loses to an archived family keeps its fitness: it is the
        best its family has in this population, and the representative is not here
        to breed in its place."""
        fit = np.array(fit, dtype=float, copy=True)
        live = [i for i in range(len(population)) if np.isfinite(fit[i])]
        if not live:
            return fit
        size = {i: len(population[i].nodes()) for i in live}
        live.sort(key=lambda i: (-fit[i], size[i], i))
        keys = [population[i].key() for i in live]
        rows = np.stack([self._family_rows(population[i], k) for i, k in zip(live, keys)], axis=1)
        thr = self.archive_dedup
        c_gen = CompactRankIndex.pairwise(rows)
        kept: List[int] = []                        # positions in `live`
        leader: Dict[int, int] = {}                 # demoted position -> the kept one it joined
        for p in range(len(live)):
            near = [q for q in kept if c_gen[p, q] >= thr]
            if near:
                leader[p] = max(near, key=lambda q: c_gen[p, q])
                fit[live[p]] = -np.inf
                self._gen_dup += 1
                self.dup_demoted += 1
            else:
                kept.append(p)
        c_arch = (self._families.corr(rows[:, kept]) if len(self._families)
                  else np.full((len(kept), 0), np.nan))
        taken: set = set()                          # rows replaced this generation (now hold a kept tree)
        family: Dict[int, str] = {}                 # kept position -> archived key of its family
        for j, p in enumerate(kept):
            i, k = live[p], keys[p]
            if k in self._slot_of:                  # it IS a family's representative
                family[p] = k
                continue
            c = c_arch[j].copy()
            # a replaced row now holds a tree kept in this same generation, and kept
            # trees do not rank alike by construction
            c[list(taken)] = np.nan
            slot = int(np.nanargmax(c)) if np.isfinite(c).any() else -1
            if slot >= 0 and c[slot] >= thr:
                rep = self._slot_key[slot]
                rep_e, rep_f = self._archive[rep]
                if fit[i] > rep_f or (fit[i] == rep_f and size[i] < len(rep_e.nodes())):
                    del self._archive[rep], self._slot_of[rep]
                    self._absorbed[k] = (self._absorbed.pop(rep, set()) | {rep}) - {k}
                    self._member_layers[k] = self._member_layers.pop(rep, set()) | one_layer_keys(rep_e)
                    self._archive[k] = (population[i], float(fit[i]))
                    self._slot_of[k], self._slot_key[slot] = slot, k
                    self._families.set(slot, rows[:, p])
                    taken.add(slot)
                    family[p] = k
                else:
                    self._absorbed[rep].add(k)
                    self._member_layers[rep] |= one_layer_keys(population[i])
                    family[p] = rep
            else:
                self._slot_of[k] = self._families.add(rows[:, p])
                self._slot_key.append(k)
                self._archive[k] = (population[i], float(fit[i]))
                self._absorbed[k] = set()
                self._member_layers[k] = set()
                family[p] = k
        for p, q in leader.items():
            if keys[p] != family[q]:
                self._absorbed[family[q]].add(keys[p])
                self._member_layers[family[q]] |= one_layer_keys(population[live[p]])
        return fit

    # -- evolution -------------------------------------------------------------------
    def _select(self, population, fit, clusters) -> Expression:
        """Within-cluster tournament: cluster drawn uniformly, best-of-k inside it."""
        cluster = clusters[self._rng.randrange(len(clusters))]
        idx = self._rng.sample(list(cluster), min(self.tournament_size, len(cluster)))
        return population[max(idx, key=lambda i: fit[i])]

    def _mutate(self, expr: Expression) -> Expression:
        # Five kinds, not three: operator-swap refines a structure instead of
        # discarding it, and hoist is the only move that makes a tree SHORTER --
        # without it a population drifts toward deep wrappers around one primitive.
        kinds = ["subtree", "operator", "hoist", "window", "constant"]
        self._rng.shuffle(kinds)
        for k in kinds:
            child = {"subtree": self.generator.mutate_subtree, "operator": self.generator.mutate_operator,
                     "hoist": self.generator.mutate_hoist, "window": self.generator.mutate_window,
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
        if self._pending_exemplars:
            self.initial_exemplars = self._add_exemplars(self._pending_exemplars)
            self._pending_exemplars = []
        # raw: every scored tree; fit: what selection sees (with archive_dedup on, a
        # family's lesser members are -inf there). The block's ban counts use raw, as
        # they did before families existed -- they measure where the population's
        # mass went, rewrites included, which is exactly what a ban is for.
        population = self.generator.generate_population(self.population_size)
        raw = self.fitness(population)
        fit = self._archive_update(population, raw)
        best = float(np.max(fit)) if np.isfinite(fit).any() else float("-inf")
        for g in range(n_gen):
            population = self._next_generation(population, fit)
            self._gen_similar = self._gen_dup = 0
            raw = self.fitness(population)
            fit = self._archive_update(population, raw)
            finite = fit[np.isfinite(fit)]
            if finite.size:
                best = max(best, float(finite.max()))
            if on_generation is not None:
                on_generation(g + 1, n_gen, {"best": best, "gen_best": float(finite.max()) if finite.size else float("nan"),
                                             "gen_mean": float(finite.mean()) if finite.size else float("nan"),
                                             "archive": len(self._archive), "valid": int(finite.size),
                                             **self._similar_info()})
            if self.checkpoint_every and self.checkpoint_dir and (g + 1) % self.checkpoint_every == 0:
                self.write_checkpoint(g + 1)
            if self.rolling_block and (g + 1) % self.rolling_block == 0 and g + 1 < n_gen:
                banned = self._roll(population, raw, g + 1)
                population = self.generator.generate_population(self.population_size)
                self._gen_similar = self._gen_dup = 0
                raw = self.fitness(population)
                fit = self._archive_update(population, raw)
                # The rebuilt population is a real, evaluated generation: report it as
                # one. This line used to print placeholders -- "gen_best +nan ... valid 0"
                # -- which on gpu5 read as a whole block's first population failing,
                # while those same trees had just added 48 entries to the archive.
                finite = fit[np.isfinite(fit)]
                if finite.size:
                    best = max(best, float(finite.max()))
                if on_generation is not None:
                    on_generation(g + 1, n_gen, {"best": best,
                                                 "gen_best": float(finite.max()) if finite.size else float("nan"),
                                                 "gen_mean": float(finite.mean()) if finite.size else float("nan"),
                                                 "archive": len(self._archive),
                                                 "valid": int(finite.size), "rolled": len(banned),
                                                 **self._similar_info()})
        if self.rolling_block and self.on_block is not None and len(self._archive) > len(self._block_seen):
            self._roll(population, raw, n_gen)          # the final, unrolled block still gets scored
        report = self._commit()
        report.generations = n_gen
        report.evaluated = self.evaluated
        report.archive_size = len(self._archive)
        report.best_fitness = best
        report.similar_refused = self.similar_refused
        report.dup_demoted = self.dup_demoted
        report.seconds = time.perf_counter() - t0
        report.engine_seconds = self.engine_seconds
        return report

    def memory_report(self) -> Dict[str, float]:
        """Bytes in the three structures that grow with the search, plus the process
        and its cgroup. Printed every generation so the next silent death comes with
        a memory curve instead of a guess: gpu14 left none."""
        ban, fam = self._rank_ban, self._families
        cur, cap = cgroup_memory()
        return {"cache_items": float(len(self._panel_cache)), "cache_bytes": float(self._cache_bytes),
                "exemplars": float(len(ban)) if ban is not None else 0.0,
                "exemplar_bytes": float(ban.index.nbytes) if ban is not None else 0.0,
                "families": float(len(fam)) if fam is not None else 0.0,
                "family_bytes": float(fam.nbytes) if fam is not None else 0.0,
                "rss_anon": rss_anon(), "cgroup_current": cur, "cgroup_max": cap}

    def memory_line(self) -> str:
        r = self.memory_report()
        g = 1024.0 ** 3
        out = (f"rss {r['rss_anon'] / g:.1f}G cache {r['cache_bytes'] / g:.1f}G/{int(r['cache_items'])} "
               f"exem {r['exemplar_bytes'] / g:.2f}G/{int(r['exemplars'])} "
               f"fam {r['family_bytes'] / g:.2f}G/{int(r['families'])}")
        if np.isfinite(r["cgroup_current"]) and np.isfinite(r["cgroup_max"]):
            out += f" cg {r['cgroup_current'] / g:.0f}/{r['cgroup_max'] / g:.0f}G"
        return out

    def _similar_info(self) -> Dict[str, int]:
        """Rank-ban and family counters for the generation line (each only when on):
        ``similar`` refused by the rank ban, ``dup`` demoted as a lesser member of a
        family in the same generation -- and ``mem``, always."""
        out: Dict[str, int] = {}
        if self._rank_ban is not None:
            out.update(similar=self._gen_similar, exemplars=len(self._rank_ban))
        if self._families is not None:
            out["dup"] = self._gen_dup
        out["mem"] = self.memory_line()
        return out

    def _commit(self) -> MiningReport:
        # top_k None: twice the pool's capacity, or the whole archive when the pool
        # has none. 0 also means the whole archive (it used to commit nothing).
        top_k = self.top_k
        if top_k is None:
            top_k = 2 * self.pool.capacity if self.pool.capacity else 0
        ranked = sorted(self._archive.values(), key=lambda t: t[1], reverse=True)
        if top_k:
            ranked = ranked[: int(top_k)]
        exprs = [e for e, _ in ranked]
        panels = self._panels(exprs) if exprs else []
        adm = self.pool.admit(zip(exprs, panels))
        return MiningReport(generations=0, evaluated=0, archive_size=0, added=adm.added,
                            commit_reasons=adm.reasons, commit_examples=adm.examples)

    def archive(self) -> List[Tuple[Expression, float]]:
        return sorted(self._archive.values(), key=lambda t: t[1], reverse=True)

    def write_checkpoint(self, generation: int, metrics_top_n: int = 10) -> Optional[Path]:
        """Ranked archive -> ``archive_gen<N>.csv``, written atomically so a kill
        during the write cannot leave a half file.

        Every row carries what its fitness was made of when the fitness is not
        plain |IC| -- ``score_ic`` (Pearson) / ``score_rank_ic``, signed, and
        ``score_turnover`` of the decile book -- and, with archive_dedup on,
        ``dups``: how many other trees of its family (daily ranks alike to the
        threshold) the run has met. The top ``metrics_top_n`` rows also carry IC and
        return-based metrics when their panel is still in the cache -- free, because
        the panel is already in memory, and it is what makes a checkpoint readable
        mid-run rather than a list of expressions with one number each. Of those,
        ``sharpe`` / ``annualized_return`` / ``max_drawdown`` are no-cost and for
        the factor AS WRITTEN (IC -0.024 with sharpe -3.3 is a strong factor, traded
        the other way); ``sharpe_net`` is after ``cost_per_side`` and traded in its
        IC's direction -- the number to read. Deeper rows, and any row whose panel
        has been evicted, carry no metrics; members.csv at the end is where
        everything gets measured properly."""
        if self.checkpoint_dir is None:
            return None
        from ..metrics import backtest_metrics, cs_pearson, ic_summary

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"archive_gen{generation:03d}.csv"
        tmp = path.with_suffix(".csv.tmp")
        scores = ["score_ic", "score_rank_ic", "score_turnover", "dups"]
        cols = ["ic_mean", "icir", "rank_ic", "coverage", "n_days",
                "annualized_return", "sharpe", "max_drawdown", "turnover", "sharpe_net"]
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write("rank,fitness," + ",".join(scores + cols) + ",expression\n")
            for i, (expr, fit) in enumerate(self.archive()):
                k = expr.key()
                sc = self._scores.get(k, {})
                vals = {f"score_{n}": sc[n] for n in ("ic", "rank_ic", "turnover") if n in sc}
                if self._families is not None:
                    vals["dups"] = len(self._absorbed.get(k, ()))
                if i < metrics_top_n:
                    p = self._panel_cache.get(k)
                    if p is not None:
                        pan = np.asarray(p, dtype=np.float64)
                        ics = cs_pearson(pan, self.target)
                        vals.update(ic_summary(pan, self.target))
                        vals.update(backtest_metrics(ics, pan, self.target))
                        if self.cost_per_side > 0 and np.isfinite(vals.get("ic_mean", np.nan)):
                            sign = -1.0 if vals["ic_mean"] < 0 else 1.0
                            vals["sharpe_net"] = backtest_metrics(sign * ics, sign * pan, self.target,
                                                                  cost_per_side=self.cost_per_side)["sharpe_net"]
                cells = ",".join("" if c not in vals else (str(vals[c]) if c == "dups" else f"{vals[c]:.6f}")
                                 for c in scores + cols)
                text = repr(expr).replace('"', '""')
                fh.write(f'{i},{fit:.6f},{cells},"{text}"\n')
        os.replace(tmp, path)
        return path


__all__ = ["GPMiner", "MiningReport"]
