"""One-shot mining run: cube -> label -> pool -> miner -> metrics -> combination
leaderboard -> run directory. Used by ``cli mine``, ``train.py`` and
``scripts/compare_miners.py`` so all three share one code path."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

MINERS = ("gp", "alphagen", "alphaqcm", "alphaforge", "alphasage", "warmstart")
POOLS = ("mse", "meanstd", "dynamic")
LABELS = ("open_vwap15", "close", "volatility", "downside_volatility", "rank", "class", "spike", "crash")


@dataclass
class RunResult:
    run_dir: Optional[Path]
    pool_size: int
    added: int
    pool_metrics: Dict[str, float]
    members: List[dict]
    combination: Optional[Any]          # DataFrame or None
    seconds: float
    engine_seconds: float
    rejected: Dict[str, int]
    summary: str


class _Lab:
    def __init__(self, desc, compute, realized) -> None:
        self._desc, self._compute, self._realized = desc, compute, realized

    def describe(self) -> str:
        return self._desc

    def compute(self, view):
        return self._compute(view)

    def realized(self, view):
        return self._realized(view)


def make_label(kind: str, horizon: int, params: Optional[Dict[str, float]] = None) -> _Lab:
    """A label with ``compute(view) -> (D, S)`` and ``realized(view)`` (the P&L leg)."""
    from .label import Label

    if kind in ("open_vwap15", "close"):
        lab = Label(kind, horizon)
        return _Lab(lab.describe(), lab.compute, lab.compute)
    from .daily import daily_from_view
    from .metrics import daily_returns_from_close
    from .models import targets as T

    p = dict(params or {})

    def compute(view):
        r = daily_returns_from_close(daily_from_view(view, fields=["Close"], open_vwap15=False))
        if kind == "volatility":
            return T.forward_volatility(r, horizon, **p)
        if kind == "downside_volatility":
            return T.forward_downside_volatility(r, horizon, **p)
        if kind == "rank":
            return T.forward_return_rank(r, horizon, **p)
        if kind == "class":
            return T.forward_return_class(r, horizon, **p) - 1.0
        return T.forward_event_label(r, horizon, kind=kind, **p)

    return _Lab(f"{kind}(h={horizon}, {p})", compute, lambda v: Label("close", horizon).compute(v))


def build_pool(kind: str, capacity: int, target, realized, dedup: Optional[float] = 0.99,
               min_coverage: float = 0.5, meanstd_lr: float = 0.05, meanstd_steps: int = 200,
               dynamic_window: int = 60):
    from .mining.pool import DynamicRollingPool, MeanStdPool, MsePool

    kw = dict(dedup_threshold=dedup, min_coverage=min_coverage, realized_returns=realized)
    if kind == "mse":
        return MsePool(capacity, target, **kw)
    if kind == "meanstd":
        return MeanStdPool(capacity, target, lr=meanstd_lr, n_steps=meanstd_steps, **kw)
    if kind == "dynamic":
        return DynamicRollingPool(capacity, target, window=dynamic_window, **kw)
    raise ValueError(f"unknown pool {kind!r}; choose from {POOLS}")


def run_mining(
    store: Union[str, Path, Any],
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    max_symbols: Optional[int] = None,
    symbols: Optional[Sequence[str]] = None,
    warmup: int = 60,
    label: str = "open_vwap15",
    label_params: Optional[Dict[str, float]] = None,
    horizon: int = 1,
    miner: str = "gp",
    budget: int = 480,
    seed: int = 0,
    device: str = "cpu",
    chunk_symbols: Optional[int] = None,
    chunk_days: Optional[int] = None,
    pool: str = "mse",
    pool_capacity: int = 10,
    dedup: Optional[float] = 0.99,
    min_coverage: float = 0.5,
    meanstd_lr: float = 0.05,
    meanstd_steps: int = 200,
    dynamic_window: int = 60,
    seed_library: Optional[Union[str, Path]] = None,
    presqueeze: Optional[dict] = None,
    allow_boolean_ops: bool = False,
    allow_resource_risk: bool = False,
    skip_kernels: Sequence[str] = (),
    checkpoint_every: int = 0,
    engine_progress_every: float = 0.0,
    workdir: Optional[Union[str, Path]] = None,
    gp_params: Optional[dict] = None,
    rl_params: Optional[dict] = None,
    max_expr_len: int = 20,
    combine_methods: Sequence[str] = ("equal", "ic", "icir", "lstsq", "ridge", "gbdt"),
    combine_train_frac: float = 0.7,
    out_root: Optional[Union[str, Path]] = None,
    run_id: Optional[str] = None,
    save_values: bool = True,
    log: Optional[Callable[[str], None]] = print,
) -> RunResult:
    from .catalog import default_registry
    from .engine.engine import CubeEngine
    from .metrics import backtest_metrics, cs_pearson, ic_summary
    from .mining import (AlphaForgeMiner, AlphaGenMiner, AlphaQCMMiner, AlphaSAGEMiner, BatchRewarder, GPMiner,
                         PPOWarmStartQCMMiner, Vocabulary)
    from .research import warmed_view
    from .store import CubeStore, SegmentedCubeStore, open_cube_store

    say = log or (lambda s: None)
    t0 = time.time()
    if not isinstance(store, (CubeStore, SegmentedCubeStore)):
        store = open_cube_store(store)
    view, trim = warmed_view(store, start, end, warmup=warmup, symbols=symbols, max_symbols=max_symbols)
    lab = make_label(label, horizon, label_params)
    target, realized = lab.compute(view), lab.realized(view)
    say(f"store {getattr(store, 'directory', store)}: {view.dates[0]}..{view.dates[-1]} {view.n_days} days x "
        f"{view.n_symbols} symbols (warm-up {trim}); label {lab.describe()}; tradable {view.tradable().mean():.1%}")
    reg = default_registry()
    pool_obj = build_pool(pool, pool_capacity, target, realized, dedup, min_coverage, meanstd_lr, meanstd_steps,
                          dynamic_window)
    engine = CubeEngine(view, device=device, chunk_symbols=chunk_symbols, chunk_days=chunk_days,
                        workdir=workdir)
    if seed_library:
        from .legacy import load_seed_library, seed_pool
        exprs, errs = load_seed_library(seed_library, n_minutes=view.n_minutes)
        exprs = [e for _, e in exprs if e.shape.degenerate]
        rewards = seed_pool(pool_obj, exprs, engine)
        say(f"seeded pool with {len(rewards)} of {len(exprs)} library factors ({len(errs)} failed to parse)")
    if presqueeze is not None:
        from .mining.boosting import PresqueezeConfig, feature_boosting_presqueeze, seed_pool_from_presqueeze
        res = feature_boosting_presqueeze(reg, engine, target,
                                          PresqueezeConfig(allow_boolean_ops=allow_boolean_ops,
                                                          allow_resource_risk=allow_resource_risk, **presqueeze))
        rewards = seed_pool_from_presqueeze(pool_obj, res, engine)
        say(f"pre-squeeze: {res.candidates_generated_total} candidates, {res.candidates_evaluated_total} scored, "
            f"{len(res.survivors)} survivors, {len(rewards)} entered the pool")

    _last_tick = [0.0]

    def engine_progress(d):
        """Inside-a-generation progress. Throttled: the engine calls back per chunk,
        which on the production cube is hundreds of times per round."""
        now = time.time()
        if now - _last_tick[0] < engine_progress_every:
            return
        _last_tick[0] = now
        eta = d["eta"]
        eta_s = "?" if eta != eta else f"{eta / 60:.0f}m"
        say(f"      ... {d['trees']} trees ({d['cached']} cached)  round {d['round']}/{d['rounds']}"
            f"  {d['done']}/{d['total']} chunks  {100 * d['frac']:.0f}%"
            f"  elapsed {d['elapsed'] / 60:.0f}m  eta {eta_s}")

    def progress(n, b, s):
        extra = "  ".join(f"{k} {v:+.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in s.items())
        say(f"  {n:5d}/{b}  {extra}")

    ckpt_dir = None
    if out_root and checkpoint_every:
        ckpt_dir = Path(out_root) / (run_id or datetime.now().strftime("%Y%m%d_%H%M%S")) / "checkpoints"
    if miner == "gp":
        gp = dict(population_size=24, max_depth=4, min_depth=2, top_k=None)
        gp.update(gp_params or {})
        m = GPMiner(reg, view, target, pool_obj, seed=seed, device=device, min_coverage=min_coverage,
                    chunk_symbols=chunk_symbols, chunk_days=chunk_days,
                    allow_boolean_ops=allow_boolean_ops, allow_resource_risk=allow_resource_risk,
                    skip_kernels=frozenset(skip_kernels), checkpoint_dir=ckpt_dir,
                    on_engine_progress=(engine_progress if engine_progress_every > 0 else None),
                    checkpoint_every=checkpoint_every, workdir=workdir, **gp)
        rep = m.mine(budget, on_generation=lambda g, n, s: progress(g, n, s))
        added, evaluated, engine_s = rep.added, rep.evaluated, rep.engine_seconds
        summary = f"{rep.generations} generations, archive {rep.archive_size}, best |IC| {rep.best_fitness:+.4f}"
        rejected = rep.commit_reasons
        panels_of = m._panels
    else:
        if miner not in MINERS:
            raise ValueError(f"unknown miner {miner!r}; choose from {MINERS}")
        vocab = Vocabulary(reg, n_minutes=view.n_minutes, allow_boolean_ops=allow_boolean_ops,
                           allow_resource_risk=allow_resource_risk)
        rew = BatchRewarder(engine, pool_obj)
        cls = {"alphagen": AlphaGenMiner, "alphaqcm": AlphaQCMMiner, "alphaforge": AlphaForgeMiner,
               "alphasage": AlphaSAGEMiner, "warmstart": PPOWarmStartQCMMiner}[miner]
        m = cls(vocab, rew, max_expr_len=max_expr_len, seed=seed, **(rl_params or {}))
        rep = m.mine(budget, on_progress=progress)
        added, evaluated, engine_s = rep.added, rep.evaluated, rep.engine_seconds
        summary = f"{rep.episodes} episodes, {rep.completed} completed expressions, best reward {rep.best_reward:+.4f}"
        rejected = rep.rejected
        panels_of = rew.panels
    say(f"done {time.time() - t0:.1f}s (engine {engine_s:.1f}s): {summary}; {evaluated} evaluations; "
        f"added {len(added)}; pool {pool_obj.size}")
    for k, v in sorted(rejected.items(), key=lambda kv: -kv[1])[:8]:
        say(f"   x{v:<4d} {k}")
    if pool_obj.size == 0:
        say("pool is empty; nothing to report")
        return RunResult(None, 0, len(added), {}, [], None, time.time() - t0, engine_s, dict(rejected), summary)

    ens = pool_obj.test_ensemble()
    say("pool: " + "  ".join(f"{k} {v:+.4f}" for k, v in ens.items()))
    rows = []
    panels = panels_of(pool_obj.exprs())
    realized = np.asarray(pool_obj.realized, dtype=np.float64)[trim:]
    for i, (mem, p) in enumerate(zip(pool_obj.members, panels)):
        row = {"index": i, "expression": repr(mem.expr), "weight": float(pool_obj.weights[i])}
        if p is not None:
            pan = p[trim:].astype(float)
            row.update(ic_summary(pan, target[trim:]))
            # Return-based metrics per factor, not only for the pool and the
            # combinations: a high IC earned on a handful of days, or on a corner of
            # the universe, shows up here as a Sharpe that does not match. Same
            # function and same top_frac as the pool line, so the numbers are
            # directly comparable. Diagnostic only -- one holding period, no costs.
            bt = backtest_metrics(cs_pearson(pan, target[trim:]), pan, realized)
            row.update({k: bt[k] for k in ("annualized_return", "sharpe", "max_drawdown") if k in bt})
        rows.append(row)
        say(f"  [{i:2d}] w={row['weight']:+.4f} IC={row.get('ic_mean', float('nan')):+.4f}"
            f" sharpe={row.get('sharpe', float('nan')):+.2f}  {row['expression']}")

    table = None
    if combine_methods:
        from .combine.post_combine import compare, leaderboard
        results, signs = compare(pool_obj.z_columns(), target, realized, methods=tuple(combine_methods),
                                 train_frac=combine_train_frac, align=True, seed=seed)
        table = leaderboard(results)
        cols = [c for c in ("ic_mean", "icir", "rank_ic", "sharpe", "coverage", "error") if c in table.columns]
        say(f"combination (out-of-sample after {combine_train_frac:.0%} of days; {int((signs < 0).sum())} flipped):")
        say(table[cols].to_string(float_format=lambda v: f"{v:+.4f}"))

    run_dir = None
    if out_root:
        run_dir = Path(out_root) / (run_id or datetime.now().strftime("%Y%m%d_%H%M%S"))
        (run_dir / "values").mkdir(parents=True, exist_ok=True)
        import pandas as pd
        if save_values:
            for i, p in enumerate(panels):
                if p is not None:
                    np.save(run_dir / "values" / f"{i:03d}.npy", p[trim:].astype(np.float32))
        pd.DataFrame(rows).to_csv(run_dir / "members.csv", index=False)
        np.save(run_dir / "combined.npy", pool_obj.combined()[trim:].astype(np.float32))
        if table is not None:
            table.to_csv(run_dir / "combination.csv")
        (run_dir / "meta.json").write_text(json.dumps({
            "store": str(getattr(store, "directory", store)), "start": start, "end": end, "dates": view.dates[trim:],
            "symbols": view.symbols, "label": lab.describe(), "miner": miner, "pool": pool, "budget": budget,
            "seed": seed, "pool_metrics": {k: float(v) for k, v in ens.items()}, "rejected": dict(rejected),
            "seconds": round(time.time() - t0, 1), "engine_seconds": round(engine_s, 1),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        say(f"wrote {run_dir}")
    return RunResult(run_dir, pool_obj.size, len(added), {k: float(v) for k, v in ens.items()}, rows, table,
                     time.time() - t0, engine_s, dict(rejected), summary)


__all__ = ["run_mining", "RunResult", "make_label", "build_pool", "MINERS", "POOLS", "LABELS"]
