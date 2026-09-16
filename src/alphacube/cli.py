"""Command line: ``python -m alphacube.cli {synth,research,mine,seed} ...``

``mine`` is the one-shot entry the previous project's ``train.py`` provided:
read the cube -> label -> pool -> any miner -> metrics -> combination
leaderboard -> everything on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .pipeline import LABELS, MINERS, POOLS


def _parse_exprs(items: List[str], file: Optional[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if file:
        text = Path(file).read_text(encoding="utf-8")
        if file.endswith(".json"):
            out.update(json.loads(text))
        else:
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                name, _, src = line.partition("=")
                out[name.strip()] = src.strip()
    for item in items:
        name, _, src = item.partition("=")
        if not src:
            src, name = name, f"f{len(out)}"
        out[name.strip()] = src.strip()
    return out


def _kv(items: List[str]) -> Dict[str, float]:
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        try:
            out[k.strip()] = float(v)
        except ValueError:
            out[k.strip()] = v
    return out


def cmd_synth(a) -> int:
    from .synthetic import write_synthetic_store

    st = write_synthetic_store(a.out, n_symbols=a.symbols, n_days=a.days, n_minutes=a.minutes, seed=a.seed,
                               suspended_frac=a.suspended, plant_beta=a.plant)
    print(f"wrote {a.out}: {st.n_symbols} symbols x {st.n_days} days x {st.n_minutes} minutes, fields {st.fields}")
    return 0


def cmd_research(a) -> int:
    from .label import Label
    from .research import evaluate_factors
    from .store import open_cube_store

    exprs = _parse_exprs(a.expr, a.file)
    if a.legacy:
        from .legacy import load_legacy_factors
        exprs = {**load_legacy_factors(), **exprs}
    if not exprs:
        print("no expressions given (use --expr name=..., --file, or --legacy)", file=sys.stderr)
        return 2
    if a.label not in ("open_vwap15", "close"):
        print("research supports the return labels open_vwap15 / close", file=sys.stderr)
        return 2
    out = Path(a.out) / (a.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")) if a.out else None
    evaluate_factors(open_cube_store(a.store), exprs, start=a.start, end=a.end, max_symbols=a.max_symbols,
                     label=Label(a.label, a.horizon), device=a.device, out_dir=out,
                     chunk_symbols=a.chunk_symbols, chunk_days=a.chunk_days)
    return 0


def cmd_calculate(a) -> int:
    from .calculate import export_run_to_parquet, run_factor_calculation

    exprs = _parse_exprs(a.expr, a.file)
    if a.legacy:
        from .legacy import load_legacy_factors
        exprs = {**load_legacy_factors(), **exprs}
    if not exprs:
        print("no expressions given", file=sys.stderr)
        return 2
    run_id = a.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_factor_calculation(exprs, a.store, save_root=a.out or "factor_runs", run_id=run_id, start=a.start, end=a.end,
                           max_symbols=a.max_symbols, factor_batch_size=a.batch_size, device=a.device,
                           chunk_symbols=a.chunk_symbols, chunk_days=a.chunk_days,
                           defer_resource_risk=not a.allow_resource_risk, resume=not a.no_resume)
    if a.parquet:
        info = export_run_to_parquet(Path(a.out or "factor_runs") / run_id, a.parquet, store=a.store)
        print(f"parquet: {info}")
    return 0


def cmd_mine(a) -> int:
    from .pipeline import run_mining

    gp = dict(population_size=a.population, max_depth=a.max_depth, min_depth=a.min_depth, top_k=a.top_k,
              panel_cache_items=a.panel_cache)
    res = run_mining(
        a.store, start=a.start, end=a.end, max_symbols=a.max_symbols, warmup=a.warmup, label=a.label,
        label_params=_kv(a.label_param), horizon=a.horizon, miner=a.miner, budget=a.budget, seed=a.seed,
        device=a.device, chunk_symbols=a.chunk_symbols, chunk_days=a.chunk_days, pool=a.pool,
        pool_capacity=a.pool_capacity, dedup=a.dedup, min_coverage=a.min_coverage, meanstd_lr=a.meanstd_lr,
        meanstd_steps=a.meanstd_steps, dynamic_window=a.dynamic_window, seed_library=a.seed_library,
        presqueeze=(dict(n_iterations=a.presqueeze) if a.presqueeze else None), gp_params=gp,
        allow_boolean_ops=a.allow_boolean_ops, allow_resource_risk=a.allow_resource_risk,
        checkpoint_every=a.checkpoint_every,
        workdir=a.workdir,
        max_expr_len=a.max_expr_len, combine_methods=tuple(x for x in a.combine.split(",") if x),
        combine_train_frac=a.combine_train_frac, out_root=a.out, run_id=a.run_id,
    )
    return 0 if res.pool_size else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="alphacube")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="write a synthetic cube store")
    s.add_argument("--out", required=True)
    s.add_argument("--symbols", type=int, default=50)
    s.add_argument("--days", type=int, default=250)
    s.add_argument("--minutes", type=int, default=241)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--suspended", type=float, default=0.02)
    s.add_argument("--plant", type=float, default=0.0, help="plant a day_std(Ret) -> next-day return signal")
    s.set_defaults(fn=cmd_synth)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--store", required=True, help="cube store directory (V2 minute-store layout)")
    common.add_argument("--start", default=None, help="YYYYMMDD, snaps to the first trading day >= it")
    common.add_argument("--end", default=None)
    common.add_argument("--max-symbols", type=int, default=None)
    common.add_argument("--label", default="open_vwap15", choices=LABELS)
    common.add_argument("--label-param", action="append", default=[], metavar="KEY=VALUE")
    common.add_argument("--horizon", type=int, default=1)
    common.add_argument("--device", default="cpu")
    common.add_argument("--workdir", default=None,
                        help="where boundary memmaps go. One boundary is a whole (S, D, M) float32 cube "
                             "(9.8 GB at 5219x1942x241), so this must NOT land on a small system disk. "
                             "Defaults to $ALPHACUBE_WORKDIR, else <project>/tmp")
    common.add_argument("--chunk-symbols", type=int, default=None)
    common.add_argument("--chunk-days", type=int, default=None)
    common.add_argument("--out", default=None, help="run directory parent")
    common.add_argument("--run-id", default=None)

    r = sub.add_parser("research", parents=[common], help="evaluate expressions -> metrics table")
    r.add_argument("--expr", action="append", default=[], help="name=expression (repeatable)")
    r.add_argument("--file", default=None, help="text file of name=expression lines, or a JSON object")
    r.add_argument("--legacy", action="store_true", help="add the 1498-factor legacy corpus")
    r.set_defaults(fn=cmd_research)

    c = sub.add_parser("calculate", parents=[common], help="production factor values (resumable) + parquet export")
    c.add_argument("--expr", action="append", default=[])
    c.add_argument("--file", default=None)
    c.add_argument("--legacy", action="store_true")
    c.add_argument("--batch-size", type=int, default=64)
    c.add_argument("--allow-resource-risk", action="store_true", help="do not defer ts_pearson-class operators")
    c.add_argument("--no-resume", action="store_true")
    c.add_argument("--parquet", default=None, help="also export the run to this parquet root")
    c.set_defaults(fn=cmd_calculate)

    m = sub.add_parser("mine", parents=[common], help="mine a factor pool with any miner")
    m.add_argument("--miner", default="gp", choices=MINERS)
    m.add_argument("--pool", default="mse", choices=POOLS)
    m.add_argument("--budget", type=int, default=480)
    m.add_argument("--population", type=int, default=24, help="GP population")
    m.add_argument("--min-depth", type=int, default=2)
    m.add_argument("--max-depth", type=int, default=4)
    m.add_argument("--max-expr-len", type=int, default=20, help="RL token budget per expression")
    m.add_argument("--pool-capacity", type=int, default=10)
    m.add_argument("--checkpoint-every", type=int, default=0,
                   help="write the ranked archive to <run>/checkpoints/archive_gen<N>.csv every N "
                        "generations. Without it the expressions exist only in memory until the whole "
                        "budget is spent, so an interrupted run leaves nothing")
    m.add_argument("--panel-cache", type=int, default=512,
                   help="how many evaluated (D, S) panels GP keeps, LRU. A miss costs a full re-read of "
                        "every field the tree touches -- 9.1 GB each on the production cube -- so on a big "
                        "machine this is the cheapest speed knob there is. One panel is n_days x n_symbols x 4 "
                        "bytes (40 MB at 1942 x 5219), so 512 is ~21 GB and 2048 is ~83 GB")
    m.add_argument("--top-k", type=int, default=None)
    m.add_argument("--dedup", type=float, default=0.99)
    m.add_argument("--min-coverage", type=float, default=0.5)
    m.add_argument("--meanstd-lr", type=float, default=0.05)
    m.add_argument("--meanstd-steps", type=int, default=200)
    m.add_argument("--dynamic-window", type=int, default=60)
    m.add_argument("--warmup", type=int, default=60, help="day-axis warm-up rows read before --start")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--seed-library", default=None, help="seed the pool from a seed_alpha_library yaml first")
    m.add_argument("--presqueeze", type=int, default=0, help="feature-boosting pre-squeeze iterations before mining (0 = off)")
    m.add_argument("--allow-resource-risk", action="store_true",
                   help="let the search use ts_pearson / nt_ts_pearson / nt_ts_spearman, whose cost is "
                        "super-linear in the stock count (the NT pair kernels are 5219^2 pairs x 1942 days "
                        "on the production cube). Off by default; calculate.py has always deferred them")
    m.add_argument("--allow-boolean-ops", action="store_true",
                   help="let the search use the 29 strictly-0/1 operators (eq, is_gt0, cs_quantile_q50, ...). "
                        "Off by default: they cannot be a root, but arithmetic launders them into one "
                        "(pow_05(eq(a, b))) and the near-constant indicator that results scores a spurious IC")
    m.add_argument("--combine", default="equal,ic,icir,lstsq,ridge,gbdt",
                   help="comma list of combination methods for the out-of-sample leaderboard ('' to skip)")
    m.add_argument("--combine-train-frac", type=float, default=0.7)
    m.set_defaults(fn=cmd_mine)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
