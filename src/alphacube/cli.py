"""Command line: ``python -m alphacube.cli {synth,research,mine,seed} ...``

``mine`` is the one-shot entry the previous project's ``train.py`` provided:
read the cube -> label -> pool -> any miner -> metrics -> combination
leaderboard -> everything on disk.
"""

from __future__ import annotations

import argparse
import os
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

    frac = os.environ.get("ALPHACUBE_GPU_FRACTION")
    if frac and str(a.device).startswith("cuda"):
        # 同一张卡上同时跑几个挖掘时给每个进程的显存封顶;引擎的 OOM 在 tolerant 模式下
        # 只会让节点静悄悄地失败,不会报错
        import torch
        torch.cuda.set_per_process_memory_fraction(float(frac))
        print(f"GPU memory capped at {float(frac):.0%} of the card")

    if a.fitness in ("long", "long_hit") and a.cost_per_side <= 0:
        raise SystemExit("--fitness long / long_hit 把换手按 --cost-per-side 从多头书里直接扣;不给它(或给 0)就是在奖励高换手")
    gp = dict(population_size=a.population, max_depth=a.max_depth, min_depth=a.min_depth, top_k=a.top_k,
              panel_cache_items=a.panel_cache,
              exemplar_cache=(a.exemplar_cache or None), exemplar_cache_gb=a.exemplar_cache_gb,
              panel_cache_bytes=int(a.panel_cache_gb * 1024 ** 3),
              banned_substructures=tuple(a.ban),
              ban_rank_corr=a.ban_rank_corr, fitness_kind=a.fitness, turnover_penalty=a.turnover_penalty,
              archive_dedup=a.archive_dedup)
    if a.rl_min_solo is not None:
        gp["rl_min_solo"] = a.rl_min_solo
    if a.seed_population:
        gp.update(seed_population=a.seed_population, seed_keep=a.seed_keep, seed_max_depth=a.seed_max_depth,
                  seed_per_block=a.seed_per_block)
    if a.ban_archive:
        from .pipeline import ban_exemplars_from
        if not a.ban_rank_corr or not a.ban:
            raise SystemExit("--ban-archive needs --ban-rank-corr and the --ban keys whose factors it should load")
        gp["ban_exemplars"] = ban_exemplars_from(a.ban_archive, a.ban)
        print(f"rank-ban exemplars from {len(a.ban_archive)} archive(s): {len(gp['ban_exemplars'])} expressions")
    res = run_mining(
        a.store, start=a.start, end=a.end, max_symbols=a.max_symbols, warmup=a.warmup, label=a.label,
        label_params=_kv(a.label_param), horizon=a.horizon, miner=a.miner, budget=a.budget, seed=a.seed,
        device=a.device, chunk_symbols=a.chunk_symbols, chunk_days=a.chunk_days, pool=a.pool,
        pool_capacity=a.pool_capacity, dedup=a.dedup, min_coverage=a.min_coverage,
        min_scored_days=a.min_scored_days, cost_per_side=a.cost_per_side, turnover_cap=a.turnover_cap, meanstd_lr=a.meanstd_lr,
        meanstd_steps=a.meanstd_steps, dynamic_window=a.dynamic_window, seed_library=a.seed_library,
        presqueeze=(dict(n_iterations=a.presqueeze) if a.presqueeze else None), gp_params=gp,
        allow_boolean_ops=a.allow_boolean_ops, allow_resource_risk=a.allow_resource_risk,
        checkpoint_every=a.checkpoint_every, rolling_block=a.rolling_block, ban_top_n=a.ban_top_n,
        block_combine_top=a.block_combine_top, engine_progress_every=a.progress_every,
        skip_kernels=tuple(x for x in a.skip_kernels.split(",") if x),
        workdir=a.workdir,
        max_expr_len=a.max_expr_len, combine_methods=tuple(x for x in a.combine.split(",") if x),
        combine_train_frac=a.combine_train_frac, out_root=a.out, run_id=a.run_id,
        smooth_halflives=tuple(float(x) for x in a.smooth_halflives.split(",") if x),
        mirror_root=a.mirror_root, universe=(a.universe or None),
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
    common.add_argument("--mirror-root", default=None,
                        help="D-major mirrors built by scripts/build_mirror.py (a store may be read-only); "
                             "D rounds then read contiguous (D, S, M) blocks instead of strided (S, D, M) ones")
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
    m.add_argument("--universe", default="",
                   help="股票池掩码 .npy(旁边要有同名 .json 记日期和股票顺序,scripts/universe.py 写的):"
                        "只在进过池子的票上算因子,标签和因子面板的池外格子都置空")
    m.add_argument("--pool", default="mse", choices=POOLS)
    m.add_argument("--budget", type=int, default=480)
    m.add_argument("--population", type=int, default=24, help="GP population")
    m.add_argument("--min-depth", type=int, default=2)
    m.add_argument("--max-depth", type=int, default=4)
    m.add_argument("--max-expr-len", type=int, default=20, help="RL token budget per expression")
    m.add_argument("--pool-capacity", type=int, default=10,
                   help="members the pool keeps; beyond it the weakest |weight| is pruned. 0 = no cap: every candidate that passes the gates (finite IC, coverage, scored days, dedup) stays, and the final commit offers the whole archive")
    m.add_argument("--progress-every", type=float, default=0.0,
                   help="seconds between inside-a-generation progress lines (0 = off). A generation is one "
                        "engine call, so without this the log is silent for its whole duration -- 58 minutes "
                        "for a cold full-market population -- and a slow generation looks like a hang")
    m.add_argument("--skip-kernels", default="",
                   help="comma list of operators the search may BREED but never EVALUATE (scored -inf "
                        "unevaluated). Unlike dropping them from the vocabulary, this leaves every RNG "
                        "draw where it was, so a run reproduces an earlier one exactly up to the first "
                        "refusal -- use it with --allow-resource-risk to replay a run that was poisoned "
                        "by an expensive operator")
    m.add_argument("--ban", action="append", default=[], metavar="KEY",
                   help="depth-1 structure banned from the first generation, as the block reports print "
                        "it, e.g. --ban 'O:log(F:Amt)' --ban 'O:nt_ts_argmax(F:High)|w3'. Repeatable")
    m.add_argument("--ban-rank-corr", type=float, default=0.0,
                   help="also refuse any candidate whose daily cross-sectional ranks correlate (absolute, "
                        "pooled over the sampled days) at least this much with an exemplar of a banned "
                        "structure -- so a monotone rewrite such as log(pow_1_5(Amt)) for a banned log(Amt) "
                        "is caught (0 = off). Exemplars: every archived factor containing a structure a "
                        "rolling block bans, plus --ban-archive")
    m.add_argument("--ban-archive", action="append", default=[], metavar="CSV",
                   help="checkpoint CSV of an earlier run; its expressions that contain a --ban key become "
                        "rank-ban exemplars from the first generation. Repeatable")
    m.add_argument("--rolling-block", type=int, default=0,
                   help="restart the population every N generations (0 = off, one long run). A single run "
                        "converges onto one primitive and spends the rest of its budget re-wrapping it")
    m.add_argument("--ban-top-n", type=int, default=0,
                   help="at each rolling restart, forbid the N depth-1 substructures the finished block "
                        "used most, e.g. day_ar1_coefficient(Volume). Banned everywhere -- fresh trees, "
                        "mutations and crossovers -- so the next block has to look elsewhere")
    m.add_argument("--checkpoint-every", type=int, default=0,
                   help="write the ranked archive to <run>/checkpoints/archive_gen<N>.csv every N "
                        "generations. Without it the expressions exist only in memory until the whole "
                        "budget is spent, so an interrupted run leaves nothing")
    m.add_argument("--panel-cache", type=int, default=512,
                   help="how many evaluated (D, S) panels GP keeps, LRU. A miss costs a full re-read of "
                        "every field the tree touches -- 9.1 GB each on the production cube -- so on a big "
                        "machine this is the cheapest speed knob there is. One panel is n_days x n_symbols x 4 "
                        "bytes (40 MB at 1942 x 5219), so 512 is ~21 GB and 2048 is ~83 GB")
    m.add_argument("--exemplar-cache", default="tmp/exemplar_cache",
                   help="ban exemplar 的秩指纹缓存目录(空字符串 = 关掉)。每轮开跑要把 ban-archive 里的"
                        "表达式全评估一遍只为拿指纹:gpu17 的 1673 条花了 2 小时 04 分。一条指纹 3.07 MB,"
                        "读回来几秒。视图(日期范围/股票数/采样)一变旧缓存自动作废并删除")
    m.add_argument("--exemplar-cache-gb", type=float, default=12.0,
                   help="缓存上限,超了按最近使用时间淘汰(0 = 不限)")
    m.add_argument("--panel-cache-gb", type=float, default=12.0,
                   help="hard cap on the same LRU in GiB -- whichever of the two binds first wins. "
                        "The item cap alone does not know how big a panel is, and it does not shrink "
                        "when the rest of the run grows: gpu14 ran --panel-cache 1024 (41 GB) beside a "
                        "23 GB ban index in a 90 GiB container and died at a block boundary. 0 = no "
                        "byte cap")
    m.add_argument("--top-k", type=int, default=None,
                   help="archive leaders offered to the pool at the end (default: 2 x pool capacity; the whole archive when the pool has no cap; 0 = the whole archive)")
    m.add_argument("--block-combine-top", type=int, default=120,
                   help="archive leaders each rolling-block report scores (0 = every archived expression)")
    m.add_argument("--dedup", type=float, default=0.99)
    m.add_argument("--min-coverage", type=float, default=0.5,
                   help="share of the label's finite cells on which the factor must be finite")
    m.add_argument("--cost-per-side", type=float, default=0.0,
                   help="one-way trading cost as a fraction (0.0015 = 15bp). Every unit of weight that "
                        "moves pays it, buy or sell, so a full round trip of the book costs twice this. "
                        "0 keeps the old cost-free numbers")
    m.add_argument("--turnover-cap", type=float, default=None,
                   help="also report the same factors under a daily turnover ceiling (0.3 = 30%% of the "
                        "book). When the full trade to target would exceed it, the book is blended toward "
                        "the target instead -- path dependent, so today's holding starts tomorrow's trade")
    m.add_argument("--min-scored-days", type=float, default=0.5,
                   help="share of usable days on which the factor must actually PRODUCE an IC. "
                        "--min-coverage only asks whether the cell is finite, which a panel of zeros "
                        "passes everywhere; this asks whether the day had any cross-sectional spread")
    m.add_argument("--meanstd-lr", type=float, default=0.05)
    m.add_argument("--meanstd-steps", type=int, default=200)
    m.add_argument("--dynamic-window", type=int, default=60)
    m.add_argument("--warmup", type=int, default=60, help="day-axis warm-up rows read before --start")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--seed-library", default=None, help="seed the pool from a seed_alpha_library yaml first")
    m.add_argument("--seed-population", default=None,
                   help="GP only: start each population partly from seed trees -- 'legacy' (the hand-written "
                        "corpus) or a csv with an expression column; ranked by train fitness inside the run")
    m.add_argument("--seed-keep", type=int, default=0, help="seeds per population (0 = half the population)")
    m.add_argument("--seed-max-depth", type=int, default=0, help="drop seeds deeper than this (0 = no cap)")
    m.add_argument("--seed-per-block", action="store_true",
                   help="each rolling block starts from the next-best seeds instead of a random population")
    m.add_argument("--presqueeze", type=int, default=0, help="feature-boosting pre-squeeze iterations before mining (0 = off)")
    m.add_argument("--allow-resource-risk", action="store_true",
                   help="let the search use the NT-face pair kernels (nt_ts_pearson, nt_ts_spearman) and "
                        "in_multireg. Every pair kernel is O(windows x d x N^2) -- 27M pairs per window at "
                        "N=5219 -- but the ND entries (ts_pearson, ts_spearman) run once over the panel and "
                        "stay searchable by default, while the NT twins run once per day, ~241x more, and "
                        "do not")
    m.add_argument("--allow-boolean-ops", action="store_true",
                   help="let the search use the 29 strictly-0/1 operators (eq, is_gt0, cs_quantile_q50, ...). "
                        "Off by default: they cannot be a root, but arithmetic launders them into one "
                        "(pow_05(eq(a, b))) and the near-constant indicator that results scores a spurious IC")
    m.add_argument("--combine", default="equal,ic,icir,lstsq,ridge,gbdt",
                   help="comma list of combination methods for the out-of-sample leaderboard ('' to skip)")
    m.add_argument("--combine-train-frac", type=float, default=0.7)
    m.add_argument("--smooth-halflives", default="",
                   help="comma-separated EMA half-lives (days) to try on every combination: the one with the "
                        "best after-cost Sharpe on the TRAINING days is traded on the test days "
                        "(halflife / sharpe_smooth / turnover_smooth); 0 = no smoothing is always a candidate. "
                        "Empty = off")
    m.add_argument("--rl-min-solo", type=float, default=None,
                   help="RL only: a candidate whose own fitness (already net of turnover under the long kinds, or "
                        "after --turnover-penalty) is below this does not enter the pool; it is paid that fitness "
                        "(floored at half the invalid-expression penalty) the first time and the penalty on repeats. "
                        "Unset = the pool admits on combination IC alone")
    m.add_argument("--fitness", default="ic", choices=("ic", "rank_ic", "long", "long_hit"),
                   help="GP fitness base: |mean daily Pearson IC| (ic) or |mean daily rank IC| (rank_ic); "
                        "long = the factor traded on its own as a long-only top-decile book (net excess Sharpe "
                        "and return after --cost-per-side, plus IC / rank IC on the long half; "
                        "mining/scoring.LongScorer); long_hit = long-side hit rates plus several long-only "
                        "books built from the factor's cross-sectional rank (mining/scoring.LongHitScorer). "
                        "--turnover-penalty is not used with the long kinds")
    m.add_argument("--turnover-penalty", type=float, default=0.0,
                   help="GP fitness = |IC or rank IC| - this x daily turnover of the top/bottom decile book "
                        "(0 = off). 0.14 is 15 bp a side in rank-IC units on the gpu4-gpu8 leaders")
    m.add_argument("--archive-dedup", type=float, default=0.0,
                   help="treat GP trees whose daily ranks correlate >= this as one factor family: the archive "
                        "keeps each family's best, and within a generation only that one breeds (0 = off)")
    m.set_defaults(fn=cmd_mine)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
