"""把 checkpoint 里的前 N 条表达式重新求值,跑一遍和块结算同一口径的池 + 组合层(服务器上跑)。

    cd /root/autodl-tmp/AlphaCube && PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
      /root/miniconda3/envs/quant/bin/python scripts/combine_archive.py \\
      mining_runs/gpu9/checkpoints/archive_gen010.csv --top 120 --gpu-fraction 0.045 \\
      --out _bench/gpu9_gen010_combine

口径与 pipeline.run_mining 的块结算(score_archive)一致:warm-up 60 天的视图、open_vwap15 h=1
(realized = target)、MsePool(容量 30,dedup 0.95,coverage 0.5,scored days 0.5)、方向按训练段
IC 对齐、后 30% 交易日样本外、单边 15bp、限换手 30%,外加组合后 EMA 平滑(半衰期只在训练段
按扣成本夏普选,0 = 不平滑永远是候选)。用途:给中断的 run、或早于某个组合层功能的 run 补块结算。

--gpu-fraction 0 = 只用 CPU(慢,但不碰 GPU);>0 给本进程显存封顶 —— 旁边有任务在跑时用 0.045:
任务里的 CUDA OOM 会被当成单个节点失败静默吞掉,不会崩。GPU 上求值失败的表达式自动退回 CPU。
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--store", default="/root/autodl-tmp/qmt_daily_data/fast_factor_compute")
    ap.add_argument("--start", default="20180102")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--top", type=int, default=120, help="取 checkpoint 前 N 条求值(checkpoint 按适应度排好序)")
    ap.add_argument("--capacity", type=int, default=30)
    ap.add_argument("--dedup", type=float, default=0.95)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--cost-per-side", type=float, default=0.0015)
    ap.add_argument("--turnover-cap", type=float, default=0.30)
    ap.add_argument("--smooth-halflives", default="1,2,3,5,10,20", help="逗号分隔;空串 = 不试平滑")
    ap.add_argument("--methods", default="equal,ic,icir,lstsq,ridge,gbdt,lightgbm,xgboost")
    ap.add_argument("--gpu-fraction", type=float, default=0.045, help="0 = 只用 CPU")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import csv

    import torch
    dev = "cpu" if a.gpu_fraction == 0 else "cuda"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.gpu_fraction)

    from alphacube.combine.post_combine import compare, leaderboard
    from alphacube.engine.engine import CubeEngine
    from alphacube.label import Label
    from alphacube.parser import parse
    from alphacube.pipeline import _TABLE_COLUMNS, build_pool
    from alphacube.research import warmed_view
    from alphacube.store import open_cube_store

    rows = list(csv.DictReader(open(a.archive, encoding="utf-8")))[: a.top]
    exprs = []
    for r in rows:
        try:
            exprs.append(parse(r["expression"]))
        except Exception as exc:  # noqa: BLE001 - an old archive may hold a retired operator
            print(f"  解析失败(跳过): {r['expression'][:60]} -> {exc}", flush=True)
    print(f"读入 {len(exprs)} 条表达式(前 {a.top} 条,{a.archive})", flush=True)

    view, trim = warmed_view(open_cube_store(a.store), a.start, a.end, warmup=a.warmup)
    target = Label("open_vwap15", 1).compute(view).astype(np.float64)
    print(f"视图 {view.n_days} 天(warm-up {trim})x {view.n_symbols} 只;label 有限格点 "
          f"{100 * np.isfinite(target).mean():.1f}%", flush=True)

    t0 = time.time()
    kw = {} if dev == "cpu" else dict(chunk_symbols=5, chunk_days=2)
    res = CubeEngine(view, device=dev, **kw).evaluate(exprs, tolerant=True)
    panels = list(res.panels)
    if res.errors and dev == "cuda":                   # what does not fit under the GPU cap goes to the CPU
        idx = sorted(res.errors)
        r2 = CubeEngine(view, device="cpu").evaluate([exprs[i] for i in idx], tolerant=True)
        for i, p in zip(idx, r2.panels):
            panels[i] = p
    print(f"{dev}: 求值 {len(exprs)} 条 {time.time() - t0:.0f}s,仍失败 {sum(p is None for p in panels)} 条",
          flush=True)

    pool = build_pool("mse", a.capacity, target, target, a.dedup, 0.5, min_scored_days=0.5)
    adm = pool.admit(zip(exprs, panels))
    print(f"\n候选 {len(exprs)} -> 进池 {pool.size}")
    for k, v in sorted(adm.reasons.items(), key=lambda kv: -kv[1])[:6]:
        print(f"   被拒 {v:>3}  {k}")
    if pool.size == 0:
        print("池为空,无法组合")
        return 1

    halflives = tuple(float(x) for x in a.smooth_halflives.split(",") if x)
    methods = tuple(x for x in a.methods.split(",") if x)
    t0 = time.time()
    results, signs = compare(pool.z_columns(), target, target, methods=methods, train_frac=a.train_frac,
                             align=True, seed=0, cost_per_side=a.cost_per_side, turnover_cap=a.turnover_cap,
                             smooth_halflives=halflives)
    table = leaderboard(results)
    cols = [c for c in _TABLE_COLUMNS if c in table.columns]
    print(f"\n=== 组合(样本外:后 {100 * (1 - a.train_frac):.0f}% 交易日;{int((signs < 0).sum())} 条翻转;"
          f"单边 {a.cost_per_side * 1e4:.0f}bp;{time.time() - t0:.0f}s)===")
    print(table[cols].to_string(float_format=lambda v: f"{v:+.4f}"))

    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "combination.csv")
        with open(out / "members.csv", "w", encoding="utf-8") as fh:
            fh.write("index,weight,ic,expression\n")
            for i, (m, w) in enumerate(zip(pool.members, pool.weights)):
                fh.write(f'{i},{w:.6f},{m.ic:.6f},"{m.expr!r}"\n')
        print(f"\n写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
