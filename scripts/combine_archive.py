"""把 checkpoint 里的因子表达式跑一遍下游组合层。

正常流程里这一步只在 40 代全部跑完后执行,所以一次被中断的挖掘什么也拿不到。
checkpoint 存的是表达式文本,重新求值就能补上这一步 —— 代价是再算一遍面板。

    python scripts/combine_archive.py <archive.csv> [--top 120] [--capacity 30]
"""
import argparse, csv, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--store", default="/root/autodl-tmp/qmt_daily_data/fast_factor_compute")
    ap.add_argument("--start", default="20180102")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--top", type=int, default=120, help="按 |IC| 取前 N 条求值")
    ap.add_argument("--capacity", type=int, default=30)
    ap.add_argument("--dedup", type=float, default=0.95)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from alphacube.combine.post_combine import compare, leaderboard
    from alphacube.engine.engine import CubeEngine
    from alphacube.label import Label
    from alphacube.mining.pool import MsePool
    from alphacube.parser import parse
    from alphacube.store import open_cube_store

    rows = list(csv.DictReader(open(a.archive, encoding="utf-8")))
    rows = rows[: a.top]
    print(f"读入 {len(rows)} 条表达式(共 {a.archive})", flush=True)

    view = open_cube_store(a.store).view(start=a.start, end=a.end)
    print(f"store: {view.n_days} 天 x {view.n_symbols} 只", flush=True)
    target = Label("open_vwap15", 1).compute(view)
    print(f"label 有限格点 {100*np.isfinite(target).mean():.1f}%", flush=True)

    exprs, texts = [], []
    for r in rows:
        try:
            exprs.append(parse(r["expression"]))
            texts.append(r["expression"])
        except Exception as exc:
            print(f"  解析失败(跳过): {r['expression'][:60]} -> {exc}", flush=True)

    t0 = time.time()
    last = [0.0]

    def prog(ri, nr, done, total):
        if time.time() - last[0] < 60:
            return
        last[0] = time.time()
        el = time.time() - t0
        f = done / total if total else 0
        eta = (el / f - el) / 60 if f > 0.01 else float("nan")
        print(f"   ... round {ri+1}/{nr}  {done}/{total} units  {100*f:.0f}%  "
              f"elapsed {el/60:.0f}m  eta {eta:.0f}m", flush=True)

    res = CubeEngine(view, device=a.device).evaluate(exprs, tolerant=True, on_progress=prog)
    print(f"求值完成 {time.time()-t0:.0f}s,失败 {len(res.errors)} 条", flush=True)

    pool = MsePool(a.capacity, target, dedup_threshold=a.dedup, realized_returns=target)
    kept = []
    reasons = {}
    for e, t, p in zip(exprs, texts, res.panels):
        if p is None:
            reasons["求值失败"] = reasons.get("求值失败", 0) + 1
            continue
        try:
            pool.try_add(e, p.astype(np.float64))
            kept.append(t)
        except Exception as exc:
            k = type(exc).__name__
            reasons[k] = reasons.get(k, 0) + 1
    print(f"\n进池 {pool.size} 条 / 尝试 {len(exprs)} 条")
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"   被拒 {v:>3}  {k}")

    if pool.size == 0:
        print("池为空,无法组合")
        return 1

    print("\n=== 池内成员 ===")
    for i, (m, w) in enumerate(zip(pool.members, pool.weights)):
        print(f"  [{i:2d}] w={w:+.5f}  IC={m.ic:+.4f}  {m.expr!r}")

    print(f"\n=== 池整体(样本内)===")
    for k, v in pool.test_ensemble().items():
        print(f"  {k:20s} {v:+.4f}")

    methods = ("equal", "ic", "icir", "lstsq", "ridge", "gbdt", "lightgbm", "xgboost")
    print(f"\n=== 组合方法对比(样本外:后 {100*(1-a.train_frac):.0f}% 交易日)===", flush=True)
    results, signs = compare(pool.z_columns(), target, target, methods=methods,
                             train_frac=a.train_frac, align=True, seed=0)
    table = leaderboard(results)
    cols = [c for c in ("ic_mean", "icir", "rank_ic", "annualized_return", "sharpe",
                        "max_drawdown", "coverage", "error") if c in table.columns]
    print(f"(方向对齐:{int((signs < 0).sum())} 条被翻转)")
    print(table[cols].to_string(float_format=lambda v: f"{v:+.4f}"))

    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        table.to_csv(Path(a.out) / "combination.csv")
        with open(Path(a.out) / "members.csv", "w", encoding="utf-8") as fh:
            fh.write("index,weight,ic,expression\n")
            for i, (m, w) in enumerate(zip(pool.members, pool.weights)):
                fh.write(f'{i},{w:.6f},{m.ic:.6f},"{repr(m.expr)}"\n')
        print(f"\n写入 {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
