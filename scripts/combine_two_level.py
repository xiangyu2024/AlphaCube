"""两层合成:族内把同一想法的多种写法合成成一个"族因子",再在族之间合成。

为什么分两层:4157 个因子在 |rank corr| 0.7 处切出 1597 族,一个族就是同一个想法
的不同写法(gpu15 前 20 名是一个母题的 20 种)。从族里"挑代表"要么丢掉信息
(只留 1 个),要么把冗余带进上层(留 3 个)。整族合成成一个因子,既摊平了各写法的
特异噪声,又只占上层的一个位置。

两层都把 equal / ic / icir / lstsq / ridge 全试一遍 —— 族内合成的最佳方法和
族间合成的未必相同:族内成员高度相关(拟合权重容易过拟合),族间成员近乎正交
(拟合权重才有发挥空间)。

族内权重和族间权重都只在训练段拟合;最后一律再报一次 EMA 平滑版(半衰期在训练段
选),因为实测平滑把换手的边际成本从 -10.1 降到 -1.1 夏普/单位换手。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

METHODS = ("equal", "ic", "icir", "lstsq", "ridge")


def fit_weights(cols: List[np.ndarray], target: np.ndarray, train: slice, method: str,
                ridge_scale: float = 1e-3) -> np.ndarray:
    """按 method 在训练段拟合一组权重(成员已按训练段 IC 方向对齐)。"""
    from alphacube.metrics import cs_pearson

    k = len(cols)
    if k == 1:
        return np.ones(1)
    if method == "equal":
        return np.full(k, 1.0 / k)
    if method in ("ic", "icir"):
        s = np.zeros(k)
        for j, c in enumerate(cols):
            ics = cs_pearson(c[train], target[train])
            f = ics[np.isfinite(ics)]
            if f.size < 2:
                continue
            m = float(f.mean())
            s[j] = abs(m) if method == "ic" else (abs(m / f.std()) if f.std() > 1e-12 else 0.0)
        return s / s.sum() if s.sum() > 1e-12 else np.full(k, 1.0 / k)
    X = np.stack([np.nan_to_num(c[train], nan=0.0).reshape(-1) for c in cols]).astype(np.float32)
    y = np.nan_to_num(target[train], nan=0.0).reshape(-1).astype(np.float32)
    G = (X @ X.T).astype(np.float64)
    b = (X @ y).astype(np.float64)
    lam = (ridge_scale if method == "ridge" else 0.0) * float(np.trace(G)) / max(k, 1)
    try:
        return np.linalg.solve(G + lam * np.eye(k), b)
    except np.linalg.LinAlgError:
        # 族内成员高度共线(本来就是同一个想法的不同写法),OLS 的正规方程常常奇异。
        # 退回最小范数解,而不是悄悄变成等权 —— 那会让"lstsq"这一行名不副实。
        return np.linalg.pinv(G + lam * np.eye(k), rcond=1e-10) @ b


def blend(cols: List[np.ndarray], w: np.ndarray) -> np.ndarray:
    acc = np.zeros(cols[0].shape, dtype=np.float64)
    seen = np.zeros(cols[0].shape, dtype=bool)
    for c, wi in zip(cols, w):
        fin = np.isfinite(c)
        acc += wi * np.where(fin, c, 0.0)
        seen |= fin
    acc[~seen] = np.nan
    return acc


def report(sig: np.ndarray, target: np.ndarray, train: slice, test: slice, cost: float,
           halflives=(0.0, 1, 2, 3, 5, 10, 20)) -> dict:
    """测试段指标,外加"半衰期在训练段选"的平滑版。"""
    from alphacube.metrics import cs_pearson, ema_rows, long_short_weights, portfolio_returns, rank_rows

    best = None
    for hl in halflives:
        sm = ema_rows(sig, hl) if hl else sig
        w = long_short_weights(sm[train], 0.1)
        net, _, _ = portfolio_returns(w, target[train], cost)
        f = net[np.isfinite(net)]
        s = float(f.mean() / f.std() * np.sqrt(252)) if f.size > 2 and f.std() > 0 else -np.inf
        if best is None or s > best[0]:
            best = (s, hl)
    out = {"halflife": best[1]}
    for tag, s2 in (("", sig), ("sm_", ema_rows(sig, best[1]) if best[1] else sig)):
        w = long_short_weights(s2[test], 0.1)
        net, gross, turn = portfolio_returns(w, target[test], cost)
        f, g = net[np.isfinite(net)], gross[np.isfinite(gross)]
        eq = np.cumsum(f)
        out[tag + "net"] = float(f.mean() / f.std() * np.sqrt(252)) if f.size > 2 and f.std() > 0 else np.nan
        out[tag + "gross"] = float(g.mean() / g.std() * np.sqrt(252)) if g.size > 2 and g.std() > 0 else np.nan
        out[tag + "mdd"] = float((eq - np.maximum.accumulate(eq)).min())
        out[tag + "turnover"] = float(np.nanmean(turn))
    ics = cs_pearson(rank_rows(sig[test]), rank_rows(target[test]))
    fi = ics[np.isfinite(ics)]
    out["rank_ic"] = float(fi.mean()) if fi.size else np.nan
    out["icir"] = float(fi.mean() / fi.std()) if fi.size > 1 and fi.std() > 0 else np.nan
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="_bench/allfac")
    ap.add_argument("--store", default="/root/autodl-tmp/qmt_daily_data/fast_factor_compute")
    ap.add_argument("--label", default="open_vwap15")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--cost", type=float, default=0.0015)
    ap.add_argument("--cluster", type=float, default=0.7)
    ap.add_argument("--max-members", type=int, default=60, help="一族最多用这么多个成员(按适应度取)")
    ap.add_argument("--max-families", type=int, default=100,
                    help="只用质量最好的这么多个族:族因子要同时留在内存里做 5x5 的交叉,"
                         "241 族 x 5 方法 x 40 MB(float32) 已经是 48 GB")
    ap.add_argument("--within", default="all", help="族内合成方法:all 或逗号分隔")
    ap.add_argument("--out", default="_bench/two_level.json")
    ap.add_argument("--buyable", action="store_true",
                    help="标签剔除次日开盘封涨停、买不进的那部分收益(Label(buyable=True))")
    a = ap.parse_args(argv)

    from alphacube.label import Label
    from alphacube.metrics import cs_pearson, nanmean_quiet, zscore_rows
    from alphacube.research import warmed_view
    from alphacube.store import open_cube_store
    from select_factors import cluster_by_correlation, fitness, pairwise_abs_corr, selectable_rows, MIN_FIT, MIN_RIC

    d = Path(a.dir)
    t0 = time.time()
    view, _ = warmed_view(open_cube_store(a.store), "20180102", "20251231", warmup=60)
    target = Label(a.label, 1, buyable=a.buyable).compute(view).astype(np.float64)
    D, S = target.shape
    split = max(1, min(D - 1, int(D * a.train_frac)))
    train, test = slice(0, split), slice(split, D)

    rows = selectable_rows(d / "metrics.csv")
    num = lambda r, k: float(r[k]) if r[k] not in ("", "nan", "None") else np.nan   # noqa: E731
    fp = np.load(d / "fp_train_250x512.npy", mmap_mode="r")
    labels = cluster_by_correlation(pairwise_abs_corr(np.asarray(fp)), a.cluster)
    print(f"{len(rows)} 个健康因子 -> {labels.max()+1} 族 ({time.time()-t0:.0f}s)", flush=True)

    fams = []
    for fam in np.unique(labels):
        idx = list(np.flatnonzero(labels == fam))
        best_fit = max(fitness(rows[j]) for j in idx)
        best_ric = max(abs(num(rows[j], "rank_ic_train")) for j in idx)
        if len(idx) <= 3 and not (best_fit >= MIN_FIT or best_ric >= MIN_RIC):
            continue
        idx.sort(key=lambda j: -fitness(rows[j]))
        fams.append((int(fam), idx[: a.max_members], best_fit))
    fams.sort(key=lambda t: -t[2])
    fams = fams[: a.max_families]
    print(f"过门槛的族 {len(fams)} 个(成员数 中位 {int(np.median([len(f[1]) for f in fams]))},"
          f"最大 {max(len(f[1]) for f in fams)}) ({time.time()-t0:.0f}s)", flush=True)

    within = METHODS if a.within == "all" else tuple(x for x in a.within.split(",") if x)
    composites: Dict[str, List[np.ndarray]] = {m: [] for m in within}
    fam_rows = []
    for n, (fam, idx, _) in enumerate(fams):
        cols = []
        for j in idx:
            z = zscore_rows(np.load(d / "panels" / f"{int(rows[j]['i']):05d}.npy").astype(np.float64))
            if nanmean_quiet(cs_pearson(z[train], target[train])) < 0:
                z = -z
            cols.append(z)
        for m in within:
            composites[m].append(blend(cols, fit_weights(cols, target, train, m)).astype(np.float32))
        fam_rows.append({"cluster": fam, "n": len(idx),
                         "runs": sorted({rows[j]["run"] for j in idx}),
                         "example": rows[idx[0]]["expression"][:90]})
        del cols
        if (n + 1) % 25 == 0:
            print(f"  族内合成 {n+1}/{len(fams)} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n=== 第一层:族内合成方法的比较(把所有族因子等权相加之后的表现)===", flush=True)
    print(f"{'族内方法':>10} {'净夏普':>7} {'毛夏普':>7} {'换手':>7} {'rankIC':>8} {'ICIR':>7} {'平滑后':>7} {'半衰期':>6}")
    for m in within:
        sig = blend(composites[m], np.full(len(composites[m]), 1.0 / len(composites[m])))
        r = report(sig, target, train, test, a.cost)
        print(f"{m:>10} {r['net']:+7.2f} {r['gross']:+7.2f} {r['turnover']*100:6.1f}% "
              f"{r['rank_ic']:+8.4f} {r['icir']:+7.3f} {r['sm_net']:+7.2f} {r['halflife']:6.0f}", flush=True)

    print(f"\n=== 第二层:族内 x 族间 的全部组合 ===", flush=True)
    print(f"{'族内':>8} {'族间':>8} {'净夏普':>7} {'毛夏普':>7} {'换手':>7} {'rankIC':>8} {'ICIR':>7} {'平滑后':>7} {'半衰期':>6}")
    best_all = []
    for m in within:
        comp = composites[m]
        for m2 in METHODS:
            w = fit_weights(comp, target, train, m2)
            r = report(blend(comp, w), target, train, test, a.cost)
            best_all.append((max(r["net"], r["sm_net"]), m, m2, r))
    for _, m, m2, r in sorted(best_all, reverse=True):
        print(f"{m:>8} {m2:>8} {r['net']:+7.2f} {r['gross']:+7.2f} {r['turnover']*100:6.1f}% "
              f"{r['rank_ic']:+8.4f} {r['icir']:+7.3f} {r['sm_net']:+7.2f} {r['halflife']:6.0f}", flush=True)
    Path(a.out).write_text(json.dumps({"families": fam_rows}, ensure_ascii=False), encoding="utf-8")
    print(f"\n族清单写到 {a.out} ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
