"""重建冠军信号:两层合成(族内 ridge x 族间 lstsq),只在训练段拟合,存成 (D, S) float32。

原来在 _bench/deploy_check.py(不在 git 里)。配置不变:按 |rank corr| 0.7 聚类(只用训练段指纹),
成员 <= 3 的小族要过质量门槛,按训练段适应度取最好的 100 族,每族最多 60 个成员。
适应度 = |训练段 rank IC| - 0.14 x **训练段**换手(metrics.csv 的 turnover_train)。2026-10-03 之前用的是
全样本换手(含测试段的因子值),之前建的冠军 / families.json 都带这个偏差;旧 metrics.csv 要重新打分。

--buyable:拟合用的标签剔除"次日开盘封涨停、买不进"的那部分收益(Label(buyable=True))。
打分(metrics.csv 的训练段 rank IC,决定哪些族入选、族内成员顺序)也要用同一个标签重跑:
    score_all_factors.py <dir> --buyable

测试段报告对每个信号都用两种标签各算一遍(原标签 / 剔除封板),新旧信号可以直接比:
多空十分位书(原始与平滑后的净夏普)、rank IC、ICIR、纯多头前 20%·exp1 的超额夏普。
平滑半衰期一律在训练段选,用的是该信号拟合时的标签。

  cd /root/autodl-tmp/AlphaCube && PYTHONPATH=src python -u scripts/build_champion.py \\
      --dir _bench/allfac_buyable --buyable --out _bench/champion_signal_buyable.npy \\
      --compare 原冠军=_bench/champion_signal.npy
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

HALFLIVES = (0.0, 1, 2, 3, 5, 10, 20)
FP_DAYS, FP_SYMBOLS = 250, 512          # 和 select_factors.py 的训练段指纹一致


def _num(r: dict, k: str) -> float:
    return float(r[k]) if r[k] not in ("", "nan", "None") else np.nan


def healthy_rows(d: Path) -> List[dict]:
    from select_factors import selectable_rows

    return selectable_rows(d / "metrics.csv")


def pick_families(rows: List[dict], fp: np.ndarray, cluster: float, max_families: int,
                  max_members: int) -> List[List[int]]:
    """只看训练段:聚类用训练段指纹,小族门槛、族内成员排序、族的排名都用训练段适应度。"""
    from select_factors import MIN_FIT, MIN_RIC, cluster_by_correlation, fitness, pairwise_abs_corr

    def fit(j):
        return fitness(rows[j])

    labels = cluster_by_correlation(pairwise_abs_corr(fp), cluster)
    fams = []
    for fam in np.unique(labels):
        idx = list(np.flatnonzero(labels == fam))
        best_fit = max(fit(j) for j in idx)
        best_ric = max(abs(_num(rows[j], "rank_ic_train")) for j in idx)
        if len(idx) <= 3 and not (best_fit >= MIN_FIT or best_ric >= MIN_RIC):
            continue                               # 小族要过质量门槛:"和谁都不像"多半是噪声
        idx.sort(key=lambda j: -fit(j))
        fams.append((idx[:max_members], best_fit))
    fams.sort(key=lambda t: -t[1])
    print(f"{len(rows)} 个健康因子 -> {labels.max() + 1} 族,过门槛 {len(fams)} 族,取前 {max_families}", flush=True)
    return [f[0] for f in fams[:max_families]]


def universe_fingerprints(panels: Path, indices: Sequence[int], days: slice, universe: np.ndarray,
                          out: Path) -> np.ndarray:
    """只看股票池的训练段指纹:列 = 指纹窗口里进过池子的票,池外的格子当缺失,逐日秩的 z 分数。"""
    from alphacube.metrics import rank_rows, zscore_rows

    if out.exists():
        return np.load(out, mmap_mode="r")
    u = universe[days]
    cols = np.flatnonzero(u.any(0))
    um = u[:, cols]
    fp = np.empty((len(indices), um.size), dtype=np.float32)
    t0 = time.time()
    for j, i in enumerate(indices):
        sub = np.asarray(np.load(panels / f"{int(i):05d}.npy", mmap_mode="r")[days][:, cols], dtype=np.float64)
        sub[~um] = np.nan
        fp[j] = np.nan_to_num(zscore_rows(rank_rows(sub)), nan=0.0).reshape(-1)
        if (j + 1) % 1000 == 0:
            print(f"  指纹 {j + 1}/{len(indices)} ({time.time() - t0:.0f}s)", flush=True)
    np.save(out, fp)
    return fp


def build(d: Path, rows: List[dict], fams: List[List[int]], target: np.ndarray, train: slice,
          save: Optional[Path] = None, universe: Optional[np.ndarray] = None) -> np.ndarray:
    """族内 ridge -> 族间 lstsq。save 给了就把 100 个族因子存成 (n_fam, D, S) float32 的 .npy
    (约 4 GB),之后换族间权重的实验不用再花 20 分钟重建。"""
    from combine_two_level import blend, fit_weights
    from alphacube.metrics import cs_pearson, nanmean_quiet, zscore_rows

    t0 = time.time()
    keep = (np.lib.format.open_memmap(save, mode="w+", dtype=np.float32, shape=(len(fams),) + target.shape)
            if save is not None else None)
    comp = []
    for n, idx in enumerate(fams):
        cols = []
        for j in idx:
            pan = np.load(d / "panels" / f"{int(rows[j]['i']):05d}.npy").astype(np.float64)
            if universe is not None:
                pan[~universe] = np.nan                 # 截面标准化只在池子里做
            z = zscore_rows(pan)
            if nanmean_quiet(cs_pearson(z[train], target[train])) < 0:
                z = -z
            cols.append(z)
        comp.append(blend(cols, fit_weights(cols, target, train, "ridge")).astype(np.float32))
        if keep is not None:
            keep[n] = comp[-1]
        del cols
        if (n + 1) % 25 == 0:
            print(f"  族内合成 {n + 1}/{len(fams)} ({time.time() - t0:.0f}s)", flush=True)
    if keep is not None:
        keep.flush()
        del keep
    comp64 = [c.astype(np.float64) for c in comp]
    del comp
    return blend(comp64, fit_weights(comp64, target, train, "lstsq"))


def evaluate(sig: np.ndarray, target: np.ndarray, split: int, fit_target: np.ndarray,
             cost: float = 0.0015) -> Dict[str, float]:
    """测试段指标。半衰期在训练段上用 fit_target 选(多空书按多空净夏普、纯多头按超额夏普)。"""
    from alphacube.metrics import cs_pearson, ema_rows, long_short_weights, portfolio_returns, rank_rows
    from long_only_grid import equal_weight_universe, long_only_weights, rank_order, stats

    D = sig.shape[0]
    train, test = slice(0, split), slice(split, D)
    sm = {hl: (ema_rows(sig, hl) if hl else sig) for hl in HALFLIVES}

    def ls(s, days, tgt):
        net, gross, turn = portfolio_returns(long_short_weights(s[days], 0.1), tgt[days], cost)
        return net, gross, turn

    hl_ls = max(HALFLIVES, key=lambda hl: np.nan_to_num(stats(ls(sm[hl], train, fit_target)[0])["sharpe"], nan=-9e9))
    net0, gross0, turn0 = ls(sig, test, target)
    net1, _, turn1 = ls(sm[hl_ls], test, target)
    ics = cs_pearson(rank_rows(sig[test]), rank_rows(target[test]))
    fi = ics[np.isfinite(ics)]

    bench_w = equal_weight_universe(fit_target)
    bench_tr = portfolio_returns(bench_w[train], fit_target[train], 0.0)[0]
    orders = {}

    def lo(hl, days, tgt, bench):
        if hl not in orders:
            orders[hl] = rank_order(sm[hl])
        o, n = orders[hl]
        w = long_only_weights(sm[hl][days], 0.2, 0, "exp1", cached=(o[days], n[days]))
        net, _, turn = portfolio_returns(w, tgt[days], cost)
        return net - bench, net, turn

    hl_lo = max(HALFLIVES, key=lambda hl: np.nan_to_num(stats(lo(hl, train, fit_target, bench_tr)[0])["sharpe"],
                                                        nan=-9e9))
    bench_te = portfolio_returns(equal_weight_universe(target)[test], target[test], 0.0)[0]
    ex, ab, turn2 = lo(hl_lo, test, target, bench_te)
    s_ex, s_ab = stats(ex), stats(ab)
    return {"ls_net": stats(net0)["sharpe"], "ls_gross": stats(gross0)["sharpe"], "ls_turn": float(np.nanmean(turn0)),
            "ls_sm_net": stats(net1)["sharpe"], "ls_sm_ann": stats(net1)["ann"], "ls_hl": hl_ls,
            "rank_ic": float(fi.mean()) if fi.size else np.nan,
            "icir": float(fi.mean() / fi.std()) if fi.size > 1 and fi.std() > 0 else np.nan,
            "lo_ex_sharpe": s_ex["sharpe"], "lo_ex_ann": s_ex["ann"], "lo_ex_mdd": s_ex["mdd"],
            "lo_abs_sharpe": s_ab["sharpe"], "lo_turn": float(np.nanmean(turn2)), "lo_hl": hl_lo}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="有 metrics.csv、panels/ 的目录;没有训练段指纹就现建一份")
    ap.add_argument("--store", default="/root/autodl-tmp/qmt_daily_data/fast_factor_compute")
    ap.add_argument("--buyable", action="store_true", help="拟合用剔除了开盘封涨停的标签")
    ap.add_argument("--cluster", type=float, default=0.7)
    ap.add_argument("--max-families", type=int, default=100)
    ap.add_argument("--max-members", type=int, default=60)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--out", required=True, help="新信号写到这里(.npy)")
    ap.add_argument("--compare", action="append", default=[], help="name=path.npy,和新信号一起报告")
    ap.add_argument("--universe", default="", help="股票池掩码 (D, S) bool(scripts/universe.py):打分、"
                    "截面标准化、拟合、评估都只用池子里的票,输出信号池外为 NaN")
    a = ap.parse_args(argv)

    from alphacube.label import Label, open_vwap15
    from alphacube.research import warmed_view
    from alphacube.store import open_cube_store
    from select_factors import build_train_fingerprints

    t0 = time.time()
    d = Path(a.dir)
    view, _ = warmed_view(open_cube_store(a.store), "20180102", "20251231", warmup=60)
    price = open_vwap15(view)
    plain = Label("open_vwap15", 1).compute(view, price=price)
    clean = Label("open_vwap15", 1, buyable=True).compute(view, price=price)
    D, S = plain.shape
    split = max(1, min(D - 1, int(D * a.train_frac)))
    train = slice(0, split)
    universe = None
    if a.universe:
        universe = np.load(a.universe)
        if universe.shape != plain.shape:
            raise SystemExit(f"universe {universe.shape} != label {plain.shape}")
        plain[~universe] = np.nan
        clean[~universe] = np.nan
        print(f"股票池 {a.universe}:每天平均 {universe.sum(1).mean():.0f} 只", flush=True)
    dropped = np.isfinite(plain) & ~np.isfinite(clean)
    for name, sl in (("训练段", train), ("测试段", slice(split, D))):
        print(f"{name}:剔除的标签 {dropped[sl].sum()} 格,占有标签的 "
              f"{dropped[sl].sum() / max(np.isfinite(plain[sl]).sum(), 1):.3%}", flush=True)
    fit_target = clean if a.buyable else plain
    print(f"拟合用的标签:{'剔除开盘封涨停' if a.buyable else '原标签'} ({time.time() - t0:.0f}s)", flush=True)

    rows = healthy_rows(d)
    fp_path = d / f"fp_train_{FP_DAYS}x{FP_SYMBOLS}.npy"
    days = slice(max(0, split - FP_DAYS), split)
    cols = np.linspace(0, S - 1, num=min(FP_SYMBOLS, S)).astype(int)
    if universe is None:
        fp = np.asarray(build_train_fingerprints(d / "panels", [int(r["i"]) for r in rows], days, cols, fp_path))
    else:
        fp = np.asarray(universe_fingerprints(d / "panels", [int(r["i"]) for r in rows], days, universe,
                                              d / f"fp_train_{FP_DAYS}x_universe.npy"))
    if fp.shape[0] != len(rows):
        raise SystemExit(f"{fp_path} 有 {fp.shape[0]} 行,健康因子 {len(rows)} 个 —— 指纹和因子清单对不上")
    fams = pick_families(rows, fp, a.cluster, a.max_families, a.max_members)
    del fp
    fam_path = Path(a.out).with_suffix(".families.npy")
    sig = build(d, rows, fams, fit_target, train, save=fam_path, universe=universe)
    if universe is not None:
        sig[~universe] = np.nan
    np.save(a.out, sig.astype(np.float32))
    import json
    Path(a.out).with_suffix(".families.json").write_text(json.dumps(
        [{"members": [int(rows[j]["i"]) for j in idx], "example": rows[idx[0]]["expression"][:120]}
         for idx in fams], ensure_ascii=False), encoding="utf-8")
    print(f"族因子存到 {fam_path}", flush=True)
    print(f"写出 {a.out} ({time.time() - t0:.0f}s)", flush=True)

    signals = [("新信号", sig, fit_target)]
    for spec in a.compare:
        name, path = spec.split("=", 1)
        other = np.load(path).astype(np.float64)
        if universe is not None:
            other[~universe] = np.nan                   # 对照的信号也只在池子里比
        signals.append((name, other, plain if "buyable" not in path else clean))
    print(f"\n=== 测试段(半衰期在训练段选)===")
    print(f"{'信号':<10} {'评估标签':<10} | {'多空净夏普':>8} {'毛夏普':>7} {'换手':>6} {'平滑后':>7} {'hl':>3} "
          f"{'rankIC':>8} {'ICIR':>6} | {'纯多头超额夏普':>12} {'超额年化':>7} {'超额回撤':>7} {'hl':>3}")
    for name, s, ft in signals:
        for lab_name, tgt in (("原标签", plain), ("剔除封板", clean)):
            r = evaluate(s, tgt, split, ft)
            print(f"{name:<10} {lab_name:<10} | {r['ls_net']:+8.2f} {r['ls_gross']:+7.2f} {r['ls_turn']:6.1%} "
                  f"{r['ls_sm_net']:+7.2f} {r['ls_hl']:3g} {r['rank_ic']:+8.4f} {r['icir']:+6.3f} | "
                  f"{r['lo_ex_sharpe']:+12.2f} {r['lo_ex_ann']:+7.1%} {r['lo_ex_mdd']:+7.3f} {r['lo_hl']:3g}", flush=True)
    print(f"\n({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
