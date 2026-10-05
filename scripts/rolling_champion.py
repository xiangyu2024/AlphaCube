"""两层合成的滚动版:族内、族间权重都每 step 天用"截至当天已经实现的最近 window 天"重新拟合。

静态版(build_champion.build)在训练段拟合一次族内 ridge、族间 lstsq,原样用到测试段两年多。线性合成很吃
regime shift:沪深300 里训练段的强因子到了测试段可能变弱、甚至反向。滚动版在测试段里也持续重拟合 ——
第 t 天的权重只用 t 天收盘时已经知道的标签,不算前视。

**标签什么时候知道**:标签 target[s] = OpenVwap15[s+2] / OpenVwap15[s+1] − 1(第 s 天的信号,s+1 开盘买、
s+2 开盘卖)。第 t 天收盘做决定时,已经实现的标签只到 s = t − 2,所以拟合窗口是 [t−1−window, t−1)。
(`combine/post_combine.rolling_combination` 用的是 [t−window, t),包含了 t−1 那天还没实现的标签。)

**每次重拟合做的事和静态版完全一样,只是数据换成窗口**:
1. 族内:成员因子(当天在股票池里截面标准化)对标签做 ridge(lam = 1e-3 x trace/k,和 fit_weights 一样);
2. 用新的族内权重算出窗口内的族因子,族间做 lstsq / ridge / 滚动 rank IC 加权;
3. 新权重用到接下来 step 天;下一次重拟合前不变。
**族间层之前先把族因子缩放到同一尺度**(除以窗口内的合并标准差,每次重拟合一个常数):单成员族的"族因子"就是
标准分(标准差约 1),多成员族是 ridge 拟合出来的收益预测(标准差约 1e-3);lstsq 不在乎列的尺度(缩放后结果不变),
ridge 的惩罚和 rank IC 加权却会被大尺度的族主导。

**交叉拟合(--crossfit K,可选)**:族内、族间都在同一个窗口里拟合时,族因子在窗口内是样本内的,成员越多、
窗口越短越虚高,族间层就偏重大族。K >= 2 时把窗口按时间切成 K 块,每块的族因子用"其余块拟合的族内权重"算
(块边界两侧各留 LABEL_LAG 天不进拟合),族间层只在这些样本外的族因子上拟合;用到下一段的族因子仍用整窗拟合的权重。

族的划分沿用 build_champion 的结果:聚类只用训练段指纹,挑成员、挑族用的 fit = |训练段 rank IC| − 0.14 x 训练段换手。
(2026-10-03 之前换手是全样本的,含测试段的因子值;那之前写出的 families.json 都带这个偏差,要用重新打分后的
build_champion 结果。)
训练段里的滚动预测还用到了"训练段整段聚出来的族",训练段数字带一点样本内成分。

**变体怎么挑**:窗口 x 族间方法 x 是否交叉拟合,全部在共同起点 s0 之后的训练段上比;纯多头以真实回测扫描的
训练段超额夏普为准(scripts/sweep_table.py --pick),多空以训练段平滑多空夏普为准(本脚本 JSON 里的 chosen_ls)。
测试段只报数。

**--mode oos(样本外堆叠,全市场用这个)**:上面那种每次重拟合都要把所有成员读进内存,在窗口里重算族因子 ——
全市场一个面板 40 MB,几百上千个成员装不下。oos 模式逐族处理:族内 ridge 照样每 step 天在 [t−1−W, t−1) 上拟合,
权重只往后用,得到每天的**样本外族因子**(落盘成 memmap);族间层在这些样本外族因子的历史上滚动拟合(同样只用
已实现的标签,且只用族因子已经有值的天,至少 min_between 天)。族因子天然是样本外的,不需要交叉拟合;
族间层同样先缩放到同一尺度。

  cd /root/autodl-tmp/AlphaCube && PYTHONPATH=src python -u scripts/rolling_champion.py \\
      --dir _bench/legacy_hs300 --families _bench/legacy_signal_hs300.families.json \\
      --universe _bench/universe/hs300_proxy_v1.npy --out-prefix _bench/universe/roll_legacy_hs300
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

RIDGE_SCALE = 1e-3          # 和 combine_two_level.fit_weights 一样
LABEL_LAG = 2               # target[s] 在 s + 2 天才实现(次日开盘买、再下一日开盘卖)
BETWEEN = ("lstsq", "ridge", "ric")
WINDOW_MODE_BUDGET = 16 * 1024 ** 3   # window 模式把所有成员(float64)读进内存;超过就要求用 --mode oos


def moments(cols: Sequence[np.ndarray], target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """逐日 (XᵀX, Xᵀy),缺失值记 0 —— 和 fit_weights 同口径(它把训练段所有格子摊平、NaN 记 0 再做正规方程),
    所以窗口内逐日相加 = 在这个窗口上调 fit_weights。返回带前导 0 的累计和,窗口和 = C[hi] − C[lo]。"""
    T = target.shape[0]
    k = len(cols)
    y = np.nan_to_num(target, nan=0.0)
    X = np.stack([np.nan_to_num(c, nan=0.0) for c in cols], axis=-1)          # (T, N, k)
    gram = np.matmul(X.transpose(0, 2, 1), X)
    rhs = np.matmul(X.transpose(0, 2, 1), y[:, :, None])[:, :, 0]
    cg = np.zeros((T + 1, k, k))
    cb = np.zeros((T + 1, k))
    np.cumsum(gram, axis=0, out=cg[1:])
    np.cumsum(rhs, axis=0, out=cb[1:])
    return cg, cb


def solve(G: np.ndarray, b: np.ndarray, method: str) -> np.ndarray:
    """fit_weights 的正规方程那一段,原样搬过来(ridge 的 lam 跟着 trace 走,奇异时退回最小范数解)。"""
    k = G.shape[0]
    if k == 1:
        return np.ones(1)
    lam = (RIDGE_SCALE if method == "ridge" else 0.0) * float(np.trace(G)) / max(k, 1)
    try:
        return np.linalg.solve(G + lam * np.eye(k), b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(G + lam * np.eye(k), rcond=1e-10) @ b


def combine(cols: Sequence[np.ndarray], w: np.ndarray) -> np.ndarray:
    """加权和;所有成员都缺值的格子留 NaN(和 combine_two_level.blend 一样)。"""
    acc = np.zeros(cols[0].shape, dtype=np.float64)
    seen = np.zeros(cols[0].shape, dtype=bool)
    for c, wi in zip(cols, w):
        fin = np.isfinite(c)
        acc += wi * np.where(fin, c, 0.0)
        seen |= fin
    acc[~seen] = np.nan
    return acc


def window(t: int, size: int, min_days: int) -> Optional[Tuple[int, int]]:
    """第 t 天收盘时能用的标签下标 [lo, hi):hi = t − LABEL_LAG + 1(最后一个已实现的标签是 t − 2)。
    size = 0 表示扩张窗口(从头开始,攒够 min_days 天才开始);固定窗口攒满 size 天就开始。"""
    hi = t - LABEL_LAG + 1
    lo = 0 if size == 0 else hi - size
    need = min_days if size == 0 else size
    if lo < 0 or hi - lo < need:
        return None
    return lo, hi


def rank_ic_weights(fam_window: Sequence[np.ndarray], target_window: np.ndarray) -> np.ndarray:
    """族因子在窗口里的逐日 rank IC 均值(带符号),按绝对值之和归一 —— post_combine roll_ric 的口径。"""
    from alphacube.metrics import cs_pearson, rank_rows

    tr = rank_rows(target_window)
    s = np.array([np.nanmean(cs_pearson(rank_rows(f), tr)) if np.isfinite(f).any() else 0.0 for f in fam_window])
    s = np.nan_to_num(s)
    tot = float(np.abs(s).sum())
    return s / tot if tot > 1e-12 else np.full(len(s), 1.0 / len(s))


def unit_scale(fam_win: Sequence[np.ndarray], fam_app: Sequence[np.ndarray]) -> Tuple[list, list]:
    """每个族因子除以它在窗口里的合并标准差(一个常数):族间 lstsq 不变,ridge / rank IC 加权不再被尺度主导。"""
    out_w, out_a = [], []
    for fw, fa in zip(fam_win, fam_app):
        sd = float(np.nanstd(fw)) if np.isfinite(fw).any() else 0.0
        k = 1.0 / sd if sd > 1e-15 else 1.0
        out_w.append(fw * k)
        out_a.append(fa * k)
    return out_w, out_a


def crossfit_family(mem: Sequence[np.ndarray], cg: np.ndarray, cb: np.ndarray, lo: int, hi: int, k: int) -> np.ndarray:
    """窗口 [lo, hi) 里的样本外族因子:按时间切 k 块,每块用其余块(块边界两侧各去掉 LABEL_LAG 天)拟合的族内 ridge。"""
    edges = np.linspace(lo, hi, k + 1).round().astype(int)
    G_all, b_all = cg[hi] - cg[lo], cb[hi] - cb[lo]
    out = np.full((hi - lo,) + mem[0].shape[1:], np.nan)
    for a, b in zip(edges[:-1], edges[1:]):
        pa, pb = max(lo, a - LABEL_LAG), min(hi, b + LABEL_LAG)
        w = solve(G_all - (cg[pb] - cg[pa]), b_all - (cb[pb] - cb[pa]), "ridge")
        out[a - lo:b - lo] = combine([m[a:b] for m in mem], w)
    return out


def rolling_two_level(members: Sequence[Sequence[np.ndarray]], target: np.ndarray, size: int, step: int = 20,
                      betweens: Sequence[str] = ("lstsq",), min_days: int = 250, member_moments=None,
                      crossfit: int = 0) -> Tuple[Dict[str, np.ndarray], List[dict]]:
    """members[f] = 第 f 族的成员面板(已截面标准化,(T, N),缺失 NaN)。返回 ({族间方法: 合成信号}, 重拟合记录)。

    同一个窗口下几种族间方法共用族内那一层(族内 ridge 只拟合一次)。第一次重拟合在第一个"窗口够 min_days 天"
    的交易日;之前的信号是 NaN。"""
    bad = [b for b in betweens if b not in BETWEEN]
    if bad:
        raise ValueError(f"族间方法只能是 {BETWEEN},不认识 {bad}")
    T, N = target.shape
    mom = member_moments if member_moments is not None else [moments(m, target) for m in members]
    out = {b: np.full((T, N), np.nan) for b in betweens}
    history: List[dict] = []
    starts = [t for t in range(T) if window(t, size, min_days) is not None]
    if not starts:
        raise ValueError(f"窗口 {size} / 最少 {min_days} 天:{T} 天里一次都拟合不了")
    for t in range(starts[0], T, step):
        lo, hi = window(t, size, min_days)
        stop = min(t + step, T)
        w_in = [solve(cg[hi] - cg[lo], cb[hi] - cb[lo], "ridge") for cg, cb in mom]
        if crossfit >= 2:
            fam_win = [crossfit_family(mem, cg, cb, lo, hi, crossfit) for mem, (cg, cb) in zip(members, mom)]
        else:
            fam_win = [combine([m[lo:hi] for m in mem], w) for mem, w in zip(members, w_in)]
        fam_app = [combine([m[t:stop] for m in mem], w) for mem, w in zip(members, w_in)]
        fam_win, fam_app = unit_scale(fam_win, fam_app)
        rec = {"t": t, "lo": lo, "hi": hi}
        X = y = None
        for b in betweens:
            if b == "ric":
                w_bt = rank_ic_weights(fam_win, target[lo:hi])
            else:
                if X is None:
                    X = np.stack([np.nan_to_num(f, nan=0.0).reshape(-1) for f in fam_win])
                    y = np.nan_to_num(target[lo:hi], nan=0.0).reshape(-1)
                w_bt = solve(X @ X.T, X @ y, b)
            out[b][t:stop] = combine(fam_app, w_bt)
            rec[b] = w_bt.tolist()
        history.append(rec)
    return out, history


def moments_chunked(cols: Sequence[np.ndarray], target: np.ndarray, chunk: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """和 moments() 一样的累计逐日矩,但按天分块、成员用 float32 叠 —— 全市场 60 个成员一次叠完要 4.9 GB。"""
    T = target.shape[0]
    k = len(cols)
    cg = np.zeros((T + 1, k, k))
    cb = np.zeros((T + 1, k))
    for a in range(0, T, chunk):
        b = min(T, a + chunk)
        X = np.stack([np.nan_to_num(c[a:b], nan=0.0).astype(np.float32) for c in cols], axis=-1)   # (d, N, k)
        y = np.nan_to_num(target[a:b], nan=0.0).astype(np.float32)
        Xt = X.transpose(0, 2, 1)
        cg[a + 1:b + 1] = np.matmul(Xt, X).astype(np.float64)
        cb[a + 1:b + 1] = np.matmul(Xt, y[:, :, None])[:, :, 0].astype(np.float64)
    np.cumsum(cg, axis=0, out=cg)
    np.cumsum(cb, axis=0, out=cb)
    return cg, cb


def oos_family_factor(members: Sequence[np.ndarray], mom, size: int, step: int, min_days: int,
                      out: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
    """族内 ridge 每 step 天在 [t−1−size, t−1) 上拟合、只往后用 -> 样本外族因子 (T, N);返回 (族因子, 第一次拟合的 t)。"""
    cg, cb = mom
    T, N = members[0].shape
    F = out if out is not None else np.full((T, N), np.nan, dtype=np.float32)
    F[:] = np.nan
    starts = [t for t in range(T) if window(t, size, min_days) is not None]
    if not starts:
        raise ValueError(f"窗口 {size} / 最少 {min_days} 天:{T} 天里一次都拟合不了")
    for t in range(starts[0], T, step):
        lo, hi = window(t, size, min_days)
        w = solve(cg[hi] - cg[lo], cb[hi] - cb[lo], "ridge")
        F[t:min(t + step, T)] = combine([m[t:min(t + step, T)] for m in members], w)
    return F, starts[0]


def oos_stack(F: np.ndarray, target: np.ndarray, size: int, t1: int, step: int = 20, betweens: Sequence[str] = ("lstsq",),
              min_days: int = 250, min_between: int = 250) -> Tuple[Dict[str, np.ndarray], List[dict]]:
    """F = (n_fam, T, N) 样本外族因子(t1 起有值)。族间层第 t 天用 [max(lo, t1), hi) 上的族因子和已实现标签拟合。"""
    from alphacube.metrics import cs_pearson, rank_rows

    bad = [b for b in betweens if b not in BETWEEN]
    if bad:
        raise ValueError(f"族间方法只能是 {BETWEEN},不认识 {bad}")
    n, T, N = F.shape
    y = np.nan_to_num(target, nan=0.0)
    gram = np.zeros((T + 1, n, n))
    rhs = np.zeros((T + 1, n))
    s1 = np.zeros((T + 1, n))
    s2 = np.zeros((T + 1, n))
    cnt = np.zeros((T + 1, n))
    for d in range(T):
        f = np.asarray(F[:, d, :], dtype=np.float64)
        fin = np.isfinite(f)
        f0 = np.where(fin, f, 0.0)
        gram[d + 1] = f0 @ f0.T
        rhs[d + 1] = f0 @ y[d]
        s1[d + 1], s2[d + 1], cnt[d + 1] = f0.sum(1), (f0 * f0).sum(1), fin.sum(1)
    for arr in (gram, rhs, s1, s2, cnt):
        np.cumsum(arr, axis=0, out=arr)
    ic = None
    if "ric" in betweens:
        tr = rank_rows(target)
        ic = np.stack([cs_pearson(rank_rows(np.asarray(F[f], dtype=np.float64)), tr) for f in range(n)])   # (n, T)
        cic = np.zeros((n, T + 1))
        cicn = np.zeros((n, T + 1))
        np.cumsum(np.nan_to_num(ic), axis=1, out=cic[:, 1:])
        np.cumsum(np.isfinite(ic), axis=1, out=cicn[:, 1:])
    out = {b: np.full((T, N), np.nan) for b in betweens}
    history: List[dict] = []
    for t in range(T):
        wnd = window(t, size, min_days)
        if wnd is None:
            continue
        lo, hi = max(wnd[0], t1), wnd[1]
        if hi - lo >= min_between:
            break
    else:
        raise ValueError("族间层一次都拟合不了(样本外族因子的天数不够)")
    for t in range(t, T, step):
        wnd = window(t, size, min_days)
        lo, hi = max(wnd[0], t1), wnd[1]
        stop = min(t + step, T)
        c = cnt[hi] - cnt[lo]
        mean = (s1[hi] - s1[lo]) / np.maximum(c, 1)
        sd = np.sqrt(np.maximum((s2[hi] - s2[lo]) / np.maximum(c, 1) - mean ** 2, 0.0))
        sc = np.where(sd > 1e-15, 1.0 / np.where(sd > 1e-15, sd, 1.0), 1.0)          # 统一尺度
        G = (gram[hi] - gram[lo]) * np.outer(sc, sc)
        b = (rhs[hi] - rhs[lo]) * sc
        seg = [np.asarray(F[f, t:stop], dtype=np.float64) * sc[f] for f in range(n)]
        rec = {"t": t, "lo": lo, "hi": hi}
        for m in betweens:
            if m == "ric":
                k = np.maximum(cicn[:, hi] - cicn[:, lo], 1)
                sco = np.where(cicn[:, hi] - cicn[:, lo] >= 2, (cic[:, hi] - cic[:, lo]) / k, 0.0)
                tot = float(np.abs(sco).sum())
                w = sco / tot if tot > 1e-12 else np.full(n, 1.0 / n)
            else:
                w = solve(G, b, m)
            out[m][t:stop] = combine(seg, w)
            rec[m] = w.tolist()
        history.append(rec)
    return out, history


def load_family(d: Path, fam: dict, cols: np.ndarray, universe: Optional[np.ndarray]) -> List[np.ndarray]:
    """一个族的成员面板(float32):给了股票池就池外置 NaN 再截面标准化,否则全市场截面标准化(和 build_champion 一样)。"""
    from eval_all_factors import drop_from_cache
    from alphacube.metrics import zscore_rows

    u = universe[:, cols] if universe is not None else None
    mem = []
    for i in fam["members"]:
        path = d / "panels" / f"{int(i):05d}.npy"
        pan = np.asarray(np.load(path, mmap_mode="r")[:, cols], dtype=np.float64)
        drop_from_cache(path.resolve())
        if u is not None:
            pan[~u] = np.nan
        mem.append(zscore_rows(pan).astype(np.float32))
    return mem


def load_members(d: Path, families: List[dict], cols: np.ndarray,
                 universe: Optional[np.ndarray]) -> List[List[np.ndarray]]:
    """每族成员面板只取进过股票池的列,池外置 NaN,逐日截面标准化(和 build_champion.build 一样);不给股票池就是全市场。"""
    from eval_all_factors import drop_from_cache
    from alphacube.metrics import zscore_rows

    u = universe[:, cols] if universe is not None else None
    out, t0 = [], time.time()
    for n, fam in enumerate(families):
        mem = []
        for i in fam["members"]:
            path = d / "panels" / f"{int(i):05d}.npy"
            pan = np.asarray(np.load(path, mmap_mode="r")[:, cols], dtype=np.float64)
            drop_from_cache(path.resolve())
            if u is not None:
                pan[~u] = np.nan
            mem.append(zscore_rows(pan))
        out.append(mem)
        if (n + 1) % 25 == 0:
            print(f"  读入 {n + 1}/{len(families)} 族 ({time.time() - t0:.0f}s)", flush=True)
    return out


def segment_stats(sig: np.ndarray, target: np.ndarray, s0: int, split: int) -> Dict[str, float]:
    """共同起点 s0 之后:训练段 [s0, split) / 测试段 [split, D) 的多空(平滑半衰期训练段选)和纯多头代理。"""
    from build_champion import evaluate
    import ls_report

    r = ls_report.report(sig[s0:], target[s0:], split - s0)
    ev = evaluate(sig[s0:], target[s0:], split - s0, target[s0:])
    sm = r["smoothed"]
    return {"hl": r["halflife"], "train_ls_sm": sm["train"]["sharpe"], "train_ls_sm_ann": sm["train"]["ann"],
            "test_ls_sm": sm["test"]["sharpe"], "test_ls_sm_ann": sm["test"]["ann"], "test_ls_mdd": sm["test"]["mdd"],
            "test_turn": sm["test"]["turnover"], "test_rank_ic": ev["rank_ic"], "test_icir": ev["icir"],
            "test_lo_ex": ev["lo_ex_sharpe"], "test_lo_ann": ev["lo_ex_ann"], "lo_hl": ev["lo_hl"]}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="打过分的目录(panels/、_target.npy,剔除封板、池外 NaN)")
    ap.add_argument("--families", required=True, help="build_champion 写的 <out>.families.json")
    ap.add_argument("--universe", default="", help="股票池掩码;不给 = 全市场(只有 --mode oos 支持)")
    ap.add_argument("--mode", choices=("window", "oos"), default="window",
                    help="window = 每次重拟合在窗口内重算族因子(成员全进内存);oos = 样本外堆叠(逐族处理,全市场用)")
    ap.add_argument("--min-between", type=int, default=250, help="oos:族间层至少要这么多天的样本外族因子")
    ap.add_argument("--keep-F", action="store_true", help="oos:保留样本外族因子的 memmap(每个窗口约 4 GB),默认用完就删")
    ap.add_argument("--static", default="", help="静态版信号(对照,同一个共同起点上评估)")
    ap.add_argument("--windows", default="", help="窗口天数,0 = 扩张窗口;默认 window 模式 250,500,750,0,oos 模式 250,500,0")
    ap.add_argument("--between", default="lstsq,ridge,ric")
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--min-days", type=int, default=250, help="扩张窗口至少攒够这么多天才开始")
    ap.add_argument("--crossfit", default="0,5", help="交叉拟合的块数,逗号分隔;0 = 不交叉拟合")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args(argv)

    t0 = time.time()
    d = Path(a.dir)
    target_full = np.load(d / "_target.npy").astype(np.float64)
    universe = None
    if a.universe:
        universe = np.load(a.universe)
        if universe.shape != target_full.shape:
            raise SystemExit(f"股票池 {universe.shape} 和标签 {target_full.shape} 对不上")
        target_full[~universe] = np.nan
        cols = np.flatnonzero(universe.any(0))
    else:
        cols = np.arange(target_full.shape[1])
    target = target_full[:, cols]
    if np.isfinite(target).sum() < 0.01 * target.size:
        raise SystemExit(f"股票池里几乎没有标签({int(np.isfinite(target).sum())} 格):{d}/_target.npy 是不是按别的池子打的分?")
    D = target.shape[0]
    split = max(1, min(D - 1, int(D * a.train_frac)))
    dates = [str(x) for x in np.load(d / "dates.npy")]
    families = json.loads(Path(a.families).read_text(encoding="utf-8"))
    n_mem = sum(len(f["members"]) for f in families)
    if a.mode == "window" and D * len(cols) * n_mem * 8 > WINDOW_MODE_BUDGET:
        raise SystemExit(f"window 模式要把 {n_mem} 个成员 x {D} 天 x {len(cols)} 只全读进内存"
                         f"(约 {D * len(cols) * n_mem * 8 / 1024 ** 3:.0f} GB);全市场或大股票池用 --mode oos")
    print(f"{len(families)} 族、{sum(len(f['members']) for f in families)} 个成员;{D} 天 x {len(cols)} 只"
          f"{'(进过池子的)' if universe is not None else '(全市场)'};切分点 {dates[split]};模式 {a.mode}", flush=True)
    windows = [int(x) for x in (a.windows or ("250,500,0" if a.mode == "oos" else "250,500,750,0")).split(",")]
    if a.mode == "oos":
        short = [w for w in windows if 0 < w < a.min_between]
        if short:                                         # 族间层在这些窗口里永远凑不够天数:先报,别等几小时
            raise SystemExit(f"窗口 {short} 比 --min-between {a.min_between} 短,族间层一次都拟合不了")
    methods = [x for x in a.between.split(",") if x]
    sigs: Dict[str, np.ndarray] = {}
    hist: Dict[str, list] = {}
    out_prefix = Path(a.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    if a.mode == "oos":
        Fs = {w: np.lib.format.open_memmap(f"{out_prefix}_F_w{w or 'exp'}.npy", mode="w+", dtype=np.float32,
                                           shape=(len(families), D, len(cols))) for w in windows}
        t1 = {}
        for f, fam in enumerate(families):
            mem = load_family(d, fam, cols, universe)
            mom_f = moments_chunked(mem, target)
            for w in windows:
                _, t1[w] = oos_family_factor(mem, mom_f, w, a.step, a.min_days, out=Fs[w][f])
            del mem, mom_f
            if (f + 1) % 20 == 0:
                print(f"  样本外族因子 {f + 1}/{len(families)} 族 ({time.time() - t0:.0f}s)", flush=True)
        for w in windows:
            Fs[w].flush()
            res, h = oos_stack(Fs[w], target, w, t1[w], a.step, methods, a.min_days, a.min_between)
            if not a.keep_F:                              # 中间结果,每个窗口约 4 GB
                path = Path(Fs[w].filename)
                del Fs[w]
                from eval_all_factors import drop_from_cache
                drop_from_cache(path)
                path.unlink()
            tag = f"oos_w{w or 'exp'}"
            hist[tag] = h
            for m in methods:
                sigs[f"{tag}_{m}"] = res[m]
            first = int(np.argmax(np.isfinite(res[methods[0]]).any(1)))
            print(f"  {tag}:族内 {dates[t1[w]]} 起、族间 {len(h)} 次重拟合,{dates[first]} 起有信号 "
                  f"({time.time() - t0:.0f}s)", flush=True)
    else:
        members = load_members(d, families, cols, universe)
        mom = [moments(m, target) for m in members]
        print(f"成员读入、逐日矩算完 ({time.time() - t0:.0f}s)", flush=True)
    for cf in ([int(x) for x in a.crossfit.split(",")] if a.mode == "window" else []):
        for w in windows:
            tag = f"w{w or 'exp'}" + (f"_cf{cf}" if cf >= 2 else "")
            res, h = rolling_two_level(members, target, w, a.step, methods, a.min_days, mom, crossfit=cf)
            hist[tag] = h
            for m in methods:
                sigs[f"{tag}_{m}"] = res[m]
            first = int(np.argmax(np.isfinite(res[methods[0]]).any(1)))
            print(f"  {tag}:{len(h)} 次重拟合,{dates[first]} 起有信号 ({time.time() - t0:.0f}s)", flush=True)

    s0 = max(int(np.argmax(np.isfinite(s).any(1))) for s in sigs.values())
    if s0 >= split - 60:
        raise SystemExit(f"共同起点 {dates[s0]} 离切分点 {dates[split]} 太近,训练段没法挑")
    print(f"\n共同起点 {dates[s0]}(之后所有变体都有信号);训练段 {split - s0} 天,测试段 {D - split} 天", flush=True)
    rows = {}
    if a.static:
        st = np.load(a.static).astype(np.float64)[:, cols]
        rows["静态(训练段一次拟合)"] = segment_stats(st, target, s0, split)
    for name, s in sigs.items():
        rows[name] = segment_stats(s, target, s0, split)
    print(f"{'变体':<20} {'hl':>3} | {'训练多空(平滑)':>12} | {'测试多空(平滑)':>12} {'年化':>7} {'回撤':>7} {'换手':>6} | "
          f"{'测试 rankIC':>10} {'ICIR':>6} | {'纯多头代理':>9} {'年化':>7}")
    for name, r in rows.items():
        print(f"{name:<20} {r['hl']:3g} | {r['train_ls_sm']:+12.2f} | {r['test_ls_sm']:+12.2f} {r['test_ls_sm_ann']:+7.1%} "
              f"{r['test_ls_mdd']:+7.1%} {r['test_turn']:6.1%} | {r['test_rank_ic']:+10.4f} {r['test_icir']:+6.3f} | "
              f"{r['test_lo_ex']:+9.2f} {r['test_lo_ann']:+7.1%}", flush=True)

    rolled = [k for k in rows if k in sigs]
    chosen_ls = max(rolled, key=lambda k: np.nan_to_num(rows[k]["train_ls_sm"], nan=-9e9))
    print(f"\n按训练段平滑多空夏普挑出:{chosen_ls}(训练 {rows[chosen_ls]['train_ls_sm']:+.2f} -> 测试 "
          f"{rows[chosen_ls]['test_ls_sm']:+.2f});纯多头按真实回测扫描的训练段挑(sweep_table.py --pick)", flush=True)
    for name, s in sigs.items():
        full = np.full(target_full.shape, np.nan, dtype=np.float32)
        full[:, cols] = s.astype(np.float32)
        if universe is not None:
            full[~universe] = np.nan
        np.save(f"{out_prefix}_{name}.npy", full)
    Path(f"{out_prefix}.json").write_text(json.dumps(
        {"dates": dates, "split": split, "s0": s0, "train_start": dates[s0], "step": a.step,
         "label_lag": LABEL_LAG, "stats": rows, "history": hist, "chosen_ls": chosen_ls,
         "families": [f.get("example", "") for f in families]}, ensure_ascii=False), encoding="utf-8")
    print(f"\n信号写到 {out_prefix}_<变体>.npy;权重历史和统计在 {out_prefix}.json ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
