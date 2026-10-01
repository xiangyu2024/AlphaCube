"""只为纯多头重新拟合族间权重。

冠军的族间权重是全截面 lstsq:回归误差平方和由两头的极端值主导,而夏普 4 的大头在空头腿
(docs/可交易回测.md;空头腿相对全市场 +49%/年,多头腿 +11%)。纯多头只交易头部,权重该
为"头部里谁更好"服务。这里换几种拟合目标,族因子(build_champion 存下的 .families.npy)不变:

  全截面            原做法(= 冠军)
  条件 前 50% / 30%  只用"当天冠军信号排在前 50% / 30%"的样本拟合 —— 在纯多头真正挑票的区域里分高下
  秩标签            标签换成当天截面秩的 z 分数,极端值不再主导
  条件前 30% + 秩标签

条件拟合的模型只见过头部样本,**不能拿去给全市场打分**:第一版直接这么用,训练段本身都是
负的(−1.4 / −3.2)—— 它把冠军排在尾部、自己从没见过的票推到了前面。所以条件拟合一律"两段式":
先用冠军(平滑后)筛出前 top,再在筛出来的票里按条件模型(同样平滑)排序;筛选外的票保持冠军
的次序,永远排在筛选内的后面。

全部只用训练段。每种权重都走一遍真实回测(100 万、A 股实际费率、上市不足 60 天不买、
等权 0.5%、不调 50%、保留前 30%,半衰期 {5, 10} 在训练段选),外加纯多头前 20%·exp1 和十分位
多空(项目口径)做参照。

  cd /root/autodl-tmp/AlphaCube && PYTHONPATH=src python -u scripts/long_side_fit.py \\
      --families _bench/champion_signal_buyable.families.npy --base _bench/champion_signal_buyable.npy \\
      --target _bench/allfac_buyable/_target.npy --inputs _bench/tradable/market_20180102_20251231.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def day_ranks(x: np.ndarray) -> np.ndarray:
    """一行的截面分位(0 = 最高),NaN 保持 NaN。"""
    out = np.full(x.shape, np.nan)
    f = np.isfinite(x)
    n = int(f.sum())
    if n:
        o = np.argsort(-x[f], kind="stable")
        r = np.empty(n)
        r[o] = (np.arange(n) + 0.5) / n
        out[f] = r
    return out


def rank_z(y: np.ndarray) -> np.ndarray:
    """一行标签 -> 截面秩的 z 分数(NaN 保持)。"""
    r = day_ranks(-y)                              # 秩:低 -> 高,方向不影响 z 分数的用法
    f = np.isfinite(r)
    if f.sum() < 2:
        return np.full(y.shape, np.nan)
    z = np.full(y.shape, np.nan)
    v = r[f]
    z[f] = (v - v.mean()) / v.std()
    return z


def fit_between(fams: np.ndarray, target: np.ndarray, base: np.ndarray, days: range,
                top: Optional[float] = None, rank_target: bool = False, ridge: float = 0.0) -> np.ndarray:
    """族间权重:逐日累加 Gram(X'X)和 X'y,只用 days 上(可选:base 排名前 top)的样本。
    约定和 combine_two_level.fit_weights 一样:因子缺失当 0,**标签缺失也当 0 收益**(不剔除样本),
    所以"全截面"那一行就是冠军本身的族间权重。"""
    k = fams.shape[0]
    G = np.zeros((k, k))
    b = np.zeros(k)
    for t in days:
        y = rank_z(target[t]) if rank_target else target[t]
        X = np.asarray(fams[:, t, :], dtype=np.float64)
        m = np.isfinite(X).any(0)
        if top is not None:
            m &= day_ranks(base[t]) <= top
        if m.sum() < 10:
            continue
        Xm = np.nan_to_num(X[:, m], nan=0.0)
        G += Xm @ Xm.T
        b += Xm @ np.nan_to_num(y[m], nan=0.0)
    lam = ridge * float(np.trace(G)) / max(k, 1)
    try:
        return np.linalg.solve(G + lam * np.eye(k), b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(G + lam * np.eye(k), rcond=1e-10) @ b


def two_stage(base: np.ndarray, inner: np.ndarray, top: float) -> np.ndarray:
    """(D, S):base 排名前 top 的票按 inner 排序、得分落在 (1, 2];其余按 base 排序、落在 [0, 1)。
    只有次序有意义。两边各自没有分的格子是 NaN。"""
    out = np.full(base.shape, np.nan)
    for t in range(base.shape[0]):
        pb = day_ranks(base[t])
        screen = pb <= top
        pi = np.full(base.shape[1], np.nan)
        m = screen & np.isfinite(inner[t])
        if m.any():
            pi[m] = day_ranks(np.where(m, inner[t], np.nan))[m]
        out[t] = np.where(screen, 2.0 - np.nan_to_num(pi, nan=1.0), 1.0 - pb)
    return out


def blend_days(fams: np.ndarray, w: np.ndarray) -> np.ndarray:
    k, D, S = fams.shape
    out = np.full((D, S), np.nan, dtype=np.float32)
    for t in range(D):
        X = np.asarray(fams[:, t, :], dtype=np.float64)
        fin = np.isfinite(X)
        acc = (np.where(fin, X, 0.0) * w[:, None]).sum(0)
        out[t] = np.where(fin.any(0), acc, np.nan)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", required=True)
    ap.add_argument("--base", required=True, help="冠军信号(条件拟合用它排名)")
    ap.add_argument("--target", required=True, help="拟合用的标签 (D, S),score_all_factors 存的 _target.npy")
    ap.add_argument("--inputs", required=True, help="tradable_backtest prepare 的 npz")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default="_bench/long_side")
    a = ap.parse_args(argv)

    import tradable_backtest as tb
    from alphacube.label import forward_return
    from alphacube.metrics import ema_rows, long_short_weights, portfolio_returns
    from long_only_grid import equal_weight_universe, long_only_weights, rank_order, stats

    t0 = time.time()
    fams = np.load(a.families)                     # 4 GB 整个读进内存,比逐天走内存映射快得多
    base = np.load(a.base).astype(np.float64)
    target = np.load(a.target).astype(np.float64)
    k, D, S = fams.shape
    split = max(1, min(D - 1, int(D * a.train_frac)))
    train_days = range(0, split)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{k} 个族因子 x {D} 天 x {S} 只;训练段 {split} 天", flush=True)

    variants = {"全截面": dict(), "条件前50%": dict(top=0.5), "条件前30%": dict(top=0.3),
                "秩标签": dict(rank_target=True), "条件前30%+秩标签": dict(top=0.3, rank_target=True)}
    signals: Dict[str, np.ndarray] = {}
    for name, kw in variants.items():
        w = fit_between(fams, target, base, train_days, **kw)
        sig = blend_days(fams, w)
        np.save(out_dir / f"{name}.npy", sig)
        signals[name] = sig.astype(np.float64)
        if name == "全截面":
            f = np.isfinite(sig) & np.isfinite(base)
            print(f"  核对:全截面和冠军信号的相关 {np.corrcoef(sig[f], base[f])[0, 1]:.6f}(应接近 1)", flush=True)
        print(f"  {name}: 权重 |w| 前 5 = {np.round(np.sort(np.abs(w))[::-1][:5], 4)} ({time.time() - t0:.0f}s)",
              flush=True)

    # ---- 评估
    mkt = tb.load_market(a.inputs)
    lab_fill = forward_return(mkt.adj, 1, 1)
    fin = np.isfinite(lab_fill)
    bench = np.where(fin.any(1), np.nansum(lab_fill, 1) / np.maximum(fin.sum(1), 1), np.nan)
    real = dict(capital=1e6, costs=tb.RETAIL, min_age=60, cap=0.005, band=0.5, keep=0.3)
    plain = forward_return(mkt.adj_label, 1, 1)
    bw = equal_weight_universe(plain)
    b_tr = portfolio_returns(bw[:split], plain[:split], 0.0)[0]
    b_te = portfolio_returns(bw[split:], plain[split:], 0.0)[0]
    print(f"\n=== 测试段;真实回测 = 100 万、A 股实际费率、等权 0.5%、不调 50%、保留前 30%(半衰期训练段选)===")
    print(f"{'拟合目标':<16} | {'真实 训练':>8} {'测试超额夏普':>10} {'超额年化':>7} {'超额回撤':>7} {'费用/年':>7} {'hl':>3} | "
          f"{'前20%exp1 测试':>12} | {'多空 测试':>8}")
    for name, sig in signals.items():
        top = variants[name].get("top")
        best = None
        for hl in (5.0, 10.0):
            sm = ema_rows(sig, hl)
            if top is not None:                   # 两段式:冠军筛头部,条件模型在头部里排
                sm = two_stage(ema_rows(base, hl), sm, top)
            tr = tb.summarize(tb.simulate(mkt, sm, 0, split, compound=False, **real), bench, 1e6)
            if best is None or tr["ex_sharpe"] > best[1]["ex_sharpe"]:
                best = (hl, tr, sm)
        hl, tr, sm = best
        te = tb.summarize(tb.simulate(mkt, sm, split, D, compound=True, **real), bench, 1e6)
        o, n = rank_order(sm)
        w_lo = long_only_weights(sm[split:], 0.2, 0, "exp1", cached=(o[split:], n[split:]))
        lo = stats(portfolio_returns(w_lo, plain[split:], 0.0015)[0] - b_te)["sharpe"]
        ls = stats(portfolio_returns(long_short_weights(sm[split:], 0.1), plain[split:], 0.0015)[0])["sharpe"]
        print(f"{name:<16} | {tr['ex_sharpe']:+8.2f} {te['ex_sharpe']:+10.2f} {te['ex_ann']:+7.1%} {te['ex_mdd']:+7.3f} "
              f"{te['fees_yr']:7.2%} {hl:3g} | {lo:+12.2f} | {ls:+8.2f}", flush=True)
    print(f"\n({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
