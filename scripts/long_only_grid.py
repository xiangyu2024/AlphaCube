"""纯多头的组合构建网格:选股规则 x 加权规则 x 平滑,全部扣费评估。

为什么只做纯多头:A 股融券做空个股券源少、成本高,多空的夏普落不了地。
所有历史结论(+3.98 等)都是多空口径,必须重测。

选股规则:前 5%/10%/20%,或固定只数 20/50/100/200。
加权规则:
  equal      等权
  rank       按在所选集合内的秩线性加权
  value      按因子值 MinMax 到 [0,1] 之后加权
  exp1/3/5   exp(beta x MinMax),beta 越大越集中在头部
  softmax    exp(beta x 组内 z 分数),对异常值比 MinMax 稳健
基准:等权全市场(当天有标签的所有股票),超额 = 组合净收益 - 基准净收益。
成本:单边 15bp,换手按 sum|dw|/2 计(纯多头总权重 1,就是单边换手)。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SELECTIONS: Tuple[Tuple[str, float, int], ...] = (
    ("前5%", 0.05, 0), ("前10%", 0.10, 0), ("前20%", 0.20, 0),
    ("前20只", 0.0, 20), ("前50只", 0.0, 50), ("前100只", 0.0, 100), ("前200只", 0.0, 200),
)
WEIGHTINGS = ("equal", "rank", "value", "exp1", "exp3", "exp5", "softmax")
HALFLIVES = (0.0, 1, 2, 3, 5, 10, 20)


def rank_order(alpha: np.ndarray):
    """一次性的降序排名索引和每天的有效只数。49 种构建共用,不要每种重排一遍
    (1942 x 5220 的 argsort 约 2 秒,392 次就是 13 分钟)。"""
    a = np.asarray(alpha, dtype=np.float64)
    fin = np.isfinite(a)
    return np.argsort(np.where(fin, a, -np.inf), axis=1, kind="stable")[:, ::-1], fin.sum(axis=1)


def long_only_weights(alpha: np.ndarray, frac: float, count: int, scheme: str,
                      cached=None) -> np.ndarray:
    """(D, S) 因子值 -> (D, S) 纯多头权重,每天总权重 1。"""
    a = np.asarray(alpha, dtype=np.float64)
    D, S = a.shape
    order, n = cached if cached is not None else rank_order(a)
    k = (np.full(D, count, dtype=np.int64) if count
         else np.maximum(1, np.floor(n * frac).astype(np.int64)))
    k = np.minimum(k, np.maximum(n, 1))
    assert order.shape[0] == D, "cached 的排序和传入的面板天数不一致"
    w = np.zeros((D, S), dtype=np.float64)
    for t in range(D):
        kk = int(k[t])
        if n[t] < 2 or kk < 1:
            continue
        pick = order[t, :kk]
        v = a[t, pick]
        if scheme == "equal":
            raw = np.ones(kk)
        elif scheme == "rank":
            raw = np.arange(kk, 0, -1, dtype=np.float64)        # 第 1 名权重最大
        else:
            lo, hi = float(v.min()), float(v.max())
            mm = (v - lo) / (hi - lo) if hi > lo else np.full(kk, 0.5)
            if scheme == "value":
                raw = mm
            elif scheme.startswith("exp"):
                raw = np.exp(float(scheme[3:]) * mm)
            elif scheme == "softmax":
                sd = float(v.std())
                z = (v - v.mean()) / sd if sd > 1e-12 else np.zeros(kk)
                raw = np.exp(np.clip(z, -10, 10))
            else:
                raise ValueError(scheme)
        tot = raw.sum()
        if tot <= 0:
            raw, tot = np.ones(kk), float(kk)
        w[t, pick] = raw / tot
    return w


def equal_weight_universe(target: np.ndarray) -> np.ndarray:
    fin = np.isfinite(target)
    n = np.maximum(fin.sum(axis=1), 1)
    return np.where(fin, 1.0 / n[:, None], 0.0)


def stats(net: np.ndarray) -> Dict[str, float]:
    f = net[np.isfinite(net)]
    if f.size < 3 or f.std() == 0:
        return {"sharpe": np.nan, "ann": np.nan, "mdd": np.nan}
    eq = np.cumsum(f)
    return {"sharpe": float(f.mean() / f.std() * np.sqrt(252)),
            "ann": float(f.mean() * 252),
            "mdd": float((eq - np.maximum.accumulate(eq)).min())}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("signals", nargs="+", help="信号文件 name=path.npy")
    ap.add_argument("--store", default="/root/autodl-tmp/qmt_daily_data/fast_factor_compute")
    ap.add_argument("--label", default="open_vwap15")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--cost", type=float, default=0.0015)
    ap.add_argument("--top", type=int, default=15, help="打印前多少行")
    a = ap.parse_args(argv)

    from alphacube.label import Label
    from alphacube.metrics import ema_rows, portfolio_returns
    from alphacube.research import warmed_view
    from alphacube.store import open_cube_store

    view, _ = warmed_view(open_cube_store(a.store), "20180102", "20251231", warmup=60)
    target = Label(a.label, 1).compute(view).astype(np.float64)
    D = target.shape[0]
    split = max(1, min(D - 1, int(D * a.train_frac)))
    train, test = slice(0, split), slice(split, D)
    bench_w = equal_weight_universe(target)
    bench_net = portfolio_returns(bench_w[test], target[test], 0.0)[0]
    bs = stats(bench_net)
    print(f"等权全市场基准(不计成本):夏普 {bs['sharpe']:+.2f} 年化 {bs['ann']:+.1%} "
          f"最大回撤 {bs['mdd']:+.3f}\n")

    for spec in a.signals:
        name, path = spec.split("=", 1)
        sig = np.load(path).astype(np.float64)
        smoothed = {hl: (ema_rows(sig, hl) if hl else sig) for hl in HALFLIVES}
        orders = {hl: rank_order(smoothed[hl]) for hl in HALFLIVES}   # 每个半衰期排一次,共用
        bench_train = portfolio_returns(bench_w[train], target[train], 0.0)[0]
        rows = []
        for label, frac, count in SELECTIONS:
            for scheme in WEIGHTINGS:
                # 半衰期在训练段上按**纯多头超额夏普**选
                best_hl, best_s = 0.0, -np.inf
                for hl in HALFLIVES:
                    w = long_only_weights(smoothed[hl][train], frac, count, scheme,
                                          cached=(orders[hl][0][train], orders[hl][1][train]))
                    net = portfolio_returns(w, target[train], a.cost)[0]
                    exc = net - bench_train
                    s = stats(exc)["sharpe"]
                    if np.isfinite(s) and s > best_s:
                        best_hl, best_s = hl, s
                w = long_only_weights(smoothed[best_hl][test], frac, count, scheme,
                                      cached=(orders[best_hl][0][test], orders[best_hl][1][test]))
                net, gross, turn = portfolio_returns(w, target[test], a.cost)
                exc = net - bench_net
                abs_, ex = stats(net), stats(exc)
                rows.append((ex["sharpe"], label, scheme, best_hl, abs_, ex, float(np.nanmean(turn))))
        rows.sort(reverse=True, key=lambda r: (r[0] if np.isfinite(r[0]) else -9e9))
        print(f"=== {name} ===")
        print(f"{'选股':>8} {'加权':>8} {'半衰期':>5} | {'超额夏普':>8} {'超额年化':>8} {'超额回撤':>8} "
              f"| {'绝对夏普':>8} {'绝对年化':>8} {'绝对回撤':>8} {'换手':>7}")
        for s, lab, sch, hl, ab, ex, to in rows[: a.top]:
            print(f"{lab:>8} {sch:>8} {hl:5.0f} | {ex['sharpe']:+8.2f} {ex['ann']:+8.1%} {ex['mdd']:+8.3f} "
                  f"| {ab['sharpe']:+8.2f} {ab['ann']:+8.1%} {ab['mdd']:+8.3f} {to:7.1%}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
