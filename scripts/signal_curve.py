"""信号强度和次日收益的关系:按截面分位分桶,只看买得进的票。

买得进 = 次日开盘不封涨停、上市满 60 天、次日有成交价。每天在这些票里排分位,算每桶相对
当天等权的日均超额(bp)和 t 值,训练段、测试段分开报。docs/可交易回测.md §5 那张"驼峰"表
就是它。

  cd /root/autodl-tmp/AlphaCube && PYTHONPATH=src python -u scripts/signal_curve.py \\
      --inputs _bench/tradable/market_20180102_20251231.npz --halflife 10 \\
      --signal 原冠军=_bench/champion_signal.npy --signal 新冠军=_bench/champion_signal_buyable.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

EDGES = (0, .002, .005, .01, .02, .05, .1, .2, .5, 1.0)


def bucket_curve(score: np.ndarray, ret: np.ndarray, eligible: np.ndarray,
                 edges: Sequence[float] = EDGES) -> Dict[str, np.ndarray]:
    """每天在 eligible & 有分 & 有收益的票里排分位(0 = 信号最高),返回每桶逐日的平均超额
    (相对当天这些票的等权),形状 (D, n_buckets),没票的格子 NaN。"""
    D = score.shape[0]
    nb = len(edges) - 1
    out = np.full((D, nb), np.nan)
    for t in range(D):
        ok = eligible[t] & np.isfinite(score[t]) & np.isfinite(ret[t])
        n = int(ok.sum())
        if n < 100:
            continue
        s, r = score[t, ok], ret[t, ok]
        pct = np.empty(n)
        pct[np.argsort(-s, kind="stable")] = (np.arange(n) + 0.5) / n
        ex = r - r.mean()
        for b in range(nb):
            m = (pct > edges[b]) & (pct <= edges[b + 1])
            if m.any():
                out[t, b] = ex[m].mean()
    return out


def summarize(curve: np.ndarray):
    """每桶 (日均 bp, t 值)。"""
    res = []
    for b in range(curve.shape[1]):
        v = curve[:, b]
        v = v[np.isfinite(v)]
        res.append((v.mean() * 1e4, v.mean() / v.std() * np.sqrt(v.size)) if v.size > 2 else (np.nan, np.nan))
    return res


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--signal", action="append", required=True, help="name=path.npy")
    ap.add_argument("--halflife", type=float, default=10.0)
    ap.add_argument("--min-age", type=int, default=60)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--universe", default="", help="股票池掩码(scripts/universe.py):只在池子里排分位")
    ap.add_argument("--edges", default="", help="分位边界,逗号分隔;默认 0,.002,.005,.01,.02,.05,.1,.2,.5,1")
    a = ap.parse_args(argv)

    import tradable_backtest as tb
    from alphacube.label import forward_return
    from alphacube.metrics import ema_rows

    m = tb.load_market(a.inputs)                       # 补过科创板开盘价,和真实回测同一套价格
    ret = forward_return(m.adj, 1, 1)
    D, S = ret.shape
    split = max(1, min(D - 1, int(D * a.train_frac)))
    eligible = np.zeros((D, S), dtype=bool)
    e = np.arange(1, D)
    eligible[:-1] = np.isfinite(m.adj[e]) & ~m.up_lock[e] & ((e[:, None] - m.list_day[None, :]) >= a.min_age)
    if a.universe:
        from universe import load_universe
        eligible &= load_universe(a.universe, (D, S))
    edges = tuple(float(x) for x in a.edges.split(",")) if a.edges else EDGES
    head = "  ".join(f"{lo * 100:g}-{hi * 100:g}%" for lo, hi in zip(edges[:-1], edges[1:]))
    for spec in a.signal:
        name, path = spec.split("=", 1)
        sig = np.load(path).astype(np.float64)
        sm = ema_rows(sig, a.halflife) if a.halflife else sig
        curve = bucket_curve(sm, ret, eligible, edges)
        print(f"\n=== {name}(hl{a.halflife:g})按信号分位的日均超额 bp(t 值)===\n分位: {head}")
        for seg, sl in (("训练", slice(0, split)), ("测试", slice(split, D))):
            print(f"{seg}: " + "  ".join(f"{m_:+5.1f}({t_:+4.1f})" for m_, t_ in summarize(curve[sl])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
