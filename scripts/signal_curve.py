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


def _ranked(score_t: np.ndarray, ret_t: np.ndarray, elig_t: np.ndarray):
    """先在 eligible & 有分的票里排分位(只用第 t 天已知的),再丢掉没有次日收益的(停牌到 t+2 的)。
    返回 (分位, 超额, 有收益的掩码),都对应排名的那批票;票太少返回 None。"""
    ok = elig_t & np.isfinite(score_t)
    n = int(ok.sum())
    if n < 100:
        return None
    s, r = score_t[ok], ret_t[ok]
    pct = np.empty(n)
    order = np.argsort(-s, kind="stable")
    pct[order] = (np.arange(n) + 0.5) / n
    has = np.isfinite(r)
    if has.sum() < 2:
        return None
    ex = np.where(has, r - r[has].mean(), np.nan)
    return pct, ex, has, order


def bucket_curve(score: np.ndarray, ret: np.ndarray, eligible: np.ndarray,
                 edges: Sequence[float] = EDGES) -> Dict[str, np.ndarray]:
    """每天在 eligible & 有分的票里排分位(0 = 信号最高;不看次日收益在不在),返回每桶有收益的票逐日的平均超额
    (相对当天这些票的等权),形状 (D, n_buckets),没票的格子 NaN。"""
    D = score.shape[0]
    nb = len(edges) - 1
    out = np.full((D, nb), np.nan)
    for t in range(D):
        rk = _ranked(score[t], ret[t], eligible[t])
        if rk is None:
            continue
        pct, ex, has, _ = rk
        for b in range(nb):
            m = has & (pct > edges[b]) & (pct <= edges[b + 1])
            if m.any():
                out[t, b] = ex[m].mean()
    return out


def topn_curve(score: np.ndarray, ret: np.ndarray, eligible: np.ndarray, ns: Sequence[int]) -> np.ndarray:
    """每天信号最强的前 N 只(在 eligible & 有分的票里排,不看次日收益在不在)里有收益的那些的等权超额,
    (D, len(ns))。对应"最多持 N 只、等权"的毛收益(不含成本、整手、成交量限制;停牌的票没有收益,不计入)。"""
    D = score.shape[0]
    out = np.full((D, len(ns)), np.nan)
    for t in range(D):
        rk = _ranked(score[t], ret[t], eligible[t])
        if rk is None:
            continue
        _, ex, has, order = rk
        for j, k in enumerate(ns):
            if k <= order.size:
                top = order[:k]
                top = top[has[top]]
                if top.size:
                    out[t, j] = ex[top].mean()
    return out


def summarize(curve: np.ndarray):
    """每桶 (日均 bp, t 值)。"""
    res = []
    for b in range(curve.shape[1]):
        v = curve[:, b]
        v = v[np.isfinite(v)]
        res.append((v.mean() * 1e4, v.mean() / v.std() * np.sqrt(v.size)) if v.size > 2 else (np.nan, np.nan))
    return res


def plot(json_path: str, out: str, title: str = "") -> int:
    """本地画图:每个信号一栏,横轴是信号分位档(左边信号最高),柱子是日均超额 bp,训练段浅色、测试段深色。"""
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tearsheet import AXIS, GRID, INK, INK2, MUTED, SURFACE, _cjk_font

    d = json.loads(Path(json_path).read_text(encoding="utf-8"))
    font = _cjk_font()
    plt.rcParams.update({"font.sans-serif": [font] if font else [], "font.family": "sans-serif",
                         "axes.unicode_minus": False, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE, "font.size": 10})
    edges = d["edges"]
    labels = [f"{lo * 100:g}-{hi * 100:g}%" for lo, hi in zip(edges[:-1], edges[1:])]
    names = list(d["signals"])
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 3.6 * len(names) + 0.8), sharex=True, squeeze=False)
    x = np.arange(len(labels))
    for ax, name in zip(axes[:, 0], names):
        rec = d["signals"][name]
        tr, te = np.array(rec["训练"]["bucket_bp"], float), np.array(rec["测试"]["bucket_bp"], float)
        ax.bar(x - 0.2, tr, width=0.38, color="#9cc3ee", label="训练段")
        ax.bar(x + 0.2, te, width=0.38, color="#2a78d6", label="测试段")
        for xi, v in zip(x, te):
            ax.annotate(f"{v:+.1f}", (xi + 0.2, v), xytext=(0, 3 if v >= 0 else -11), textcoords="offset points",
                        ha="center", fontsize=8, color=INK)
        ax.axhline(0, color=AXIS, lw=1)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_ylabel("次日超额(bp / 天)")
        ax.set_title(name, loc="left", fontsize=11, color=INK)
        ax.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK)
    axes[-1, 0].set_xticks(x)
    axes[-1, 0].set_xticklabels(labels, rotation=30, ha="right")
    axes[-1, 0].set_xlabel("信号分位(左边信号最高;每天只在买得进的票里排)")
    fig.suptitle(title or f"按信号分位的次日超额(hl{d['halflife']:g},相对当天买得进的票等权)", x=0.01, ha="left",
                 fontsize=13, color=INK)
    fig.text(0.01, 0.005, f"买得进 = 次日有价、开盘不封涨停、上市满 60 天;每天约 {d['n_eligible'][0]:.0f}(训练)/ "
             f"{d['n_eligible'][1]:.0f}(测试)只。毛收益,不含成本。", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out, dpi=170)
    print(f"写出 {out}(字体 {font})")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "plot":
        pp = argparse.ArgumentParser()
        pp.add_argument("--json", required=True)
        pp.add_argument("--out", required=True)
        pp.add_argument("--title", default="")
        b = pp.parse_args(argv[1:])
        return plot(b.json, b.out, b.title)
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--signal", action="append", required=True, help="name=path.npy")
    ap.add_argument("--halflife", type=float, default=10.0)
    ap.add_argument("--min-age", type=int, default=60)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--universe", default="", help="股票池掩码(scripts/universe.py):只在池子里排分位")
    ap.add_argument("--edges", default="", help="分位边界,逗号分隔;默认 0,.002,.005,.01,.02,.05,.1,.2,.5,1")
    ap.add_argument("--topn", default="", help="另报信号最强的前 N 只等权的超额,逗号分隔,如 50,100,300")
    ap.add_argument("--out", default="", help="结果另存 JSON(画图用)")
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
    ns = [int(x) for x in a.topn.split(",") if x]
    n_elig = eligible.sum(1)
    print(f"每天买得进的票:训练段平均 {n_elig[:split].mean():.0f} 只,测试段 {n_elig[split:].mean():.0f} 只")
    saved = {"edges": list(edges), "topn": ns, "split": split, "halflife": a.halflife,
             "n_eligible": [float(n_elig[:split].mean()), float(n_elig[split:].mean())], "signals": {}}
    for spec in a.signal:
        name, path = spec.split("=", 1)
        sig = np.load(path).astype(np.float64)
        sm = ema_rows(sig, a.halflife) if a.halflife else sig
        curve = bucket_curve(sm, ret, eligible, edges)
        print(f"\n=== {name}(hl{a.halflife:g})按信号分位的日均超额 bp(t 值)===\n分位: {head}")
        rec = {}
        for seg, sl in (("训练", slice(0, split)), ("测试", slice(split, D))):
            sm_ = summarize(curve[sl])
            print(f"{seg}: " + "  ".join(f"{m_:+5.1f}({t_:+4.1f})" for m_, t_ in sm_), flush=True)
            rec[seg] = {"bucket_bp": [m_ for m_, _ in sm_], "bucket_t": [t_ for _, t_ in sm_]}
        if ns:
            top = topn_curve(sm, ret, eligible, ns)
            print("前 N 只等权:" + "  ".join(f"N={k}" for k in ns))
            for seg, sl in (("训练", slice(0, split)), ("测试", slice(split, D))):
                st = summarize(top[sl])
                print(f"{seg}: " + "  ".join(f"{m_:+5.1f}({t_:+4.1f})" for m_, t_ in st), flush=True)
                rec[seg].update(topn_bp=[m_ for m_, _ in st], topn_t=[t_ for _, t_ in st])
        saved["signals"][name] = rec
    if a.out:
        import json
        Path(a.out).write_text(json.dumps(saved, ensure_ascii=False, default=float), encoding="utf-8")
        print(f"写出 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
