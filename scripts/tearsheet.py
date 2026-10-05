"""全样本表现:100 万从 2018-01-02 起滚到 2025-12-31,策略 vs 大盘,出指标和累计净值图。

策略 = docs/可交易回测.md §10 训练段选出的配置:新冠军(剔除封板标签拟合),平滑半衰期 5,
等权单票 0.5%,保留前 50%,不调 100%,上市不足 60 天不买,A 股实际费率,整手,开盘 15 分钟 VWAP 成交。
**信号权重和这些参数都是在 2023-08-08 之前拟合、挑选的:这之前是样本内,之后才是样本外**,指标分段报。

基准和策略用同一套成交价和时点(t+1 开盘 15 分钟 VWAP 到 t+2,后复权,含分红):
  等权全市场          所有有价股票等权,也就是超额一直用的基准
  流通市值加权全市场  按决策日的流通市值加权,相当于"大盘"
  沪深300(近似)       立方体里没有指数行情,用 daily_capital 搭的:每年 1 月、7 月第一个交易日调样,
                      在过去 250 天日均成交额前 50% 的票里,取日均总市值前 300 只,按流通市值加权

子命令:
  compute  (服务器)跑策略、算基准、打印指标,存 JSON
  plot     (本地)从 JSON 画图,中文字体用本机的
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

STRATEGY = dict(cap=0.005, band=1.0, keep=0.5, pool_frac=0.2, participation=0.1, min_age=60)
HALFLIFE = 5.0
SERIES = ("策略", "等权全市场", "流通市值加权全市场", "沪深300(近似)")


# ---------------------------------------------------------------------------
# 基准
# ---------------------------------------------------------------------------

def load_caps(cap_dir: str, dates: Sequence[str], symbols: Sequence[str]):
    """(D, S) 的流通市值、总市值(元),daily_capital/YYYYMMDD.parquet,没有的格子 NaN。"""
    import pandas as pd

    col = {s: i for i, s in enumerate(symbols)}
    D, S = len(dates), len(symbols)
    cmv = np.full((D, S), np.nan)
    tmv = np.full((D, S), np.nan)
    for t, d in enumerate(dates):
        f = Path(cap_dir) / f"{d}.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f, columns=["symbol", "circulating_market_value", "total_market_value"])
        idx = df["symbol"].map(col)
        ok = idx.notna().to_numpy()
        ii = idx[ok].astype(int).to_numpy()
        cmv[t, ii] = df["circulating_market_value"].to_numpy(dtype=np.float64)[ok]
        tmv[t, ii] = df["total_market_value"].to_numpy(dtype=np.float64)[ok]
    return cmv, tmv


def weighted_returns(ret: np.ndarray, w: np.ndarray) -> np.ndarray:
    """每天按 w 加权的平均收益,只算 ret 和 w 都有、w > 0 的格子。"""
    ok = np.isfinite(ret) & np.isfinite(w) & (w > 0)
    num = np.where(ok, ret * np.where(ok, w, 0.0), 0.0).sum(1)
    den = np.where(ok, w, 0.0).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def index_members(tmv: np.ndarray, amount: np.ndarray, list_day: np.ndarray, dates: Sequence[str],
                  n: int = 300, liquid: float = 0.5, window: int = 250, min_age: int = 90) -> np.ndarray:
    """(D, S) bool:沪深300 式的成分股近似。第一天和每年 1 月、7 月的第一个交易日调样,
    其余日子沿用。在上市满 min_age 天、过去 window 天日均成交额排前 liquid 的票里,
    取过去 window 天日均总市值最大的 n 只。"""
    from ls_anatomy import trailing_mean

    D, S = tmv.shape
    tmv_avg = trailing_mean(tmv, window)
    amt_avg = trailing_mean(amount, window)
    member = np.zeros((D, S), dtype=bool)
    cur = np.zeros(S, dtype=bool)
    for t in range(D):
        if t == 0 or (dates[t][:6] != dates[t - 1][:6] and dates[t][4:6] in ("01", "07")):
            ok = np.isfinite(tmv_avg[t]) & np.isfinite(amt_avg[t]) & ((t - list_day) >= min_age)
            idx = np.flatnonzero(ok)
            if idx.size:
                liq = idx[np.argsort(-amt_avg[t, idx], kind="stable")][: max(1, int(idx.size * liquid))]
                top = liq[np.argsort(-tmv_avg[t, liq], kind="stable")][:n]
                cur = np.zeros(S, dtype=bool)
                cur[top] = True
        member[t] = cur
    return member


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------

def _drawdown(nav: np.ndarray):
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1.0
    trough = int(np.argmin(dd))
    top = int(np.argmax(nav[: trough + 1]))
    rec = np.flatnonzero(nav[trough:] >= nav[top])
    return float(dd.min()), top, trough, (trough + int(rec[0])) if rec.size else None


def perf(r: np.ndarray, dates: Sequence[str], rb: Optional[np.ndarray] = None) -> Dict[str, object]:
    """一段日收益的常用指标。dates 是每个收益的结算日(和 r 等长)。无风险利率按 0 算。
    rb 给了就再算相对它的:超额年化(几何)、跟踪误差、信息比率、超额回撤、beta、alpha、上下行捕获。"""
    r = np.asarray(r, dtype=np.float64)
    n = r.size
    nav = np.cumprod(1.0 + r)
    yrs = n / 252.0
    mdd, top, trough, rec = _drawdown(nav)
    neg = np.minimum(r, 0.0)
    out: Dict[str, object] = {
        "start": dates[0], "end": dates[-1], "days": n,
        "total": float(nav[-1] - 1.0),
        "cagr": float(nav[-1] ** (1.0 / yrs) - 1.0),
        "vol": float(r.std() * math.sqrt(252)),
        "sharpe": float(r.mean() / r.std() * math.sqrt(252)),
        "sortino": float(r.mean() * 252 / (math.sqrt((neg ** 2).mean()) * math.sqrt(252))),
        "mdd": mdd, "mdd_peak": dates[top], "mdd_trough": dates[trough],
        "mdd_recover": dates[rec] if rec is not None else None,
        "win_day": float((r > 0).mean()),
    }
    out["calmar"] = out["cagr"] / abs(mdd) if mdd < 0 else float("nan")
    months: Dict[str, float] = {}
    for d, x in zip(dates, r):
        months[d[:6]] = (1.0 + months.get(d[:6], 0.0)) * (1.0 + x) - 1.0
    mv = np.array(list(months.values()))
    out.update(win_month=float((mv > 0).mean()), best_month=float(mv.max()), worst_month=float(mv.min()))
    if rb is not None:
        rb = np.asarray(rb, dtype=np.float64)
        ex = r - rb
        nav_b = np.cumprod(1.0 + rb)
        rel = nav / nav_b
        rel_mdd, rtop, rtrough, _ = _drawdown(rel)
        beta = float(np.cov(r, rb)[0, 1] / rb.var(ddof=1))
        up, dn = rb > 0, rb < 0
        out.update(
            ex_cagr=float(rel[-1] ** (1.0 / yrs) - 1.0),
            te=float(ex.std() * math.sqrt(252)),
            ir=float(ex.mean() / ex.std() * math.sqrt(252)),
            ex_mdd=rel_mdd, ex_mdd_peak=dates[rtop], ex_mdd_trough=dates[rtrough],
            beta=beta, alpha=float((r.mean() - beta * rb.mean()) * 252),
            up_capture=float(r[up].mean() / rb[up].mean()), down_capture=float(r[dn].mean() / rb[dn].mean()),
            win_day_ex=float((ex > 0).mean()))
    return out


def yearly(series: Dict[str, np.ndarray], dates: Sequence[str]) -> List[dict]:
    yrs = sorted({d[:4] for d in dates})
    y_of = np.array([d[:4] for d in dates])
    rows = []
    for y in yrs:
        m = y_of == y
        row = {"year": y, "days": int(m.sum())}
        for k, r in series.items():
            row[k] = float(np.prod(1.0 + r[m]) - 1.0)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# compute(服务器)
# ---------------------------------------------------------------------------

def compute(a: argparse.Namespace) -> int:
    import tradable_backtest as tb
    from alphacube.label import forward_return
    from alphacube.metrics import ema_rows

    mkt = tb.load_market(a.inputs)
    D, S = mkt.adj.shape
    split = max(1, min(D - 1, int(D * a.train_frac)))
    sig = np.load(a.signal).astype(np.float64)
    ret = forward_return(mkt.adj, 1, 1)                 # 决策日 t:t+1 开盘 15 分钟 VWAP -> t+2
    cfg = dict(STRATEGY)
    for k, v in (("cap", a.cap), ("keep", a.keep), ("band", a.band), ("pool_frac", a.pool), ("max_names", a.max_names)):
        if v is not None:
            cfg[k] = v
    hl = a.halflife if a.halflife is not None else HALFLIFE
    sm = ema_rows(sig, hl)
    universe = None
    uname = ""
    if a.universe:
        from universe import load_universe
        universe = load_universe(a.universe, (D, S))
        sm[~universe] = np.nan                           # 先平滑再圈池子
        uname = a.universe_name or universe_name(a.universe)
    t0 = 0
    if a.start:                                          # 滚动拟合的信号前面是空的:从有信号的那天开始投钱
        t0 = next((i for i, x in enumerate(mkt.dates) if str(x) >= a.start), D)
        if t0 >= split:
            raise SystemExit(f"--start {a.start} 不在训练段里")
    res = tb.simulate(mkt, sm, t0, D, capital=a.capital, costs=tb.RETAIL, compound=True, **cfg)
    t = res["t"].astype(int)
    end = [mkt.dates[i + 2] for i in t]                  # 每个日收益在 t+2 开盘后结算

    cmv, tmv = load_caps(a.capital_dir, mkt.dates, mkt.symbols)
    amount = np.load(a.amount)
    # 股票池就是沪深300 近似名单时,沪深300 基准直接用这份名单;别的池子(比如它的补集)另外搭真正的沪深300 近似
    member = universe if uname == "沪深300" else index_members(tmv, amount, mkt.list_day, mkt.dates)
    fin = np.isfinite(ret)
    r_ew = np.where(fin.any(1), np.nansum(ret, 1) / np.maximum(fin.sum(1), 1), np.nan)
    r_cw = weighted_returns(ret, cmv)
    r_300 = weighted_returns(ret, np.where(member, cmv, np.nan))
    if universe is None:
        series = {"策略": res["ret"], "等权全市场": r_ew[t], "流通市值加权全市场": r_cw[t], "沪深300(近似)": r_300[t]}
        base, lines = "等权全市场", list(SERIES)
    else:
        r_u = weighted_returns(ret, np.where(universe, 1.0, np.nan))
        base = "沪深300成分等权(近似)" if uname == "沪深300" else f"{uname}等权"
        series = {"策略": res["ret"], base: r_u[t], "沪深300(近似)": r_300[t], "等权全市场": r_ew[t]}
        lines = ["策略", base, "沪深300(近似)"]
    for k, v in series.items():
        if not np.isfinite(v).all():
            raise SystemExit(f"{k} 有 {int((~np.isfinite(v)).sum())} 个缺失日")

    segs = {"全样本": t >= 0, "样本内(训练段)": t < split, "样本外(测试段)": t >= split}
    metrics = {}
    for seg, m in segs.items():
        e = [d for d, k in zip(end, m) if k]
        metrics[seg] = {k: perf(v[m], e, None if k == base else series[base][m]) for k, v in series.items()}
        metrics[seg]["策略_vs_沪深300"] = perf(series["策略"][m], e, series["沪深300(近似)"][m])
        metrics[seg]["运作"] = {"turnover": float(res["turnover"][m].mean()), "fees_yr": float(res["fees"][m].mean() * 252),
                              "n_hold": float(res["n_hold"][m].mean()), "cash": float(res["cash"][m].mean())}
    out = {"capital": a.capital, "split_date": mkt.dates[split + 1], "config": {**cfg, "halflife": hl,
           "costs": tb.RETAIL.name, "signal": a.signal, "universe": a.universe, "universe_name": uname},
           "order": list(series), "base": base, "plot": lines, "title": a.title, "insample_note": a.insample_note,
           "dates": end, "series": {k: v.tolist() for k, v in series.items()},
           "n_hold": res["n_hold"].tolist(), "cash": res["cash"].tolist(), "turnover": res["turnover"].tolist(),
           "metrics": metrics, "yearly": yearly(series, end),
           "index_members_avg": float(member[t].sum(1).mean())}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, default=float), encoding="utf-8")
    report(out)
    print(f"\n写出 {a.out}")
    return 0


def report(out: dict) -> None:
    pct = lambda x: f"{x:+.1%}" if x is not None and np.isfinite(x) else "—"   # noqa: E731
    order, base = out.get("order", list(SERIES)), out.get("base", "等权全市场")
    for seg, ms in out["metrics"].items():
        s = ms["策略"]
        print(f"\n=== {seg}:{s['start']} ~ {s['end']}({s['days']} 天);超额相对 {base} ===")
        print(f"{'':<14} {'累计':>8} {'年化':>7} {'波动':>6} {'夏普':>6} {'最大回撤':>8} {'卡玛':>5} | "
              f"{'超额年化':>8} {'信息比率':>8} {'超额回撤':>8} {'beta':>5}")
        for k in order:
            m = ms[k]
            print(f"{k:<14} {m['total']:+8.1%} {m['cagr']:+7.1%} {m['vol']:6.1%} {m['sharpe']:+6.2f} {m['mdd']:+8.1%} "
                  f"{m['calmar']:5.2f} | {pct(m.get('ex_cagr'))} {m.get('ir', float('nan')):+8.2f} "
                  f"{pct(m.get('ex_mdd'))} {m.get('beta', float('nan')):5.2f}")
        v = ms["策略_vs_沪深300"]
        print(f"策略相对沪深300(近似):超额年化 {v['ex_cagr']:+.1%} 信息比率 {v['ir']:+.2f} beta {v['beta']:.2f}")
        o = ms["运作"]
        print(f"运作:日均单边换手 {o['turnover']:.2%},年化费用 {o['fees_yr']:.2%},平均持股 {o['n_hold']:.0f} 只,"
              f"平均现金 {o['cash']:.1%};策略最大回撤 {s['mdd_peak']} -> {s['mdd_trough']},"
              f"{'修复于 ' + s['mdd_recover'] if s['mdd_recover'] else '尚未修复'}")
    print("\n=== 分年 ===")
    print(f"{'年':<6} " + " ".join(f"{k:>14}" for k in order) + f" {'超额(对' + base + ')':>12}")
    for row in out["yearly"]:
        ex = (1 + row["策略"]) / (1 + row[base]) - 1
        print(f"{row['year']:<6} " + " ".join(f"{row[k]:+14.1%}" for k in order) + f" {ex:+12.1%}")


# ---------------------------------------------------------------------------
# plot(本地)
# ---------------------------------------------------------------------------

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLORS = {"策略": "#2a78d6", "等权全市场": "#eb6834", "流通市值加权全市场": "#1baf7a", "沪深300(近似)": "#eda100",
          "沪深300成分等权(近似)": "#e87ba4"}


def _color(k: str) -> str:
    """别的股票池的"池内等权"基准(比如"非沪深300等权")和沪深300 成分等权同一个颜色。"""
    return COLORS.get(k, COLORS["沪深300成分等权(近似)"])


def universe_name(path: str) -> str:
    """从名单旁边的 .json 读 rule:补集(universe.py --complement-of)叫"非沪深300",否则当作沪深300 近似名单。"""
    meta = Path(path).with_suffix(".json")
    rule = json.loads(meta.read_text(encoding="utf-8")).get("rule", "") if meta.exists() else ""
    return "非沪深300" if "补集" in rule else "沪深300"


def _cjk_font() -> Optional[str]:
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("PingFang SC", "Hiragino Sans GB", "Heiti TC", "STHeiti", "Songti SC", "Arial Unicode MS",
                 "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Micro Hei", "SimHei"):
        if name in have:
            return name
    return None


def plot(a: argparse.Namespace) -> int:
    import datetime as dt

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    out = json.loads(Path(a.json).read_text(encoding="utf-8"))
    font = _cjk_font()
    plt.rcParams.update({"font.sans-serif": [font] if font else [], "font.family": "sans-serif",
                         "axes.unicode_minus": False, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE, "font.size": 10})
    days = [dt.datetime.strptime(d, "%Y%m%d") for d in out["dates"]]
    split = dt.datetime.strptime(out["split_date"], "%Y%m%d")
    cap = out["capital"]
    series = {k: np.array(v) for k, v in out["series"].items()}
    nav = {k: cap * np.cumprod(1 + v) for k, v in series.items()}
    x0 = days[0] - (days[1] - days[0])                    # 起点:投入 100 万那天
    xs = [x0] + days

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True, gridspec_kw={"height_ratios": [3.2, 1.5, 1.5]})
    for ax in axes:
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.axvspan(split, xs[-1], color="#f0efec", zorder=0, linewidth=0)
        ax.axvline(split, color=MUTED, linewidth=1.0)

    # (1) 累计净值,对数轴,单位万元
    lines = out.get("plot", list(SERIES))
    base = out.get("base", "等权全市场")
    ax = axes[0]
    for k in lines:
        y = np.concatenate([[cap], nav[k]]) / 1e4
        ax.plot(xs, y, color=_color(k), linewidth=2.2 if k == "策略" else 1.6, label=k, zorder=3 if k == "策略" else 2)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 1.5, 2.0, 3.0, 5.0, 7.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("账户净值(万元,对数轴)")
    ends = sorted(lines, key=lambda k: -nav[k][-1])
    last_y = None
    for k in ends:                                        # 线尾直接标名称和期末值,挤在一起时往下错开
        y = nav[k][-1] / 1e4
        if last_y is not None and y > last_y / 1.12:
            y = last_y / 1.12
        ax.annotate(f"{k} {nav[k][-1] / 1e4:,.0f} 万", xy=(xs[-1], nav[k][-1] / 1e4), xytext=(xs[-1], y),
                    textcoords="data", xycoords="data", fontsize=9.5, color=INK, va="center",
                    ha="left", annotation_clip=False)
        ax.plot([xs[-1]], [nav[k][-1] / 1e4], marker="o", markersize=4.5, color=_color(k), zorder=4)
        last_y = y
    ms_all, ms_oos = out["metrics"]["全样本"], out["metrics"]["样本外(测试段)"]
    s_all, s_oos = ms_all["策略"], ms_oos["策略"]
    title = out.get("title") or f"100 万纯多头:{out['dates'][0][:4]}-{out['dates'][-1][:4]} 全样本(竖线右边是样本外)"
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=24)
    ax.text(0, 1.015, f"全样本 年化 {s_all['cagr']:+.1%} · 夏普 {s_all['sharpe']:.2f} · 最大回撤 {s_all['mdd']:.1%} · "
            f"相对{base} 超额年化 {s_all['ex_cagr']:+.1%} 信息比率 {s_all['ir']:.2f}    |    "
            f"样本外 年化 {s_oos['cagr']:+.1%} · 信息比率 {s_oos['ir']:.2f}",
            transform=ax.transAxes, fontsize=9.5, color=INK2)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK)
    ax.text(split, ax.get_ylim()[0] * 1.04, "  样本外 →", color=INK2, fontsize=9, va="bottom")
    note = out.get("insample_note") or "样本内:信号权重和参数在这一段拟合、挑选"
    ax.text(split, ax.get_ylim()[0] * 1.04, note + "  ", color=INK2, fontsize=9,
            va="bottom", ha="right")

    # (2) 相对强弱:策略 / 基准
    ax = axes[1]
    for k in dict.fromkeys((base, "沪深300(近似)")):
        rel = np.concatenate([[1.0], nav["策略"] / nav[k]])
        ax.plot(xs, rel, color=_color(k), linewidth=1.6, label=f"策略 ÷ {k}")
        ax.annotate(f"{rel[-1]:.2f}", xy=(xs[-1], rel[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=9, color=INK, va="center")
    ax.axhline(1.0, color=AXIS, linewidth=1.0)
    ax.set_ylabel("相对净值(起点 = 1)")
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)

    # (3) 回撤
    ax = axes[2]
    for k in dict.fromkeys(("策略", base, "沪深300(近似)")):
        v = np.concatenate([[cap], nav[k]])
        dd = v / np.maximum.accumulate(v) - 1
        ax.plot(xs, dd * 100, color=_color(k), linewidth=1.9 if k == "策略" else 1.3, label=k)
    ax.set_ylabel("回撤(%)")
    # 左下角压着 2018 年的回撤、右下角挨着 2024 年初的回撤,放在 2019 下半年到 2022 之间那块空白
    if out["dates"][0][:4] == "2018":
        ax.legend(loc="lower left", bbox_to_anchor=(0.19, 0.0), frameon=False, fontsize=9, labelcolor=INK, ncol=3)
    else:                                                 # 不是从 2018 年起(--start):那块空白不在原处,让它自己找
        ax.legend(loc="best", frameon=False, fontsize=9, labelcolor=INK, ncol=3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(xs[0], xs[-1])

    c = out["config"]
    uname = c.get("universe_name") or "沪深300"
    where = "" if not c.get("universe") else ("、只在沪深300 近似成分里选" if uname == "沪深300" else f"、只在{uname}的票里选")
    fig.text(0.01, 0.005, "基准与策略同一套成交价(次日开盘 15 分钟 VWAP)、含分红;沪深300 是用市值数据搭的近似(半年调样、流通市值加权)。"
             f"策略:等权单票 {c['cap']:.1%}{'、最多 ' + str(c['max_names']) + ' 只' if c.get('max_names') else ''}、"
             f"保留前 {c['keep']:.0%}、半衰期 {c['halflife']:g}、A 股实际费率(佣金最低 5 元)、整手"
             f"{where}。",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.015, 0.9, 1))
    fig.savefig(a.out, dpi=170)
    print(f"写出 {a.out}(字体 {font})")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute")
    c.add_argument("--inputs", default="_bench/tradable/market_20180102_20251231.npz")
    c.add_argument("--signal", default="_bench/champion_signal_buyable.npy")
    c.add_argument("--capital-dir", default="/root/autodl-tmp/qmt_daily_data/daily_capital")
    c.add_argument("--amount", default="_bench/tradable/amount_day.npy")
    c.add_argument("--capital", type=float, default=1e6)
    c.add_argument("--train-frac", type=float, default=0.7)
    c.add_argument("--out", default="_bench/tradable/fullsample_1m.json")
    c.add_argument("--universe", default="", help="股票池掩码(scripts/universe.py);给了就只在池子里选股,基准换成池内")
    c.add_argument("--cap", type=float, default=None)
    c.add_argument("--keep", type=float, default=None)
    c.add_argument("--band", type=float, default=None)
    c.add_argument("--pool", type=float, default=None)
    c.add_argument("--halflife", type=float, default=None)
    c.add_argument("--title", default="")
    c.add_argument("--start", default="", help="从这天(YYYYMMDD)开始投入资金;默认第一天")
    c.add_argument("--max-names", type=int, default=None, help="最多持有几只(默认不限)")
    c.add_argument("--universe-name", default="", help="股票池叫什么(默认从名单 .json 的 rule 判断:沪深300 / 非沪深300)")
    c.add_argument("--insample-note", default="", help="净值图上样本内那段的说明文字(默认:信号权重和参数在这一段拟合、挑选)")
    p = sub.add_parser("plot")
    p.add_argument("--json", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return compute(a) if a.cmd == "compute" else plot(a)


if __name__ == "__main__":
    raise SystemExit(main())
