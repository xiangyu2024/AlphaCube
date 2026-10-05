"""合成信号的十分位多空书报告:训练段 / 测试段分开,存每日收益,给文档留底。

口径和 build_champion.evaluate 完全一样:多前 10%、空后 10%,等权,总敞口 1(多 0.5 / 空 0.5),单边费用 15bp
(调仓的每一单位权重都付),EMA 平滑的半衰期在训练段按扣费多空夏普选;年化 = 日均 x 252(算术),
回撤在累计收益(相加)上算。另外把毛收益拆成两条腿:多头腿 = 前 10% 等权 − 当天全部有信号的票等权,
空头腿 = 全部等权 − 后 10% 等权,看多空差价从哪边来(两条腿相加 x 0.5 = 毛多空)。

  cd /root/autodl-tmp/AlphaCube && PYTHONPATH=src python -u scripts/ls_report.py \\
      --signal 原始因子库=_bench/legacy_signal.npy --signal 挖掘冠军=_bench/champion_signal_buyable.npy \\
      --target _bench/legacy/_target.npy --out _bench/tradable/ls_legacy_full.json
  # 本地画图(和全样本净值图同一个三联版式;--only 定顺序,第一个减第二个是中间栏)
  python3 scripts/ls_report.py plot --style tearsheet --json ls_legacy_full.json --only 原始因子库,挖掘冠军 \\
      --label 原始因子库=手写库,全市场 --label 挖掘冠军=挖掘冠军,全市场 --out docs/figures/ls_legacy_full.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

HALFLIVES = (0.0, 1, 2, 3, 5, 10, 20)


def leg_returns(sig: np.ndarray, fwd: np.ndarray, frac: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """(多头腿, 空头腿) 的毛日收益,都相对当天有信号、有标签的票等权。"""
    from alphacube.metrics import long_short_weights

    w = long_short_weights(sig, frac)
    ok = np.isfinite(sig) & np.isfinite(fwd)
    r = np.where(ok, fwd, 0.0)
    n = ok.sum(1)
    ew = np.where(n > 0, r.sum(1) / np.maximum(n, 1), np.nan)
    long_w, short_w = np.clip(w, 0, None) * 2, np.clip(-w, 0, None) * 2
    has = (long_w.sum(1) > 0) & (short_w.sum(1) > 0)
    long_r = np.where(has, (long_w * r).sum(1) - ew, np.nan)
    short_r = np.where(has, ew - (short_w * r).sum(1), np.nan)
    return long_r, short_r


def period_stats(net: np.ndarray, gross: np.ndarray, turn: np.ndarray, legs) -> Dict[str, float]:
    from long_only_grid import stats

    s = stats(net)
    out = {"ann": s["ann"], "sharpe": s["sharpe"], "mdd": s["mdd"], "gross_sharpe": stats(gross)["sharpe"],
           "gross_ann": stats(gross)["ann"], "turnover": float(np.nanmean(turn))}
    for name, r in zip(("long_leg", "short_leg"), legs):
        ls = stats(r)
        out[f"{name}_ann"], out[f"{name}_sharpe"] = ls["ann"], ls["sharpe"]
    return out


def report(sig: np.ndarray, target: np.ndarray, split: int, cost: float = 0.0015) -> dict:
    from alphacube.metrics import ema_rows, long_short_weights, portfolio_returns
    from long_only_grid import stats

    def book(s, days):
        return portfolio_returns(long_short_weights(s[days], 0.1), target[days], cost)

    train, test = slice(0, split), slice(split, sig.shape[0])
    sm = {hl: (ema_rows(sig, hl) if hl else sig) for hl in HALFLIVES}
    hl = max(HALFLIVES, key=lambda h: np.nan_to_num(stats(book(sm[h], train)[0])["sharpe"], nan=-9e9))
    out = {"halflife": hl}
    for tag, h in (("raw", 0.0), ("smoothed", hl)):
        legs = leg_returns(sm[h], target)
        out[tag] = {}
        for per, days in (("train", train), ("test", test)):        # 两段各自建书,和 build_champion 一样
            net, gross, turn = book(sm[h], days)
            out[tag][per] = period_stats(net, gross, turn, [x[days] for x in legs])
        net = book(sm[h], slice(None))[0]                           # 画图用:一路连着的全区间
        out[tag]["net"] = [None if not np.isfinite(x) else float(x) for x in net]
    return out


LINE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")


def plot(json_path: str, out: str, title: str = "") -> int:
    """本地画图:每个信号平滑后(训练段选的半衰期)的多空扣费累计收益(相加),竖线右边是测试段。"""
    import datetime as dt

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from tearsheet import AXIS, GRID, INK, INK2, MUTED, SURFACE, _cjk_font

    d = json.loads(Path(json_path).read_text(encoding="utf-8"))
    font = _cjk_font()
    plt.rcParams.update({"font.sans-serif": [font] if font else [], "font.family": "sans-serif",
                         "axes.unicode_minus": False, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE, "font.size": 10})
    days = [dt.datetime.strptime(x, "%Y%m%d") for x in d["dates"]]
    split = days[d["split"]]
    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=170)
    ax.axvspan(split, days[-1], color="#efeeea", zorder=0)
    ax.axvline(split, color=MUTED, lw=1)
    for (name, r), color in zip(d["signals"].items(), LINE_COLORS):
        net = np.array([np.nan if v is None else v for v in r["smoothed"]["net"]])
        cum = np.cumsum(np.nan_to_num(net)) * 100
        te = r["smoothed"]["test"]
        ax.plot(days, cum, color=color, lw=1.8, label=f"{name}(hl{r['halflife']:g};测试段 年化 {te['ann']:+.0%},"
                                                       f"夏普 {te['sharpe']:+.2f},回撤 {te['mdd']:+.0%})")
        ax.annotate(f"{name} {cum[-1]:+.0f}%", (days[-1], cum[-1]), xytext=(6, 0), textcoords="offset points",
                    fontsize=9, color=INK, va="center")
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_ylabel("累计收益(%,逐日相加,扣单边 15bp)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title(title or "十分位多空书(多前 10% / 空后 10%,总敞口 1)", loc="left", fontsize=13, color=INK, pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    ax.text(split, ax.get_ylim()[0], "  样本外 →", color=INK2, fontsize=9, va="bottom")
    fig.text(0.01, 0.01, "标签:次日开盘 15 分钟 VWAP 收益,剔除开盘封涨停买不进的;半衰期只在训练段选。"
             "多空书不受整手、涨跌停、融券约束,是信号质量的上限,不是可交易收益。", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 0.93, 1))
    fig.savefig(out)
    print(f"写出 {out}(字体 {font})")
    return 0


def plot_tearsheet(json_path: str, out: str, only: Sequence[str] = (), labels: Optional[Dict[str, str]] = None,
                   title: str = "") -> int:
    """和 tearsheet.py 全样本净值图同一个版式的三联图,画平滑后(训练段选的半衰期)的多空扣费收益:
    (1) 累计收益(逐日相加,和文档表格里的年化、回撤同一口径);(2) 两两之差(恰好两个信号时);(3) 回撤。
    标题下面的分段数字取 JSON 里两段各自建书的统计(= 文档表格),曲线是一路连着建书的全区间。年化、夏普两者
    只差测试段第一天的建仓费;回撤不一样:分段的样本外回撤从切分点重新算高点,曲线的回撤带着样本内的高点
    (沪深300 手写库 −26.4 对 −29.1),脚注里写明。"""
    d = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if only:
        missing = [k for k in only if k not in d["signals"]]
        if missing:
            raise SystemExit(f"{json_path} 里没有 {missing},有的是 {list(d['signals'])}")
        sigs = {k: d["signals"][k] for k in only}
    else:
        sigs = dict(d["signals"])

    import datetime as dt

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from tearsheet import AXIS, GRID, INK, INK2, MUTED, SURFACE, _cjk_font

    labels = labels or {}
    name = {k: labels.get(k, k) for k in sigs}
    color = {k: LINE_COLORS[i % len(LINE_COLORS)] for i, k in enumerate(sigs)}
    font = _cjk_font()
    plt.rcParams.update({"font.sans-serif": [font] if font else [], "font.family": "sans-serif",
                         "axes.unicode_minus": False, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE, "font.size": 10})
    days = [dt.datetime.strptime(x, "%Y%m%d") for x in d["dates"]]
    split = days[d["split"]]
    x0 = days[0] - (days[1] - days[0])                    # 起点:建书那天,累计收益 0
    xs = [x0] + days
    cum = {}
    for k, r in sigs.items():
        net = np.array([np.nan if v is None else v for v in r["smoothed"]["net"]], dtype=float)
        cum[k] = np.concatenate([[0.0], np.cumsum(np.nan_to_num(net))]) * 100

    pair = len(sigs) == 2
    n_panels = 3 if pair else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 11 if pair else 8.8), sharex=True,
                             gridspec_kw={"height_ratios": [3.2, 1.5, 1.5][:n_panels] if pair else [3.2, 1.5]})
    for ax in axes:
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.axvspan(split, xs[-1], color="#f0efec", zorder=0, linewidth=0)
        ax.axvline(split, color=MUTED, linewidth=1.0)

    # (1) 累计收益
    ax = axes[0]
    for i, k in enumerate(sigs):
        ax.plot(xs, cum[k], color=color[k], linewidth=2.2 if i == 0 else 1.8, label=f"{name[k]}(hl{sigs[k]['halflife']:g})",
                zorder=3 if i == 0 else 2)
    span = max(c.max() for c in cum.values()) - min(c.min() for c in cum.values()) or 1.0
    order = sorted(sigs, key=lambda k: -cum[k][-1])
    ys, last = [], None
    for k in order:                                       # 线尾直接标名称和期末值,挤在一起时往下错开
        y = cum[k][-1]
        if last is not None and y > last - 0.045 * span:
            y = last - 0.045 * span
        ys.append(y)
        last = y
    floor = min(c.min() for c in cum.values())
    if ys and ys[-1] < floor:                              # 错开后掉出坐标轴下沿:整串往上挪回来
        ys = [y + floor - ys[-1] for y in ys]
    for k, y in zip(order, ys):
        ax.annotate(f"{name[k]} {cum[k][-1]:+.0f}%", xy=(xs[-1], cum[k][-1]), xytext=(xs[-1] + dt.timedelta(days=12), y),
                    textcoords="data",
                    fontsize=9.5, color=INK, va="center", ha="left", annotation_clip=False)
        ax.plot([xs[-1]], [cum[k][-1]], marker="o", markersize=4.5, color=color[k], zorder=4)
    ax.set_ylabel("累计收益(%,逐日相加)")
    ax.set_title(title or f"十分位多空书,平滑后:{d['dates'][0][:4]}-{d['dates'][-1][:4]} 全样本(竖线右边是样本外)",
                 loc="left", fontsize=13, color=INK, pad=12 + 14 * len(sigs))
    for i, k in enumerate(sigs):
        tr, te = sigs[k]["smoothed"]["train"], sigs[k]["smoothed"]["test"]
        ax.text(0, 1.015 + 0.045 * (len(sigs) - 1 - i),
                f"{name[k]}:样本内 年化 {tr['ann']:+.1%} · 夏普 {tr['sharpe']:.2f} · 最大回撤 {tr['mdd']:.1%}    |    "
                f"样本外 年化 {te['ann']:+.1%} · 夏普 {te['sharpe']:.2f} · 最大回撤 {te['mdd']:.1%} · 换手 {te['turnover']:.1%}/天",
                transform=ax.transAxes, fontsize=9.5, color=color[k])
    ax.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK)
    lo = ax.get_ylim()[0]
    ax.text(split, lo, "  样本外 →", color=INK2, fontsize=9, va="bottom")
    ax.text(split, lo, "样本内:合成权重和半衰期在这一段拟合、挑选  ", color=INK2, fontsize=9, va="bottom", ha="right")

    # (2) 两者之差
    if pair:
        a_k, b_k = list(sigs)
        ax = axes[1]
        diff = cum[a_k] - cum[b_k]
        ax.plot(xs, diff, color=INK2, linewidth=1.6, label=f"{name[a_k]} 减 {name[b_k]}")
        ax.axhline(0.0, color=AXIS, linewidth=1.0)
        ax.annotate(f"{diff[-1]:+.0f}", xy=(xs[-1], diff[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=9, color=INK, va="center")
        ax.set_ylabel("累计收益之差(百分点)")
        # 差值从 0 起步:往上走时左上角是空的,往下走时左下角是空的(沪深300 那张往下走,放左上会压线)
        ax.legend(loc="upper left" if diff[-1] >= 0 else "lower left", frameon=False, fontsize=9, labelcolor=INK)

    # (3) 回撤:累计收益离此前最高点差多少(百分点,和表格里的回撤同一口径)
    ax = axes[-1]
    for i, k in enumerate(sigs):
        dd = cum[k] - np.maximum.accumulate(cum[k])
        ax.plot(xs, dd, color=color[k], linewidth=1.9 if i == 0 else 1.3, label=name[k])
    ax.set_ylabel("回撤(百分点)")
    ax.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK, ncol=len(sigs))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(xs[0], xs[-1])

    # 中文没有空格,matplotlib 的自动换行会在第一个空格处折断、还会压到横轴刻度:手动分两行
    fig.text(0.01, 0.006, f"多前 10% / 空后 10%,等权,总敞口 1(多 0.5 / 空 0.5),每天按信号重建、调仓单边 {d['cost'] * 1e4:g}bp;"
             "收益逐日相加(恒定敞口,不复利)。标签:次日开盘 15 分钟 VWAP 收益,剔除开盘封涨停买不进的。\n"
             "副标题的分段数字是每段单独建书算的(样本外回撤从切分点起算高点),曲线和回撤栏是全区间连着的一本书。"
             "多空书不受整手、涨跌停、融券约束,是信号质量的上限,不是可交易收益。", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 0.9, 1))
    fig.savefig(out, dpi=170)
    print(f"写出 {out}(字体 {font})")
    for k in sigs:
        dd = cum[k] - np.maximum.accumulate(cum[k])
        i = int(np.argmin(dd))
        j = int(np.argmax(cum[k][:i + 1]))
        print(f"  {name[k]}:期末累计 {cum[k][-1]:+.1f}%,全区间最大回撤 {dd[i]:.1f} 个百分点"
              f"({xs[j]:%Y-%m-%d} -> {xs[i]:%Y-%m-%d})")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "plot":
        pp = argparse.ArgumentParser()
        pp.add_argument("--json", required=True)
        pp.add_argument("--out", required=True)
        pp.add_argument("--title", default="")
        pp.add_argument("--style", choices=("simple", "tearsheet"), default="simple",
                        help="tearsheet = 和全样本净值图同一个三联版式(累计 / 两者之差 / 回撤)")
        pp.add_argument("--only", default="", help="只画这几个信号,逗号分隔,按给的顺序")
        pp.add_argument("--label", action="append", default=[], help="改图上的名字:原名=新名")
        b = pp.parse_args(argv[1:])
        if b.style == "tearsheet":
            only = [x.strip() for x in b.only.split(",") if x.strip()]
            bad = [x for x in b.label if "=" not in x]
            if bad:
                raise SystemExit(f"--label 要写成 原名=新名:{bad}")
            labels = dict(x.split("=", 1) for x in b.label)
            known = set(json.loads(Path(b.json).read_text(encoding="utf-8"))["signals"])
            if set(labels) - known:
                raise SystemExit(f"--label 里的 {sorted(set(labels) - known)} 不在 {b.json} 里,有的是 {sorted(known)}")
            return plot_tearsheet(b.json, b.out, only, labels, b.title)
        if b.only or b.label:
            raise SystemExit("--only / --label 只对 --style tearsheet 有用")
        return plot(b.json, b.out, b.title)
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", action="append", required=True, help="name=path.npy,可以给多个")
    ap.add_argument("--target", required=True, help="score_all_factors 存的 _target.npy(剔除封板的标签)")
    ap.add_argument("--dates", default="_bench/allfac/dates.npy")
    ap.add_argument("--universe", default="", help="股票池掩码:池外的信号和标签都当缺失")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--cost", type=float, default=0.0015)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    target = np.load(a.target).astype(np.float64)
    dates = [str(x) for x in np.load(a.dates)]
    D = target.shape[0]
    split = max(1, min(D - 1, int(D * a.train_frac)))
    member = np.load(a.universe) if a.universe else None
    if member is not None:
        target = np.where(member, target, np.nan)
    res: Dict[str, dict] = {}
    print(f"十分位多空书,单边 {a.cost:.2%},切分点 {dates[split]}" + (f",股票池 {a.universe}" if a.universe else ""))
    head = (f"{'信号':<14} {'段':<4} {'平滑':>5} | {'年化':>7} {'夏普':>6} {'回撤':>7} {'换手':>6} | {'毛夏普':>6} {'毛年化':>7} | "
            f"{'多头腿年化':>9} {'空头腿年化':>9}")
    print(head)
    for spec in a.signal:
        name, path = spec.split("=", 1)
        sig = np.load(path).astype(np.float64)
        if member is not None:
            sig = np.where(member, sig, np.nan)
        r = res[name] = report(sig, target, split, a.cost)
        for tag in ("raw", "smoothed"):
            for per in ("train", "test"):
                s = r[tag][per]
                lab = "原始" if tag == "raw" else f"hl{r['halflife']:g}"
                print(f"{name:<14} {('训练' if per == 'train' else '测试'):<4} {lab:>5} | {s['ann']:+7.1%} {s['sharpe']:+6.2f} "
                      f"{s['mdd']:+7.1%} {s['turnover']:6.1%} | {s['gross_sharpe']:+6.2f} {s['gross_ann']:+7.1%} | "
                      f"{s['long_leg_ann']:+9.1%} {s['short_leg_ann']:+9.1%}", flush=True)
    Path(a.out).write_text(json.dumps({"dates": dates, "split": split, "cost": a.cost, "universe": a.universe,
                                       "signals": res}, ensure_ascii=False), encoding="utf-8")
    print(f"写出 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
