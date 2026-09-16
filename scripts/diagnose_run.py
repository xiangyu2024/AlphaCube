"""为什么一次挖掘跑出 0 个有效候选:逐层定位,只读。

    PYTHONPATH=src python scripts/diagnose_run.py --store <计算根目录> [--start --end --max-symbols]

GP 的 fitness 只在三种情况下留 -inf:面板求值失败、IC 非有限、覆盖率低于闸值。
所以 "valid 0" 一定坏在这条链上的某一环,本脚本按顺序把每一环的数字打出来:

  1 store 轴与字段
  2 每个字段的有限值占比(以及开盘前 16 分钟单独看一眼 —— open_vwap15 只看这一段)
  3 label 的价格腿:open_vwap15 的三道闸各自筛掉多少(complete / 成交量>0 / 高低价区间)
  4 label 的收益腿:forward_return 之后还剩多少
  5 一批最简单的因子:覆盖率与 IC 各是多少

哪一环先掉到 0,问题就在那一环。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import warnings  # noqa: E402

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)


def pct(x) -> str:
    return f"{100.0 * float(x):6.2f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--max-symbols", type=int, default=60)
    ap.add_argument("--label", default="open_vwap15", choices=("open_vwap15", "close"))
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--min-coverage", type=float, default=0.5)
    a = ap.parse_args()

    from alphacube.engine.engine import CubeEngine
    from alphacube.label import OPEN_VWAP_WINDOW, Label, daily_close
    from alphacube.metrics import MIN_PAIRS, batched_ic
    from alphacube.parser import parse
    from alphacube.store import open_cube_store

    store = open_cube_store(Path(a.store))
    view = store.view(start=a.start, end=a.end, max_symbols=a.max_symbols)
    trad = view.tradable().T                                    # (D, S)
    cells = int(trad.sum())
    print(f"\n[1] store {a.store}")
    print(f"    {view.n_days} days {view.dates[0]}..{view.dates[-1]} x {view.n_symbols} symbols "
          f"x {view.n_minutes} minutes; fields {tuple(view.fields)}")
    print(f"    tradable stock-days {pct(trad.mean())} ({cells} cells)")
    if cells == 0:
        print("    >> 没有可交易的股票日,后面都免谈")
        return 1

    print(f"\n[2] 字段有限值占比(分母 = 可交易股票日 x 分钟)")
    lo_m, hi_m = OPEN_VWAP_WINDOW
    hi_m = min(hi_m, view.n_minutes)
    for f in view.fields:
        arr = view.read_s(f, 0, view.n_symbols)                 # (S, D, M)
        fin = np.isfinite(arr).transpose(1, 0, 2)[trad]         # (cells, M)
        head = fin[:, lo_m:hi_m]
        print(f"    {f:8s} 全天 {pct(fin.mean())}   开盘窗口[{lo_m}:{hi_m}) {pct(head.mean())}"
              f"   窗口内整段无缺 {pct(head.all(-1).mean())}")

    print(f"\n[3] label 价格腿 ({a.label})")
    if a.label == "open_vwap15":
        vol = view.read_s("Volume", 0, view.n_symbols)[..., lo_m:hi_m].astype(np.float64)
        amt = view.read_s("Amt", 0, view.n_symbols)[..., lo_m:hi_m].astype(np.float64)
        complete = (np.isfinite(vol).all(-1) & np.isfinite(amt).all(-1)).T
        tv = np.where(complete.T, vol.sum(-1), np.nan).T
        ta = np.where(complete.T, amt.sum(-1), np.nan).T
        with np.errstate(all="ignore"):
            v = np.where(tv > 0, ta / tv, np.nan)
        print(f"    闸1 窗口内 Volume/Amt 整段有限   {pct((complete & trad).sum() / cells)}")
        print(f"    闸2 且窗口成交量 > 0             {pct((np.isfinite(v) & trad).sum() / cells)}")
        if {"High", "Low"} <= set(view.fields):
            hi_p = np.nanmax(view.read_s("High", 0, view.n_symbols)[..., lo_m:hi_m].astype(np.float64), -1).T
            lo_p = np.nanmin(view.read_s("Low", 0, view.n_symbols)[..., lo_m:hi_m].astype(np.float64), -1).T
            v2 = np.where((v >= lo_p) & (v <= hi_p), v, np.nan)
            print(f"    闸3 且落在窗口 High/Low 之间     {pct((np.isfinite(v2) & trad).sum() / cells)}")
        price = Label(a.label, a.horizon).price(view)
    else:
        price = daily_close(view)
        print(f"    daily_close 有限                 {pct((np.isfinite(price) & trad).sum() / cells)}")
    price_ok = float((np.isfinite(price) & trad).sum() / cells)
    print(f"    价格腿最终有限占比               {pct(price_ok)}")

    target = Label(a.label, a.horizon).compute(view, price=price)
    tgt_fin = np.isfinite(target)
    tgt_cells = int(tgt_fin.sum())
    per_day = tgt_fin.sum(axis=1)                                # 每个交易日有多少只股票有 label
    eaten = 1 + a.horizon if a.label == "open_vwap15" else a.horizon
    print(f"\n[4] label 收益腿 horizon={a.horizon}: 有限占比 {pct(tgt_cells / cells)} "
          f"({tgt_cells} cells)   (未来一期缺失会自然吃掉最后 {eaten} 天)")
    print(f"    每日有限只数:中位数 {int(np.median(per_day))}  最小 {int(per_day.min())}  "
          f"最大 {int(per_day.max())}   IC 要求每日 >= MIN_PAIRS={MIN_PAIRS} 只")
    usable_days = int((per_day >= MIN_PAIRS).sum())
    print(f"    满足 MIN_PAIRS 的交易日 {usable_days}/{view.n_days}")
    if tgt_cells == 0:
        print("    >> label 全是 NaN:任何因子的 IC 都会是 NaN,GP 必然 valid 0。问题在这一层。")
        return 1
    if usable_days == 0:
        print(f"    >> 没有任何一天凑得齐 {MIN_PAIRS} 只:batched_ic 对每个候选都返回 NaN,GP 必然 valid 0。")
        return 1

    print(f"\n[5] 一批最简单的因子,用 GP 自己的闸(batched_ic + 覆盖率 >= {a.min_coverage},"
          f"分母 = label 的有限格点)")
    import torch
    exprs = ["day_mean(Close)", "day_std(Ret)", "day_sum(Volume)", "day_last(Close)",
             "cs_rank(day_mean(Close))", "ts_mean(day_last(Close), 5)",
             "day_std(in_mean_rolling(Close, 5))", "day_mean(nt_cs_rank(Ret))"]
    parsed = [parse(e) for e in exprs]
    res = CubeEngine(view).evaluate(parsed, tolerant=True)
    live = [i for i in range(len(exprs)) if i not in res.errors and res.panels[i] is not None]
    ics = {}
    if live:
        tensors = [torch.from_numpy(res.panels[i]).float() for i in live]
        vals = batched_ic(tensors, torch.from_numpy(np.ascontiguousarray(target)).float()).numpy()
        ics = dict(zip(live, vals.astype(float)))
    n_ok = 0
    for i, e in enumerate(exprs):
        if i not in ics:
            print(f"    {e:34s} 求值失败: {res.errors.get(i, 'panel is None')[:70]}")
            continue
        cov = float((np.isfinite(res.panels[i]) & tgt_fin).sum() / max(tgt_cells, 1))
        ic = ics[i]
        gate = ("IC 非有限(没有一天同时凑齐因子与 label)" if not np.isfinite(ic)
                else f"覆盖率 {cov:.2f} < {a.min_coverage}" if cov < a.min_coverage else "OK")
        n_ok += gate == "OK"
        print(f"    {e:34s} 覆盖率 {pct(cov)}  IC {ic:+.4f}  -> {gate}")
    print(f"\n[结论] {n_ok}/{len(exprs)} 条基础因子能过闸。", end=" ")
    print("链路是通的,若 GP 仍 valid 0,问题在搜索文法/参数而不是数据。" if n_ok
          else "一条都过不了 —— 按上面第一个掉到 0 的环去查。")
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
