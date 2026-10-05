"""因子在单只股票上是不是"持续选错":选进来却一直亏 / 没选它却一直赚,分开看。

两种错误:
- **高分时亏**(假阳性):第 t 天信号排在前 q_hi(默认 5%,接近实盘持仓)的票,过去 window 天里"被选中那些天"的
  平均超额 hi_err。如果 hi_err 低的票今天被选中时也更差 —— 组合里应该排除它们(黑名单)。
- **低分时赚**(假阴性):排名在 lo_from(默认 20%,候选池之外)以后的票,过去 window 天里"排名低那些天"的平均超额
  lo_err。如果 lo_err 高的票今天还在赚、而且赚得比已选中的票多 —— 组合里应该额外选入(白名单)。

超额 = 次日收益 − 当天"买得进、有信号"的票的等权(和 signal_curve.py 同口径;买得进 = 次日有价、开盘不封涨停、
上市满 60 天)。信号先按实盘用的半衰期平滑,每天在买得进的票里排分位。

**只用已实现的标签**:第 t 天的历史统计只用 s <= t − 2 的超额(标签 target[s] 到 s + 2 天才实现)。

输出:
1. 逐日样本外检验:今天被选中的票按 hi_err 三等分,各组今天的平均超额(bp / 天,t 值);低分票里 lo_err 最高的 5%
   今天的平均超额,和已选中的票比;训练段 / 测试段分开。
2. 逐年持续性:每只股票相邻两年 hi_err(lo_err)的截面相关。
3. 可选 --write-prefix:写出黑名单 / 白名单修改后的(已平滑)信号,给 tradable_backtest.py sweep --halflives 0 用。

  cd /root/autodl-tmp/AlphaCube && PYTHONPATH=src python -u scripts/stock_errors.py \\
      --inputs _bench/tradable/market_20180102_20251231.npz --signal 挖掘冠军=_bench/champion_signal_buyable.npy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

LABEL_LAG = 2


def daily_pct(score: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """每天在 eligible & 有分的票里排分位(0 = 信号最高),其余 NaN。"""
    D, S = score.shape
    pct = np.full((D, S), np.nan)
    for t in range(D):
        ok = eligible[t] & np.isfinite(score[t])
        n = int(ok.sum())
        if n == 0:
            continue
        idx = np.flatnonzero(ok)
        order = idx[np.argsort(-score[t, ok], kind="stable")]
        pct[t, order] = (np.arange(n) + 0.5) / n
    return pct


def excess(ret: np.ndarray, pct: np.ndarray) -> np.ndarray:
    """次日收益减去当天有分位的票的等权;没分位的格子 NaN。"""
    ok = np.isfinite(pct) & np.isfinite(ret)
    r = np.where(ok, ret, np.nan)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(r, axis=1, keepdims=True)
    return np.where(ok, r - mean, np.nan)


def trailing(ex: np.ndarray, mask: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """第 t 天:过去 window 天(只到 t − 2)里 mask 为真那些天的超额之和、天数 -> (均值, 天数),(D, S)。"""
    D, S = ex.shape
    hit = mask & np.isfinite(ex)
    cs = np.zeros((D + 1, S))
    cn = np.zeros((D + 1, S))
    np.cumsum(np.where(hit, ex, 0.0), axis=0, out=cs[1:])
    np.cumsum(hit, axis=0, out=cn[1:])
    hi = np.clip(np.arange(D) - LABEL_LAG + 1, 0, D)             # 只到 t − 2:下标 < t − 1
    lo = np.clip(hi - window, 0, D)
    s, n = cs[hi] - cs[lo], cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan), n


def group_means(ex: np.ndarray, member: np.ndarray, key: np.ndarray, n: np.ndarray, min_n: int,
                cuts: Sequence[float]) -> np.ndarray:
    """每天在 member 里,按 key(历史成绩,至少 min_n 天)的截面分位 cuts 分组,返回各组今天的平均超额
    (D, len(cuts)-1 + 1);最后一列是"没有足够历史"的那组。"""
    D = ex.shape[0]
    k = len(cuts) - 1
    out = np.full((D, k + 1), np.nan)
    for t in range(D):
        m = member[t] & np.isfinite(ex[t])
        has = m & (n[t] >= min_n) & np.isfinite(key[t])
        if has.sum() >= 3 * k:
            q = np.quantile(key[t, has], cuts)
            for g in range(k):
                lo_, hi_ = q[g], q[g + 1]
                gm = has & (key[t] >= lo_) & ((key[t] < hi_) if g < k - 1 else (key[t] <= hi_))
                if gm.any():
                    out[t, g] = ex[t, gm].mean()
        no = m & ~has
        if no.any():
            out[t, k] = ex[t, no].mean()
    return out


def mean_t(x: np.ndarray) -> Tuple[float, float, int]:
    v = x[np.isfinite(x)]
    if v.size < 3 or v.std() == 0:
        return np.nan, np.nan, int(v.size)
    return float(v.mean() * 1e4), float(v.mean() / v.std() * np.sqrt(v.size)), int(v.size)


def yearly_persistence(ex: np.ndarray, mask: np.ndarray, years: np.ndarray, min_n: int) -> list:
    """每只股票每年 mask 那些天的平均超额;相邻两年都有至少 min_n 天的票,做截面相关(Pearson / Spearman)。"""
    from scipy.stats import spearmanr

    ys = sorted(set(years.tolist()))
    hit = mask & np.isfinite(ex)
    per = {}
    for y in ys:
        rows = years == y
        s = np.where(hit[rows], ex[rows], 0.0).sum(0)
        c = hit[rows].sum(0)
        per[y] = (np.where(c >= min_n, s / np.maximum(c, 1), np.nan), c)
    out = []
    for a, b in zip(ys[:-1], ys[1:]):
        x, y_ = per[a][0], per[b][0]
        ok = np.isfinite(x) & np.isfinite(y_)
        if ok.sum() >= 30:
            out.append({"from": int(a), "to": int(b), "n": int(ok.sum()), "pearson": float(np.corrcoef(x[ok], y_[ok])[0, 1]),
                        "spearman": float(spearmanr(x[ok], y_[ok]).correlation)})
    return out


def modified_signals(sm: np.ndarray, pct: np.ndarray, hi_err: np.ndarray, n_hi: np.ndarray, lo_err: np.ndarray,
                     n_lo: np.ndarray, min_hi: int, min_lo: int, black_frac: float, white_frac: float,
                     lo_from: float) -> Dict[str, np.ndarray]:
    """黑名单:当天"有足够历史"的票里 hi_err 最差的 black_frac 不选(信号置 NaN);
    白名单:排名在 lo_from 之后、lo_err 最好的 white_frac(占当天有分位的票)强制排到当天最前面,彼此按 lo_err 排
    (抬到某个分位进不了组合:保留区的老持仓和更靠前的新票先把钱用掉了)。pct 只能用第 t 天收盘时已知的信息排
    (不能用"次日是否封板"),历史统计 hi_err / lo_err 本来就只到 t − 2。返回 black、white 和白名单掩码 white_mask。"""
    D, S = sm.shape
    black = sm.copy()
    white = sm.copy()
    mask = np.zeros((D, S), dtype=bool)
    for t in range(D):
        has = np.isfinite(pct[t]) & (n_hi[t] >= min_hi) & np.isfinite(hi_err[t])
        if has.sum() >= 50:
            thr = np.quantile(hi_err[t, has], black_frac)
            black[t, has & (hi_err[t] <= thr)] = np.nan
        cand = np.isfinite(pct[t]) & (pct[t] > lo_from) & (n_lo[t] >= min_lo) & np.isfinite(lo_err[t])
        n_all = int(np.isfinite(pct[t]).sum())
        k = int(round(white_frac * n_all))
        if k > 0 and cand.sum() > k:
            idx = np.flatnonzero(cand)
            top = idx[np.argsort(-lo_err[t, idx], kind="stable")[:k]]
            fin = np.isfinite(sm[t])
            mx, sd = float(np.max(sm[t, fin])), float(np.std(sm[t, fin])) or 1.0
            white[t, top] = mx + sd * 1e-3 * np.arange(k, 0, -1)     # 最前面,lo_err 越好越靠前
            mask[t, top] = True
    return {"black": black, "white": white, "white_mask": mask}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--signal", action="append", required=True, help="name=path.npy")
    ap.add_argument("--halflife", type=float, default=5.0)
    ap.add_argument("--q-hi", type=float, default=0.05, help="排名前这一比例算'被选中'")
    ap.add_argument("--lo-from", type=float, default=0.2, help="排名在这之后算'低分'(候选池之外)")
    ap.add_argument("--window", type=int, default=250)
    ap.add_argument("--min-hi", type=int, default=20, help="被选中至少这么多天才算有历史")
    ap.add_argument("--min-lo", type=int, default=100)
    ap.add_argument("--min-age", type=int, default=60)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--black-frac", type=float, default=0.2)
    ap.add_argument("--white-frac", type=float, default=0.02)
    ap.add_argument("--write-prefix", default="", help="写出 <prefix>_<name>_{base,black,white}.npy(已平滑)")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    import tradable_backtest as tb
    from alphacube.label import forward_return
    from alphacube.metrics import ema_rows

    m = tb.load_market(a.inputs)
    ret = forward_return(m.adj, 1, 1)
    D, S = ret.shape
    split = max(1, min(D - 1, int(D * a.train_frac)))
    eligible = np.zeros((D, S), dtype=bool)
    e = np.arange(1, D)
    eligible[:-1] = np.isfinite(m.adj[e]) & ~m.up_lock[e] & ((e[:, None] - m.list_day[None, :]) >= a.min_age)
    years = np.array([int(str(d)[:4]) for d in m.dates])
    segs = (("训练", slice(0, split)), ("测试", slice(split, D)))
    report: Dict[str, dict] = {}
    for spec in a.signal:
        name, path = spec.split("=", 1)
        sm = ema_rows(np.load(path).astype(np.float64), a.halflife)
        pct = daily_pct(sm, eligible)
        ex = excess(ret, pct)
        sel = pct <= a.q_hi
        low = pct > a.lo_from
        hi_err, n_hi = trailing(ex, sel, a.window)
        lo_err, n_lo = trailing(ex, low, a.window)
        rec: Dict[str, dict] = {}
        print(f"\n=== {name}(hl{a.halflife:g};选中 = 前 {a.q_hi:.0%},低分 = {a.lo_from:.0%} 以后;历史窗口 {a.window} 天)===")

        g_sel = group_means(ex, sel, hi_err, n_hi, a.min_hi, (0, 1 / 3, 2 / 3, 1))
        all_sel = np.array([np.nanmean(ex[t, sel[t]]) if sel[t].any() else np.nan for t in range(D)])
        print("1) 高分时亏:今天被选中的票,按'过去被选中时的平均超额'三等分 -> 今天的平均超额 bp(t 值)")
        print(f"   {'':4} {'历史最差 1/3':>14} {'中间 1/3':>14} {'历史最好 1/3':>14} {'历史不够':>14} {'全部选中':>14} {'最差−其余':>14}")
        for seg, sl in segs:
            cols = [mean_t(g_sel[sl, j]) for j in range(4)] + [mean_t(all_sel[sl])]
            diff = g_sel[sl, 0] - np.nanmean(g_sel[sl, 1:3], axis=1)
            cols.append(mean_t(diff))
            print(f"   {seg:4} " + " ".join(f"{c[0]:+8.1f}({c[1]:+4.1f})" for c in cols), flush=True)
            rec[f"hi_{seg}"] = {"worst": cols[0], "mid": cols[1], "best": cols[2], "nohist": cols[3], "all": cols[4],
                                "worst_minus_rest": cols[5]}

        g_low = group_means(ex, low, lo_err, n_lo, a.min_lo, (0, 0.95, 1))      # 第 1 组 = 其余,第 2 组 = 历史最好 5%
        print("2) 低分时赚:排名靠后的票,'过去低分时的平均超额'最好的 5% -> 今天的平均超额 bp(t 值),和被选中的票比")
        print(f"   {'':4} {'低分里其余':>14} {'历史最好 5%':>14} {'全部选中':>14} {'最好5% − 选中':>14}")
        for seg, sl in segs:
            diff = g_low[sl, 1] - all_sel[sl]
            cols = [mean_t(g_low[sl, 0]), mean_t(g_low[sl, 1]), mean_t(all_sel[sl]), mean_t(diff)]
            print(f"   {seg:4} " + " ".join(f"{c[0]:+8.1f}({c[1]:+4.1f})" for c in cols), flush=True)
            rec[f"lo_{seg}"] = {"rest": cols[0], "best5": cols[1], "selected": cols[2], "best5_minus_selected": cols[3]}

        p_hi = yearly_persistence(ex, sel, years, a.min_hi)
        p_lo = yearly_persistence(ex, low, years, a.min_lo)
        print("3) 逐年持续性:相邻两年每只股票平均超额的截面相关(Pearson / Spearman,只数)")
        for lab, pp in (("被选中时", p_hi), ("低分时", p_lo)):
            print(f"   {lab}: " + "  ".join(f"{r['from']}->{r['to']} {r['pearson']:+.2f}/{r['spearman']:+.2f}({r['n']})"
                                       for r in pp))
        rec["yearly_hi"], rec["yearly_lo"] = p_hi, p_lo
        report[name] = rec

        if a.write_prefix:
            # 组合里的排名只能用第 t 天收盘时已知的:有信号、到 t+1 上市满 min_age 天;封板、停牌交给回测去处理
            age_ok = ((np.arange(D)[:, None] + 1 - m.list_day[None, :]) >= a.min_age)
            pct_t = daily_pct(sm, age_ok)
            mods = modified_signals(sm, pct_t, hi_err, n_hi, lo_err, n_lo, a.min_hi, a.min_lo, a.black_frac,
                                    a.white_frac, a.lo_from)
            for k_, arr in (("base", sm), ("black", mods["black"]), ("white", mods["white"])):
                np.save(f"{a.write_prefix}_{name}_hl{a.halflife:g}_{k_}.npy", arr.astype(np.float32))
            np.save(f"{a.write_prefix}_{name}_hl{a.halflife:g}_white_mask.npy", mods["white_mask"])
            print(f"   写出 {a.write_prefix}_{name}_hl{a.halflife:g}_{{base,black,white}}.npy(已平滑,回测用 --halflives 0)")
    if a.out:
        Path(a.out).write_text(json.dumps(report, ensure_ascii=False, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
