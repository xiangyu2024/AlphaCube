#!/usr/bin/env python
"""头部多头因子是不是几种朴素的风格排序?(交接文档 H4 / H5)

三个因果的日度风格面板,值越大越是预期做多的那一头(第 t 行只用 <= t 的数据,窗口含当天、至少 10 个有效日):
  illiq     = -log(过去 20 天"开盘 15 分钟成交额"的均值),成交额 = vol15 x raw(不复权的 15 分钟 VWAP)
  lowvol    = -(过去 20 天后复权日收益的标准差,ddof=1)
  lowprice  = -log(不复权收盘价)
注意:行情 npz 里的 close 是**后复权价**(和 adj 同一个价格空间,close/adj 的中位数是 1),不是不复权价,
所以不复权收盘价 = close / fac,复权收盘价就是 close 本身(不能再乘 fac)。运行时会量这两个比值,不对就退出。

做三件事:
  1. 三个风格排序自己按 long_report.evaluate 的同一把尺子打分(训练段挑方向、测试段固定;原样和 EMA5 平滑后)。
     另外按字面 -log(close)(后复权价)打一个"lowprice_adj"当对照,不进回归。
  2. 每组前 20 名因子"实际使用的信号"(sign x 面板,做半衰期 5 的因果 EMA)和三个风格的逐日截面 Spearman,
     训练段 / 测试段分别取日均。
  3. 残差检验:逐日把因子的截面秩(z 分数)对三个风格秩(z 分数,带截距)做 OLS,残差当新信号,再按同一把尺子打分:
       resid       = 对 sign x 面板(未平滑)回归 -> evaluate 的"原样 / EMA5"两列和原因子的两列一一对应
       resid_used  = 对"实际使用的平滑后信号"回归 -> 看 evaluate 的"原样"列(平滑后信号去掉风格以后能不能交易)
     同时量残差(EMA5 后)和风格的残余相关,应该接近 0;以及三个风格秩对因子秩的逐日 R^2。

输入:行情 npz(_bench/tradable/...),三个库的 report 和面板;输出 _bench/long_reward/exposure/ 下面的文件。

  cd /root/autodl-tmp/AlphaCube && PYTHONPATH=src python -u scripts/style_exposure.py --workers 3
"""
from __future__ import annotations

import os

if __name__ == "__main__":      # 作为脚本跑时,在 numpy 之前把 BLAS 线程压到 1(机器是共享的,3 个 worker 各开线程会超配)
    for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_k, "1")

import argparse
import csv
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import long_report as lr  # noqa: E402  evaluate() 就是"同一把尺子"

STYLES = ("illiq", "lowvol", "lowprice")
STYLE_NOTE = {"illiq": "-log(20日开盘15分钟成交额均值)", "lowvol": "-20日日收益标准差", "lowprice": "-log(不复权收盘价)",
              "lowprice_adj": "-log(close),close 是后复权价(对照)"}
EXTRA_STYLE = "lowprice_adj"       # 只打分,不进暴露 / 回归
# (显示名, report 文件, lib 名, 面板目录, 排名用的训练段适应度)
GROUPS = (("多头奖励", "_bench/long_reward/report.json", "多头奖励", "_bench/allfac_long", "fit_train"),
          ("老奖励对照", "_bench/long_reward/report.json", "老奖励对照", "_bench/allfac_base", "fit_train"),
          ("第二版奖励", "_bench/long_reward/report2_partial.json", "第二版奖励", "_bench/allfac_hit", "hit_fit_train"))
_G: dict = {}


# ---------------------------------------------------------------------------
# 风格面板(纯 numpy,因果)
# ---------------------------------------------------------------------------

def _positive(a: np.ndarray) -> np.ndarray:
    """float64 拷贝;NaN / inf / 非正 -> NaN。"""
    a = np.asarray(a, dtype=np.float64)
    return np.where(np.isfinite(a) & (a > 0), a, np.nan)


def rolling_mean_std(x: np.ndarray, window: int = 20, min_valid: int = 10, ddof: int = 1,
                     want_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """逐日滚动(含当天、共 window 行)的均值和标准差,只数有限值;窗口里有效值少于 min_valid 个就是 NaN。
    因果:第 t 行只用到 <= t 的行(开头不足 window 行时用已有的行)。标准差用两遍法(先减均值),不用平方和相减。"""
    x = np.asarray(x, dtype=np.float64)
    D, S = x.shape
    fin_all = np.isfinite(x)
    x0 = np.where(fin_all, x, 0.0)
    mean = np.full((D, S), np.nan)
    std = np.full((D, S), np.nan) if want_std else None
    need = max(int(min_valid), 1)
    for t in range(D):
        lo = max(0, t - window + 1)
        fin = fin_all[lo:t + 1]
        n = fin.sum(0)
        ok = n >= need
        m = x0[lo:t + 1].sum(0) / np.maximum(n, 1)
        mean[t] = np.where(ok, m, np.nan)
        if want_std:
            dev = np.where(fin, x0[lo:t + 1] - m, 0.0)
            std[t] = np.where(ok & (n > ddof), np.sqrt((dev * dev).sum(0) / np.maximum(n - ddof, 1)), np.nan)
    return mean, std


def daily_returns(adj_close: np.ndarray) -> np.ndarray:
    """后复权收盘价 -> 日收益 c_t / c_t-1 - 1;前一天或当天缺(停牌)就是 NaN,不跨停牌算多日收益。"""
    c = _positive(adj_close)
    r = np.full(c.shape, np.nan)
    r[1:] = c[1:] / c[:-1] - 1.0
    return r


def style_panels(adj_close: np.ndarray, raw_close: np.ndarray, amount: np.ndarray,
                 window: int = 20, min_valid: int = 10) -> Dict[str, np.ndarray]:
    """三个风格面板(float32,值越大越是预期做多的那一头)。输入都是 (D, S);NaN / 非正 -> NaN。
    illiq    -log(window 日成交额均值);lowvol   -(window 日日收益标准差);lowprice -log(不复权收盘价)。"""
    mean_amt, _ = rolling_mean_std(_positive(amount), window, min_valid, want_std=False)
    _, sd = rolling_mean_std(daily_returns(adj_close), window, min_valid)
    out = {"illiq": -np.log(mean_amt), "lowvol": -sd, "lowprice": -np.log(_positive(raw_close))}
    return {k: v.astype(np.float32) for k, v in out.items()}


def style_inputs(close: np.ndarray, fac: np.ndarray, raw: np.ndarray, vol15: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """npz 字段 -> (后复权收盘价, 不复权收盘价, 开盘 15 分钟成交额)。close 已经是后复权价,不复权价 = close / fac;
    成交额 = 开盘 15 分钟成交股数 x 不复权 VWAP(= 那 15 分钟的 sum Amt)。"""
    adj_close = np.asarray(close, dtype=np.float64)
    raw_close = adj_close / _positive(fac)
    amount = np.asarray(vol15, dtype=np.float64) * np.asarray(raw, dtype=np.float64)
    return adj_close, raw_close, amount


def check_price_space(close, adj, raw, fac, sample_days: int = 200) -> Dict[str, float]:
    """close 必须是后复权价(和 adj 同一价格空间),close / fac 必须是不复权价(和 raw 同一价格空间);否则风格面板整个错位。"""
    idx = np.linspace(0, close.shape[0] - 1, num=min(sample_days, close.shape[0])).astype(int)
    c, a, r, f = (np.asarray(v[idx], dtype=np.float64) for v in (close, adj, raw, fac))
    with np.errstate(all="ignore"):
        ok = np.isfinite(c) & np.isfinite(a) & np.isfinite(r) & np.isfinite(f) & (c > 0) & (a > 0) & (r > 0) & (f > 0)
        r_adj, r_raw = float(np.median((c / a)[ok])), float(np.median((c / f / r)[ok]))
        r_wrong = float(np.median((c / r)[ok]))
    if not (abs(r_adj - 1) < 0.05 and abs(r_raw - 1) < 0.05):
        raise SystemExit(f"价格空间不对:median(close/adj)={r_adj:.3f}, median(close/fac/raw)={r_raw:.3f}(都应该≈1)。"
                         f"close 不是后复权价,先弄清 npz 再跑")
    return {"close_over_adj": r_adj, "close_over_fac_over_raw": r_raw, "close_over_raw": r_wrong}


# ---------------------------------------------------------------------------
# 截面秩、逐日 Spearman、逐日回归取残差(纯 numpy)
# ---------------------------------------------------------------------------

def rank_avg(v: np.ndarray) -> np.ndarray:
    """平均秩(从 1 开始,并列取平均),等价于 scipy.stats.rankdata。"""
    v = np.asarray(v, dtype=np.float64)
    n = v.size
    if n == 0:
        return v.copy()
    order = np.argsort(v)                                 # 并列的平均秩与排序是否稳定无关
    sv = v[order]
    start = np.concatenate(([True], sv[1:] != sv[:-1]))
    first = np.flatnonzero(start)
    last = np.concatenate((first[1:], [n])) - 1
    out = np.empty(n)
    out[order] = ((first + last) / 2.0 + 1.0)[np.cumsum(start) - 1]
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x - x.mean(), y - y.mean()
    den = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / den) if den > 0 else float("nan")


def daily_rank_corr(a: np.ndarray, b: np.ndarray, min_common: int = 30) -> np.ndarray:
    """逐日截面 Spearman(两边都有限的票上取秩再算 Pearson);共同的票少于 min_common 只或一边没有离散度的日子是 NaN。(D,)"""
    D = a.shape[0]
    out = np.full(D, np.nan)
    for t in range(D):
        m = np.isfinite(a[t]) & np.isfinite(b[t])
        if m.sum() >= min_common:
            out[t] = _pearson(rank_avg(a[t][m]), rank_avg(b[t][m]))
    return out


def period_means(day_values: np.ndarray, split: int) -> Tuple[float, float]:
    """逐日数组 -> (训练段 [0, split) 均值, 测试段 [split, D) 均值),只数有限值,没有就是 NaN。"""
    out = []
    for part in (day_values[:split], day_values[split:]):
        part = part[np.isfinite(part)]
        out.append(float(part.mean()) if part.size else float("nan"))
    return out[0], out[1]


def _zrank(v: np.ndarray) -> Optional[np.ndarray]:
    r = rank_avg(v)
    sd = r.std()
    return None if not sd > 0 else (r - r.mean()) / sd


def neutralize(factor: np.ndarray, styles: Sequence[np.ndarray], min_common: int = 30
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """逐日:把因子的截面秩(z 分数)对各风格的截面秩(z 分数,带截距)做 OLS(np.linalg.lstsq),取残差。
    只在因子和所有风格都有限的票上做、秩也在这批票里取;不足 min_common 只或有零离散度的日子整行 NaN。
    返回 (残差面板 float64, 每天的 R^2, 每天用到的票数),后两个对没做的日子是 NaN / 0。"""
    D, S = factor.shape
    res = np.full((D, S), np.nan)
    r2 = np.full(D, np.nan)
    used = np.zeros(D, dtype=np.int64)
    for t in range(D):
        m = np.isfinite(factor[t])
        for s in styles:
            m &= np.isfinite(s[t])
        n = int(m.sum())
        if n < min_common:
            continue
        y = _zrank(factor[t][m])
        cols = [_zrank(s[t][m]) for s in styles]
        if y is None or any(c is None for c in cols):
            continue
        X = np.column_stack([np.ones(n)] + cols)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        e = y - X @ beta
        res[t, m] = e
        r2[t] = 1.0 - float((e * e).sum()) / float((y * y).sum())
        used[t] = n
    return res, r2, used


# ---------------------------------------------------------------------------
# 选因子、汇总
# ---------------------------------------------------------------------------

def select_top(rows: Sequence[dict], lib: str, key: str = "fit_train", n: int = 20) -> List[dict]:
    """某个库里没被门槛挡掉的因子,按训练段分数 key 从高到低取前 n 个(并列按 i 小的在前)。"""
    ok = [r for r in rows if r["lib"] == lib and not r["gated_out"] and _finite(r.get(key))]
    ok.sort(key=lambda r: (-r[key], r["i"]))
    return ok[:n]


def _finite(x) -> bool:
    return x is not None and bool(np.isfinite(x))


def _med(vals: Sequence[float]) -> float:
    a = np.asarray(list(vals), dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


def _npos(vals: Sequence[float]) -> int:
    a = np.asarray(list(vals), dtype=float)
    return int((a[np.isfinite(a)] > 0).sum())


def summarize_group(rows: Sequence[dict]) -> dict:
    """一组因子(每个是 factor_row 的输出)-> 风格相关的中位数、原样 vs 残差的测试段夏普中位数、为正的个数。"""
    out = {"n": len(rows), "corr": {}, "r2": {}, "resid_exposure": {}}
    for s in STYLES:
        out["corr"][s] = {p: _med(r["corr"][s][p] for r in rows) for p in ("train", "test")}
        out["resid_exposure"][s] = {p: _med(r["resid_exposure"][s][p] for r in rows) for p in ("train", "test")}
    out["r2"] = {p: _med(r["r2"][p] for r in rows) for p in ("train", "test")}
    cols = {"orig_raw": ("orig", "test_sharpe"), "orig_smooth": ("orig", "test_sharpe_smooth"),
            "resid_raw": ("resid", "test_sharpe"), "resid_smooth": ("resid", "test_sharpe_smooth"),
            "residused_raw": ("resid_used", "test_sharpe")}
    for name, (part, key) in cols.items():
        vals = [r[part].get(key, float("nan")) for r in rows]
        out[f"med_{name}"], out[f"pos_{name}"] = _med(vals), _npos(vals)
        out[f"valid_{name}"] = int(np.isfinite(np.asarray(vals, dtype=float)).sum())
    return out


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------

def _init(target_path: str, split: int, cost: float, min_coverage: float) -> None:
    import torch
    from alphacube.mining.scoring import LongScorer

    torch.set_num_threads(1)
    tgt = np.asarray(np.load(target_path, mmap_mode="r"), dtype=np.float64)
    _G.update(split=split, train=LongScorer(tgt[:split], cost_per_side=cost, min_coverage=min_coverage),
              test=LongScorer(tgt[split:], cost_per_side=cost, min_coverage=0.0))


def _evaluate(panel: np.ndarray) -> dict:
    return lr.evaluate(panel, _G["train"], _G["test"], _G["split"])


def compact(ev: dict) -> dict:
    """evaluate 的输出 -> 要留的几个数。被门槛挡掉的只留标记。"""
    out = {"gated_out": bool(ev["gated_out"]), "fit_train": float(ev["fit_train"]),
           "distinct": float(ev.get("distinct", float("nan")))}
    if out["gated_out"]:
        return out
    out["sign"] = float(ev["sign"])
    for tag, part in (("", "train"), ("", "test"), ("_smooth", "train_smooth"), ("_smooth", "test_smooth")):
        pre = "train" if part.startswith("train") else "test"
        for src, dst in (("long_sharpe", "sharpe"), ("long_ann", "ann"), ("long_turnover", "turnover"),
                         ("long_rank_ic", "rank_ic")):
            out[f"{pre}_{dst}{tag}"] = float(ev[part].get(src, float("nan")))
    out["train_tie"], out["train_code"] = float(ev["train"]["long_tie"]), float(ev["train"]["long_code"])
    return out


def _exposures(signal: np.ndarray, styles: Dict[str, np.ndarray], split: int, min_common: int) -> Dict[str, dict]:
    out = {}
    for s in STYLES:
        tr, te = period_means(daily_rank_corr(signal, styles[s], min_common), split)
        out[s] = {"train": tr, "test": te}
    return out


def _style_job(name: str) -> dict:
    try:
        panel = _G["styles"][name]
        ev = _evaluate(np.asarray(panel, dtype=np.float64))
        cov = float(np.isfinite(panel).mean())
        return {"kind": "style", "name": name, "note": STYLE_NOTE[name], "finite_share": cov,
                "names_per_day": float(np.isfinite(panel).sum(1).mean()), **compact(ev)}
    except Exception:
        return {"kind": "style", "name": name, "error": traceback.format_exc()}


def _factor_job(job: dict) -> dict:
    try:
        return _factor_row(job)
    except Exception:
        return {"kind": "factor", "group": job["group"], "i": job["i"], "error": traceback.format_exc()}


def _factor_row(job: dict) -> dict:
    from alphacube.metrics import ema_rows

    split, mc, hl = _G["split"], _G["min_common"], lr.SMOOTH
    styles = _G["styles"]
    sl = [styles[s] for s in STYLES]
    p = np.asarray(np.load(Path(job["dir"]) / "panels" / f"{job['i']:05d}.npy"), dtype=np.float64)
    p[~np.isfinite(p)] = np.nan
    orig = _evaluate(p)                                   # 用现在的打分器重算一遍原因子,和 report 里存的对账
    sign = float(job["sign"])
    check = {"sign_equal": bool(orig.get("sign") == sign), "fit_diff": float(abs(orig["fit_train"] - job["fit_train"])),
             "test_sharpe_diff": float(abs(orig["test"]["long_sharpe"] - job["rep_test_sharpe"])) if not orig["gated_out"] else None,
             "test_sharpe_smooth_diff": float(abs(orig["test_smooth"]["long_sharpe"] - job["rep_test_sharpe_smooth"]))
             if not orig["gated_out"] else None}
    signed = sign * p
    del p
    used = ema_rows(signed, hl)                           # 实际使用的信号:sign x 面板,再做半衰期 5 的因果 EMA
    corr = _exposures(used, styles, split, mc)
    res_a, r2, n_used = neutralize(signed, sl, mc)        # 对未平滑的信号回归
    r2_tr, r2_te = period_means(r2, split)
    ev_a = _evaluate(res_a)
    expo_a = _exposures(ema_rows(res_a, hl), styles, split, mc)       # 残差(EMA5 后)和风格的残余相关
    del res_a, signed
    res_b, _, _ = neutralize(used, sl, mc)                # 对实际使用的平滑后信号回归
    ev_b = _evaluate(res_b)
    del res_b, used
    return {"kind": "factor", "group": job["group"], "lib": job["lib"], "i": job["i"], "run": job["run"],
            "expression": job["expression"], "sign": sign, "fit_train": job["fit_train"],
            "hit_fit_train": job.get("hit_fit_train"), "corr": corr, "r2": {"train": r2_tr, "test": r2_te},
            "names_used_mean": float(n_used[n_used > 0].mean()) if (n_used > 0).any() else float("nan"),
            "orig": compact(orig), "orig_check": check, "resid": compact(ev_a), "resid_exposure": expo_a,
            "resid_used": compact(ev_b)}


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def _f(x, fmt="+.2f", scale=1.0) -> str:
    return f"{'nan':>6s}" if x is None or not np.isfinite(x) else format(x * scale, fmt)


def print_style_table(styles: Dict[str, dict]) -> None:
    print("\n=== (a) 三个风格排序自己的成绩(同一把尺子:头部 10% 等权、减全市场等权、单边 15bp;训练段挑方向,测试段方向固定) ===")
    print("                      |        原样: 训练夏普 | 测试夏普 测试年化 测试换手 |     EMA5: 训练夏普 | 测试夏普 测试年化 测试换手 | 覆盖")
    for name, s in styles.items():
        if s.get("gated_out"):
            print(f"  {name:12s} 被门槛挡掉(fit={s['fit_train']}, distinct={s['distinct']:.3f})")
            continue
        print(f"  {name:12s} 方向{s['sign']:+.0f} | 训练 {_f(s['train_sharpe'])} | 测试 {_f(s['test_sharpe'])} "
              f"{_f(s['test_ann'], '+.1f', 100)}% {_f(s['test_turnover'], '.3f')} | 训练 {_f(s['train_sharpe_smooth'])} | "
              f"测试 {_f(s['test_sharpe_smooth'])} {_f(s['test_ann_smooth'], '+.1f', 100)}% "
              f"{_f(s['test_turnover_smooth'], '.3f')} | {s['finite_share']:.2f}   {s['note']}")


def print_group_tables(groups: Dict[str, dict]) -> None:
    print("\n=== (b) 前 20 名因子对三个风格的暴露:实际使用的信号(EMA5)和风格的逐日截面 Spearman,训练段 / 测试段日均,组内中位数 ===")
    print("  组             n |      illiq 训练/测试 |     lowvol 训练/测试 |   lowprice 训练/测试 | 三风格秩的 R^2 训练/测试")
    for g, s in groups.items():
        c = s["corr"]
        print(f"  {g:10s} {s['n']:3d} | " + " | ".join(f"{_f(c[k]['train'])} / {_f(c[k]['test'])}" for k in STYLES)
              + f" | {_f(s['r2']['train'], '.2f')} / {_f(s['r2']['test'], '.2f')}")
    print("\n=== (c) 去掉三个风格以后(逐日因子秩对三风格秩回归取残差,再按同一把尺子打分;测试段夏普,组内中位数,括号是为正的个数 / 有数的个数) ===")
    print("  组             n |  原样: 原因子  残差        | EMA5: 原因子  残差            | 残差(对平滑后信号回归,原样列)")
    for g, s in groups.items():
        def cell(k):
            return f"{_f(s['med_' + k])} ({s['pos_' + k]}/{s['valid_' + k]})"
        print(f"  {g:10s} {s['n']:3d} |  {cell('orig_raw'):15s} {cell('resid_raw'):15s} | {cell('orig_smooth'):15s} "
              f"{cell('resid_smooth'):15s} | {cell('residused_raw')}")
    print("\n  残差(EMA5 后)和三个风格的残余相关(应≈0,训练/测试中位数):")
    for g, s in groups.items():
        e = s["resid_exposure"]
        print(f"  {g:10s} " + " | ".join(f"{k} {_f(e[k]['train'])} / {_f(e[k]['test'])}" for k in STYLES))


def print_top_resid(rows: Sequence[dict], k: int = 3) -> List[dict]:
    ok = [r for r in rows if np.isfinite(r["resid"].get("test_sharpe_smooth", float("nan")))]
    ok.sort(key=lambda r: -r["resid"]["test_sharpe_smooth"])
    print(f"\n=== (d) 残差 EMA5 测试夏普最高的 {k} 个 ===")
    for r in ok[:k]:
        o, e = r["orig"], r["resid"]
        print(f"  [{r['group']} i={r['i']} {r['run']}] {r['expression']}\n"
              f"      原因子 测试夏普 原样 {_f(o['test_sharpe'])} / EMA5 {_f(o['test_sharpe_smooth'])};残差 测试夏普 原样 "
              f"{_f(e['test_sharpe'])} / EMA5 {_f(e['test_sharpe_smooth'])}(年化 {_f(e['test_ann_smooth'], '+.1f', 100)}%,"
              f"换手 {_f(e['test_turnover_smooth'], '.3f')});风格相关 测试 "
              + " ".join(f"{s} {_f(r['corr'][s]['test'])}" for s in STYLES))
    return ok[:k]


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o)} 不能写进 json")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def check_alignment(market: dict, target_path: str, dirs: Sequence[str]) -> dict:
    """行情 npz 和因子面板 / 标签是不是同一套 日期 x 股票:日期逐个相等;标签 = adj[t+2]/adj[t+1]-1(抽样天);
    因子面板有值的格子上收盘价都有值。npz 里没有股票列表之外的独立来源,列顺序靠后两条验证。"""
    dates = [str(d) for d in market["dates"]]
    for d in dirs:
        dd = [str(x) for x in np.load(Path(d) / "dates.npy")]
        if dd != dates:
            raise SystemExit(f"{d}/dates.npy 和行情 npz 的日期不一致({len(dd)} vs {len(dates)})")
    tgt = np.load(target_path, mmap_mode="r")
    D, S = tgt.shape
    if (D, S) != market["adj"].shape:
        raise SystemExit(f"标签形状 {tgt.shape} 和行情 {market['adj'].shape} 不一致")
    adj = market["adj"]
    days = np.linspace(0, D - 3, num=60).astype(int)
    diff, n_cmp = 0.0, 0
    for t in days:
        lab = adj[t + 2] / adj[t + 1] - 1.0
        m = np.isfinite(lab) & np.isfinite(tgt[t])
        n_cmp += int(m.sum())
        if m.any():
            diff = max(diff, float(np.abs(lab[m] - np.asarray(tgt[t])[m]).max()))
    if n_cmp < 10000 or diff > 1e-9:
        raise SystemExit(f"标签和行情的 adj 对不上:比了 {n_cmp} 格,最大差 {diff:g}")
    fin_close = np.isfinite(market["close"])
    bad = {}
    for d in dirs:
        f0 = sorted((Path(d) / "panels").glob("*.npy"))[0]
        pf = np.isfinite(np.load(f0, mmap_mode="r"))
        bad[d] = float((pf & ~fin_close).sum() / max(int(pf.sum()), 1))
        if bad[d] > 1e-3:
            raise SystemExit(f"{f0}:有值的格子里 {bad[d]:.3%} 在收盘价缺失的格子上,列顺序可能不一致")
    return {"dates_equal": True, "label_cells_compared": n_cmp, "label_max_abs_diff": diff, "panel_cells_on_nan_close": bad}


def load_jobs(top: int, panels_cfg: Sequence[tuple]) -> Tuple[List[dict], Dict[str, int]]:
    jobs, counts = [], {}
    for group, report, lib, d, key in panels_cfg:
        rep = json.loads(Path(report).read_text(encoding="utf-8"))
        man = {int(r["i"]): r["expression"] for r in csv.DictReader(open(Path(d) / "manifest.csv", encoding="utf-8"))}
        if rep["split"] != _G["split"]:
            raise SystemExit(f"{report} 的训练段长度 {rep['split']} 和这里的 {_G['split']} 不一致")
        for r in select_top(rep["rows"], lib, key, top):
            if man.get(r["i"]) != r["expression"]:
                raise SystemExit(f"{group} i={r['i']}:manifest 里的表达式和 report 对不上")
            jobs.append({"group": group, "lib": lib, "dir": d, "i": int(r["i"]), "run": r.get("run", ""),
                         "expression": r["expression"], "sign": r["sign"], "fit_train": r["fit_train"],
                         "hit_fit_train": r.get("hit_fit_train"), "rep_test_sharpe": r["test"]["long_sharpe"],
                         "rep_test_sharpe_smooth": r["test_smooth"]["long_sharpe"]})
        counts[group] = sum(j["group"] == group for j in jobs)
    return jobs, counts


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="_bench/tradable/market_20180102_20251231.npz")
    ap.add_argument("--target", default="_bench/trainto/allfac_buyable/_target.npy")
    ap.add_argument("--out-dir", default="_bench/long_reward/exposure")
    ap.add_argument("--top", type=int, default=20, help="每组取训练段分数最高的几个;0 = 只给风格排序打分")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--min-valid", type=int, default=10)
    ap.add_argument("--min-common", type=int, default=30)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--cost", type=float, default=0.0015)
    ap.add_argument("--min-coverage", type=float, default=0.8)
    a = ap.parse_args(argv)

    t0 = time.time()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(a.market, allow_pickle=False)
    market = {k: z[k] for k in ("dates", "symbols", "adj", "raw", "vol15", "close", "fac")}
    D, S = market["close"].shape
    split = max(1, min(D - 1, int(D * a.train_frac)))
    _G.update(split=split, min_common=a.min_common)
    print(f"行情 {D} 天 x {S} 只;训练段 {split} 天,测试段 {D - split} 天", flush=True)
    space = check_price_space(market["close"], market["adj"], market["raw"], market["fac"])
    print(f"价格空间:median(close/adj)={space['close_over_adj']:.4f},median(close/fac/raw)="
          f"{space['close_over_fac_over_raw']:.4f}(都≈1:close 是后复权价);median(close/raw)={space['close_over_raw']:.2f}"
          f"(≫1,所以不能直接拿 close 当不复权价)", flush=True)
    align = check_alignment(market, a.target, sorted({g[3] for g in GROUPS}))
    print(f"对齐检查通过:日期逐个相等;标签 = adj[t+2]/adj[t+1]-1(比了 {align['label_cells_compared']} 格,最大差 "
          f"{align['label_max_abs_diff']:g});因子面板有值的格子落在收盘价缺失处的比例 {align['panel_cells_on_nan_close']}", flush=True)

    adj_close, raw_close, amount = style_inputs(market["close"], market["fac"], market["raw"], market["vol15"])
    styles = style_panels(adj_close, raw_close, amount, a.window, a.min_valid)
    styles[EXTRA_STYLE] = (-np.log(_positive(adj_close))).astype(np.float32)
    del adj_close, raw_close, amount
    for k, v in styles.items():
        np.save(out_dir / f"style_{k}.npy", v)
        print(f"  风格面板 {k:13s} 有值 {np.isfinite(v).mean():.3f},每天 {np.isfinite(v).sum(1).mean():.0f} 只", flush=True)
    mat = {}
    for i, x in enumerate(STYLES):                          # 三个风格之间的逐日秩相关(解释共线性用)
        for y in STYLES[i + 1:]:
            mat[f"{x}|{y}"] = dict(zip(("train", "test"), period_means(daily_rank_corr(styles[x], styles[y], a.min_common), split)))
    print("  风格两两逐日 Spearman(训练/测试):" + ";".join(f"{k} {v['train']:+.2f}/{v['test']:+.2f}" for k, v in mat.items()),
          flush=True)
    _G["styles"] = styles
    del market
    jobs, counts = load_jobs(a.top, GROUPS) if a.top > 0 else ([], {})
    print(f"因子:{counts};风格 {len(styles)} 个;共 {len(jobs) + len(styles)} 个任务,{a.workers} 个 worker "
          f"({time.time() - t0:.0f}s)", flush=True)

    style_res: Dict[str, dict] = {}
    rows: List[dict] = []
    errors: List[str] = []
    jl = open(out_dir / "factors.jsonl", "w", encoding="utf-8")
    with ProcessPoolExecutor(a.workers, mp_context=get_context("fork"), initializer=_init,
                             initargs=(a.target, split, a.cost, a.min_coverage)) as ex:
        futs = [ex.submit(_style_job, n) for n in styles] + [ex.submit(_factor_job, j) for j in jobs]
        for n, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if "error" in r:
                errors.append(f"{r.get('name') or (r.get('group'), r.get('i'))}\n{r['error']}")
                print(f"  !! 出错 {r.get('name') or (r.get('group'), r.get('i'))}\n{r['error']}", flush=True)
            elif r["kind"] == "style":
                style_res[r["name"]] = r
                if len(style_res) == len(styles):                  # 风格任务排在最前面,先把这张表打出来
                    print_style_table({k: style_res[k] for k in styles})
            else:
                rows.append(r)
                jl.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")
                jl.flush()
            if n % 6 == 0 or n == len(futs):
                print(f"  {n}/{len(futs)} ({time.time() - t0:.0f}s)", flush=True)
    jl.close()

    order = {g[0]: k for k, g in enumerate(GROUPS)}
    rows.sort(key=lambda r: (order[r["group"]], -float(r["hit_fit_train"] if r["group"] == "第二版奖励" else r["fit_train"])))
    groups = {g[0]: summarize_group([r for r in rows if r["group"] == g[0]]) for g in GROUPS if counts.get(g[0])}
    if groups:
        print_group_tables(groups)
    top3 = print_top_resid(rows) if rows else []
    chk = [r["orig_check"] for r in rows]
    if chk:
        bad = [r for r in rows if not (r["orig_check"]["sign_equal"] and r["orig_check"]["fit_diff"] < 1e-6)]
        print(f"\n对账:重算的原因子和 report 里存的比,方向一致且训练适应度差 < 1e-6 的 {len(rows) - len(bad)}/{len(rows)} 个;"
              f"测试夏普最大差 {max((c['test_sharpe_diff'] or 0) for c in chk):.2g}(原样)/ "
              f"{max((c['test_sharpe_smooth_diff'] or 0) for c in chk):.2g}(EMA5)", flush=True)
    out = {"meta": {"split": split, "days": D, "symbols": S, "window": a.window, "min_valid": a.min_valid,
                    "min_common": a.min_common, "smooth_halflife": lr.SMOOTH, "cost": a.cost, "top": a.top,
                    "min_coverage": a.min_coverage, "price_space": space, "alignment": align,
                    "note": "npz 的 close 是后复权价;不复权收盘价 = close / fac;lowprice_adj 是按字面 -log(close) 的对照,不进回归",
                    "styles_rank_corr": mat},
           "styles": {k: style_res[k] for k in styles if k in style_res},
           "factors": rows, "groups": groups, "top_resid": [{"group": r["group"], "i": r["i"], "expression": r["expression"]}
                                                           for r in top3], "errors": errors}
    (out_dir / "exposure.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
    print(f"\n写出 {out_dir / 'exposure.json'}({time.time() - t0:.0f}s);出错 {len(errors)} 个", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
