"""开盘能不能成交:涨跌停封板的判定。

标签 ``open_vwap15`` 假设决策日 t 的仓位在 t+1 开盘 15 分钟按 VWAP 建立。可要是 t+1 开盘
就封死在涨停上,这笔多头根本买不进 —— 标签照样给它记上涨停后的收益,合成权重就会学去追
这些买不进的票(训练段信号前 22 只里 12% 次日开盘封涨停,见 docs/可交易回测.md §3)。

这里只用立方体本身能看到的东西判定:昨天的收盘价、今天开盘窗口里的最高价和最低价、
按板块和日期定的涨跌幅限制。立方体里没有 ST 标记,ST 的 5% 限幅靠"一字"规则兜住。
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def board_limit(symbols: Sequence[str], dates: Sequence[str]) -> np.ndarray:
    """(D, S) 涨跌停幅度。主板 10%;科创板 20%;创业板 2020-08-24 注册制起 20%;
    北交所 30%。ST(5%)立方体里没有标记,靠 `limit_locks` 的"一字"规则兜住。"""
    D, S = len(dates), len(symbols)
    lim = np.full((D, S), 0.10)
    star = np.array([s.startswith(("688", "689")) for s in symbols], dtype=bool)
    gem = np.array([s.startswith(("300", "301", "302")) for s in symbols], dtype=bool)
    bse = np.array([s.startswith(("4", "8", "92")) for s in symbols], dtype=bool)
    after = np.array([d >= "20200824" for d in dates], dtype=bool)
    lim[:, star] = 0.20
    lim[np.ix_(after, gem)] = 0.20
    lim[:, bse] = 0.30
    return lim


def ffill_rows(x: np.ndarray) -> np.ndarray:
    """沿天前向填充(每列独立)。"""
    D = x.shape[0]
    idx = np.where(np.isfinite(x), np.arange(D)[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = x[idx, np.arange(x.shape[1])[None, :]]
    seen = np.maximum.accumulate(np.isfinite(x), axis=0)
    return np.where(seen, out, np.nan)


def limit_locks(hi: np.ndarray, lo: np.ndarray, close: np.ndarray, lim: np.ndarray,
                tol: float = 0.006, one_price_gap: float = 0.045) -> Tuple[np.ndarray, np.ndarray]:
    """(D, S) 的 (封涨停, 封跌停):开盘窗口里是不是封死在涨/跌停上。

    参考价是上一个有收盘价的交易日的收盘(后复权价的比值跨除权日连续,不用另外除权)。
    封涨停 = 窗口里的**最低价**都在涨停价附近(tol 吸收 0.01 元的取整);
    封跌停 = 窗口里的**最高价**都在跌停价附近。另加一条"一字"规则:整个窗口只有一个价、
    且离昨收超过 4.5% —— 兜住立方体里认不出来的 ST(5% 限幅)。没有昨收(上市首日)不判。"""
    prev = np.full_like(close, np.nan, dtype=np.float64)
    prev[1:] = ffill_rows(close.astype(np.float64))[:-1]
    with np.errstate(all="ignore"):
        up_gap = lo / prev - 1.0
        dn_gap = hi / prev - 1.0
        one = np.isfinite(hi) & np.isfinite(lo) & (hi <= lo * (1 + 1e-6))
        up = (up_gap >= lim - tol) | (one & (dn_gap >= one_price_gap))
        down = (dn_gap <= -lim + tol) | (one & (up_gap <= -one_price_gap))
    ok = np.isfinite(prev) & np.isfinite(hi) & np.isfinite(lo)
    return up & ok, down & ok


def window_range(view, window: Tuple[int, int], chunk: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """(D, S) 的 (窗口最高价, 窗口最低价),价格空间和 High/Low 字段一致(后复权)。"""
    import warnings

    lo_m, hi_m = window
    hi_m = min(hi_m, view.n_minutes)
    hi = np.full((view.n_days, view.n_symbols), np.nan)
    lo = np.full((view.n_days, view.n_symbols), np.nan)
    for a in range(0, view.n_symbols, chunk):
        b = min(a + chunk, view.n_symbols)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)      # 停牌日整窗都是 NaN
            hi[:, a:b] = np.nanmax(view.read_s("High", a, b)[..., lo_m:hi_m], axis=-1).T
            lo[:, a:b] = np.nanmin(view.read_s("Low", a, b)[..., lo_m:hi_m], axis=-1).T
    bad = ~view.tradable().T
    hi[bad] = np.nan
    lo[bad] = np.nan
    return hi, lo


def entry_locks(view, window: Tuple[int, int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """(D, S) 的 (开盘封涨停, 开盘封跌停),窗口默认就是标签的开盘 15 分钟。"""
    from .label import OPEN_VWAP_WINDOW, daily_close

    hi, lo = window_range(view, window or OPEN_VWAP_WINDOW)
    return limit_locks(hi, lo, daily_close(view), board_limit(view.symbols, view.dates))


__all__ = ["board_limit", "ffill_rows", "limit_locks", "window_range", "entry_locks"]
