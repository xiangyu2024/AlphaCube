"""单只股票上的持续选错:历史统计只用已实现的标签、和暴力算法一致、能认出真实的持续偏差。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stock_errors as se                                    # noqa: E402


def test_trailing_matches_brute_force_and_stops_at_t_minus_2():
    rng = np.random.default_rng(0)
    D, S, W = 60, 7, 15
    ex = rng.normal(size=(D, S))
    ex[rng.random((D, S)) < 0.1] = np.nan
    mask = rng.random((D, S)) < 0.4
    mean, n = se.trailing(ex, mask, W)
    for t in (2, 17, 40, 59):
        lo, hi = max(0, t - 1 - W), t - 1                       # 下标 [t − 1 − W, t − 1):最后一天是 t − 2
        for s in range(S):
            v = [ex[k, s] for k in range(lo, max(lo, hi)) if mask[k, s] and np.isfinite(ex[k, s])]
            assert n[t, s] == len(v)
            if v:
                assert np.isclose(mean[t, s], np.mean(v))
    bumped = ex.copy()
    bumped[39:] = 99.0                                          # 第 40 天收盘时还没实现的
    assert np.array_equal(se.trailing(bumped, mask, W)[0][40], mean[40], equal_nan=True)
    poked = ex.copy()
    poked[38] = 99.0                                            # t − 2:已实现,应该进统计
    assert not np.array_equal(se.trailing(poked, mask, W)[0][40], mean[40], equal_nan=True)


def test_persistent_losers_show_up_in_the_worst_group():
    rng = np.random.default_rng(1)
    D, S = 400, 300
    bias = np.where(np.arange(S) < 60, -0.01, 0.0)              # 前 60 只被选中时总是亏 1%
    score = rng.normal(size=(D, S))
    eligible = np.ones((D, S), bool)
    pct = se.daily_pct(score, eligible)
    sel = pct <= 0.3
    ret = rng.normal(0, 0.01, (D, S)) + np.where(sel, bias, 0.0)
    ex = se.excess(ret, pct)
    hi_err, n_hi = se.trailing(ex, sel, 200)
    g = se.group_means(ex, sel, hi_err, n_hi, 20, (0, 1 / 3, 2 / 3, 1))
    late = slice(250, D)
    assert np.nanmean(g[late, 0]) < np.nanmean(g[late, 2]) - 0.004
    yp = se.yearly_persistence(ex, sel, np.repeat(np.arange(4), 100), 10)
    assert all(r["spearman"] > 0.3 for r in yp)


def test_blacklist_drops_the_worst_and_whitelist_lifts_low_ranked_earners():
    rng = np.random.default_rng(2)
    D, S = 3, 200
    sm = rng.normal(size=(D, S))
    pct = se.daily_pct(sm, np.ones((D, S), bool))
    hi_err = rng.normal(size=(D, S))
    lo_err = rng.normal(size=(D, S))
    n = np.full((D, S), 500.0)
    mods = se.modified_signals(sm, pct, hi_err, n, lo_err, n, 20, 100, 0.2, 0.02, 0.2)
    t = 1
    dropped = np.isnan(mods["black"][t]) & np.isfinite(sm[t])
    assert dropped.sum() == 40 and hi_err[t, dropped].max() <= np.quantile(hi_err[t], 0.2) + 1e-12
    lifted = mods["white"][t] != sm[t]
    assert lifted.sum() == 4 and (pct[t, lifted] > 0.2).all()
    assert np.array_equal(lifted, mods["white_mask"][t])
    assert (mods["white"][t, lifted] > sm[t].max()).all()          # 强制排到最前面
    idx = np.flatnonzero(lifted)
    by_score = idx[np.argsort(-mods["white"][t, idx])]
    by_err = idx[np.argsort(-lo_err[t, idx])]
    assert np.array_equal(by_score, by_err)                        # 彼此按 lo_err 排,不按列顺序
