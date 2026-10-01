"""全样本表现的几个算子:回撤和日期、加权收益、半年调样的成分股近似。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import tearsheet as ts                                       # noqa: E402


def test_perf_drawdown_dates_and_recovery():
    r = np.array([0.1, -0.5, 0.2, 1.0])                     # 净值 1.1, 0.55, 0.66, 1.32
    d = ["20240102", "20240103", "20240201", "20240202"]
    p = ts.perf(r, d)
    assert p["mdd"] == pytest.approx(-0.5)
    assert (p["mdd_peak"], p["mdd_trough"], p["mdd_recover"]) == ("20240102", "20240103", "20240202")
    assert p["total"] == pytest.approx(0.32)
    assert p["win_month"] == pytest.approx(0.5)              # 1 月 -45%,2 月 +32%


def test_perf_relative_identities():
    rng = np.random.default_rng(0)
    rb = rng.normal(0.0005, 0.01, 500)
    r = 1.2 * rb + 0.0004 + rng.normal(0, 0.002, 500)
    p = ts.perf(r, [f"2024{i % 12 + 1:02d}01" for i in range(500)], rb)
    assert p["beta"] == pytest.approx(1.2, abs=0.05)
    assert p["ir"] == pytest.approx((r - rb).mean() / (r - rb).std() * np.sqrt(252))
    same = ts.perf(rb, ["20240101"] * 500, rb)
    assert same["ex_cagr"] == pytest.approx(0.0) and same["ex_mdd"] == pytest.approx(0.0)


def test_weighted_returns_skip_missing():
    ret = np.array([[0.1, 0.2, np.nan], [0.0, np.nan, 0.3]])
    w = np.array([[1.0, 3.0, 5.0], [np.nan, 2.0, 2.0]])
    np.testing.assert_allclose(ts.weighted_returns(ret, w), [(0.1 + 0.6) / 4, 0.3])


def test_index_members_rebalance_only_in_january_and_july():
    dates = ["20231229", "20240102", "20240103", "20240201", "20240701", "20240702"]
    D, S = len(dates), 4
    tmv = np.tile([4.0, 3.0, 2.0, 1.0], (D, 1))
    tmv[2:, 3] = 100.0                                       # 第 3 只 1 月 3 日起变成最大
    amount = np.ones((D, S))
    m = ts.index_members(tmv, amount, np.full(S, -10_000), dates, n=2, liquid=1.0, window=1)
    assert m[0].tolist() == [True, True, False, False]       # 第一天调样
    assert m[2].tolist() == m[1].tolist() == [True, True, False, False]   # 1 月 2 日调过,之后不动
    assert m[4].tolist() == [True, False, False, True]       # 7 月第一个交易日才换进来
