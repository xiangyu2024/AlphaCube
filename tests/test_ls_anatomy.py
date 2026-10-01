"""多空书拆成两条腿:不加限制时必须和原来的多空书逐日相等;限制只剔"交易不到"的票。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ls_anatomy as la                                     # noqa: E402

from alphacube.metrics import long_short_weights, portfolio_returns


def _panel(seed=0, D=60, S=50):
    rng = np.random.default_rng(seed)
    alpha = rng.normal(size=(D, S))
    alpha[rng.random((D, S)) < 0.05] = np.nan
    label = 0.01 * alpha + rng.normal(0, 0.02, (D, S))
    label[rng.random((D, S)) < 0.05] = np.nan
    return alpha, label


def test_legs_without_restrictions_are_the_long_short_book():
    alpha, label = _panel()
    every = np.ones_like(alpha, dtype=bool)
    _, short_ex, ls, short_pool, _ = la.legs(alpha, label, every, every, cost=0.0015)
    # _panel 里 alpha 缺失的格子标签也缺失(标签由 alpha 生成),所以空头池就是全市场,两列相等
    np.testing.assert_allclose(short_pool, short_ex, atol=1e-12)
    net, _, _ = portfolio_returns(long_short_weights(alpha, 0.1), label, 0.0015)
    np.testing.assert_allclose(ls, net, atol=1e-12)


def test_short_leg_against_its_own_pool_strips_the_pool_effect():
    alpha, label = _panel(2)
    every = np.ones_like(alpha, dtype=bool)
    half = np.zeros_like(alpha, dtype=bool)
    half[:, :25] = True                                     # 空头只在前 25 只里做
    _, short_ex, _, short_pool, _ = la.legs(alpha, label, every, half)
    fin = np.isfinite(label)
    r_mkt = np.where(fin, label, 0).sum(1) / fin.sum(1)
    pool = fin[:, :25]
    r_pool = np.where(pool, label[:, :25], 0).sum(1) / pool.sum(1)
    np.testing.assert_allclose(short_ex - short_pool, r_mkt - r_pool, atol=1e-12)


def test_leg_weights_pick_within_the_eligible_names():
    alpha = np.tile(np.arange(10, dtype=float), (2, 1))      # 第 9 只最高
    ok = np.ones_like(alpha, dtype=bool)
    ok[:, 9] = False                                          # 最高那只交易不到 -> 顺延到第 8 只
    w = la.leg_weights(alpha, ok, 0.25, True)               # 9 只能交易 -> floor(9 x 0.25) = 2 只
    np.testing.assert_allclose(w[0], [0] * 7 + [0.5, 0.5, 0])
    w = la.leg_weights(alpha, ok, 0.25, False)
    np.testing.assert_allclose(w[0], [0.5, 0.5] + [0] * 8)


def test_anatomy_aligns_locks_to_the_trading_day():
    alpha, label = _panel(1, D=40, S=40)
    up = np.zeros_like(alpha, dtype=bool)
    down = np.zeros_like(alpha, dtype=bool)
    liq = np.zeros_like(alpha)
    base = la.anatomy(alpha, label, up, down, liq, slice(0, 40), liquid_cuts=())
    assert base["原标签"] == base["开盘能成交"]                 # 没有封板,两种口径一样
    # 决策日 t 的前 10% 在成交日 t+1 全部封涨停 -> 多头腿换一批票,空头腿不变
    for t in range(39):
        top = np.argsort(np.where(np.isfinite(alpha[t]), alpha[t], -np.inf))[-4:]
        up[t + 1, top] = True
    out = la.anatomy(alpha, label, up, down, liq, slice(0, 40), liquid_cuts=())
    assert out["开盘能成交"]["long_ann"] != out["原标签"]["long_ann"]
    assert out["开盘能成交"]["short_ann"] == out["原标签"]["short_ann"]


def test_trailing_mean():
    x = np.array([[1.0], [2.0], [np.nan], [4.0]])
    np.testing.assert_allclose(la.trailing_mean(x, 2), [[1.0], [1.5], [2.0], [4.0]])
