"""可交易回测的记账:整百股、单票上限、停牌/涨跌停、费用、除权。

全部用手搭的小行情,每个数都能心算;要锁住的是"真账户"的几条硬规矩,
而不是哪个参数下夏普多少。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import tradable_backtest as tb        # noqa: E402

ZERO = tb.Costs("零费用", rate=0.0)


def market(adj, raw=None, vol=None, up=None, down=None):
    adj = np.asarray(adj, dtype=np.float64)
    D, S = adj.shape
    return tb.Market(
        dates=[f"202401{d + 1:02d}" for d in range(D)], adj=adj,
        raw=adj.copy() if raw is None else np.asarray(raw, dtype=np.float64),
        vol15=np.full((D, S), 1e9) if vol is None else np.asarray(vol, dtype=np.float64),
        up_lock=np.zeros((D, S), bool) if up is None else np.asarray(up, bool),
        down_lock=np.zeros((D, S), bool) if down is None else np.asarray(down, bool))


def shares(m, res, k):
    """第 k 个决策日成交后的不复权股数。"""
    e = int(res["t"][k]) + 1
    return res["units"][k] * m.adj[e] / m.raw[e]


# --------------------------------------------------------------------- 费用

def test_flat_cost_is_proportional_both_sides():
    assert tb.FLAT15.one(10_000, False, "20240102") == pytest.approx(15.0)
    assert tb.FLAT15.one(10_000, True, "20240102") == pytest.approx(15.0)
    assert tb.FLAT15.one(0.0, True, "20240102") == 0.0


def test_retail_cost_minimum_commission_and_stamp_duty_by_date():
    # 5000 元:佣金 1.25 元 < 5 元 -> 收 5 元;比例部分 5.1bp
    buy = tb.RETAIL.one(5000, False, "20240102")
    assert buy == pytest.approx(5000 * 0.00051 + 5.0)
    # 卖出加印花税:2023-08-28 起万 5,之前千 1
    assert tb.RETAIL.one(5000, True, "20240102") - buy == pytest.approx(2.5)
    assert tb.RETAIL.one(5000, True, "20230825") - buy == pytest.approx(5.0)
    # 大单佣金按费率
    assert tb.RETAIL.one(1e6, False, "20240102") == pytest.approx(1e6 * (0.00051 + 0.00025))
    v = tb.RETAIL.many(np.array([0.0, 5000.0]), True, "20240102")
    assert v[0] == 0.0 and v[1] == pytest.approx(tb.RETAIL.one(5000, True, "20240102"))


# --------------------------------------------------------------- 涨跌停判定

def test_board_limits():
    lim = tb.board_limit(["600000.SH", "300750.SZ", "688981.SH", "000001.SZ"], ["20200821", "20200824"])
    assert lim[:, 0].tolist() == [0.10, 0.10]
    assert lim[:, 1].tolist() == [0.10, 0.20]          # 创业板注册制之后 20%
    assert lim[:, 2].tolist() == [0.20, 0.20]
    assert lim[:, 3].tolist() == [0.10, 0.10]


def test_ffill_rows():
    x = np.array([[np.nan, 1.0], [2.0, np.nan], [np.nan, np.nan], [3.0, 4.0]])
    np.testing.assert_array_equal(tb.ffill_rows(x), [[np.nan, 1.0], [2.0, 1.0], [2.0, 1.0], [3.0, 4.0]])


def test_limit_locks():
    # 第 0 天收盘 10 元;第 1 天开盘 15 分钟的高低价
    close = np.array([[10.0] * 5, [np.nan] * 5])
    hi = np.array([[np.nan] * 5, [11.00, 10.50, 9.00, 10.30, 10.98]])
    lo = np.array([[np.nan] * 5, [10.99, 10.50, 9.00, 10.00, 10.20]])
    lim = np.full((2, 5), 0.10)
    up, down = tb.limit_locks(hi, lo, close, lim)
    # 0: 整个窗口都在涨停价 -> 封涨停;1: 一字 +5%(ST)-> 封涨停;
    # 2: 一字 -10% -> 封跌停;3: 正常;4: 摸到涨停但成交过 10.20 -> 买得进
    assert up[1].tolist() == [True, True, False, False, False]
    assert down[1].tolist() == [False, False, True, False, False]
    assert not up[0].any() and not down[0].any()       # 没有昨收就不判


# ---------------------------------------------------------------- 交易单位

def test_lot_rules_by_board():
    qmin, step = tb.lot_rules(["600000.SH", "300750.SZ", "688981.SH"])
    assert qmin.tolist() == [100, 100, 200]
    assert step.tolist() == [100, 100, 1]


def test_qty_floor():
    assert tb.qty_floor(250, 100, 100) == 200
    assert tb.qty_floor(99, 100, 100) == 0
    assert tb.qty_floor(250, 200, 1) == 250            # 科创板:过了 200 股按 1 股递增
    assert tb.qty_floor(199.5, 200, 1) == 0
    np.testing.assert_allclose(tb.qty_floor_vec(np.array([250.0, 99.0, 250.7]), np.array([100.0, 100.0, 200.0]),
                                                np.array([100.0, 100.0, 1.0])), [200, 0, 250])


def test_star_market_needs_200_shares_then_steps_by_one():
    m = market(np.full((3, 3), 10.0))
    m.lot_min, m.lot_step = tb.lot_rules(["688001.SH", "688002.SH", "600000.SH"])
    score = np.tile([3.0, 2.0, 1.0], (3, 1))
    # 上限 2500 元:科创板第 1 只买 250 股;第 2 只同样 250 股;主板只能 200 股
    res = tb.simulate(m, score, 0, 1, capital=10_000, cap=0.25, pool_frac=1.0, costs=ZERO, keep_units=True)
    np.testing.assert_allclose(shares(m, res, 0), [250, 250, 200])
    # 上限 1500 元:科创板不够 200 股 -> 一股不买;主板 100 股
    res = tb.simulate(m, score, 0, 1, capital=10_000, cap=0.15, pool_frac=1.0, costs=ZERO, keep_units=True)
    assert shares(m, res, 0).tolist() == [0, 0, 100]


def test_fill_prices_uses_raw_times_adjustment_factor():
    adj = np.array([[60.0, 10.0], [np.nan, 11.0], [np.nan, np.nan]])
    raw = np.array([[10.0, 10.0], [10.5, 11.0], [11.0, np.nan]])
    fac = np.array([[6.0, 1.0], [6.0, 1.0], [np.nan, np.nan]])
    out = tb.fill_prices(adj, raw, fac)
    # 第 1 天按当天因子补;第 2 天因子缺失,沿用前一天;没成交(raw 缺)的不补
    np.testing.assert_allclose(out, [[60.0, 10.0], [63.0, 11.0], [66.0, np.nan]])


# ------------------------------------------------------------------ 建仓

def test_greedy_fills_by_score_to_cap_in_whole_lots():
    m = market(np.full((4, 6), 10.0))
    score = np.tile(np.array([6.0, 5, 4, 3, 2, 1]), (4, 1))
    res = tb.simulate(m, score, 0, 2, capital=10_000, cap=0.25, pool_frac=1.0, costs=ZERO,
                      keep_units=True)
    sh = shares(m, res, 0)
    # 每只上限 2500 元 = 2.5 手 -> 2 手;前 4 只各 200 股,剩 1980 元给第 5 只买 1 手
    np.testing.assert_allclose(sh, [200, 200, 200, 200, 100, 0])
    assert np.allclose(sh % 100, 0)
    assert np.all(sh * 10.0 <= 0.25 * 10_000 + 1e-9)
    np.testing.assert_allclose(res["ret"], 0.0, atol=1e-12)     # 价格不动、没费用 -> 净值不动


def test_unaffordable_lot_is_skipped():
    # 第 0 只 1 手 = 5 万,超过单票上限 2 万 -> 跳过,钱给后面的
    m = market(np.tile([500.0, 10.0, 10.0], (3, 1)))
    score = np.tile([3.0, 2.0, 1.0], (3, 1))
    res = tb.simulate(m, score, 0, 1, capital=100_000, cap=0.2, pool_frac=1.0, costs=ZERO, keep_units=True)
    np.testing.assert_allclose(shares(m, res, 0), [0, 2000, 2000])


def test_pool_limits_where_spare_cash_goes():
    m = market(np.full((3, 10), 10.0))
    score = np.tile(np.arange(10, 0, -1, dtype=float), (3, 1))
    res = tb.simulate(m, score, 0, 1, capital=10_000, cap=0.5, pool_frac=0.2, costs=ZERO, keep_units=True)
    sh = shares(m, res, 0)
    assert sh[2:].sum() == 0                             # 前 20% = 2 只,之外一股不买
    np.testing.assert_allclose(sh[:2], [500, 400])       # 上限 5000 = 5 手;第 2 只只剩 4980(0.2% 预留)


def test_floor_skips_a_position_too_small_to_be_worth_opening():
    m = market(np.full((4, 6), 10.0))
    score = np.tile(np.array([6.0, 5, 4, 3, 2, 1]), (4, 1))
    res = tb.simulate(m, score, 0, 2, capital=10_000, cap=0.25, floor=0.2, pool_frac=1.0, costs=ZERO,
                      keep_units=True)
    # 没有下限时第 5 只会拿剩下的 1980 元买 1 手;下限 2000 元 -> 不开
    np.testing.assert_allclose(shares(m, res, 0), [200, 200, 200, 200, 0, 0])
    with pytest.raises(ValueError):
        tb.simulate(m, score, 0, 2, cap=0.01, floor=0.02)


def test_conviction_sizes_between_floor_and_cap_by_signal_strength():
    m = market(np.full((3, 5), 10.0))
    score = np.tile([3.0, 1.0, 0.0, -1.0, -3.0], (3, 1))      # 截面 z = [1.5, 0.5, 0, -0.5, -1.5]
    res = tb.simulate(m, score, 0, 1, capital=10_000, cap=0.4, floor=0.1, conviction=(0.0, 1.0),
                      pool_frac=1.0, lot=0, costs=ZERO, reserve=0.0, keep_units=True)
    w = shares(m, res, 0) * 10.0 / 10_000
    np.testing.assert_allclose(w, [0.4, 0.25, 0.1, 0.1, 0.1])


def test_skip_top_leaves_the_most_extreme_names_out():
    m = market(np.full((3, 5), 10.0))
    score = np.tile([5.0, 4, 3, 2, 1], (3, 1))
    res = tb.simulate(m, score, 0, 1, capital=10_000, cap=0.5, skip_top=0.2, pool_frac=1.0, costs=ZERO,
                      reserve=0.0, keep_units=True)
    np.testing.assert_allclose(shares(m, res, 0), [0, 500, 500, 0, 0])


def test_keep_zone_holds_incumbents_that_slipped_but_are_still_ranked_high():
    m = market(np.full((4, 5), 10.0))
    score = np.array([[5.0, 4, 3, 2, 1], [5.0, 3, 4, 1, 2], [5.0, 3, 4, 1, 2], [5.0, 3, 4, 1, 2]])
    kw = dict(capital=10_000, cap=0.5, pool_frac=1.0, costs=ZERO, reserve=0.0, keep_units=True)
    # 第 1 天第 1 只掉到第 3 名(分位 0.5):不保留就换成第 2 只,保留区 0.6 就不动
    res = tb.simulate(m, score, 0, 2, **kw)
    np.testing.assert_allclose(shares(m, res, 1), [500, 0, 500, 0, 0])
    res = tb.simulate(m, score, 0, 2, keep=0.6, **kw)
    np.testing.assert_allclose(shares(m, res, 1), [500, 500, 0, 0, 0])
    assert res["turnover"][1] == 0.0


def test_keep_zone_still_trims_a_holding_that_outgrew_the_cap():
    adj = np.full((4, 2), 10.0)
    adj[2:, 0] = 20.0                                         # 第 0 只翻倍:10000 元 > 上限 7500
    m = market(adj)
    score = np.tile([2.0, 1.0], (4, 1))
    res = tb.simulate(m, score, 0, 2, capital=10_000, cap=0.5, keep=0.9, pool_frac=1.0, costs=ZERO,
                      reserve=0.0, keep_units=True)
    # 目标 375 股,减仓量要整手 -> 卖 100 股,剩 400 股(仍略超上限,但不到 1 手)
    assert shares(m, res, 1)[0] == 400


def test_cash_never_negative_with_costs():
    rng = np.random.default_rng(0)
    D, S = 30, 40
    adj = 10 * np.cumprod(1 + rng.normal(0, 0.03, (D, S)), axis=0)
    m = market(adj, raw=adj / rng.uniform(1, 5, S))
    score = rng.normal(size=(D, S))
    for costs in (tb.FLAT15, tb.RETAIL):
        res = tb.simulate(m, score, 0, D, capital=200_000, cap=0.05, pool_frac=0.5, costs=costs,
                          reserve=0.0)
        assert (res["cash"] >= -1e-9).all()
        assert np.isfinite(res["ret"]).all()


# ---------------------------------------------------------- 停牌、涨跌停

def test_suspended_holding_is_frozen_at_last_price_then_sold():
    adj = np.array([[10.0, 10.0], [10.0, 10.0], [np.nan, 10.0], [np.nan, 10.0], [8.0, 10.0], [8.0, 10.0]])
    m = market(adj)
    score = np.array([[2.0, 1.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    res = tb.simulate(m, score, 0, 5, capital=10_000, cap=0.5, pool_frac=0.5, costs=ZERO, keep_units=True)
    u = res["units"]
    # 第 0 天决策 -> 第 1 天买第 0 只;第 1、2 天想换成第 1 只,但第 0 只第 2、3 天停牌卖不掉
    assert u[0][0] > 0
    assert u[1][0] == u[0][0] and u[2][0] == u[0][0]
    assert res["blocked_sell"][1] == 1 and res["blocked_sell"][2] == 1
    # 停牌期间按最后价格(10)计市值,净值不动;复牌当天按 8 元成交卖掉
    np.testing.assert_allclose(res["ret"][:2], 0.0, atol=1e-12)
    assert res["frozen"][0] == 0 and res["frozen"][1] == pytest.approx(u[0][0] * 10.0 / 10_000)
    assert u[3][0] == 0
    hold0 = u[0][0] * 10.0
    assert res["ret"][2] == pytest.approx(hold0 * (8.0 / 10.0 - 1) / 10_000)


def test_up_locked_name_cannot_be_bought_and_down_locked_cannot_be_sold():
    m = market(np.full((4, 3), 10.0), up=np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]]))
    score = np.tile([3.0, 2.0, 1.0], (4, 1))
    res = tb.simulate(m, score, 0, 1, capital=10_000, cap=0.5, pool_frac=1.0, costs=ZERO, keep_units=True)
    np.testing.assert_allclose(shares(m, res, 0), [0, 500, 400])   # 第 0 只封涨停,顺延
    assert res["blocked_buy"][0] == 1

    down = np.zeros((4, 3), bool)
    down[2, 1] = True
    m = market(np.full((4, 3), 10.0), down=down)
    score = np.array([[1.0, 3.0, 2.0], [3.0, 1.0, 2.0], [3.0, 1.0, 2.0], [3.0, 1.0, 2.0]])
    res = tb.simulate(m, score, 0, 2, capital=10_000, cap=0.5, pool_frac=0.5, costs=ZERO, keep_units=True)
    assert shares(m, res, 0)[1] == 500
    assert shares(m, res, 1)[1] == 500                   # 封跌停:想卖卖不掉


def test_new_listing_is_not_bought_until_old_enough():
    adj = np.full((7, 2), 10.0)
    adj[:3, 1] = np.nan                                  # 第 1 只第 3 天上市
    m = market(adj)
    m.list_day = np.array([-10_000, 3])
    score = np.tile([1.0, 2.0], (7, 1))                  # 信号一直更偏好新股
    res = tb.simulate(m, score, 2, 5, capital=10_000, cap=1.0, pool_frac=1.0, costs=ZERO, min_age=2,
                      reserve=0.0, keep_units=True)
    # 决策日 2、3 在第 3、4 天成交:上市第 0、1 天,不买,钱进老股;决策日 4 -> 第 5 天,满 2 天,换过去
    assert shares(m, res, 0).tolist() == [1000, 0]
    assert shares(m, res, 1).tolist() == [1000, 0]
    assert shares(m, res, 2).tolist() == [0, 1000]
    with pytest.raises(ValueError):
        tb.simulate(market(adj), score, 2, 5, min_age=2)   # 没有上市日就不能用 min_age


def test_participation_caps_trade_size():
    vol = np.full((3, 2), 1e9)
    vol[1, 0] = 3500                                     # 开盘 15 分钟只成交 3500 股 -> 最多 350 股 -> 3 手
    m = market(np.full((3, 2), 10.0), vol=vol)
    res = tb.simulate(m, np.tile([2.0, 1.0], (3, 1)), 0, 1, capital=100_000, cap=0.5, pool_frac=1.0,
                      costs=ZERO, keep_units=True)
    np.testing.assert_allclose(shares(m, res, 0), [300, 4900])


# ------------------------------------------------------------ 记账口径

def test_weights_mode_without_frictions_earns_the_weighted_return():
    rng = np.random.default_rng(1)
    D, S = 8, 4
    adj = 20 * np.cumprod(1 + rng.normal(0, 0.02, (D, S)), axis=0)
    m = market(adj)
    w = np.tile([0.4, 0.3, 0.2, 0.1], (D, 1))
    res = tb.simulate(m, None, 0, D, weights=w, cap=None, lot=0, costs=ZERO, tradability=False, reserve=0.0)
    t = res["t"].astype(int)
    expect = (w[t] * (adj[t + 2] / adj[t + 1] - 1)).sum(1)
    np.testing.assert_allclose(res["ret"], expect, rtol=1e-10, atol=1e-12)


def test_flat_cost_charged_on_traded_notional():
    m = market(np.full((5, 2), 10.0))
    w = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    res = tb.simulate(m, None, 0, 3, weights=w, cap=None, lot=0, costs=tb.FLAT15, tradability=False,
                      reserve=0.0)
    # 第 1 天:全仓买入,付 15bp(费用从同一笔钱里出,所以只买得到 1/1.0015);
    # 第 2 天:全换,卖 + 买各 15bp;第 3 天不动
    assert res["turnover"][0] == pytest.approx(0.5 / 1.0015, rel=1e-9)
    assert res["ret"][0] == pytest.approx(-0.0015 / 1.0015, rel=1e-6)
    assert res["turnover"][1] == pytest.approx(1.0, rel=1e-2)
    assert res["ret"][1] == pytest.approx(-0.003, rel=1e-2)
    assert res["turnover"][2] == pytest.approx(0.0, abs=1e-9)


def test_bonus_shares_keep_value_and_lots():
    # 第 2 天 10 送 10:不复权价减半,后复权价不动 -> 股数翻倍、市值不变
    adj = np.full((5, 1), 10.0)
    raw = np.array([[10.0], [10.0], [5.0], [5.0], [5.0]])
    m = market(adj, raw=raw)
    res = tb.simulate(m, np.ones((5, 1)), 0, 3, capital=10_000, cap=1.0, pool_frac=1.0, costs=ZERO,
                      reserve=0.0, keep_units=True)
    assert shares(m, res, 0)[0] == 1000
    assert shares(m, res, 1)[0] == pytest.approx(2000)
    np.testing.assert_allclose(res["ret"], 0.0, atol=1e-12)


def test_cash_dividend_tail_does_not_trigger_a_lot_trade():
    # 第 2 天除息:后复权价不动、不复权价降 0.3% -> 股数变成 1003 股出头,目标还是 1000 股
    adj = np.full((5, 1), 10.0)
    raw = np.array([[10.0], [10.0], [9.97], [9.97], [9.97]])
    m = market(adj, raw=raw)
    res = tb.simulate(m, np.ones((5, 1)), 0, 3, capital=10_000, cap=1.0, pool_frac=1.0, costs=ZERO,
                      reserve=0.0, keep_units=True)
    assert shares(m, res, 0)[0] == 1000
    assert shares(m, res, 1)[0] == pytest.approx(1000 * 10.0 / 9.97)    # 没有为尾巴卖掉一手
    np.testing.assert_allclose(res["turnover"][1:], 0.0, atol=1e-12)


def test_constant_capital_mode_resets_nav_each_day():
    adj = np.array([[10.0], [10.0], [11.0], [12.1], [12.1]])
    m = market(adj)
    res = tb.simulate(m, np.ones((5, 1)), 0, 3, capital=10_000, cap=1.0, pool_frac=1.0, costs=ZERO,
                      lot=0, reserve=0.0, compound=False)
    np.testing.assert_allclose(res["nav"], 10_000)
    np.testing.assert_allclose(res["ret"], [0.1, 0.1, 0.0])


def test_stats_matches_grid_convention():
    r = np.array([0.01, -0.02, 0.03, 0.0, 0.01])
    s = tb.stats(r)
    assert s["sharpe"] == pytest.approx(r.mean() / r.std() * np.sqrt(252))
    assert s["mdd"] == pytest.approx(-0.02)
    nav = np.cumprod(1 + r)
    assert s["mdd_c"] == pytest.approx((nav / np.maximum.accumulate(nav) - 1).min())
