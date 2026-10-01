"""开盘封板判定,以及标签剔除"次日开盘封涨停、买不进"的那部分收益。"""
from __future__ import annotations

import numpy as np
import pytest

from alphacube.label import Label
from alphacube.store import DAY_STATUS_CODES, build_cube_store
from alphacube.tradability import entry_locks

SYMS = ["600000.SH", "600001.SH", "688001.SH"]


def _store(tmp_path):
    """3 只 x 5 天 x 20 分钟。第 2 天:主板 600000 整个开盘窗口都在 +10% 涨停价上(封死);
    科创板 688001 同样 +10%,但它的限幅是 20%,而且窗口里有波动 -> 买得进;
    第 3 天:600001 一字跌停(-10%)。"""
    S, D, M = 3, 5, 20
    price = np.full((S, D, M), 10.0)
    rng = np.random.default_rng(0)
    price += rng.uniform(-0.02, 0.02, (S, D, M))
    price[:, :, -1] = 10.0                                   # 每天收盘 10 元
    price[0, 2, :] = 11.0                                    # 600000 第 2 天全天封死在 11 元
    price[2, 2, 1:16] = 11.0 + rng.uniform(-0.05, 0.05, 15)  # 688001 第 2 天 +10% 附近,有波动
    price[1, 3, :] = 9.0                                     # 600001 第 3 天一字跌停
    price[1, 4, :] = 9.0 + rng.uniform(-0.02, 0.02, M)       # 第 4 天在 9 元附近正常交易
    vol = np.full((S, D, M), 100.0)
    cubes = {"Open": price, "High": price, "Low": price, "Close": price, "Volume": vol,
             "Vwap": price, "Ret": price * 0, "Amt": vol * price}
    status = np.full((S, D), DAY_STATUS_CODES["TRADING"], dtype=np.int8)
    return build_cube_store(tmp_path / "locks", cubes, SYMS, [f"2024010{i + 1}" for i in range(D)],
                            day_status=status)


def test_entry_locks(tmp_path):
    view = _store(tmp_path).view()
    up, down = entry_locks(view)
    assert up.shape == (5, 3)
    assert up[2].tolist() == [True, False, False]            # 科创板 +10% 不是涨停
    assert down[3].tolist() == [False, True, False]
    assert up.sum() == 1 and down.sum() == 1
    assert not up[0].any() and not down[0].any()             # 第一天没有昨收,不判


def test_buyable_label_drops_only_entries_locked_at_limit_up(tmp_path):
    view = _store(tmp_path).view()
    plain = Label("open_vwap15", 1).compute(view)
    clean = Label("open_vwap15", 1, buyable=True).compute(view)
    # 决策日 1 -> 第 2 天开盘买入:600000 封涨停,买不进 -> NaN;其余格子一模一样
    assert np.isfinite(plain[1, 0]) and np.isnan(clean[1, 0])
    keep = np.ones(plain.shape, dtype=bool)
    keep[1, 0] = False
    np.testing.assert_array_equal(plain[keep], clean[keep])
    # 跌停买得进(卖不出是另一回事),不剔
    assert np.isfinite(clean[2, 1])
    assert "limit-up" in Label("open_vwap15", 1, buyable=True).describe()


def test_buyable_is_only_defined_for_the_opening_label(tmp_path):
    view = _store(tmp_path).view()
    with pytest.raises(ValueError):
        Label("close", 1, buyable=True).compute(view)
