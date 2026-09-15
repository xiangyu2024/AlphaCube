"""Tests for the portfolio-optimization layer (target weights -> executable
whole-share orders under market-specific constraints).

Covers: the index-alignment helper, lot rounding, turnover-budget
enforcement (both the "fits as-is" and "must scale + greedy leftover fill"
paths), untradable-name freezing, dust cleanup accounting, per-name weight
cap trimming, and the market-specific defaults (A-share lot size /
long-only-by-default vs. US 1-share lots / short-able-by-default).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphacube.portfolio import (
    AshareLotOptimizer,
    RebalancePlan,
    TurnoverBudgetOptimizer,
    USEquityOptimizer,
    align_inputs,
)

EQUITY = 1_000_000.0


# --------------------------------------------------------------------------
# alignment helper
# --------------------------------------------------------------------------


def test_align_inputs_union_and_fill_conventions():
    target_weights = pd.Series({"A": 0.1, "B": 0.2, "C": 0.05})
    prev_shares = pd.Series({"B": 100.0, "C": 50.0, "D": 10.0})
    prices = pd.Series({"A": 10.0, "C": 20.0, "D": 30.0, "E": 40.0})

    tw, ps, px, tradable = align_inputs(target_weights, prev_shares, prices)

    expected_index = pd.Index(["A", "B", "C", "D", "E"])
    assert set(tw.index) == set(expected_index)

    # target_weights: missing -> 0.0
    assert tw["A"] == 0.1
    assert tw["D"] == 0.0
    assert tw["E"] == 0.0

    # prev_shares: missing -> 0.0
    assert ps["A"] == 0.0
    assert ps["B"] == 100.0
    assert ps["D"] == 10.0

    # prices: missing -> NaN (never silently filled)
    assert np.isnan(px["B"])
    assert px["A"] == 10.0
    assert px["E"] == 40.0

    # tradable defaults to True everywhere when not supplied
    assert bool(tradable.all())


def test_align_inputs_respects_explicit_tradable_series():
    target_weights = pd.Series({"A": 0.1, "B": 0.1})
    prev_shares = pd.Series({"A": 10.0, "B": 20.0})
    prices = pd.Series({"A": 10.0, "B": 20.0, "C": 30.0})
    tradable = pd.Series({"A": False})

    tw, ps, px, aligned_tradable = align_inputs(target_weights, prev_shares, prices, tradable)

    assert aligned_tradable["A"] == False  # noqa: E712 - explicit False
    # symbols absent from the tradable series default to True
    assert aligned_tradable["B"] == True  # noqa: E712
    assert aligned_tradable["C"] == True  # noqa: E712


def test_turnover_budget_optimizer_is_abstract():
    with pytest.raises(TypeError):
        TurnoverBudgetOptimizer(
            lot_size=1,
            allow_short=True,
            turnover_limit=0.25,
            max_single_name_weight=0.1,
        )  # type: ignore[abstract]


# --------------------------------------------------------------------------
# small hand-computed example: affordable full rebalance, no scaling
# --------------------------------------------------------------------------


def test_full_rebalance_affordable_no_scaling():
    opt = AshareLotOptimizer(turnover_limit=0.25)
    symbols = ["A", "B", "C", "D", "E"]
    prices = pd.Series(100.0, index=symbols)
    prev_shares = pd.Series(0.0, index=symbols)
    # 0.02 weight each * 1,000,000 / 100 price = 200 raw shares = exactly 2 lots
    target_weights = pd.Series(0.02, index=symbols)

    plan = opt.rebalance(target_weights, prev_shares, prices, EQUITY)

    assert isinstance(plan, RebalancePlan)
    assert (plan.new_shares == 200.0).all()
    assert (plan.trades == 200.0).all()
    # no fractional lots
    assert ((plan.new_shares % opt.lot_size) == 0).all()
    assert plan.diagnostics["scaled"] is False
    assert plan.turnover_used == pytest.approx(0.10)
    assert plan.turnover_used <= plan.turnover_budget + 1e-9


# --------------------------------------------------------------------------
# scaling + predictable leftover-lot greedy fill
# --------------------------------------------------------------------------


def test_turnover_scaling_with_predictable_leftover_fill():
    # turnover_limit tight enough that the full desired rebalance (200,000)
    # doesn't fit the 150,000 budget -> forces proportional scale + leftover.
    opt = AshareLotOptimizer(turnover_limit=0.15, dust_cleanup_enabled=False)
    symbols = ["X", "Y", "Z"]
    prices = pd.Series(100.0, index=symbols)
    prev_shares = pd.Series(0.0, index=symbols)
    target_weights = pd.Series({"X": 0.10, "Y": 0.08, "Z": 0.02})

    plan = opt.rebalance(target_weights, prev_shares, prices, EQUITY)

    # Hand-computed: full desired trades are X=1000, Y=800, Z=200 shares
    # (10, 8, 2 lots) = 200,000 total vs. a 150,000 budget -> scale=0.75.
    # Proportional*truncated-to-lots: X=700 (7 lots), Y=600 (6 lots),
    # Z=100 (1 lot) = 140,000 used, leaving exactly one lot's worth
    # (10,000) of leftover budget. Priority = abs(target_weight) so X
    # (0.10) outranks Y (0.08) outranks Z (0.02) and claims the leftover
    # lot first.
    assert plan.diagnostics["scaled"] is True
    assert plan.new_shares["X"] == 800.0  # 700 + 1 leftover lot
    assert plan.new_shares["Y"] == 600.0  # unchanged after truncation
    assert plan.new_shares["Z"] == 100.0  # unchanged after truncation

    for sym in symbols:
        assert plan.new_shares[sym] % opt.lot_size == 0

    assert plan.turnover_used == pytest.approx(0.15)
    assert plan.turnover_used <= plan.turnover_budget + 1e-9


# --------------------------------------------------------------------------
# untradable-name freezing
# --------------------------------------------------------------------------


def test_untradable_name_is_frozen_regardless_of_target():
    opt = AshareLotOptimizer(turnover_limit=0.25, dust_cleanup_enabled=False)
    symbols = ["A", "B"]
    prices = pd.Series(100.0, index=symbols)
    prev_shares = pd.Series({"A": 100.0, "B": 0.0})
    target_weights = pd.Series({"A": 0.20, "B": 0.05})
    tradable = pd.Series({"A": False, "B": True})

    plan = opt.rebalance(target_weights, prev_shares, prices, EQUITY, tradable=tradable)

    assert plan.new_shares["A"] == 100.0  # unchanged despite a big target
    assert plan.trades["A"] == 0.0
    assert "A" in plan.diagnostics["names_untradable"]
    # B trades normally
    assert plan.new_shares["B"] == 500.0  # 0.05 * 1e6 / 100 = 500, already lot-aligned


def test_missing_or_nonpositive_price_freezes_symbol():
    opt = AshareLotOptimizer(turnover_limit=0.25, dust_cleanup_enabled=False)
    prices = pd.Series({"A": np.nan, "B": 0.0, "C": 100.0})
    prev_shares = pd.Series({"A": 100.0, "B": 200.0, "C": 0.0})
    # C kept below the default 0.08 max_single_name_weight cap so this test
    # isolates the price-validity freeze, not the weight cap (covered
    # separately by test_max_single_name_weight_cap_trims_post_fill).
    target_weights = pd.Series({"A": 0.1, "B": 0.1, "C": 0.05})

    plan = opt.rebalance(target_weights, prev_shares, prices, EQUITY)

    assert plan.new_shares["A"] == 100.0
    assert plan.new_shares["B"] == 200.0
    assert set(plan.diagnostics["names_untradable"]) == {"A", "B"}
    assert plan.new_shares["C"] == 500.0  # traded normally: 0.05*1e6/100=500


# --------------------------------------------------------------------------
# dust cleanup
# --------------------------------------------------------------------------


def test_dust_cleanup_zeroes_tiny_position_outside_main_budget():
    opt = AshareLotOptimizer(
        turnover_limit=0.25,
        dust_cleanup_enabled=True,
        dust_cleanup_weight_threshold=0.0005,  # 500 on 1,000,000 equity
        dust_cleanup_turnover_limit=0.05,
    )
    prices = pd.Series({"DUST": 100.0, "MAIN": 100.0})
    # DUST: tiny existing position (300 <= 500 threshold), tiny/zero target
    prev_shares = pd.Series({"DUST": 3.0, "MAIN": 0.0})
    target_weights = pd.Series({"DUST": 0.0, "MAIN": 0.05})

    plan = opt.rebalance(target_weights, prev_shares, prices, EQUITY)

    assert plan.new_shares["DUST"] == 0.0
    assert "DUST" in plan.diagnostics["dust_cleaned_names"]
    assert plan.trades["DUST"] == -3.0
    assert plan.trade_value["DUST"] == pytest.approx(-300.0)

    # MAIN's trade: 0.05 * 1,000,000 / 100 = 500 shares (5 lots) = 50,000
    assert plan.new_shares["MAIN"] == 500.0
    # main-step turnover_used reflects ONLY MAIN's trade, not the dust sale
    assert plan.turnover_used == pytest.approx(50_000.0 / EQUITY)
    # the dust sale is tracked separately
    assert plan.diagnostics["dust_turnover_used"] == pytest.approx(300.0 / EQUITY)


def test_dust_cleanup_closes_either_sign_when_shorting_allowed():
    opt = USEquityOptimizer(
        turnover_limit=0.5,
        allow_short=True,
        dust_cleanup_enabled=True,
        dust_cleanup_weight_threshold=0.0005,
        dust_cleanup_turnover_limit=0.05,
    )
    prices = pd.Series({"SHORTDUST": 50.0})
    prev_shares = pd.Series({"SHORTDUST": -4.0})  # tiny short, value = -200
    target_weights = pd.Series({"SHORTDUST": 0.0})

    plan = opt.rebalance(target_weights, prev_shares, prices, EQUITY)

    assert plan.new_shares["SHORTDUST"] == 0.0
    assert "SHORTDUST" in plan.diagnostics["dust_cleaned_names"]
    assert plan.trades["SHORTDUST"] == 4.0  # buy back to cover the tiny short


# --------------------------------------------------------------------------
# max single-name weight cap
# --------------------------------------------------------------------------


def test_max_single_name_weight_cap_trims_post_fill():
    opt = AshareLotOptimizer(
        turnover_limit=0.5,  # generous so the cap -- not turnover -- binds
        max_single_name_weight=0.08,
        dust_cleanup_enabled=False,
    )
    prices = pd.Series({"BIG": 100.0})
    prev_shares = pd.Series({"BIG": 0.0})
    target_weights = pd.Series({"BIG": 0.10})  # above the 0.08 cap

    plan = opt.rebalance(target_weights, prev_shares, prices, EQUITY)

    # Uncapped fill would be 1000 shares (10 lots, weight 0.10); capped
    # down in whole lots to fit 0.08 -> 800 shares (8 lots).
    assert plan.new_shares["BIG"] == 800.0
    assert plan.achieved_weights["BIG"] == pytest.approx(0.08)
    assert "BIG" in plan.diagnostics["capped_names"]


# --------------------------------------------------------------------------
# market-specific behavior
# --------------------------------------------------------------------------


def test_ashare_optimizer_never_produces_fractional_lots():
    opt = AshareLotOptimizer()
    symbols = ["A", "B", "C", "D"]
    prices = pd.Series({"A": 37.5, "B": 12.3, "C": 88.8, "D": 5.05})
    prev_shares = pd.Series({"A": 100.0, "B": 0.0, "C": 200.0, "D": 0.0})
    target_weights = pd.Series({"A": 0.05, "B": 0.03, "C": -0.02, "D": 0.15})

    plan = opt.rebalance(target_weights, prev_shares, prices, EQUITY)

    assert opt.lot_size == 100
    assert opt.allow_short is False
    for sym in symbols:
        assert plan.new_shares[sym] % 100 == 0
    # long-only: the negative target on C should never produce a short
    assert plan.new_shares["C"] >= 0.0


def test_us_optimizer_allows_negative_shares_when_shorting():
    opt = USEquityOptimizer()  # allow_short=True by default
    prices = pd.Series({"SHORT_ME": 50.0})
    prev_shares = pd.Series({"SHORT_ME": 0.0})
    target_weights = pd.Series({"SHORT_ME": -0.05})

    plan = opt.rebalance(target_weights, prev_shares, prices, EQUITY)

    assert opt.lot_size == 1
    assert opt.allow_short is True
    assert plan.new_shares["SHORT_ME"] < 0.0
    # -0.05 * 1,000,000 / 50 = -1000 shares, lot_size=1 so exact
    assert plan.new_shares["SHORT_ME"] == pytest.approx(-1000.0)


def test_us_optimizer_long_only_override_clips_negative_targets():
    opt = USEquityOptimizer(allow_short=False)
    prices = pd.Series({"NO_SHORT": 50.0})
    prev_shares = pd.Series({"NO_SHORT": 0.0})
    target_weights = pd.Series({"NO_SHORT": -0.05})

    plan = opt.rebalance(target_weights, prev_shares, prices, EQUITY)

    assert plan.new_shares["NO_SHORT"] == 0.0
