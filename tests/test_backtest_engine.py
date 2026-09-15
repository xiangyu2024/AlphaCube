"""Tests for the sequential multi-day backtest engine
(`alphacube.backtest`): costs.TransactionCostModel, impact.SimpleImpactModel,
metrics.{performance_stats,cost_drag_stats}, and engine.BacktestEngine.

Covers: a hand-computable zero-friction growth-rate sanity check, that
commission/impact costs actually reduce equity (and match hand-computed
sums), that the per-name weight cap actually binds, that the turnover
budget is never exceeded, the off-by-one/no-lookahead day-boundary
contract, and integration with both market-specific optimizers
(AshareLotOptimizer's whole-lot constraint, USEquityOptimizer's shorting).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphacube.backtest import (
    BacktestEngine,
    SimpleImpactModel,
    TransactionCostModel,
)
from alphacube.portfolio import AshareLotOptimizer, USEquityOptimizer

EQUITY = 1_000_000.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _dates(n_days: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=n_days)


def _deterministic_growth_price_panel(n_days: int, rates: list, p0: list) -> np.ndarray:
    """price[t, i] = p0[i] * (1 + rates[i]) ** t -- exact per-step compounding,
    so day t -> t+1 realized return for instrument i is exactly rates[i]
    on every day (a real, hand-computable ground truth)."""
    n_instruments = len(rates)
    t = np.arange(n_days).reshape(-1, 1)
    rates_arr = np.asarray(rates).reshape(1, -1)
    p0_arr = np.asarray(p0).reshape(1, -1)
    return p0_arr * (1.0 + rates_arr) ** t


class _RecordingOptimizer:
    """Thin spy wrapper around a TurnoverBudgetOptimizer: forwards
    .rebalance() unchanged but records every RebalancePlan, so tests can
    inspect per-day share counts the engine itself doesn't expose."""

    def __init__(self, inner):
        self.inner = inner
        self.plans = []

    @property
    def allow_short(self):
        return self.inner.allow_short

    def rebalance(self, *args, **kwargs):
        plan = self.inner.rebalance(*args, **kwargs)
        self.plans.append(plan)
        return plan


# ---------------------------------------------------------------------------
# 1. zero-cost, zero-impact, constant-signal sanity check
# ---------------------------------------------------------------------------


def test_zero_friction_constant_signal_matches_hand_computed_growth():
    n_days = 15
    instruments = ["A", "B"]
    rate_a = 0.01  # A grows 1%/day, deterministically
    rate_b = 0.0  # B is flat and never favored by the signal
    price_panel = _deterministic_growth_price_panel(n_days, [rate_a, rate_b], [50.0, 80.0])

    # Constant signal: only A is active (positive), B is exactly 0 (inactive
    # under long_only / _active_mask).
    signal_panel = np.tile(np.array([1.0, 0.0]), (n_days, 1))

    optimizer = AshareLotOptimizer(
        lot_size=1,  # avoid lot-rounding noise in the hand-computed check
        allow_short=False,
        turnover_limit=1.0,  # full rebalance affordable every day
        max_single_name_weight=0.95,
        dust_cleanup_enabled=False,
    )
    engine = BacktestEngine(
        optimizer=optimizer,
        cost_model=TransactionCostModel(cost_bps=0.0, min_commission=0.0),
        signal_variant="long_only",
        target_exposure=0.9,
        max_single_name_weight=0.95,
        impact_model=None,
        initial_equity=EQUITY,
    )

    result = engine.run(signal_panel, price_panel, _dates(n_days), instruments)

    n_trading_days = n_days - 1
    assert len(result.daily) == n_trading_days

    # Weight in A is ~0.9 of equity every day (turnover budget is 100%, so
    # the target is always fully achievable) and A grows exactly rate_a
    # every day by construction -> equity compounds at (1 + 0.9*rate_a).
    expected_final_equity = EQUITY * (1.0 + 0.9 * rate_a) ** n_trading_days

    assert result.summary["final_equity"] == pytest.approx(expected_final_equity, rel=2e-3)
    assert result.equity_curve.iloc[-1] == pytest.approx(expected_final_equity, rel=2e-3)

    # No costs at all in this configuration.
    assert (result.daily["transaction_cost"] == 0.0).all()
    assert (result.daily["impact_cost"] == 0.0).all()
    assert result.summary["total_cost"] == 0.0


# ---------------------------------------------------------------------------
# 2. costs actually reduce equity, and match a hand-computed sum
# ---------------------------------------------------------------------------


def _alternating_signal_panel(n_days: int, n_instruments: int) -> np.ndarray:
    """A different single instrument is favored each day (round-robin),
    forcing the optimizer to sell out of yesterday's name and buy into a
    new one every day -> near-100%-of-equity turnover every day."""
    panel = np.zeros((n_days, n_instruments))
    for t in range(n_days):
        panel[t, t % n_instruments] = 1.0
    return panel


def test_costs_reduce_equity_and_match_hand_computed_commission():
    n_days = 10
    n_instruments = 4
    rng = np.random.default_rng(0)
    # Mild, mixed per-instrument drift so growth isn't degenerate.
    rates = rng.uniform(-0.005, 0.01, size=n_instruments)
    price_panel = _deterministic_growth_price_panel(n_days, list(rates), [40.0, 60.0, 25.0, 100.0])
    signal_panel = _alternating_signal_panel(n_days, n_instruments)
    instruments = [f"I{i}" for i in range(n_instruments)]

    def make_engine(cost_bps: float) -> BacktestEngine:
        optimizer = AshareLotOptimizer(
            lot_size=1,
            allow_short=False,
            turnover_limit=1.0,
            max_single_name_weight=0.95,
            dust_cleanup_enabled=False,  # keep commission hand-computation exact
        )
        return BacktestEngine(
            optimizer=optimizer,
            cost_model=TransactionCostModel(cost_bps=cost_bps, min_commission=0.0),
            signal_variant="long_only",
            target_exposure=0.9,
            max_single_name_weight=0.95,
            impact_model=None,
            initial_equity=EQUITY,
        )

    result_free = make_engine(0.0).run(signal_panel, price_panel, _dates(n_days), instruments)
    result_cost = make_engine(20.0).run(signal_panel, price_panel, _dates(n_days), instruments)

    assert result_cost.summary["final_equity"] < result_free.summary["final_equity"]
    assert result_cost.summary["total_transaction_cost"] > 0.0

    # With dust_cleanup disabled, min_commission=0, and every symbol that
    # trades having a valid price, TransactionCostModel.compute()'s notional
    # for a day equals exactly that day's main-fill turnover dollars
    # (turnover_used * equity_before) -- so total commission should equal
    # sum(turnover_used * equity_before) * cost_bps / 1e4.
    hand_computed_total = float(
        (result_cost.daily["turnover_used"] * result_cost.daily["equity_before"]).sum() * (20.0 / 1e4)
    )
    assert result_cost.summary["total_transaction_cost"] == pytest.approx(hand_computed_total, rel=1e-6)
    assert result_cost.daily["transaction_cost"].sum() == pytest.approx(hand_computed_total, rel=1e-6)


# ---------------------------------------------------------------------------
# 3. impact cost integration: scales with trade size, and reduces equity
# ---------------------------------------------------------------------------


def _market_params(instruments, vol=0.02, eta=2.5e-6, gamma=2.5e-7) -> pd.DataFrame:
    return pd.DataFrame(
        {"volatility": vol, "temporary_impact": eta, "permanent_impact": gamma},
        index=instruments,
    )


def test_impact_cost_scales_with_trade_size_and_reduces_equity():
    n_days = 6
    instruments = ["A", "B"]
    price_panel = _deterministic_growth_price_panel(n_days, [0.0, 0.0], [50.0, 50.0])
    signal_panel = np.tile(np.array([1.0, 0.0]), (n_days, 1))
    market_params = _market_params(instruments, vol=0.02, eta=2.5e-6, gamma=2.5e-7)

    def make_engine(initial_equity: float, impact_model) -> BacktestEngine:
        optimizer = AshareLotOptimizer(
            lot_size=1,
            allow_short=False,
            turnover_limit=1.0,
            max_single_name_weight=0.95,
            dust_cleanup_enabled=False,
        )
        return BacktestEngine(
            optimizer=optimizer,
            cost_model=TransactionCostModel(cost_bps=0.0, min_commission=0.0),
            signal_variant="long_only",
            target_exposure=0.9,
            max_single_name_weight=0.95,
            impact_model=impact_model,
            initial_equity=initial_equity,
        )

    impact_model = SimpleImpactModel(market_params=market_params, horizon=1.0, n_intervals=1)

    small_equity = 100_000.0
    large_equity = 5_000_000.0  # 50x the trade size

    result_small = make_engine(small_equity, impact_model).run(
        signal_panel, price_panel, _dates(n_days), instruments
    )
    result_large = make_engine(large_equity, impact_model).run(
        signal_panel, price_panel, _dates(n_days), instruments
    )

    impact_small_day0 = result_small.daily["impact_cost"].iloc[0]
    impact_large_day0 = result_large.daily["impact_cost"].iloc[0]

    assert impact_small_day0 > 0.0
    assert impact_large_day0 > impact_small_day0

    # Almgren-Chriss expected_cost is quadratic-ish in trade size (a
    # 0.5*gamma*X^2 term plus an eta*sum(n_j^2)/tau term) -- so a 50x
    # larger trade should produce a MORE than 50x larger impact cost.
    equity_ratio = large_equity / small_equity
    impact_ratio = impact_large_day0 / impact_small_day0
    assert impact_ratio > equity_ratio

    # Same scenario/trades, impact_model=None vs configured: equity should
    # be strictly lower with impact cost included (trades are identical
    # since the impact model doesn't feed back into the optimizer).
    result_no_impact = make_engine(large_equity, None).run(
        signal_panel, price_panel, _dates(n_days), instruments
    )
    assert result_large.summary["final_equity"] < result_no_impact.summary["final_equity"]
    assert (result_no_impact.daily["impact_cost"] == 0.0).all()


def test_impact_model_skips_symbols_missing_from_market_params():
    instruments = ["A", "B"]
    # market_params only covers "A" -- "B" trades should be skipped (0
    # contribution), not raise.
    market_params = _market_params(["A"])
    impact_model = SimpleImpactModel(market_params=market_params)

    trades = pd.Series({"A": 100.0, "B": -50.0})
    prices = pd.Series({"A": 50.0, "B": 30.0})
    cost = impact_model.estimate(trades, prices)
    assert np.isfinite(cost)
    assert cost >= 0.0


# ---------------------------------------------------------------------------
# 4. position limits actually bind
# ---------------------------------------------------------------------------


def test_position_limit_binds_across_backtest():
    n_days = 10
    n_instruments = 5
    instruments = [f"I{i}" for i in range(n_instruments)]
    rng = np.random.default_rng(1)
    rates = rng.uniform(-0.01, 0.02, size=n_instruments)
    price_panel = _deterministic_growth_price_panel(n_days, list(rates), [30.0, 40.0, 50.0, 60.0, 70.0])

    # Signal heavily concentrated in instrument 0 every day, small noise
    # elsewhere -- would want to hold way more than the cap in I0 absent
    # the limit.
    signal_panel = np.tile(np.array([10.0, 0.2, 0.2, 0.2, 0.2]), (n_days, 1))

    cap = 0.05
    optimizer = AshareLotOptimizer(
        lot_size=1,
        allow_short=False,
        turnover_limit=1.0,
        max_single_name_weight=cap,
        dust_cleanup_enabled=False,
    )
    engine = BacktestEngine(
        optimizer=optimizer,
        cost_model=TransactionCostModel(cost_bps=0.0, min_commission=0.0),
        signal_variant="long_only",
        target_exposure=0.9,
        max_single_name_weight=cap,
        impact_model=None,
        initial_equity=EQUITY,
    )

    result = engine.run(signal_panel, price_panel, _dates(n_days), instruments)

    tol = 1e-3
    assert (result.daily["max_abs_weight"] <= cap + tol).all()


# ---------------------------------------------------------------------------
# 5. off-by-one / no-lookahead boundary
# ---------------------------------------------------------------------------


def test_no_lookahead_boundary_short_panel():
    n_days = 5
    instruments = ["A", "B"]
    price_panel = _deterministic_growth_price_panel(n_days, [0.01, -0.005], [50.0, 60.0])
    signal_panel = np.tile(np.array([1.0, 0.0]), (n_days, 1))

    optimizer = AshareLotOptimizer(lot_size=1, turnover_limit=1.0, dust_cleanup_enabled=False)
    engine = BacktestEngine(
        optimizer=optimizer,
        cost_model=TransactionCostModel(cost_bps=0.0, min_commission=0.0),
        initial_equity=EQUITY,
    )

    # Full range: default start_idx=0, end_idx=None -- must not crash, and
    # must never try to realize P&L off a price beyond the panel.
    result = engine.run(signal_panel, price_panel, _dates(n_days), instruments)
    assert len(result.daily) == n_days - 1
    # The last usable decision day is n_days - 2 (needs price at n_days-1
    # to realize its P&L); the very last day (n_days-1) is never a decision
    # day.
    expected_dates = _dates(n_days)[: n_days - 1]
    assert list(result.daily.index) == list(expected_dates)
    assert len(result.equity_curve) == n_days - 1

    # Explicit end_idx == n_days behaves identically (still capped at
    # n_days - 1 internally).
    result_explicit = engine.run(
        signal_panel, price_panel, _dates(n_days), instruments, start_idx=0, end_idx=n_days
    )
    assert len(result_explicit.daily) == n_days - 1
    assert result_explicit.summary["final_equity"] == pytest.approx(result.summary["final_equity"])

    # A degenerate start_idx >= effective_end range must not crash and
    # returns an empty (but well-formed) result.
    result_empty = engine.run(
        signal_panel, price_panel, _dates(n_days), instruments, start_idx=n_days - 1, end_idx=n_days
    )
    assert len(result_empty.daily) == 0
    assert result_empty.summary["final_equity"] == EQUITY
    assert result_empty.summary["n_days"] == 0


# ---------------------------------------------------------------------------
# 6. turnover budget respected across days
# ---------------------------------------------------------------------------


def test_turnover_budget_never_exceeded():
    n_days = 12
    n_instruments = 4
    instruments = [f"I{i}" for i in range(n_instruments)]
    price_panel = _deterministic_growth_price_panel(
        n_days, [0.002, -0.003, 0.001, 0.004], [30.0, 45.0, 60.0, 20.0]
    )
    signal_panel = _alternating_signal_panel(n_days, n_instruments)  # forces heavy daily turnover

    turnover_limit = 0.3
    optimizer = AshareLotOptimizer(
        lot_size=1,
        turnover_limit=turnover_limit,
        max_single_name_weight=0.95,
        dust_cleanup_enabled=True,
        dust_cleanup_turnover_limit=0.05,
    )
    engine = BacktestEngine(
        optimizer=optimizer,
        cost_model=TransactionCostModel(cost_bps=2.0, min_commission=1.0),
        signal_variant="long_only",
        target_exposure=0.9,
        max_single_name_weight=0.95,
        initial_equity=EQUITY,
    )

    result = engine.run(signal_panel, price_panel, _dates(n_days), instruments)

    tol = 1e-6
    assert (result.daily["turnover_used"] <= turnover_limit + tol).all()


# ---------------------------------------------------------------------------
# 7. integration with both AshareLotOptimizer and USEquityOptimizer
# ---------------------------------------------------------------------------


def test_ashare_optimizer_integration_whole_lots_only():
    n_days = 10
    n_instruments = 4
    instruments = [f"I{i}" for i in range(n_instruments)]
    rng = np.random.default_rng(2)
    rates = rng.uniform(-0.01, 0.01, size=n_instruments)
    price_panel = _deterministic_growth_price_panel(n_days, list(rates), [12.0, 33.0, 55.0, 8.0])
    signal_panel = rng.uniform(-1.0, 1.0, size=(n_days, n_instruments))

    inner = AshareLotOptimizer(lot_size=100)  # default: long-only, 100-share lots
    spy = _RecordingOptimizer(inner)
    engine = BacktestEngine(
        optimizer=spy,
        cost_model=TransactionCostModel(cost_bps=5.0, min_commission=5.0),
        signal_variant="long_only",
        target_exposure=0.9,
        max_single_name_weight=0.3,
        initial_equity=EQUITY,
    )

    result = engine.run(signal_panel, price_panel, _dates(n_days), instruments)

    assert len(result.daily) == n_days - 1
    assert np.isfinite(result.summary["final_equity"])

    for plan in spy.plans:
        shares = plan.new_shares.to_numpy()
        lots = shares / 100.0
        assert np.allclose(lots, np.round(lots), atol=1e-6), "AshareLotOptimizer produced a fractional lot"
        assert (shares >= -1e-9).all()  # long-only default: never short


def test_us_optimizer_integration_allows_shorting():
    n_days = 10
    n_instruments = 4
    instruments = [f"I{i}" for i in range(n_instruments)]
    rng = np.random.default_rng(3)
    rates = rng.uniform(-0.01, 0.01, size=n_instruments)
    price_panel = _deterministic_growth_price_panel(n_days, list(rates), [12.0, 33.0, 55.0, 8.0])
    signal_panel = rng.uniform(-1.0, 1.0, size=(n_days, n_instruments))

    inner = USEquityOptimizer()  # default: short-able, 1-share lots
    spy = _RecordingOptimizer(inner)
    engine = BacktestEngine(
        optimizer=spy,
        cost_model=TransactionCostModel(cost_bps=5.0, min_commission=5.0),
        signal_variant="raw_continuous",  # keeps negative (short) signal
        target_exposure=0.9,
        max_single_name_weight=0.3,
        initial_equity=EQUITY,
    )

    result = engine.run(signal_panel, price_panel, _dates(n_days), instruments)

    assert len(result.daily) == n_days - 1
    assert np.isfinite(result.summary["final_equity"])

    any_short = any((plan.new_shares < -1e-9).any() for plan in spy.plans)
    assert any_short, "expected at least one short position across the backtest"
