import numpy as np
import pandas as pd
import pytest

from alphacube.execution.almgren_chriss import (
    AlmgrenChrissParams,
    almgren_chriss_trajectory,
    annualized_vol_to_interval_vol,
    efficient_frontier,
    schedule_trade_for_symbol,
    schedule_trades_for_rebalance,
)


def make_params(**overrides) -> AlmgrenChrissParams:
    defaults = dict(
        total_shares=100_000.0,
        horizon=1.0,  # 1 trading day
        n_intervals=10,
        volatility=0.02,  # per-interval vol
        temporary_impact=2.5e-6,
        permanent_impact=2.5e-7,
        risk_aversion=0.0,
    )
    defaults.update(overrides)
    return AlmgrenChrissParams(**defaults)


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("risk_aversion", [0.0, 1e-7, 1e-6, 1e-5, 1e-3])
def test_boundary_conditions(risk_aversion):
    params = make_params(risk_aversion=risk_aversion)
    traj = almgren_chriss_trajectory(params)

    assert traj.holdings[0] == params.total_shares
    assert traj.holdings[-1] == pytest.approx(0.0, abs=1e-8)
    assert len(traj.holdings) == params.n_intervals + 1
    assert len(traj.trades) == params.n_intervals
    assert len(traj.times) == params.n_intervals + 1
    assert traj.times[0] == 0.0
    assert traj.times[-1] == pytest.approx(params.horizon)
    assert np.sum(traj.trades) == pytest.approx(params.total_shares, rel=1e-8)


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "risk_aversion,eta,gamma",
    [
        (0.0, 2.5e-6, 2.5e-7),
        (1e-6, 2.5e-6, 2.5e-7),
        (1e-4, 5e-6, 1e-6),
        (1.0, 1e-5, 1e-6),
    ],
)
def test_monotonic_nonincreasing_holdings_for_sell_program(risk_aversion, eta, gamma):
    params = make_params(risk_aversion=risk_aversion, temporary_impact=eta, permanent_impact=gamma)
    traj = almgren_chriss_trajectory(params)
    diffs = np.diff(traj.holdings)
    assert np.all(diffs <= 1e-6), f"holdings not monotonically non-increasing: {traj.holdings}"


# ---------------------------------------------------------------------------
# Zero risk-aversion -> TWAP
# ---------------------------------------------------------------------------


def test_zero_risk_aversion_is_twap():
    params = make_params(risk_aversion=0.0)
    traj = almgren_chriss_trajectory(params)

    assert traj.is_linear_limit is True
    expected_trade = params.total_shares / params.n_intervals
    assert traj.trades == pytest.approx(np.full(params.n_intervals, expected_trade), rel=1e-9)


# ---------------------------------------------------------------------------
# High risk-aversion -> front-loaded
# ---------------------------------------------------------------------------


def test_high_risk_aversion_front_loads_trading():
    params = make_params(risk_aversion=1e-4)
    traj = almgren_chriss_trajectory(params)

    assert traj.is_linear_limit is False
    assert traj.trades[0] > traj.trades[-1]
    # Front-loading should also be monotonically decreasing trade sizes
    # (standard AC hyperbolic-trajectory property).
    assert np.all(np.diff(traj.trades) <= 1e-6)


# ---------------------------------------------------------------------------
# Cost sanity: E[cost] increases, Var[cost] decreases with risk_aversion
# ---------------------------------------------------------------------------


def test_efficient_frontier_tradeoff_direction():
    params = make_params(risk_aversion=0.0)
    risk_aversions = [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    frontier = efficient_frontier(params, risk_aversions)

    print("\nEfficient frontier:\n", frontier.to_string(index=False))

    assert list(frontier["risk_aversion"]) == risk_aversions
    # expected_cost should be non-decreasing (strictly increasing once kappa
    # leaves the linear-limit regime) as risk_aversion increases
    assert np.all(np.diff(frontier["expected_cost"]) >= -1e-9)
    assert frontier["expected_cost"].iloc[-1] > frontier["expected_cost"].iloc[0]
    # cost_variance should be non-increasing as risk_aversion increases
    assert np.all(np.diff(frontier["cost_variance"]) <= 1e-9)
    assert frontier["cost_variance"].iloc[-1] < frontier["cost_variance"].iloc[0]


# ---------------------------------------------------------------------------
# Degenerate case: total_shares = 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("risk_aversion", [0.0, 1e-5])
def test_zero_shares_is_trivial(risk_aversion):
    params = make_params(total_shares=0.0, risk_aversion=risk_aversion)
    traj = almgren_chriss_trajectory(params)

    assert np.all(traj.holdings == 0.0)
    assert np.all(traj.trades == 0.0)
    assert traj.expected_cost == 0.0
    assert traj.cost_variance == 0.0
    assert traj.cost_std == 0.0
    assert not np.any(np.isnan(traj.holdings))
    assert not np.any(np.isnan(traj.trades))


# ---------------------------------------------------------------------------
# Continuous-time limiting case check (documented in module docstring)
# ---------------------------------------------------------------------------


def test_discrete_kappa_approaches_continuous_limit_for_small_tau():
    sigma, eta, gamma, lam = 0.001, 2.5e-6, 2.5e-7, 1e-6
    # Use a large n_intervals (small tau) so tau^2*kappa_tilde^2 is small.
    params = make_params(
        n_intervals=2000, horizon=1.0, volatility=sigma, temporary_impact=eta,
        permanent_impact=gamma, risk_aversion=lam,
    )
    traj = almgren_chriss_trajectory(params)
    tau = params.tau
    eta_tilde = eta - gamma * tau / 2.0
    kappa_continuous = np.sqrt(lam * sigma**2 / eta_tilde)
    assert traj.kappa == pytest.approx(kappa_continuous, rel=1e-3)


# ---------------------------------------------------------------------------
# annualized_vol_to_interval_vol helper
# ---------------------------------------------------------------------------


def test_annualized_vol_helper():
    annual_vol = 0.32
    trading_days_per_year = 252
    tau_in_days = 1.0
    sigma = annualized_vol_to_interval_vol(annual_vol, trading_days_per_year, tau_in_days)
    assert sigma == pytest.approx(annual_vol * np.sqrt(1.0 / 252))


# ---------------------------------------------------------------------------
# Bridge functions
# ---------------------------------------------------------------------------


def test_schedule_trade_for_symbol():
    traj = schedule_trade_for_symbol(
        trade_shares=50_000.0,
        horizon=1.0,
        n_intervals=13,
        volatility=0.015,
        temporary_impact=3e-6,
        permanent_impact=3e-7,
        risk_aversion=5e-6,
    )
    assert traj.holdings[0] == 50_000.0
    assert traj.holdings[-1] == pytest.approx(0.0, abs=1e-8)
    assert len(traj.trades) == 13


def test_schedule_trades_for_rebalance_skips_missing_symbols(caplog):
    trades = pd.Series({"AAPL": 10_000.0, "MSFT": -5_000.0, "GOOG": 2_000.0, "FLAT": 0.0})
    market_params = pd.DataFrame(
        {
            "volatility": [0.02, 0.018],
            "temporary_impact": [2e-6, 2.5e-6],
            "permanent_impact": [2e-7, 2.5e-7],
        },
        index=["AAPL", "MSFT"],
    )

    with caplog.at_level("WARNING"):
        result = schedule_trades_for_rebalance(
            trades, market_params, horizon=1.0, n_intervals=10, risk_aversion=1e-6
        )

    assert set(result.keys()) == {"AAPL", "MSFT"}
    assert "FLAT" not in result  # zero trade, nothing to schedule
    assert "GOOG" not in result  # missing market params, skipped not errored

    assert result["AAPL"].holdings[0] == 10_000.0
    assert result["MSFT"].holdings[0] == -5_000.0
    for traj in result.values():
        assert not np.any(np.isnan(traj.holdings))

    assert any("GOOG" in record.message for record in caplog.records)
