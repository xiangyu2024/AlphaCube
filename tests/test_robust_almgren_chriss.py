import numpy as np
import pytest

from alphacube.execution.almgren_chriss import (
    AlmgrenChrissParams,
    ExecutionTrajectory,
    almgren_chriss_trajectory,
)
from alphacube.execution.robust_almgren_chriss import (
    EllipsoidalUncertaintySet,
    RobustExecutionTrajectory,
    compare_robust_vs_nominal,
    robust_almgren_chriss_trajectory,
    worst_case_cost,
)


# ---------------------------------------------------------------------------
# Shared worked-example parameters (mirrors almgren_chriss.py's own worked
# example: ~100,000 shares, 1 trading day, 10 intervals).
# ---------------------------------------------------------------------------

TOTAL_SHARES = 100_000.0
HORIZON = 1.0
N_INTERVALS = 10
VOLATILITY = 0.02
ETA_HAT = 2.5e-6
GAMMA_HAT = 2.5e-7


def make_synthetic_trajectory(holdings, trades, times) -> ExecutionTrajectory:
    """Build an ExecutionTrajectory directly (bypassing almgren_chriss_trajectory)
    with a known, hand-picked shape, so the worst_case_cost formula can be
    checked against brute force without depending on the AC solver's own
    correctness (which is already covered by tests/test_almgren_chriss.py).
    The expected_cost/cost_variance/etc. fields are irrelevant to
    worst_case_cost (which recomputes cost-linear-coefficients from
    holdings/trades/times only), so they're filled with placeholder values.
    """
    return ExecutionTrajectory(
        holdings=np.asarray(holdings, dtype=float),
        trades=np.asarray(trades, dtype=float),
        times=np.asarray(times, dtype=float),
        expected_cost=float("nan"),
        cost_variance=float("nan"),
        cost_std=float("nan"),
        kappa=float("nan"),
        is_linear_limit=False,
    )


def _symmetric_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Matrix square root of a symmetric PD matrix via eigendecomposition
    (test-only helper, used to brute-force-sample the ellipsoid boundary
    theta = theta_hat + rho*Sigma^(1/2)*u for ||u||=1 -- see module
    docstring's derivation, which this test verifies independently)."""
    eigvals, eigvecs = np.linalg.eigh(matrix)
    assert np.all(eigvals > 0), "matrix must be positive definite"
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


# ---------------------------------------------------------------------------
# Worst-case formula correctness: closed form vs. brute-force search over
# the ellipsoid boundary, for a KNOWN linear cost function.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sigma,rho,eta_hat,gamma_hat",
    [
        (np.eye(2), 1.5, 2.5e-6, 2.5e-7),  # plain ball (sigma = identity)
        (np.array([[9e-14, -5.25e-12], [-5.25e-12, 6.25e-10]]), 2.4477, 2.5e-6, 2.5e-7),  # anisotropic + correlated
        (np.array([[4e-13, 0.0], [0.0, 1e-11]]), 3.0, 1e-6, 5e-7),  # axis-aligned, very different scales
    ],
)
def test_worst_case_cost_matches_brute_force(sigma, rho, eta_hat, gamma_hat):
    # A trajectory with distinct, nonzero c_eta and c_gamma (see
    # _cost_linear_coefficients): X=100_000 split unevenly into two trades.
    trajectory = make_synthetic_trajectory(
        holdings=[100_000.0, 30_000.0, 0.0],
        trades=[70_000.0, 30_000.0],
        times=[0.0, 0.5, 1.0],
    )
    uncertainty_set = EllipsoidalUncertaintySet(eta_hat=eta_hat, gamma_hat=gamma_hat, sigma=sigma, rho=rho)

    from alphacube.execution.robust_almgren_chriss import _cost_linear_coefficients

    c = _cost_linear_coefficients(trajectory)
    theta_hat = np.array([eta_hat, gamma_hat])
    sqrt_sigma = _symmetric_sqrt(sigma)

    angles = np.linspace(0.0, 2.0 * np.pi, 20_000)
    u = np.stack([np.cos(angles), np.sin(angles)], axis=0)  # 2 x n
    thetas = theta_hat[:, None] + rho * (sqrt_sigma @ u)  # 2 x n
    brute_force_costs = c @ thetas
    brute_force_max = float(np.max(brute_force_costs))

    closed_form = worst_case_cost(trajectory, uncertainty_set)

    assert closed_form == pytest.approx(brute_force_max, rel=1e-3)
    # Sanity: the closed form must also be >= the cost AT theta_hat itself
    # (theta_hat is always inside U, so it can never beat the max over U).
    assert closed_form >= float(c @ theta_hat) - 1e-6


# ---------------------------------------------------------------------------
# Directional/elliptical structure actually matters (isn't silently
# collapsed to a symmetric ball).
# ---------------------------------------------------------------------------


def test_elliptical_uncertainty_differs_from_symmetric_ball_of_same_average_radius():
    trajectory = make_synthetic_trajectory(
        holdings=[100_000.0, 30_000.0, 0.0],
        trades=[70_000.0, 30_000.0],
        times=[0.0, 0.5, 1.0],
    )
    rho = 2.0

    # Anisotropic, uncorrelated: eta uncertainty much larger than gamma's.
    sigma_aniso = np.array([[1e-12, 0.0], [0.0, 1e-16]])
    # A symmetric ball with the SAME average variance (trace/2) on each axis
    # and no correlation -- what you'd get if you (wrongly) collapsed the
    # ellipse to a ball instead of respecting the per-axis scale.
    avg_var = float(np.trace(sigma_aniso)) / 2.0
    sigma_ball = np.eye(2) * avg_var

    us_aniso = EllipsoidalUncertaintySet(eta_hat=ETA_HAT, gamma_hat=GAMMA_HAT, sigma=sigma_aniso, rho=rho)
    us_ball = EllipsoidalUncertaintySet(eta_hat=ETA_HAT, gamma_hat=GAMMA_HAT, sigma=sigma_ball, rho=rho)

    wc_aniso = worst_case_cost(trajectory, us_aniso)
    wc_ball = worst_case_cost(trajectory, us_ball)

    # Same "average" uncertainty magnitude (trace), very different worst
    # case once direction is respected -- the elliptical structure is doing
    # real work, not collapsing to an equivalent ball.
    relative_gap = abs(wc_aniso - wc_ball) / wc_ball
    assert relative_gap > 0.05, (
        f"expected elliptical vs. same-average-radius ball to differ meaningfully, "
        f"got wc_aniso={wc_aniso!r} wc_ball={wc_ball!r} (relative gap {relative_gap!r})"
    )

    # Also check correlation alone (same per-axis variances, only the
    # off-diagonal differs) moves the worst case.
    sig_eta2, sig_gamma2 = 1e-12, 1e-14
    sigma_uncorrelated = np.array([[sig_eta2, 0.0], [0.0, sig_gamma2]])
    sigma_correlated = np.array(
        [[sig_eta2, 0.9 * np.sqrt(sig_eta2 * sig_gamma2)], [0.9 * np.sqrt(sig_eta2 * sig_gamma2), sig_gamma2]]
    )
    us_uncorrelated = EllipsoidalUncertaintySet(eta_hat=ETA_HAT, gamma_hat=GAMMA_HAT, sigma=sigma_uncorrelated, rho=rho)
    us_correlated = EllipsoidalUncertaintySet(eta_hat=ETA_HAT, gamma_hat=GAMMA_HAT, sigma=sigma_correlated, rho=rho)
    wc_uncorrelated = worst_case_cost(trajectory, us_uncorrelated)
    wc_correlated = worst_case_cost(trajectory, us_correlated)
    assert wc_correlated > wc_uncorrelated  # positive correlation with positive c widens the worst case


# ---------------------------------------------------------------------------
# Robust trajectory sacrifices nominal performance for worst-case protection.
# ---------------------------------------------------------------------------


def _tradeoff_uncertainty_set() -> EllipsoidalUncertaintySet:
    # Tuned so that the risk_aversion search genuinely finds an interior
    # optimum away from the nominal-cost-minimizing (TWAP) corner: gamma's
    # absolute uncertainty is large (permanent impact is notoriously hard to
    # calibrate) and negatively correlated with eta's, so front-loading
    # (which raises eta-exposure but lowers gamma-exposure, see module
    # docstring) reduces worst-case cost despite raising nominal cost.
    sig_eta, sig_gamma, corr = 3e-7, 2.5e-5, -0.7
    sigma = np.array(
        [
            [sig_eta**2, corr * sig_eta * sig_gamma],
            [corr * sig_eta * sig_gamma, sig_gamma**2],
        ]
    )
    return EllipsoidalUncertaintySet.from_confidence_level(ETA_HAT, GAMMA_HAT, sigma, confidence=0.95)


def test_robust_trajectory_trades_nominal_cost_for_worst_case_protection():
    uncertainty_set = _tradeoff_uncertainty_set()

    df = compare_robust_vs_nominal(
        TOTAL_SHARES,
        HORIZON,
        N_INTERVALS,
        VOLATILITY,
        ETA_HAT,
        GAMMA_HAT,
        uncertainty_set.sigma,
        uncertainty_set.rho,
        nominal_risk_aversion=1e-6,
        risk_aversion_search_range=(1e-9, 5.0),
    )
    nominal_row = df[df["strategy"] == "nominal"].iloc[0]
    robust_row = df[df["strategy"] == "robust"].iloc[0]

    # Small (here: nonzero) sacrifice in nominal-case cost...
    assert robust_row["cost_at_theta_hat"] >= nominal_row["cost_at_theta_hat"]
    # ...for meaningful worst-case protection.
    assert robust_row["cost_at_worst_case"] < nominal_row["cost_at_worst_case"]
    improvement = (nominal_row["cost_at_worst_case"] - robust_row["cost_at_worst_case"]) / nominal_row[
        "cost_at_worst_case"
    ]
    assert improvement > 0.05, f"expected >5% worst-case improvement, got {improvement:.4f}"


# ---------------------------------------------------------------------------
# Degenerate case: rho = 0 (no uncertainty).
# ---------------------------------------------------------------------------


def test_zero_rho_worst_case_equals_nominal_cost():
    uncertainty_set = EllipsoidalUncertaintySet(eta_hat=ETA_HAT, gamma_hat=GAMMA_HAT, sigma=np.eye(2), rho=0.0)
    params = AlmgrenChrissParams(
        total_shares=TOTAL_SHARES,
        horizon=HORIZON,
        n_intervals=N_INTERVALS,
        volatility=VOLATILITY,
        temporary_impact=ETA_HAT,
        permanent_impact=GAMMA_HAT,
        risk_aversion=1e-5,
    )
    trajectory = almgren_chriss_trajectory(params)

    wc = worst_case_cost(trajectory, uncertainty_set)
    assert wc == pytest.approx(trajectory.expected_cost, rel=1e-9)


def test_zero_rho_robust_search_reduces_to_nominal_cost_minimization():
    # With rho=0, worst_case_cost(theta) == E[cost](theta_hat) exactly for
    # every trajectory, so the robust search is *identical* to minimizing
    # nominal E[cost] over risk_aversion -- which (per
    # test_efficient_frontier_tradeoff_direction in test_almgren_chriss.py)
    # is minimized in the TWAP limit, risk_aversion -> 0.
    uncertainty_set = EllipsoidalUncertaintySet(eta_hat=ETA_HAT, gamma_hat=GAMMA_HAT, sigma=np.eye(2), rho=0.0)

    robust = robust_almgren_chriss_trajectory(
        TOTAL_SHARES,
        HORIZON,
        N_INTERVALS,
        VOLATILITY,
        uncertainty_set,
        risk_aversion_search_range=(1e-9, 5.0),
    )

    # worst_case_cost must exactly equal nominal_cost when rho=0 (the
    # sqrt(...) spread term is identically zero regardless of trajectory).
    assert robust.worst_case_cost == pytest.approx(robust.nominal_cost, rel=1e-9)

    # The chosen trajectory should have converged to (very nearly) the
    # TWAP/nominal-cost-minimizing trajectory: compare directly against the
    # plain almgren_chriss_trajectory built at the search range's lower
    # bound (which is, by construction, indistinguishable from TWAP -- see
    # almgren_chriss.py's _LINEAR_LIMIT_KAPPA_T).
    twap_params = AlmgrenChrissParams(
        total_shares=TOTAL_SHARES,
        horizon=HORIZON,
        n_intervals=N_INTERVALS,
        volatility=VOLATILITY,
        temporary_impact=ETA_HAT,
        permanent_impact=GAMMA_HAT,
        risk_aversion=0.0,
    )
    twap_trajectory = almgren_chriss_trajectory(twap_params)
    assert robust.nominal_cost == pytest.approx(twap_trajectory.expected_cost, rel=1e-3)
    assert robust.trajectory.trades == pytest.approx(twap_trajectory.trades, rel=1e-3)


# ---------------------------------------------------------------------------
# Integration: the returned trajectory obeys almgren_chriss.py's own
# ExecutionTrajectory invariants.
# ---------------------------------------------------------------------------


def test_robust_trajectory_satisfies_execution_trajectory_invariants():
    uncertainty_set = _tradeoff_uncertainty_set()
    robust = robust_almgren_chriss_trajectory(
        TOTAL_SHARES,
        HORIZON,
        N_INTERVALS,
        VOLATILITY,
        uncertainty_set,
        risk_aversion_search_range=(1e-9, 5.0),
    )
    traj = robust.trajectory

    assert isinstance(robust, RobustExecutionTrajectory)
    assert isinstance(traj, ExecutionTrajectory)
    assert traj.holdings[0] == TOTAL_SHARES
    assert traj.holdings[-1] == pytest.approx(0.0, abs=1e-6)
    assert len(traj.holdings) == N_INTERVALS + 1
    assert len(traj.trades) == N_INTERVALS
    assert len(traj.times) == N_INTERVALS + 1
    assert traj.times[0] == 0.0
    assert traj.times[-1] == pytest.approx(HORIZON)
    assert np.sum(traj.trades) == pytest.approx(TOTAL_SHARES, rel=1e-6)
    # Monotonically non-increasing holdings (sell program).
    assert np.all(np.diff(traj.holdings) <= 1e-6)
    assert not np.any(np.isnan(traj.holdings))
    assert not np.any(np.isnan(traj.trades))

    assert robust.nominal_cost >= 0.0
    assert robust.worst_case_cost >= robust.nominal_cost - 1e-6
    assert robust.uncertainty_set is uncertainty_set


# ---------------------------------------------------------------------------
# EllipsoidalUncertaintySet construction / validation.
# ---------------------------------------------------------------------------


def test_from_confidence_level_rho_matches_chi_squared_quantile():
    from scipy.stats import chi2

    sigma = np.eye(2) * 1e-12
    us = EllipsoidalUncertaintySet.from_confidence_level(ETA_HAT, GAMMA_HAT, sigma, confidence=0.95)
    expected_rho = np.sqrt(chi2.ppf(0.95, df=2))
    assert us.rho == pytest.approx(expected_rho)


def test_uncertainty_set_rejects_non_positive_definite_sigma():
    with pytest.raises(ValueError):
        EllipsoidalUncertaintySet(eta_hat=ETA_HAT, gamma_hat=GAMMA_HAT, sigma=np.array([[1.0, 2.0], [2.0, 1.0]]), rho=1.0)


def test_uncertainty_set_rejects_negative_rho():
    with pytest.raises(ValueError):
        EllipsoidalUncertaintySet(eta_hat=ETA_HAT, gamma_hat=GAMMA_HAT, sigma=np.eye(2), rho=-1.0)


def test_from_confidence_level_rejects_bad_confidence():
    with pytest.raises(ValueError):
        EllipsoidalUncertaintySet.from_confidence_level(ETA_HAT, GAMMA_HAT, np.eye(2), confidence=1.5)


# ---------------------------------------------------------------------------
# compare_robust_vs_nominal: shape/columns sanity.
# ---------------------------------------------------------------------------


def test_compare_robust_vs_nominal_shape():
    uncertainty_set = _tradeoff_uncertainty_set()
    df = compare_robust_vs_nominal(
        TOTAL_SHARES,
        HORIZON,
        N_INTERVALS,
        VOLATILITY,
        ETA_HAT,
        GAMMA_HAT,
        uncertainty_set.sigma,
        uncertainty_set.rho,
    )
    assert list(df["strategy"]) == ["nominal", "robust"]
    assert list(df.columns) == ["strategy", "risk_aversion", "cost_at_theta_hat", "cost_at_worst_case"]
    assert np.all(df["cost_at_theta_hat"] >= 0)
    assert np.all(df["cost_at_worst_case"] >= df["cost_at_theta_hat"] - 1e-6)
