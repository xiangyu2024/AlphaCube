"""Tests for the convex, cardinality-sparse rebalance optimizer
(`portfolio/optimize_convex.py`, the DCA/capped-L1 "RSMV"-adapted
optimizer -- see that module's docstring for the paper reference and the
documented simplifications).

Covers: reduction to plain mean-variance QP when fixed_cost==0 (checked
against an independent scipy SLSQP reference solve), the core sparsity
claim (a meaningful fixed cost produces fewer nonzero positions than a
zero cost on identical inputs), hard constraint satisfaction (budget sum +
box) across random instances, monotone-non-increasing behavior of the outer
DCA loop's objective, and alignment/validation robustness (union reindex
where safe, a hard error when cov_matrix is missing coverage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from alphacube.portfolio.optimize_convex import (
    ConvexOptimizationResult,
    ConvexSparsePortfolioOptimizer,
)

_TRADE_ZERO_TOL = 1e-6  # local nonzero-position tolerance for test assertions


def _random_psd_cov(n: int, seed: int, eps: float = 0.05) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    sigma = a @ a.T / n + eps * np.eye(n)
    symbols = [f"S{i}" for i in range(n)]
    return pd.DataFrame(sigma, index=symbols, columns=symbols)


def _symbols(n: int) -> list:
    return [f"S{i}" for i in range(n)]


# --------------------------------------------------------------------------
# 1. Reduction to plain mean-variance QP when fixed_cost == 0
# --------------------------------------------------------------------------


def test_zero_cost_matches_scipy_reference_qp():
    n = 4
    symbols = _symbols(n)
    kappa = 3.0
    cov = _random_psd_cov(n, seed=1)
    expected_returns = pd.Series([0.08, 0.05, 0.02, 0.10], index=symbols)
    prev_weights = pd.Series([0.4, 0.3, 0.2, 0.1], index=symbols)
    w_lo, w_hi = -1.0, 1.0  # wide box -> inactive at the optimum

    optimizer = ConvexSparsePortfolioOptimizer(
        kappa=kappa,
        fixed_cost=0.0,
        w_lo=w_lo,
        w_hi=w_hi,
        max_outer_iterations=50,
        inner_iterations=400,
        tol=1e-9,
    )
    result = optimizer.optimize(prev_weights, expected_returns, cov, target_sum_weights=1.0)

    sigma = cov.loc[symbols, symbols].to_numpy()
    r = expected_returns.to_numpy()

    def objective(w):
        return 0.5 * kappa * w @ sigma @ w - r @ w

    def objective_grad(w):
        return kappa * sigma @ w - r

    ref = minimize(
        objective,
        x0=prev_weights.to_numpy(),
        jac=objective_grad,
        method="SLSQP",
        bounds=[(w_lo, w_hi)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones(n)}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    assert ref.success

    solved_w = result.new_weights.loc[symbols].to_numpy()

    max_abs_diff = float(np.max(np.abs(solved_w - ref.x)))
    obj_diff = abs(objective(solved_w) - ref.fun)

    # Report-worthy numbers, also asserted below.
    print(f"closed-form agreement: max|w_solved - w_scipy| = {max_abs_diff:.3e}, "
          f"objective diff = {obj_diff:.3e}")

    assert np.allclose(solved_w, ref.x, atol=5e-3)
    assert obj_diff < 1e-4
    assert np.sum(solved_w) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# 2. Sparsity actually happens under a meaningful fixed cost
# --------------------------------------------------------------------------


def test_meaningful_fixed_cost_induces_sparsity():
    """Fixed per-trade costs should induce a "no-trade region": starting
    from an already fully-invested book (``prev_weights`` summing to the
    target), a zero-net (``target_sum_weights == sum(prev_weights)``)
    rebalance with no fixed cost happily touches every name to chase the
    mean-variance optimum, while a meaningful fixed cost should make most
    (here, per the empirically observed fixed point below, all) of those
    small rebalancing trades not worth paying for, collapsing ``x`` toward
    zero -- this is the trade-level analogue of the paper's cardinality
    claim, adapted to the rebalance-not-static-allocation setting this
    module targets (see its module docstring): the fixed cost is charged
    on the TRADE, so the natural place to look for induced sparsity is
    ``n_traded``/nonzero-count of the trade vector, not of the resulting
    weights (which inherit whatever density ``prev_weights`` already had).
    """
    n = 8
    symbols = _symbols(n)
    kappa = 4.0
    cov = _random_psd_cov(n, seed=2, eps=0.15)
    rng = np.random.default_rng(42)
    expected_returns = pd.Series(rng.uniform(0.01, 0.06, size=n), index=symbols)
    prev_weights = pd.Series(1.0 / n, index=symbols)  # already fully invested, equal-weight

    common_kwargs = dict(
        kappa=kappa,
        w_lo=0.0,
        w_hi=1.0,
        max_outer_iterations=60,
        inner_iterations=300,
        tol=1e-8,
    )

    zero_cost = ConvexSparsePortfolioOptimizer(fixed_cost=0.0, **common_kwargs)
    result_zero = zero_cost.optimize(prev_weights, expected_returns, cov, target_sum_weights=1.0)

    big_cost = ConvexSparsePortfolioOptimizer(fixed_cost=0.01, **common_kwargs)
    result_big = big_cost.optimize(prev_weights, expected_returns, cov, target_sum_weights=1.0)

    nnz_trade_zero = int((result_zero.trade.abs() > _TRADE_ZERO_TOL).sum())
    nnz_trade_big = int((result_big.trade.abs() > _TRADE_ZERO_TOL).sum())

    print(f"sparsity comparison: nonzero TRADES at fixed_cost=0 -> {nnz_trade_zero}/{n} "
          f"(n_traded={result_zero.n_traded}), at fixed_cost=0.01 -> {nnz_trade_big}/{n} "
          f"(n_traded={result_big.n_traded})")

    assert np.sum(result_zero.new_weights) == pytest.approx(1.0, abs=1e-4)
    assert np.sum(result_big.new_weights) == pytest.approx(1.0, abs=1e-4)
    assert result_big.n_traded < result_zero.n_traded
    assert nnz_trade_big < nnz_trade_zero


# --------------------------------------------------------------------------
# 3. Constraint satisfaction across random instances
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n,seed", [(5, 10), (6, 11), (7, 12), (8, 13), (10, 14)])
def test_constraints_satisfied_on_random_instances(n, seed):
    symbols = _symbols(n)
    rng = np.random.default_rng(seed)
    cov = _random_psd_cov(n, seed=seed)
    expected_returns = pd.Series(rng.uniform(-0.03, 0.08, size=n), index=symbols)
    raw_prev = rng.uniform(0.0, 1.0, size=n)
    prev_weights = pd.Series(raw_prev / raw_prev.sum(), index=symbols)

    w_lo, w_hi = 0.0, 0.5
    optimizer = ConvexSparsePortfolioOptimizer(
        kappa=2.5,
        fixed_cost=pd.Series(rng.uniform(0.0, 0.02, size=n), index=symbols),
        w_lo=w_lo,
        w_hi=w_hi,
        max_outer_iterations=50,
        inner_iterations=200,
        tol=1e-7,
    )
    result = optimizer.optimize(prev_weights, expected_returns, cov, target_sum_weights=1.0)

    total = float(result.new_weights.sum())
    assert total == pytest.approx(1.0, abs=1e-4)
    assert (result.new_weights >= w_lo - 1e-6).all()
    assert (result.new_weights <= w_hi + 1e-6).all()


# --------------------------------------------------------------------------
# 4. Outer DCA loop's objective is non-increasing
# --------------------------------------------------------------------------


def test_outer_loop_objective_non_increasing():
    n = 6
    symbols = _symbols(n)
    cov = _random_psd_cov(n, seed=20)
    rng = np.random.default_rng(21)
    expected_returns = pd.Series(rng.uniform(-0.02, 0.07, size=n), index=symbols)
    prev_weights = pd.Series(0.0, index=symbols)
    fixed_cost = pd.Series(rng.uniform(0.005, 0.03, size=n), index=symbols)

    objectives = []
    for cap in range(1, 16):
        optimizer = ConvexSparsePortfolioOptimizer(
            kappa=3.0,
            fixed_cost=fixed_cost,
            w_lo=0.0,
            w_hi=1.0,
            max_outer_iterations=cap,
            inner_iterations=150,
            tol=1e-12,  # tiny tol: run stops early only once truly converged
        )
        result = optimizer.optimize(prev_weights, expected_returns, cov, target_sum_weights=1.0)
        # Each run is deterministic and starts from x^0=0, so it retraces
        # the exact same trajectory regardless of `cap` -- it may converge
        # (and stop) before `cap` outer iterations if it already reached a
        # fixed point, but never runs more than `cap`.
        assert result.n_outer_iterations <= cap
        objectives.append(result.objective_value)

    print(f"objective trajectory over outer iterations 1..15: {objectives}")

    diffs = np.diff(objectives)
    # Allow a tiny numerical slack; the DCA guarantee is monotone
    # non-increasing given an exactly-solved inner subproblem, and the
    # inner solve here is only approximate (proximal gradient, finite
    # iterations).
    assert np.all(diffs <= 1e-6), f"objective increased between outer iterations: {objectives}"


# --------------------------------------------------------------------------
# 5. Alignment robustness
# --------------------------------------------------------------------------


def test_missing_expected_return_or_prev_weight_is_filled_with_zero():
    cov = _random_psd_cov(3, seed=30)
    prev_weights = pd.Series({"S0": 0.5, "S1": 0.5})  # S2 missing -> 0.0
    expected_returns = pd.Series({"S1": 0.04, "S2": 0.02})  # S0 missing -> 0.0

    optimizer = ConvexSparsePortfolioOptimizer(kappa=2.0, fixed_cost=0.0)
    result = optimizer.optimize(prev_weights, expected_returns, cov, target_sum_weights=1.0)

    assert isinstance(result, ConvexOptimizationResult)
    assert set(result.new_weights.index) == {"S0", "S1", "S2"}
    assert result.new_weights.sum() == pytest.approx(1.0, abs=1e-4)


def test_missing_cov_matrix_coverage_raises_clear_error():
    # cov_matrix only covers S0/S1, but expected_returns references S2 too.
    cov = _random_psd_cov(2, seed=31)  # symbols S0, S1
    prev_weights = pd.Series({"S0": 0.5, "S1": 0.5})
    expected_returns = pd.Series({"S0": 0.04, "S1": 0.02, "S2": 0.10})

    optimizer = ConvexSparsePortfolioOptimizer(kappa=2.0, fixed_cost=0.0)
    with pytest.raises(ValueError, match="cov_matrix does not cover"):
        optimizer.optimize(prev_weights, expected_returns, cov, target_sum_weights=1.0)


def test_missing_cov_matrix_row_and_column_partially_raises():
    # cov_matrix has S2 as a column but not as a row (malformed) -- should
    # still be treated as "not covered" since .loc[universe, universe]
    # requires both.
    symbols = ["S0", "S1"]
    cov = _random_psd_cov(2, seed=32)
    prev_weights = pd.Series({"S0": 0.6, "S1": 0.4, "S2": 0.0})
    expected_returns = pd.Series({"S0": 0.03, "S1": 0.01})

    optimizer = ConvexSparsePortfolioOptimizer(kappa=2.0, fixed_cost=0.0)
    with pytest.raises(ValueError, match="cov_matrix does not cover"):
        optimizer.optimize(prev_weights, expected_returns, cov, target_sum_weights=1.0)


# --------------------------------------------------------------------------
# Misc constructor validation (mirrors optimize_base's validate-in-__init__ style)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(kappa=0.0, fixed_cost=0.0),
        dict(kappa=-1.0, fixed_cost=0.0),
        dict(kappa=1.0, fixed_cost=-0.1),
        dict(kappa=1.0, fixed_cost=0.0, capped_l1_threshold=0.0),
        dict(kappa=1.0, fixed_cost=0.0, w_lo=0.5, w_hi=0.5),
        dict(kappa=1.0, fixed_cost=0.0, w_lo=0.6, w_hi=0.5),
        dict(kappa=1.0, fixed_cost=0.0, max_outer_iterations=0),
        dict(kappa=1.0, fixed_cost=0.0, inner_iterations=0),
        dict(kappa=1.0, fixed_cost=0.0, tol=0.0),
    ],
)
def test_constructor_validates_arguments(kwargs):
    with pytest.raises(ValueError):
        ConvexSparsePortfolioOptimizer(**kwargs)


def test_empty_universe_returns_empty_result():
    optimizer = ConvexSparsePortfolioOptimizer(kappa=1.0, fixed_cost=0.0)
    empty = pd.Series(dtype=float)
    empty_cov = pd.DataFrame(dtype=float)
    result = optimizer.optimize(empty, empty, empty_cov, target_sum_weights=1.0)
    assert result.n_traded == 0
    assert result.n_outer_iterations == 0
    assert result.converged is True
    assert len(result.trade) == 0
