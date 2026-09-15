"""Convex, cardinality-sparse rebalance optimizer with per-asset fixed costs.

Adapted from Chen, Ahipasaoglu, Zhang, Yang, "Robust and Sparse Portfolio
Selection: Quantitative Insights and Efficient Algorithms" (the "RSMV"
model), arXiv:2412.19462. The paper solves the *static allocation* problem

    minimize_w   0.5*kappa*w'Sigma*w - r'w + phi' @ 1(w != 0)
    subject to   sum(w) == 1,  w_lo <= w <= w_hi

via a DC-programming outer loop (proximal DCA) around a capped-L1 surrogate
for the NP-hard cardinality indicator, with a semismooth-Newton method
(their Algorithm 2 / SN-pDCA) solving the convex inner subproblem, and a
Kurdyka-Lojasiewicz argument giving linear (local) convergence of the outer
loop.

This module keeps the paper's outer DCA structure and capped-L1 surrogate
faithful, but makes two deliberate departures, both documented inline where
they matter and both intentional simplifications, not oversights:

1. **Rebalance, not static allocation.** ``optimize_base.py``'s
   :class:`~alphacube.portfolio.optimize_base.TurnoverBudgetOptimizer` (and
   the stage-5 contract inherited from the previous project's --
   AlphaFactorMining's -- ``docs/pipeline.md``) frames portfolio
   optimization around a *previous* position and a *trade*, not a
   from-scratch weight vector -- and a fixed cost genuinely belongs on the
   trade, not the resulting weight: holding a name unchanged from yesterday
   should never cost ``phi_i`` again today, only a name that is actually
   *bought or sold* should. So this module reparameterizes the paper's ``w``
   as ``w_prev + x`` and applies the cardinality/fixed-cost term to the
   *trade* ``x``, not the post-trade weight:

       minimize_x   0.5*kappa*(w_prev+x)'Sigma*(w_prev+x) - r'(w_prev+x)
                     + phi' @ 1(x != 0)
       subject to   sum(w_prev+x) == target_sum_weights
                    w_lo <= w_prev+x <= w_hi

   The previous project's ``docs/pipeline.md`` (stage-5 section) calls the
   existing ``TurnoverBudgetOptimizer`` fill "a greedy heuristic, explicitly the
   simple version ... not a true convex/mean-variance portfolio optimizer
   with a proper risk model" and names "a more principled optimizer" against
   an actual risk model as natural future work; this module is that.  It is
   a sibling of ``TurnoverBudgetOptimizer``, not a replacement -- it has no
   notion of lot size, turnover budget, or untradable-name freezing, and
   does not produce whole-share order counts. Bridging its continuous
   ``trade``/``new_weights`` output into executable lots is left to the
   existing ``optimize_ashare.py``/``optimize_us.py`` machinery (or a future
   integration), by design.

2. **Proximal gradient descent instead of semismooth Newton for the inner
   subproblem.** The paper's inner solver works a dual reformulation with a
   semismooth-Newton method for fast (locally quadratic) convergence. That
   is a meaningfully more involved piece of numerics than the rest of this
   project's stage-5 code. This module instead solves the (still convex)
   inner subproblem with plain **projected proximal gradient descent**:
   fewer moving parts, easy to verify line by line, at the cost of a
   (theoretically) slower, linear per-inner-iteration convergence rate
   instead of Newton's superlinear rate. The outer DCA structure -- and the
   capped-L1 surrogate it linearizes each round -- is otherwise faithful to
   the paper.

Everything below operates on plain ``numpy`` arrays internally; ``pandas``
only appears at the alignment boundary (inputs/outputs), matching this
project's convention elsewhere in ``portfolio/``.

**DC decomposition of the cardinality surrogate.** For threshold ``t > 0``
and per-asset fixed cost ``phi_i``, the capped-L1 surrogate for
``1(x_i != 0)`` is ``phi_i * min(|x_i|/t, 1)``, decomposed as
``p_t(x_i) - q_t(x_i)`` with

    p_t(x_i) = (phi_i/t) * |x_i|                                  (convex)
    q_t(x_i) = phi_i * max(0, x_i/t - 1, -x_i/t - 1)               (convex)

``t`` is in the same units as ``x`` (portfolio weight, e.g. 0.01 = 1% of
equity) -- the default ``1e-3`` assumes trade sizes on the order of a few
percent of equity or larger are "real" and anything below ~10bp is noise;
callers with a very different typical trade-size scale should override it.

**Outer DCA loop** (see :meth:`ConvexSparsePortfolioOptimizer.optimize`):
at iterate ``x^k``, linearize ``q_t`` via a subgradient ``s^k``
(``s^k_i = phi_i/t`` if ``x^k_i >= t``, ``-phi_i/t`` if ``x^k_i <= -t``, else
``0``), solve the resulting convex majorized subproblem

    minimize_x   g(x) + p_t(x),   g(x) = 0.5*kappa*(w_prev+x)'Sigma(w_prev+x)
                                          - r'(w_prev+x) - <s^k, x>

by projected proximal gradient (closed-form soft-threshold prox for the
weighted-L1 ``p_t``, step size ``eta = 1/L`` with
``L = kappa * lambda_max(Sigma)`` the Lipschitz constant of ``grad g``,
computed once via ``np.linalg.eigvalsh`` and reused across every inner
iteration), then re-linearize at the new iterate and repeat until
``||x^{k+1} - x^k||_2 < tol`` or ``max_outer_iterations`` is hit.

**Projection.** After each proximal-gradient step the iterate is projected
onto ``{x : sum(w_prev+x) = target_sum_weights, w_lo <= w_prev+x <= w_hi}``,
i.e. a box intersected with a hyperplane. This has a standard closed-ish
form: the Euclidean projection of any point ``z`` onto ``lo <= x <= hi``
with ``sum(x) = c`` is ``x_i = clip(z_i - mu, lo_i, hi_i)`` for the unique
scalar Lagrange multiplier ``mu`` solving ``sum_i clip(z_i-mu,lo_i,hi_i) =
c`` -- the left side is a continuous, non-increasing, piecewise-linear
function of ``mu``, so ``mu`` is found by bisection (see
``_project_box_sum_equality``). This is used instead of a generic
``scipy.optimize.minimize`` call at every inner iteration purely for speed
(this projection runs ``max_outer_iterations * inner_iterations`` times per
``optimize()`` call); a general-purpose QP/SLSQP solver would also be
"correct" here per the assignment's stated tolerance for simplifying the
inner solver, but the bisection is a well-known, easy-to-verify closed
algorithm for exactly this box+hyperplane projection, so there is no real
accuracy/complexity tradeoff being made by hand-rolling it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd

# Anything traded by less than this (in weight units) is treated as an exact
# zero when reporting ``n_traded`` -- distinct from ``capped_l1_threshold``
# (which controls the *shape* of the DC surrogate), this is purely a
# reporting cutoff. Soft-thresholding in the inner solver drives
# uneconomical trades to exactly (or numerically indistinguishable from)
# zero, so this can safely be tight.
_TRADE_ZERO_TOL = 1e-8

# Outer-loop / inner-loop numerical floors.
_MIN_LIPSCHITZ = 1e-8
_BISECTION_ITERS = 60


def _project_box_sum_equality(
    z: np.ndarray, lo: np.ndarray, hi: np.ndarray, target_sum: float
) -> np.ndarray:
    """Euclidean projection of ``z`` onto ``{x : lo<=x<=hi, sum(x)=target_sum}``.

    Solved via bisection on the scalar Lagrange multiplier ``mu`` of the sum
    constraint: the projection is ``x_i = clip(z_i - mu, lo_i, hi_i)`` for
    the ``mu`` solving ``sum_i clip(z_i-mu,lo_i,hi_i) == target_sum`` (a
    continuous, non-increasing, piecewise-linear function of ``mu``, hence a
    unique root found by bisection).

    If ``target_sum`` is infeasible given ``lo``/``hi`` (i.e. outside
    ``[sum(lo), sum(hi)]`` -- can only happen if the caller's box/target
    combination is itself infeasible, e.g. too few assets for the requested
    exposure under a tight box), this degrades gracefully to the closest
    feasible box corner (all-``lo`` or all-``hi``) rather than raising mid
    proximal-gradient-descent; :meth:`ConvexSparsePortfolioOptimizer.optimize`
    documents this as the expected behavior for a mis-specified box.
    """
    lo_sum = float(np.sum(lo))
    hi_sum = float(np.sum(hi))
    if target_sum <= lo_sum:
        return lo.copy()
    if target_sum >= hi_sum:
        return hi.copy()

    # h(mu) = sum(clip(z-mu, lo, hi)) - target_sum is non-increasing in mu.
    # h(mu_lo) >= 0 and h(mu_hi) <= 0 by construction (see module docstring).
    mu_lo = float(np.min(z - hi))
    mu_hi = float(np.max(z - lo))

    def h(mu: float) -> float:
        return float(np.sum(np.clip(z - mu, lo, hi))) - target_sum

    h_lo = h(mu_lo)
    h_hi = h(mu_hi)
    if h_lo < 0.0:
        mu_lo -= 1.0
    if h_hi > 0.0:
        mu_hi += 1.0

    mu = 0.5 * (mu_lo + mu_hi)
    for _ in range(_BISECTION_ITERS):
        mu = 0.5 * (mu_lo + mu_hi)
        if h(mu) > 0.0:
            mu_lo = mu
        else:
            mu_hi = mu
    return np.clip(z - mu, lo, hi)


def _align_and_validate(
    prev_weights: pd.Series,
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    fixed_cost: Union[pd.Series, float],
) -> tuple:
    """Reindex ``prev_weights``/``expected_returns``/``fixed_cost`` onto the
    union of ``prev_weights``/``expected_returns`` symbols, and validate
    ``cov_matrix`` actually covers that union.

    Fill conventions (mirrors ``optimize_base.align_inputs``'s philosophy,
    adapted to this module's inputs):

    - ``prev_weights``: missing -> ``0.0`` (no prior holding).
    - ``expected_returns``: missing -> ``0.0`` (no return view -> treated as
      market-neutral for that name; a bad *return* guess is a bounded,
      recoverable mistake).
    - ``cov_matrix``: missing entries are NOT filled. A missing/garbage risk
      number silently corrupts the quadratic term far more dangerously than
      a missing return guess, so any symbol in the union without a full
      row+column in ``cov_matrix`` raises ``ValueError`` naming the symbols,
      rather than being silently imputed as 0 (spuriously "risk-free") or
      NaN (silently propagating NaN through the whole solve).
    - ``fixed_cost`` (when a ``pd.Series``): missing -> ``0.0`` (no fixed
      cost specified for that symbol -- trading it is treated as free,
      which is a safe, conservative default in the sense that it never
      *blocks* a trade the caller didn't think to price).
    """
    universe = prev_weights.index.union(expected_returns.index)

    missing_cov = universe.difference(cov_matrix.index).union(
        universe.difference(cov_matrix.columns)
    )
    if len(missing_cov) > 0:
        raise ValueError(
            "cov_matrix does not cover every symbol in prev_weights/"
            f"expected_returns: missing {sorted(str(s) for s in missing_cov)}. "
            "A missing covariance entry is not safely fillable (unlike a "
            "missing return estimate) -- supply a full covariance "
            "row+column for every traded symbol."
        )

    w_prev = prev_weights.reindex(universe, fill_value=0.0).astype(np.float64)
    r = expected_returns.reindex(universe, fill_value=0.0).astype(np.float64)
    sigma = cov_matrix.loc[universe, universe].astype(np.float64)

    if isinstance(fixed_cost, pd.Series):
        phi = fixed_cost.reindex(universe, fill_value=0.0).astype(np.float64)
    else:
        phi = pd.Series(float(fixed_cost), index=universe, dtype=np.float64)

    if (phi < 0.0).any():
        raise ValueError("fixed_cost must be >= 0 for every symbol")

    return w_prev, r, sigma, phi, universe


@dataclass
class ConvexOptimizationResult:
    """Output of a single :meth:`ConvexSparsePortfolioOptimizer.optimize` call.

    All series are indexed by the aligned (union) symbol index.
    """

    trade: pd.Series
    new_weights: pd.Series
    n_traded: int
    objective_value: float
    converged: bool
    n_outer_iterations: int


class ConvexSparsePortfolioOptimizer:
    """DCA-based convex optimizer for cardinality-sparse rebalances.

    See the module docstring for the full formulation, the rebalance-vs-
    static-allocation adaptation, and the semismooth-Newton -> proximal-
    gradient simplification. In one line: this picks the trade ``x`` that
    minimizes mean-variance disutility plus a fixed per-asset cost for every
    asset actually traded, via a DC-programming (proximal DCA) outer loop
    around a capped-L1 surrogate for the trade-cardinality term.

    Args:
        kappa: risk-aversion coefficient (multiplies the quadratic risk
            term); must be > 0.
        fixed_cost: fixed cost ``phi_i`` incurred by trading asset ``i`` at
            all (any nonzero ``x_i``), independent of trade size. Either a
            scalar (same cost for every asset) or a ``pd.Series`` indexed by
            symbol for per-asset costs. Units must match ``expected_returns``
            (e.g. both in fractional-weight/return units, not percent or
            currency) since the two are summed inside one objective.
        capped_l1_threshold: the DC surrogate's ``t`` -- the trade size
            below which the cardinality penalty ramps linearly to zero
            rather than being flat at ``phi_i``. Smaller values track true
            cardinality more closely but are numerically harder (steeper
            soft-threshold, slower outer-loop convergence near the kink).
            Default ``1e-3`` assumes trade sizes are quoted in fractional
            portfolio weight (e.g. 0.01 = 1% of equity); rescale for other
            units.
        w_lo, w_hi: scalar box constraint on post-trade weights
            ``w_prev + x``, applied uniformly to every asset (e.g.
            ``w_lo=0, w_hi=1`` for long-only; ``w_lo=-max_short`` to allow
            shorting). Per-asset box constraints are not supported by this
            module -- ``TurnoverBudgetOptimizer``'s per-name weight cap is
            the place for per-asset limits downstream.
        max_outer_iterations: cap on the DCA outer loop (each outer
            iteration re-linearizes the concave part and re-solves the
            convex majorized subproblem).
        inner_iterations: number of projected proximal-gradient steps run
            per outer iteration to (approximately) solve the convex
            subproblem. The inner loop also exits early once consecutive
            iterates stop moving (< 1e-12, sup norm), so this is a cap, not
            a fixed cost every outer iteration necessarily pays in full.
        tol: outer-loop convergence tolerance on ``||x^{k+1}-x^k||_2``.
    """

    def __init__(
        self,
        kappa: float,
        fixed_cost: Union[pd.Series, float],
        capped_l1_threshold: float = 1e-3,
        w_lo: float = 0.0,
        w_hi: float = 1.0,
        max_outer_iterations: int = 50,
        inner_iterations: int = 200,
        tol: float = 1e-6,
    ) -> None:
        if kappa <= 0.0:
            raise ValueError("kappa must be > 0")
        if capped_l1_threshold <= 0.0:
            raise ValueError("capped_l1_threshold must be > 0")
        if w_lo >= w_hi:
            raise ValueError("w_lo must be < w_hi")
        if max_outer_iterations < 1:
            raise ValueError("max_outer_iterations must be >= 1")
        if inner_iterations < 1:
            raise ValueError("inner_iterations must be >= 1")
        if tol <= 0.0:
            raise ValueError("tol must be > 0")
        if isinstance(fixed_cost, pd.Series):
            if (fixed_cost < 0.0).any():
                raise ValueError("fixed_cost must be >= 0 for every symbol")
        elif fixed_cost < 0.0:
            raise ValueError("fixed_cost must be >= 0")

        self.kappa = float(kappa)
        self.fixed_cost = fixed_cost
        self.capped_l1_threshold = float(capped_l1_threshold)
        self.w_lo = float(w_lo)
        self.w_hi = float(w_hi)
        self.max_outer_iterations = int(max_outer_iterations)
        self.inner_iterations = int(inner_iterations)
        self.tol = float(tol)

    def optimize(
        self,
        prev_weights: pd.Series,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        target_sum_weights: float = 1.0,
    ) -> ConvexOptimizationResult:
        """Solve for the trade ``x`` minimizing risk-adjusted disutility plus
        fixed per-trade costs, subject to ``sum(w_prev+x) == target_sum_weights``
        and ``w_lo <= w_prev+x <= w_hi``.

        ``target_sum_weights`` defaults to ``1.0`` (fully invested, matching
        the paper's ``sum(w)==1``); pass a smaller value (e.g. ``0.95``) to
        deliberately hold cash -- this is an equality constraint either way
        (not an inequality/upper-bound), i.e. the solve always lands exactly
        on the requested total exposure, cash or not, since the paper's
        formulation and its DC surrogate assume an equality budget
        constraint throughout.
        """
        w_prev_s, r_s, sigma_df, phi_s, universe = _align_and_validate(
            prev_weights, expected_returns, cov_matrix, self.fixed_cost
        )
        n = len(universe)
        if n == 0:
            empty = pd.Series(dtype=np.float64)
            return ConvexOptimizationResult(
                trade=empty,
                new_weights=empty,
                n_traded=0,
                objective_value=0.0,
                converged=True,
                n_outer_iterations=0,
            )

        w_prev = w_prev_s.to_numpy()
        r = r_s.to_numpy()
        sigma = sigma_df.to_numpy()
        phi = phi_s.to_numpy()
        kappa = self.kappa
        t = self.capped_l1_threshold

        lo = self.w_lo - w_prev
        hi = self.w_hi - w_prev
        target_sum_x = float(target_sum_weights) - float(w_prev.sum())

        # Lipschitz constant of grad g (quadratic term only -- p_t/q_t are
        # nonsmooth/handled separately) is kappa * lambda_max(Sigma);
        # computed once and reused for every inner proximal-gradient step.
        eigmax = float(np.linalg.eigvalsh(sigma).max())
        lipschitz = max(kappa * eigmax, _MIN_LIPSCHITZ)
        eta = 1.0 / lipschitz
        thresh = eta * phi / t

        x = np.zeros(n, dtype=np.float64)
        converged = False
        n_outer = 0

        for k in range(self.max_outer_iterations):
            n_outer = k + 1

            # Step 1: linearize the concave part q_t at the current iterate.
            s = np.where(x >= t, phi / t, np.where(x <= -t, -phi / t, 0.0))

            # Step 2: projected proximal gradient descent on the resulting
            # convex majorized subproblem, warm-started from x^k.
            x_inner = x.copy()
            for _ in range(self.inner_iterations):
                grad_g = kappa * (sigma @ (w_prev + x_inner)) - r - s
                z = x_inner - eta * grad_g
                x_half = np.sign(z) * np.maximum(np.abs(z) - thresh, 0.0)
                x_new_inner = _project_box_sum_equality(x_half, lo, hi, target_sum_x)
                inner_step = float(np.max(np.abs(x_new_inner - x_inner)))
                x_inner = x_new_inner
                if inner_step < 1e-12:
                    break

            # Step 3: outer convergence check.
            outer_step = float(np.linalg.norm(x_inner - x))
            x = x_inner
            if outer_step < self.tol:
                converged = True
                break

        new_w = w_prev + x
        capped_l1 = np.sum(phi * np.minimum(np.abs(x) / t, 1.0))
        objective = 0.5 * kappa * float(new_w @ sigma @ new_w) - float(r @ new_w) + float(capped_l1)
        n_traded = int(np.sum(np.abs(x) > _TRADE_ZERO_TOL))

        trade = pd.Series(x, index=universe)
        new_weights = pd.Series(new_w, index=universe)

        return ConvexOptimizationResult(
            trade=trade,
            new_weights=new_weights,
            n_traded=n_traded,
            objective_value=objective,
            converged=converged,
            n_outer_iterations=n_outer,
        )
