"""Robust Almgren-Chriss optimal execution under ellipsoidal impact-parameter
uncertainty.

Relationship to source material (read this before the formulas below)
-----------------------------------------------------------------------
This module was written after reading Ma & Huang, "Robust Reinforcement
Learning in Finance: Modeling Market Impact with Elliptic Uncertainty Sets"
(NeurIPS 2025, arXiv:2510.19950), hoping to find a portable closed-form
robust extension of Almgren-Chriss. **It does not provide one.** That
paper's actual result is a robust *discrete-state MDP*: uncertainty over
perturbations of the transition *kernel*, solved with robust TD-learning.
That is fundamentally different machinery from the continuous-state,
closed-form, linear-impact setting of :mod:`alphacube.execution.almgren_chriss`
-- there is no way to transcribe their theorem into this setting; a discrete
robust-MDP value-iteration result simply doesn't specialize to a closed-form
quadratic-cost trajectory formula.

What *is* borrowed from that paper is a concept, not a formula: robustness
against an **ellipsoidal ("elliptic") uncertainty set** -- a directionally
shaped region of parameter uncertainty, as opposed to a symmetric ball
around each parameter independently. Their empirical finding motivating
this choice: naive symmetric worst-case robustness was often *worse* than
not being robust at all (too conservative, defends equally against
directions of uncertainty that don't actually co-occur), while their
elliptical sets -- which capture that some parameter errors are correlated
and so some directions of joint error are far more plausible than others --
beat both the non-robust and the symmetric-robust baselines.

**Everything below this point -- the linear-cost-in-theta observation, the
Cauchy-Schwarz / ellipsoid-support-function worst-case formula, and the
risk_aversion search for a robust-optimal trajectory -- is this project's
own derivation**, built for the setting where robust optimization actually
*is* tractable in closed form: Almgren-Chriss's linear temporary/permanent
impact and quadratic expected cost. It is "inspired by" the source paper's
ellipsoidal-uncertainty-set concept, adapted because the source paper's own
result does not transfer to this setting. This mirrors how the previous project's
(AlphaFactorMining's) modules related to their source papers when a direct
transcription wasn't possible or wasn't the right scope (its
``docs/papers.md`` "This project's simplification" notes, e.g. for
AlphaSAGE/AlphaQCM/FactorGCL) -- stating plainly what's reused, what's
original, and why.

Setup
-----
The temporary/permanent impact coefficients ``eta``, ``gamma`` (see
``almgren_chriss.py``'s :class:`~alphacube.execution.almgren_chriss.AlmgrenChrissParams`)
are calibrated, not known exactly. Write ``theta = (eta, gamma)``, with a
point estimate ``theta_hat = (eta_hat, gamma_hat)`` (e.g. from a historical
impact regression) and an ellipsoidal uncertainty set around it::

    U = { theta : (theta - theta_hat)^T Sigma_inv (theta - theta_hat) <= rho^2 }

``Sigma`` (stored on :class:`EllipsoidalUncertaintySet` as ``sigma``) is a
2x2 positive-definite matrix giving the uncertainty's shape/correlation --
e.g. the covariance of the ``(eta_hat, gamma_hat)`` estimator, so that if a
single regression tends to over- or under-estimate both coefficients
together (a very plausible failure mode: both impact coefficients come out
of the same fit against the same noisy impact data), ``Sigma`` has positive
off-diagonal covariance and the ellipsoid tilts along that diagonal instead
of being a symmetric ball. ``rho`` is the uncertainty radius; see
:meth:`EllipsoidalUncertaintySet.from_confidence_level` to set it from a
desired Gaussian coverage probability via a chi-squared quantile.

Key derivation: E[cost] is linear in theta for a fixed trajectory
--------------------------------------------------------------------
From ``almgren_chriss.py`` (module docstring + ``_effective_temporary_impact``,
verified by reading the source directly, not assumed): for a trajectory with
total size ``X = holdings[0]``, per-interval trades ``n_1..n_N``, and step
length ``tau``,

    E[cost] = 0.5*gamma*X^2 + eta_tilde * sum_j(n_j^2) / tau
    eta_tilde = eta - gamma*tau/2

Substituting ``eta_tilde`` and collecting terms in ``eta`` and ``gamma``
(``X``, the ``n_j``, and ``tau`` are all fixed once the *trajectory* --
i.e. the schedule of trades -- is fixed, so this is just regrouping a sum,
not an approximation):

    E[cost](eta, gamma) = eta * (S2/tau) + gamma * 0.5*(X^2 - S2)
                         = c_eta * eta   + c_gamma * gamma
                         = c^T theta,          where S2 = sum_j n_j^2

i.e. for a FIXED trajectory, ``E[cost]`` is an exactly linear (homogeneous
degree-1, no constant term) function of ``theta = (eta, gamma)``, with
trajectory-dependent coefficients ``c = (c_eta, c_gamma)`` and no
approximation involved -- this is just the existing closed-form formula,
regrouped. See ``_cost_linear_coefficients`` below.

Worst case over an ellipsoid: Cauchy-Schwarz / support function
-------------------------------------------------------------------
Because ``E[cost](theta) = c^T theta`` is linear in ``theta`` for a fixed
trajectory, its worst case over the ellipsoid ``U`` has a closed form: this
is the classical "support function of an ellipsoid" result, derived here
explicitly (not just asserted) via the ellipsoid's parametrization.

Any ``theta`` in ``U`` can be written ``theta = theta_hat + rho * Sigma^(1/2)
* u`` for some vector ``u`` with ``||u|| <= 1`` (substitute into the
ellipsoid's defining inequality: ``(theta-theta_hat)^T Sigma_inv
(theta-theta_hat) = rho^2 * u^T Sigma^(1/2) Sigma_inv Sigma^(1/2) u = rho^2 *
u^T u = rho^2 * ||u||^2 <= rho^2`` exactly when ``||u|| <= 1``, using
``Sigma^(1/2) Sigma_inv Sigma^(1/2) = I`` for symmetric PD ``Sigma``). Then

    c^T theta = c^T theta_hat + rho * c^T Sigma^(1/2) u
              = c^T theta_hat + rho * (Sigma^(1/2) c)^T u

which, as a linear function of ``u`` over the unit ball ``||u|| <= 1``, is
maximized by Cauchy-Schwarz at ``u = Sigma^(1/2)c / ||Sigma^(1/2)c||`` (the
unit vector aligned with ``Sigma^(1/2)c``), giving max value
``rho * ||Sigma^(1/2) c||``. And ``||Sigma^(1/2)c||^2 = c^T Sigma^(1/2)
Sigma^(1/2) c = c^T Sigma c``, so:

    max_{theta in U} c^T theta = c^T theta_hat + rho * sqrt(c^T Sigma c)

No numerical optimization is needed for this step -- it's a closed-form
expression in ``c``, ``theta_hat``, ``Sigma``, ``rho``. This is exactly
:func:`worst_case_cost` below (with ``c`` computed by
``_cost_linear_coefficients`` from an already-solved
:class:`~alphacube.execution.almgren_chriss.ExecutionTrajectory`).

Finding the robust-optimal trajectory: two tiers
-----------------------------------------------------
1. **Evaluation tier** (cheap, always available): take the ordinary
   Almgren-Chriss trajectory computed at the nominal point estimate
   ``theta_hat`` (via the existing, unmodified
   ``almgren_chriss.almgren_chriss_trajectory``) for some ``risk_aversion``,
   and evaluate its worst-case cost over ``U`` via the closed form above.
   This tells you how badly a *nominally* chosen trajectory could do, but
   doesn't yet search for a better one.

2. **Robust-optimal tier** (:func:`robust_almgren_chriss_trajectory`):
   search over ``risk_aversion`` (a 1-D scan via
   ``scipy.optimize.minimize_scalar``) for the value whose nominal-AC
   trajectory (built at ``theta_hat``, exactly as in tier 1) has the
   SMALLEST worst-case cost. This reuses ``risk_aversion`` as a convenient
   one-parameter family that already sweeps from TWAP (uniform) through
   maximally front-loaded trajectories, and asks the closed-form worst-case
   formula which point in that family is most robust -- rather than trading
   off E[cost] against Var[cost] as the ordinary AC risk_aversion sweep
   does. This is a genuinely **different** optimization from the plain AC
   ``efficient_frontier``: it minimizes worst-case E[cost] over an
   uncertainty set, not a mean-variance objective under a known ``sigma``.

   Scope limitation, stated honestly: this searches within the *family* of
   trajectories that are nominal-AC-optimal for some risk_aversion (i.e. a
   1-parameter family of hyperbolic-sine schedules), not over *all* possible
   trajectories subject only to ``x_0=X, x_N=0``. A fully general robust
   trajectory optimization (minimize worst-case cost over the ellipsoid,
   jointly over all ``N-1`` free intermediate holdings) is a well-posed
   convex problem in principle -- the inner worst-case is convex in the
   trajectory (a pointwise max of the linear-in-theta functions
   ``c(x)^T theta`` over ``theta in U`` is convex in ``x`` since ``c`` is
   linear in the trajectory and the sup of affine functions is convex) --
   but is out of scope here. Restricting to the risk_aversion-parameterized
   family keeps the search 1-D, always reuses the existing, tested
   ``almgren_chriss_trajectory`` machinery unmodified, and in practice
   recovers a meaningful, verifiably-better-than-nominal robust trajectory
   (see ``tests/test_robust_almgren_chriss.py`` and
   :func:`compare_robust_vs_nominal`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

from alphacube.execution.almgren_chriss import (
    AlmgrenChrissParams,
    ExecutionTrajectory,
    almgren_chriss_trajectory,
)

# Default search window for the risk_aversion knob used as the
# robust-optimal trajectory's 1-D parameterization (see module docstring,
# "robust-optimal tier"). The lower end brackets the same near-zero range
# `tests/test_almgren_chriss.py` exercises for the TWAP limit; kept strictly
# positive (rather than 0) so the search stays in the smooth hyperbolic-sine
# branch instead of landing exactly on the TWAP branch's boundary. The upper
# end is set much higher than `almgren_chriss.py`'s own worked examples
# (which stay under 1e-3): whether front-loading meaningfully reduces
# worst-case cost depends on how large the ellipsoid's gamma-uncertainty
# component is relative to eta's, and shrinking that exposure enough to
# matter can require substantially more front-loading than the ordinary
# (non-robust) mean-variance tradeoff ever would -- see
# `tests/test_robust_almgren_chriss.py` for a worked example where the
# worst-case-minimizing risk_aversion lands around 2-3. Callers with a
# narrower plausible range should pass their own.
_DEFAULT_RISK_AVERSION_SEARCH_RANGE: Tuple[float, float] = (1e-9, 5.0)


@dataclass
class EllipsoidalUncertaintySet:
    """An ellipsoidal uncertainty set around point-estimated impact
    parameters ``theta_hat = (eta_hat, gamma_hat)``:

        U = { (eta, gamma) : (theta - theta_hat)^T sigma_inv (theta - theta_hat) <= rho^2 }

    Attributes:
        eta_hat: point estimate of the temporary-impact coefficient.
        gamma_hat: point estimate of the permanent-impact coefficient.
        sigma: 2x2 symmetric positive-definite matrix giving the
            uncertainty's shape/correlation (e.g. an estimated covariance
            matrix of the ``(eta_hat, gamma_hat)`` estimator). Note this
            plays the role of ``Sigma`` in the ellipsoid formula above, i.e.
            it is what ``sigma_inv`` in that formula is the inverse of --
            *not* an inverse itself.
        rho: uncertainty radius (>= 0). ``rho=0`` collapses ``U`` to the
            single point ``theta_hat`` (no uncertainty).
    """

    eta_hat: float
    gamma_hat: float
    sigma: np.ndarray
    rho: float

    def __post_init__(self) -> None:
        self.sigma = np.asarray(self.sigma, dtype=float)
        if self.sigma.shape != (2, 2):
            raise ValueError(f"sigma must be a 2x2 matrix, got shape {self.sigma.shape}")
        if not np.allclose(self.sigma, self.sigma.T, atol=1e-10):
            raise ValueError(f"sigma must be symmetric, got {self.sigma!r}")
        eigvals = np.linalg.eigvalsh(self.sigma)
        if np.any(eigvals <= 0):
            raise ValueError(
                f"sigma must be positive definite (eigenvalues must be > 0), got eigenvalues {eigvals!r}"
            )
        if self.rho < 0:
            raise ValueError(f"rho must be >= 0, got {self.rho}")
        if self.eta_hat < 0:
            raise ValueError(f"eta_hat must be >= 0, got {self.eta_hat}")
        if self.gamma_hat < 0:
            raise ValueError(f"gamma_hat must be >= 0, got {self.gamma_hat}")

    @property
    def theta_hat(self) -> np.ndarray:
        """(eta_hat, gamma_hat) as a length-2 array."""
        return np.array([self.eta_hat, self.gamma_hat])

    @classmethod
    def from_confidence_level(
        cls,
        eta_hat: float,
        gamma_hat: float,
        sigma: np.ndarray,
        confidence: float,
    ) -> "EllipsoidalUncertaintySet":
        """Build the uncertainty set with `rho` chosen so `U` is a
        `confidence`-level confidence region for `theta_hat`, under the
        standard assumption that the estimation error `theta_hat - theta` is
        approximately Gaussian with covariance `sigma`.

        For a 2-D Gaussian error with covariance `sigma`, the squared
        Mahalanobis distance `(theta - theta_hat)^T sigma_inv (theta -
        theta_hat)` is chi-squared distributed with 2 degrees of freedom, so
        the `confidence`-quantile of that distribution is the squared radius
        that contains the true `theta` with probability `confidence`:

            rho = sqrt(chi2.ppf(confidence, df=2))

        Args:
            confidence: desired coverage probability in (0, 1), e.g. 0.95.
        """
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"confidence must be in (0, 1), got {confidence}")
        rho = float(np.sqrt(chi2.ppf(confidence, df=2)))
        return cls(eta_hat=eta_hat, gamma_hat=gamma_hat, sigma=sigma, rho=rho)


@dataclass
class RobustExecutionTrajectory:
    """Result of :func:`robust_almgren_chriss_trajectory`."""

    trajectory: ExecutionTrajectory  # the winning AC trajectory (built at theta_hat)
    nominal_cost: float  # E[cost] at theta_hat, i.e. trajectory.expected_cost
    worst_case_cost: float  # max E[cost] over the uncertainty set
    chosen_risk_aversion: float  # the risk_aversion whose trajectory won the search
    uncertainty_set: EllipsoidalUncertaintySet


def _cost_linear_coefficients(trajectory: ExecutionTrajectory) -> np.ndarray:
    """The vector `c = (c_eta, c_gamma)` such that, for THIS fixed
    trajectory (fixed X and n_j's), `E[cost](eta, gamma) = c @ (eta, gamma)`
    exactly -- see the module docstring's "Key derivation" section for the
    algebra. Recomputed directly from `trajectory.holdings`/`.trades`/
    `.times` rather than trusting `trajectory.expected_cost` (which was
    computed at whatever (eta, gamma) the trajectory happened to be built
    with), so this works for evaluating the SAME trajectory (same n_j
    schedule) at ANY (eta, gamma), not just the one it was optimized for.
    """
    X = float(trajectory.holdings[0])
    tau = float(trajectory.times[1] - trajectory.times[0])
    trade_sq_sum = float(np.sum(trajectory.trades**2))
    c_eta = trade_sq_sum / tau
    c_gamma = 0.5 * (X**2 - trade_sq_sum)
    return np.array([c_eta, c_gamma])


def worst_case_cost(trajectory: ExecutionTrajectory, uncertainty_set: EllipsoidalUncertaintySet) -> float:
    """Closed-form worst-case E[cost] of an already-computed
    `ExecutionTrajectory` (i.e. a fixed trading schedule) over the
    `uncertainty_set`'s ellipsoid, via the Cauchy-Schwarz/support-function
    derivation in the module docstring:

        max_{theta in U} E[cost](theta) = c @ theta_hat + rho * sqrt(c @ Sigma @ c)

    where `c` are this trajectory's cost-linear-coefficients (see
    `_cost_linear_coefficients`). Note `c @ theta_hat` is itself the
    trajectory's expected cost evaluated AT theta_hat (equal to
    `trajectory.expected_cost` when the trajectory was built with
    `temporary_impact=eta_hat, permanent_impact=gamma_hat`).
    """
    c = _cost_linear_coefficients(trajectory)
    nominal_at_theta_hat = float(c @ uncertainty_set.theta_hat)
    # c @ Sigma @ c is a quadratic form of a PD matrix, so it is
    # mathematically >= 0; clip at 0 purely as a floating-point safety net.
    quad_form = float(c @ uncertainty_set.sigma @ c)
    spread = uncertainty_set.rho * np.sqrt(max(quad_form, 0.0))
    return nominal_at_theta_hat + spread


def robust_almgren_chriss_trajectory(
    total_shares: float,
    horizon: float,
    n_intervals: int,
    volatility: float,
    uncertainty_set: EllipsoidalUncertaintySet,
    risk_aversion_search_range: Tuple[float, float] = _DEFAULT_RISK_AVERSION_SEARCH_RANGE,
) -> RobustExecutionTrajectory:
    """Search over `risk_aversion` for the nominal-AC trajectory (built at
    `theta_hat` via the existing, unmodified `almgren_chriss_trajectory`)
    that minimizes worst-case E[cost] over `uncertainty_set` (see module
    docstring's "robust-optimal tier" -- this is an approximate/practical
    robust-optimal search over a 1-D family of trajectories, not a fully
    general optimization over all trajectories).

    Returns the winning trajectory plus its nominal cost (at theta_hat) and
    worst-case cost (over the ellipsoid) for comparison.
    """
    lo, hi = risk_aversion_search_range
    if lo <= 0 or hi <= lo:
        raise ValueError(
            f"risk_aversion_search_range must be (lo, hi) with 0 < lo < hi, got {risk_aversion_search_range}"
        )

    def _trajectory_for(risk_aversion: float) -> ExecutionTrajectory:
        params = AlmgrenChrissParams(
            total_shares=total_shares,
            horizon=horizon,
            n_intervals=n_intervals,
            volatility=volatility,
            temporary_impact=uncertainty_set.eta_hat,
            permanent_impact=uncertainty_set.gamma_hat,
            risk_aversion=risk_aversion,
        )
        return almgren_chriss_trajectory(params)

    def _objective(risk_aversion: float) -> float:
        return worst_case_cost(_trajectory_for(risk_aversion), uncertainty_set)

    result = minimize_scalar(_objective, bounds=(lo, hi), method="bounded")
    chosen_risk_aversion = float(result.x)

    trajectory = _trajectory_for(chosen_risk_aversion)
    nominal_cost = trajectory.expected_cost
    wc = worst_case_cost(trajectory, uncertainty_set)

    return RobustExecutionTrajectory(
        trajectory=trajectory,
        nominal_cost=nominal_cost,
        worst_case_cost=wc,
        chosen_risk_aversion=chosen_risk_aversion,
        uncertainty_set=uncertainty_set,
    )


def compare_robust_vs_nominal(
    total_shares: float,
    horizon: float,
    n_intervals: int,
    volatility: float,
    eta_hat: float,
    gamma_hat: float,
    sigma: np.ndarray,
    rho: float,
    nominal_risk_aversion: float = 1e-6,
    risk_aversion_search_range: Tuple[float, float] = _DEFAULT_RISK_AVERSION_SEARCH_RANGE,
) -> pd.DataFrame:
    """Compute both a plain nominal-optimal trajectory (ordinary
    `almgren_chriss_trajectory` at `theta_hat`, using `nominal_risk_aversion`
    -- a fixed, reasonable risk-aversion choice a practitioner might make
    while ignoring parameter uncertainty entirely) and the robust trajectory
    from `robust_almgren_chriss_trajectory`, and report each one's E[cost]
    evaluated BOTH at the nominal parameters `theta_hat` and at the
    worst case within the ellipsoidal uncertainty set.

    This is the table that demonstrates the value of robustness: the
    nominal trajectory typically looks fine (or even better) at `theta_hat`
    but degrades more at the worst case within `U`; the robust trajectory
    sacrifices a little nominal-case performance for a better worst-case
    guarantee. Mirrors the tradeoff-table style of
    `almgren_chriss.efficient_frontier`.
    """
    uncertainty_set = EllipsoidalUncertaintySet(eta_hat=eta_hat, gamma_hat=gamma_hat, sigma=sigma, rho=rho)

    nominal_params = AlmgrenChrissParams(
        total_shares=total_shares,
        horizon=horizon,
        n_intervals=n_intervals,
        volatility=volatility,
        temporary_impact=eta_hat,
        permanent_impact=gamma_hat,
        risk_aversion=nominal_risk_aversion,
    )
    nominal_trajectory = almgren_chriss_trajectory(nominal_params)
    nominal_worst_case = worst_case_cost(nominal_trajectory, uncertainty_set)

    robust = robust_almgren_chriss_trajectory(
        total_shares=total_shares,
        horizon=horizon,
        n_intervals=n_intervals,
        volatility=volatility,
        uncertainty_set=uncertainty_set,
        risk_aversion_search_range=risk_aversion_search_range,
    )

    rows = [
        {
            "strategy": "nominal",
            "risk_aversion": nominal_risk_aversion,
            "cost_at_theta_hat": nominal_trajectory.expected_cost,
            "cost_at_worst_case": nominal_worst_case,
        },
        {
            "strategy": "robust",
            "risk_aversion": robust.chosen_risk_aversion,
            "cost_at_theta_hat": robust.nominal_cost,
            "cost_at_worst_case": robust.worst_case_cost,
        },
    ]
    return pd.DataFrame(rows, columns=["strategy", "risk_aversion", "cost_at_theta_hat", "cost_at_worst_case"])
