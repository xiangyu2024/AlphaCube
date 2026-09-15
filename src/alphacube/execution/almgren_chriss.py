"""Almgren-Chriss (2000/2001) optimal execution model.

This module implements the classic closed-form Almgren-Chriss "optimal
execution of portfolio transactions" model: given a parent order of ``X``
shares to be liquidated (or acquired) over a horizon ``T``, split into ``N``
equal discrete intervals, find the trading trajectory that minimizes

    E[cost] + risk_aversion * Var[cost]

trading off expected market-impact cost against the price-uncertainty risk
of trading slowly. This sits downstream of portfolio optimization: it turns
one day's *parent* trade (shares to trade for one symbol on one rebalance
day) into an intraday execution *schedule* (how many shares to trade in
each of N sub-intervals of that day) plus a cost/risk estimate for that
schedule.

References
----------
Almgren, R. and Chriss, N. (2000/2001), "Optimal Execution of Portfolio
Transactions", Journal of Risk. The formulas below follow that paper's
Theorem 1 (expected cost / variance of a general trajectory) and Theorem 2
(closed-form optimal trajectory for the linear-impact, mean-variance
objective).

Units and sign conventions
---------------------------
- ``total_shares`` (``X``): shares to execute over the horizon. Sign is the
  caller's convention -- positive = sell/liquidate, negative = buy/acquire.
  All formulas below are even in the trajectory (they scale linearly with
  ``X``) so the sign just flows through to the trajectory and trade signs;
  ``expected_cost``/``cost_variance`` are magnitudes (>= 0) regardless of
  sign, since they are driven by ``X**2`` and ``n_j**2`` terms.
- ``horizon`` (``T``) and ``tau = T / N``: any consistent time unit (e.g.
  trading days, or minutes) -- caller's choice, just be consistent across
  ``horizon``, ``volatility``, ``temporary_impact``, and ``permanent_impact``.
- ``volatility`` (``sigma``): price volatility **per unit of tau** (i.e. the
  standard deviation of the price change over one interval of length
  ``tau``), NOT annualized. Use :func:`annualized_vol_to_interval_vol` to
  convert from an annualized figure.
- ``temporary_impact`` (``eta``) and ``permanent_impact`` (``gamma``): both
  expressed in price-per-share per unit of trading *rate* (shares/time), so
  that ``eta * (n_j / tau)`` and ``gamma * (n_j / tau)`` are price impacts
  per share, consistent with the horizon's time unit.

Derivation of the cost and variance formulas
---------------------------------------------
Let ``S_k`` be the unaffected mid-price after interval ``k``:

    S_k = S_{k-1} + sigma*sqrt(tau)*xi_k - gamma*n_k,   xi_k ~ iid N(0, 1)

(the ``-gamma*n_k`` term is the permanent impact of trading ``n_k`` shares
during interval ``k``, applied as of the end of that interval; this is the
discrete form of the paper's ``tau * g(n_k/tau)`` with linear ``g(v) =
gamma*v``). Trading during interval ``k`` executes against a temporarily
impacted price:

    S~_k = S_{k-1} - eta*(n_k/tau)

The implementation-shortfall cost relative to the paper value ``X*S_0`` is

    cost = X*S_0 - sum_{k=1}^{N} n_k * S~_k

Substituting ``S~_k`` and then ``S_{k-1}``, and using ``X = sum_k n_k`` plus
the telescoping identity ``sum_{k=i+1}^{N} n_k = x_i`` (because ``x_i`` is
exactly the shares remaining after trade ``i``, and ``x_N = 0``), the cost
splits cleanly into a deterministic part and a mean-zero stochastic part:

    cost = [ (1/2)*gamma*X^2 + eta_tilde * sum_{k=1}^{N} n_k^2/tau ]
           - sigma*sqrt(tau) * sum_{i=1}^{N-1} xi_i * x_i

where ``eta_tilde = eta - gamma*tau/2`` absorbs a cross term between the
permanent-impact and temporary-impact pieces that falls out of the algebra
(see ``_effective_temporary_impact`` below). Because the ``xi_i`` are iid
standard normal and independent of the (deterministic, given the
trajectory) ``x_i``:

    E[cost]   = (1/2)*gamma*X^2 + eta_tilde * sum_{k=1}^{N} n_k^2/tau
    Var[cost] = sigma^2 * tau * sum_{i=1}^{N-1} x_i^2
              = sigma^2 * sum_j (tau * x_j^2)   over the post-trade holdings
                x_1, ..., x_{N-1} (equivalently x_1..x_N, since x_N = 0
                contributes nothing) -- note x_0 = X is *not* included: the
                initial holding is riskless with respect to future price
                moves realized only from interval 1 onward.

This is exactly Theorem 1 of Almgren-Chriss (2000).

For the mean-variance objective ``E[cost] + lambda*Var[cost]`` with
``lambda >= 0``, Theorem 2 of the paper gives the minimizing trajectory in
closed form via the hyperbolic discretization:

    kappa_tilde^2 = lambda * sigma^2 / eta_tilde
    kappa = (1/tau) * arccosh(1 + tau^2 * kappa_tilde^2 / 2)
    x_j   = X * sinh(kappa*(T - j*tau)) / sinh(kappa*T)

As ``tau -> 0`` (or more precisely as ``tau^2*kappa_tilde^2 -> 0``),
``arccosh(1+u) ~ sqrt(2u)`` for small ``u``, so
``kappa -> (1/tau)*sqrt(tau^2*kappa_tilde^2) = kappa_tilde``, recovering the
continuous-time AC result ``kappa = sqrt(lambda*sigma^2/eta_tilde)``. This
module checks that limit explicitly (see ``_LINEAR_LIMIT_KAPPA_T`` and the
module-level test in ``tests/test_almgren_chriss.py``).

As ``lambda -> 0`` (pure cost minimization, no risk aversion), ``kappa -> 0``
and the hyperbolic trajectory degenerates to the linear/uniform (TWAP)
schedule ``x_j = X*(1 - j/N)``. Rather than rely on ``sinh``/``cosh``
numerics near ``kappa = 0`` (which is a removable but numerically delicate
``0/0``), this is implemented as an explicit branch below.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, replace
from typing import Dict, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Below this threshold on kappa*T, the sinh-trajectory ratio is numerically
# indistinguishable from (and derived analytically as the limit of) the
# linear/TWAP trajectory, so we take the analytic limit directly instead of
# evaluating sinh(~0)/sinh(~0).
_LINEAR_LIMIT_KAPPA_T = 1e-6

# Floor applied to eta_tilde if it would otherwise be <= 0 (see
# _effective_temporary_impact docstring) -- purely a numerical safety net,
# not something that should be hit with sane inputs.
_ETA_TILDE_FLOOR = 1e-12


@dataclass
class AlmgrenChrissParams:
    """Inputs to the Almgren-Chriss optimal execution problem.

    Attributes:
        total_shares: X, shares to execute over the horizon. Sign convention
            is the caller's choice, documented consistently: positive =
            sell/liquidate, negative = buy/acquire.
        horizon: T, total execution time, in whatever time unit the caller
            uses consistently across this dataclass (e.g. trading days,
            or minutes).
        n_intervals: N, number of equal discrete trading intervals.
        volatility: sigma, price volatility PER INTERVAL (per `tau =
            horizon/n_intervals`), not annualized. See
            `annualized_vol_to_interval_vol` to convert.
        temporary_impact: eta >= 0, temporary market-impact coefficient
            (price per share per unit trading rate).
        permanent_impact: gamma >= 0, permanent market-impact coefficient
            (price per share per unit trading rate).
        risk_aversion: lambda >= 0, risk-aversion parameter trading off
            E[cost] against Var[cost]. 0 = pure cost minimization (TWAP).
    """

    total_shares: float
    horizon: float
    n_intervals: int
    volatility: float
    temporary_impact: float
    permanent_impact: float
    risk_aversion: float = 0.0

    def __post_init__(self) -> None:
        if self.n_intervals <= 0 or int(self.n_intervals) != self.n_intervals:
            raise ValueError(f"n_intervals must be a positive integer, got {self.n_intervals}")
        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")
        if self.volatility < 0:
            raise ValueError(f"volatility must be >= 0, got {self.volatility}")
        if self.temporary_impact < 0:
            raise ValueError(f"temporary_impact must be >= 0, got {self.temporary_impact}")
        if self.permanent_impact < 0:
            raise ValueError(f"permanent_impact must be >= 0, got {self.permanent_impact}")
        if self.risk_aversion < 0:
            raise ValueError(f"risk_aversion must be >= 0, got {self.risk_aversion}")

    @property
    def tau(self) -> float:
        """Length of one discrete trading interval, T/N."""
        return self.horizon / self.n_intervals


def annualized_vol_to_interval_vol(annual_vol: float, trading_days_per_year: float, tau_in_days: float) -> float:
    """Convert an annualized volatility to per-tau (per-interval) volatility.

    sigma_per_tau = annual_vol * sqrt(tau_in_days / trading_days_per_year)

    Args:
        annual_vol: annualized price volatility (e.g. 0.30 for 30%/year).
        trading_days_per_year: typically ~252.
        tau_in_days: length of one execution interval, in trading days
            (e.g. if horizon is in minutes, tau_in_days = tau_minutes /
            minutes_per_trading_day).
    """
    return annual_vol * np.sqrt(tau_in_days / trading_days_per_year)


@dataclass
class ExecutionTrajectory:
    """Result of solving the Almgren-Chriss trajectory optimization."""

    holdings: np.ndarray  # x_0..x_N, length N+1 (shares remaining to trade)
    trades: np.ndarray  # n_1..n_N, length N (shares traded during interval j)
    times: np.ndarray  # 0, tau, 2*tau, ..., T, length N+1
    expected_cost: float
    cost_variance: float
    cost_std: float
    kappa: float
    is_linear_limit: bool  # True if risk_aversion ~ 0 and the TWAP branch was used


def _effective_temporary_impact(eta: float, gamma: float, tau: float) -> float:
    """eta_tilde = eta * (1 - gamma*tau/(2*eta)), the AC-paper "effective"
    temporary-impact coefficient after netting out a permanent-impact cross
    term that appears in the discrete-time cost derivation (see module
    docstring). Algebraically this equals ``eta - gamma*tau/2``; we compute
    it in that form directly so eta=0 doesn't trigger a division, while it
    remains exactly the quantity the paper calls eta~.
    """
    eta_tilde = eta - gamma * tau / 2.0
    if eta_tilde <= 0:
        warnings.warn(
            f"eta_tilde = eta - gamma*tau/2 = {eta_tilde!r} <= 0 (tau is large relative to "
            "eta/gamma for this discretization); the model is not well-posed in this regime. "
            f"Clamping eta_tilde to {_ETA_TILDE_FLOOR!r}.",
            stacklevel=3,
        )
        eta_tilde = _ETA_TILDE_FLOOR
    return eta_tilde


def _linear_trajectory(X: float, N: int) -> np.ndarray:
    """Analytic TWAP limit x_j = X*(1 - j/N), used both when risk_aversion
    is exactly 0 and when kappa*T is small enough that the hyperbolic
    trajectory is numerically indistinguishable from this limit."""
    j = np.arange(N + 1)
    holdings = X * (1.0 - j / N)
    holdings[0] = X
    holdings[-1] = 0.0
    return holdings


def almgren_chriss_trajectory(params: AlmgrenChrissParams) -> ExecutionTrajectory:
    """Solve for the mean-variance-optimal Almgren-Chriss execution trajectory.

    Minimizes E[cost] + risk_aversion * Var[cost] over trajectories x_0=X,
    x_N=0. See module docstring for the full derivation.
    """
    X = params.total_shares
    T = params.horizon
    N = int(params.n_intervals)
    sigma = params.volatility
    eta = params.temporary_impact
    gamma = params.permanent_impact
    lam = params.risk_aversion
    tau = params.tau

    times = np.linspace(0.0, T, N + 1)

    if X == 0:
        holdings = np.zeros(N + 1)
        trades = np.zeros(N)
        return ExecutionTrajectory(
            holdings=holdings,
            trades=trades,
            times=times,
            expected_cost=0.0,
            cost_variance=0.0,
            cost_std=0.0,
            kappa=0.0,
            is_linear_limit=True,
        )

    eta_tilde = _effective_temporary_impact(eta, gamma, tau)

    kappa = 0.0
    is_linear_limit = True
    if lam > 0 and sigma > 0:
        kappa_tilde_sq = lam * sigma**2 / eta_tilde
        arg = 1.0 + (tau**2) * kappa_tilde_sq / 2.0
        if arg < 1.0:
            # Only possible via floating-point noise since kappa_tilde_sq >= 0
            # guarantees arg >= 1 analytically.
            warnings.warn(
                f"arccosh argument {arg!r} < 1 due to floating-point error; clamping to 1.0.",
                stacklevel=2,
            )
            arg = 1.0
        kappa = np.arccosh(arg) / tau
        if kappa * T > _LINEAR_LIMIT_KAPPA_T:
            is_linear_limit = False

    if is_linear_limit:
        holdings = _linear_trajectory(X, N)
    else:
        j = np.arange(N + 1)
        holdings = X * np.sinh(kappa * (T - j * tau)) / np.sinh(kappa * T)
        holdings[0] = X
        holdings[-1] = 0.0

    trades = -np.diff(holdings)  # n_j = x_{j-1} - x_j

    expected_cost = 0.5 * gamma * X**2 + eta_tilde * float(np.sum(trades**2)) / tau
    # Var[cost] = sigma^2 * tau * sum_{j=1}^{N} x_j^2 -- post-trade holdings
    # only (x_0 = X excluded; see derivation in the module docstring).
    cost_variance = sigma**2 * tau * float(np.sum(holdings[1:] ** 2))
    cost_std = float(np.sqrt(max(cost_variance, 0.0)))

    return ExecutionTrajectory(
        holdings=holdings,
        trades=trades,
        times=times,
        expected_cost=expected_cost,
        cost_variance=cost_variance,
        cost_std=cost_std,
        kappa=kappa,
        is_linear_limit=is_linear_limit,
    )


def efficient_frontier(params_template: AlmgrenChrissParams, risk_aversions: Sequence[float]) -> pd.DataFrame:
    """Sweep risk_aversion values, holding everything else in
    `params_template` fixed, and return the resulting (risk_aversion,
    expected_cost, cost_variance, cost_std) efficient frontier of execution
    strategies -- the classic AC tool for picking a risk-aversion lambda by
    seeing the cost/risk it buys you.
    """
    rows = []
    for lam in risk_aversions:
        p = replace(params_template, risk_aversion=lam)
        traj = almgren_chriss_trajectory(p)
        rows.append(
            {
                "risk_aversion": lam,
                "expected_cost": traj.expected_cost,
                "cost_variance": traj.cost_variance,
                "cost_std": traj.cost_std,
            }
        )
    return pd.DataFrame(rows, columns=["risk_aversion", "expected_cost", "cost_variance", "cost_std"])


def schedule_trade_for_symbol(
    trade_shares: float,
    horizon: float,
    n_intervals: int,
    volatility: float,
    temporary_impact: float,
    permanent_impact: float,
    risk_aversion: float = 0.0,
) -> ExecutionTrajectory:
    """Thin convenience wrapper: turn a single symbol's one-day parent trade
    (the kind of number that comes out of a portfolio optimizer's
    `RebalancePlan.trades` Series) plus that symbol's market-impact
    parameters into an intraday Almgren-Chriss execution schedule.
    """
    params = AlmgrenChrissParams(
        total_shares=trade_shares,
        horizon=horizon,
        n_intervals=n_intervals,
        volatility=volatility,
        temporary_impact=temporary_impact,
        permanent_impact=permanent_impact,
        risk_aversion=risk_aversion,
    )
    return almgren_chriss_trajectory(params)


def schedule_trades_for_rebalance(
    trades: pd.Series,
    market_params: pd.DataFrame,
    horizon: float,
    n_intervals: int,
    risk_aversion: float = 0.0,
) -> Dict[str, "ExecutionTrajectory"]:
    """The realistic entry point: given one day's rebalance trade list
    (symbol -> signed shares to trade) and a per-symbol market-impact
    parameter table (`market_params`, indexed by symbol, with columns
    `volatility`, `temporary_impact`, `permanent_impact`), produce an
    intraday execution schedule for every symbol with a nonzero trade.

    Symbols with a nonzero trade but missing from `market_params` are
    skipped (not erroring) and logged at WARNING level with the full list
    of skipped symbols.
    """
    result: Dict[str, ExecutionTrajectory] = {}
    skipped = []
    for symbol, trade_shares in trades.items():
        if trade_shares == 0:
            continue
        if symbol not in market_params.index:
            skipped.append(symbol)
            continue
        row = market_params.loc[symbol]
        result[symbol] = schedule_trade_for_symbol(
            trade_shares=float(trade_shares),
            horizon=horizon,
            n_intervals=n_intervals,
            volatility=float(row["volatility"]),
            temporary_impact=float(row["temporary_impact"]),
            permanent_impact=float(row["permanent_impact"]),
            risk_aversion=risk_aversion,
        )

    if skipped:
        logger.warning(
            "schedule_trades_for_rebalance: skipping %d symbol(s) missing from market_params: %s",
            len(skipped),
            skipped,
        )

    return result
