"""Optional market-impact-cost bridge from the backtest engine to
``execution/almgren_chriss.py``.

Running the FULL Almgren-Chriss intraday optimal trajectory (potentially
many intervals) for every nonzero trade on every single day of a
multi-year backtest would be far more expensive than the rest of the
engine's per-day work, and the backtest loop only needs a same-day cost
NUMBER (not an intraday execution schedule) to fold into that day's P&L.
:class:`SimpleImpactModel` is the adapter: it reuses
``execution.almgren_chriss.schedule_trades_for_rebalance`` directly (rather
than reimplementing anything) with a small, configurable ``n_intervals``
(default ``1`` -- "execute the whole trade in one shot this day") and sums
the resulting ``expected_cost`` across every symbol that traded, using this
model's ``market_params`` table. Symbols missing from ``market_params`` are
skipped with a zero contribution, exactly matching
``schedule_trades_for_rebalance``'s own (log-and-skip, not error) behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from ..execution.almgren_chriss import schedule_trades_for_rebalance


@dataclass
class SimpleImpactModel:
    """Per-day market-impact cost estimate, bridging to Almgren-Chriss.

    Attributes:
        market_params: per-symbol table indexed by symbol with columns
            ``volatility``, ``temporary_impact``, ``permanent_impact`` --
            the same shape ``execution.almgren_chriss.schedule_trades_for_rebalance``
            already expects. Symbols absent from this table contribute
            ``0.0`` impact cost (skipped, not an error) when they trade.
        horizon: execution horizon handed to the underlying Almgren-Chriss
            solve for each day's rebalance (same time unit as
            ``volatility``/impact coefficients in ``market_params``; default
            ``1.0`` == "one day").
        n_intervals: number of intraday sub-intervals to solve for (kept
            small/cheap by design -- e.g. ``1`` = "one shot", or a small
            configurable N like ``5``). This is a coarse single-day cost
            estimate, not a real execution schedule to be dispatched.
        risk_aversion: Almgren-Chriss risk-aversion lambda used for this
            estimate; ``0.0`` (pure cost minimization / TWAP-equivalent for
            ``n_intervals=1``, where there's only one interval anyway) by
            default.
    """

    market_params: pd.DataFrame
    horizon: float = 1.0
    n_intervals: int = 1
    risk_aversion: float = 0.0

    def __post_init__(self) -> None:
        if self.horizon <= 0.0:
            raise ValueError("horizon must be > 0")
        if self.n_intervals < 1:
            raise ValueError("n_intervals must be >= 1")

    def estimate(self, trades: pd.Series, prices: Optional[pd.Series] = None) -> float:
        """Sum of expected Almgren-Chriss impact cost across this day's
        nonzero trades. ``prices`` is accepted for interface symmetry with
        :meth:`~alphacube.backtest.costs.TransactionCostModel.compute` but
        is not used here -- impact cost is driven entirely by
        ``market_params`` (volatility/impact coefficients), not by price
        level, per the Almgren-Chriss cost formula.
        """
        nonzero_trades = trades[trades.fillna(0.0) != 0.0]
        if nonzero_trades.empty:
            return 0.0

        schedules = schedule_trades_for_rebalance(
            nonzero_trades,
            self.market_params,
            horizon=self.horizon,
            n_intervals=self.n_intervals,
            risk_aversion=self.risk_aversion,
        )
        return float(sum(traj.expected_cost for traj in schedules.values()))
