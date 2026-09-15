"""Execution layer: turns one day's parent trade order per symbol (from
portfolio optimization) into an intraday execution schedule/cost estimate,
via the classic Almgren-Chriss (2000) optimal execution model.
"""

from alphacube.execution.almgren_chriss import (
    AlmgrenChrissParams,
    ExecutionTrajectory,
    almgren_chriss_trajectory,
    annualized_vol_to_interval_vol,
    efficient_frontier,
    schedule_trade_for_symbol,
    schedule_trades_for_rebalance,
)

__all__ = [
    "AlmgrenChrissParams",
    "ExecutionTrajectory",
    "almgren_chriss_trajectory",
    "annualized_vol_to_interval_vol",
    "efficient_frontier",
    "schedule_trade_for_symbol",
    "schedule_trades_for_rebalance",
]
