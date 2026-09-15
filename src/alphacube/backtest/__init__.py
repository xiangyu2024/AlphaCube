"""Sequential, multi-day backtest engine.

Chains ``portfolio/construct.py`` (signal -> weights) ->
``portfolio/optimize_*.py`` (weights -> executable share orders, under
turnover/lot/position-limit constraints) -> a dollar transaction-cost model
(``costs.py``) -> an optional market-impact-cost bridge to
``execution/almgren_chriss.py`` (``impact.py``) into an actual day-over-day
loop with carried-forward state (shares, equity) and a real equity curve
(``engine.py``), plus the performance/cost-drag reporting to go with it
(``metrics.py``).
"""

from __future__ import annotations

from .costs import TransactionCostModel
from .engine import BacktestEngine, BacktestResult, DailyRecord
from .impact import SimpleImpactModel
from .metrics import cost_drag_stats, performance_stats

__all__ = [
    "TransactionCostModel",
    "SimpleImpactModel",
    "BacktestEngine",
    "BacktestResult",
    "DailyRecord",
    "performance_stats",
    "cost_drag_stats",
]
