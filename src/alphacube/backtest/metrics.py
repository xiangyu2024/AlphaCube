"""Performance and cost-drag metrics for a real (sequential) equity curve.

`alphacube/metrics.py`'s `backtest_metrics` already has a Sharpe/max-drawdown/
cumulative-return pattern, but it's computed over a *decile long-short
diagnostic* (no state carried forward, no costs) -- a fundamentally
different input shape from a real day-by-day equity curve. This module is
the real-equity-curve counterpart: it deliberately matches that module's
formula conventions (same annualization convention: `(1+mean_r)**periods -
1` for annualized return, `mean_r/std_r*sqrt(periods)` for Sharpe, running-
max drawdown on `cumprod(1+r)`) for consistency across the codebase, without
importing/reusing it directly since the inputs aren't interchangeable.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, pd.Series]


def performance_stats(daily_returns: ArrayLike, periods_per_year: int = 252) -> dict:
    """Standard return/risk stats from a plain array/Series of per-day net
    returns (``net_pnl / equity_before`` for each day).

    Returns:
        ``{'ann_return', 'ann_vol', 'sharpe', 'max_drawdown', 'win_rate',
        'total_return', 'calmar_ratio'}``. All ``nan`` if ``daily_returns``
        has no valid (non-NaN) observations.
    """
    r = np.asarray(daily_returns, dtype=np.float64)
    r = r[~np.isnan(r)]

    if r.size == 0:
        return {
            "ann_return": float("nan"),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "win_rate": float("nan"),
            "total_return": float("nan"),
            "calmar_ratio": float("nan"),
        }

    mean_r = float(np.mean(r))
    std_r = float(np.std(r))

    ann_return = float((1.0 + mean_r) ** periods_per_year - 1.0)
    ann_vol = float(std_r * np.sqrt(periods_per_year))
    sharpe = float((mean_r / std_r) * np.sqrt(periods_per_year)) if std_r > 0 else float("nan")

    cumulative = np.cumprod(1.0 + r)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative / running_max - 1.0
    max_drawdown = float(np.min(drawdown))

    win_rate = float(np.mean(r > 0))
    total_return = float(cumulative[-1] - 1.0)
    calmar_ratio = float(ann_return / abs(max_drawdown)) if max_drawdown != 0.0 else float("nan")

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "total_return": total_return,
        "calmar_ratio": calmar_ratio,
    }


def cost_drag_stats(daily: pd.DataFrame, periods_per_year: int = 252) -> dict:
    """"How much did costs/impact actually eat into returns" report.

    Args:
        daily: the ``BacktestResult.daily`` DataFrame (one row per trading
            day, columns include ``gross_pnl``, ``net_pnl``,
            ``transaction_cost``, ``impact_cost``, ``equity_before``,
            ``turnover_used``).

    Returns:
        ``{'total_transaction_cost', 'total_impact_cost', 'total_cost',
        'gross_total_return', 'net_total_return', 'cost_drag_annualized',
        'avg_turnover'}``.
    """
    if daily is None or len(daily) == 0:
        return {
            "total_transaction_cost": 0.0,
            "total_impact_cost": 0.0,
            "total_cost": 0.0,
            "gross_total_return": float("nan"),
            "net_total_return": float("nan"),
            "cost_drag_annualized": float("nan"),
            "avg_turnover": float("nan"),
        }

    total_transaction_cost = float(daily["transaction_cost"].sum())
    total_impact_cost = float(daily["impact_cost"].sum())
    total_cost = total_transaction_cost + total_impact_cost

    gross_returns = (daily["gross_pnl"] / daily["equity_before"]).to_numpy(dtype=np.float64)
    net_returns = (daily["net_pnl"] / daily["equity_before"]).to_numpy(dtype=np.float64)

    gross_stats = performance_stats(gross_returns, periods_per_year=periods_per_year)
    net_stats = performance_stats(net_returns, periods_per_year=periods_per_year)

    cost_drag_annualized = gross_stats["ann_return"] - net_stats["ann_return"]

    return {
        "total_transaction_cost": total_transaction_cost,
        "total_impact_cost": total_impact_cost,
        "total_cost": total_cost,
        "gross_total_return": gross_stats["total_return"],
        "net_total_return": net_stats["total_return"],
        "cost_drag_annualized": float(cost_drag_annualized),
        "avg_turnover": float(daily["turnover_used"].mean()),
    }
