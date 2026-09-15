"""Dollar transaction-cost (commission) model for the backtest engine.

Nothing in the existing ``portfolio``/``execution`` layers computes an
explicit dollar commission: :class:`~alphacube.portfolio.optimize_base.RebalancePlan`
tracks turnover as a *fraction of equity* (``turnover_used``), and
``execution/almgren_chriss.py`` models market-impact cost, not brokerage/
exchange commission. This module fills that specific gap with a simple,
realistic percentage-commission-plus-minimum-fee (and optional sell-side
tax, e.g. China A-share stamp duty) model, operating on the same
``pd.Series`` (indexed by symbol) convention as
:attr:`~alphacube.portfolio.optimize_base.RebalancePlan.trades`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TransactionCostModel:
    """Percentage commission with a per-symbol minimum fee, plus an
    optional sell-only tax.

    Attributes:
        cost_bps: percentage commission, in basis points of traded notional
            (e.g. ``2.0`` == 2 bps == 0.02%), charged on both buys and sells.
        min_commission: minimum dollar commission charged for any symbol
            that trades a nonzero amount at a valid price, even if
            ``notional * cost_bps/1e4`` would be smaller (the common
            "$X minimum ticket" broker convention).
        sell_tax_bps: additional tax, in basis points, charged only on
            sell-direction notional (e.g. China A-share stamp duty, which
            is sell-only). ``0.0`` by default (most markets/accounts don't
            have one).
    """

    cost_bps: float = 2.0
    min_commission: float = 5.0
    sell_tax_bps: float = 0.0

    def __post_init__(self) -> None:
        if self.cost_bps < 0.0:
            raise ValueError("cost_bps must be >= 0")
        if self.min_commission < 0.0:
            raise ValueError("min_commission must be >= 0")
        if self.sell_tax_bps < 0.0:
            raise ValueError("sell_tax_bps must be >= 0")

    def compute(self, trades: pd.Series, prices: pd.Series) -> dict:
        """Compute this rebalance's dollar transaction cost.

        Args:
            trades: signed shares traded per symbol (same convention as
                :attr:`RebalancePlan.trades` -- positive = buy, negative =
                sell), indexed by symbol.
            prices: price per symbol used to value the trades (same
                convention as the price series passed to ``rebalance``).

        Returns:
            ``{'commission': float, 'sell_tax': float,
            'transaction_cost': float, 'charged_order_count': int}``.
            A symbol is only "charged" (contributes to ``commission`` /
            ``charged_order_count``) if it has a nonzero, non-NaN trade AND
            a valid (finite, positive) price -- a trade against a missing/
            bad price is not something that could actually execute, so it's
            not charged a commission either.
        """
        trades = trades.astype(np.float64)
        prices = prices.reindex(trades.index).astype(np.float64)

        valid_price = prices.notna() & (prices > 0.0)
        nonzero_trade = trades.notna() & (trades != 0.0)
        charged_mask = valid_price & nonzero_trade

        if charged_mask.any():
            notional = trades[charged_mask].abs() * prices[charged_mask]
            commission = notional * (self.cost_bps / 1e4)
            commission = commission.clip(lower=self.min_commission)
            total_commission = float(commission.sum())
        else:
            total_commission = 0.0

        sell_mask = charged_mask & (trades < 0.0)
        if sell_mask.any():
            sell_notional = float((trades[sell_mask].abs() * prices[sell_mask]).sum())
        else:
            sell_notional = 0.0
        sell_tax = sell_notional * (self.sell_tax_bps / 1e4)

        transaction_cost = total_commission + sell_tax

        return {
            "commission": total_commission,
            "sell_tax": float(sell_tax),
            "transaction_cost": float(transaction_cost),
            "charged_order_count": int(charged_mask.sum()),
        }
