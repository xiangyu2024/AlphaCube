"""China A-share turnover-budgeted optimizer.

A-share-specific mechanics (see :mod:`optimize_base` for the shared
algorithm, implemented once and reused here unchanged):

- ``lot_size=100``: A-shares settle in whole "lots" of 100 shares -- you
  cannot buy/sell 37 shares, only multiples of 100.
- ``allow_short=False`` by default: retail/most strategies here are
  long-only or long-biased since shorting is restricted/expensive. This is
  a *default*, not hard-coded -- margin/quant-fund accounts that can
  actually short should pass ``allow_short=True``.
- T+1 settlement (a share bought today can't be sold until the next
  trading day) is NOT modeled inside this optimizer -- it produces a target
  rebalance from whatever price series it's handed, and the "today's close
  vs. tomorrow's execution price" timing question belongs to the caller
  (the daily backtest loop), per the contract documented in
  :meth:`~alphacube.portfolio.optimize_base.TurnoverBudgetOptimizer.rebalance`.
"""

from __future__ import annotations

from .optimize_base import TurnoverBudgetOptimizer


class AshareLotOptimizer(TurnoverBudgetOptimizer):
    """China A-share optimizer: 100-share lots, long-only by default.

    Defaults (all overridable via constructor kwargs):

    - ``turnover_limit=0.25``: allow up to a quarter of equity to trade in
      dollar (well, RMB) terms per rebalance -- a reasonably active but not
      hyperactive cadence for a lot-constrained, mostly-long book.
    - ``max_single_name_weight=0.08``: kept inside the commonly-used
      0.05-0.10 single-name concentration band; picked as a middle-of-range,
      tunable default, not derived from any specific risk model.
    - ``dust_cleanup_enabled=True`` with a small ``0.0005`` (5 bps of
      equity) threshold and its own ``0.05`` (5%) budget: A-share lot
      rounding routinely strands small positions (e.g. a name decayed to a
      few hundred RMB after roundoff) that aren't worth continuing to hold
      but also aren't worth spending main turnover budget to trim.
    """

    market_name = "ashare"

    def __init__(
        self,
        lot_size: int = 100,
        allow_short: bool = False,
        turnover_limit: float = 0.25,
        max_single_name_weight: float = 0.08,
        dust_cleanup_enabled: bool = True,
        dust_cleanup_weight_threshold: float = 0.0005,
        dust_cleanup_turnover_limit: float = 0.05,
    ) -> None:
        super().__init__(
            lot_size=lot_size,
            allow_short=allow_short,
            turnover_limit=turnover_limit,
            max_single_name_weight=max_single_name_weight,
            dust_cleanup_enabled=dust_cleanup_enabled,
            dust_cleanup_weight_threshold=dust_cleanup_weight_threshold,
            dust_cleanup_turnover_limit=dust_cleanup_turnover_limit,
        )
