"""US equities turnover-budgeted optimizer.

US-specific mechanics (see :mod:`optimize_base` for the shared algorithm,
implemented once and reused here unchanged):

- ``lot_size=1``: shares trade continuously, so any integer share count is a
  fine common baseline. Fractional-share trading is increasingly available
  at many brokers, but ``lot_size`` here is an integer whole-share count by
  design (see :class:`~alphacube.portfolio.optimize_base.TurnoverBudgetOptimizer`),
  so the default is kept at a whole share (1); fractional-share execution,
  if a venue supports it, is a concern for the execution layer downstream,
  not this optimizer.
- ``allow_short=True`` by default: shorting is much more freely available
  in US equities than in A-shares, so this optimizer defaults to allowing
  it (still overridable per account/mandate).
"""

from __future__ import annotations

from .optimize_base import TurnoverBudgetOptimizer


class USEquityOptimizer(TurnoverBudgetOptimizer):
    """US equity optimizer: 1-share lots by default, short-able by default.

    Defaults (all overridable via constructor kwargs):

    - ``turnover_limit=0.5``: set higher than the A-share default (0.25).
      This is a judgment call, not a hard fact -- continuous (non-lot-
      constrained) share sizing plus freer shorting typically supports a
      more active/larger rebalance cadence than a lot-constrained,
      long-biased book, so the default budget headroom is larger. Tune per
      strategy turnover profile.
    - ``max_single_name_weight=0.10``: single-name cap, slightly looser than
      the A-share default since concentration risk tends to be managed
      differently (and shorting lets a single name's *net* exposure stay
      controlled even at a larger gross weight).
    - ``dust_cleanup_enabled=True`` with a smaller ``0.0002`` (2 bps of
      equity) threshold than the A-share default: at ``lot_size=1`` there's
      no lot-rounding forcing awkward small residuals the way 100-share
      lots do, so dust is less of an issue -- but cheap cleanup is still
      worth doing since it's essentially free and small stragglers still
      accumulate from decay/turnover-budget partial fills. Its own small
      ``0.02`` (2%) budget is kept separate from the main turnover budget.
    """

    market_name = "us"

    def __init__(
        self,
        lot_size: int = 1,
        allow_short: bool = True,
        turnover_limit: float = 0.5,
        max_single_name_weight: float = 0.10,
        dust_cleanup_enabled: bool = True,
        dust_cleanup_weight_threshold: float = 0.0002,
        dust_cleanup_turnover_limit: float = 0.02,
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
