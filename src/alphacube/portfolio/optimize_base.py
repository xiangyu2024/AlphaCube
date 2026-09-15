"""Shared turnover-budgeted rebalance-to-executable-orders algorithm.

Sits downstream of ``portfolio/construct.py`` (which produces a *target
weight* ``pd.Series`` indexed by instrument symbol, values roughly in
``[-1, 1]``, summing near some target gross/long exposure -- not necessarily
achievable exactly) and upstream of ``execution/`` (which is handed the
resulting whole-share order sizes and actually works them into the market).

This module implements the ONE algorithm both markets share
(:class:`TurnoverBudgetOptimizer`); ``optimize_ashare.py`` / ``optimize_us.py``
only set market-specific config defaults (lot size, short-ability, turnover
budget, ...) via subclassing -- they do not re-implement any of the fill
logic below.

**Timing contract.** ``rebalance`` is a pure function of the target weights,
previous share counts, a price series, and available equity -- it has no
notion of "today" vs "tomorrow" or T+1 settlement. Decision weights and
lot-rounding are computed strictly as-of the price series passed in; it is
the caller's responsibility (e.g. the daily backtest loop) to decide which
day's prices those are (typically: size today's rebalance off today's close,
even though it will actually execute at tomorrow's open/price under T+1
settlement). Do not assume inside this module which day a price belongs to.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_EPS = 1e-9


def align_inputs(
    target_weights: pd.Series,
    prev_shares: pd.Series,
    prices: pd.Series,
    tradable: Optional[pd.Series] = None,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Reindex the four rebalance inputs onto a common (union) symbol index.

    Index-mismatch bugs -- a symbol present in ``target_weights`` but not yet
    in ``prev_shares`` (a brand-new name), or present in ``prev_shares`` but
    dropped from today's ``target_weights`` (should be sold to zero), or
    missing a price -- are a classic source of silent wrong-answer trading
    bugs, so this alignment is centralized and unit-tested on its own.

    Fill conventions:

    - ``target_weights``: missing -> ``0.0`` (no signal -> no target position).
    - ``prev_shares``: missing -> ``0.0`` (no prior holding).
    - ``prices``: missing -> ``NaN`` (deliberately NOT filled -- a missing
      price makes a symbol untradable this rebalance, handled by the caller
      via the untradable-freeze step; silently filling it with 0 or a stale
      value would be worse than leaving it NaN).
    - ``tradable``: missing -> ``True`` when a ``tradable`` series is given
      but a symbol is absent from it; if ``tradable`` itself is ``None``,
      every symbol is treated as tradable (subject to the price check).
    """
    index = target_weights.index.union(prev_shares.index).union(prices.index)
    if tradable is not None:
        index = index.union(tradable.index)

    aligned_weights = target_weights.reindex(index, fill_value=0.0).astype(np.float64)
    aligned_shares = prev_shares.reindex(index, fill_value=0.0).astype(np.float64)
    aligned_prices = prices.reindex(index).astype(np.float64)

    if tradable is None:
        aligned_tradable = pd.Series(True, index=index)
    else:
        aligned_tradable = tradable.reindex(index, fill_value=True).astype(bool)

    return aligned_weights, aligned_shares, aligned_prices, aligned_tradable


@dataclass
class RebalancePlan:
    """Output of a single :meth:`TurnoverBudgetOptimizer.rebalance` call.

    All series are indexed by the same (aligned) symbol index as the inputs.
    """

    new_shares: pd.Series
    trades: pd.Series
    trade_value: pd.Series
    turnover_used: float
    turnover_budget: float
    achieved_weights: pd.Series
    diagnostics: dict = field(default_factory=dict)


class TurnoverBudgetOptimizer(ABC):
    """Turns a target weight vector into whole-lot executable share orders.

    Concrete market subclasses (:class:`~alphacube.portfolio.optimize_ashare.AshareLotOptimizer`,
    :class:`~alphacube.portfolio.optimize_us.USEquityOptimizer`) only differ
    in ``lot_size``/``allow_short``/budget defaults -- the fill algorithm
    itself lives here, once, and is shared.

    Algorithm (see :meth:`rebalance`):

    1. Freeze names that are untradable or missing a valid price.
    2. Convert target weights to whole lots (floor for long-only, round for
       long/short).
    3. Optional cheap "dust cleanup" pass, funded by its own small budget,
       run before the main fill.
    4. Main turnover-budgeted fill: take the full desired rebalance if it
       fits the turnover budget; otherwise scale every trade down
       proportionally, truncate to whole lots, and greedily hand out the
       leftover (lot-rounding) budget in signal-priority order.
    5. Trim any name that ends up over ``max_single_name_weight``.
    """

    def __init__(
        self,
        lot_size: int,
        allow_short: bool,
        turnover_limit: float,
        max_single_name_weight: float,
        dust_cleanup_enabled: bool = True,
        dust_cleanup_weight_threshold: float = 0.0005,
        dust_cleanup_turnover_limit: float = 0.02,
    ) -> None:
        if lot_size < 1:
            raise ValueError("lot_size must be >= 1")
        if not (0.0 < turnover_limit <= 1.0):
            raise ValueError("turnover_limit must be in (0, 1]")
        if not (0.0 < max_single_name_weight <= 1.0):
            raise ValueError("max_single_name_weight must be in (0, 1]")
        if dust_cleanup_weight_threshold < 0.0:
            raise ValueError("dust_cleanup_weight_threshold must be >= 0")
        if dust_cleanup_turnover_limit < 0.0:
            raise ValueError("dust_cleanup_turnover_limit must be >= 0")

        self.lot_size = int(lot_size)
        self.allow_short = bool(allow_short)
        self.turnover_limit = float(turnover_limit)
        self.max_single_name_weight = float(max_single_name_weight)
        self.dust_cleanup_enabled = bool(dust_cleanup_enabled)
        self.dust_cleanup_weight_threshold = float(dust_cleanup_weight_threshold)
        self.dust_cleanup_turnover_limit = float(dust_cleanup_turnover_limit)

    @property
    @abstractmethod
    def market_name(self) -> str:
        """Short identifier for the market this optimizer targets (e.g. ``"ashare"``, ``"us"``)."""

    # -- main entry point ------------------------------------------------

    def rebalance(
        self,
        target_weights: pd.Series,
        prev_shares: pd.Series,
        prices: pd.Series,
        equity: float,
        tradable: Optional[pd.Series] = None,
    ) -> RebalancePlan:
        """Compute this rebalance's executable share orders.

        Decision weights are computed as-of the ``prices`` series passed in
        -- the caller chooses which day's prices those are (see module
        docstring on the T+1 timing contract). ``equity`` is total account
        equity in the same currency as ``prices``.

        Args:
            target_weights: desired weight per symbol, roughly in [-1, 1].
            prev_shares: current share count per symbol before this rebalance.
            prices: price per symbol used to size this rebalance.
            equity: total account equity (same currency as ``prices``).
            tradable: optional per-symbol tradability flag; ``False`` (or a
                missing/non-positive/NaN price) freezes that symbol's shares
                for this rebalance.
        """
        if equity <= 0:
            raise ValueError("equity must be > 0")

        target_weights, prev_shares, prices, tradable = align_inputs(
            target_weights, prev_shares, prices, tradable
        )

        diagnostics: dict = {}

        # --- step 1: freeze untradable / bad-price names -----------------
        price_valid = prices.notna() & (prices > 0)
        frozen_mask = (~tradable) | (~price_valid)
        names_untradable = sorted(str(s) for s in target_weights.index[frozen_mask])
        diagnostics["names_untradable"] = names_untradable

        active_index = target_weights.index[~frozen_mask]

        # new_shares starts as a copy of the original (aligned) prev_shares;
        # frozen names are simply never touched again below.
        new_shares = prev_shares.astype(np.float64).copy()

        if len(active_index) == 0:
            trades = new_shares - prev_shares
            trade_value = trades * 0.0
            diagnostics.update(
                dust_cleaned_names=[],
                dust_turnover_used=0.0,
                capped_names=[],
                total_desired_turnover=0.0,
                scaled=False,
            )
            return RebalancePlan(
                new_shares=new_shares,
                trades=trades,
                trade_value=trade_value,
                turnover_used=0.0,
                turnover_budget=self.turnover_limit,
                achieved_weights=new_shares * prices / equity,
                diagnostics=diagnostics,
            )

        tw = target_weights.loc[active_index].copy()
        ps = prev_shares.loc[active_index].copy()
        px = prices.loc[active_index].copy()

        # --- step 2: weight -> raw shares -> whole lots -------------------
        if not self.allow_short:
            tw = tw.clip(lower=0.0)

        raw_shares = tw * equity / px
        raw_lots = raw_shares / self.lot_size
        if self.allow_short:
            # Round to nearest lot: can go either direction since a short
            # position on one name can net against a long elsewhere.
            target_lots = np.round(raw_lots)
        else:
            # Floor (never round up) so a long-only fill never slightly
            # overspends available capital on a name -- there's no offsetting
            # short position anywhere else to net the overshoot against.
            target_lots = np.floor(raw_lots)
        target_lots = pd.Series(target_lots, index=active_index, dtype=np.float64)

        # --- step 3: dust cleanup (own budget, runs before the main fill) -
        dust_cleaned: List[str] = []
        dust_turnover_dollars = 0.0
        if self.dust_cleanup_enabled and len(active_index) > 0:
            cur_val = (ps * px).abs()
            target_val = (target_lots * self.lot_size * px).abs()
            thresh = self.dust_cleanup_weight_threshold * equity
            dust_mask = (cur_val <= thresh) & (target_val <= thresh) & (ps != 0.0)
            candidates = list(cur_val.index[dust_mask])
            candidates.sort(key=lambda s: (cur_val[s], str(s)))

            dust_budget = self.dust_cleanup_turnover_limit * equity
            cum = 0.0
            for sym in candidates:
                trade_val = abs(0.0 - ps[sym]) * px[sym]
                if cum + trade_val > dust_budget + _EPS:
                    continue
                cum += trade_val
                dust_cleaned.append(str(sym))
                new_shares[sym] = 0.0

            dust_turnover_dollars = cum
            # Resolved: pulled out of the active set before the main step.
            if dust_cleaned:
                ps = ps.drop(index=dust_cleaned)
                px = px.drop(index=dust_cleaned)
                tw = tw.drop(index=dust_cleaned)
                target_lots = target_lots.drop(index=dust_cleaned)

        diagnostics["dust_cleaned_names"] = dust_cleaned
        diagnostics["dust_turnover_used"] = dust_turnover_dollars / equity

        main_index = target_lots.index

        # --- step 4: main turnover-budgeted fill --------------------------
        budget_dollars = self.turnover_limit * equity
        final_trade_shares = pd.Series(0.0, index=main_index)

        if len(main_index) > 0:
            full_target_shares = target_lots * self.lot_size
            full_trade_shares = full_target_shares - ps
            full_trade_value = full_trade_shares * px
            total_turnover_dollars = full_trade_value.abs().sum()

            diagnostics["total_desired_turnover"] = total_turnover_dollars / equity

            if total_turnover_dollars <= budget_dollars + _EPS:
                final_trade_shares = full_trade_shares.copy()
                diagnostics["scaled"] = False
            else:
                diagnostics["scaled"] = True
                scale = budget_dollars / total_turnover_dollars
                scaled_shares = full_trade_shares * scale
                scaled_lots = np.trunc(scaled_shares / self.lot_size)
                final_trade_shares = pd.Series(
                    scaled_lots.to_numpy() * self.lot_size, index=main_index
                )

                used_dollars = (final_trade_shares * px).abs().sum()
                leftover = budget_dollars - used_dollars
                remaining_shares = full_trade_shares - final_trade_shares

                priority_order = sorted(
                    main_index, key=lambda s: (-abs(tw[s]), str(s))
                )
                for sym in priority_order:
                    if leftover <= _EPS:
                        break
                    rem = remaining_shares[sym]
                    if rem == 0.0:
                        continue
                    direction = 1.0 if rem > 0 else -1.0
                    max_lots_by_target = np.floor(abs(rem) / self.lot_size + _EPS)
                    if max_lots_by_target < 1:
                        continue
                    lot_value = self.lot_size * px[sym]
                    if lot_value <= 0:
                        continue
                    max_lots_by_budget = np.floor(leftover / lot_value + _EPS)
                    n_lots = min(max_lots_by_target, max_lots_by_budget)
                    if n_lots < 1:
                        continue
                    delta = direction * n_lots * self.lot_size
                    final_trade_shares[sym] += delta
                    remaining_shares[sym] -= delta
                    leftover -= abs(delta) * px[sym]

        new_shares_active = ps + final_trade_shares

        # --- step 5: per-name weight cap ----------------------------------
        capped_names: List[str] = []
        if len(main_index) > 0:
            achieved = (new_shares_active * px / equity).abs()
            cap = self.max_single_name_weight
            over = achieved.index[achieved > cap + _EPS]
            for sym in over:
                price = px[sym]
                max_abs_shares = cap * equity / price
                max_lots = np.floor(max_abs_shares / self.lot_size + _EPS)
                sign = np.sign(new_shares_active[sym]) or 1.0
                trimmed = sign * max_lots * self.lot_size
                if abs(trimmed) < abs(new_shares_active[sym]) - _EPS:
                    new_shares_active[sym] = trimmed
                    capped_names.append(str(sym))
        diagnostics["capped_names"] = sorted(capped_names)

        new_shares.loc[new_shares_active.index] = new_shares_active

        # --- assemble outputs ----------------------------------------------
        trades = new_shares - prev_shares
        trade_value = (trades * prices).where(trades != 0.0, 0.0)

        main_turnover_dollars = trade_value.loc[main_index].abs().sum() if len(main_index) else 0.0
        turnover_used = main_turnover_dollars / equity

        achieved_weights = new_shares * prices / equity

        return RebalancePlan(
            new_shares=new_shares,
            trades=trades,
            trade_value=trade_value,
            turnover_used=turnover_used,
            turnover_budget=self.turnover_limit,
            achieved_weights=achieved_weights,
            diagnostics=diagnostics,
        )
