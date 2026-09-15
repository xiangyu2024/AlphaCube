"""Sequential, multi-day backtest loop.

The project already has real cost/impact/position-limit building blocks --
:class:`~alphacube.portfolio.optimize_base.TurnoverBudgetOptimizer` enforces
turnover budgets, lot sizes, and per-name weight caps;
``execution/almgren_chriss.py`` computes real market-impact cost;
``portfolio/construct.py`` turns signals into weights -- but nothing chains
them into an actual day-over-day loop that carries state (shares, equity)
forward and produces a real equity curve. ``TurnoverBudgetOptimizer.rebalance``
is deliberately stateless (see its module docstring): it has no notion of
"today" vs "tomorrow". :class:`BacktestEngine` is that caller: it owns the
day-over-day state and decides, explicitly, which day's prices mean what.

**Timing contract (see ``optimize_base.py``'s own docstring; the previous
project's ``docs/pipeline.md`` spelled out the same convention).** Each day
``t``: size the rebalance off day ``t``'s close
(the "decision price"), then mark the resulting NEW position to market
using day ``t+1``'s close vs. day ``t``'s close (the "T -> T+1" convention:
you decide today, your position (at its NEW size) earns tomorrow's move).
This is a real modeling choice, not the only valid one -- an equally
defensible alternative marks the OLD (pre-rebalance) position to the day's
realized return and only the NEW position from t+1 onward. We use the
NEW-position convention (spelled out here and at the ``gross_pnl`` line
below) for simplicity: it means "you're fully invested in today's target
by the time today's move happens" rather than modeling an execution lag
within the realized-return calculation itself (T+1 settlement / execution
timing is instead captured by sizing off day ``t``'s price and realizing
against day ``t+1``'s price, not by lagging which shares participate).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ..portfolio.construct import cap_and_renormalize_weights, transform_signal
from .costs import TransactionCostModel
from .metrics import cost_drag_stats, performance_stats


@dataclass
class DailyRecord:
    """One day's worth of backtest bookkeeping."""

    date: Any
    equity_before: float
    equity_after: float
    gross_pnl: float
    transaction_cost: float
    impact_cost: float
    net_pnl: float
    turnover_used: float
    n_positions: int
    max_abs_weight: float
    n_untradable: int


@dataclass
class BacktestResult:
    """Output of :meth:`BacktestEngine.run`."""

    daily: pd.DataFrame
    equity_curve: pd.Series
    summary: dict = field(default_factory=dict)


class BacktestEngine:
    """Chains ``construct.py`` -> a ``TurnoverBudgetOptimizer`` -> costs ->
    (optional) impact -> a day-over-day equity curve.

    Args:
        optimizer: a :class:`~alphacube.portfolio.optimize_base.TurnoverBudgetOptimizer`
            instance (``AshareLotOptimizer``/``USEquityOptimizer``, or any
            object exposing a compatible ``.rebalance(target_weights,
            prev_shares, prices, equity, tradable=None) -> RebalancePlan``
            method plus an ``allow_short`` attribute -- the engine is
            designed primarily against ``TurnoverBudgetOptimizer``'s
            signature, but doesn't hard-require the concrete class).
        cost_model: dollar commission model, see ``costs.py``.
        signal_variant: passed straight through to
            ``portfolio.construct.transform_signal``.
        signal_variant_params: extra kwargs for ``transform_signal``
            (e.g. ``{"temperature": 0.5}`` for ``"softmax_long"``).
        target_exposure / max_single_name_weight: passed to
            ``cap_and_renormalize_weights`` *before* handing the weights to
            ``optimizer`` -- note the optimizer applies its OWN
            ``max_single_name_weight`` again downstream (its own
            per-name cap trim, step 5 of ``TurnoverBudgetOptimizer.rebalance``);
            the two are independent knobs that both end up bounding the
            final achieved weight (whichever is tighter binds).
        impact_model: optional adapter (e.g. ``impact.SimpleImpactModel``)
            exposing ``.estimate(trades, prices) -> float``. ``None`` (the
            default) is a fully valid "friction-free except for commission"
            configuration.
        initial_equity: starting account equity.
    """

    def __init__(
        self,
        optimizer,
        cost_model: TransactionCostModel,
        signal_variant: str = "long_only",
        signal_variant_params: Optional[dict] = None,
        target_exposure: float = 0.9,
        max_single_name_weight: float = 0.1,
        impact_model=None,
        initial_equity: float = 1_000_000.0,
    ) -> None:
        if initial_equity <= 0:
            raise ValueError("initial_equity must be > 0")

        self.optimizer = optimizer
        self.cost_model = cost_model
        self.signal_variant = signal_variant
        self.signal_variant_params = dict(signal_variant_params or {})
        self.target_exposure = float(target_exposure)
        self.max_single_name_weight = float(max_single_name_weight)
        self.impact_model = impact_model
        self.initial_equity = float(initial_equity)

    def run(
        self,
        signal_panel: np.ndarray,
        price_panel: np.ndarray,
        dates: Sequence,
        instruments: Sequence[str],
        start_idx: int = 0,
        end_idx: Optional[int] = None,
    ) -> BacktestResult:
        """Run the day-by-day loop over ``[start_idx, end_idx)``.

        ``signal_panel``/``price_panel`` are ``(n_days, n_instruments)``
        numpy arrays, already aligned (same shape, same date/instrument
        order) -- this method does not fetch or align data itself.

        **No-lookahead boundary.** Realizing day ``t``'s P&L needs day
        ``t+1``'s price, so the loop can never actually process ``t ==
        n_days - 1`` (there is no day-after-the-last-day price to realize
        against). The effective end is therefore
        ``min(end_idx or n_days, n_days - 1)`` -- i.e. the loop runs through
        at most ``n_days - 2`` inclusive, producing at most ``n_days - 1``
        daily records. This is enforced unconditionally (not just when
        ``end_idx`` is omitted), so passing ``end_idx=n_days`` is safe and
        behaves identically to omitting it.
        """
        signal_panel = np.asarray(signal_panel, dtype=np.float64)
        price_panel = np.asarray(price_panel, dtype=np.float64)
        instruments = list(instruments)
        dates = list(dates)

        if signal_panel.shape != price_panel.shape:
            raise ValueError(
                f"signal_panel shape {signal_panel.shape} != price_panel shape {price_panel.shape}"
            )
        n_days, n_instruments = price_panel.shape
        if len(dates) != n_days:
            raise ValueError(f"len(dates)={len(dates)} != n_days={n_days}")
        if len(instruments) != n_instruments:
            raise ValueError(f"len(instruments)={len(instruments)} != n_instruments={n_instruments}")

        if end_idx is None:
            end_idx = n_days
        # Never attempt to process the last day: there is no t+1 price to
        # realize its P&L against (see module/method docstring).
        effective_end = min(end_idx, n_days - 1)

        equity = self.initial_equity
        prev_shares = pd.Series(0.0, index=instruments)
        records = []

        for t in range(start_idx, max(start_idx, effective_end)):
            date = dates[t]
            price_t = price_panel[t]
            price_tp1 = price_panel[t + 1]

            price_valid_t = np.isfinite(price_t) & (price_t > 0.0)
            tradable = pd.Series(price_valid_t, index=instruments)
            prices_t = pd.Series(price_t, index=instruments)

            signal_t = pd.Series(signal_panel[t], index=instruments)
            raw_weights = transform_signal(signal_t, self.signal_variant, **self.signal_variant_params)
            capped = cap_and_renormalize_weights(
                raw_weights,
                self.target_exposure,
                self.max_single_name_weight,
                allow_short=bool(getattr(self.optimizer, "allow_short", False)),
            )

            equity_before = equity

            plan = self.optimizer.rebalance(
                capped.weights, prev_shares, prices_t, equity_before, tradable=tradable
            )

            cost = self.cost_model.compute(plan.trades, prices_t)
            transaction_cost = float(cost["transaction_cost"])

            impact_cost = 0.0
            if self.impact_model is not None:
                impact_cost = float(self.impact_model.estimate(plan.trades, prices_t))

            # --- realize P&L: mark the NEW post-rebalance position to
            # market using t -> t+1 return (see module docstring for the
            # explicit modeling-choice statement). Instruments with an
            # invalid price at t or t+1 contribute zero realized return
            # rather than NaN/inf (a broken price series shouldn't corrupt
            # the whole day's P&L).
            valid_move = price_valid_t & np.isfinite(price_tp1) & (price_tp1 > 0.0)
            realized_return = np.zeros(n_instruments, dtype=np.float64)
            realized_return[valid_move] = price_tp1[valid_move] / price_t[valid_move] - 1.0

            new_shares = plan.new_shares.reindex(instruments).fillna(0.0)
            new_shares_arr = new_shares.to_numpy(dtype=np.float64)
            gross_pnl = float(np.sum(new_shares_arr * price_t * realized_return))

            net_pnl = gross_pnl - transaction_cost - impact_cost
            equity_after = equity_before + net_pnl

            achieved_weights = plan.achieved_weights.reindex(instruments).fillna(0.0)
            n_positions = int((new_shares != 0.0).sum())
            max_abs_weight = float(achieved_weights.abs().max()) if len(achieved_weights) else 0.0
            n_untradable = int((~tradable).sum())

            records.append(
                DailyRecord(
                    date=date,
                    equity_before=equity_before,
                    equity_after=equity_after,
                    gross_pnl=gross_pnl,
                    transaction_cost=transaction_cost,
                    impact_cost=impact_cost,
                    net_pnl=net_pnl,
                    turnover_used=float(plan.turnover_used),
                    n_positions=n_positions,
                    max_abs_weight=max_abs_weight,
                    n_untradable=n_untradable,
                )
            )

            prev_shares = plan.new_shares
            equity = equity_after

        if records:
            daily = pd.DataFrame([asdict(r) for r in records]).set_index("date")
            equity_curve = daily["equity_after"].copy()
            equity_curve.name = "equity"
            net_returns = (daily["net_pnl"] / daily["equity_before"]).to_numpy(dtype=np.float64)
        else:
            daily = pd.DataFrame(
                columns=[
                    "equity_before",
                    "equity_after",
                    "gross_pnl",
                    "transaction_cost",
                    "impact_cost",
                    "net_pnl",
                    "turnover_used",
                    "n_positions",
                    "max_abs_weight",
                    "n_untradable",
                ]
            )
            daily.index.name = "date"
            equity_curve = pd.Series(dtype=np.float64, name="equity")
            net_returns = np.array([], dtype=np.float64)

        summary: Dict[str, Any] = {
            **performance_stats(net_returns),
            **cost_drag_stats(daily),
            "n_days": int(len(daily)),
            "final_equity": float(equity),
        }

        return BacktestResult(daily=daily, equity_curve=equity_curve, summary=summary)
