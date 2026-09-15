"""Portfolio construction + portfolio optimization.

``portfolio/construct.py`` (portfolio construction) turns a raw per-day
cross-sectional signal into a target-weight ``pd.Series``; the
``optimize_*``/``optimize_base`` modules (portfolio optimization) consume
that target-weight ``pd.Series`` and turn it into an actual
``RebalancePlan`` of whole-lot share orders under market-specific
constraints (turnover budget, per-name weight cap, lot size,
long-only/short-able), ready to hand to ``execution/``.
"""

from __future__ import annotations

from .construct import (
    KNOWN_VARIANTS,
    CappedWeights,
    blend_signals_by_rolling_icir,
    cap_and_renormalize_weights,
    is_supported_variant,
    top_n_from_variant,
    transform_signal,
)
from .optimize_ashare import AshareLotOptimizer
from .optimize_base import RebalancePlan, TurnoverBudgetOptimizer, align_inputs
from .optimize_us import USEquityOptimizer

__all__ = [
    "RebalancePlan",
    "TurnoverBudgetOptimizer",
    "align_inputs",
    "AshareLotOptimizer",
    "USEquityOptimizer",
    "KNOWN_VARIANTS",
    "CappedWeights",
    "blend_signals_by_rolling_icir",
    "cap_and_renormalize_weights",
    "is_supported_variant",
    "top_n_from_variant",
    "transform_signal",
]
