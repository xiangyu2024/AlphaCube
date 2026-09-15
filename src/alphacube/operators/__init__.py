"""Cleaned, vectorized quant-operator library.

Layout:
    common.py                shared elementwise ops + generic rolling primitives
    nd_cs.py / nd_ts.py       ND (time x stock) cross-sectional / time-series ops
    dt_cross.py / dt_in.py / dt_day.py
                              DT (day x minute) cross-day / inday / day-collapse ops
    dt_spectral.py            DT FFT / spectral-energy-ratio ops (day-collapsing;
                              a bit-exact replay of the legacy implementation,
                              see that module's docstring)
    legacy_specialized_nd.py / legacy_specialized_dt.py
                              exotic / one-off / environment-specific code, kept for
                              reference only, excluded from the generic operator
                              vocabulary below

`operator_dict_ND` / `operator_dict_DT` mirror the OLD `ND_operators.py` /
`DT_operators.py` module-level dicts (same purpose: the eval/exec namespace
`all_mode_fac.py`'s factor-expression strings resolve operator names
against), built here from the new, cleaned, bug-fixed modules instead of the
old ones. `OPERATOR_REGISTRY` (from `common`) carries the mode/category/arity
metadata used to generate `configs/operators.yaml` for the AlphaFactorMining
alpha-search project.

## Backward-compatible aliases

A few functions were renamed or fixed during cleanup because the OLD name
was actively bug-triggering or ambiguous (see each module's docstring for
the full story). Where the OLD code (verified against the one live-executed
factor script we found, `all_mode_fac.py`) called a name directly, that
exact name is kept working here, mapped to the corrected implementation:

- `abs`  -> `common.abs_` (renamed to avoid shadowing the Python builtin
  inside the library itself; the dict key `'abs'` is unaffected by that and
  works exactly as before).
- `cross_zscore_rolling` -> `dt_in.in_zscore_rolling` (the old function was
  named/categorized as a cross-day op but actually rolled along the minute
  axis — moved to the `in_*` family under a correctly-descriptive name;
  aliased back to the old name here so nothing that calls it by the old
  name breaks).
- `in_quantile` -> `dt_in.in_quantile_rolling` (the OLD file defined
  `in_quantile` TWICE with unrelated signatures; Python's last-definition-
  wins meant the ROLLING-WINDOW version was what actually executed at
  runtime, despite an unrelated non-rolling version appearing earlier in
  the file. This alias reproduces that actual runtime behavior. The
  non-rolling "is this value in the row's top quantile" version is now
  `dt_in.in_top_quantile_flag` — call it under that name going forward if
  that's the one you need).
- `cross_product` in `operator_dict_DT` now points at the CORRECT day-axis
  rolling product (`dt_cross.cross_product`). This is a deliberate bug fix,
  not a faithful-replay alias: the old file also defined `cross_product`
  twice, and the shadowing SECOND definition that actually ran was an
  unrelated minute-axis sliding-window product — clearly an accidental
  naming collision (nothing suggests two intentionally different operators
  sharing one name), not two valid semantics. That minute-axis operator is
  preserved under its own honest name, `in_local_window_product`, so no
  functionality is lost; it's just no longer silently aliased to
  `cross_product`.
"""

from __future__ import annotations

from typing import Dict

from . import common
from . import nd_cs
from . import nd_ts
from . import dt_cross
from . import dt_in
from . import dt_day
from . import dt_spectral
from .common import OPERATOR_REGISTRY, op  # noqa: F401  (re-exported for callers)

# a couple of small compat-only pieces that don't belong in the generic
# vocabulary but are needed to fully replace old runtime behavior
from .common import rolling_reduce as _rolling_reduce, nanproduct_last as _nanproduct_last


def in_local_window_product(tensor, n):
    """Rolling product along the minute axis. This is the operator the old
    `DT_operators.py`'s SECOND (shadowing) `cross_product(tensor, n)`
    definition actually computed, despite its misleading name. Kept under
    an honest name; not part of the generic `dt_cross`/`dt_in` vocabulary
    files themselves so it doesn't get tagged into the alpha-search
    registry twice under two names.
    """
    return _rolling_reduce(tensor, n, _nanproduct_last, axis=-1)


def _shared_elementwise_names():
    return [name for name, meta in OPERATOR_REGISTRY.items() if meta["mode"] == "shared"]


def _mode_names(mode: str):
    return [name for name, meta in OPERATOR_REGISTRY.items() if meta["mode"] == mode]


def _build_operator_dict(mode: str) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for name in _shared_elementwise_names() + _mode_names(mode):
        result[name] = OPERATOR_REGISTRY[name]["fn"]
    return result


operator_dict_ND: Dict[str, object] = _build_operator_dict("nd")
operator_dict_DT: Dict[str, object] = _build_operator_dict("dt")

# --- ND compat aliases ---
operator_dict_ND["abs"] = common.abs_
operator_dict_ND["rank"] = common.rank
operator_dict_ND["indneutralize"] = common.indneutralize

# --- DT compat aliases ---
operator_dict_DT["abs"] = common.abs_
operator_dict_DT["cross_zscore_rolling"] = dt_in.in_zscore_rolling
operator_dict_DT["in_quantile"] = dt_in.in_quantile_rolling
operator_dict_DT["cross_product"] = dt_cross.cross_product
operator_dict_DT["in_local_window_product"] = in_local_window_product
operator_dict_DT["rank"] = common.rank
operator_dict_DT["indneutralize"] = common.indneutralize

__all__ = [
    "common", "nd_cs", "nd_ts", "dt_cross", "dt_in", "dt_day", "dt_spectral",
    "OPERATOR_REGISTRY", "op",
    "operator_dict_ND", "operator_dict_DT",
    "in_local_window_product",
]
