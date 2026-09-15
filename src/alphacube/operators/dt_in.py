"""Intraday ("in") DT operators — roll/reduce along `dim=-1` (M, the
intraday-minute axis within a single day).

DT tensors are shape `(D, M)` = (day, intraday-minute). Every rolling/
row-reducing operator here uses `operators.common`'s `axis=-1` primitives
(the SAME `rolling_windows`/`rolling_reduce`/`rolling_pair_reduce`/
`rolling_ols` machinery `dt_cross.py` uses with `axis=0` — same code,
different axis).

Ported from the original `/Users/wxy/Desktop/code/DT_operators.py` `in_*`
family (~50+ functions). The "whole-row" reducers (`in_mean_full`,
`in_sum`, `in_median`, `in_max`, `in_min`, `in_stddev_full`, `in_skew_full`,
`in_kurt_full`, `in_last`, `in_first`) collapse to two-line `common`
row-reducer + `.unsqueeze(-1).expand_as(x)` wrappers; the `*_rolling`
family collapses to `rolling_reduce(..., axis=-1)` one-liners, eliminating
~10 hand-written Python-loop "expanding window" branches in one stroke.
The regression family (`in_regbeta`/`in_regalpha`/`in_regres`/
`in_reg_tvalue`/`in_reg_fvalue`) is now a direct `rolling_ols(x, y, d,
axis=-1)[key]` call each — the original instead transposed `x`/`y`,
called the DAY-axis `cross_reg*` family on the transpose, then transposed
the result back; that indirection is gone, replaced by just passing
`axis=-1` to the same shared primitive `cross_reg*` uses with `axis=0`.

## Bugs fixed while porting

- **`in_r2`'s axis-convention bug.** Despite the `in_` prefix, the
  original rolled along `dim=0` (`x.unfold(0, d, 1)`) — the DAY axis, not
  the minute axis its `in_regbeta` sibling (correctly) uses. Fixed here
  to `rolling_pair_reduce(x, y, d, "corr", axis=-1) ** 2`, matching every
  other `in_reg*` function's axis.
- **`in_quantile` duplicate-name shadowing.** The original defined
  `in_quantile` TWICE: once as `in_quantile(x, quantile, valid_mask=None,
  return_bool=True)` — "flag values in the top quantile of the row",
  unrelated to rolling — and again as `in_quantile(x, q, d)`, a genuine
  rolling-window quantile along the minute axis. The second silently
  shadowed the first at import time. Renamed here to two unambiguous
  names: `in_top_quantile_flag` (the flag version) and
  `in_quantile_rolling` (the rolling version, consistent with the
  `in_*_rolling` naming already used by `in_mean_rolling`/`in_min_rolling`/
  etc).
- **`cross_zscore_rolling`'s axis-convention violation, relocated here.**
  Despite living in the original's `cross_*` section and being named/
  registered as a day-axis op, tracing its `tensor.t()` → pad → `unfold`
  chain shows the window is actually built along the MINUTE axis. It's
  reimplemented here as `in_zscore_rolling` (an `in_*` op, correctly
  named for what it actually computes), NOT left under its old misleading
  `cross_` name in `dt_cross.py`.
- **`in_stddev_full`'s inconsistent denominator.** The original divided
  by `count - 1` (sample variance) while every sibling row-reducer in the
  file (`_stddev`, `day_std`, ...) divides by `count` (population
  variance). Standardized here on population variance via `common._stddev`
  for consistency with the rest of the operator family.

## Renames / API changes vs. the original (flagged clearly, since these
change public names)

- `in_quantile` → split into `in_top_quantile_flag` + `in_quantile_rolling`
  (see above).
- `cross_zscore_rolling` → `in_zscore_rolling` (moved from `dt_cross.py`,
  see above).
- `in_multiregbeta`/`in_multiregres` are NEW (didn't exist in the
  original under those names — only a flag-driven `in_multireg(x_list, y,
  d, output=...)` existed). Added for API symmetry with `dt_cross.py`'s
  `cross_multireg`/`cross_multiregbeta`/`cross_multiregres` trio, which
  already existed as three separate functions; `in_multireg` here mirrors
  `cross_multireg`'s simplified dict-of-independent-per-predictor-fits
  return shape instead of the original's `output=`/`drop_first_t=`
  flag API. See `dt_cross.py`'s module docstring for why the
  multi-predictor family is an independent-per-predictor simplification
  rather than a true joint multivariate solve.

## Dropped while porting (not carried into this file)

- `in_shift_left`/`in_shift_right` — exact special cases of the more
  general `in_shift(x, d, fill_value)` (`d=1`/`d=-1`); redundant.
- `repeat(x)` — a hardcoded-`N=240` special case of `in_expand(x, N)`;
  redundant.
- `in_mean_rolling_multi`/`in_stddev_rolling_multi` — multi-tensor
  composition helpers not referenced anywhere in the requested operator
  surface; the same result is obtainable by combining per-tensor
  `in_mean_rolling`/`in_stddev_rolling` calls at the expression level.
- `delete_open_close` — a one-off pipeline artifact (hardcodes column 0
  and the last 3 columns to NaN for a specific data layout); moved to
  `legacy_specialized_dt.py` instead (see that file).

## Moved to `dt_spectral.py` (see that file's docstring)

- `fft_with_nan_handling`/`energy_ratio_n`/`energy_ratio_band_n`/
  `energy_ratio_topm`/`energy_ratio_topm_minus_q` — a self-contained FFT/
  spectral-energy feature family, orthogonal to the rolling-stat taxonomy
  this file covers. Originally left behind as "a candidate for its own
  `dt_spectral.py` later if needed"; that file NOW EXISTS and the family
  has been ported into it (the live research factor set calls all four
  `energy_ratio_*` functions). Note they are tagged `category="day"`, not
  `"in"`: despite operating along the minute axis they CONSUME it,
  returning `(D, 1)`, so they belong to the day-collapsing taxonomy rather
  than this file's shape-preserving one. `dt_spectral.py` is also a
  deliberate BIT-EXACT replay — unlike this file, no legacy bug in that
  family was fixed, only documented.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from . import common as C
from .common import (
    EPS,
    decay_linear_last,
    first_valid_last,
    get_nan_tensor,
    last_valid_last,
    nanargmax_last,
    nanargmin_last,
    nanmax_last,
    nanmean_last,
    nanmedian_last,
    nanmin_last,
    nanproduct_last,
    nanrank_last,
    nanstd_last,
    nansum_last,
    op,
    rank as _common_rank,
    rolling_ols,
    rolling_pair_reduce,
    rolling_reduce,
    rolling_windows,
    scale_last_ratio,
)

__all__ = [
    # whole-row broadcast reducers
    "in_skew_full", "in_kurt_full", "in_scale", "in_mean_full", "in_demean",
    "in_sum", "in_median", "in_max", "in_min", "in_stddev_full", "in_rank",
    "in_last", "in_first",
    # plain minute-axis indexing
    "in_delay", "in_shift", "in_delta", "in_half", "in_slice", "in_expand",
    # pair-rolling
    "in_cov", "in_corr", "in_autocorr",
    # rolling reducers
    "in_sum_rolling", "in_mean_rolling", "in_min_rolling", "in_max_rolling",
    "in_med", "in_argmin", "in_argmax", "in_rank_rolling", "in_scale_rolling",
    "in_product_rolling", "in_decay_linear_rolling", "in_stddev_rolling",
    "in_quantile_rolling", "in_top_quantile_flag", "in_zscore_rolling",
    # rolling OLS
    "in_regbeta", "in_regalpha", "in_regres", "in_reg_tvalue", "in_reg_fvalue",
    "in_r2", "in_multiregbeta", "in_multiregres", "in_multireg",
    # neighbor / masking utilities
    "in_no_neighbor", "in_is_neighbor", "in_mask_nearby_ones",
    "in_left_one_distance_or_idx", "in_left_one_distance_zero",
    # top-k masked family
    "sliding_top_mean_masked", "sliding_bottom_mean_masked",
    "sliding_top_sum_masked", "sliding_bottom_sum_masked",
    "in_topk_masked", "in_topk_masked_exact",
    # interval-gap statistics
    "in_interval_std", "in_interval_skew", "in_interval_kurt", "in_interval_std_vmap",
    # sustain / cumsum / boxcox / misc generic primitives
    "in_sustain", "in_sustain_mean", "in_sustain_stddev", "in_sustain_product",
    "in_sorted_cumsum_forward", "in_sorted_cumsum_backward", "in_boxcox",
    "in_relative_distance_p", "in_relative_distance_m", "in_local_mean",
    "in_conditional_choose", "in_slice_from_start", "in_slice_to_end",
    "in_forward_valid_part",
]


# ==========================================================================
# whole-row broadcast reducers ("_full" family): common row-reducer +
# `.unsqueeze(-1).expand_as(x)`
# ==========================================================================

@op("dt", "in", arity=1)
def in_skew_full(x: torch.Tensor) -> torch.Tensor:
    return C._skew(x).unsqueeze(-1).expand_as(x)


@op("dt", "in", arity=1)
def in_kurt_full(x: torch.Tensor) -> torch.Tensor:
    return C._kurt(x).unsqueeze(-1).expand_as(x)


@op("dt", "in", arity=1)
def in_mean_full(x: torch.Tensor) -> torch.Tensor:
    return C._mean(x).unsqueeze(-1).expand_as(x)


@op("dt", "in", arity=1)
def in_demean(x: torch.Tensor) -> torch.Tensor:
    return x - in_mean_full(x)


@op("dt", "in", arity=1)
def in_sum(x: torch.Tensor) -> torch.Tensor:
    return C._sum(x).unsqueeze(-1).expand_as(x)


@op("dt", "in", arity=1)
def in_median(x: torch.Tensor) -> torch.Tensor:
    return C._median(x).unsqueeze(-1).expand_as(x)


@op("dt", "in", arity=1)
def in_max(x: torch.Tensor) -> torch.Tensor:
    return nanmax_last(x).unsqueeze(-1).expand_as(x)


@op("dt", "in", arity=1)
def in_min(x: torch.Tensor) -> torch.Tensor:
    return nanmin_last(x).unsqueeze(-1).expand_as(x)


@op("dt", "in", arity=1)
def in_stddev_full(x: torch.Tensor) -> torch.Tensor:
    return C._stddev(x).unsqueeze(-1).expand_as(x)


@op("dt", "in", arity=1)
def in_rank(x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None, pct: bool = True) -> torch.Tensor:
    """Whole-row rank — `common.rank` already reduces/ranks along `dim=-1`,
    exactly the minute axis this operator needs, so it's a direct
    passthrough (no reimplementation needed).
    """
    return _common_rank(x, valid_mask=valid_mask, pct=pct)


@op("dt", "in", arity=1)
def in_last(x: torch.Tensor) -> torch.Tensor:
    """Last non-NaN value of each row, broadcast across the row.

    The original's `in_last` took the LITERAL last column (`x[..., -1:]`),
    ignoring NaN entirely, unlike its day-level analogue `day_last` (which
    IS NaN-aware). Made NaN-aware here for consistency with `day_last`,
    via the shared `common.last_valid_last` (same helper `day_last` uses).
    """
    return last_valid_last(x).unsqueeze(-1).expand_as(x)


@op("dt", "in", arity=1)
def in_first(x: torch.Tensor) -> torch.Tensor:
    """First non-NaN value of each row, broadcast across the row (see
    `in_last`'s docstring — same NaN-awareness fix, mirroring `day_first`).
    """
    return first_valid_last(x).unsqueeze(-1).expand_as(x)


@op("dt", "in", arity=1)
def in_scale(x: torch.Tensor, constant: float = 0.0) -> torch.Tensor:
    """`x / nansum(row) + constant`. The original used `torch.sum`
    (NOT nan-aware) for the denominator, so a single NaN anywhere in the
    row would poison the whole row's output; switched to `nansum` here,
    consistent with every other `common`-backed reducer in this file.
    """
    denom = torch.nansum(x, dim=-1, keepdim=True)
    denom = torch.where(denom.abs() < EPS, torch.ones_like(denom), denom)
    return x / denom + constant


# ==========================================================================
# plain minute-axis indexing (lag / shift / diff / slicing)
# ==========================================================================

@op("dt", "in", arity=1, requires_window=True)
def in_delay(x: torch.Tensor, d: int, fill_zero_for_input_leakage: bool = False) -> torch.Tensor:
    N, T = x.shape
    d = int(d)
    if d >= T:
        result = x.new_full((N, T), float("nan"))
    else:
        result = x.new_full((N, T), float("nan"))
        result[:, d:] = x[:, : T - d]
    if fill_zero_for_input_leakage:
        valid_mask = ~torch.isnan(x[:, :d])
        result[:, :d][valid_mask] = 0.0
    return result


@op("dt", "in", arity=1, requires_window=True)
def in_shift(x: torch.Tensor, d: int, fill_value: float = -1.0) -> torch.Tensor:
    """Shift right by `d` (or left, if `d < 0`) along the minute axis,
    filling vacated cells with `fill_value`.
    """
    B, T = x.shape
    d = int(d)
    if d == 0:
        return x.clone()
    result = torch.full_like(x, fill_value)
    if d > 0:
        if d < T:
            result[:, d:] = x[:, :-d]
    else:
        d_abs = -d
        if d_abs < T:
            result[:, :d_abs] = x[:, d_abs:]
    return result


@op("dt", "in", arity=1, requires_window=True)
def in_delta(x: torch.Tensor, d: int) -> torch.Tensor:
    d = int(d)
    T, N = x.shape
    res = get_nan_tensor(x.shape, x.device, x.dtype)
    if d < N:
        res[:, d:] = x[:, d:] - x[:, :-d]
    return res


@op("dt", "in", arity=1)
def in_half(x: torch.Tensor, front: bool = True) -> torch.Tensor:
    N = x.size(1)
    if N == 241:
        # V2 column 0 is the 09:30 auction.  The legacy half-day operators
        # describe the two 120-minute continuous-auction sessions.
        return x[:, 1:121] if front else x[:, 121:241]
    if N % 2 != 0:
        raise ValueError("Number of columns must be even.")
    mid = N // 2
    return x[:, :mid] if front else x[:, mid:]


@op("dt", "in", arity=1)
def in_slice(x: torch.Tensor, start: int, end: int) -> torch.Tensor:
    return x[:, start : end + 1]


@op("dt", "in", arity=1)
def in_expand(x: torch.Tensor, N: int = 241) -> torch.Tensor:
    """Broadcast a `(D, 1)` tensor out to `(D, N)`."""
    return x.expand(-1, N)


# ==========================================================================
# pair-rolling cov / corr / autocorr (axis=-1)
# ==========================================================================

@op("dt", "in", arity=2, requires_window=True, pair=True)
def in_cov(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_pair_reduce(x, y, int(d), "cov", axis=-1)


@op("dt", "in", arity=2, requires_window=True, pair=True)
def in_corr(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_pair_reduce(x, y, int(d), "corr", axis=-1)


@op("dt", "in", arity=1, requires_window=True)
def in_autocorr(x: torch.Tensor, d: int, shift: int) -> torch.Tensor:
    shift = int(shift)
    res = get_nan_tensor(x.shape, x.device, x.dtype)
    if shift < x.shape[1]:
        res[:, shift:] = in_corr(x[:, shift:], x[:, :-shift], d)
    return res


# ==========================================================================
# rolling reducers (axis=-1) — the `rolling_reduce`/`rolling_windows`
# one-liners that replace ~10 hand-written Python-loop branches
# ==========================================================================

@op("dt", "in", arity=1, requires_window=True)
def in_sum_rolling(x: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_reduce(x, int(d), nansum_last, axis=-1)


@op("dt", "in", arity=1, requires_window=True)
def in_mean_rolling(x: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_reduce(x, int(d), nanmean_last, axis=-1)


@op("dt", "in", arity=1, requires_window=True)
def in_min_rolling(x: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_reduce(x, int(d), nanmin_last, axis=-1)


@op("dt", "in", arity=1, requires_window=True)
def in_max_rolling(x: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_reduce(x, int(d), nanmax_last, axis=-1)


@op("dt", "in", arity=1, requires_window=True)
def in_med(x: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_reduce(x, int(d), nanmedian_last, axis=-1)


@op("dt", "in", arity=1, requires_window=True)
def in_argmin(x: torch.Tensor, d: int, pct: bool = True) -> torch.Tensor:
    return rolling_reduce(x, int(d), lambda w: nanargmin_last(w, pct=pct), axis=-1)


@op("dt", "in", arity=1, requires_window=True)
def in_argmax(x: torch.Tensor, d: int, pct: bool = True) -> torch.Tensor:
    return rolling_reduce(x, int(d), lambda w: nanargmax_last(w, pct=pct), axis=-1)


@op("dt", "in", arity=1, requires_window=True)
def in_rank_rolling(x: torch.Tensor, d: int, pct: bool = True) -> torch.Tensor:
    return rolling_reduce(x, int(d), lambda w: nanrank_last(w, pct=pct), axis=-1)


@op("dt", "in", arity=1, requires_window=True)
def in_scale_rolling(x: torch.Tensor, d: int, constant: float = 0.0) -> torch.Tensor:
    w = rolling_windows(x, int(d), axis=-1)
    return scale_last_ratio(x, w) + constant


@op("dt", "in", arity=1, requires_window=True)
def in_product_rolling(x: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_reduce(x, int(d), nanproduct_last, axis=-1)


@op("dt", "in", arity=1, requires_window=True)
def in_decay_linear_rolling(x: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_reduce(x, int(d), decay_linear_last, axis=-1)


@op("dt", "in", arity=1, requires_window=True)
def in_stddev_rolling(x: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_reduce(x, int(d), nanstd_last, axis=-1)


@op("dt", "in", arity=1, requires_window=True)
def in_quantile_rolling(x: torch.Tensor, q: float, d: int) -> torch.Tensor:
    """Rolling-window quantile along the minute axis. Renamed from the
    original's second `in_quantile(x, q, d)` definition — see module
    docstring for the duplicate-name shadowing this resolves.
    """
    w = rolling_windows(x, int(d), axis=-1)
    return torch.nanquantile(w, q, dim=-1)


@op("dt", "in", arity=1)
def in_top_quantile_flag(
    x: torch.Tensor, quantile: float, valid_mask: Optional[torch.Tensor] = None, return_bool: bool = True
) -> torch.Tensor:
    """Flag values in the top `quantile` fraction of each row (NOT a
    rolling-window op). Renamed from the original's first `in_quantile`
    definition — see module docstring.

    Also fixes a latent crash: the original did `flags[~valid_mask] =
    torch.nan` unconditionally, including on the `return_bool=True`
    (default) path where `flags` is a bool tensor — assigning float NaN
    into a bool-dtype tensor. Here the bool path just leaves invalid
    entries `False` (a bool tensor cannot represent NaN); only the
    `return_bool=False` (float) path NaN-masks invalid entries.
    """
    if not (0.0 <= quantile <= 1.0):
        raise ValueError("quantile must be in [0, 1]")
    if valid_mask is None:
        valid_mask = torch.isfinite(x)
    x_masked = torch.where(valid_mask, x, torch.full_like(x, float("nan")))
    thresholds = torch.nanquantile(x_masked, q=1 - quantile, dim=-1, keepdim=True)
    flagged = valid_mask & (x >= thresholds)
    if return_bool:
        return flagged
    flags = flagged.to(torch.float32)
    return torch.where(valid_mask, flags, torch.full_like(flags, float("nan")))


@op("dt", "in", arity=1, requires_window=True)
def in_zscore_rolling(x: torch.Tensor, d: int) -> torch.Tensor:
    """Rolling z-score along the minute axis (0 and NaN both treated as
    invalid input, matching the original's masking convention). Moved
    and renamed from the original's `cross_zscore_rolling` — see module
    docstring for why (it never actually rolled along the day axis).
    """
    d = int(d)
    invalid = (x == 0) | torch.isnan(x)
    data = torch.where(invalid, torch.full_like(x, float("nan")), x)
    mean = rolling_reduce(data, d, nanmean_last, axis=-1)
    std = rolling_reduce(data, d, nanstd_last, axis=-1)
    z = (x - mean) / (std + EPS)
    return torch.where(invalid, torch.zeros_like(z), z)


# ==========================================================================
# rolling single-predictor OLS (axis=-1) — direct `rolling_ols` calls;
# the original instead transposed x/y, called the day-axis `cross_reg*`,
# then transposed back. That indirection is gone.
# ==========================================================================

@op("dt", "in", arity=2, requires_window=True, pair=True)
def in_regbeta(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_ols(x, y, int(d), axis=-1)["beta"]


@op("dt", "in", arity=2, requires_window=True, pair=True)
def in_regalpha(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_ols(x, y, int(d), axis=-1)["alpha"]


@op("dt", "in", arity=2, requires_window=True, pair=True)
def in_regres(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_ols(x, y, int(d), axis=-1)["resid"]


@op("dt", "in", arity=2, requires_window=True, pair=True)
def in_reg_tvalue(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_ols(x, y, int(d), axis=-1)["tvalue"]


@op("dt", "in", arity=2, requires_window=True, pair=True)
def in_reg_fvalue(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_ols(x, y, int(d), axis=-1)["fvalue"]


@op("dt", "in", arity=2, requires_window=True, pair=True)
def in_r2(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    """Rolling R^2 along the minute axis. FIXES the original's
    axis-convention bug (it rolled along `dim=0`, the day axis, despite
    the `in_` prefix — see module docstring).
    """
    r = rolling_pair_reduce(x, y, int(d), "corr", axis=-1)
    return (r ** 2).clamp(0.0, 1.0)


@op("dt", "in", arity=1, requires_window=True)
def in_multiregbeta(x_list: List[torch.Tensor], y: torch.Tensor, d: int) -> List[torch.Tensor]:
    """NEW — added for API symmetry with `dt_cross.py`'s
    `cross_multiregbeta` (see module docstring). Independent rolling beta
    of `y` against each predictor, axis=-1.
    """
    d = int(d)
    return [rolling_ols(x, y, d, axis=-1)["beta"] for x in x_list]


@op("dt", "in", arity=1, requires_window=True)
def in_multiregres(x_list: List[torch.Tensor], y: torch.Tensor, d: int) -> torch.Tensor:
    """NEW — added for API symmetry with `dt_cross.py`'s
    `cross_multiregres` (see module docstring)."""
    betas = in_multiregbeta(x_list, y, d)
    res = y
    for x, beta in zip(x_list, betas):
        res = res - beta * x
    return res


@op("dt", "in", arity=1, requires_window=True)
def in_multireg(x_list: List[torch.Tensor], y: torch.Tensor, d: int) -> Dict[str, object]:
    """Bundle of independent per-predictor rolling OLS results, axis=-1.

    Replaces the original's `in_multireg(x_list, y, d, output=...,
    drop_first_t=...)` flag-driven API (which delegated to the buggy
    `cross_multireg_core` — see `dt_cross.py`'s module docstring for that
    bug) with the same simple dict shape `cross_multireg` returns. This
    is a deliberate API simplification, flagged here since it changes the
    call signature from the original.
    """
    d = int(d)
    results = [rolling_ols(x, y, d, axis=-1) for x in x_list]
    return {
        "alpha": [r["alpha"] for r in results],
        "beta": [r["beta"] for r in results],
        "tvalue": [r["tvalue"] for r in results],
        "fvalue": [r["fvalue"] for r in results],
        "resid": in_multiregres(x_list, y, d),
    }


# ==========================================================================
# neighbor / masking utilities (generic, reusable — kept close to the
# original, already reasonably vectorized)
# ==========================================================================

@op("dt", "in", arity=1)
def in_no_neighbor(x: torch.Tensor) -> torch.Tensor:
    """1 where both the left and right neighbor are 0 (edges treated as
    having an implicit 0 neighbor outside the row)."""
    B, T = x.shape
    left = torch.cat([torch.zeros(B, 1, device=x.device, dtype=x.dtype), x[:, :-1]], dim=1)
    right = torch.cat([x[:, 1:], torch.zeros(B, 1, device=x.device, dtype=x.dtype)], dim=1)
    return ((left == 0) & (right == 0)).float()


@op("dt", "in", arity=1)
def in_is_neighbor(x: torch.Tensor) -> torch.Tensor:
    """1 where either the left or right neighbor is 1."""
    B, T = x.shape
    left = torch.cat([torch.zeros(B, 1, device=x.device, dtype=x.dtype), x[:, :-1]], dim=1)
    right = torch.cat([x[:, 1:], torch.zeros(B, 1, device=x.device, dtype=x.dtype)], dim=1)
    return ((left == 1) | (right == 1)).float()


@op("dt", "in", arity=1, requires_window=True)
def in_mask_nearby_ones(x: torch.Tensor, d: int) -> torch.Tensor:
    """Greedily keep only every `d`-apart `1` in each row (scanning
    left-to-right), masking the rest to NaN. `x` should contain only `1`
    and `NaN`.
    """
    if d < 0:
        raise ValueError("d must be non-negative")
    result = x.clone()
    ones = torch.nonzero(~torch.isnan(x))
    if ones.numel() == 0:
        return result
    _, idx = torch.sort(ones[:, 0] * x.shape[1] + ones[:, 1])
    sorted_ones = ones[idx]
    last_anchor = torch.full((x.shape[0],), -(10 ** 9), dtype=torch.long, device=x.device)
    for i in range(sorted_ones.size(0)):
        b = int(sorted_ones[i, 0].item())
        pos = int(sorted_ones[i, 1].item())
        if pos > last_anchor[b] + d:
            last_anchor[b] = pos
        else:
            result[b, pos] = float("nan")
    return result


@op("dt", "in", arity=1)
def in_left_one_distance_or_idx(x: torch.Tensor) -> torch.Tensor:
    """Distance to the nearest `1` to the left (1-based); if none, the
    column index + 1."""
    B, L = x.shape
    device = x.device
    col_idx = torch.arange(1, L + 1, device=device, dtype=torch.float32)
    masked = torch.where(x.bool(), col_idx.unsqueeze(0), torch.full_like(col_idx.unsqueeze(0), float("-inf")))
    left_col = torch.cummax(masked, dim=1)[0]
    dist = col_idx.unsqueeze(0) - left_col + 1
    dist = torch.where(left_col.isinf(), col_idx.unsqueeze(0), dist)
    return dist.long()


@op("dt", "in", arity=1)
def in_left_one_distance_zero(x: torch.Tensor) -> torch.Tensor:
    """Distance to the nearest `1` to the left (0-based); if none, `0`."""
    B, L = x.shape
    device = x.device
    col_idx = torch.arange(L, device=device, dtype=torch.float32)
    masked = torch.where(x.bool(), col_idx.unsqueeze(0), torch.full_like(col_idx.unsqueeze(0), float("-inf")))
    left_col = torch.cummax(masked, dim=1)[0]
    dist = col_idx.unsqueeze(0) - left_col
    dist = torch.where(left_col.isinf(), torch.zeros_like(dist), dist)
    return dist.long()


# ==========================================================================
# top-k masked family (generic reusable primitives — already vectorized
# in the original, kept close to it)
# ==========================================================================

def _sliding_windows(x: torch.Tensor, w: int) -> torch.Tensor:
    """`(T, F)` -> `(T, w, F)` causal rolling windows, NaN-padded."""
    T, Feat = x.shape
    pad = w - 1
    x_pad = F.pad(x.T.unsqueeze(0), (pad, 0), value=float("nan")).squeeze(0)  # [F, T+pad]
    return x_pad.as_strided(
        size=(Feat, T, w), stride=(x_pad.stride(0), x_pad.stride(1), x_pad.stride(1))
    ).permute(1, 2, 0)  # [T, w, F]


def _topk_masked(x: torch.Tensor, y: torch.Tensor, w: int, p: int, largest: bool) -> torch.Tensor:
    device = x.device
    x_win = _sliding_windows(x, w)
    y_win = _sliding_windows(y, w)
    x_filled = torch.where(torch.isnan(x_win), torch.full_like(x_win, float("-inf") if largest else float("inf")), x_win)
    valid_cnt = (~torch.isnan(x_win)).sum(dim=1)
    k_vals = ((valid_cnt + 1) // p).clamp(min=1, max=w)
    max_k = min(int(k_vals.max().item()), w)
    _, topk_idx = torch.topk(x_filled, max_k, dim=1, largest=largest, sorted=False)
    y_top = y_win.gather(1, topk_idx)
    mask = torch.arange(max_k, device=device).view(1, max_k, 1) < k_vals.unsqueeze(1)
    return torch.where(mask, y_top, torch.full_like(y_top, float("nan")))


@op("dt", "in", arity=2, requires_window=True, pair=True)
def sliding_top_mean_masked(x: torch.Tensor, y: torch.Tensor, w: int, p: int) -> torch.Tensor:
    return torch.nanmean(_topk_masked(x, y, w, p, largest=True), dim=1)


@op("dt", "in", arity=2, requires_window=True, pair=True)
def sliding_bottom_mean_masked(x: torch.Tensor, y: torch.Tensor, w: int, p: int) -> torch.Tensor:
    return torch.nanmean(_topk_masked(x, y, w, p, largest=False), dim=1)


@op("dt", "in", arity=2, requires_window=True, pair=True)
def sliding_top_sum_masked(x: torch.Tensor, y: torch.Tensor, w: int, p: int) -> torch.Tensor:
    return torch.nansum(_topk_masked(x, y, w, p, largest=True), dim=1)


@op("dt", "in", arity=2, requires_window=True, pair=True)
def sliding_bottom_sum_masked(x: torch.Tensor, y: torch.Tensor, w: int, p: int) -> torch.Tensor:
    return torch.nansum(_topk_masked(x, y, w, p, largest=False), dim=1)


@op("dt", "in", arity=2, pair=True)
def in_topk_masked(x: torch.Tensor, y: torch.Tensor, p: int, largest: bool = True) -> torch.Tensor:
    B, D = x.shape
    device = x.device
    x_filled = torch.where(torch.isnan(x), torch.full_like(x, float("-inf") if largest else float("inf")), x)
    valid_cnt = (~torch.isnan(x)).sum(dim=1)
    k_vals = ((valid_cnt + 1) // p).clamp(min=1, max=D)
    max_k = int(k_vals.max().item())
    _, topk_indices = torch.topk(x_filled, max_k, dim=1, largest=largest)
    result = torch.full_like(y, float("nan"))
    row_indices = torch.arange(B, device=device).unsqueeze(1)
    mask = torch.arange(max_k, device=device).unsqueeze(0) < k_vals.unsqueeze(1)
    result[row_indices, topk_indices] = torch.where(mask, y[row_indices, topk_indices], torch.full_like(y[row_indices, topk_indices], float("nan")))
    return result


@op("dt", "in", arity=2, pair=True)
def in_topk_masked_exact(x: torch.Tensor, y: torch.Tensor, k: int, largest: bool = True) -> torch.Tensor:
    B, D = x.shape
    device = x.device
    if k <= 0:
        raise ValueError("k must be positive")
    fill_value = float("-inf") if largest else float("inf")
    x_filled = torch.where(torch.isnan(x), torch.full_like(x, fill_value), x)
    valid_cnt = (~torch.isnan(x)).sum(dim=1)
    k_eff = torch.minimum(torch.full_like(valid_cnt, k), valid_cnt)
    max_k = int(k_eff.max().item())
    if max_k == 0:
        return torch.full_like(y, float("nan"))
    _, topk_indices = torch.topk(x_filled, max_k, dim=1, largest=largest)
    arange = torch.arange(max_k, device=device).unsqueeze(0)
    mask = arange < k_eff.unsqueeze(1)
    result = torch.full_like(y, float("nan"))
    row_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, max_k)
    result[row_indices[mask], topk_indices[mask]] = y[row_indices[mask], topk_indices[mask]]
    return result


# ==========================================================================
# interval-gap statistics (distance between consecutive 1s in each row)
# ==========================================================================

def _gap_distances(row: torch.Tensor) -> Optional[torch.Tensor]:
    one_positions = torch.nonzero(row == 1).squeeze(-1)
    if one_positions.numel() < 2:
        return None
    return (one_positions[1:] - one_positions[:-1]).float()


@op("dt", "in", arity=1)
def in_interval_std(x: torch.Tensor) -> torch.Tensor:
    """Std-dev of the gaps between consecutive `1`s in each row."""
    B = x.shape[0]
    result = torch.zeros(B, 1, device=x.device)
    for b in range(B):
        d = _gap_distances(x[b])
        if d is not None and d.numel() > 1:
            result[b, 0] = torch.std(d)
    return result


@op("dt", "in", arity=1)
def in_interval_skew(x: torch.Tensor) -> torch.Tensor:
    """Skewness of the gaps between consecutive `1`s in each row."""
    B = x.shape[0]
    result = torch.zeros(B, 1, device=x.device)
    for b in range(B):
        d = _gap_distances(x[b])
        if d is not None and d.numel() > 1:
            mean, std = torch.mean(d), torch.std(d)
            if std > 0:
                result[b, 0] = torch.mean(((d - mean) / std) ** 3)
    return result


@op("dt", "in", arity=1)
def in_interval_kurt(x: torch.Tensor) -> torch.Tensor:
    """Excess kurtosis of the gaps between consecutive `1`s in each row."""
    B = x.shape[0]
    result = torch.zeros(B, 1, device=x.device)
    for b in range(B):
        d = _gap_distances(x[b])
        if d is not None and d.numel() > 1:
            mean, std = torch.mean(d), torch.std(d)
            if std > 0:
                result[b, 0] = torch.mean(((d - mean) / std) ** 4) - 3.0
    return result


@op("dt", "in", arity=1)
def in_interval_std_vmap(x: torch.Tensor) -> torch.Tensor:
    """`vmap`-based variant of `in_interval_std`; falls back to it if
    `functorch`/`torch.vmap` isn't available."""
    try:
        from functorch import vmap

        def single_row(row: torch.Tensor) -> torch.Tensor:
            d = _gap_distances(row)
            if d is not None and d.numel() > 1:
                return torch.std(d)
            return torch.tensor(0.0, device=row.device)

        return vmap(single_row)(x).unsqueeze(-1)
    except Exception:
        # `vmap` cannot batch over `nonzero` (dynamic output shape) — this
        # is a fundamental vmap limitation, not something fixable here, so
        # fall back to the plain Python-loop version on ANY failure
        # (missing functorch OR a runtime vmap error), not just ImportError.
        return in_interval_std(x)


# ==========================================================================
# sustain / cumsum / boxcox / misc generic primitives (kept close to the
# original — already reasonably vectorized, reusable building blocks)
# ==========================================================================

def _get_rowwise_windows(mask: torch.Tensor, data: torch.Tensor, d: int, start_offset: int = 0):
    """For every `mask[i, j] == 1`, gather the window `data[i, j+start_offset
    : j+start_offset+d]` (NaN where it runs off the row)."""
    assert mask.shape == data.shape, "mask and data must have the same shape"
    N, T = mask.shape
    row_idx, col_idx = torch.where(mask == 1)
    K = row_idx.size(0)
    if K == 0:
        empty = torch.zeros(0, d, dtype=data.dtype, device=data.device)
        valid = torch.zeros(0, d, dtype=torch.bool, device=data.device)
        return row_idx, col_idx, empty, valid
    offsets = torch.arange(d, device=mask.device)
    window_cols = col_idx.unsqueeze(1) + start_offset + offsets
    valid_mask = (window_cols >= 0) & (window_cols < T)
    clamped_cols = torch.clamp(window_cols, 0, T - 1)
    window_vals = data[row_idx.unsqueeze(1), clamped_cols]
    window_vals = torch.where(valid_mask, window_vals, torch.full_like(window_vals, float("nan")))
    return row_idx, col_idx, window_vals, valid_mask


@op("dt", "in", arity=2, requires_window=True, pair=True)
def in_sustain_mean(mask: torch.Tensor, data: torch.Tensor, d: int, start_offset: int = 0, num: bool = False) -> torch.Tensor:
    """At each `mask==1` trigger, the mean of `data[i, j:j+d]`."""
    N, T = mask.shape
    row_idx, col_idx, window_vals, _ = _get_rowwise_windows(mask, data, d, start_offset)
    result = torch.zeros(N, T, dtype=data.dtype, device=data.device) if num else get_nan_tensor((N, T), data.device, data.dtype)
    if row_idx.numel() == 0:
        return result
    result[row_idx, col_idx] = torch.nanmean(window_vals, dim=1)
    return result


@op("dt", "in", arity=2, requires_window=True, pair=True)
def in_sustain_stddev(mask: torch.Tensor, data: torch.Tensor, d: int, start_offset: int = 0, num: bool = False) -> torch.Tensor:
    """At each `mask==1` trigger, the (NaN-aware) std-dev of `data[i, j:j+d]`."""
    N, T = mask.shape
    row_idx, col_idx, window_vals, _ = _get_rowwise_windows(mask, data, d, start_offset)
    fill_value = 0.0 if num else float("nan")
    result = torch.full((N, T), fill_value, dtype=data.dtype, device=data.device)
    if row_idx.numel() == 0:
        return result
    means = torch.nanmean(window_vals, dim=1, keepdim=True)
    var = torch.nanmean((window_vals - means) ** 2, dim=1)
    result[row_idx, col_idx] = torch.sqrt(var)
    return result


@op("dt", "in", arity=2, requires_window=True, pair=True)
def in_sustain_product(mask: torch.Tensor, data: torch.Tensor, d: int, num: bool = False) -> torch.Tensor:
    """At each `mask==1` trigger, the product of `data[i, j:j+d]`."""
    N, T = mask.shape
    row_idx, col_idx, window_vals, valid_mask = _get_rowwise_windows(mask, data, d)
    result = torch.zeros(N, T, dtype=data.dtype, device=data.device) if num else get_nan_tensor((N, T), data.device, data.dtype)
    if row_idx.numel() == 0:
        return result
    filled_vals = torch.where(valid_mask, window_vals, torch.ones_like(window_vals))
    products = torch.prod(filled_vals, dim=1)
    any_valid = valid_mask.any(dim=1)
    products = torch.where(any_valid, products, torch.full_like(products, float("nan")))
    result[row_idx, col_idx] = products
    return result


@op("dt", "in", arity=1, requires_window=True)
def in_sustain(x: torch.Tensor, d: int, num: bool = False) -> torch.Tensor:
    """Propagate every `1` in `x` RIGHT `d` positions (minutes), inclusive
    of itself. Fully vectorized scatter, no Python loop."""
    d = int(d)
    if d <= 0:
        raise ValueError("d must be a positive integer")
    N, T = x.shape
    ones = x == 1
    row_idx, col_idx = torch.where(ones)
    if row_idx.numel() == 0:
        return torch.zeros((N, T), dtype=x.dtype, device=x.device) if num else get_nan_tensor((N, T), x.device, x.dtype)
    k = torch.arange(d, device=x.device)
    col_expanded = col_idx.unsqueeze(1) + k
    valid_mask = col_expanded < T
    col_expanded = torch.clamp(col_expanded, max=T - 1)
    target_cols = col_expanded[valid_mask]
    target_rows = row_idx.unsqueeze(1).expand_as(col_expanded)[valid_mask]
    mask = torch.zeros(N, T, dtype=torch.bool, device=x.device)
    mask[target_rows, target_cols] = True
    result = torch.zeros((N, T), dtype=x.dtype, device=x.device) if num else get_nan_tensor((N, T), x.device, x.dtype)
    result[mask] = 1.0
    return result


@op("dt", "in", arity=2, pair=True)
def in_sorted_cumsum_forward(a: torch.Tensor, b: torch.Tensor, nan_policy: str = "omit") -> torch.Tensor:
    """Sort each row of `a`, apply the same permutation to `b`, forward
    (increasing-`a`) cumulative sum of `b`, NaN-safe."""
    B, N = a.shape
    mask_a, mask_b = torch.isnan(a), torch.isnan(b)
    mask_valid = ~(mask_a | mask_b)
    key = torch.where(mask_a, torch.full_like(a, float("inf") if nan_policy == "mask_last" else float("-inf")), a)
    _, sort_idx = torch.sort(key, dim=1)
    sorted_b = torch.gather(b, 1, sort_idx)
    sorted_valid = torch.gather(mask_valid, 1, sort_idx)
    sorted_b_safe = torch.where(sorted_valid, sorted_b, torch.zeros_like(sorted_b))
    cumsum = torch.cumsum(sorted_b_safe, dim=1)
    return torch.where(sorted_valid, cumsum, torch.full_like(cumsum, float("nan")))


@op("dt", "in", arity=2, pair=True)
def in_sorted_cumsum_backward(a: torch.Tensor, b: torch.Tensor, nan_policy: str = "omit") -> torch.Tensor:
    """Same as `in_sorted_cumsum_forward` but cumulating from the
    largest-`a` end backward."""
    B, N = a.shape
    mask_a, mask_b = torch.isnan(a), torch.isnan(b)
    mask_valid = ~(mask_a | mask_b)
    key = torch.where(mask_a, torch.full_like(a, float("inf") if nan_policy == "mask_last" else float("-inf")), a)
    _, sort_idx = torch.sort(key, dim=1)
    sorted_b = torch.gather(b, 1, sort_idx)
    sorted_valid = torch.gather(mask_valid, 1, sort_idx)
    sorted_b_safe = torch.where(sorted_valid, sorted_b, torch.zeros_like(sorted_b))
    flipped_b = torch.flip(sorted_b_safe, dims=[1])
    cumsum_backward = torch.flip(torch.cumsum(flipped_b, dim=1), dims=[1])
    return torch.where(sorted_valid, cumsum_backward, torch.full_like(cumsum_backward, float("nan")))


@op("dt", "in", arity=1)
def in_boxcox(x: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Signed-log transform, then row-standardized. `0` is treated as
    invalid (mapped to NaN before the transform, like the original)."""
    x = torch.where(x == 0.0, torch.full_like(x, float("nan")), x)
    x_t = torch.sign(x) * torch.log(torch.abs(x) + epsilon)
    mean = torch.nanmean(x_t, dim=1, keepdim=True)
    std = torch.sqrt(torch.nanmean((x_t - mean) ** 2, dim=1, keepdim=True) + EPS)
    return (x_t - mean) / std


@op("dt", "in", arity=1)
def in_relative_distance_p(x: torch.Tensor) -> torch.Tensor:
    """`x/prefix_min + x/prefix_max`, where `prefix_min`/`prefix_max` are
    the running min/max EXCLUDING the current position (column 0 -> NaN)."""
    cummin, _ = torch.cummin(x, dim=1)
    cummax, _ = torch.cummax(x, dim=1)
    prefix_min = torch.full_like(cummin, float("inf"))
    prefix_max = torch.full_like(cummax, float("-inf"))
    prefix_min[:, 1:] = cummin[:, :-1]
    prefix_max[:, 1:] = cummax[:, :-1]
    safe_min = torch.where(torch.isfinite(prefix_min), prefix_min, x + 1)
    safe_max = torch.where(torch.isfinite(prefix_max), prefix_max, x - 1)
    result = x / (safe_min + EPS) + x / (safe_max + EPS)
    result = result.clone()
    result[:, 0] = float("nan")
    return result


@op("dt", "in", arity=1)
def in_relative_distance_m(x: torch.Tensor) -> torch.Tensor:
    """`x/prefix_min - x/prefix_max` (see `in_relative_distance_p`)."""
    cummin, _ = torch.cummin(x, dim=1)
    cummax, _ = torch.cummax(x, dim=1)
    prefix_min = torch.full_like(cummin, float("inf"))
    prefix_max = torch.full_like(cummax, float("-inf"))
    prefix_min[:, 1:] = cummin[:, :-1]
    prefix_max[:, 1:] = cummax[:, :-1]
    safe_min = torch.where(torch.isfinite(prefix_min), prefix_min, x + 1)
    safe_max = torch.where(torch.isfinite(prefix_max), prefix_max, x - 1)
    result = x / (safe_min + EPS) - x / (safe_max + EPS)
    result = result.clone()
    result[:, 0] = float("nan")
    return result


@op("dt", "in", arity=1, requires_window=True)
def in_local_mean(x: torch.Tensor, d: int) -> torch.Tensor:
    """`result[i, j] = nanmean(x[i, j-d+1 : j+d])` — a centered (non-causal)
    local window of radius `d-1`, unlike the causal `in_mean_rolling`."""
    if d <= 0:
        raise ValueError("d must be a positive integer")
    N, T = x.shape
    if d == 1:
        return x.clone()
    window_size = 2 * d - 1
    offsets = torch.arange(-d + 1, d, device=x.device)
    col_indices = torch.arange(T, device=x.device).unsqueeze(1) + offsets
    valid = (col_indices >= 0) & (col_indices < T)
    clamped = torch.clamp(col_indices, 0, T - 1)
    expanded = x.unsqueeze(1).expand(N, T, T)
    gathered = torch.gather(expanded, dim=2, index=clamped.unsqueeze(0).expand(N, T, window_size))
    window_vals = torch.where(valid.unsqueeze(0).expand_as(gathered), gathered, torch.full_like(gathered, float("nan")))
    return torch.nanmean(window_vals, dim=2)


@op("dt", "in", arity=4)
def in_conditional_choose(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Row-wise choose `x` or `y` by comparing scalars `z`/`t`: `z>t` -> `x`,
    else `y`; NaN in one of `z`/`t` picks the other's side; NaN in both ->
    NaN."""
    z, t = z.squeeze(-1), t.squeeze(-1)
    z_nan, t_nan = torch.isnan(z), torch.isnan(t)
    only_z_nan, only_t_nan = z_nan & ~t_nan, ~z_nan & t_nan
    neither_nan = ~z_nan & ~t_nan
    out = torch.full_like(x, float("nan"))
    out[only_z_nan] = y[only_z_nan]
    out[only_t_nan] = x[only_t_nan]
    comp = z[neither_nan] > t[neither_nan]
    out[neither_nan] = torch.where(comp.unsqueeze(-1), x[neither_nan], y[neither_nan])
    return out


@op("dt", "in", arity=2, pair=True)
def in_slice_from_start(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Row `i` keeps `x[i, 0:y[i]+1]`, rest NaN; NaN `y[i]` -> whole row NaN."""
    B, N = x.shape
    device, dtype = x.device, x.dtype
    result = get_nan_tensor((B, N), device, dtype)
    y_flat = y.squeeze(1)
    is_valid = ~torch.isnan(y_flat)
    indices = torch.where(is_valid, y_flat, torch.full_like(y_flat, -1.0)).long()
    if is_valid.any():
        col_indices = torch.arange(N, device=device).expand(B, N)
        mask = col_indices <= indices.unsqueeze(1)
        result[is_valid] = torch.where(mask[is_valid], x[is_valid], result[is_valid])
    return result


@op("dt", "in", arity=2, pair=True)
def in_slice_to_end(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Row `i` keeps `x[i, y[i]:]`, rest NaN; NaN `y[i]` -> whole row NaN."""
    B, N = x.shape
    device, dtype = x.device, x.dtype
    result = get_nan_tensor((B, N), device, dtype)
    y_flat = y.squeeze(1)
    is_valid = ~torch.isnan(y_flat)
    indices = torch.where(is_valid, y_flat, torch.full_like(y_flat, float(N))).long()
    if is_valid.any():
        col_indices = torch.arange(N, device=device).expand(B, N)
        mask = col_indices >= indices.unsqueeze(1)
        result[is_valid] = torch.where(mask[is_valid], x[is_valid], result[is_valid])
    return result


@op("dt", "in", arity=1)
def in_forward_valid_part(x: torch.Tensor) -> torch.Tensor:
    """Forward-fill: `result[i, j]` = the most recent valid (nonzero,
    non-NaN) value strictly BEFORE `j`; NaN if none."""
    D, T = x.shape
    device = x.device
    is_valid = (x != 0) & (~torch.isnan(x))
    x_filled = torch.where(is_valid, x, torch.full_like(x, float("nan")))
    col_indices = torch.arange(T, device=device)
    valid_indices = torch.where(is_valid, col_indices, torch.full_like(col_indices, -1))
    shifted = torch.full_like(valid_indices, -1)
    if T > 1:
        shifted[:, 1:] = valid_indices[:, :-1]
    prev_cummax = torch.cummax(shifted, dim=1)[0]
    gather_indices = torch.where(prev_cummax == -1, torch.zeros_like(prev_cummax), prev_cummax)
    gathered = torch.gather(x_filled, 1, gather_indices)
    return torch.where(prev_cummax == -1, torch.full_like(gathered, float("nan")), gathered)
