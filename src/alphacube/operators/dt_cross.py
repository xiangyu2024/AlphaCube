"""Cross-day ("cross") DT operators — roll along `dim=0` (D, the day axis).

DT tensors are shape `(D, M)` = (day, intraday-minute). Every operator here
rolls a window of `d` DAYS at a fixed minute-of-day, using `operators.common`'s
`axis=0` rolling primitives (the same machinery the `nd_ts.py` ts_* family
uses with `axis=0` for its own `(T, N)` convention — same code, different
axis label).

Ported from the original `/Users/wxy/Desktop/code/DT_operators.py` `cross_*`
family (~30 functions, each hand-rolled with a separate "full window"
vectorized branch and a Python-loop "expanding window" branch). Rebuilt on
`rolling_windows`/`rolling_reduce`/`rolling_pair_reduce`/`rolling_ols`,
which fold both branches into one NaN-safe pad-then-unfold pass — almost
every function below is a one- or two-line wrapper.

## Bugs fixed while porting (see the module-level docstrings on each
function below for the specific one; summarized here for a quick scan):

- **`cross_product` name collision (HIGH SEVERITY).** The original file
  defines `cross_product` TWICE: once (~line 539) as the correct day-axis
  rolling product, and again (~line 1908) as an unrelated row-wise
  sliding-window product along the *minute* axis with signature
  `cross_product(tensor, n)`. Python silently keeps only the second
  definition, so the first (the one actually implied by the `cross_`
  prefix and this module's day-axis convention) was DEAD CODE — nothing
  in the original file could ever call it. `cross_product` here is the
  correct day-axis rolling product. The second (minute-axis) definition
  is really an `in_*`-style operator mislabeled with a `cross_` name; it
  belongs in `dt_in.py`, not here (flagged for that file, not
  reimplemented in this one — no point duplicating it).
- **`cross_rank`'s shape-destroying bug.** The original did
  `x_valid = x[mask]` with a 2-D boolean mask, which FLATTENS to 1-D —
  splicing values across columns before windowing — whenever any NaN is
  present anywhere in `x`. `rolling_windows` + `nanrank_last` never
  index with a 2-D bool mask; they operate on the full `(D, M)` tensor
  throughout and let the NaN-aware reducer skip invalid entries
  per-window, so column independence is preserved even with 5%+ NaNs.
- **`cross_multireg_core`'s broken exception fallback** (used by
  `in_multireg`, not by anything here) called
  `zip(['alpha','beta','tvalue','fvalue','resid'])` with a single
  iterable — `ValueError` on the very first genuine failure. Not
  reproduced anywhere in this file's multi-predictor family (see
  `cross_multireg` below, which uses a plain per-predictor loop with no
  such fallback).
- **`cross_zscore_rolling`'s axis-convention violation.** Named and
  registered like a `cross_*` (day-axis) op, but it actually transposes
  and rolls along the MINUTE axis internally (verified by tracing the
  `unfold` call through the transpose — the window dim is built from the
  padded minute axis, not the day axis). It has been moved to `dt_in.py`
  as `in_zscore_rolling`, NOT reimplemented here under its old,
  misleading name.
- **`_max`/`_min`'s broken fallback**: `except Exception: return
  cross_max(x, y)` passed a *tensor* `y` where `cross_max(x, d: int)`
  expects an integer window — nonsensical, and `cross_max`/`cross_min`
  were never a substitute for elementwise `torch.maximum`/`torch.minimum`
  in the first place. Not ported at all: elementwise max/min needs
  nothing beyond `torch.maximum`/`torch.minimum` directly (already
  exception-free), so adding a `cross_`-namespaced wrapper here would
  just be a distraction.

## Deliberate simplification: multi-predictor regression family

The original `cross_multireg` (dict of alpha/beta/tvalue/fvalue/resid) is
a genuine *joint* multivariate rolling OLS via windowed normal equations
(`torch.linalg.solve`/`pinv` on a `(D, d, k+1)` design matrix per day).
`cross_multiregbeta`/`cross_multiregres` (defined separately, further down
in the same original file) instead treat each predictor with its OWN
INDEPENDENT single-predictor regression against `y` — a different,
simpler model. Rather than reimplement the joint solve (which would need
a new k-predictor rolling-OLS primitive in `common.py`, well beyond a
"rare, genuinely needed" shared helper), `cross_multireg` here is
rebuilt the same way `cross_multiregbeta`/`cross_multiregres` already
worked: a small Python loop (O(k) predictors, not O(D) days) delegating
each predictor to `rolling_ols`. This is a real semantics change from the
original `cross_multireg` (independent per-predictor coefficients, not
joint/partialled-out ones) — flagged clearly here and in the port report.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .common import (
    EPS,
    decay_linear_last,
    get_nan_tensor,
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
    rolling_ols,
    rolling_pair_reduce,
    rolling_reduce,
    rolling_windows,
    scale_last_ratio,
)

__all__ = [
    "cross_delay",
    "cross_shift",
    "cross_delta",
    "cross_cov",
    "cross_corr",
    "cross_autocorr",
    "cross_quantile",
    "cross_sum",
    "cross_mean",
    "cross_min",
    "cross_max",
    "cross_med",
    "cross_argmin",
    "cross_argmax",
    "cross_rank",
    "cross_scale",
    "cross_product",
    "cross_decay_linear",
    "cross_stddev",
    "cross_regbeta",
    "cross_regalpha",
    "cross_regres",
    "cross_reg_tvalue",
    "cross_reg_fvalue",
    "cross_multireg",
    "cross_multiregbeta",
    "cross_multiregres",
    "cross_pooled_beta",
    "cross_pooled_alpha",
    "cross_pooled_resid1",
    "cross_outlier",
    "cross_hurst",
    "cross_sustain",
    "cross_fangzheng",
]


# --------------------------------------------------------------------------
# lag / shift / diff (plain day-axis indexing, no windowing needed)
# --------------------------------------------------------------------------

@op("dt", "cross", arity=1, requires_window=True)
def cross_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    """`x` shifted DOWN (later days see earlier values) by `d` days."""
    d = int(d)
    res = get_nan_tensor(x.shape, x.device, x.dtype)
    if 0 < d < x.shape[0]:
        res[d:] = x[:-d]
    elif d == 0:
        res = x.clone()
    return res


@op("dt", "cross", arity=1, requires_window=True)
def cross_shift(x: torch.Tensor, d: int) -> torch.Tensor:
    """`x` shifted UP (earlier days see later values) by `d` days."""
    d = int(d)
    res = get_nan_tensor(x.shape, x.device, x.dtype)
    if 0 < d < x.shape[0]:
        res[:-d] = x[d:]
    elif d == 0:
        res = x.clone()
    return res


@op("dt", "cross", arity=1, requires_window=True)
def cross_delta(x: torch.Tensor, d: int) -> torch.Tensor:
    """`x[t] - x[t-d]` along the day axis."""
    d = int(d)
    res = get_nan_tensor(x.shape, x.device, x.dtype)
    if 0 < d < x.shape[0]:
        res[d:] = x[d:] - x[:-d]
    return res


# --------------------------------------------------------------------------
# pair-rolling cov / corr / autocorr
# --------------------------------------------------------------------------

@op("dt", "cross", arity=2, requires_window=True, pair=True)
def cross_cov(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_pair_reduce(x, y, int(d), "cov", axis=0)


@op("dt", "cross", arity=2, requires_window=True, pair=True)
def cross_corr(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_pair_reduce(x, y, int(d), "corr", axis=0)


@op("dt", "cross", arity=1, requires_window=True)
def cross_autocorr(x: torch.Tensor, d: int, shift: int) -> torch.Tensor:
    """Rolling correlation of `x` against itself lagged by `shift` days."""
    shift = int(shift)
    res = get_nan_tensor(x.shape, x.device, x.dtype)
    if shift < x.shape[0]:
        res[shift:] = cross_corr(x[shift:], x[:-shift], d)
    return res


# --------------------------------------------------------------------------
# rolling distribution / order statistics
# --------------------------------------------------------------------------

@op("dt", "cross", arity=1, requires_window=True)
def cross_quantile(x: torch.Tensor, q: float, d: int) -> torch.Tensor:
    w = rolling_windows(x, int(d), axis=0)
    return torch.nanquantile(w, q, dim=-1)


@op("dt", "cross", arity=1, requires_window=True)
def cross_sum(x: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_reduce(x, int(d), nansum_last, axis=0)


@op("dt", "cross", arity=1, requires_window=True)
def cross_mean(x: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_reduce(x, int(d), nanmean_last, axis=0)


@op("dt", "cross", arity=1, requires_window=True)
def cross_min(x: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_reduce(x, int(d), nanmin_last, axis=0)


@op("dt", "cross", arity=1, requires_window=True)
def cross_max(x: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_reduce(x, int(d), nanmax_last, axis=0)


@op("dt", "cross", arity=1, requires_window=True)
def cross_med(x: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_reduce(x, int(d), nanmedian_last, axis=0)


@op("dt", "cross", arity=1, requires_window=True)
def cross_argmin(x: torch.Tensor, d: int, pct: bool = True) -> torch.Tensor:
    return rolling_reduce(x, int(d), lambda w: nanargmin_last(w, pct=pct), axis=0)


@op("dt", "cross", arity=1, requires_window=True)
def cross_argmax(x: torch.Tensor, d: int, pct: bool = True) -> torch.Tensor:
    return rolling_reduce(x, int(d), lambda w: nanargmax_last(w, pct=pct), axis=0)


@op("dt", "cross", arity=1, requires_window=True)
def cross_rank(x: torch.Tensor, d: int, pct: bool = True) -> torch.Tensor:
    """Rolling rank of the current day's value within the trailing `d`-day
    window. FIXES the original's `x[mask]` (2-D boolean indexing) bug,
    which flattened the whole `(D, M)` tensor to 1-D whenever any NaN was
    present anywhere in `x`, splicing values across columns before
    windowing (see module docstring). Here `rolling_windows`/
    `nanrank_last` operate on the full 2-D tensor throughout — no
    flattening, so column `j`'s ranks never depend on column `k != j`'s
    values.
    """
    return rolling_reduce(x, int(d), lambda w: nanrank_last(w, pct=pct), axis=0)


@op("dt", "cross", arity=1, requires_window=True)
def cross_scale(x: torch.Tensor, d: int, constant: float = 0.0) -> torch.Tensor:
    w = rolling_windows(x, int(d), axis=0)
    return scale_last_ratio(x, w) + constant


@op("dt", "cross", arity=1, requires_window=True)
def cross_product(x: torch.Tensor, d: int) -> torch.Tensor:
    """Rolling `d`-day product, NaN treated as the multiplicative identity.

    This is the CORRECT day-axis semantics for the name `cross_product`
    (see module docstring for the name-collision bug this fixes: the
    original file's second, minute-axis `cross_product(tensor, n)`
    definition silently shadowed this one).
    """
    return rolling_reduce(x, int(d), nanproduct_last, axis=0)


@op("dt", "cross", arity=1, requires_window=True)
def cross_decay_linear(x: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_reduce(x, int(d), decay_linear_last, axis=0)


@op("dt", "cross", arity=1, requires_window=True)
def cross_stddev(x: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_reduce(x, int(d), nanstd_last, axis=0)


# --------------------------------------------------------------------------
# rolling single-predictor OLS (beta/alpha/resid/tvalue/fvalue in one call)
# --------------------------------------------------------------------------

@op("dt", "cross", arity=2, requires_window=True, pair=True)
def cross_regbeta(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_ols(x, y, int(d), axis=0)["beta"]


@op("dt", "cross", arity=2, requires_window=True, pair=True)
def cross_regalpha(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_ols(x, y, int(d), axis=0)["alpha"]


@op("dt", "cross", arity=2, requires_window=True, pair=True)
def cross_regres(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_ols(x, y, int(d), axis=0)["resid"]


@op("dt", "cross", arity=2, requires_window=True, pair=True)
def cross_reg_tvalue(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_ols(x, y, int(d), axis=0)["tvalue"]


@op("dt", "cross", arity=2, requires_window=True, pair=True)
def cross_reg_fvalue(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_ols(x, y, int(d), axis=0)["fvalue"]


# --------------------------------------------------------------------------
# multi-predictor family — independent per-predictor loop (see module
# docstring for why this deliberately does NOT reproduce the original
# joint-multivariate `cross_multireg`'s matrix solve)
# --------------------------------------------------------------------------

@op("dt", "cross", arity=1, requires_window=True)
def cross_multiregbeta(x_list: List[torch.Tensor], y: torch.Tensor, d: int) -> List[torch.Tensor]:
    """Independent rolling beta of `y` against each predictor in `x_list`
    (each predictor regressed on its own, NOT jointly/partialled-out).
    """
    d = int(d)
    return [rolling_ols(x, y, d, axis=0)["beta"] for x in x_list]


@op("dt", "cross", arity=1, requires_window=True)
def cross_multiregres(x_list: List[torch.Tensor], y: torch.Tensor, d: int) -> torch.Tensor:
    """`y` minus the sum of each predictor's own independent rolling
    `beta * x` contribution (matches the original's definition exactly).
    """
    betas = cross_multiregbeta(x_list, y, d)
    res = y
    for x, beta in zip(x_list, betas):
        res = res - beta * x
    return res


@op("dt", "cross", arity=1, requires_window=True)
def cross_multireg(x_list: List[torch.Tensor], y: torch.Tensor, d: int) -> Dict[str, object]:
    """Bundle of independent per-predictor rolling OLS results (see module
    docstring: a deliberate simplification of the original's joint
    multivariate solve). Returns `{'alpha': [...], 'beta': [...],
    'tvalue': [...], 'fvalue': [...], 'resid': Tensor}`, one alpha/beta/
    tvalue/fvalue per predictor (list, aligned with `x_list`) and a single
    combined residual (`y` minus every predictor's own contribution, via
    `cross_multiregres`).
    """
    d = int(d)
    results = [rolling_ols(x, y, d, axis=0) for x in x_list]
    return {
        "alpha": [r["alpha"] for r in results],
        "beta": [r["beta"] for r in results],
        "tvalue": [r["tvalue"] for r in results],
        "fvalue": [r["fvalue"] for r in results],
        "resid": cross_multiregres(x_list, y, d),
    }


# --------------------------------------------------------------------------
# pooled regression (concat two (x, y) samples, regress, slice back)
# --------------------------------------------------------------------------

@op("dt", "cross", arity=4, requires_window=True)
def cross_pooled_beta(
    x1: torch.Tensor, x2: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor, d: int
) -> torch.Tensor:
    T = x1.shape[0]
    x_full = torch.cat([x1, x2], dim=0)
    y_full = torch.cat([y1, y2], dim=0)
    return cross_regbeta(x_full, y_full, d)[:T]


@op("dt", "cross", arity=4, requires_window=True)
def cross_pooled_alpha(
    x1: torch.Tensor, x2: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor, d: int
) -> torch.Tensor:
    T = x1.shape[0]
    x_full = torch.cat([x1, x2], dim=0)
    y_full = torch.cat([y1, y2], dim=0)
    return cross_regalpha(x_full, y_full, d)[:T]


@op("dt", "cross", arity=4, requires_window=True)
def cross_pooled_resid1(
    x1: torch.Tensor, x2: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor, d: int
) -> torch.Tensor:
    T = x1.shape[0]
    x_full = torch.cat([x1, x2], dim=0)
    y_full = torch.cat([y1, y2], dim=0)
    beta = cross_regbeta(x_full, y_full, d)[:T]
    alpha = cross_regalpha(x_full, y_full, d)[:T]
    return y1 - (alpha + beta * x1)


# --------------------------------------------------------------------------
# misc rolling stats
# --------------------------------------------------------------------------

@op("dt", "cross", arity=1, requires_window=True)
def cross_outlier(x: torch.Tensor, d: int, n: float, m: float) -> torch.Tensor:
    """Flag `+1`/`-1`/`0` for above/below/within a rolling mean+`m`*std
    threshold, where the mean/std are computed only from the bottom `n`%
    (by rolling rank) of each window.

    REWORKED from the original: the original computed `mean`/`std` via
    this file's row-reducing `_mean`/`_stddev` (`dim=-1`, i.e. across
    MINUTES) applied to a `(D, M)` selection, then broadcast a `(D,)`
    threshold against the full `(D, M)` tensor — a shape mismatch that
    raises whenever `M != D` (found while porting; not one of the
    pre-confirmed bugs, but real). Reworked here to compute the
    mean/std from each day's own rolling `d`-day window instead, which
    is shape-safe by construction and matches the "outlier relative to
    its trailing window" intent implied by `cross_`'s day-axis semantics.
    """
    d = int(d)
    rank_pct = cross_rank(x, d, pct=True)
    mask = rank_pct < (n / 100.0)
    selected = torch.where(mask, x, torch.full_like(x, float("nan")))
    sw = rolling_windows(selected, d, axis=0)
    mean = nanmean_last(sw)
    std = nanstd_last(sw)
    threshold = mean + m * std
    return torch.where(
        x > threshold,
        torch.ones_like(x),
        torch.where(x < threshold, -torch.ones_like(x), torch.zeros_like(x)),
    )


@op("dt", "cross", arity=1, requires_window=True)
def cross_hurst(x: torch.Tensor, d: int) -> torch.Tensor:
    """Rolling `d`-day Hurst exponent (rescaled-range method). Matches the
    original's behavior of leaving the first `d-1` days as NaN (the
    original never had an expanding-window branch for Hurst at all —
    unlike its other `cross_*` siblings — since R/S over a tiny sample is
    not meaningful; preserved intentionally rather than "fixed" into
    computing something noisy).
    """
    d = int(d)
    w = rolling_windows(x, d, axis=0)
    mean = w.nanmean(dim=-1, keepdim=True)
    demean = w - mean
    cs = demean.cumsum(dim=-1)
    valid_cs = ~torch.isnan(cs)
    r_max = torch.where(valid_cs, cs, torch.full_like(cs, float("-inf"))).max(dim=-1).values
    r_min = torch.where(valid_cs, cs, torch.full_like(cs, float("inf"))).min(dim=-1).values
    R = r_max - r_min
    S = torch.sqrt(torch.nansum(demean.pow(2), dim=-1) / max(d - 1, 1))
    RS = R / S
    hurst = torch.log(RS) / torch.log(torch.tensor(float(d), device=x.device, dtype=x.dtype))
    hurst = torch.where(S < 1e-9, torch.full_like(hurst, float("nan")), hurst)
    if d > 1:
        hurst[: d - 1] = float("nan")
    return hurst


@op("dt", "cross", arity=1, requires_window=True)
def cross_sustain(x: torch.Tensor, d: int, num: bool = False) -> torch.Tensor:
    """Propagate every `1` in `x` DOWN `d` rows (days), inclusive of
    itself. `num=True` fills untouched cells with `0`, else `NaN`. Fully
    vectorized scatter (no Python loop over rows/days) — ported near
    verbatim, since it's a scatter/mask op rather than a windowed
    reduction and there's no `rolling_*` primitive that fits it.
    """
    d = int(d)
    if d <= 0:
        raise ValueError("d must be a positive integer")
    N, T = x.shape
    ones = x == 1
    row_idx, col_idx = torch.where(ones)
    fill = torch.zeros((N, T), dtype=x.dtype, device=x.device) if num else get_nan_tensor((N, T), x.device, x.dtype)
    if row_idx.numel() == 0:
        return fill
    k = torch.arange(d, device=x.device)
    row_expanded = row_idx.unsqueeze(1) + k
    col_expanded = col_idx.unsqueeze(1).expand_as(row_expanded)
    valid_mask = (row_idx.unsqueeze(1) + k) < N
    target_rows = row_expanded[valid_mask]
    target_cols = col_expanded[valid_mask]
    mask = torch.zeros(N, T, dtype=torch.bool, device=x.device)
    mask[target_rows, target_cols] = True
    fill[mask] = 1.0
    return fill


@op("dt", "cross", arity=1, requires_window=True)
def cross_fangzheng(x: torch.Tensor, d: int) -> torch.Tensor:
    return cross_mean(x, d) + cross_stddev(x, d)
