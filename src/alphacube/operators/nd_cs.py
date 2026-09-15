"""Cross-sectional ("cs") ND operators.

ND tensors are shape `(T, N)` = (time, stock). Every operator here reduces
or ranks along `dim=-1` (across stocks, at a fixed time).

Ported from the original `/Users/wxy/Desktop/code/ND_operators.py` `cs_*`
family, rebuilt on top of `operators.common`:

- `cs_rank`/`indneutralize` are not reimplemented at all — they are
  byte-identical (module-level-global aside) to `common.rank`/
  `common.indneutralize`, so we just import and re-export them.
- `cs_mean`/`cs_sum`/`cs_median` reuse `common._mean`/`_sum`/`_median`
  as the reduction core (masking invalid entries to NaN first, then
  broadcasting the reduced row statistic back out to the input shape),
  rather than re-deriving nanmean/nansum/nanmedian by hand.
- Every function accepts an optional `valid_mask` that defaults to
  `torch.isfinite(x)` (the old module-level `tensor_mask` global was
  `None`, so every function that fell back to it crashed the instant no
  explicit mask was supplied — see the `tensor_mask` bug note below).
- `cs_sum`'s old `yes: bool` flag (which, on the `yes=False` default
  path, silently *ignored* any `valid_mask` the caller passed in) has
  been removed. `valid_mask` is now always honored.
- `cs_mask_top_replace_mean`'s `for i in range(N)` Python loop (looping
  over ROWS/time steps despite the variable being confusingly named `N`
  in the original) is replaced by a fully vectorized rank-and-threshold.
"""

from __future__ import annotations

from typing import Optional

import torch

from .common import (
    EPS,
    _mean,
    _median,
    _sum,
    indneutralize,
    op,
    rank as _common_rank,
)

__all__ = [
    "cs_rank",
    "rank",
    "indneutralize",
    "cs_scale",
    "cs_mean",
    "cs_demean",
    "cs_sum",
    "cs_median",
    "cs_quantile",
    "cs_stddev_full",
    "cs_mask_top_replace_mean",
    "cs_mask_top_replace_mean_neg",
    "cs_regres",
    "cs_normalize",
]


def _valid(x: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Resolve the effective valid mask: `isfinite(x)`, ANDed with the
    caller's `valid_mask` if one was given. Replaces the old crash-prone
    `if valid_mask is None: valid_mask = tensor_mask` (`tensor_mask` was a
    module-level global hardcoded to `None` in the original file, so
    ~11 functions there raised on the very first call unless a caller
    happened to pass an explicit mask).
    """
    finite = torch.isfinite(x)
    return finite if valid_mask is None else (finite & valid_mask)


# --------------------------------------------------------------------------
# rank / industry-neutralize — identical to common.py, just re-exported
# --------------------------------------------------------------------------

# The original file defined `rank`/`cs_rank` twice with byte-identical
# bodies. `common.rank` is the one canonical implementation; `cs_rank` is
# kept as the primary Alpha101-style exported name (that's the name this
# family is normally called by), with `rank` re-exported too so nothing
# that imported the old bare name breaks.
@op("nd", "cs", arity=1)
def cs_rank(x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None, pct: bool = True) -> torch.Tensor:
    return _common_rank(x, valid_mask=valid_mask, pct=pct)


rank = _common_rank
# indneutralize is imported directly above and re-exported as-is (already
# registered under its own name by common.py's @op decorator).


# --------------------------------------------------------------------------
# scale / mean / demean / sum / median
# --------------------------------------------------------------------------

@op("nd", "cs", arity=1)
def cs_scale(x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None, constant: float = 0.0) -> torch.Tensor:
    mask = _valid(x, valid_mask)
    x_safe = torch.where(mask, x, torch.zeros_like(x))
    x_sum = x_safe.sum(dim=-1, keepdim=True)
    x_sum = torch.where(x_sum.abs() < EPS, torch.ones_like(x_sum), x_sum)
    return x / x_sum + constant


@op("nd", "cs", arity=1)
def cs_mean(x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    mask = _valid(x, valid_mask)
    masked = torch.where(mask, x, torch.full_like(x, float("nan")))
    return _mean(masked).unsqueeze(-1).expand_as(x)


@op("nd", "cs", arity=1)
def cs_demean(x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    return x - cs_mean(x, valid_mask)


@op("nd", "cs", arity=1)
def cs_sum(x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Row-wise (cross-sectional) sum, broadcast back to input shape.

    The original had a `yes: bool` flag: `yes=False` (the default) threw
    away any `valid_mask` argument entirely and summed over *every*
    finite value with no way to restrict to a universe; `yes=True` was
    needed to get masking at all. That flag is gone — `valid_mask` (when
    given) is now always honored, same as every other `cs_*` function.
    """
    mask = _valid(x, valid_mask)
    masked = torch.where(mask, x, torch.full_like(x, float("nan")))
    return _sum(masked).unsqueeze(-1).expand_as(x)


@op("nd", "cs", arity=1)
def cs_median(x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    mask = _valid(x, valid_mask)
    masked = torch.where(mask, x, torch.full_like(x, float("nan")))
    med = _median(masked)
    count = mask.sum(dim=-1)
    med = torch.where(count > 0, med, torch.full_like(med, float("nan")))
    return med.unsqueeze(-1).expand_as(x)


# --------------------------------------------------------------------------
# quantile-band flag
# --------------------------------------------------------------------------

@op("nd", "cs", arity=1)
def cs_quantile(
    x: torch.Tensor,
    quantile: float,
    valid_mask: Optional[torch.Tensor] = None,
    return_bool: bool = True,
) -> torch.Tensor:
    """Flag entries in the top `quantile` band of each row (e.g.
    `quantile=0.2` flags the top 20% by value per row).

    Bug fix vs. the original: when `return_bool=True` the original did
    `flags[~mask] = torch.nan` on a *bool* tensor. Bool tensors can't
    represent NaN — PyTorch casts the NaN scalar to `True` (Python
    `bool(float('nan'))` is `True`), so every masked-out/invalid entry
    silently flipped to "flagged" instead of being excluded. Since
    `in_quantile = mask & (x >= q_val)` is already `False` wherever
    `mask` is `False`, that assignment was both wrong and redundant —
    it's simply dropped here for the bool path. The float path
    (`return_bool=False`) still NaNs out invalid entries, which bool
    dtype cannot do.
    """
    if not (0.0 <= quantile <= 1.0):
        raise ValueError("quantile must be in [0, 1]")
    mask = _valid(x, valid_mask)
    x_masked = torch.where(mask, x, torch.full_like(x, float("nan")))
    q_val = torch.nanquantile(x_masked, q=1.0 - quantile, dim=-1, keepdim=True)
    in_quantile = mask & (x >= q_val)
    if return_bool:
        return in_quantile
    flags = torch.where(in_quantile, torch.ones_like(x), torch.zeros_like(x))
    return torch.where(mask, flags, torch.full_like(flags, float("nan")))


# --------------------------------------------------------------------------
# stddev (sample, ddof=1)
# --------------------------------------------------------------------------

@op("nd", "cs", arity=1)
def cs_stddev_full(x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Row-wise SAMPLE stddev (ddof=1, i.e. divide by count-1), broadcast
    back to input shape. Kept as sample stddev (not delegated to
    `common._stddev`, which is a *population* stddev / ddof=0 reducer —
    the two aren't the same statistic, so reusing `_stddev` here would
    silently change the numeric result) but still fully vectorized.
    """
    mask = _valid(x, valid_mask)
    x_safe = torch.where(mask, x, torch.zeros_like(x))
    count = mask.sum(dim=-1, keepdim=True).to(x.dtype)
    mean_x = torch.where(count > 0, x_safe.sum(dim=-1, keepdim=True) / count, torch.zeros_like(count))
    sq_diff = torch.where(mask, (x_safe - mean_x) ** 2, torch.zeros_like(x))
    sum_sq = sq_diff.sum(dim=-1, keepdim=True)
    var = torch.where(count > 1, sum_sq / (count - 1), torch.full_like(count, float("nan")))
    std = torch.sqrt(var.clamp(min=0))
    std = torch.where(count > 1, std, torch.full_like(std, float("nan")))
    return std.expand_as(x)


# --------------------------------------------------------------------------
# top-band winsorize-to-mean (vectorized)
# --------------------------------------------------------------------------

@op("nd", "cs", arity=1)
def cs_mask_top_replace_mean(
    x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None, top_ratio: float = 0.1
) -> torch.Tensor:
    """Replace the top `top_ratio` fraction (by value, per row) of each
    row with that row's mean.

    The original implementation used `torch.topk` to find candidate top
    indices and then a Python `for i in range(N)` loop (`N` there is
    actually the number of ROWS/time steps, not stocks — a misleading
    variable name) to build the final replace-mask row by row. Replaced
    here with: rank each row (ascending, valid-aware, via `common.rank`
    with `pct=False`), threshold by `top_ratio`, `torch.where` to
    replace — one vectorized pass, no loop.
    """
    assert 0.0 <= top_ratio <= 1.0
    mask = _valid(x, valid_mask)

    x_masked = torch.where(mask, x, torch.full_like(x, float("nan")))
    row_means = _mean(x_masked).unsqueeze(-1).expand_as(x)

    count = mask.sum(dim=-1, keepdim=True).to(x.dtype)
    n_top = torch.ceil(count * top_ratio)
    integer_rank = _common_rank(x, valid_mask=mask, pct=False)  # ascending, NaN where invalid
    # Top `n_top` largest values per row <=> rank strictly greater than
    # (count - n_top). NaN ranks (invalid entries) never satisfy this.
    top_mask = mask & (integer_rank > (count - n_top))

    return torch.where(top_mask, row_means, x)


@op("nd", "cs", arity=1)
def cs_mask_top_replace_mean_neg(
    x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None, top_ratio: float = 0.1
) -> torch.Tensor:
    """Bottom-band counterpart: replace the *lowest* `top_ratio` fraction
    of each row with that row's mean, via the negate-flip-negate trick.
    """
    return -cs_mask_top_replace_mean(-x, valid_mask, top_ratio)


# --------------------------------------------------------------------------
# cross-sectional regression residual
# --------------------------------------------------------------------------

@op("nd", "cs", arity=2)
def cs_regres(Y: torch.Tensor, X: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Cross-sectional (per-row, across stocks) OLS residual of
    `Y ~ alpha + beta*X`, one independent regression per time step.
    """
    mask = torch.isfinite(X) & torch.isfinite(Y)
    if valid_mask is not None:
        mask = mask & valid_mask

    X_ = torch.where(mask, X, torch.zeros_like(X))
    Y_ = torch.where(mask, Y, torch.zeros_like(Y))

    n_valid = mask.sum(dim=-1, keepdim=True).to(X.dtype)
    x_mean = X_.sum(dim=-1, keepdim=True) / n_valid.clamp(min=1)
    y_mean = Y_.sum(dim=-1, keepdim=True) / n_valid.clamp(min=1)

    x_centered = torch.where(mask, X_ - x_mean, torch.zeros_like(X))
    y_centered = torch.where(mask, Y_ - y_mean, torch.zeros_like(Y))

    cov = (x_centered * y_centered).sum(dim=-1, keepdim=True)
    var = (x_centered * x_centered).sum(dim=-1, keepdim=True)
    beta = torch.where(var > EPS, cov / var, torch.zeros_like(var))

    y_pred = beta * (X - x_mean) + y_mean
    residuals = Y - y_pred
    return torch.where(mask, residuals, torch.full_like(residuals, float("nan")))


# --------------------------------------------------------------------------
# row z-score normalize
# --------------------------------------------------------------------------

@op("nd", "cs", arity=1)
def cs_normalize(x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Row-wise (cross-sectional) z-score: `(x - row_mean) / row_std`
    (sample std, ddof=1). Rows with a single valid entry get 0 (a
    lone point has no meaningful spread); rows with a zero-variance
    (constant) set of valid entries also get 0 instead of a 0/0 NaN.
    """
    mask = _valid(x, valid_mask)

    x_safe = torch.where(mask, x, torch.zeros_like(x))
    count = mask.sum(dim=-1, keepdim=True).to(x.dtype)
    count_safe = torch.where(count > 0, count, torch.ones_like(count))
    mean = x_safe.sum(dim=-1, keepdim=True) / count_safe

    x_centered = torch.where(mask, x - mean, torch.zeros_like(x))
    sum_sq = (x_centered ** 2).sum(dim=-1, keepdim=True)
    var = torch.where(count > 1, sum_sq / (count - 1), torch.full_like(count, float("nan")))
    std = torch.sqrt(var.clamp(min=0.0))

    z = x_centered / std

    out = torch.full_like(x, float("nan"))
    ge2 = count > 1
    std_gt0 = std > 0
    std_eq0 = std <= 0
    out = torch.where(mask & ge2 & std_gt0, z, out)
    out = torch.where(mask & ge2 & std_eq0, torch.zeros_like(x), out)
    out = torch.where(mask & (count == 1), torch.zeros_like(x), out)
    return out
