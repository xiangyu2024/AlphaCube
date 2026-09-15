"""Shared elementwise ops + generic vectorized rolling-window primitives.

Single source of truth for everything that was duplicated (often with subtle
divergences) across the old `ND_operators.py` and `DT_operators.py`.

## Tensor conventions

- **ND** tensors: shape `(T, N)` = (time/day, stock). Cross-sectional ("cs")
  ops reduce/rank along `dim=-1` (N, across stocks at a fixed time).
  Time-series ("ts") ops roll along `dim=0` (T, across time for each stock).
- **DT** tensors: shape `(D, M)` = (day, intraday-minute), typically one
  symbol at a time. "cross" ops roll along `dim=0` (D, across days).
  "in" (inday) ops roll/reduce along `dim=-1` (M, across minutes within a
  day). "day" ops collapse `dim=-1` (M) to one value per day, shape `(D,)`
  or `(D, 1)`.

## The core trick: pad-then-unfold

Every rolling-window operator in the old files had two separately-written
code paths — a vectorized "full window" branch (valid once `t >= d-1`) and a
hand-written Python-loop "expanding window" branch for the first `d-1`
positions — and in several operators (`ts_mean`, `ts_corr`, `ts_stddev`,
the `*_reg*` family, ...) the two branches used inconsistent NaN handling,
and in several others (`ts_min`, `ts_max`, `ts_med`, `ts_argmin`,
`ts_argmax`, `ts_rank`, and their `cross_*`/`in_*` counterparts) the loop
branch had an outright bug (missing `dim=0`, reducing across the wrong
axis).

`rolling_windows` below eliminates the split entirely: left-pad the series
with `d-1` NaNs along the rolling axis, then a single `unfold` call produces
one window per original timestep, where the earliest windows are simply
"padded" with NaNs that every nan-aware reducer below ignores. One code
path, no Python loop, no branch-mismatch bugs.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Union

import torch

EPS = 1e-9

Number = Union[int, float]


# --------------------------------------------------------------------------
# lightweight operator-metadata registry (feeds the AlphaFactorMining
# OperatorRegistry / configs/operators.yaml generation step)
# --------------------------------------------------------------------------

OPERATOR_REGISTRY: Dict[str, dict] = {}


def op(mode: str, category: str, arity: int = 1, requires_window: bool = False, pair: bool = False):
    """Tag a function with taxonomy metadata and register it.

    mode: "nd" | "dt" | "shared" (shared = elementwise/utility, valid in both)
    category: "cs" | "ts" (nd)  |  "in" | "cross" | "day" (dt)  |  "elementwise"
    arity: number of Expression-tensor operands (excludes window/scalar params)
    requires_window: whether the function takes an integer window `d`
    pair: whether it's a 2-tensor-operand (pair-rolling) op, e.g. cov/corr
    """
    def deco(fn):
        OPERATOR_REGISTRY[fn.__name__] = dict(
            mode=mode, category=category, arity=arity,
            requires_window=requires_window, pair=pair, fn=fn,
        )
        return fn
    return deco


# --------------------------------------------------------------------------
# tensor construction helpers
# --------------------------------------------------------------------------

def get_nan_tensor(shape, device, dtype) -> torch.Tensor:
    return torch.full(shape, float("nan"), dtype=dtype, device=device)


def nan_tensor(x: torch.Tensor) -> torch.Tensor:
    """Deprecated alias kept for backward compatibility."""
    return get_nan_tensor(x.shape, x.device, x.dtype)


def create_like(input_tensor: torch.Tensor, target, dtype=None, device=None) -> torch.Tensor:
    device = device or input_tensor.device
    if dtype is None:
        if isinstance(target, torch.Tensor):
            dtype = target.dtype
        elif isinstance(target, bool):
            dtype = torch.bool
        elif isinstance(target, int):
            dtype = torch.int64
        elif isinstance(target, float) or (
            isinstance(target, (int, float)) and (math.isinf(target) or math.isnan(target))
        ):
            dtype = torch.float32
        else:
            dtype = input_tensor.dtype

    if isinstance(target, float) and math.isnan(target):
        value = torch.tensor(float("nan"), dtype=dtype, device=device)
    elif isinstance(target, float) and math.isinf(target):
        value = torch.tensor(
            float("inf") if target > 0 else float("-inf"), dtype=dtype, device=device
        )
    else:
        try:
            value = torch.tensor(target, dtype=dtype, device=device)
        except Exception as exc:
            raise ValueError(f"无法将目标值 {target} 转换为类型 {dtype}") from exc

    return torch.full(
        size=input_tensor.shape, fill_value=value.item(), dtype=dtype, device=device,
        layout=input_tensor.layout,
    )


def _as_tensor_operand(x, like: torch.Tensor) -> torch.Tensor:
    return create_like(like, x) if isinstance(x, (int, float)) else x


# --------------------------------------------------------------------------
# elementwise arithmetic / logic (single canonical version)
# --------------------------------------------------------------------------

@op("shared", "elementwise", arity=2)
def add(a, b, fill_nan_with_zero: bool = False) -> torch.Tensor:
    """Elementwise add. `fill_nan_with_zero=True` treats a lone NaN operand as
    0 (both-NaN still yields NaN); default is standard NaN-propagating add.
    """
    if isinstance(a, (int, float)) and isinstance(b, torch.Tensor):
        a = create_like(b, a)
    elif isinstance(b, (int, float)) and isinstance(a, torch.Tensor):
        b = create_like(a, b)
    if not fill_nan_with_zero:
        return a + b
    a_nan, b_nan = torch.isnan(a), torch.isnan(b)
    result = torch.where(a_nan, torch.zeros_like(a), a) + torch.where(b_nan, torch.zeros_like(b), b)
    return torch.where(a_nan & b_nan, torch.full_like(result, float("nan")), result)


@op("shared", "elementwise", arity=2)
def sub(a, b, fill_nan_with_zero: bool = False) -> torch.Tensor:
    if isinstance(a, (int, float)) and isinstance(b, torch.Tensor):
        a = create_like(b, a)
    elif isinstance(b, (int, float)) and isinstance(a, torch.Tensor):
        b = create_like(a, b)
    if not fill_nan_with_zero:
        return a - b
    a_nan, b_nan = torch.isnan(a), torch.isnan(b)
    result = torch.where(a_nan, torch.zeros_like(a), a) - torch.where(b_nan, torch.zeros_like(b), b)
    return torch.where(a_nan & b_nan, torch.full_like(result, float("nan")), result)


@op("shared", "elementwise", arity=2)
def mul(x, y) -> torch.Tensor:
    if isinstance(x, (int, float)) and isinstance(y, torch.Tensor):
        x = create_like(y, x)
    elif isinstance(y, (int, float)) and isinstance(x, torch.Tensor):
        y = create_like(x, y)
    return torch.mul(x, y)


@op("shared", "elementwise", arity=2)
def div(x, y) -> torch.Tensor:
    if isinstance(x, (int, float)) and isinstance(y, torch.Tensor):
        x = create_like(y, x)
    elif isinstance(y, (int, float)) and isinstance(x, torch.Tensor):
        y = create_like(x, y)
    return torch.divide(x, y)


@op("shared", "elementwise", arity=2)
def eq(x, y) -> torch.Tensor:
    x = _as_tensor_operand(x, y) if isinstance(y, torch.Tensor) else x
    y = _as_tensor_operand(y, x) if isinstance(x, torch.Tensor) else y
    return torch.eq(x, y)


@op("shared", "elementwise", arity=1)
def inv(x: torch.Tensor) -> torch.Tensor:
    return torch.divide(1.0, x)


@op("shared", "elementwise", arity=1)
def abs_(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x)


@op("shared", "elementwise", arity=1)
def neg(x: torch.Tensor) -> torch.Tensor:
    return torch.negative(x)


@op("shared", "elementwise", arity=1)
def sqrt(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.abs(x))


@op("shared", "elementwise", arity=1)
def log(x: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.abs(x) > EPS, torch.log(torch.abs(x)), torch.full_like(x, float("nan")))


@op("shared", "elementwise", arity=1)
def pow_(x: torch.Tensor, e) -> torch.Tensor:
    """Signed power: `sign(x)*|x|^e` for `e<=1` (keeps sign through
    fractional powers), plain `x**e` otherwise. `e` may be a tensor
    (broadcasts) or a python scalar.
    """
    if isinstance(e, (int, float)):
        e = torch.tensor(e, dtype=x.dtype, device=x.device)
    x_b, e_b = torch.broadcast_tensors(x, e)
    return torch.where(e_b <= 1.0, torch.sign(x_b) * torch.pow(x_b.abs(), e_b), torch.pow(x_b, e_b))


@op("shared", "elementwise", arity=1)
def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


@op("shared", "elementwise", arity=1)
def sign(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x)


@op("shared", "elementwise", arity=1)
def signedpower(x: torch.Tensor, e: float = 2.0) -> torch.Tensor:
    return torch.sign(x) * torch.pow(torch.abs(x), e)


def fill_with_quantile(x: torch.Tensor, q: float) -> torch.Tensor:
    """Fills the whole tensor with its single GLOBAL nanquantile (not per-row)."""
    return torch.full_like(x, torch.nanquantile(x, q).item())


@op("shared", "elementwise", arity=1)
def is_gt(x: torch.Tensor, value, numeric: bool = True) -> torch.Tensor:
    result = x > value
    return result.int() if numeric else result


@op("shared", "elementwise", arity=1)
def is_lt(x: torch.Tensor, value, numeric: bool = True) -> torch.Tensor:
    result = x < value
    return result.int() if numeric else result


@op("shared", "elementwise", arity=2)
def bool_and(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return x1 & x2


@op("shared", "elementwise", arity=2)
def bool_or(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return x1 | x2


def _to_bool(x: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isnan(x), torch.zeros_like(x, dtype=torch.bool), x != 0)


@op("shared", "elementwise", arity=2)
def logical_and(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.logical_and(_to_bool(x), _to_bool(y))


@op("shared", "elementwise", arity=2)
def logical_or(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.logical_or(_to_bool(x), _to_bool(y))


@op("shared", "elementwise", arity=1)
def logical_not(x: torch.Tensor) -> torch.Tensor:
    return torch.logical_not(_to_bool(x))


@op("shared", "elementwise", arity=2)
def logical_xor(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.logical_xor(_to_bool(x), _to_bool(y))


def logical_and_n(*tensors: torch.Tensor) -> torch.Tensor:
    assert len(tensors) >= 2, "至少需要两个输入张量"
    result = tensors[0]
    for t in tensors[1:]:
        result = torch.logical_and(result, t)
    return result


@op("shared", "elementwise", arity=3)
def select_where(x: torch.Tensor, y, z) -> torch.Tensor:
    y = create_like(x, y) if isinstance(y, (int, float)) else y
    z = create_like(x, z) if isinstance(z, (int, float)) else z
    return torch.where(x.bool(), y, z)


def fillna(x: torch.Tensor, fill_value, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    res = torch.where(x.isfinite(), x, torch.full_like(x, fill_value) if not isinstance(fill_value, torch.Tensor) else fill_value)
    if valid_mask is not None:
        res = torch.where(valid_mask, res, torch.full_like(res, float("nan")))
    return res


# --------------------------------------------------------------------------
# masked cross-sectional rank / neutralize (canonical, NaN-safe)
# --------------------------------------------------------------------------

@op("nd", "cs", arity=1)
def rank(x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None, pct: bool = True) -> torch.Tensor:
    """Rank along the last dim (ascending), NaN-safe. `valid_mask` defaults
    to `isfinite(x)` — the old code's crash-prone reliance on a mutable
    module-level `tensor_mask` global has been removed.
    """
    if valid_mask is None:
        valid_mask = torch.isfinite(x)
    x_rank = x.clone()
    # Invalid entries are pushed ABOVE every valid value (not below) so they
    # land at the end of the sort and don't shift valid entries' ranks.
    x_rank[~valid_mask] = torch.finfo(x.dtype).max
    ranks = (torch.argsort(torch.argsort(x_rank, dim=-1, descending=False), dim=-1) + 1).to(x.dtype)
    if pct:
        valid_count = valid_mask.sum(dim=-1, keepdim=True).to(x.dtype)
        ranks = torch.where(valid_count > 0, ranks / valid_count, torch.full_like(ranks, float("nan")))
    ranks = torch.where(valid_mask, ranks, torch.full_like(ranks, float("nan")))
    return ranks


@op("nd", "cs", arity=2)
def indneutralize(
    features: torch.Tensor, inds: torch.Tensor, valid_mask: Optional[torch.Tensor] = None, eps: float = 1e-6
) -> torch.Tensor:
    """Industry Z-score neutralization, vectorized via `scatter_add_`.
    Invalid entries (per `valid_mask`, default `isfinite(features)`) are
    grouped into their own bucket so they can't pollute other industries'
    statistics, then re-masked to NaN in the output.
    """
    if valid_mask is None:
        valid_mask = torch.isfinite(features)
    device = features.device
    T, N = features.shape

    flat_features = features.reshape(-1).float()
    flat_inds = inds.reshape(-1).long()
    flat_valid = valid_mask.reshape(-1)

    adjusted_inds = torch.where(flat_valid, flat_inds, torch.full_like(flat_inds, -1))
    unique_inds, inverse_indices = torch.unique(adjusted_inds, return_inverse=True)
    inverse_indices = inverse_indices.long()

    num_unique = len(unique_inds)
    sum_per_ind = torch.zeros(num_unique, device=device, dtype=torch.float32)
    sq_sum_per_ind = torch.zeros(num_unique, device=device, dtype=torch.float32)
    count_per_ind = torch.zeros(num_unique, device=device, dtype=torch.float32)
    ones = torch.ones_like(flat_features)

    masked_features = torch.where(flat_valid, flat_features, torch.zeros_like(flat_features))
    sum_per_ind.scatter_add_(0, inverse_indices, masked_features)
    sq_sum_per_ind.scatter_add_(0, inverse_indices, masked_features ** 2)
    count_per_ind.scatter_add_(0, inverse_indices, torch.where(flat_valid, ones, torch.zeros_like(ones)))

    count_per_ind = torch.clamp(count_per_ind, min=1e-8)
    mean_per_ind = sum_per_ind / count_per_ind
    variance_per_ind = (sq_sum_per_ind / count_per_ind) - mean_per_ind ** 2
    std_per_ind = torch.sqrt(torch.clamp(variance_per_ind, min=eps))

    neutralized = (flat_features - mean_per_ind[inverse_indices]) / std_per_ind[inverse_indices]
    neutralized = torch.where(flat_valid, neutralized, torch.full_like(neutralized, float("nan")))
    return neutralized.reshape(T, N)


# --------------------------------------------------------------------------
# last-dim (cross-sectional / row) statistical reducers
# --------------------------------------------------------------------------

def _sum(x: torch.Tensor) -> torch.Tensor:
    value = x.nansum(dim=-1)
    return torch.where(
        torch.isfinite(x).any(dim=-1), value, torch.full_like(value, float("nan"))
    )


def _mean(x: torch.Tensor) -> torch.Tensor:
    return x.nanmean(dim=-1)


def _median(x: torch.Tensor) -> torch.Tensor:
    return x.nanmedian(dim=-1).values


def _stddev(x: torch.Tensor) -> torch.Tensor:
    x_demean = x - x.nanmean(dim=-1, keepdim=True)
    return torch.sqrt(torch.nanmean(torch.pow(x_demean, 2), dim=-1))


def _skew(x: torch.Tensor) -> torch.Tensor:
    x_demeaned = x - x.nanmean(dim=-1, keepdim=True)
    x_std = torch.sqrt(torch.nanmean(torch.pow(x_demeaned, 2), dim=-1, keepdim=True))
    zscore = torch.pow(div(x_demeaned, x_std), 3)
    return _mean(zscore)


def _kurt(x: torch.Tensor) -> torch.Tensor:
    x_demeaned = x - x.nanmean(dim=-1, keepdim=True)
    x_var = torch.nanmean(torch.pow(x_demeaned, 2), dim=-1, keepdim=True)
    x_4 = torch.pow(x_demeaned, 4)
    return div(_mean(x_4), torch.pow(_mean(x_var), 2)) - 3


def _qua(x: torch.Tensor, q: float) -> torch.Tensor:
    return div(x.nanquantile(q, dim=-1), x.nanmean(dim=-1))


# --------------------------------------------------------------------------
# THE core rolling-window primitive
# --------------------------------------------------------------------------

def rolling_windows(x: torch.Tensor, d: int, axis: int = 0) -> torch.Tensor:
    """Causal rolling windows of length `d` along `axis`, left-padded with
    NaN so every original position (including the first `d-1`) gets a
    window. Output shape = `x.shape` with a new trailing window-dim of
    size `d` appended (the `axis` dim's length is unchanged).
    """
    d = int(d)
    pad_shape = list(x.shape)
    pad_shape[axis] = d - 1
    pad = torch.full(pad_shape, float("nan"), dtype=x.dtype, device=x.device)
    x_padded = torch.cat([pad, x], dim=axis)
    return x_padded.unfold(axis, d, 1)


def nan_count(w: torch.Tensor) -> torch.Tensor:
    return (~torch.isnan(w)).sum(dim=-1)


def nanmin_last(w: torch.Tensor) -> torch.Tensor:
    filled = torch.where(torch.isnan(w), torch.full_like(w, float("inf")), w)
    out = filled.min(dim=-1).values
    return torch.where(torch.isnan(w).all(dim=-1), torch.full_like(out, float("nan")), out)


def nanmax_last(w: torch.Tensor) -> torch.Tensor:
    filled = torch.where(torch.isnan(w), torch.full_like(w, float("-inf")), w)
    out = filled.max(dim=-1).values
    return torch.where(torch.isnan(w).all(dim=-1), torch.full_like(out, float("nan")), out)


def nanmedian_last(w: torch.Tensor) -> torch.Tensor:
    return w.nanmedian(dim=-1).values


def first_valid_last(w: torch.Tensor) -> torch.Tensor:
    """First non-NaN value along the last dim; NaN if the slice has none.
    Shared by DT's `day_first`/`in_first` — same contract as `nanmin_last`/
    `nanmax_last` above: works identically on a whole row or an
    already-unfolded rolling window.
    """
    d = w.shape[-1]
    idx = torch.arange(d, device=w.device).expand_as(w)
    valid = ~torch.isnan(w)
    idx_when_valid = torch.where(valid, idx, torch.full_like(idx, d))
    first_idx = idx_when_valid.min(dim=-1, keepdim=True).values
    has_valid = first_idx < d
    first_idx_safe = torch.where(has_valid, first_idx, torch.zeros_like(first_idx))
    val = torch.gather(w, -1, first_idx_safe)
    return torch.where(has_valid, val, torch.full_like(val, float("nan"))).squeeze(-1)


def last_valid_last(w: torch.Tensor) -> torch.Tensor:
    """Last non-NaN value along the last dim; NaN if the slice has none.
    Shared by DT's `day_last`/`in_last`, same contract as `first_valid_last`.
    """
    d = w.shape[-1]
    idx = torch.arange(d, device=w.device).expand_as(w)
    valid = ~torch.isnan(w)
    idx_when_valid = torch.where(valid, idx, torch.full_like(idx, -1))
    last_idx = idx_when_valid.max(dim=-1, keepdim=True).values
    has_valid = last_idx >= 0
    last_idx_safe = torch.where(has_valid, last_idx, torch.zeros_like(last_idx))
    val = torch.gather(w, -1, last_idx_safe)
    return torch.where(has_valid, val, torch.full_like(val, float("nan"))).squeeze(-1)


def nanargmin_last(w: torch.Tensor, pct: bool = True) -> torch.Tensor:
    """Position of the min value within the window, counted from the END
    (0 = the most recent point), optionally as a fraction of the window's
    valid-count. NaN if the window has no valid values.
    """
    d = w.shape[-1]
    filled = torch.where(torch.isnan(w), torch.full_like(w, float("inf")), w)
    idx_from_start = filled.argmin(dim=-1)
    idx_from_end = (d - 1 - idx_from_start).to(w.dtype)
    allnan = torch.isnan(w).all(dim=-1)
    idx_from_end = torch.where(allnan, torch.full_like(idx_from_end, float("nan")), idx_from_end)
    if pct:
        count = nan_count(w).to(w.dtype)
        idx_from_end = torch.where(count > 0, (idx_from_end + 1) / count, torch.full_like(idx_from_end, float("nan")))
    return idx_from_end


def nanargmax_last(w: torch.Tensor, pct: bool = True) -> torch.Tensor:
    d = w.shape[-1]
    filled = torch.where(torch.isnan(w), torch.full_like(w, float("-inf")), w)
    idx_from_start = filled.argmax(dim=-1)
    idx_from_end = (d - 1 - idx_from_start).to(w.dtype)
    allnan = torch.isnan(w).all(dim=-1)
    idx_from_end = torch.where(allnan, torch.full_like(idx_from_end, float("nan")), idx_from_end)
    if pct:
        count = nan_count(w).to(w.dtype)
        idx_from_end = torch.where(count > 0, (idx_from_end + 1) / count, torch.full_like(idx_from_end, float("nan")))
    return idx_from_end


def nanrank_last(w: torch.Tensor, pct: bool = True) -> torch.Tensor:
    """NaN-safe rank of the window's LAST (current) value among all valid
    values in the window: `count(valid window values <= current)`. One-shot
    vectorized (no argsort, no loop) — replaces the old `ts_rank`/
    `cross_rank`/`in_rank_rolling`'s buggy expanding-window loops.
    """
    last = w[..., -1:]
    valid = ~torch.isnan(w)
    le = (w <= last) & valid
    r = le.sum(dim=-1).to(w.dtype)
    if pct:
        count = valid.sum(dim=-1).to(w.dtype)
        r = torch.where(count > 0, r / count, torch.full_like(r, float("nan")))
    return r


def nansum_last(w: torch.Tensor) -> torch.Tensor:
    value = torch.nansum(w, dim=-1)
    return torch.where(
        torch.isfinite(w).any(dim=-1), value, torch.full_like(value, float("nan"))
    )


def nanmean_last(w: torch.Tensor) -> torch.Tensor:
    return w.nanmean(dim=-1)


def nanstd_last(w: torch.Tensor) -> torch.Tensor:
    mean = w.nanmean(dim=-1, keepdim=True)
    sq = torch.where(torch.isnan(w), torch.zeros_like(w), (w - mean) ** 2)
    count = nan_count(w).clamp(min=1).to(w.dtype)
    return torch.sqrt(sq.sum(dim=-1) / count)


def nanvar_last(w: torch.Tensor) -> torch.Tensor:
    return nanstd_last(w) ** 2


def nanproduct_last(w: torch.Tensor) -> torch.Tensor:
    filled = torch.where(torch.isnan(w), torch.ones_like(w), w)
    value = filled.prod(dim=-1)
    return torch.where(
        torch.isfinite(w).any(dim=-1), value, torch.full_like(value, float("nan"))
    )


def nanskew_last(w: torch.Tensor) -> torch.Tensor:
    """NaN-safe skewness of the window's last dim. Added for `ts_skew`
    (ND) / rolling skew (DT) — no rolling skew reducer existed before;
    follows the same masked-moment pattern as `nanstd_last`.
    """
    mean = w.nanmean(dim=-1, keepdim=True)
    diff = torch.where(torch.isnan(w), torch.zeros_like(w), w - mean)
    count = nan_count(w).clamp(min=1).to(w.dtype)
    var = (diff ** 2).sum(dim=-1) / count
    std = torch.sqrt(var)
    m3 = (diff ** 3).sum(dim=-1) / count
    return torch.where(std > EPS, m3 / (std ** 3 + EPS), torch.full_like(std, float("nan")))


def nankurt_last(w: torch.Tensor) -> torch.Tensor:
    """NaN-safe (excess) kurtosis of the window's last dim. Added for
    `ts_kurt` (ND) / rolling kurtosis (DT) — same masked-moment pattern
    as `nanstd_last`/`nanskew_last`.
    """
    mean = w.nanmean(dim=-1, keepdim=True)
    diff = torch.where(torch.isnan(w), torch.zeros_like(w), w - mean)
    count = nan_count(w).clamp(min=1).to(w.dtype)
    var = (diff ** 2).sum(dim=-1) / count
    m4 = (diff ** 4).sum(dim=-1) / count
    return torch.where(var > EPS, m4 / (var ** 2 + EPS) - 3.0, torch.full_like(var, float("nan")))


def decay_linear_last(w: torch.Tensor) -> torch.Tensor:
    """Linearly-weighted (most recent = heaviest) mean over valid entries."""
    d = w.shape[-1]
    weights = torch.arange(1, d + 1, dtype=w.dtype, device=w.device)
    valid = ~torch.isnan(w)
    w_filled = torch.where(valid, w, torch.zeros_like(w))
    weighted_sum = (w_filled * weights).sum(dim=-1)
    weight_total = (valid.to(w.dtype) * weights).sum(dim=-1)
    return torch.where(weight_total > EPS, weighted_sum / weight_total, torch.full_like(weighted_sum, float("nan")))


# --------------------------------------------------------------------------
# generic rolling reducers built on rolling_windows (drive ts_*/cross_*/in_*)
# --------------------------------------------------------------------------

def rolling_reduce(x: torch.Tensor, d: int, reducer, axis: int = 0) -> torch.Tensor:
    return reducer(rolling_windows(x, d, axis))


def rolling_pair_reduce(x: torch.Tensor, y: torch.Tensor, d: int, kind: str, axis: int = 0) -> torch.Tensor:
    """Shared rolling cov/corr for a PAIR of tensors — one implementation
    replaces `ts_cov`/`ts_corr`/`cross_cov`/`cross_corr`/`in_cov`/`in_corr`.
    """
    wx = rolling_windows(x, d, axis)
    wy = rolling_windows(y, d, axis)
    valid = ~(torch.isnan(wx) | torch.isnan(wy))
    count = valid.sum(dim=-1).clamp(min=1).to(x.dtype)
    x0 = torch.where(valid, wx, torch.zeros_like(wx))
    y0 = torch.where(valid, wy, torch.zeros_like(wy))
    mean_x = x0.sum(dim=-1) / count
    mean_y = y0.sum(dim=-1) / count
    dx = torch.where(valid, wx - mean_x.unsqueeze(-1), torch.zeros_like(wx))
    dy = torch.where(valid, wy - mean_y.unsqueeze(-1), torch.zeros_like(wy))
    cov = (dx * dy).sum(dim=-1) / count
    if kind == "cov":
        return cov
    var_x = (dx ** 2).sum(dim=-1) / count
    var_y = (dy ** 2).sum(dim=-1) / count
    return cov / (torch.sqrt(var_x * var_y) + EPS)


def rolling_ols(x: torch.Tensor, y: torch.Tensor, d: int, axis: int = 0) -> Dict[str, torch.Tensor]:
    """Shared rolling single-predictor OLS `y ~ alpha + beta*x`. One
    implementation replaces the old `ts_regbeta/regalpha/regres/
    reg_tvalue/reg_fvalue` AND `cross_reg*` AND the `in_reg*` transpose-
    wrappers around them — all five statistics computed in one pass.
    """
    wx = rolling_windows(x, d, axis)
    wy = rolling_windows(y, d, axis)
    valid = ~(torch.isnan(wx) | torch.isnan(wy))
    count = valid.sum(dim=-1).to(x.dtype)
    count_safe = count.clamp(min=1)
    x0 = torch.where(valid, wx, torch.zeros_like(wx))
    y0 = torch.where(valid, wy, torch.zeros_like(wy))
    mean_x = x0.sum(dim=-1) / count_safe
    mean_y = y0.sum(dim=-1) / count_safe
    dx = torch.where(valid, wx - mean_x.unsqueeze(-1), torch.zeros_like(wx))
    dy = torch.where(valid, wy - mean_y.unsqueeze(-1), torch.zeros_like(wy))
    cov = (dx * dy).sum(dim=-1)
    var_x = (dx ** 2).sum(dim=-1)
    nan_col = torch.full_like(var_x, float("nan"))
    beta = torch.where(var_x > EPS, cov / var_x, nan_col)
    alpha = mean_y - beta * mean_x

    last_valid = valid[..., -1]
    last_x = wx[..., -1]
    last_y = wy[..., -1]
    pred_last = alpha + beta * last_x
    resid = torch.where(last_valid, last_y - pred_last, nan_col)

    pred = alpha.unsqueeze(-1) + beta.unsqueeze(-1) * wx
    rss = torch.where(valid, (wy - pred) ** 2, torch.zeros_like(wy)).sum(dim=-1)
    df = (count - 2).clamp(min=1)
    mse = rss / df
    se_beta = torch.sqrt(mse / (var_x + EPS))
    tvalue = torch.where(count > 2, beta / (se_beta + EPS), nan_col)

    tss = (dy ** 2).sum(dim=-1)
    ess = tss - rss
    fvalue = torch.where(count > 2, ess / (mse + EPS), nan_col)

    return {"beta": beta, "alpha": alpha, "resid": resid, "tvalue": tvalue, "fvalue": fvalue}


def scale_last_ratio(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """`x / nansum(window)` — shared by `ts_scale`/`cross_scale`/`in_scale_rolling`."""
    denom = torch.nansum(w, dim=-1)
    denom = torch.where(denom.abs() < EPS, torch.ones_like(denom), denom)
    return x / denom
