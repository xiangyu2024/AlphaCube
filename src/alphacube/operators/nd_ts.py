"""Time-series ("ts") ND operators.

ND tensors are shape `(T, N)` = (time, stock). Every operator here rolls
along `dim=0` (time), independently per stock (column), with an integer
window `d`.

Ported from the original `/Users/wxy/Desktop/code/ND_operators.py` `ts_*`
family, rebuilt on top of `operators.common`'s `rolling_windows` /
`rolling_reduce` / `rolling_pair_reduce` / `rolling_ols` / `nan*_last` /
`decay_linear_last` / `scale_last_ratio` primitives.

## Why this file is mostly one-liners

Every rolling operator in the original file was hand-written with TWO
separate code paths: a vectorized "full window" branch (`x.unfold(...)`,
valid once `t >= d-1`) and a hand-rolled Python-loop "expanding window"
branch for the first `d-1` rows. `common.rolling_windows` eliminates the
split by left-padding with NaN before unfolding, so a single call to a
NaN-aware reducer handles every row. Confirmed bugs this fixes
structurally (not just patches around):

- `ts_min`/`ts_max`/`ts_med`/`ts_argmin`/`ts_argmax`/`ts_rank`: the
  original's expanding-window loop called `torch.min`/`max`/`median`/
  `argmin`/`argsort` on `x[:i+1]` WITHOUT `dim=0` — for a `(T, N)` tensor
  that silently reduces across BOTH time and stock, or sorts along the
  wrong axis, for every one of the first `d-1` rows. Completely wrong
  output there. `nanmin_last`/`nanmax_last`/etc. only ever reduce a
  window's trailing dim, so this class of bug can't recur.
- `ts_mean`/`ts_corr`/`ts_stddev`/the `ts_reg*` family: the "full window"
  and "expanding window" branches used inconsistent NaN handling (e.g.
  `ts_mean`'s expanding branch used a raw `cumsum`/`count` that does NOT
  skip NaNs, while its full-window branch used `nanmean`). Since every
  function below has exactly one code path, there's nothing for the two
  branches to disagree about.
"""

from __future__ import annotations

from typing import List

import torch

from .common import (
    _mean as _cs_mean_reduce,
    _stddev as _cs_stddev_reduce,
    decay_linear_last,
    get_nan_tensor,
    nanargmax_last,
    nanargmin_last,
    nankurt_last,
    nanmax_last,
    nanmean_last,
    nanmedian_last,
    nanmin_last,
    nanproduct_last,
    nanrank_last,
    nanskew_last,
    nanstd_last,
    nansum_last,
    nanvar_last,
    op,
    rolling_ols,
    rolling_pair_reduce,
    rolling_reduce,
    rolling_windows,
    scale_last_ratio,
    sub,
)

__all__ = [
    "ts_delay",
    "ts_shift",
    "ts_delta",
    "ts_sum",
    "ts_mean",
    "ts_std",
    "ts_stddev",
    "ts_var",
    "ts_min",
    "ts_max",
    "ts_med",
    "ts_argmin",
    "ts_argmax",
    "ts_rank",
    "ts_scale",
    "ts_product",
    "ts_decay_linear",
    "ts_skew",
    "ts_kurt",
    "ts_cov",
    "ts_corr",
    "ts_autocorr",
    "ts_quantile",
    "ts_regbeta",
    "ts_regalpha",
    "ts_regres",
    "ts_reg_tvalue",
    "ts_reg_fvalue",
    "ts_multiregbeta",
    "ts_multiregres",
    "ts_outlier",
    "ts_pearson",
    "ts_spearman",
    "ts_hurst",
]


# --------------------------------------------------------------------------
# lag / shift / diff
# --------------------------------------------------------------------------

@op("nd", "ts", arity=1, requires_window=True)
def ts_delay(x: torch.Tensor, d) -> torch.Tensor:
    """Lag by `d` (a.k.a. `Ref`): `res[t] = x[t-d]`."""
    d = int(d)
    res = get_nan_tensor(x.shape, x.device, x.dtype)
    if d > 0:
        res[d:] = x[:-d]
    return res


@op("nd", "ts", arity=1, requires_window=True)
def ts_shift(x: torch.Tensor, d) -> torch.Tensor:
    """Lead by `d`: `res[t] = x[t+d]`."""
    d = int(d)
    res = get_nan_tensor(x.shape, x.device, x.dtype)
    if d > 0:
        res[:-d] = x[d:]
    return res


@op("nd", "ts", arity=1, requires_window=True)
def ts_delta(x: torch.Tensor, d) -> torch.Tensor:
    return sub(x, ts_delay(x, d))


# --------------------------------------------------------------------------
# rolling sum / mean / std / var
# --------------------------------------------------------------------------

@op("nd", "ts", arity=1, requires_window=True)
def ts_sum(x: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_reduce(x, d, nansum_last, axis=0)


@op("nd", "ts", arity=1, requires_window=True)
def ts_mean(x: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_reduce(x, d, nanmean_last, axis=0)


@op("nd", "ts", arity=1, requires_window=True)
def ts_std(x: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_reduce(x, d, nanstd_last, axis=0)


@op("nd", "ts", arity=1, requires_window=True)
def ts_stddev(x: torch.Tensor, d: int) -> torch.Tensor:
    """Alias of `ts_std` — `ts_stddev` was the original file's name."""
    return ts_std(x, d)


@op("nd", "ts", arity=1, requires_window=True)
def ts_var(x: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_reduce(x, d, nanvar_last, axis=0)


# --------------------------------------------------------------------------
# rolling min / max / median / argmin / argmax / rank
# --------------------------------------------------------------------------

@op("nd", "ts", arity=1, requires_window=True)
def ts_min(x: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_reduce(x, d, nanmin_last, axis=0)


@op("nd", "ts", arity=1, requires_window=True)
def ts_max(x: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_reduce(x, d, nanmax_last, axis=0)


@op("nd", "ts", arity=1, requires_window=True)
def ts_med(x: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_reduce(x, d, nanmedian_last, axis=0)


@op("nd", "ts", arity=1, requires_window=True)
def ts_argmin(x: torch.Tensor, d: int, pct: bool = True) -> torch.Tensor:
    return rolling_reduce(x, d, lambda w: nanargmin_last(w, pct=pct), axis=0)


@op("nd", "ts", arity=1, requires_window=True)
def ts_argmax(x: torch.Tensor, d: int, pct: bool = True) -> torch.Tensor:
    return rolling_reduce(x, d, lambda w: nanargmax_last(w, pct=pct), axis=0)


@op("nd", "ts", arity=1, requires_window=True)
def ts_rank(x: torch.Tensor, d: int, pct: bool = True) -> torch.Tensor:
    return rolling_reduce(x, d, lambda w: nanrank_last(w, pct=pct), axis=0)


# --------------------------------------------------------------------------
# scale / product / decay
# --------------------------------------------------------------------------

@op("nd", "ts", arity=1, requires_window=True)
def ts_scale(x: torch.Tensor, d: int, constant: float = 0.0) -> torch.Tensor:
    """`x[t] / sum(window ending at t) + constant`."""
    w = rolling_windows(x, d, axis=0)
    return scale_last_ratio(x, w) + constant


@op("nd", "ts", arity=1, requires_window=True)
def ts_product(x: torch.Tensor, d) -> torch.Tensor:
    """Canonical name for the rolling product (the original file also had
    a bare top-level `product(x, d)` doing the same thing — dropped as a
    duplicate; `ts_product` is the one name kept).
    """
    return rolling_reduce(x, d, nanproduct_last, axis=0)


@op("nd", "ts", arity=1, requires_window=True)
def ts_decay_linear(x: torch.Tensor, d) -> torch.Tensor:
    return rolling_reduce(x, d, decay_linear_last, axis=0)


# --------------------------------------------------------------------------
# rolling skew / kurt
# --------------------------------------------------------------------------

@op("nd", "ts", arity=1, requires_window=True)
def ts_skew(x: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_reduce(x, d, nanskew_last, axis=0)


@op("nd", "ts", arity=1, requires_window=True)
def ts_kurt(x: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_reduce(x, d, nankurt_last, axis=0)


# --------------------------------------------------------------------------
# rolling cov / corr / autocorr / quantile
# --------------------------------------------------------------------------

@op("nd", "ts", arity=2, requires_window=True, pair=True)
def ts_cov(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_pair_reduce(x, y, d, "cov", axis=0)


@op("nd", "ts", arity=2, requires_window=True, pair=True)
def ts_corr(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_pair_reduce(x, y, d, "corr", axis=0)


@op("nd", "ts", arity=1, requires_window=True)
def ts_autocorr(x: torch.Tensor, d: int, shift: int) -> torch.Tensor:
    shift = int(shift)
    res = get_nan_tensor(x.shape, x.device, x.dtype)
    res[shift:] = ts_corr(x[shift:], x[:-shift], d)
    return res


@op("nd", "ts", arity=1, requires_window=True)
def ts_quantile(x: torch.Tensor, q: float, d: int) -> torch.Tensor:
    w = rolling_windows(x, d, axis=0)
    return torch.nanquantile(w, q, dim=-1)


# --------------------------------------------------------------------------
# rolling OLS family — ALL five statistics come from one `rolling_ols` call
# --------------------------------------------------------------------------

@op("nd", "ts", arity=2, requires_window=True, pair=True)
def ts_regbeta(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_ols(x, y, d, axis=0)["beta"]


@op("nd", "ts", arity=2, requires_window=True, pair=True)
def ts_regalpha(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    return rolling_ols(x, y, d, axis=0)["alpha"]


@op("nd", "ts", arity=2, requires_window=True, pair=True)
def ts_regres(x: torch.Tensor, y: torch.Tensor, d) -> torch.Tensor:
    return rolling_ols(x, y, int(d), axis=0)["resid"]


@op("nd", "ts", arity=2, requires_window=True, pair=True)
def ts_reg_tvalue(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    """t-value of the rolling beta. The original had NO incremental-window
    implementation at all for this (its expanding-window branch was a
    `pass`/comment, i.e. the first `d-1` rows were silently left NaN with
    no attempt made) — `rolling_ols` gives every row a value in one pass.
    """
    return rolling_ols(x, y, d, axis=0)["tvalue"]


@op("nd", "ts", arity=2, requires_window=True, pair=True)
def ts_reg_fvalue(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    """F-value of the rolling regression. Same missing-incremental-window
    bug as `ts_reg_tvalue` in the original, fixed the same way.
    """
    return rolling_ols(x, y, d, axis=0)["fvalue"]


@op("nd", "ts", arity=2, requires_window=True)
def ts_multiregbeta(x_list: List[torch.Tensor], y: torch.Tensor, d) -> List[torch.Tensor]:
    """Rolling beta of `y` against each predictor in `x_list`
    INDEPENDENTLY (`d`-window univariate OLS per predictor, not a joint
    multivariate regression) — one `rolling_ols` call per predictor.

    Bug fix vs. the original: the original allocated its result as
    `get_nan_tensor(shape=(F, T))` (`F` = number of predictors, `T` =
    `len(y)`) — for an ND `y` of shape `(T, N)` with `N > 1` that drops
    the stock dimension entirely, and the subsequent
    `beta[d-1:] = torch.where(valid, cov/var, nan)` assignment (RHS shape
    `(T-d+1, N)`, LHS slice shape `(T-d+1,)`) would raise a shape-mismatch
    error for any `N != 1`. Each `rolling_ols(...)["beta"]` call here
    correctly returns the full `(T, N)` per-predictor beta.
    """
    if not x_list:
        return []
    d = int(d)
    return [rolling_ols(x, y, d, axis=0)["beta"] for x in x_list]


@op("nd", "ts", arity=2, requires_window=True)
def ts_multiregres(x_list: List[torch.Tensor], y: torch.Tensor, d) -> torch.Tensor:
    betas = ts_multiregbeta(x_list, y, d)
    res = y
    for x, beta in zip(x_list, betas):
        res = res - beta * x
    return res


# --------------------------------------------------------------------------
# rank-band outlier flag
# --------------------------------------------------------------------------

@op("nd", "ts", arity=1, requires_window=True)
def ts_outlier(x: torch.Tensor, d, n: float, m: float) -> torch.Tensor:
    """Flag `x[t, i]` as +1/-1/0 relative to a mean+m*std threshold, where
    the threshold is built from the CROSS-SECTIONAL (per-row, across
    stocks — `common._mean`/`_stddev`, `dim=-1`) mean/std of whichever
    entries in that row have a `ts_rank(x, d)` percentile below `n/100`.
    This mixed ts+cs shape matches the original's intent.

    Bug fix vs. the original: `_mean(selected)`/`_stddev(selected)`
    reduce `dim=-1` WITHOUT `keepdim`, producing shape `(T,)`; the
    original then did `is_gt(x, threshold)` with `x` shape `(T, N)` and
    `threshold` shape `(T,)`. Torch broadcasts trailing dims, so that
    only works by coincidence when `T == N`; for `T != N` (the normal
    case — time steps and stock counts are essentially never equal) it
    raises a shape-mismatch error. Fixed here with an explicit
    `.unsqueeze(-1)` so `threshold` broadcasts against `x` correctly.
    """
    d = int(d)
    rank_pct = ts_rank(x, d, pct=True)
    mask = rank_pct < (n / 100.0)
    selected = torch.where(mask, x, torch.full_like(x, float("nan")))

    mean = _cs_mean_reduce(selected).unsqueeze(-1)
    std = _cs_stddev_reduce(selected).unsqueeze(-1)
    threshold = mean + m * std

    return torch.where(
        x > threshold,
        torch.ones_like(x),
        torch.where(x < threshold, -torch.ones_like(x), torch.zeros_like(x)),
    )


# --------------------------------------------------------------------------
# rolling cross-stock correlation, aggregated per stock
# --------------------------------------------------------------------------

#: Peak bytes `ts_pearson` may hold in its `(chunk, N, N)` working set. 512MB
#: keeps a 300-stock run at one chunk of ~740 days and a full 5215-stock market
#: at 4 days per chunk; raise it on a big-RAM box to trade memory for fewer
#: passes. Only the pairwise term is budgeted -- the `(T, N)` output and the
#: `(T, d, N)` window view are linear in N and not worth counting.
TS_PEARSON_PAIRWISE_BUDGET_BYTES = 512 * 1024 * 1024


def _ts_pearson_chunk_rows(n_days: int, n_stocks: int, itemsize: int) -> int:
    """How many windows fit in the pairwise budget, at least one."""
    per_row = max(1, n_stocks * n_stocks * itemsize * 3)  # cov + counts + corr
    return max(1, min(n_days, TS_PEARSON_PAIRWISE_BUDGET_BYTES // per_row))


@op("nd", "ts", arity=1, requires_window=True)
def ts_pearson(data: torch.Tensor, d: int, abs_corr: bool = False, agg: str = "mean") -> torch.Tensor:
    """For each stock `i` and each rolling window ending at `t`: the
    Pearson correlation of `i`'s window against every OTHER stock `j`'s
    window, aggregated (`mean` or `std`) over `j`.

    Memory: the original built the pairwise products as a literal
    `(T, d, N, N)` tensor, which is quadratic in the universe AND carries the
    window length as a factor -- 2111 days x 300 stocks at window 50 wanted
    ~9.3GB for the bool mask plus ~37GB for the float products, and a full
    5215-stock market wanted ~5.4TB for a SINGLE day. It was consequently
    deferred as a resource risk (`alphamine.factor_contract`) and unusable in
    search. Two changes fix that without changing what is computed:

    1. **The `d` axis folds into a matmul.** `x_centered` is already zero
       wherever the window is NaN, so masking the pairwise product is a no-op
       (`0 * anything == 0`) and
       `sum_k [common_mask * xc_i * xc_j] == einsum('tki,tkj->tij', xc, xc)`.
       The same holds for the pairwise observation count over the float mask.
       Both reduce straight to `(T, N, N)`; the `(T, d, N, N)` intermediate
       never exists.
    2. **The remaining `(T, N, N)` term is chunked over time**, which is exact
       because each window's result depends on no other window.

    Peak working set is therefore `TS_PEARSON_PAIRWISE_BUDGET_BYTES`, not a
    function of the data size. Time is still O(T d N^2) -- the pairwise
    correlation is genuinely quadratic in the universe, and that is a property
    of the factor, not of this implementation.

    NOT bit-identical to the pre-rewrite version: a matmul accumulates in a
    different order than an explicit element-wise `sum(dim=1)`, so results
    differ in trailing ULPs (~1e-7 relative on float32). The change is an
    algebraic identity in exact arithmetic, and it applies to every caller at
    once, so no two code paths disagree with each other -- but factor values
    stored before the rewrite will not reproduce to the last bit.
    """
    if agg not in ("mean", "std"):
        raise ValueError("agg must be 'mean' or 'std'")
    D, N = data.shape
    d = int(d)
    if d <= 1 or D < d:
        return torch.full((D, N), float("nan"), device=data.device, dtype=data.dtype)

    windows = data.unfold(0, d, 1).transpose(1, 2)  # [T, d, N]
    T = windows.size(0)
    mask = ~torch.isnan(windows)

    x_mean = torch.nanmean(windows, dim=1, keepdim=True)
    x_centered = torch.where(mask, windows - x_mean, torch.zeros_like(windows))
    x_var = torch.nanmean(x_centered ** 2, dim=1)
    x_std = torch.sqrt(x_var + 1e-8)

    diag = torch.eye(N, dtype=torch.bool, device=data.device).unsqueeze(0)
    result_per_stock = torch.empty((T, N), device=data.device, dtype=windows.dtype)
    step = _ts_pearson_chunk_rows(T, N, windows.element_size())

    for lo in range(0, T, step):
        hi = min(lo + step, T)
        xc = x_centered[lo:hi]                                  # [c, d, N]
        maskf = mask[lo:hi].to(windows.dtype)                   # [c, d, N]

        # sum_k common_mask * xc_i * xc_j, and sum_k common_mask -- both as
        # batched matmuls straight to [c, N, N] (see docstring point 1).
        cov = torch.bmm(xc.transpose(1, 2), xc)
        num_valid = torch.bmm(maskf.transpose(1, 2), maskf)

        valid = num_valid >= 2
        cov = cov / num_valid.clamp(min=1)
        cov = torch.where(valid, cov, torch.full_like(cov, float("nan")))

        std_chunk = x_std[lo:hi]
        std_outer = std_chunk.unsqueeze(-1) * std_chunk.unsqueeze(-2)
        corr = cov / (std_outer + 1e-8)
        corr = torch.where(~diag, corr, torch.full_like(corr, float("nan")))
        if abs_corr:
            corr = corr.abs()

        if agg == "mean":
            result_per_stock[lo:hi] = torch.nanmean(corr, dim=2)
        else:
            count = (~torch.isnan(corr)).sum(dim=2).to(windows.dtype)
            m = torch.nanmean(corr, dim=2, keepdim=True)
            var = torch.nanmean((corr - m) ** 2, dim=2)
            result_per_stock[lo:hi] = torch.where(
                count >= 2, torch.sqrt(var), torch.full_like(var, float("nan")))

    if d == D:
        return result_per_stock
    final = torch.full((D, N), float("nan"), device=data.device, dtype=data.dtype)
    final[d - 1:, :] = result_per_stock
    return final


@op("nd", "ts", arity=1, requires_window=True)
def ts_spearman(data: torch.Tensor, d: int, abs_corr: bool = False, agg: str = "mean") -> torch.Tensor:
    """Rolling rank (Spearman) correlation counterpart to `ts_pearson`.

    NOTE ON THE REMAINING PYTHON LOOP: this keeps the original's
    `for i in range(N)` outer loop over stocks (vectorizing only the
    `j != i` inner computation). A fully `ts_pearson`-style `(T, d, N, N)`
    all-pairs tensor is possible in principle, but ranks require an
    `argsort` per window per stock BEFORE the pairwise step, which makes
    the fully vectorized version substantially more memory (an extra
    `(T, d, N)` rank tensor broadcast to `(T, d, N, N)` pairwise products,
    same as Pearson) without a correspondingly large win, since the
    per-stock rank computation itself is already vectorized outside the
    loop. Given this task's priority is fixing outright bugs and
    dropping dead/redundant code over maximally vectorizing every exotic
    function, the loop (over stocks, i.e. `O(N)`, not `O(T)`) is kept.
    """
    if agg not in ("mean", "std"):
        raise ValueError("agg must be 'mean' or 'std'")
    D, N = data.shape
    d = int(d)
    if d <= 1 or D < d:
        return torch.full((D, N), float("nan"), device=data.device, dtype=data.dtype)

    windows = data.unfold(0, d, 1).transpose(1, 2)  # [T, d, N]
    T = windows.size(0)
    mask = ~torch.isnan(windows)

    x_filled = torch.where(mask, windows, torch.full_like(windows, float("inf")))
    ranks = torch.argsort(torch.argsort(x_filled, dim=1, stable=True), dim=1, stable=True).to(windows.dtype)
    ranks = torch.where(mask, ranks, torch.full_like(ranks, float("nan")))

    rank_mean = torch.nanmean(ranks, dim=1, keepdim=True)
    ranks_centered = ranks - rank_mean
    rank_var = torch.nanmean(ranks_centered ** 2, dim=1)
    rank_std = torch.sqrt(rank_var + 1e-8)

    result_per_stock = torch.full((T, N), float("nan"), device=data.device, dtype=data.dtype)

    for i in range(N):
        rc_i = ranks_centered[:, :, i]
        std_i = rank_std[:, i]
        mask_i = mask[:, :, i]

        j_indices = [j for j in range(N) if j != i]
        if not j_indices:
            continue

        rc_j_all = ranks_centered[:, :, j_indices]
        std_j_all = rank_std[:, j_indices]
        mask_j_all = mask[:, :, j_indices]

        valid = mask_i.unsqueeze(-1) & mask_j_all
        prod = rc_i.unsqueeze(-1) * rc_j_all
        prod_masked = torch.where(valid, prod, torch.zeros_like(prod))

        sum_prod = prod_masked.sum(dim=1)
        num_valid = valid.sum(dim=1).to(windows.dtype)
        cov = sum_prod / num_valid.clamp(min=1)

        corr = cov / (std_i.unsqueeze(-1) * std_j_all + 1e-8)
        corr = torch.where(num_valid >= 2, corr, torch.full_like(corr, float("nan")))
        if abs_corr:
            corr = corr.abs()

        if agg == "mean":
            result = torch.nanmean(corr, dim=1)
        else:
            count = (~torch.isnan(corr)).sum(dim=1)
            m = torch.nanmean(corr, dim=1, keepdim=True)
            var = torch.nanmean((corr - m) ** 2, dim=1)
            result = torch.where(count >= 2, torch.sqrt(var), torch.full_like(var, float("nan")))

        result_per_stock[:, i] = result

    if d == D:
        return result_per_stock
    final = torch.full((D, N), float("nan"), device=data.device, dtype=data.dtype)
    final[d - 1:, :] = result_per_stock
    return final


# --------------------------------------------------------------------------
# Hurst exponent (R/S analysis) — already fully vectorized in the
# original; ported onto `rolling_windows` for a single code path.
# --------------------------------------------------------------------------

@op("nd", "ts", arity=1, requires_window=True)
def ts_hurst(x: torch.Tensor, d: int) -> torch.Tensor:
    d = int(d)
    w = rolling_windows(x, d, axis=0)  # (T, N, d); NaN-padded for t < d-1

    mean = w.nanmean(dim=-1, keepdim=True)
    demean = w - mean
    # Cumsum along the window axis (oldest -> newest). NaN-padding sits
    # at the OLDEST end of an under-filled window, so it poisons the
    # cumsum from the start for t < d-1 -- exactly reproducing the
    # original's "leave the first d-1 rows NaN" behavior in the same
    # single pass that computes the full-window rows.
    cumsum = demean.cumsum(dim=-1)
    valid = ~torch.isnan(cumsum)
    r_max = torch.where(valid, cumsum, torch.full_like(cumsum, float("-inf"))).max(dim=-1).values
    r_min = torch.where(valid, cumsum, torch.full_like(cumsum, float("inf"))).min(dim=-1).values
    r_d = r_max - r_min

    s_d = torch.sqrt(torch.nansum(demean ** 2, dim=-1) / (d - 1))

    rs = r_d / s_d
    log_d = torch.log(torch.tensor(float(d), dtype=x.dtype, device=x.device))
    hurst = torch.log(rs) / log_d
    return torch.where(s_d < 1e-9, torch.full_like(hurst, float("nan")), hurst)
