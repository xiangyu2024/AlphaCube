"""Day-collapse ("day") DT operators — reduce `dim=-1` (M, the
intraday-minute axis) down to a single value per day, shape `(D, 1)`.

DT tensors are shape `(D, M)` = (day, intraday-minute). Every function
here returns `(D, 1)` (matching the original's `keepdim=True` convention,
so results broadcast cleanly against `(D, M)` tensors in downstream
expressions without an extra `.unsqueeze(-1)` at the call site).

Ported from the original `/Users/wxy/Desktop/code/DT_operators.py` `day_*`
family. The simple statistical reducers (`day_sum`/`day_mean`/`day_median`/
`day_max`/`day_min`/`day_std`/`day_var`/`day_skew`/`day_kurt`/`day_first`/
`day_last`/`day_product`) collapse to one-line `common` row-reducer calls
(`common._sum`/`_mean`/`_median`/`_stddev`/`_skew`/`_kurt`/
`first_valid_last`/`last_valid_last`/`nanmax_last`/`nanmin_last`/
`nanproduct_last`, all already NaN-safe `dim=-1` reducers) + `.unsqueeze(-1)`.

## Bugs fixed while porting

- **`day_product` silently leaked NaN.** The original's "treat 0 as the
  multiplicative identity" mask was built with `tensor != 0`, and under
  IEEE-754, `NaN != 0` is `True` — so NaN entries were NOT filtered out
  by the mask at all (only literal `0`s were), and any NaN anywhere in
  the row poisoned the whole row's product. Fixed here via
  `common.nanproduct_last`, which correctly treats NaN (not `0`) as the
  identity element.
- **`day_quantile` redundant Python-loop fallback.** The original defined
  a fully vectorized `day_quantile_efficient` AND a separate row-by-row
  Python-loop `day_quantile`, with `day_quantile_efficient` falling back
  to the loop version only on an (essentially unreachable in modern
  PyTorch) `AttributeError`. Collapsed here into ONE vectorized
  `day_quantile(x, q=0.5)` using `torch.nanquantile` directly; the
  `_efficient` name and the loop fallback are both gone.
- **`day_cov`/`day_corr`** implemented directly (whole-row NaN-safe cov/
  corr, `>= 2` valid pairs required) rather than routed through
  `common.rolling_pair_reduce` with `d = M`: reusing the rolling primitive
  would unfold an `O(M)`-wide window at every one of the `M` positions
  (`O(M^2)` work) just to keep the single last (full-window) slice — pure
  overhead for a whole-row statistic. The math is identical to what
  `rolling_pair_reduce` computes at its last position; it's just done
  once instead of `M` times.

## Moved to `legacy_specialized_dt.py` (see that file's docstring for why)

`day_tide_speed`/`day_tide_change`, the `day_pdf_*` family, `day_q_stat_std`
is kept HERE (see below), `compute_bucket_entropy`, `masked_multivariate_
regression` + wrappers, `top_frac_sum_masked`, `day_longest_run_of_label`
(moved together with `in_longest_run_of_label`, since one calls the other).

`day_q_stat_std` (Ljung-Box Q-statistic std-dev) is kept in THIS file, not
legacy: unlike the PDF/tide/entropy family, it takes no pipeline-specific
hardcoded markers and is a standalone, generically useful diagnostic
statistic (autocorrelation-based), so it's treated as a normal `day_*` op.

## Phantom functions (found, NOT implemented)

`day_value_interval`, `day_distance_interval`, `day_speed_rate`,
`day_change_rate` appear in the original source (~lines 4904-4911) but are
textually INSIDE a stray triple-quoted string literal — they never
execute, don't exist at runtime, and were never callable. Not ported;
there's no real behavior to preserve, only dead text.
"""

from __future__ import annotations

import torch

from . import common as C
from .common import EPS, first_valid_last, last_valid_last, nanmax_last, nanmin_last, nanproduct_last, op

__all__ = [
    "day_sum", "day_mean", "day_median", "day_max", "day_min", "day_var",
    "day_std", "day_skew", "day_kurt", "day_first", "day_last", "day_open",
    "day_close", "day_quantile", "day_cov", "day_corr", "day_product",
    "day_argmax", "day_argmin", "day_locate", "day_hurst",
    "day_durbin_watson", "day_autocorr", "day_ar1_coefficient", "day_q_stat_std",
]


# --------------------------------------------------------------------------
# simple statistical reducers
# --------------------------------------------------------------------------

@op("dt", "day", arity=1)
def day_sum(x: torch.Tensor) -> torch.Tensor:
    return C._sum(x).unsqueeze(-1)


@op("dt", "day", arity=1)
def day_mean(x: torch.Tensor) -> torch.Tensor:
    return C._mean(x.float()).unsqueeze(-1)


@op("dt", "day", arity=1)
def day_median(x: torch.Tensor) -> torch.Tensor:
    return C._median(x).unsqueeze(-1)


@op("dt", "day", arity=1)
def day_max(x: torch.Tensor) -> torch.Tensor:
    return nanmax_last(x).unsqueeze(-1)


@op("dt", "day", arity=1)
def day_min(x: torch.Tensor) -> torch.Tensor:
    return nanmin_last(x).unsqueeze(-1)


@op("dt", "day", arity=1)
def day_std(x: torch.Tensor) -> torch.Tensor:
    return C._stddev(x).unsqueeze(-1)


@op("dt", "day", arity=1)
def day_var(x: torch.Tensor) -> torch.Tensor:
    return day_std(x) ** 2


@op("dt", "day", arity=1)
def day_skew(x: torch.Tensor) -> torch.Tensor:
    return C._skew(x).unsqueeze(-1)


@op("dt", "day", arity=1)
def day_kurt(x: torch.Tensor) -> torch.Tensor:
    return C._kurt(x).unsqueeze(-1)


@op("dt", "day", arity=1)
def day_first(x: torch.Tensor) -> torch.Tensor:
    return first_valid_last(x).unsqueeze(-1)


@op("dt", "day", arity=1)
def day_last(x: torch.Tensor) -> torch.Tensor:
    return last_valid_last(x).unsqueeze(-1)


@op("dt", "day", arity=1)
def day_open(x: torch.Tensor) -> torch.Tensor:
    """Alias for `day_first` (first non-NaN value of the day)."""
    return day_first(x)


@op("dt", "day", arity=1)
def day_close(x: torch.Tensor) -> torch.Tensor:
    """Alias for `day_last` (last non-NaN value of the day)."""
    return day_last(x)


@op("dt", "day", arity=1)
def day_quantile(x: torch.Tensor, q: float = 0.5) -> torch.Tensor:
    """Per-day quantile of the row. Collapses the original's redundant
    `day_quantile_efficient` (vectorized) + `day_quantile` (Python-loop
    fallback) pair into one vectorized function — see module docstring.
    """
    return torch.nanquantile(x, q, dim=-1, keepdim=True)


@op("dt", "day", arity=1)
def day_product(x: torch.Tensor) -> torch.Tensor:
    """Per-day product, NaN treated as the multiplicative identity. FIXES
    the original's NaN leak (see module docstring) via
    `common.nanproduct_last`.
    """
    return nanproduct_last(x).unsqueeze(-1)


@op("dt", "day", arity=1)
def day_argmax(x: torch.Tensor) -> torch.Tensor:
    """Column index (0-based, float) of each row's max, ignoring NaN; NaN
    if the whole row is NaN."""
    not_all_nan = ~torch.all(torch.isnan(x), dim=1, keepdim=True)
    masked = torch.where(torch.isnan(x), torch.full_like(x, float("-inf")), x)
    argmax_vals = torch.argmax(masked, dim=1, keepdim=True).to(torch.float32)
    return torch.where(not_all_nan, argmax_vals, torch.full_like(argmax_vals, float("nan")))


@op("dt", "day", arity=1)
def day_argmin(x: torch.Tensor) -> torch.Tensor:
    """Column index (0-based, float) of each row's min, ignoring NaN; NaN
    if the whole row is NaN."""
    not_all_nan = ~torch.all(torch.isnan(x), dim=1, keepdim=True)
    masked = torch.where(torch.isnan(x), torch.full_like(x, float("inf")), x)
    argmin_vals = torch.argmin(masked, dim=1, keepdim=True).to(torch.float32)
    return torch.where(not_all_nan, argmin_vals, torch.full_like(argmin_vals, float("nan")))


@op("dt", "day", arity=1)
def day_locate(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """`result[i] = x[i, y[i]]`; NaN if `y[i]` is NaN or out of bounds."""
    B, N = x.shape
    device, dtype = x.device, x.dtype
    result = torch.full((B, 1), float("nan"), dtype=dtype, device=device)
    y_flat = y.squeeze(1)
    is_not_nan = ~torch.isnan(y_flat)
    if not is_not_nan.any():
        return result
    indices = y_flat.long()
    indices_clamped = torch.clamp(indices, 0, N - 1)
    gathered = x[torch.arange(B, device=device), indices_clamped]
    in_bounds = (indices >= 0) & (indices < N)
    valid_mask = is_not_nan & in_bounds
    result[valid_mask, 0] = gathered[valid_mask]
    return result


# --------------------------------------------------------------------------
# pair reducers (whole-row cov/corr, direct — see module docstring for why
# not routed through `rolling_pair_reduce`)
# --------------------------------------------------------------------------

@op("dt", "day", arity=2, pair=True)
def day_cov(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    valid = ~(torch.isnan(x) | torch.isnan(y))
    count = valid.sum(dim=-1, keepdim=True).to(x.dtype)
    count_safe = count.clamp(min=1)
    x0 = torch.where(valid, x, torch.zeros_like(x))
    y0 = torch.where(valid, y, torch.zeros_like(y))
    mean_x = x0.sum(dim=-1, keepdim=True) / count_safe
    mean_y = y0.sum(dim=-1, keepdim=True) / count_safe
    dx = torch.where(valid, x - mean_x, torch.zeros_like(x))
    dy = torch.where(valid, y - mean_y, torch.zeros_like(y))
    cov = (dx * dy).sum(dim=-1, keepdim=True) / count_safe
    return torch.where(count >= 2, cov, torch.full_like(cov, float("nan")))


@op("dt", "day", arity=2, pair=True)
def day_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    valid = ~(torch.isnan(x) | torch.isnan(y))
    count = valid.sum(dim=-1, keepdim=True).to(x.dtype)
    count_safe = count.clamp(min=1)
    x0 = torch.where(valid, x, torch.zeros_like(x))
    y0 = torch.where(valid, y, torch.zeros_like(y))
    mean_x = x0.sum(dim=-1, keepdim=True) / count_safe
    mean_y = y0.sum(dim=-1, keepdim=True) / count_safe
    dx = torch.where(valid, x - mean_x, torch.zeros_like(x))
    dy = torch.where(valid, y - mean_y, torch.zeros_like(y))
    cov = (dx * dy).sum(dim=-1, keepdim=True)
    var_x = (dx ** 2).sum(dim=-1, keepdim=True)
    var_y = (dy ** 2).sum(dim=-1, keepdim=True)
    corr = cov / (torch.sqrt(var_x * var_y) + EPS)
    return torch.where(count >= 2, corr, torch.full_like(corr, float("nan")))


# --------------------------------------------------------------------------
# whole-row time-series diagnostics (Hurst / Durbin-Watson / autocorr /
# AR(1) / Ljung-Box Q std) — already vectorized in the original, ported
# with light cleanup
# --------------------------------------------------------------------------

@op("dt", "day", arity=1)
def day_hurst(x: torch.Tensor) -> torch.Tensor:
    """Per-day Hurst exponent (rescaled-range method) over the WHOLE row
    (not a rolling window — see `dt_cross.cross_hurst` for the rolling
    day-axis version)."""
    N, T = x.shape
    x_mean = torch.nanmean(x, dim=1, keepdim=True)
    x_demean = x - x_mean
    x_cumsum = torch.cumsum(x_demean, dim=1)
    cmin = torch.where(torch.isnan(x_cumsum), torch.full_like(x_cumsum, float("inf")), x_cumsum).min(dim=1)[0]
    cmax = torch.where(torch.isnan(x_cumsum), torch.full_like(x_cumsum, float("-inf")), x_cumsum).max(dim=1)[0]
    R = cmax - cmin
    S = torch.sqrt(torch.nanmean(x_demean ** 2, dim=1))
    valid_mask = (S >= 1e-9) & torch.isfinite(R) & torch.isfinite(S)
    RS = R / S
    hurst = torch.log(RS) / torch.log(torch.tensor(float(T), dtype=x.dtype, device=x.device))
    res = torch.full((N, 1), float("nan"), dtype=x.dtype, device=x.device)
    res[valid_mask, 0] = hurst[valid_mask]
    return res


@op("dt", "day", arity=1)
def day_durbin_watson(x: torch.Tensor) -> torch.Tensor:
    """Per-row Durbin-Watson statistic; NaN if <2 valid points or the row
    is constant."""
    D, T = x.shape
    if T < 2:
        return torch.full((D, 1), float("nan"), dtype=x.dtype, device=x.device)
    valid_mask = ~torch.isnan(x)
    n_valid = valid_mask.sum(dim=1, keepdim=True)
    has_enough_data = n_valid >= 2

    diff = x[:, 1:] - x[:, :-1]
    diff_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
    numerator = torch.where(diff_mask, diff ** 2, torch.zeros_like(diff)).sum(dim=1, keepdim=True)

    x_sq_masked = torch.where(valid_mask, x ** 2, torch.zeros_like(x))
    denominator = x_sq_masked.sum(dim=1, keepdim=True)

    dw = torch.where(
        has_enough_data & (denominator > 1e-8),
        numerator / (denominator + 1e-8),
        torch.full_like(x[:, :1], float("nan")),
    )
    mean_x = torch.where(valid_mask, x, torch.zeros_like(x)).sum(dim=1, keepdim=True) / n_valid.clamp(min=1)
    var_x = x_sq_masked.sum(dim=1, keepdim=True) / n_valid.clamp(min=1) - mean_x ** 2
    is_constant = var_x < 1e-8
    return torch.where(is_constant, torch.full_like(dw, float("nan")), dw)


@op("dt", "day", arity=1)
def day_autocorr(x: torch.Tensor, n: int) -> torch.Tensor:
    """Per-row lag-`n` autocorrelation."""
    D, T = x.shape
    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if n >= T:
        return torch.full((D, 1), float("nan"), dtype=x.dtype, device=x.device)

    valid_mask = ~torch.isnan(x)
    x_unfolded = x.unfold(1, n + 1, 1)
    mask_unfolded = valid_mask.unfold(1, n + 1, 1)
    x_t, x_lag = x_unfolded[..., -1], x_unfolded[..., 0]
    pair_mask = mask_unfolded[..., -1] & mask_unfolded[..., 0]
    num_valid_pairs = pair_mask.sum(dim=1, keepdim=True)
    has_data = num_valid_pairs >= 1

    x_t_valid = torch.where(pair_mask, x_t, torch.zeros_like(x_t))
    x_lag_valid = torch.where(pair_mask, x_lag, torch.zeros_like(x_lag))
    mean_x_t = x_t_valid.sum(dim=1, keepdim=True) / num_valid_pairs.clamp(min=1)
    mean_x_lag = x_lag_valid.sum(dim=1, keepdim=True) / num_valid_pairs.clamp(min=1)
    x_t_c = x_t_valid - mean_x_t
    x_lag_c = x_lag_valid - mean_x_lag

    cov = (x_t_c * x_lag_c * pair_mask).sum(dim=1, keepdim=True)
    var_t = (x_t_c ** 2 * pair_mask).sum(dim=1, keepdim=True)
    var_lag = (x_lag_c ** 2 * pair_mask).sum(dim=1, keepdim=True)
    denom = torch.sqrt(var_t * var_lag + 1e-8)
    autocorr = cov / denom
    invalid = ~has_data | (denom < 1e-8)
    return torch.where(invalid, torch.full_like(autocorr, float("nan")), autocorr)


@op("dt", "day", arity=1)
def day_ar1_coefficient(x: torch.Tensor) -> torch.Tensor:
    """Per-row AR(1) OLS coefficient `phi` for `x_t = c + phi*x_{t-1} + e`."""
    D, T = x.shape
    if T < 2:
        return torch.full((D, 1), float("nan"), dtype=x.dtype, device=x.device)
    valid_mask = ~torch.isnan(x)
    n_valid = valid_mask.sum(dim=1, keepdim=True)
    has_enough_data = n_valid >= 2

    y, y_mask = x[:, 1:], valid_mask[:, 1:]
    x_lag, x_lag_mask = x[:, :-1], valid_mask[:, :-1]
    pair_mask = y_mask & x_lag_mask
    num_pairs = pair_mask.sum(dim=1, keepdim=True)

    y_valid = torch.where(pair_mask, y, torch.zeros_like(y))
    x_lag_valid = torch.where(pair_mask, x_lag, torch.zeros_like(x_lag))
    mean_y = y_valid.sum(dim=1, keepdim=True) / num_pairs.clamp(min=1)
    mean_x = x_lag_valid.sum(dim=1, keepdim=True) / num_pairs.clamp(min=1)
    y_c = y_valid - mean_y
    x_c = x_lag_valid - mean_x

    cov = (y_c * x_c * pair_mask).sum(dim=1, keepdim=True)
    var_x = (x_c * x_c * pair_mask).sum(dim=1, keepdim=True)
    phi = torch.where(
        has_enough_data & (var_x > 1e-8), cov / (var_x + 1e-8), torch.full_like(cov, float("nan"))
    )
    return torch.where(var_x < 1e-8, torch.full_like(phi, float("nan")), phi)


@op("dt", "day", arity=1)
def day_q_stat_std(x: torch.Tensor, max_lag: int = 20) -> torch.Tensor:
    """Std-dev of the Ljung-Box Q-statistic sequence (lags `1..max_lag`)
    for each row. The `max_lag`-length loop is `O(max_lag)`, not
    `O(D)`/`O(T)` — a small, fixed-size fitting loop, not a hand-rolled
    windowing loop, so it's left as-is rather than vectorized further.
    """
    D, T = x.shape
    device, dtype = x.device, x.dtype
    max_lag = min(int(max_lag), T - 1)
    if max_lag < 1:
        return torch.full((D, 1), float("nan"), device=device, dtype=dtype)

    valid_mask = ~torch.isnan(x)
    n_valid = valid_mask.sum(dim=1, keepdim=True)
    has_enough_data = n_valid >= 2

    autocorr_list = []
    for k in range(1, max_lag + 1):
        if T <= k:
            autocorr_list.append(torch.full((D, 1), float("nan"), device=device, dtype=dtype))
            continue
        x_t, x_t_k = x[:, k:], x[:, :-k]
        pair_mask = valid_mask[:, k:] & valid_mask[:, :-k]
        num_pairs = pair_mask.sum(dim=1, keepdim=True)
        has_pairs = num_pairs >= 1

        x_t_valid = torch.where(pair_mask, x_t, torch.zeros_like(x_t))
        x_t_k_valid = torch.where(pair_mask, x_t_k, torch.zeros_like(x_t_k))
        mean_t = x_t_valid.sum(dim=1, keepdim=True) / num_pairs.clamp(min=1)
        mean_tk = x_t_k_valid.sum(dim=1, keepdim=True) / num_pairs.clamp(min=1)
        c_t = x_t_valid - mean_t
        c_tk = x_t_k_valid - mean_tk

        cov = (c_t * c_tk * pair_mask).sum(dim=1, keepdim=True)
        var_t = (c_t ** 2 * pair_mask).sum(dim=1, keepdim=True)
        var_tk = (c_tk ** 2 * pair_mask).sum(dim=1, keepdim=True)
        denom = torch.sqrt(var_t * var_tk + 1e-8)
        rho_k = torch.where(
            has_pairs & (denom > 1e-8), cov / denom, torch.full_like(cov, float("nan"))
        )
        autocorr_list.append(rho_k)

    autocorr_stack = torch.cat(autocorr_list, dim=1)  # (D, K)
    T_t = torch.tensor(float(T), device=device, dtype=dtype)
    K = autocorr_stack.shape[1]
    lag_idx = torch.arange(1, K + 1, device=device, dtype=dtype).unsqueeze(0)
    denom = T_t - lag_idx
    term = autocorr_stack.pow(2) / denom
    q_stats = T_t * (T_t + 2) * torch.cumsum(term, dim=1)
    q_std = torch.std(q_stats, dim=1, keepdim=True)
    return torch.where(has_enough_data, q_std, torch.full_like(q_std, float("nan")))
