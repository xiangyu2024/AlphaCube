"""Pure-numpy forward-looking label construction for the model-aggregate layer.

Every function here takes a `(n_days, n_instruments)` price-or-return panel
plus a `horizon` and returns a `(n_days, n_instruments)` label panel of the
same shape. All of them derive from the same forward-return primitive that
:meth:`alphacube.daily.DailyData.forward_returns` already uses for the
mining/eval side: `price[t+h] / price[t] - 1`.

**No-lookahead contract.** A label at day `t` may only depend on data at days
`> t` (the "future" relative to `t`) up to and including `t + horizon`. The
trailing `horizon` rows of every panel returned here are therefore NaN, not
wrapped-around garbage — there is never a modular/negative-index access into
the array. Every function is unit-tested for this explicitly.

These are plain numpy/pandas functions with zero dependency on the mining
machinery (`engine/`, `mining/`) — callers evaluate a pool member's
expression (or anything else) into a daily feature panel themselves and
pass the resulting `(n_days, n_instruments)` arrays in here.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def forward_return(
    price_or_return_panel: np.ndarray, horizon: int, from_returns: bool = False
) -> np.ndarray:
    """Forward `horizon`-day return, `(n_days, n_instruments)` in, same shape out.

    If ``from_returns=False`` (default), `price_or_return_panel` is a price
    panel and this computes ``price[t+horizon] / price[t] - 1``.

    If ``from_returns=True``, `price_or_return_panel` is already a *daily*
    return panel (``returns[t] = price[t] / price[t-1] - 1``) and this
    computes the equivalent compounded forward return by chaining the next
    `horizon` daily returns: ``prod_{k=1..horizon}(1 + returns[t+k]) - 1``.
    This is algebraically identical to the price-panel formula when
    `returns` was itself derived from `price` via day-over-day pct change.

    The trailing `horizon` rows are NaN (no future price/return exists yet
    for those days) — never wrapped-around.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    panel = np.asarray(price_or_return_panel, dtype=np.float64)
    n_days, n_instruments = panel.shape
    out = np.full((n_days, n_instruments), np.nan, dtype=np.float64)

    valid_n = n_days - horizon
    if valid_n <= 0:
        return out

    if not from_returns:
        price = panel
        with np.errstate(divide="ignore", invalid="ignore"):
            out[:valid_n] = price[horizon:] / price[:valid_n] - 1.0
        return out

    returns = panel
    compounded = np.ones((valid_n, n_instruments), dtype=np.float64)
    for k in range(1, horizon + 1):
        compounded *= 1.0 + returns[k : k + valid_n]
    out[:valid_n] = compounded - 1.0
    return out


def forward_volatility(returns_panel: np.ndarray, horizon: int) -> np.ndarray:
    """Rolling-forward realized std of daily returns over the window
    `[t+1, t+horizon]` (inclusive), population std (ddof=0).

    If any daily return in the forward window is NaN, the result for that
    `(t, instrument)` is NaN (no silent partial-window computation).
    The trailing `horizon` rows are NaN.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    returns = np.asarray(returns_panel, dtype=np.float64)
    n_days, n_instruments = returns.shape
    out = np.full((n_days, n_instruments), np.nan, dtype=np.float64)

    valid_n = n_days - horizon
    if valid_n <= 0:
        return out

    window = np.stack([returns[k : k + valid_n] for k in range(1, horizon + 1)], axis=0)
    with np.errstate(invalid="ignore"):
        out[:valid_n] = np.std(window, axis=0, ddof=0)
    return out


def forward_downside_volatility(
    returns_panel: np.ndarray, horizon: int, mar: float = 0.0
) -> np.ndarray:
    """Semi-deviation: same forward window as :func:`forward_volatility`, but
    the std is computed only over the daily returns strictly below `mar`
    (minimum acceptable return, default 0.0) within that window.

    If a `(t, instrument)` window has zero returns below `mar`, the result
    is NaN (there's nothing to compute a downside std from). The trailing
    `horizon` rows are NaN.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    returns = np.asarray(returns_panel, dtype=np.float64)
    n_days, n_instruments = returns.shape
    out = np.full((n_days, n_instruments), np.nan, dtype=np.float64)

    valid_n = n_days - horizon
    if valid_n <= 0:
        return out

    window = np.stack([returns[k : k + valid_n] for k in range(1, horizon + 1)], axis=0)
    below = window < mar
    count = below.sum(axis=0)
    safe_count = np.maximum(count, 1)

    with np.errstate(invalid="ignore"):
        mean = np.where(below, window, 0.0).sum(axis=0) / safe_count
        sq = np.where(below, (window - mean[np.newaxis, :, :]) ** 2, 0.0)
        var = sq.sum(axis=0) / safe_count
        std = np.sqrt(var)

    out[:valid_n] = np.where(count >= 1, std, np.nan)
    return out


def forward_return_rank(returns_panel: np.ndarray, horizon: int, pct: bool = True) -> np.ndarray:
    """Per-day cross-sectional rank of :func:`forward_return` (computed with
    ``from_returns=True``), in `[0, 1]` if `pct` else raw rank `1..n_valid`.

    NaN-safe (``na_option="keep"``): instruments with a NaN forward return
    that day keep NaN rank rather than corrupting the others' ranks.
    """
    fwd = forward_return(returns_panel, horizon, from_returns=True)
    ranks = pd.DataFrame(fwd).rank(axis=1, pct=pct, na_option="keep")
    return ranks.to_numpy()


def forward_return_class(
    returns_panel: np.ndarray,
    horizon: int,
    top_frac: float = 0.3,
    bottom_frac: float = 0.3,
) -> np.ndarray:
    """Per-day cross-sectional 3-class label from :func:`forward_return`:
    0 = bottom `bottom_frac`, 2 = top `top_frac`, 1 = everything in between.

    Thresholds are the day's own `bottom_frac`/`(1 - top_frac)` quantiles of
    the *valid* (non-NaN) forward returns that day (``np.quantile``, linear
    interpolation). Returned as a float panel (0.0/1.0/2.0) so NaN can mark
    both the tail rows and any instrument missing a forward return that day.
    """
    if not (0.0 < top_frac < 1.0) or not (0.0 < bottom_frac < 1.0):
        raise ValueError("top_frac and bottom_frac must be in (0, 1)")
    if top_frac + bottom_frac > 1.0:
        raise ValueError("top_frac + bottom_frac must be <= 1")

    fwd = forward_return(returns_panel, horizon, from_returns=True)
    n_days, _ = fwd.shape
    out = np.full_like(fwd, np.nan)

    for t in range(n_days):
        row = fwd[t]
        mask = ~np.isnan(row)
        if mask.sum() < 1:
            continue
        valid = row[mask]
        lo = np.quantile(valid, bottom_frac)
        hi = np.quantile(valid, 1.0 - top_frac)
        cls = np.where(valid <= lo, 0.0, np.where(valid >= hi, 2.0, 1.0))
        out[t, mask] = cls
    return out


def forward_event_label(
    returns_panel: np.ndarray,
    horizon: int,
    kind: str,
    threshold: Optional[float] = None,
    quantile: Optional[float] = None,
) -> np.ndarray:
    """Binary (0/1) event label from :func:`forward_return`.

    `kind` is `"spike"` (label 1 where the forward return is large/positive)
    or `"crash"` (label 1 where it's large/negative). Exactly one of
    `threshold`/`quantile` must be given:

    - `threshold` is an absolute forward-return level. For `kind="spike"`,
      label = 1 where ``forward_return > threshold``. For `kind="crash"`,
      label = 1 where ``forward_return < -threshold`` (pass a positive
      magnitude; the sign flip is automatic).
    - `quantile` is a per-day cross-sectional quantile of that day's valid
      forward returns (``np.quantile``), used directly as the cutoff for
      both kinds — e.g. `quantile=0.95` with `kind="spike"` labels the top
      5%; `quantile=0.05` with `kind="crash"` labels the bottom 5%.

    Returned as a float panel (0.0/1.0) with NaN for the trailing `horizon`
    rows and any instrument missing a forward return that day.
    """
    if kind not in ("spike", "crash"):
        raise ValueError("kind must be 'spike' or 'crash'")
    if (threshold is None) == (quantile is None):
        raise ValueError("exactly one of threshold or quantile must be given")

    fwd = forward_return(returns_panel, horizon, from_returns=True)
    n_days, _ = fwd.shape
    out = np.full_like(fwd, np.nan)

    for t in range(n_days):
        row = fwd[t]
        mask = ~np.isnan(row)
        if mask.sum() < 1:
            continue
        valid = row[mask]
        if threshold is not None:
            cutoff = threshold if kind == "spike" else -threshold
        else:
            cutoff = np.quantile(valid, quantile)
        lbl = (valid > cutoff) if kind == "spike" else (valid < cutoff)
        out[t, mask] = lbl.astype(np.float64)
    return out
