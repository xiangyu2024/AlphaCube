"""Portfolio construction: raw signal -> target portfolio weight vector.

Pipeline stage: alpha mining -> model aggregate -> **portfolio construction**
(this module) -> portfolio optimization (`portfolio/optimize_ashare.py`,
`portfolio/optimize_us.py`) -> execution (`execution/`).

Everything here is single-day, cross-sectional, and ``pandas``-based: a day's
signal (a model prediction or an already-blended score) is a ``pd.Series``
indexed by instrument symbol, and every function in this module takes and/or
returns that same shape. We use ``pd.Series`` rather than raw ``numpy``
arrays deliberately -- at this layer, instrument alignment/indexing is the
whole game (signals, caps, and blends all need to line up by symbol, and
symbol universes can differ day to day), and a ``pd.Series`` carries that
alignment for free.

Three independent pieces live here:

- ``transform_signal`` -- turn one day's raw signal into an unnormalized
  target-weight-shaped ``pd.Series`` using one of several named "execution
  variants" (long/short, long-only, rank-based, softmax, top-N, ...).
- ``cap_and_renormalize_weights`` -- take an unnormalized weight vector and
  rescale it to hit a target gross/net exposure while enforcing a per-name
  cap, returning a small diagnostic-carrying result rather than a bare
  ``Series`` since downstream (portfolio optimization) needs to know whether
  the cap made the target exposure unreachable.
- ``blend_signals_by_rolling_icir`` -- combine multiple named signal panels
  (e.g. several models' predictions) into one blended score panel using
  rolling information-coefficient-based weights, ahead of the two functions
  above (which then operate on the blended panel one day at a time).

This module is intentionally self-contained: pure functions, no dependency
on `models/`, `portfolio/optimize_*`, or `execution/` (the sibling layers
other engineers are building), and no new third-party dependency beyond the
project's existing ``pandas``/``numpy``.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..metrics import zscore_rows

# ---------------------------------------------------------------------------
# transform_signal
# ---------------------------------------------------------------------------

#: Variant names handled directly by name (as opposed to the parametric
#: ``top<N>_long`` family, see `top_n_from_variant`/`is_supported_variant`).
KNOWN_VARIANTS = (
    "raw_continuous",
    "long_short",  # alias of raw_continuous
    "long_only",
    "shift_to_long",
    "demean_shift_long",
    "rank_long",
    "softmax_long",
    "top_positive_long",
    "top_rank_long",
)

_TOPN_RE = re.compile(r"^top(\d+)_long$")


def top_n_from_variant(variant: str) -> Optional[int]:
    """Parse ``N`` out of a ``"top<N>_long"`` variant string, else ``None``.

    ``"top0_long"`` (N == 0, a degenerate "keep nothing" request) is treated
    as *not* a valid topN variant and returns ``None`` -- callers that only
    check ``top_n_from_variant(...) is not None`` therefore correctly reject
    it too, via `is_supported_variant`.
    """
    match = _TOPN_RE.match(variant)
    if not match:
        return None
    n = int(match.group(1))
    return n if n > 0 else None


def is_supported_variant(variant: str) -> bool:
    """True if `transform_signal` knows how to handle ``variant``.

    Other engineers (config validation, CLI arg parsing, etc.) should import
    this rather than hand-rolling their own membership/regex check.
    """
    return variant in KNOWN_VARIANTS or top_n_from_variant(variant) is not None


def _clean_signal(signal: pd.Series) -> pd.Series:
    """Coerce to numeric float and replace NaN/inf with 0, before any
    variant-specific logic runs. A NaN signal is treated as "no signal"
    (0), not as "unknown"/dropped, and specifically not as a negative
    value -- which matters for the no-short variants' ``clip(lower=0)``
    spirit: a NaN shouldn't accidentally end up clipped to a large
    negative-turned-zero the same way a real negative signal would, it
    should just be inert from the start.
    """
    s = pd.to_numeric(signal, errors="coerce").astype(float)
    s = s.replace([np.inf, -np.inf], np.nan)
    return s.fillna(0.0)


def _active_mask(s: pd.Series) -> pd.Series:
    """"Active" names = nonzero raw (cleaned) signal."""
    return s != 0.0


def _percentile_rank(s: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank in (0, 1] (1/n .. 1.0)."""
    return s.rank(pct=True, method="average")


def _demean_shift_long(s: pd.Series) -> pd.Series:
    out = pd.Series(0.0, index=s.index)
    active = _active_mask(s)
    if not active.any():
        return out
    x = s - s[active].mean()
    out.loc[active] = x[active] - x[active].min()
    return out


def _rank_long(s: pd.Series) -> pd.Series:
    out = pd.Series(0.0, index=s.index)
    active = _active_mask(s)
    if not active.any():
        return out
    r = _percentile_rank(s[active])
    out.loc[active] = r**2
    return out


def _softmax_long(s: pd.Series, temperature: float) -> pd.Series:
    active = _active_mask(s)
    if not active.any():
        return pd.Series(0.0, index=s.index)
    x = s[active]
    std = x.std(ddof=0)
    if not np.isfinite(std) or std < 1e-12:
        # Degenerate day (no dispersion among active names): softmax has
        # nothing to differentiate on, fall back to plain long-only shape.
        return s.clip(lower=0.0)
    z = ((x - x.mean()) / std).clip(-8.0, 8.0) / temperature
    numerator = np.exp(z - z.max())
    out = pd.Series(0.0, index=s.index)
    out.loc[active] = numerator
    return out


def _top_positive_long(s: pd.Series, top_quantile: float) -> pd.Series:
    out = pd.Series(0.0, index=s.index)
    positive = s[s > 0.0]
    if positive.empty:
        return out
    threshold = positive.quantile(top_quantile)
    keep = positive[positive > threshold]
    out.loc[keep.index] = keep
    return out


def _top_rank_long(s: pd.Series, top_quantile: float) -> pd.Series:
    out = pd.Series(0.0, index=s.index)
    active = _active_mask(s)
    if not active.any():
        return out
    r = _percentile_rank(s[active])
    keep = r[r >= top_quantile]
    out.loc[keep.index] = keep**2
    return out


def _topn_long(s: pd.Series, n: int) -> pd.Series:
    out = pd.Series(0.0, index=s.index)
    if n <= 0 or s.empty:
        return out
    selected_idx = s.sort_values(ascending=False).index[:n]
    selected = s.loc[selected_idx]
    r = _percentile_rank(selected)
    out.loc[selected_idx] = r**2
    return out


def transform_signal(signal: pd.Series, variant: str, **params) -> pd.Series:
    """Turn one day's raw cross-sectional signal into a weight-shaped Series.

    `signal` is a `pd.Series` indexed by instrument symbol for a SINGLE day.
    `variant` selects the transform (see module docstring / `KNOWN_VARIANTS`
    / the ``top<N>_long`` pattern). Unrecognized variants raise `ValueError`
    -- callers that need to validate config-supplied variant strings ahead
    of time should use `is_supported_variant` instead of catching this.

    Every variant first runs the input through `_clean_signal` (numeric
    coercion, NaN/inf -> 0) before any variant-specific logic.
    """
    s = _clean_signal(signal)

    if variant in ("raw_continuous", "long_short"):
        return s.clip(-1.0, 1.0)

    if variant == "long_only":
        return s.clip(-1.0, 1.0).clip(lower=0.0)

    if variant == "shift_to_long":
        if np.isclose(s.max(), s.min()):
            return pd.Series(0.0, index=s.index)
        return s - s.min()

    if variant == "demean_shift_long":
        return _demean_shift_long(s)

    if variant == "rank_long":
        return _rank_long(s)

    if variant == "softmax_long":
        temperature = float(params.get("temperature", 1.0))
        return _softmax_long(s, temperature=temperature)

    if variant == "top_positive_long":
        top_quantile = float(params.get("top_quantile", 0.80))
        return _top_positive_long(s, top_quantile=top_quantile)

    if variant == "top_rank_long":
        top_quantile = float(params.get("top_quantile", 0.80))
        return _top_rank_long(s, top_quantile=top_quantile)

    n = top_n_from_variant(variant)
    if n is not None:
        return _topn_long(s, n=n)

    raise ValueError(
        f"Unsupported transform_signal variant: {variant!r}. "
        f"Known variants: {KNOWN_VARIANTS}, or 'top<N>_long'."
    )


# ---------------------------------------------------------------------------
# cap_and_renormalize_weights
# ---------------------------------------------------------------------------


@dataclass
class CappedWeights:
    """Result of `cap_and_renormalize_weights`.

    `achieved_exposure`/`converged` let downstream (portfolio optimization)
    tell "hit the target exactly" apart from "the per-name cap made the
    target exposure unreachable given this many names" without having to
    re-derive it from `weights` itself.
    """

    weights: pd.Series
    achieved_exposure: float
    converged: bool
    iterations: int


def _exposure(weights: pd.Series, allow_short: bool) -> float:
    return float(weights.abs().sum()) if allow_short else float(weights.sum())


def cap_and_renormalize_weights(
    raw_weights: pd.Series,
    target_exposure: float,
    max_single_name_weight: float,
    allow_short: bool = False,
    max_iterations: int = 50,
) -> CappedWeights:
    """Scale `raw_weights` to hit `target_exposure` while capping any single
    name's absolute weight at `max_single_name_weight`.

    Exposure is ``sum(weights)`` when ``allow_short=False`` (long-only) or
    ``sum(abs(weights))`` when ``allow_short=True`` (gross).

    Algorithm ("waterfill"): proportionally scale all weights to hit the
    target exposure; any name that now exceeds the cap is clipped to the cap
    and locked, and the *remaining* target exposure (target minus what the
    locked names now contribute) is re-derived from the still-free names'
    original relative proportions -- which is equivalent to redistributing
    the excess proportionally across them, since it's a uniform rescale of
    their previous values. Repeat until no free name exceeds the cap, the
    free set is exhausted (cap makes the target unreachable), or
    `max_iterations` is hit.

    Returns a `CappedWeights` rather than a bare Series: if the cap makes
    `target_exposure` genuinely unreachable (e.g. `max_single_name_weight`
    too small for `target_exposure` given the number of names), this
    returns the best-effort (fully-capped, cap-respecting) weights with
    ``converged=False`` and ``achieved_exposure < target_exposure`` rather
    than silently returning a wrong-total-exposure result.
    """
    index = raw_weights.index
    raw = raw_weights.astype(float)
    max_cap = abs(float(max_single_name_weight))
    eps = 1e-12

    if len(index) == 0:
        return CappedWeights(weights=raw.copy(), achieved_exposure=0.0, converged=True, iterations=0)

    locked_values = pd.Series(0.0, index=index)
    locked = pd.Series(False, index=index)
    weights = locked_values.copy()
    converged = False
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        free_mask = ~locked
        if not free_mask.any():
            # Every name is pinned at the cap; nothing left to redistribute
            # into. This is the "target unreachable" terminal state.
            weights = locked_values.copy()
            break

        locked_exposure = _exposure(locked_values[locked], allow_short) if locked.any() else 0.0
        remaining_target = target_exposure - locked_exposure

        free_raw = raw[free_mask]
        free_raw_exposure = _exposure(free_raw, allow_short)

        if abs(free_raw_exposure) <= eps:
            # Free names carry no raw exposure to scale from. If there's no
            # remaining target left either, we're trivially done (converged);
            # otherwise the remaining target is unreachable from here.
            weights = locked_values.copy()
            weights[free_mask] = 0.0
            converged = abs(remaining_target) <= eps
            break

        scale = remaining_target / free_raw_exposure
        scaled_free = free_raw * scale

        weights = locked_values.copy()
        weights[free_mask] = scaled_free

        over_free = (weights.abs() > max_cap + eps) & free_mask
        if not over_free.any():
            converged = True
            break

        newly_locked_idx = weights.index[over_free]
        locked_values.loc[newly_locked_idx] = np.sign(weights.loc[newly_locked_idx]) * max_cap
        locked.loc[newly_locked_idx] = True
    else:
        # Loop finished all max_iterations without breaking (converged or
        # exhausted). Guard against returning a still-over-cap result: clip
        # whatever remains as a final safety net.
        pass

    if not converged:
        # Whatever state `weights` is in (unreachable-terminal, zero-free-
        # exposure, or max_iterations exhausted mid-flight), make sure the
        # cap is never violated in the returned result even if the exposure
        # target wasn't hit.
        over = weights.abs() > max_cap + eps
        if over.any():
            weights.loc[over] = np.sign(weights.loc[over]) * max_cap

    achieved = _exposure(weights, allow_short)
    return CappedWeights(
        weights=weights, achieved_exposure=achieved, converged=converged, iterations=iterations
    )


# ---------------------------------------------------------------------------
# blend_signals_by_rolling_icir
# ---------------------------------------------------------------------------


def _zscore_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-day (row-wise) cross-sectional z-score of a (n_days, n_instruments)
    panel. Thin pandas wrapper around `alphacube.metrics.zscore_rows`
    (the existing numpy implementation) so we don't duplicate the logic.
    """
    z = zscore_rows(panel.to_numpy(dtype=float))
    return pd.DataFrame(z, index=panel.index, columns=panel.columns)


def _daily_ic_series(signal_panel: pd.DataFrame, target_panel: pd.DataFrame, rank: bool) -> pd.Series:
    """Per-day cross-sectional Pearson (or Spearman, if `rank`) correlation
    between `signal_panel` and `target_panel`, both (n_days, n_instruments).
    """
    method = "spearman" if rank else "pearson"
    out = pd.Series(np.nan, index=signal_panel.index, dtype=float)
    for date in signal_panel.index:
        x = signal_panel.loc[date]
        y = target_panel.loc[date]
        mask = x.notna() & y.notna()
        if int(mask.sum()) < 2:
            continue
        xs, ys = x[mask], y[mask]
        if xs.std(ddof=0) == 0 or ys.std(ddof=0) == 0:
            continue
        out.loc[date] = xs.corr(ys, method=method)
    return out


def _warn_if_target_looks_unlagged(
    signal_panels: Dict[str, pd.DataFrame], realized_target_panel: pd.DataFrame
) -> None:
    """Heuristic leakage smell-check, not a hard requirement (see docstring
    of `blend_signals_by_rolling_icir` for why lagging is entirely the
    caller's responsibility): warn if `realized_target_panel` has valid data
    as recent as (or more recent than) every signal panel's most recent
    valid row, since a target that isn't lagged behind the signals it's
    meant to be evaluated against is the classic embargo/lookahead bug.
    """
    target_valid = realized_target_panel.dropna(how="all")
    if target_valid.empty:
        return
    target_last = target_valid.index.max()

    signal_lasts = []
    for panel in signal_panels.values():
        valid = panel.dropna(how="all")
        if not valid.empty:
            signal_lasts.append(valid.index.max())
    if not signal_lasts:
        return

    # Most conservative (earliest) "last available" date across signals.
    signal_last = min(signal_lasts)
    if target_last >= signal_last:
        warnings.warn(
            "blend_signals_by_rolling_icir: realized_target_panel's last "
            f"non-NaN row ({target_last!r}) is not earlier than the earliest "
            f"'last available row' among signal_panels ({signal_last!r}). "
            "This is the classic embargo/lookahead smell: realized_target_panel "
            "must already be lagged (shifted back by the forward-return "
            "horizon) by the CALLER before calling this function -- it is not "
            "inferred or applied here. Proceeding, but double-check upstream "
            "lagging.",
            stacklevel=2,
        )


def blend_signals_by_rolling_icir(
    signal_panels: Dict[str, pd.DataFrame],
    realized_target_panel: pd.DataFrame,
    *,
    lookback: int = 60,
    min_obs: int = 20,
    min_abs_ic: float = 0.0,
    icir_clip: float = 2.0,
    prefer_rank_ic: bool = True,
) -> pd.DataFrame:
    """Combine multiple named signal panels into one blended score panel
    using rolling information-coefficient-based (ICIR) weighting.

    Every value in `signal_panels` and `realized_target_panel` must be a
    `(n_days, n_instruments)` DataFrame sharing the exact same
    `DatetimeIndex` and instrument columns.

    **Embargo/lag responsibility is entirely the caller's.** `realized_target_panel`
    for day `d` typically depends on future data up to `d + horizon` (e.g. a
    forward return). This function does NOT know or infer `horizon` -- the
    caller must pass in a `realized_target_panel` that has already been
    shifted/lagged so that, as of day `d`, using it does not leak
    information from beyond day `d`. As a (soft, heuristic-only) sanity
    check, a `UserWarning` is raised if `realized_target_panel`'s last
    non-NaN row looks suspiciously recent relative to the signal panels'
    (see `_warn_if_target_looks_unlagged`) -- this is a smell detector, not
    a guarantee of correctness.

    For each day `t` and signal `name`: compute the daily cross-sectional IC
    (Pearson, or Spearman/rank-IC if `prefer_rank_ic`) between that signal
    and `realized_target_panel` for every day, then take the trailing
    `lookback`-day mean/std of that per-day IC series (requiring at least
    `min_obs` valid trailing observations, else NaN for that day/signal).
    `icir = mean_ic / std_ic` (falling back to `mean_ic` if `std_ic` is
    ~0/NaN), clipped to `[-icir_clip, icir_clip]`. Signals with
    `|mean_ic| < min_abs_ic` that day (or without enough trailing
    observations) get weight 0. Surviving signals' weights are L1-normalized
    (`icir / sum(|icir|)`) each day.

    The blended value for day `t`, instrument `i` is
    ``sum(weight_name * zscore(signal_panels[name].loc[t]))[i]`` where
    `zscore` is a per-day cross-sectional z-score computed fresh each day.

    Days before `lookback + min_obs` days of history exist are NaN by
    design -- there is no equal-weight (or any other) fallback for the
    warmup period; NaN is the honest answer for "not enough history yet",
    and it's on the caller to decide how to handle warmup NaNs.

    Returns
    -------
    A `(n_days, n_instruments)` DataFrame, same index/columns as
    `realized_target_panel`.
    """
    if not signal_panels:
        raise ValueError("signal_panels must contain at least one signal")

    names = list(signal_panels)
    index = realized_target_panel.index
    columns = realized_target_panel.columns

    for name, panel in signal_panels.items():
        if not panel.index.equals(index) or not panel.columns.equals(columns):
            raise ValueError(
                f"signal panel '{name}' is not aligned with realized_target_panel "
                "(all signal panels and realized_target_panel must share the exact "
                "same DatetimeIndex and instrument columns)"
            )

    _warn_if_target_looks_unlagged(signal_panels, realized_target_panel)

    zscored = {name: _zscore_panel(panel) for name, panel in signal_panels.items()}

    ic_df = pd.DataFrame(
        {name: _daily_ic_series(panel, realized_target_panel, rank=prefer_rank_ic) for name, panel in signal_panels.items()}
    )

    mean_ic_df = ic_df.rolling(window=lookback, min_periods=min_obs).mean()
    std_ic_df = ic_df.rolling(window=lookback, min_periods=min_obs).std()

    blended = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)

    # Hard warmup gate: no output before `lookback + min_obs` elapsed days,
    # regardless of how early pandas' rolling(min_periods=...) would
    # otherwise be willing to produce a (partial-window) value. This is a
    # deliberately conservative reading of "not enough history yet".
    warmup_days = lookback + min_obs
    n_days = len(index)

    for t in range(warmup_days - 1, n_days):
        date = index[t]
        mean_ic = mean_ic_df.loc[date]
        std_ic = std_ic_df.loc[date]

        eligible = mean_ic.notna()
        if not eligible.any():
            continue

        icir = mean_ic / std_ic
        icir = icir.where(std_ic.abs() > 1e-12, mean_ic)
        icir = icir.clip(-icir_clip, icir_clip)

        eligible = eligible & (mean_ic.abs() >= min_abs_ic)
        icir = icir.where(eligible, 0.0)

        l1 = icir.abs().sum()
        if l1 <= 1e-12:
            continue  # no surviving signal today: leave this day NaN

        weights_t = icir / l1

        cross_section = pd.DataFrame({name: zscored[name].loc[date] for name in names})
        blended.loc[date] = cross_section.mul(weights_t, axis=1).sum(axis=1)

    return blended
