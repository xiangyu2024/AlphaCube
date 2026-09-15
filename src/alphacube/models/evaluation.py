"""Scoring for model-aggregate prediction panels: IC, rolling per-stock IC,
long-short Sharpe variants, and magnitude/calibration metrics.

Adjacent to `metrics.py` (which scores mined expression panels against
forward returns for the mining reward signal), not a replacement for it:
this module scores arbitrary `(n_days, n_instruments)` prediction panels
against whatever target panel `targets.py`/`trainer.py` produced, which may
not be a return at all (volatility, rank, class, event probability).

IC-style metrics (#1, #2) always compare `prediction` against
`realized_target` — whatever that target's units are. Sharpe-style metrics
(#3, #4) always trade on `returns_panel_for_pnl`, the *actual* realized
return panel, regardless of what the model predicted — a volatility model's
predictions still rank instruments, and that ranking's P&L is measured on
real returns, not on realized volatility.

IC/Sharpe are scale-invariant (rank-based); per explicit request, magnitude
matters too (`注意，绝对值重要`), so #5 reports bias/RMSE/calibration slope
pooled across all valid `(day, instrument)` pairs.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_MIN_CROSS_SECTIONAL_PAIRS = 5


def _daily_pearson(a: np.ndarray, b: np.ndarray, min_pairs: int = _MIN_CROSS_SECTIONAL_PAIRS) -> np.ndarray:
    """Per-day Pearson correlation between two (n_days, n_instruments)
    arrays; days with fewer than `min_pairs` finite pairs are NaN."""
    n_days = a.shape[0]
    out = np.full(n_days, np.nan, dtype=np.float64)
    for t in range(n_days):
        x, y = a[t], b[t]
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < min_pairs:
            continue
        xs, ys = x[mask], y[mask]
        if np.std(xs) == 0 or np.std(ys) == 0:
            continue
        out[t] = np.corrcoef(xs, ys)[0, 1]
    return out


def _daily_rank(x: np.ndarray) -> np.ndarray:
    return pd.DataFrame(x).rank(axis=1, na_option="keep").to_numpy()


def cross_sectional_ic(
    prediction_panel: np.ndarray, realized_target_panel: np.ndarray
) -> Dict[str, float]:
    """Metric #1: per-day cross-sectional Pearson IC and Spearman rank IC.

    Days with fewer than 5 finite `(prediction, target)` pairs are skipped.
    Returns `mean_ic`/`std_ic`/`icir` and the rank-IC equivalents.
    """
    pred = np.asarray(prediction_panel, dtype=np.float64)
    tgt = np.asarray(realized_target_panel, dtype=np.float64)

    ic_series = _daily_pearson(pred, tgt)
    rank_ic_series = _daily_pearson(_daily_rank(pred), _daily_rank(tgt))

    mean_ic, std_ic = float(np.nanmean(ic_series)), float(np.nanstd(ic_series))
    mean_rank_ic, std_rank_ic = float(np.nanmean(rank_ic_series)), float(np.nanstd(rank_ic_series))
    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "icir": mean_ic / std_ic if std_ic > 0 else float("nan"),
        "mean_rank_ic": mean_rank_ic,
        "std_rank_ic": std_rank_ic,
        "rank_icir": mean_rank_ic / std_rank_ic if std_rank_ic > 0 else float("nan"),
    }


def rolling_single_stock_ic(
    prediction_panel: np.ndarray,
    realized_target_panel: np.ndarray,
    window: int = 60,
    min_periods: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Metric #2: for EACH instrument independently, the rolling-window
    correlation between its own prediction series and its own realized-
    target series (time-series predictive skill, not cross-sectional
    ranking quality — fundamentally different from `cross_sectional_ic`).

    Returns `(rolling_ic_panel, summary)` where `rolling_ic_panel` is
    `(n_days, n_instruments)` and `summary` has the mean/median/std, across
    instruments, of each instrument's own time-averaged rolling IC.
    """
    pred = np.asarray(prediction_panel, dtype=np.float64)
    tgt = np.asarray(realized_target_panel, dtype=np.float64)
    n_days, n_instruments = pred.shape
    min_periods = min_periods if min_periods is not None else max(5, window // 3)

    out = np.full((n_days, n_instruments), np.nan, dtype=np.float64)
    for i in range(n_instruments):
        s_pred = pd.Series(pred[:, i])
        s_tgt = pd.Series(tgt[:, i])
        out[:, i] = s_pred.rolling(window, min_periods=min_periods).corr(s_tgt).to_numpy()

    # An instrument whose window is ever entirely constant (e.g. a sparsely
    # predicted event-classifier column) yields an all-NaN rolling-IC
    # column; nanmean/nanstd on that "empty slice" is expected here, not a
    # bug, so the RuntimeWarning is suppressed the same way
    # metrics.py's zscore_rows does for the analogous situation.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        per_stock_time_avg = np.nanmean(out, axis=0)
        summary = {
            "mean_of_stock_ic": float(np.nanmean(per_stock_time_avg)),
            "median_of_stock_ic": float(np.nanmedian(per_stock_time_avg)),
            "std_of_stock_ic": float(np.nanstd(per_stock_time_avg)),
        }
    return out, summary


def _annualized_sharpe(daily_returns: np.ndarray, periods_per_year: int = 252) -> float:
    valid = daily_returns[np.isfinite(daily_returns)]
    if valid.size < 2:
        return float("nan")
    std = float(np.std(valid))
    if std == 0:
        return float("nan")
    return float(np.mean(valid) / std * np.sqrt(periods_per_year))


def _continuous_long_short_daily_returns(
    prediction_panel: np.ndarray, returns_panel_for_pnl: np.ndarray, frac: float
) -> np.ndarray:
    """Metric #3 helper: each day, select the top/bottom `frac` of
    instruments by prediction (restricted to instruments with a finite
    realized return), weight each selected name by its *demeaned*
    prediction value (continuous, not equal-weight) normalized to unit
    gross exposure, then compute that day's portfolio return on realized
    returns. Demeaning within the selected universe guarantees dollar
    neutrality (net weight = 0) by construction.
    """
    pred = np.asarray(prediction_panel, dtype=np.float64)
    pnl_ret = np.asarray(returns_panel_for_pnl, dtype=np.float64)
    n_days = pred.shape[0]
    out = np.full(n_days, np.nan, dtype=np.float64)

    for t in range(n_days):
        p, r = pred[t], pnl_ret[t]
        mask = np.isfinite(p) & np.isfinite(r)
        n_valid = int(mask.sum())
        if n_valid < 2:
            continue
        idx = np.flatnonzero(mask)
        vals = p[idx]
        k = max(1, int(np.floor(n_valid * frac)))
        order = np.argsort(vals)
        universe = np.union1d(order[:k], order[-k:])
        universe_vals = vals[universe]
        demeaned = universe_vals - universe_vals.mean()
        gross = np.sum(np.abs(demeaned))
        if gross == 0:
            continue
        weights = demeaned / gross
        out[t] = float(np.sum(weights * r[idx[universe]]))
    return out


def _decile_long_short_daily_returns(
    prediction_panel: np.ndarray, returns_panel_for_pnl: np.ndarray, decile_frac: float
) -> np.ndarray:
    """Metric #4 helper: fixed equal-weight top-`decile_frac` long,
    bottom-`decile_frac` short (spread return), restricted to instruments
    with a finite realized return."""
    pred = np.asarray(prediction_panel, dtype=np.float64)
    pnl_ret = np.asarray(returns_panel_for_pnl, dtype=np.float64)
    n_days = pred.shape[0]
    out = np.full(n_days, np.nan, dtype=np.float64)

    for t in range(n_days):
        p, r = pred[t], pnl_ret[t]
        mask = np.isfinite(p) & np.isfinite(r)
        n_valid = int(mask.sum())
        if n_valid < 2:
            continue
        idx = np.flatnonzero(mask)
        vals, rets = p[idx], r[idx]
        k = max(1, int(np.floor(n_valid * decile_frac)))
        order = np.argsort(vals)
        bottom_idx, top_idx = order[:k], order[-k:]
        out[t] = float(rets[top_idx].mean() - rets[bottom_idx].mean())
    return out


@dataclass
class EvaluationResult:
    """Flat, tabulate-friendly result for one (target, horizon) combination."""

    target_name: str
    horizon: int
    n_valid_pairs: int
    mean_ic: float
    std_ic: float
    icir: float
    mean_rank_ic: float
    std_rank_ic: float
    rank_icir: float
    rolling_ic_mean_of_stock: float
    rolling_ic_median_of_stock: float
    rolling_ic_std_of_stock: float
    long_short_sharpe: float
    long_short_decile_sharpe: float
    mean_bias: float
    rmse: float
    calibration_slope: float
    calibration_intercept: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_target(
    prediction_panel: np.ndarray,
    realized_target_panel: np.ndarray,
    returns_panel_for_pnl: np.ndarray,
    target_name: str = "target",
    horizon: int = 0,
    rolling_window: int = 60,
    long_short_frac: float = 0.3,
    decile_frac: float = 0.1,
    periods_per_year: int = 252,
) -> EvaluationResult:
    """Compute all five metric groups for one prediction panel and pack them
    into one flat `EvaluationResult`.

    `realized_target_panel` is whatever `prediction_panel` was trained to
    predict (return, volatility, rank, ...); `returns_panel_for_pnl` is
    always the actual realized-return panel, used only for the Sharpe-style
    metrics (#3/#4) so P&L is always measured on real returns even when the
    model's target isn't a return.
    """
    pred = np.asarray(prediction_panel, dtype=np.float64)
    tgt = np.asarray(realized_target_panel, dtype=np.float64)
    pnl_ret = np.asarray(returns_panel_for_pnl, dtype=np.float64)

    ic_metrics = cross_sectional_ic(pred, tgt)
    _, rolling_summary = rolling_single_stock_ic(pred, tgt, window=rolling_window)

    ls_sharpe = _annualized_sharpe(
        _continuous_long_short_daily_returns(pred, pnl_ret, long_short_frac), periods_per_year
    )
    decile_sharpe = _annualized_sharpe(
        _decile_long_short_daily_returns(pred, pnl_ret, decile_frac), periods_per_year
    )

    # Magnitude/calibration metrics (#5), pooled across all valid pairs.
    mask = np.isfinite(pred) & np.isfinite(tgt)
    n_valid_pairs = int(mask.sum())
    pred_valid, tgt_valid = pred[mask], tgt[mask]
    if n_valid_pairs >= 2 and np.std(pred_valid) > 0:
        mean_bias = float(np.mean(pred_valid - tgt_valid))
        rmse = float(np.sqrt(np.mean((pred_valid - tgt_valid) ** 2)))
        slope, intercept = np.polyfit(pred_valid, tgt_valid, 1)
        calibration_slope, calibration_intercept = float(slope), float(intercept)
    else:
        mean_bias = rmse = calibration_slope = calibration_intercept = float("nan")

    return EvaluationResult(
        target_name=target_name,
        horizon=horizon,
        n_valid_pairs=n_valid_pairs,
        long_short_sharpe=ls_sharpe,
        long_short_decile_sharpe=decile_sharpe,
        mean_bias=mean_bias,
        rmse=rmse,
        calibration_slope=calibration_slope,
        calibration_intercept=calibration_intercept,
        rolling_ic_mean_of_stock=rolling_summary["mean_of_stock_ic"],
        rolling_ic_median_of_stock=rolling_summary["median_of_stock_ic"],
        rolling_ic_std_of_stock=rolling_summary["std_of_stock_ic"],
        **ic_metrics,
    )


def leaderboard(results: List[EvaluationResult]) -> pd.DataFrame:
    """Tabulate a list of `EvaluationResult` into one leaderboard DataFrame,
    sorted by ICIR descending."""
    df = pd.DataFrame([r.to_dict() for r in results])
    if not df.empty and "icir" in df:
        df = df.sort_values("icir", ascending=False, na_position="last").reset_index(drop=True)
    return df
