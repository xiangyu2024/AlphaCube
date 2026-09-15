"""Cross-sectional IC and friends on (D, S) panels. ``isfinite`` masks
everywhere (inf is not a value), a minimum of 5 valid pairs per day, and a
zero-variance day is NaN, so the numpy path (research / pool gate) and the
torch path (GP fitness) agree on which days count.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import warnings

import numpy as np
import pandas as pd
import torch

MIN_PAIRS = 5


def cs_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    n = valid.sum(axis=1)
    az = np.where(valid, a, 0.0)
    bz = np.where(valid, b, 0.0)
    nf = np.maximum(n, 1)
    am = az.sum(1) / nf
    bm = bz.sum(1) / nf
    ac = np.where(valid, a - am[:, None], 0.0)
    bc = np.where(valid, b - bm[:, None], 0.0)
    cov = (ac * bc).sum(1)
    va = (ac ** 2).sum(1)
    vb = (bc ** 2).sum(1)
    ok = (n >= MIN_PAIRS) & (va > 0) & (vb > 0)
    out = np.full(a.shape[0], np.nan)
    out[ok] = cov[ok] / np.sqrt(va[ok] * vb[ok])
    return out


def rank_rows(x: np.ndarray) -> np.ndarray:
    return pd.DataFrame(np.asarray(x, dtype=np.float64)).rank(axis=1).to_numpy()


def zscore_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)         # all-NaN rows are legitimate
        m = np.nanmean(x, axis=1, keepdims=True)
        s = np.nanstd(x, axis=1, keepdims=True)
        z = (x - m) / s
    z[~np.isfinite(z)] = np.nan
    return z


def nanmean_quiet(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    f = x[np.isfinite(x)]
    return float(f.mean()) if f.size else float("nan")


def ic_summary(panel: np.ndarray, target: np.ndarray, ann_factor: float = 252.0) -> Dict[str, float]:
    ic = cs_pearson(panel, target)
    ric = cs_pearson(rank_rows(panel), rank_rows(target))
    f = ic[np.isfinite(ic)]
    out = {
        "ic_mean": nanmean_quiet(ic),
        "ic_std": float(f.std()) if f.size > 1 else float("nan"),
        "icir": float(f.mean() / f.std()) if f.size > 1 and f.std() > 0 else float("nan"),
        "ic_tstat": float(f.mean() / f.std() * np.sqrt(f.size)) if f.size > 1 and f.std() > 0 else float("nan"),
        "ic_pos": float((f > 0).mean()) if f.size else float("nan"),
        "rank_ic": nanmean_quiet(ric),
        "coverage": float(np.isfinite(panel).mean()),
        "n_days": int(f.size),
    }
    return out


def quantile_longshort(panel: np.ndarray, fwd: np.ndarray, quantiles: int = 5) -> Dict[str, float]:
    """Equal-weight top-minus-bottom quantile daily P&L on ``fwd`` (no costs)."""
    r = rank_rows(panel)
    with np.errstate(invalid="ignore"):
        n = np.isfinite(r).sum(axis=1, keepdims=True)
        q = np.floor((r - 1) / np.maximum(n, 1) * quantiles)
    q = np.where(np.isfinite(r) & np.isfinite(fwd), q, np.nan)
    top = np.where(q == quantiles - 1, fwd, np.nan)
    bot = np.where(q == 0, fwd, np.nan)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ls = np.nanmean(top, axis=1) - np.nanmean(bot, axis=1)
    ls = ls[np.isfinite(ls)]
    if ls.size < 2:
        return {"ls_ann_ret": float("nan"), "ls_sharpe": float("nan"), "ls_maxdd": float("nan"), "ls_win": float("nan")}
    equity = np.cumsum(ls)
    dd = equity - np.maximum.accumulate(equity)
    return {
        "ls_ann_ret": float(ls.mean() * 252),
        "ls_sharpe": float(ls.mean() / ls.std() * np.sqrt(252)) if ls.std() > 0 else float("nan"),
        "ls_maxdd": float(dd.min()),
        "ls_win": float((ls > 0).mean()),
    }


def batched_ic(tensors: Sequence[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    """Mean daily cross-sectional Pearson IC per factor, in torch (GP fitness)."""
    out = torch.full((len(tensors),), float("nan"), dtype=torch.float32)
    with torch.no_grad():
        tfin = torch.isfinite(target)
        for i, f in enumerate(tensors):
            valid = torch.isfinite(f) & tfin
            n = valid.sum(dim=1)
            fz = torch.where(valid, f, torch.zeros_like(f))
            tz = torch.where(valid, target, torch.zeros_like(target))
            nf = n.clamp(min=1).float()
            fm = fz.sum(1) / nf
            tm = tz.sum(1) / nf
            fc = torch.where(valid, f - fm.unsqueeze(1), torch.zeros_like(f))
            tc = torch.where(valid, target - tm.unsqueeze(1), torch.zeros_like(target))
            cov = (fc * tc).sum(1)
            fv = fc.pow(2).sum(1)
            tv = tc.pow(2).sum(1)
            ok = (n >= MIN_PAIRS) & (fv > 0) & (tv > 0)
            days = cov[ok] / (fv[ok].sqrt() * tv[ok].sqrt())
            if days.numel() > 0:
                out[i] = days.mean().cpu()
    return out


def sampled_abs_corr(a: np.ndarray, b: np.ndarray, sample_days: int = 250) -> float:
    D = a.shape[0]
    idx = np.linspace(0, D - 1, num=min(sample_days, D)).astype(int)
    x, y = a[idx].reshape(-1), b[idx].reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan")
    x, y = x[m], y[m]
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(abs(np.corrcoef(x, y)[0, 1]))




# ---- ported from the previous project's eval.calculator ---------------------------

cross_sectional_pearson = cs_pearson


def daily_returns_from_close(data, close_field: str = "Close") -> np.ndarray:
    """returns[t] = price[t] / price[t-1] - 1 (row 0 NaN): the convention
    ``models.targets`` expects, NOT ``forward_returns`` (which is indexed at t)."""
    price = np.asarray(data.get_field(close_field), dtype=np.float64)
    out = np.full_like(price, np.nan)
    if price.shape[0] > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            out[1:] = price[1:] / price[:-1] - 1.0
    return out


def combine_panels(panels: Sequence[np.ndarray], weights: Sequence[float], nan_as_zero: bool = True) -> np.ndarray:
    """Weighted sum of per-day z-scored panels (the pool's combined signal)."""
    w = np.asarray(weights, dtype=np.float64)
    if len(panels) != len(w):
        raise ValueError("panels and weights must have the same length")
    out = None
    for p, wi in zip(panels, w):
        z = zscore_rows(p)
        if nan_as_zero:
            z = np.nan_to_num(z, nan=0.0)
        out = wi * z if out is None else out + wi * z
    return out


def _long_short_returns(alpha: np.ndarray, fwd: np.ndarray, top_frac: float) -> np.ndarray:
    n_days = alpha.shape[0]
    out = np.full(n_days, np.nan)
    for t in range(n_days):
        a, r = alpha[t], fwd[t]
        m = np.isfinite(a) & np.isfinite(r)
        n = int(m.sum())
        if n < 2:
            continue
        av, rv = a[m], r[m]
        k = max(1, int(np.floor(n * top_frac)))
        order = np.argsort(av)
        out[t] = rv[order[-k:]].mean() - rv[order[:k]].mean()
    return out


def backtest_metrics(ic_series: np.ndarray, alpha_values: Optional[np.ndarray] = None,
                     forward_returns: Optional[np.ndarray] = None, top_frac: float = 0.1,
                     periods_per_year: int = 252) -> Dict[str, float]:
    """ICIR from the IC series plus, when alpha/returns are given, an equal-weight
    top-minus-bottom ``top_frac`` portfolio's annualised return / Sharpe / max
    drawdown (diagnostic: one holding period, no costs, no limits)."""
    ic = np.asarray(ic_series, dtype=np.float64)
    f = ic[np.isfinite(ic)]
    ic_mean = float(f.mean()) if f.size else float("nan")
    ic_std = float(f.std()) if f.size else float("nan")
    out: Dict[str, float] = {"ic_mean": ic_mean, "ic_std": ic_std,
                             "icir": float(ic_mean / ic_std) if ic_std > 0 else float("nan")}
    if alpha_values is not None and forward_returns is not None:
        port = _long_short_returns(np.asarray(alpha_values, dtype=np.float64),
                                   np.asarray(forward_returns, dtype=np.float64), top_frac)
        v = port[np.isfinite(port)]
        if v.size:
            mean, std = float(v.mean()), float(v.std())
            cum = np.cumprod(1.0 + v)
            dd = cum / np.maximum.accumulate(cum) - 1.0
            out.update({"annualized_return": float((1.0 + mean) ** periods_per_year - 1.0),
                        "sharpe": float(mean / std * np.sqrt(periods_per_year)) if std > 0 else float("nan"),
                        "max_drawdown": float(dd.min())})
        else:
            out.update({"annualized_return": float("nan"), "sharpe": float("nan"), "max_drawdown": float("nan")})
    return out


__all__ = ["MIN_PAIRS", "cs_pearson", "cross_sectional_pearson", "rank_rows", "zscore_rows", "nanmean_quiet",
           "ic_summary", "quantile_longshort", "batched_ic", "sampled_abs_corr", "daily_returns_from_close",
           "combine_panels", "backtest_metrics"]
