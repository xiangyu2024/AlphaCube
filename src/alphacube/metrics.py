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
    """Per-row average ranks (ties averaged) of the finite cells, NaN elsewhere.
    pandas ranks inf as a value -- the largest one -- so an inf cell used to get
    rank n here while every IC and every book in this module skips it."""
    x = np.asarray(x, dtype=np.float64)
    return pd.DataFrame(np.where(np.isfinite(x), x, np.nan)).rank(axis=1).to_numpy()


def zscore_rows(x: np.ndarray) -> np.ndarray:
    """Per-row z-scores over the finite cells; NaN elsewhere. ``nanmean`` /
    ``nanstd`` take inf as a value, so ONE inf cell made the row's mean inf and
    every z-score of that day NaN: gpu12's third block was built from
    ``log(add(...))`` factors with a -inf on some stock each day, and its pool
    lost the whole test period that way -- coverage 0 for every weighted
    combination, a constant signal for the trees."""
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.isfinite(x), x, np.nan)
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


def torch_rank_rows(x) -> torch.Tensor:
    """``rank_rows`` in torch (float64): per-row average ranks of the finite cells."""
    return _avg_ranks_t(torch.as_tensor(np.asarray(x)))[0]


def _avg_ranks_t(x: torch.Tensor):
    """Average ranks (1-based, ties averaged) of each row's finite cells, NaN
    elsewhere, plus what the decile book reuses: the stable ascending order with
    the non-finite cells last, and the per-row finite count."""
    fin = torch.isfinite(x)
    s, order = torch.sort(torch.where(fin, x, torch.full_like(x, float("inf"))), dim=1, stable=True)
    lo = torch.searchsorted(s, s, right=False)          # a tie block spans sorted positions lo..hi-1
    hi = torch.searchsorted(s, s, right=True)
    avg = (lo + hi + 1).to(torch.float64) / 2.0
    ranks = torch.empty(x.shape, dtype=torch.float64).scatter_(1, order, avg)
    ranks[~fin] = float("nan")
    return ranks, order, fin.sum(dim=1)


def _cs_corr_days_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``cs_pearson`` in torch: one correlation per day over the pairwise-finite
    cells, NaN below ``MIN_PAIRS`` or without variance on either side."""
    valid = torch.isfinite(a) & torch.isfinite(b)
    n = valid.sum(dim=1)
    zero = torch.zeros((), dtype=a.dtype)
    nf = n.clamp(min=1).to(a.dtype)
    am = torch.where(valid, a, zero).sum(1) / nf
    bm = torch.where(valid, b, zero).sum(1) / nf
    ac = torch.where(valid, a - am.unsqueeze(1), zero)
    bc = torch.where(valid, b - bm.unsqueeze(1), zero)
    cov, va, vb = (ac * bc).sum(1), (ac * ac).sum(1), (bc * bc).sum(1)
    ok = (n >= MIN_PAIRS) & (va > 0) & (vb > 0)
    out = torch.full((a.shape[0],), float("nan"), dtype=a.dtype)
    out[ok] = cov[ok] / torch.sqrt(va[ok] * vb[ok])
    return out


def _book_turnover_t(order: torch.Tensor, n: torch.Tensor, top_frac: float) -> float:
    """Mean daily turnover of ``long_short_weights`` built from a stable sort order,
    as ``portfolio_returns`` counts it (0.5 * sum|w_t - w_t-1|, from an empty book)."""
    D, S = order.shape
    k = torch.clamp(torch.floor(n.to(torch.float64) * top_frac), min=1).to(torch.int64)
    pos = torch.arange(S).unsqueeze(0)
    ok = (n >= 2).unsqueeze(1)
    long_ = ok & (pos >= (n - k).unsqueeze(1)) & (pos < n.unsqueeze(1))
    short = ok & (pos < k.unsqueeze(1))
    kk = k.to(torch.float64).unsqueeze(1)
    ws = torch.where(long_, 0.5 / kk, torch.where(short, -0.5 / kk, torch.zeros((), dtype=torch.float64)))
    w = torch.zeros((D, S), dtype=torch.float64).scatter_(1, order, ws)
    prev = torch.cat([torch.zeros((1, S), dtype=torch.float64), w[:-1]])
    return float((0.5 * (w - prev).abs().sum(1)).mean())


def rank_ic_turnover(panel, target_ranks: torch.Tensor, top_frac: float = 0.1):
    """GP fitness ingredients from one sort of the panel: the daily rank-IC series
    (``cs_pearson(rank_rows(panel), rank_rows(target))`` -- ``ic_summary``'s
    ``rank_ic`` -- as a numpy array) and the mean daily turnover of the
    top/bottom ``top_frac`` book (``backtest_metrics``' ``turnover``; it does not
    depend on the factor's sign). ``target_ranks`` is ``torch_rank_rows(target)``."""
    ranks, order, n = _avg_ranks_t(torch.as_tensor(np.asarray(panel)))
    return _cs_corr_days_t(ranks, target_ranks).numpy(), _book_turnover_t(order, n, top_frac)


def ema_rows(panel: np.ndarray, halflife: float) -> np.ndarray:
    """Causal per-stock EMA along days: ``s_t = a * x_t + (1 - a) * s_t-1`` with
    ``a = 1 - 0.5 ** (1 / halflife)``. A missing day leaves the state as it was and
    is missing in the output (nothing to trade that day); a stock's first finite
    value starts its state. ``halflife <= 0`` returns the panel unchanged."""
    x = np.asarray(panel, dtype=np.float64)
    if halflife <= 0:
        return x
    a = 1.0 - 0.5 ** (1.0 / float(halflife))
    out = np.full_like(x, np.nan)
    state = np.full(x.shape[1], np.nan)
    for t in range(x.shape[0]):
        row = x[t]
        fin = np.isfinite(row)
        fresh = fin & ~np.isfinite(state)
        state = np.where(fresh, row, state)
        upd = fin & ~fresh
        state[upd] = a * row[upd] + (1.0 - a) * state[upd]
        out[t, fin] = state[fin]
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


def long_short_weights(alpha: np.ndarray, top_frac: float) -> np.ndarray:
    """(D, S) alpha -> (D, S) target weights: equal-weight long the top ``top_frac``,
    short the bottom, gross exposure 1 (so +0.5 long / -0.5 short) and net 0.
    Days with too few names are all-zero rather than NaN, so a gap in the alpha
    means "hold nothing", not "hold the previous book forever".

    Ties at a boundary go by symbol order (a stable sort). numpy's default sort
    is not stable, and which of the tied names it put in the book depended on
    the rest of the row: nudging one mid-ranked name changed the book in 12 of
    200 tied rows, i.e. turnover charged for trades nobody would make. With
    turnover now part of the GP fitness (``rank_ic_turnover`` builds the same
    book in torch) that noise would also have been selected against."""
    alpha = np.asarray(alpha, dtype=np.float64)
    D, S = alpha.shape
    fin = np.isfinite(alpha)
    n = fin.sum(axis=1)
    order = np.argsort(np.where(fin, alpha, np.inf), axis=1, kind="stable")     # non-finite last
    k = np.maximum(1, np.floor(n * top_frac).astype(np.int64))
    pos = np.arange(S)[None, :]
    ok = (n >= 2)[:, None]
    long_ = ok & (pos >= (n - k)[:, None]) & (pos < n[:, None])
    short = ok & (pos < k[:, None])
    w_sorted = np.where(long_, 0.5 / k[:, None], np.where(short, -0.5 / k[:, None], 0.0))
    w = np.zeros((D, S), dtype=np.float64)
    np.put_along_axis(w, order, w_sorted, axis=1)
    return w


def apply_turnover_cap(target: np.ndarray, cap: Optional[float]) -> np.ndarray:
    """Walk the book forward, never trading more than ``cap`` in a day.

    Turnover is ``0.5 * sum|w_new - w_old|`` -- the share of a gross-1 book that
    changes hands. Blending the old book toward the target is linear in that
    measure, so ``w = w_old + lam * (target - w_old)`` turns over exactly
    ``lam * full_turnover`` and the needed ``lam`` is ``cap / full_turnover``,
    clipped at 1.

    This is path dependent by construction: today's book is the starting point
    for tomorrow's trade, so it cannot be computed day by day in isolation --
    which is the whole reason a capped strategy is not just the target with a
    smaller cost subtracted.
    """
    if cap is None:
        return target
    D = target.shape[0]
    out = np.zeros_like(target)
    held = np.zeros(target.shape[1], dtype=np.float64)
    for t in range(D):
        full = 0.5 * np.abs(target[t] - held).sum()
        lam = 1.0 if full <= cap or full <= 0 else cap / full
        held = held + lam * (target[t] - held)
        out[t] = held
    return out


def portfolio_returns(weights: np.ndarray, fwd: np.ndarray, cost_per_side: float = 0.0):
    """Daily P&L of a held book, after the cost of getting into it.

    ``weights[t]`` is the book held into day ``t``'s forward return, and the trade
    that establishes it is charged on the way in: every unit of weight that moves
    pays ``cost_per_side`` whether it is a buy or a sell, so a round trip of the
    whole book costs ``2 * cost_per_side``. Returns (net, gross, turnover)."""
    D = weights.shape[0]
    gross = np.full(D, np.nan)
    turn = np.zeros(D)
    prev = np.zeros(weights.shape[1], dtype=np.float64)
    for t in range(D):
        w, r = weights[t], fwd[t]
        traded = np.abs(w - prev).sum()
        turn[t] = 0.5 * traded
        m = np.isfinite(r)
        gross[t] = float((w[m] * r[m]).sum()) if m.any() else np.nan
        prev = w
    net = gross - turn * 2.0 * cost_per_side
    return net, gross, turn


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


def _curve_stats(v: np.ndarray, periods_per_year: int) -> Dict[str, float]:
    f = v[np.isfinite(v)]
    if f.size < 2:
        return {"annualized_return": float("nan"), "sharpe": float("nan"), "max_drawdown": float("nan")}
    mean, std = float(f.mean()), float(f.std())
    cum = np.cumprod(1.0 + f)
    dd = cum / np.maximum.accumulate(cum) - 1.0
    return {"annualized_return": float((1.0 + mean) ** periods_per_year - 1.0),
            "sharpe": float(mean / std * np.sqrt(periods_per_year)) if std > 0 else float("nan"),
            "max_drawdown": float(dd.min())}


def backtest_metrics(ic_series: np.ndarray, alpha_values: Optional[np.ndarray] = None,
                     forward_returns: Optional[np.ndarray] = None, top_frac: float = 0.1,
                     periods_per_year: int = 252, cost_per_side: float = 0.0,
                     turnover_cap: Optional[float] = None) -> Dict[str, float]:
    """ICIR from the IC series, plus an equal-weight top-minus-bottom ``top_frac``
    portfolio measured two ways when ``cost_per_side`` or ``turnover_cap`` is given:

    - **full**: rebalance all the way to the target book every day. Maximum
      responsiveness, maximum turnover -- and at A-share costs the turnover is
      what decides whether a daily-rebalanced factor survives at all.
    - **capped**: never trade more than ``turnover_cap`` of the book in a day,
      blending toward the target when the full trade would exceed it. Path
      dependent: today's holding is tomorrow's starting point.

    Both are still diagnostics -- one holding period, no borrow, no limits, no
    market impact beyond the linear cost -- but with costs charged they are at
    least the right shape for deciding between factors."""
    ic = np.asarray(ic_series, dtype=np.float64)
    f = ic[np.isfinite(ic)]
    ic_mean = float(f.mean()) if f.size else float("nan")
    ic_std = float(f.std()) if f.size else float("nan")
    out: Dict[str, float] = {"ic_mean": ic_mean, "ic_std": ic_std,
                             "icir": float(ic_mean / ic_std) if ic_std > 0 else float("nan")}
    if alpha_values is not None and forward_returns is not None:
        alpha = np.asarray(alpha_values, dtype=np.float64)
        fwd = np.asarray(forward_returns, dtype=np.float64)
        target = long_short_weights(alpha, top_frac)

        net, gross, turn = portfolio_returns(target, fwd, cost_per_side)
        out.update(_curve_stats(gross, periods_per_year))          # the no-cost headline, unchanged
        out["turnover"] = float(np.nanmean(turn))
        if cost_per_side > 0:
            for k, v in _curve_stats(net, periods_per_year).items():
                out[f"{k}_net"] = v

        if turnover_cap is not None:
            capped = apply_turnover_cap(target, turnover_cap)
            c_net, c_gross, c_turn = portfolio_returns(capped, fwd, cost_per_side)
            src = c_net if cost_per_side > 0 else c_gross
            for k, v in _curve_stats(src, periods_per_year).items():
                out[f"{k}_capped"] = v
            out["turnover_capped"] = float(np.nanmean(c_turn))
    return out


__all__ = ["MIN_PAIRS", "long_short_weights", "apply_turnover_cap", "portfolio_returns",
           "cs_pearson", "cross_sectional_pearson", "rank_rows", "zscore_rows", "nanmean_quiet",
           "ic_summary", "quantile_longshort", "batched_ic", "sampled_abs_corr", "daily_returns_from_close",
           "combine_panels", "backtest_metrics", "torch_rank_rows", "rank_ic_turnover", "ema_rows"]
