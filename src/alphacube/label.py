"""Backtest labels computed straight from the cube.

``open_vwap15``: the production label of the previous project. The factor is
known after the close of day t; the position is entered at day t+1's
09:31-09:45 VWAP and exited at day t+1+h's:

    target[t] = V[t+1+h] / V[t+1] - 1,   V = sum(Amt) / sum(Volume) over bars 1..15

NaN rules (unchanged): any missing bar in the window, zero total volume, or
a VWAP outside [min Low, max High] of the window. Bar 0 (09:30 auction) is
excluded on the 241 grid; on a shorter synthetic grid the window is clipped.

``close``: Close[t+h] / Close[t] - 1 on the last finite minute close of
each stock-day (the paper convention; not executable at the close itself).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import warnings

import numpy as np

from .store import CubeView

OPEN_VWAP_WINDOW = (1, 16)          # minute indices [1, 16) = 09:31 .. 09:45 on the V2 grid
OPEN_VWAP_FIELD = "OpenVwap15"
# How many of the window's minutes must carry a usable price. Measured on the
# production store (80 symbols, 2023): every tradable stock-day has at least 3,
# 99.98% have >= 5, and only 67% have all 15 -- so demanding the full window
# throws away a third of the panel, while 5 costs 0.02% and still rules out a
# "15-minute VWAP" built from one stray minute.
OPEN_VWAP_MIN_MINUTES = 5


def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.where(np.isfinite(a) & np.isfinite(b) & (b > 0), a / np.where(b > 0, b, np.nan), np.nan)


def _last_finite(cube: np.ndarray) -> np.ndarray:
    """(A, B, M) -> (A, B): last finite value along M (NaN if none)."""
    finite = np.isfinite(cube)
    idx = cube.shape[-1] - 1 - np.argmax(finite[..., ::-1], axis=-1)
    out = np.take_along_axis(cube, idx[..., None], axis=-1)[..., 0]
    return np.where(finite.any(axis=-1), out, np.nan)


def daily_close(view: CubeView, chunk: int = 256) -> np.ndarray:
    """(D, S) float64 last finite minute close per stock-day (masked by tradability)."""
    out = np.full((view.n_days, view.n_symbols), np.nan)
    for lo in range(0, view.n_symbols, chunk):
        hi = min(lo + chunk, view.n_symbols)
        out[:, lo:hi] = _last_finite(view.read_s("Close", lo, hi).astype(np.float64)).T
    out[~view.tradable().T] = np.nan
    return out


def open_vwap15(view: CubeView, window: Tuple[int, int] = OPEN_VWAP_WINDOW, chunk: int = 256,
                min_minutes: int = OPEN_VWAP_MIN_MINUTES) -> np.ndarray:
    """(D, S) float64 volume-weighted average price over the opening window.

    **Price space matters here.** In the production V2 store the price fields
    (Open/High/Low/Close/Vwap) are back-adjusted while ``Amt`` and ``Volume``
    are the raw traded yuan and shares, so ``Amt / Volume`` is a price in the
    *unadjusted* space -- off by a per-stock factor that ran from 2.5x to 113x
    across a sample of 12 symbols (and is constant within a symbol to ~1%, which
    is what identifies it as an adjustment factor rather than a unit slip).
    Mixing it with adjusted High/Low made the sanity check below reject 100% of
    stock-days, the label came out entirely NaN, and every mined candidate was
    rejected for a non-finite IC.

    So the average is taken over the per-minute ``Vwap`` field, which lives in
    the same space as High/Low, weighted by ``Volume``. Numerator and
    denominator share one mask: 5.5% of minutes carry volume but no Vwap, and
    counting their volume in the denominator only would bias the result down.
    ``Amt / Volume`` remains the fallback for a cube with no ``Vwap`` field
    (synthetic stores, and any unadjusted store where the two agree anyway).
    """
    lo_m, hi_m = window
    hi_m = min(hi_m, view.n_minutes)
    if hi_m - lo_m < 1:
        raise ValueError(f"minute grid too short for window {window}")
    fields = set(view.fields)
    have_range = {"High", "Low"} <= fields
    use_vwap = "Vwap" in fields
    if not use_vwap and not {"Volume", "Amt"} <= fields:
        raise KeyError("open_vwap15 needs a Vwap field, or Volume and Amt, in the cube")
    if "Volume" not in fields:
        raise KeyError("open_vwap15 needs Volume in the cube")
    need = max(1, min(int(min_minutes), hi_m - lo_m))
    out = np.full((view.n_days, view.n_symbols), np.nan)
    for lo in range(0, view.n_symbols, chunk):
        hi = min(lo + chunk, view.n_symbols)
        vol = view.read_s("Volume", lo, hi)[..., lo_m:hi_m].astype(np.float64)
        price = (view.read_s("Vwap", lo, hi)[..., lo_m:hi_m].astype(np.float64) if use_vwap
                 else _ratio(view.read_s("Amt", lo, hi)[..., lo_m:hi_m].astype(np.float64), vol))
        good = np.isfinite(price) & np.isfinite(vol) & (vol > 0)
        with np.errstate(all="ignore"):
            num = np.where(good, price * vol, 0.0).sum(-1)
            den = np.where(good, vol, 0.0).sum(-1)
            v = np.where((den > 0) & (good.sum(-1) >= need), num / den, np.nan)
        if have_range:
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN suspended days
                hi_p = np.nanmax(view.read_s("High", lo, hi)[..., lo_m:hi_m].astype(np.float64), axis=-1)
                lo_p = np.nanmin(view.read_s("Low", lo, hi)[..., lo_m:hi_m].astype(np.float64), axis=-1)
            v = np.where((v >= lo_p) & (v <= hi_p), v, np.nan)
        out[:, lo:hi] = v.T
    out[~view.tradable().T] = np.nan
    return out


def forward_return(price: np.ndarray, horizon: int, entry_lag: int) -> np.ndarray:
    """target[t] = price[t+lag+h] / price[t+lag] - 1 on a (D, S) price panel."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    D = price.shape[0]
    out = np.full_like(price, np.nan, dtype=np.float64)
    span = entry_lag + horizon
    if span < D:
        with np.errstate(divide="ignore", invalid="ignore"):
            out[: D - span] = price[span:] / price[entry_lag: D - horizon] - 1.0
    return out


@dataclass(frozen=True)
class Label:
    kind: str = "open_vwap15"        # "open_vwap15" | "close"
    horizon: int = 1

    def price(self, view: CubeView) -> np.ndarray:
        if self.kind == "open_vwap15":
            return open_vwap15(view)
        if self.kind == "close":
            return daily_close(view)
        raise ValueError(f"unknown label kind {self.kind!r}")

    def compute(self, view: CubeView, price: Optional[np.ndarray] = None) -> np.ndarray:
        p = self.price(view) if price is None else price
        lag = 1 if self.kind == "open_vwap15" else 0
        return forward_return(p, self.horizon, lag)

    def describe(self) -> str:
        if self.kind == "open_vwap15":
            return f"OpenVwap15[t+1+{self.horizon}] / OpenVwap15[t+1] - 1"
        return f"Close[t+{self.horizon}] / Close[t] - 1"


__all__ = ["Label", "open_vwap15", "daily_close", "forward_return", "OPEN_VWAP_WINDOW", "OPEN_VWAP_FIELD",
           "OPEN_VWAP_MIN_MINUTES"]
