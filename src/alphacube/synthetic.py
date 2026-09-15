"""Synthetic cube for tests and local runs: a market with cross-sectional
factor structure, intraday price paths, U-shaped volume, suspended and
not-yet-listed stock-days. Shapes follow the V2 contract (241 minutes, index
0 = 09:30 auction) unless told otherwise.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .store import DAY_STATUS_CODES, CubeStore, build_cube_store


def trading_dates(n_days: int, start_year: int = 2020) -> List[str]:
    """Weekday calendar as YYYYMMDD strings (no holidays; monotone is all we need)."""
    days = np.busday_offset(np.datetime64(f"{start_year}-01-02"), np.arange(n_days), roll="forward")
    return [str(d).replace("-", "") for d in days]


def make_synthetic_cube(n_symbols: int = 12, n_days: int = 80, n_minutes: int = 241,
                        seed: int = 0, suspended_frac: float = 0.02, unlisted_days: int = 0,
                        plant_beta: float = 0.0,
                        ) -> Tuple[Dict[str, np.ndarray], List[str], List[str], np.ndarray]:
    """``plant_beta`` > 0 plants a known signal: day d's close-to-close return is
    ``plant_beta * z_cs(intraday volatility on day d-1) + noise``, so the factor
    ``day_std(Ret)`` predicts next-day return with a positive IC (``Label("close", 1)``).
    Used to prove the research entry and the GP recover a signal that is really there."""
    rng = np.random.default_rng(seed)
    S, D, M = n_symbols, n_days, n_minutes
    # daily close: common factor + idiosyncratic random walk
    beta = rng.normal(1.0, 0.3, S)
    market = rng.normal(0.0, 0.01, D).cumsum()
    idio = rng.normal(0.0, 0.02, (S, D)).cumsum(axis=1)
    log_close = np.log(rng.uniform(5, 50, S))[:, None] + beta[:, None] * market[None, :] + idio
    daily_close = np.exp(log_close)                                   # (S, D)
    prev_close = np.concatenate([daily_close[:, :1] * 0.99, daily_close[:, :-1]], axis=1)
    # intraday path: bridge from prev close to today's close with noise
    t = np.linspace(0.0, 1.0, M)
    noise = rng.normal(0.0, 0.0015, (S, D, M)).cumsum(axis=2)
    noise -= noise[:, :, -1:] * t[None, None, :]                       # Brownian bridge -> 0 at close
    path = prev_close[:, :, None] * np.exp(np.log(daily_close / prev_close)[:, :, None] * t + noise)
    close = path
    open_ = np.concatenate([prev_close[:, :, None] * (1 + rng.normal(0, 0.002, (S, D, 1))), close[:, :, :-1]], axis=2)
    spread = np.abs(rng.normal(0.0, 0.0015, (S, D, M)))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    ushape = 1.0 + 2.0 * (np.abs(t - 0.5) * 2) ** 2                  # U-shaped intraday volume
    base_vol = rng.lognormal(10.0, 0.6, (S, 1, 1)) * rng.lognormal(0.0, 0.3, (S, D, 1))
    volume = np.round(base_vol * ushape[None, None, :] * rng.lognormal(0.0, 0.5, (S, D, M)))
    volume[:, :, 0] *= 3.0                                             # auction bar
    amt = volume * (open_ + close) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(volume > 0, amt / volume, np.nan)
    ret = close / open_ - 1.0
    if plant_beta:
        vol = ret.std(axis=2)                                          # (S, D) intraday volatility
        z = (vol - vol.mean(0)) / (vol.std(0) + 1e-12)                 # cross-sectional z per day
        shock = np.zeros((S, D))
        shock[:, 1:] = plant_beta * 0.02 * z[:, :-1] + rng.normal(0, 0.02, (S, D - 1))
        scale = np.cumprod(1.0 + shock, axis=1)[:, :, None]
        open_, high, low, close = open_ * scale, high * scale, low * scale, close * scale
        amt = volume * (open_ + close) / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            vwap = np.where(volume > 0, amt / volume, np.nan)
    cubes = {"Open": open_, "High": high, "Low": low, "Close": close,
             "Volume": volume, "Vwap": vwap, "Ret": ret, "Amt": amt}
    status = np.full((S, D), DAY_STATUS_CODES["TRADING"], dtype=np.int8)
    if suspended_frac > 0:
        susp = rng.random((S, D)) < suspended_frac
        status[susp] = DAY_STATUS_CODES["SUSPENDED"]
    if unlisted_days > 0:
        status[-1, :unlisted_days] = DAY_STATUS_CODES["OUTSIDE_LIFECYCLE"]
    dead = status != DAY_STATUS_CODES["TRADING"]
    for arr in cubes.values():
        arr[dead] = np.nan
    symbols = [f"SH.6{i:05d}" for i in range(S)]
    dates = trading_dates(D)
    return cubes, symbols, dates, status


def write_synthetic_store(out_dir, **kwargs) -> CubeStore:
    cubes, symbols, dates, status = make_synthetic_cube(**kwargs)
    return build_cube_store(out_dir, cubes, symbols, dates, day_status=status)


__all__ = ["make_synthetic_cube", "write_synthetic_store", "trading_dates"]
