"""Synthetic cube for tests and local runs: a market with cross-sectional
factor structure, intraday price paths, U-shaped volume, suspended and
not-yet-listed stock-days. Shapes follow the V2 contract (241 minutes, index
0 = 09:30 auction) unless told otherwise.

Symbol spelling is ``<prefix>.<market>`` (``"600000.SH"``), the format
``tradability.board_limit`` (hence ``dailyfields.derive``) actually keys off
-- must_fix #2, measured: the previous ``"SH.6xxxxx"`` spelling put the market
code FIRST, so every STAR/GEM/BSE board rule missed and ``board_limit``
returned a uniform 0.10 plane on every synthetic cube. See
``_SPECIAL_BOARD_SYMBOLS`` / ``_synthetic_symbols``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .store import DAY_STATUS_CODES, CubeStore, build_cube_store

# One of each special board, pinned so every rule in tradability.board_limit
# fires regardless of n_symbols: STAR (688/689), GEM (300/301/302), BSE
# (4/8/92). Always the LAST few symbols -- see _synthetic_symbols -- so a
# short prefix of the full list (store.view(max_symbols=k) with a small k) is
# the main-board-only case dailyfields.derive already treats as normal.
_SPECIAL_BOARD_SYMBOLS: Tuple[str, ...] = ("688001.SH", "300001.SZ", "830001.BJ")

# The three stored daily fields make_synthetic_cube can hand back identifiable
# planes for -- the same three names as dailyfields.STORED_FIELDS / catalog.
# STORED_DAILY_FIELDS, duplicated here rather than imported (this module
# already does not import dailyfields.py, and the three codebase precedents
# for this exact tuple -- catalog.py, dailyfields.py -- are each their own
# literal copy too; tests/test_daily_leaf_typing.py::test_dailyfields_tuples_
# agree_with_catalog_py is the kind of check that would catch a drift).
_SYNTHETIC_DAILY_FIELDS: Tuple[str, ...] = ("FloatCap", "TotalCap", "FloatShares")

# Step of the DATE term in the identifiable planes (see make_synthetic_cube's
# ``daily_fields``). Must be (a) far ABOVE float32's spacing at the plane's
# magnitude, or the float32 cast in dailyfields.resolve erases the date axis,
# and (b) far BELOW the 1e9 step of the SYMBOL term, or the two axes collide.
# 1e6 satisfies both for n_symbols up to ~8000 (float32 spacing is m / 2**23,
# so it stays under 1e6 while 1e9 * n_symbols < 8.4e12) and n_days up to 1000
# (1e6 * n_days < 1e9) -- orders of magnitude past any test fixture;
# tests/test_daily_fields_store.py::test_identifiable_planes_stay_identifiable_
# after_the_float32_cast asserts the property at the sizes actually used.
_DATE_STEP: float = 1e6


def _synthetic_symbols(n_symbols: int) -> List[str]:
    """``n_symbols`` symbols, mostly main board (``f"{600000+i}.SH"``) with the
    LAST few replaced by one STAR, one GEM and one BSE symbol -- see
    _SPECIAL_BOARD_SYMBOLS. For ``n_symbols`` smaller than the three special
    boards, as many of them are kept as fit (main board first)."""
    n_special = min(len(_SPECIAL_BOARD_SYMBOLS), n_symbols)
    n_main = n_symbols - n_special
    return [f"{600000 + i}.SH" for i in range(n_main)] + list(_SPECIAL_BOARD_SYMBOLS[:n_special])


def trading_dates(n_days: int, start_year: int = 2020) -> List[str]:
    """Weekday calendar as YYYYMMDD strings (no holidays; monotone is all we need)."""
    days = np.busday_offset(np.datetime64(f"{start_year}-01-02"), np.arange(n_days), roll="forward")
    return [str(d).replace("-", "") for d in days]


def make_synthetic_cube(n_symbols: int = 12, n_days: int = 80, n_minutes: int = 241,
                        seed: int = 0, suspended_frac: float = 0.02, unlisted_days: int = 0,
                        plant_beta: float = 0.0, daily_fields: Sequence[str] = (),
                        ) -> Tuple[Dict[str, np.ndarray], List[str], List[str], np.ndarray, Dict[str, np.ndarray]]:
    """``plant_beta`` > 0 plants a known signal: day d's close-to-close return is
    ``plant_beta * z_cs(intraday volatility on day d-1) + noise``, so the factor
    ``day_std(Ret)`` predicts next-day return with a positive IC (``Label("close", 1)``).
    Used to prove the research entry and the GP recover a signal that is really there.

    ``daily_fields``: names from ``("FloatCap", "TotalCap", "FloatShares")`` to
    also return as a FIFTH element -- a dict of (S, D) float64 planes over the
    SAME symbols/dates, with IDENTIFIABLE (not random) values: ``FloatCap[s, d]
    = 1e9 * (s + 1) + _DATE_STEP * d`` where ``s``/``d`` are plain 0-based
    POSITION indices (not symbol/date values), so a bug that aligns EITHER axis
    by position instead of by label is visible rather than silently plausible;
    ``TotalCap = FloatCap / 0.6``, ``FloatShares = FloatCap / 10.0``.

    DEVIATION from the design's literal ``+ d`` (design step 7), measured: the
    date term has to survive ``dailyfields.resolve``'s float32 cast, and it did
    not. float32's spacing at 1e9/5e9/1e10 is 64/512/1024, so ``+ d`` with
    d in 0..59 quantised away completely -- ``CubeView.daily("FloatCap")`` came
    back with 1 or 2 distinct values per symbol ROW (measured) instead of 60,
    i.e. each row was constant along the date axis. Sabotaging
    ``DailyFieldStore.align`` to index the DATE axis by POSITION then left both
    view-level WP3 tests green: the date half of "view independence" asserted
    nothing. ``_DATE_STEP`` keeps the same ~1e9 magnitude and the same
    by-POSITION identifiability while putting the date term far above that
    spacing. Keyword-gated: empty by
    default, so every existing caller keeps today's 4-tuple-shaped contents at
    this position (write_synthetic_store below unpacks five unconditionally,
    so its own arity never depends on this argument) and today's RNG draw
    sequence exactly -- the planes are plain index arithmetic, not ``rng``, so
    asking for them costs no extra random draw either."""
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
    symbols = _synthetic_symbols(S)
    dates = trading_dates(D)
    planes: Dict[str, np.ndarray] = {}
    if daily_fields:
        unknown = [f for f in daily_fields if f not in _SYNTHETIC_DAILY_FIELDS]
        if unknown:
            raise ValueError(f"make_synthetic_cube: daily_fields has unknown name(s) {unknown}; "
                             f"the stored daily fields are {_SYNTHETIC_DAILY_FIELDS}")
        float_cap = (1e9 * (np.arange(S, dtype=np.float64)[:, None] + 1.0)
                     + _DATE_STEP * np.arange(D, dtype=np.float64)[None, :])
        available = {"FloatCap": float_cap, "TotalCap": float_cap / 0.6, "FloatShares": float_cap / 10.0}
        planes = {name: available[name] for name in daily_fields}
    return cubes, symbols, dates, status, planes


def write_synthetic_store(out_dir, **kwargs) -> CubeStore:
    cubes, symbols, dates, status, _daily_planes = make_synthetic_cube(**kwargs)
    return build_cube_store(out_dir, cubes, symbols, dates, day_status=status)


def write_daily_fields(out_dir, planes: Dict[str, np.ndarray], symbols: Sequence[str],
                       dates: Sequence[str]) -> Path:
    """Write one ``dailyfields.DailyFieldStore`` directory: ``daily_<name>.npy``
    (float64, ``(len(symbols), len(dates))``) for each plane, plus the
    ``meta.json`` naming the symbol/date labels those rows/columns are in --
    exactly, and only, what ``DailyFieldStore`` reads."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    symbols = [str(s) for s in symbols]
    dates = [str(d) for d in dates]
    names = sorted(planes)
    for name in names:
        arr = np.asarray(planes[name], dtype=np.float64)
        if arr.shape != (len(symbols), len(dates)):
            raise ValueError(f"write_daily_fields: {name!r} shape {arr.shape} != {(len(symbols), len(dates))}")
        np.save(out / f"daily_{name}.npy", np.ascontiguousarray(arr))
    (out / "meta.json").write_text(
        json.dumps({"fields": names, "symbols": symbols, "dates": dates}, ensure_ascii=False), encoding="utf-8")
    return out


__all__ = ["make_synthetic_cube", "write_synthetic_store", "write_daily_fields", "trading_dates"]
