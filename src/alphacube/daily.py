"""Daily ``(D, S)`` panels derived from the cube -- the bridge to every stage
that consumes day-frequency data (pools, combination, models, portfolio,
backtest). ``DailyData`` is the successor of the previous project's
``FeatureData`` (same duck-typed surface: ``get_field`` / ``n_days`` /
``n_instruments`` / ``forward_returns`` / ``dates`` / ``instruments``), but it
is never a data *source*: every daily field is a cube expression evaluated by
the engine (``DAILY_FIELD_EXPRESSIONS``), so a daily ``Close`` here is by
construction ``day_last(Close)`` of the minute cube.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

DAILY_FIELD_EXPRESSIONS: Dict[str, str] = {
    "Open": "day_first(Open)",
    "High": "day_max(High)",
    "Low": "day_min(Low)",
    "Close": "day_last(Close)",
    "Volume": "day_sum(Volume)",
    "Amount": "day_sum(Amt)",
    # NOT Amt/Volume: in the production store the price fields are back-adjusted
    # while Amt and Volume are raw, so their ratio is a price in another space
    # (median 0.108x Close, and the daily store's own Vwap is the adjusted one).
    # The average is over the minute Vwap field, weighted by Volume. The
    # denominator has to skip the 5.5% of minutes that carry volume but no Vwap,
    # or it is biased low by a median 5%; ``mul(0.0, ...)`` is how the expression
    # language says that -- 0 * NaN is NaN, so those minutes drop out of the
    # NaN-aware day_sum, and every other minute contributes Volume - 0.
    "Vwap": "div(day_sum(mul(Vwap, Volume)), day_sum(sub(Volume, mul(0.0, mul(Vwap, Volume)))))",
    "Return": "sub(div(day_last(Close), ts_delay(day_last(Close), 1)), 1)",
}


class DailyData:
    """A panel of daily fields, each shaped ``(n_days, n_instruments)`` float64."""

    def __init__(self, fields: Dict[str, np.ndarray], dates: Sequence, instruments: Sequence[str],
                 close_field: str = "Close", tradable_mask: Optional[np.ndarray] = None) -> None:
        if not fields:
            raise ValueError("DailyData requires at least one field")
        self.dates = pd.DatetimeIndex(pd.to_datetime([str(d) for d in dates]))
        self.instruments = list(instruments)
        expected = (len(self.dates), len(self.instruments))
        self._fields: Dict[str, np.ndarray] = {}
        for name, arr in fields.items():
            arr = np.asarray(arr, dtype=np.float64)
            if arr.shape != expected:
                raise ValueError(f"field {name!r} has shape {arr.shape}, expected {expected}")
            self._fields[name] = arr
        self._close_field = close_field
        self._tradable: Optional[np.ndarray] = None
        if tradable_mask is not None:
            mask = np.asarray(tradable_mask, dtype=bool)
            if mask.shape != expected:
                raise ValueError(f"tradable_mask has shape {mask.shape}, expected {expected}")
            self._tradable = mask

    # -- protocol ---------------------------------------------------------------------
    @property
    def n_days(self) -> int:
        return len(self.dates)

    @property
    def n_instruments(self) -> int:
        return len(self.instruments)

    @property
    def field_names(self) -> List[str]:
        return list(self._fields)

    def has_field(self, name: str) -> bool:
        return name in self._fields

    def get_field(self, name: str) -> np.ndarray:
        try:
            return self._fields[name]
        except KeyError:
            raise KeyError(f"unknown daily field {name!r}; available: {sorted(self._fields)}") from None

    def add_field(self, name: str, panel: np.ndarray) -> None:
        panel = np.asarray(panel, dtype=np.float64)
        if panel.shape != (self.n_days, self.n_instruments):
            raise ValueError(f"panel shape {panel.shape} != {(self.n_days, self.n_instruments)}")
        self._fields[name] = panel

    def get_tradable_mask(self) -> np.ndarray:
        if self._tradable is None:
            self._tradable = np.ones((self.n_days, self.n_instruments), dtype=bool)
        return self._tradable

    def slice_days(self, start: int, end: int) -> "DailyData":
        return DailyData({k: v[start:end] for k, v in self._fields.items()}, self.dates[start:end],
                         self.instruments, self._close_field,
                         None if self._tradable is None else self._tradable[start:end])

    def window(self, end_idx: int, length: int) -> "DailyData":
        return self.slice_days(max(0, end_idx - length), end_idx)

    def forward_returns(self, horizon: int = 1, field_name: Optional[str] = None) -> np.ndarray:
        """price[t+h] / price[t] - 1 (trailing ``h`` rows NaN)."""
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        price = self.get_field(field_name or self._close_field)
        out = np.full_like(price, np.nan, dtype=np.float64)
        n = price.shape[0]
        if horizon < n:
            with np.errstate(divide="ignore", invalid="ignore"):
                out[: n - horizon] = price[horizon:] / price[: n - horizon] - 1.0
        return out


def daily_from_view(view, fields: Optional[Sequence[str]] = None, open_vwap15: bool = True,
                    device: str = "cpu", chunk_symbols: Optional[int] = None,
                    chunk_days: Optional[int] = None) -> DailyData:
    """Evaluate the daily field expressions over a ``CubeView``."""
    from .engine.engine import CubeEngine
    from .label import OPEN_VWAP_FIELD, open_vwap15 as _ov
    from .parser import parse

    names = list(fields) if fields is not None else list(DAILY_FIELD_EXPRESSIONS)
    unknown = [n for n in names if n not in DAILY_FIELD_EXPRESSIONS]
    if unknown:
        raise KeyError(f"no daily definition for {unknown}; known: {sorted(DAILY_FIELD_EXPRESSIONS)}")
    engine = CubeEngine(view, device=device, chunk_symbols=chunk_symbols, chunk_days=chunk_days)
    res = engine.evaluate([parse(DAILY_FIELD_EXPRESSIONS[n]) for n in names])
    panels = {n: p.astype(np.float64) for n, p in zip(names, res.panels)}
    if open_vwap15:
        panels[OPEN_VWAP_FIELD] = _ov(view)
    return DailyData(panels, view.dates, view.symbols, tradable_mask=view.tradable().T)


def synthetic_daily(n_days: int = 250, n_instruments: int = 20, seed: int = 0,
                    start: str = "2020-01-02") -> DailyData:
    """Random-walk OHLCV daily panels for stages that only need day-frequency
    inputs (the previous project's ``generate_synthetic_ohlcv`` + CSV round trip,
    without the CSV)."""
    rng = np.random.default_rng(seed)
    close = 20 * np.exp(np.cumsum(rng.normal(0.0, 0.015, (n_days, n_instruments)), axis=0))
    open_ = close * (1 + rng.normal(0, 0.003, close.shape))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, close.shape)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, close.shape)))
    volume = rng.lognormal(12.0, 0.5, close.shape)
    amount = volume * (open_ + close) / 2
    ret = np.vstack([np.full((1, n_instruments), np.nan), close[1:] / close[:-1] - 1])
    dates = pd.bdate_range(start, periods=n_days)
    return DailyData({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume,
                      "Amount": amount, "Vwap": amount / volume, "Return": ret},
                     dates, [f"S{i:03d}" for i in range(n_instruments)])


__all__ = ["DailyData", "DAILY_FIELD_EXPRESSIONS", "daily_from_view", "synthetic_daily"]
