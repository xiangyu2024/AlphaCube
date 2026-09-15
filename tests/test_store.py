import json

import numpy as np
import pytest

from alphacube.store import DAY_STATUS_CODES, CubeStore, build_cube_store


def test_meta_is_v2_layout(store):
    meta = json.loads((store.directory / "dt_meta.json").read_text())
    for key in ("fields", "n_symbols", "n_days", "n_minutes", "symbols", "dates", "day_status_file", "day_status_codes"):
        assert key in meta
    assert meta["day_status_codes"] == DAY_STATUS_CODES
    assert (store.directory / "dtfield_Close.npy").exists() and (store.directory / "day_status.npy").exists()
    reopened = CubeStore(store.directory)
    assert reopened.fields == store.fields and reopened.dates == store.dates


def test_view_selection_and_reads(store):
    S, D, M = store.n_symbols, store.n_days, store.n_minutes
    full = np.asarray(store.field("Close"))
    v = store.view(max_symbols=4, start=store.dates[5], end=store.dates[20])
    assert (v.n_symbols, v.n_days, v.n_minutes) == (4, 16, M)
    assert np.array_equal(v.read_s("Close", 1, 3), full[1:3, 5:21, :], equal_nan=True)
    assert np.array_equal(v.read_d("Close", 2, 5), full[:4, 7:10, :], equal_nan=True)
    assert v.tradable().shape == (4, 16)
    picks = [store.symbols[i] for i in (7, 2, 9)]
    w = store.view(symbols=picks)
    assert w.symbols == picks
    assert np.array_equal(w.read_s("Volume", 0, 3), full[[7, 2, 9]] if False else np.asarray(store.field("Volume"))[[7, 2, 9]], equal_nan=True)
    assert np.array_equal(w.read_d("Volume", 0, 2), np.asarray(store.field("Volume"))[[7, 2, 9], 0:2], equal_nan=True)
    assert w.read_s("Close", 0, 1).flags.writeable


def test_date_bounds_snap_to_trading_days(store):
    d0, d1 = store.date_range("20200101", store.dates[10])   # New Year's Day is not a trading day
    assert d0 == 0 and d1 == 11
    with pytest.raises(ValueError):
        store.date_range(store.dates[-1], store.dates[0])
    with pytest.raises(KeyError):
        store.view(symbols=["nope"])


def test_tradable_matches_status(store):
    st = np.asarray(store.day_status())
    assert np.array_equal(store.tradable(), np.isin(st, [1, 2]))
    assert (st == DAY_STATUS_CODES["OUTSIDE_LIFECYCLE"]).sum() == 8
    assert (st == DAY_STATUS_CODES["SUSPENDED"]).sum() > 0


def test_build_validates_axes(tmp_path):
    cubes = {"Close": np.ones((2, 3, 4)), "Volume": np.ones((2, 3, 5))}
    with pytest.raises(ValueError):
        build_cube_store(tmp_path / "bad", cubes, ["a", "b"], ["1", "2", "3"])
    with pytest.raises(ValueError):
        build_cube_store(tmp_path / "bad2", {"Close": np.ones((2, 3, 4))}, ["a"], ["1", "2", "3"])
