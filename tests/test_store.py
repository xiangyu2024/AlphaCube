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


def test_strided_cube_reads_switch_off_kernel_read_ahead(tmp_path, monkeypatch):
    """A cube is (S, D, M), so the D round's read -- many symbols, few days --
    wants ~1 KB out of every 1.9 MB stride, and the kernel's read-ahead fetches a
    window around each crumb. Measured on the production store (9.8 GB field, 6
    days x 5207 symbols, cold ranges): 5649 MB read for the 28.7 MB wanted, 6.9 s,
    against 13 MB and 0.37 s under MADV_RANDOM -- 37 minutes a field over 1942
    days against 2. gpu10's first D round crawled at 3 days a minute because of it.
    The contiguous pattern must keep read-ahead (47 MB/s without it), so the advice
    follows the pattern, and only changes when the pattern does."""
    import mmap

    import numpy as np

    from alphacube import store as store_mod
    from alphacube.store import advise_access, open_cube_store
    from alphacube.synthetic import write_synthetic_store

    st = open_cube_store(write_synthetic_store(tmp_path / "cube", n_symbols=12, n_days=40, n_minutes=8,
                                               seed=2).directory)
    calls = []
    real = store_mod.advise_access
    monkeypatch.setattr(store_mod, "advise_access", lambda arr, random: calls.append(bool(random)) or real(arr, random))

    field = np.asarray(st.field("Close"))
    rows = slice(0, 12)
    strided = st.read_cube("Close", rows, 5, 7)              # many symbols, few days
    assert calls == [True]
    assert np.array_equal(strided, field[rows, 5:7, :], equal_nan=True)
    st.read_cube("Close", rows, 9, 11)                       # same pattern: no second syscall
    assert calls == [True]
    contiguous = st.read_cube("Close", slice(0, 2), 0, 40)   # few symbols, all days
    assert calls == [True, False]
    assert np.array_equal(contiguous, field[0:2, 0:40, :], equal_nan=True)
    st.read_cube("Close", slice(2, 4), 0, 40)
    assert calls == [True, False]

    # and the advice really reaches a real mapping on this platform
    if getattr(mmap, "MADV_RANDOM", None) is not None:
        assert advise_access(st.field("Close"), random=True)
        assert advise_access(st.field("Close"), random=False)
    assert advise_access(np.zeros(4), random=True) is False, "a plain array is not a mapping"
