"""Segmented base+segments store, D-major mirror, open_cube_store: every
layout reads bit-identically to the monolithic store and the engine agrees."""

import json

import numpy as np
import pytest

from alphacube.engine.engine import CubeEngine
from alphacube.parser import parse
from alphacube.store import (CubeStore, SegmentedCubeStore, build_cube_store, build_d_major_mirror, open_cube_store,
                             write_segment)
from alphacube.synthetic import make_synthetic_cube


EXPRS = ["day_std(in_mean_rolling(Close, 5))", "cs_rank(ts_mean(day_std(Ret), 5))", "day_mean(nt_cs_rank(Close))",
         "day_last(nt_cs_rank(cross_mean(Close, 3)))", "ret_1d"]


@pytest.fixture(scope="module")
def layouts(tmp_path_factory):
    root = tmp_path_factory.mktemp("layouts")
    cubes, symbols, dates, status = make_synthetic_cube(n_symbols=9, n_days=50, n_minutes=24, seed=7, suspended_frac=0.05)
    mono = build_cube_store(root / "mono", cubes, symbols, dates, status)
    # segmented: base = first 30 days, segment A = next 12 days, segment B = last 8 days with one NEW symbol
    comp = root / "comp"
    build_cube_store(comp / "minute", {k: v[:, :30] for k, v in cubes.items()}, symbols, dates[:30], status[:, :30])
    write_segment(comp, "2024-02", {k: v[:, 30:42] for k, v in cubes.items()}, symbols, dates[30:42], status[:, 30:42])
    extra = {k: np.concatenate([v[:, 42:], v[:1, 42:] * 1.01], axis=0) for k, v in cubes.items()}
    write_segment(comp, "2024-03", extra, symbols + ["NEW.1"], dates[42:],
                  np.concatenate([status[:, 42:], np.ones((1, 8), dtype=np.int8)], axis=0))
    return mono, comp, cubes, status


def test_segmented_reads_match_monolithic(layouts):
    mono, comp, cubes, status = layouts
    seg = open_cube_store(comp)
    assert isinstance(seg, SegmentedCubeStore) and isinstance(open_cube_store(mono.directory), CubeStore)
    assert seg.dates == mono.dates and seg.symbols == mono.symbols + ["NEW.1"] and seg.n_days == 50
    rows = slice(0, mono.n_symbols)
    for name in ("Close", "Volume"):
        a = seg.read_cube(name, rows, 25, 47)                     # spans base + both segments
        b = mono.read_cube(name, rows, 25, 47)
        assert np.array_equal(a, b, equal_nan=True)
        c = seg.read_cube(name, np.array([8, 2, 9]), 40, 50)      # fancy rows incl. the new symbol
        assert np.array_equal(c[:2], mono.read_cube(name, np.array([8, 2]), 40, 50), equal_nan=True)
        assert np.isnan(c[2, :2]).all() and np.isfinite(c[2, 2:]).any()   # NEW.1 exists only in segment B
    assert np.array_equal(seg.read_status(rows, 0, 50), mono.read_status(rows, 0, 50))
    assert (seg.read_status(np.array([9]), 0, 42) == 0).all()      # OUTSIDE_LIFECYCLE before it appears
    assert seg.contract_hash != mono.contract_hash and len(seg.contract["segment_meta_sha256"]) == 2
    base_only = SegmentedCubeStore(comp, include_segments=False)
    assert base_only.n_days == 30 and base_only.symbols == mono.symbols


def test_segmented_index_is_validated(layouts, tmp_path):
    mono, comp, cubes, status = layouts
    idx = comp / "incremental" / "index.json"
    payload = json.loads(idx.read_text())
    payload["segments"][0]["meta_sha256"] = "0" * 64
    bad = tmp_path / "bad"
    import shutil
    shutil.copytree(comp, bad)
    (bad / "incremental" / "index.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        SegmentedCubeStore(bad)


def test_engine_identical_across_layouts(layouts):
    mono, comp, cubes, status = layouts
    seg = SegmentedCubeStore(comp)
    exprs = [parse(s) for s in EXPRS]
    a = CubeEngine(mono.view(), chunk_symbols=4, chunk_days=13).evaluate(exprs)
    b = CubeEngine(seg.view(symbols=mono.symbols), chunk_symbols=4, chunk_days=13).evaluate(exprs)
    for x, y in zip(a.panels, b.panels):
        assert np.array_equal(x, y, equal_nan=True)
    full = CubeEngine(seg.view(), chunk_days=7).evaluate(exprs)        # incl. NEW.1: extra column, NaN before it lists
    assert full.panels[0].shape == (50, 10) and np.isnan(full.panels[0][:42, 9]).all()


def test_d_major_mirror(layouts, tmp_path):
    mono, comp, cubes, status = layouts
    import shutil
    shutil.copytree(mono.directory, tmp_path / "m")
    st = CubeStore(tmp_path / "m")
    build_d_major_mirror(st, fields=["Close", "Ret"], days_per_chunk=7)
    assert sorted(st.meta["d_major_files"]) == ["Close", "Ret"] and st.mirror("Close").shape == (50, 9, 24)
    reopened = CubeStore(tmp_path / "m")
    for rows in (slice(0, 9), np.array([4, 1, 7])):
        for d0, d1 in ((0, 3), (10, 50), (20, 21)):
            assert np.array_equal(reopened.read_cube("Close", rows, d0, d1), mono.read_cube("Close", rows, d0, d1), equal_nan=True)
    exprs = [parse(s) for s in EXPRS]
    a = CubeEngine(reopened.view(), chunk_days=5).evaluate(exprs)
    b = CubeEngine(mono.view(), chunk_days=5).evaluate(exprs)
    for x, y in zip(a.panels, b.panels):
        assert np.array_equal(x, y, equal_nan=True)


def test_boundary_files_never_land_on_the_system_temp_dir(view, tmp_path, monkeypatch):
    """A boundary is a whole (S, D, M) float32 cube -- 9.8 GB on the production
    universe -- and the default used to be the system temp directory, which on the
    server is a 30 GB overlay shared with conda and the IDE server. A full-market
    run filled it and was killed with nothing in the log. The default now follows
    the project tree, and $ALPHACUBE_WORKDIR overrides it."""
    import tempfile
    from pathlib import Path

    from alphacube.engine.engine import CubeEngine, default_workdir

    monkeypatch.delenv("ALPHACUBE_WORKDIR", raising=False)
    project = Path(__file__).resolve().parents[1]
    assert default_workdir() == project / "tmp"
    assert Path(tempfile.gettempdir()) not in (default_workdir() or Path("/")).parents

    monkeypatch.setenv("ALPHACUBE_WORKDIR", str(tmp_path / "wd"))
    assert default_workdir() == tmp_path / "wd" and (tmp_path / "wd").is_dir()

    # and the engine actually writes its boundaries there
    seen = []
    real = tempfile.TemporaryDirectory

    def spy(*a, **kw):
        seen.append(kw.get("dir"))
        return real(*a, **kw)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", spy)
    # a full-width value wanted by both an S-mode round (cross_*) and a D-mode one
    # (nt_cs_*) has to be written out between them -- that is the boundary
    expr = parse("day_mean(nt_cs_rank(in_mean_rolling(cross_delay(Close, 2), 3)))")
    res = CubeEngine(view, chunk_symbols=3, chunk_days=7).evaluate([expr], tolerant=True)
    assert res.plan.boundaries, "expected this tree to materialise a boundary"
    assert seen and all(d == tmp_path / "wd" for d in seen), seen


def _segmented_truth(layouts):
    """What the segmented store holds at every (symbol, day): the monolithic cube
    for the nine base symbols; NEW.1 only on segment B's days, NaN / OUTSIDE_LIFECYCLE
    before it lists."""
    mono, comp, cubes, status = layouts
    cube = {}
    for name in ("Close", "Volume"):
        t = np.full((10, 50, mono.n_minutes), np.nan, dtype=np.float32)
        t[:9] = mono.field(name)
        t[9, 42:] = np.asarray(cubes[name][:1, 42:] * 1.01, dtype=np.float32)[0]
        cube[name] = t
    st = np.zeros((10, 50), dtype=np.int8)
    st[:9] = mono.day_status()
    st[9, 42:] = 1
    return cube, st


def test_segmented_reads_match_truth_for_every_index_form(layouts):
    """Random slices, unsorted and repeated index arrays, day ranges inside one
    source or across all three. (The dict-based reader returned NaN for the first
    copy of a repeated row: it mapped each symbol to its LAST output position.)"""
    mono, comp, cubes, status = layouts
    seg = SegmentedCubeStore(comp)
    cube, st = _segmented_truth(layouts)
    rng = np.random.default_rng(0)
    for _ in range(300):
        d0 = int(rng.integers(0, 50))
        d1 = int(rng.integers(d0 + 1, 51))
        if rng.random() < 0.5:
            a = int(rng.integers(0, 10))
            rows = slice(a, int(rng.integers(a + 1, 11)))
        else:
            rows = rng.integers(0, 10, size=int(rng.integers(1, 13)))
        for name in ("Close", "Volume"):
            got = seg.read_cube(name, rows, d0, d1)
            assert got.dtype == np.float32 and np.array_equal(got, cube[name][rows, d0:d1], equal_nan=True)
        assert np.array_equal(seg.read_status(rows, d0, d1), st[rows, d0:d1])


def test_segmented_contiguous_reads_are_stripe_copies(layouts, monkeypatch):
    """A contiguous request must reach each source memmap as basic slices. The
    previous reader indexed all three axes with np.ix_ -- the minute axis too --
    which gathers element by element: 21-28x slower than a slice on the production
    store's chunk shapes (58 ms vs 2.1-2.7 ms for 15 symbols x 1942 days)."""
    mono, comp, cubes, status = layouts
    seg = SegmentedCubeStore(comp)
    keys = []

    class Spy:
        def __init__(self, arr):
            self.arr = arr

        def __getitem__(self, key):
            keys.append(key)
            return self.arr[key]

    for src in seg._sources:
        field = src.store.field
        monkeypatch.setattr(src.store, "field", lambda name, _f=field: Spy(_f(name)))
    seg.read_cube("Close", slice(0, 9), 25, 47)          # base + both segments
    seg.read_cube("Close", slice(2, 6), 0, 30)           # base only (the production case)
    seg.read_cube("Close", slice(0, 10), 40, 50)         # NEW.1 is absent from segment A
    assert len(keys) == 6
    assert all(isinstance(k, slice) for key in keys for k in key), keys


def test_segmented_reads_advise_every_segment_they_touch(layouts, monkeypatch):
    """SegmentedCubeStore reads its segments' fields directly, so the read-ahead
    advice CubeStore.read_cube gives never reached them -- and the production
    store IS segmented. gpu12's third-block rebuild ran its D round at 5 days a
    minute, 987 MB/s off the disk, with no "rr" on any field mapping in
    /proc/<pid>/smaps (the boundary files had it)."""
    from alphacube import store as store_mod

    mono, comp, cubes, status = layouts
    seg = SegmentedCubeStore(comp)
    calls = []
    real = store_mod.advise_access
    monkeypatch.setattr(store_mod, "advise_access",
                        lambda arr, random: calls.append((id(arr), bool(random))) or real(arr, random))
    strided = seg.read_cube("Close", slice(0, 9), 29, 31)       # 9 symbols x 2 days, base + segment A
    assert [r for _, r in calls] == [True, True] and len({a for a, _ in calls}) == 2, calls
    calls.clear()
    seg.read_cube("Close", slice(0, 9), 27, 29)                 # same pattern, base only: already advised
    assert calls == []
    contiguous = seg.read_cube("Close", slice(0, 2), 0, 50)     # 2 symbols x all days: every segment
    assert [r for _, r in calls] == [False, False, False], calls
    assert np.array_equal(strided, mono.read_cube("Close", slice(0, 9), 29, 31), equal_nan=True)
    assert np.array_equal(contiguous, mono.read_cube("Close", slice(0, 2), 0, 50), equal_nan=True)


def test_external_d_major_mirror_serves_the_segmented_strided_reads(layouts, tmp_path, monkeypatch):
    """The production store is read-only, segmented, and has no D-major mirror, so
    every D-round read takes ~1 KB out of each 1.9 MB stride of the (S, D, M)
    fields -- the reason a D round ran 100 days a minute on gpu13 with the page
    cache full. A mirror built into a directory of our own serves those reads from
    contiguous (D, S, M) blocks: bit-identical, nothing written into the store,
    and only ever for the store it was built from."""
    from alphacube.engine.engine import CubeEngine
    from alphacube.store import MIRROR_META_FILE

    mono, comp, cubes, status = layouts
    plain = SegmentedCubeStore(comp)
    files_before = sorted(str(p.relative_to(comp)) for p in comp.rglob("*"))
    mroot = tmp_path / "mirror"
    for src in plain._sources:
        build_d_major_mirror(src.store, out_dir=mroot / src.root.relative_to(comp) / "minute")
    assert sorted(str(p.relative_to(comp)) for p in comp.rglob("*")) == files_before, "the store is read-only"
    assert not list(mroot.rglob("*.tmp"))

    seg = SegmentedCubeStore(comp, mirror_root=mroot)
    assert all(s.store.mirror("Close") is not None for s in seg._sources)
    fetched = []
    for s in seg._sources:
        real = s.store.advised_field
        monkeypatch.setattr(s.store, "advised_field", lambda name, strided, _r=real: fetched.append(strided) or _r(name, strided))
    reads = [("Close", slice(0, 9), 29, 31), ("Vwap", np.array([0, 3, 9, 4, 7]), 40, 42),
             ("Ret", slice(0, 10), 25, 47), ("Close", slice(2, 6), 0, 30), ("High", np.array([8, 1]), 10, 50)]
    for name, rows, d0, d1 in reads:
        got, want = seg.read_cube(name, rows, d0, d1), plain.read_cube(name, rows, d0, d1)
        assert np.array_equal(got, want, equal_nan=True), (name, rows, d0, d1)
    assert fetched.count(True) == 0, "a strided read must come from the mirror, not the (S, D, M) field"
    assert fetched.count(False) > 0, "contiguous reads still take the field itself"

    # a mirror built for another store is never read
    other = CubeStore(comp / "minute", mirror_dir=tmp_path / "wrong")
    build_d_major_mirror(CubeStore(mono.directory), out_dir=tmp_path / "wrong")
    assert (tmp_path / "wrong" / MIRROR_META_FILE).is_file() and other.mirror("Close") is None

    view_m, view_p = seg.view(), plain.view()
    for e in EXPRS:
        a = CubeEngine(view_m, chunk_symbols=4, chunk_days=3).evaluate([parse(e)]).panels[0]
        b = CubeEngine(view_p, chunk_symbols=4, chunk_days=3).evaluate([parse(e)]).panels[0]
        assert np.array_equal(a, b, equal_nan=True), e
