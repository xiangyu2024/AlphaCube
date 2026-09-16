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
