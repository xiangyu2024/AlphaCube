"""WP9 -- ``scripts/build_daily_fields.py``, the server build script for the
stored daily fields (``dailyfields.STORED_FIELDS``: FloatCap, TotalCap,
FloatShares). See docs/alphacube_steering_design.md, section "Building the
data", and work package "### WP9".

Builds a hand-written ``daily_capital/<date>.parquet`` directory and runs the
script against a synthetic cube store (``write_synthetic_store`` -- the
digit-first symbol spelling from WP3 so ``dailyfields.require_board_rule``,
which this script also runs per its own docstring's second call site, passes
on every fixture here for free).

DEVIATION, recorded here (see this package's ``notes``, not asserted as fact
in the script's own docstring beyond its own DEVIATION paragraph): the
design leaves FloatShares's source column name unspecified ("a share column
if one exists"). ``build_daily_fields.SHARE_COLUMN_CANDIDATES`` is this
implementation's documented guess; the tests below exercise both the
present- and absent-share-column paths against that guess rather than
against a real production schema, which was not available this session.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_daily_fields as bdf                               # noqa: E402

from alphacube.dailyfields import DailyFieldStore, resolve     # noqa: E402
from alphacube.synthetic import write_synthetic_store          # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_cap_file(cap_dir: Path, date: str, symbols: Sequence[str], cap: Sequence[float],
                    tot: Sequence[float], shares: Optional[Sequence[float]] = None,
                    share_column: str = "circulating_shares") -> None:
    cap_dir.mkdir(parents=True, exist_ok=True)
    data = {"symbol": list(symbols), "circulating_market_value": list(cap),
           "total_market_value": list(tot)}
    if shares is not None:
        data[share_column] = list(shares)
    pd.DataFrame(data).to_parquet(cap_dir / f"{date}.parquet")


def _store(tmp_path, **kw):
    """4 symbols (digit-first -- one main board, one STAR, one GEM, one BSE,
    per synthetic._synthetic_symbols) x however many days ``kw`` asks for."""
    kw.setdefault("n_symbols", 4)
    kw.setdefault("n_days", 3)
    kw.setdefault("n_minutes", 2)
    kw.setdefault("seed", 0)
    return write_synthetic_store(tmp_path / "cube", **kw)


# ---------------------------------------------------------------------------
# the primary build: shapes, NaN placement, meta.json, DailyFieldStore re-read
# ---------------------------------------------------------------------------

def test_build_writes_correct_planes_with_a_missing_date_and_a_missing_symbol(tmp_path):
    store = _store(tmp_path)
    symbols, dates = store.symbols, store.dates
    assert len(symbols) == 4 and len(dates) == 3           # the "3 dates x 4 symbols" fixture shape

    cap_dir = tmp_path / "daily_capital"
    # date 0: all four symbols
    _write_cap_file(cap_dir, dates[0], symbols, cap=[1e9, 2e9, 3e9, 4e9], tot=[2e9, 4e9, 6e9, 8e9],
                    shares=[1e8, 2e8, 3e8, 4e8])
    # date 1: MISSING entirely -- no parquet file written at all
    # date 2: one symbol absent (drop the last one)
    _write_cap_file(cap_dir, dates[2], symbols[:3], cap=[1.1e9, 2.1e9, 3.1e9], tot=[2.1e9, 4.1e9, 6.1e9],
                    shares=[1.1e8, 2.1e8, 3.1e8])

    out = tmp_path / "daily_fields"
    rc = bdf.main(["--store", str(store.directory), "--cap-dir", str(cap_dir), "--out", str(out),
                  "--min-coverage", "0.0"])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert set(meta["fields"]) == {"FloatCap", "TotalCap", "FloatShares"}
    assert meta["symbols"] == symbols and meta["dates"] == dates
    assert meta["n_symbols"] == 4 and meta["n_days"] == 3
    assert meta["columns_used"] == {"FloatCap": "circulating_market_value",
                                    "TotalCap": "total_market_value",
                                    "FloatShares": "circulating_shares"}
    assert set(meta["coverage"]) == {"FloatCap", "TotalCap", "FloatShares"}
    for name in ("FloatCap", "TotalCap", "FloatShares"):
        assert meta["coverage"][name] == pytest.approx(7 / 12)
        assert meta["date_range"][name] == [dates[0], dates[2]]
    assert "built_at" in meta and "source_paths" in meta
    assert meta["source_paths"]["cap_dir"] == str(cap_dir.resolve())
    assert meta["source_paths"]["store"] == str(Path(store.directory).resolve())

    nan = np.nan
    expect = {
        "FloatCap": np.array([[1e9, nan, 1.1e9], [2e9, nan, 2.1e9], [3e9, nan, 3.1e9], [4e9, nan, nan]]),
        "TotalCap": np.array([[2e9, nan, 2.1e9], [4e9, nan, 4.1e9], [6e9, nan, 6.1e9], [8e9, nan, nan]]),
        "FloatShares": np.array([[1e8, nan, 1.1e8], [2e8, nan, 2.1e8], [3e8, nan, 3.1e8], [4e8, nan, nan]]),
    }
    for name, exp in expect.items():
        got = np.load(out / f"daily_{name}.npy")
        assert got.dtype == np.float64
        assert got.shape == (4, 3)
        np.testing.assert_array_equal(got, exp)

    # "immediately readable by DailyFieldStore"
    dfs = DailyFieldStore(out)
    assert set(dfs.fields) == {"FloatCap", "TotalCap", "FloatShares"}
    assert dfs.symbols == symbols and dfs.dates == dates
    np.testing.assert_array_equal(np.asarray(dfs.plane("FloatCap")), expect["FloatCap"])


def test_log_float_cap_resolves_from_the_written_plane_without_the_script_writing_it(tmp_path):
    store = _store(tmp_path)
    symbols, dates = store.symbols, store.dates
    cap_dir = tmp_path / "daily_capital"
    for d in dates:
        _write_cap_file(cap_dir, d, symbols, cap=[1e9, 2e9, 3e9, 4e9], tot=[2e9, 4e9, 6e9, 8e9])
    out = tmp_path / "daily_fields"
    assert bdf.main(["--store", str(store.directory), "--cap-dir", str(cap_dir), "--out", str(out)]) == 0

    assert not (out / "daily_LogFloatCap.npy").exists()     # step 6: no Log* plane is ever written
    dfs = DailyFieldStore(out)
    got = resolve("LogFloatCap", symbols, dates, store=dfs)
    raw = np.load(out / "daily_FloatCap.npy")
    assert raw.dtype == np.float64
    expect = np.log(raw).astype(np.float32)
    np.testing.assert_array_equal(got, expect)


# ---------------------------------------------------------------------------
# the coverage floor: a total symbol-format mismatch must exit non-zero and
# show BOTH symbol samples, and must write nothing at --out.
# ---------------------------------------------------------------------------

def test_mismatched_symbols_exit_nonzero_on_the_coverage_floor_and_show_both_samples(tmp_path):
    store = _store(tmp_path)
    dates = store.dates
    foreign = ["AAAA.XX", "BBBB.XX", "CCCC.XX", "DDDD.XX"]      # zero overlap with store.symbols
    cap_dir = tmp_path / "daily_capital"
    for d in dates:
        _write_cap_file(cap_dir, d, foreign, cap=[1.0, 2.0, 3.0, 4.0], tot=[1.0, 2.0, 3.0, 4.0])

    out = tmp_path / "daily_fields"
    with pytest.raises(SystemExit) as ei:
        bdf.main(["--store", str(store.directory), "--cap-dir", str(cap_dir), "--out", str(out)])
    msg = str(ei.value)
    assert store.symbols[0] in msg                 # the store's own first symbol ...
    assert "AAAA.XX" in msg                         # ... next to the parquet's foreign one
    assert "coverage" in msg.lower() or "min-coverage" in msg
    assert not out.exists()                         # nothing written on a failed build


# ---------------------------------------------------------------------------
# must_fix #2 / step 2: refuse an --out inside --cap-dir or the store dir.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("which", ["cap_dir", "store_dir"])
def test_refuses_an_out_nested_inside_a_source_directory(tmp_path, which):
    cap_dir = tmp_path / "daily_capital"
    cap_dir.mkdir()
    store_dir = tmp_path / "cube"
    store_dir.mkdir()
    out = (cap_dir if which == "cap_dir" else store_dir) / "daily_fields"
    with pytest.raises(SystemExit, match="refusing to run"):
        bdf.main(["--store", str(store_dir), "--cap-dir", str(cap_dir), "--out", str(out)])


def test_refuses_an_out_exactly_equal_to_a_source_directory(tmp_path):
    cap_dir = tmp_path / "daily_capital"
    cap_dir.mkdir()
    store_dir = tmp_path / "cube"
    store_dir.mkdir()
    with pytest.raises(SystemExit, match="refusing to run"):
        bdf.main(["--store", str(store_dir), "--cap-dir", str(cap_dir), "--out", str(cap_dir)])


def test_an_out_that_is_merely_a_sibling_of_the_sources_is_not_refused(tmp_path):
    """Non-vacuous: the refuse check must not be so broad it rejects the
    normal layout (--out next to, not inside, the source directories)."""
    store = _store(tmp_path)
    cap_dir = tmp_path / "daily_capital"
    for d in store.dates:
        _write_cap_file(cap_dir, d, store.symbols, cap=[1e9] * 4, tot=[2e9] * 4)
    out = tmp_path / "daily_fields"          # sibling of cube/ and daily_capital/, not inside either
    assert bdf.main(["--store", str(store.directory), "--cap-dir", str(cap_dir), "--out", str(out),
                     "--min-coverage", "0.0"]) == 0
    assert out.is_dir()


# ---------------------------------------------------------------------------
# step 7: a build interrupted before the rename leaves nothing a
# DailyFieldStore will open.
# ---------------------------------------------------------------------------

def test_an_interrupted_write_leaves_nothing_a_daily_field_store_will_open(tmp_path, monkeypatch):
    planes = {"FloatCap": np.array([[1.0, 2.0], [3.0, 4.0]]), "TotalCap": np.array([[5.0, 6.0], [7.0, 8.0]])}
    out = tmp_path / "daily_fields"

    real_save = bdf.np.save
    calls = {"n": 0}

    def flaky_save(path, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:                 # let the FIRST field write, blow up on the second
            raise OSError("simulated interruption")
        return real_save(path, *a, **kw)

    monkeypatch.setattr(bdf.np, "save", flaky_save)
    with pytest.raises(OSError, match="simulated interruption"):
        bdf.write_daily_field_store(out, planes, ["A", "B"], ["20200101", "20200102"], {})

    assert not out.exists()                                   # nothing was ever renamed into place
    with pytest.raises(FileNotFoundError):
        DailyFieldStore(out)
    # and no stray temp directory was left behind next to it either
    leftovers = [p for p in tmp_path.glob(".daily_fields.*") if p.is_dir()]
    assert leftovers == []


def test_rerunning_against_an_existing_nonempty_out_refuses_rather_than_overwriting(tmp_path):
    """write_daily_field_store's own extension beyond the literal "os.replace
    it into place" (see this package's notes/deviations): os.replace cannot
    swap a directory onto a non-empty one, so a stale --out from a previous
    build must be refused, loudly, with the old directory left intact --
    never silently deleted and never left half up-to-date."""
    out = tmp_path / "daily_fields"
    out.mkdir()
    (out / "meta.json").write_text('{"fields": [], "symbols": [], "dates": []}', encoding="utf-8")
    (out / "sentinel.txt").write_text("old build", encoding="utf-8")

    planes = {"FloatCap": np.array([[1.0, 2.0]])}
    with pytest.raises(SystemExit, match="already exists"):
        bdf.write_daily_field_store(out, planes, ["A"], ["20200101", "20200102"], {})

    assert (out / "sentinel.txt").read_text(encoding="utf-8") == "old build"   # untouched
    assert not (out / "daily_FloatCap.npy").exists()                          # new data never landed


# ---------------------------------------------------------------------------
# --only, and FloatShares's "only if a real share column exists" rule.
# ---------------------------------------------------------------------------

def test_floatshares_is_omitted_when_no_share_column_is_present(tmp_path):
    store = _store(tmp_path)
    cap_dir = tmp_path / "daily_capital"
    for d in store.dates:
        _write_cap_file(cap_dir, d, store.symbols, cap=[1e9] * 4, tot=[2e9] * 4, shares=None)
    out = tmp_path / "daily_fields"
    assert bdf.main(["--store", str(store.directory), "--cap-dir", str(cap_dir), "--out", str(out),
                     "--min-coverage", "0.0"]) == 0
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert set(meta["fields"]) == {"FloatCap", "TotalCap"}
    assert not (out / "daily_FloatShares.npy").exists()
    assert "FloatShares" not in meta["columns_used"]


@pytest.mark.parametrize("candidate", ["circulating_shares", "float_shares", "share_count"])
def test_floatshares_is_built_from_whichever_candidate_column_is_actually_present(tmp_path, candidate):
    store = _store(tmp_path)
    cap_dir = tmp_path / "daily_capital"
    for d in store.dates:
        _write_cap_file(cap_dir, d, store.symbols, cap=[1e9] * 4, tot=[2e9] * 4, shares=[9e7] * 4,
                        share_column=candidate)
    out = tmp_path / "daily_fields"
    assert bdf.main(["--store", str(store.directory), "--cap-dir", str(cap_dir), "--out", str(out),
                     "--min-coverage", "0.0"]) == 0
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["columns_used"]["FloatShares"] == candidate
    np.testing.assert_array_equal(np.load(out / "daily_FloatShares.npy"), np.full((4, 3), 9e7))


def test_only_restricts_the_built_fields_even_when_more_columns_are_available(tmp_path):
    store = _store(tmp_path)
    cap_dir = tmp_path / "daily_capital"
    for d in store.dates:
        _write_cap_file(cap_dir, d, store.symbols, cap=[1e9] * 4, tot=[2e9] * 4, shares=[9e7] * 4)
    out = tmp_path / "daily_fields"
    assert bdf.main(["--store", str(store.directory), "--cap-dir", str(cap_dir), "--out", str(out),
                     "--only", "FloatCap", "--min-coverage", "0.0"]) == 0
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["fields"] == ["FloatCap"]
    assert not (out / "daily_TotalCap.npy").exists()
    assert not (out / "daily_FloatShares.npy").exists()


def test_only_with_an_unknown_field_name_exits_nonzero(tmp_path):
    columns = ["symbol", "circulating_market_value", "total_market_value"]
    with pytest.raises(SystemExit, match="Nonsense"):
        bdf.choose_columns(columns, ["Nonsense"])


def test_only_requesting_an_unbuildable_field_drops_it_without_raising(capsys):
    columns = ["symbol", "circulating_market_value", "total_market_value"]     # no share column
    mapping = bdf.choose_columns(columns, ["FloatCap", "FloatShares"])
    assert mapping == {"FloatCap": "circulating_market_value"}
    assert "FloatShares" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# schema-probe failure modes.
# ---------------------------------------------------------------------------

def test_choose_columns_raises_clearly_when_a_required_column_is_missing():
    columns = ["symbol", "circulating_market_value"]            # total_market_value missing
    with pytest.raises(SystemExit, match="total_market_value"):
        bdf.choose_columns(columns, None)


def test_probe_schema_raises_when_cap_dir_has_no_parquet_files_at_all(tmp_path):
    empty_cap_dir = tmp_path / "daily_capital"
    empty_cap_dir.mkdir()
    with pytest.raises(SystemExit, match="no .parquet files"):
        bdf.probe_schema(empty_cap_dir, ["20200101", "20200102"])


def test_probe_schema_falls_back_to_a_directory_listing_when_no_date_matches(tmp_path):
    cap_dir = tmp_path / "daily_capital"
    _write_cap_file(cap_dir, "20190101", ["600000.SH"], cap=[1e9], tot=[2e9])     # not a store date
    path, columns, symbols = bdf.probe_schema(cap_dir, ["20200101", "20200102"])
    assert path.name == "20190101.parquet"
    assert "circulating_market_value" in columns
    assert symbols == ["600000.SH"]


# ---------------------------------------------------------------------------
# --dry-run: reports and validates, writes nothing.
# ---------------------------------------------------------------------------

def test_dry_run_writes_nothing_but_still_runs_the_coverage_gate(tmp_path, capsys):
    store = _store(tmp_path)
    cap_dir = tmp_path / "daily_capital"
    for d in store.dates:
        _write_cap_file(cap_dir, d, store.symbols, cap=[1e9] * 4, tot=[2e9] * 4)
    out = tmp_path / "daily_fields"
    rc = bdf.main(["--store", str(store.directory), "--cap-dir", str(cap_dir), "--out", str(out),
                  "--dry-run"])
    assert rc == 0
    assert not out.exists()
    assert "--dry-run" in capsys.readouterr().out


def test_dry_run_still_exits_nonzero_below_the_coverage_floor(tmp_path):
    store = _store(tmp_path)
    cap_dir = tmp_path / "daily_capital"
    for d in store.dates:
        _write_cap_file(cap_dir, d, ["ZZZZ.XX"] * 4, cap=[1e9] * 4, tot=[2e9] * 4)
    out = tmp_path / "daily_fields"
    with pytest.raises(SystemExit):
        bdf.main(["--store", str(store.directory), "--cap-dir", str(cap_dir), "--out", str(out),
                 "--dry-run"])
    assert not out.exists()


# ---------------------------------------------------------------------------
# the store's own full symbol list must still satisfy require_board_rule
# (dailyfields.py's own documented second call site for this function).
# ---------------------------------------------------------------------------

def _bad_symbol_store(tmp_path):
    """A store whose symbols put the market prefix LAST ('SH.600000'), so
    ``dailyfields.require_board_rule`` can never fire a board rule on it."""
    from alphacube.store import build_cube_store
    from alphacube.store import DAY_STATUS_CODES

    S, D, M = 2, 3, 2
    cubes = {k: np.ones((S, D, M)) for k in ("Open", "High", "Low", "Close", "Volume", "Vwap", "Ret", "Amt")}
    status = np.full((S, D), DAY_STATUS_CODES["TRADING"], dtype=np.int8)
    return build_cube_store(tmp_path / "bad_cube", cubes, ["SH.600000", "SH.600001"],
                            ["20200101", "20200102", "20200103"], day_status=status)


def test_a_store_whose_symbols_can_never_match_a_board_rule_is_refused(tmp_path):
    bad_store = _bad_symbol_store(tmp_path)
    cap_dir = tmp_path / "daily_capital"
    out = tmp_path / "daily_fields"
    with pytest.raises(SystemExit, match="STAR.*GEM.*BSE|digit"):
        bdf.main(["--store", str(bad_store.directory), "--cap-dir", str(cap_dir), "--out", str(out)])


# ---------------------------------------------------------------------------
# $ALPHACUBE_DAILY_FIELDS: the server shell's own value must not reach the
# label-only store open. open_cube_store defaults daily_root to that env var
# (store.py:444-445), and the design documents it as THE way to point
# AlphaCube at the daily store -- so on the server it is most naturally set to
# exactly this script's --out, which on a first build does not exist yet.
# ---------------------------------------------------------------------------

def _tiny_build_inputs(tmp_path):
    store = _store(tmp_path)
    cap_dir = tmp_path / "daily_capital"
    for d in store.dates:
        _write_cap_file(cap_dir, d, store.symbols, cap=[1e9] * 4, tot=[2e9] * 4)
    return store, cap_dir


def test_builds_with_alphacube_daily_fields_pointing_at_the_not_yet_built_out(tmp_path, monkeypatch):
    """The regression: this is the script that CREATES the daily store, so a
    shell that already exports ALPHACUBE_DAILY_FIELDS=<out> must not make it
    die on ``FileNotFoundError: <out>/meta.json`` before it does anything."""
    store, cap_dir = _tiny_build_inputs(tmp_path)
    out = tmp_path / "not_built_yet"
    monkeypatch.setenv("ALPHACUBE_DAILY_FIELDS", str(out))          # the very dir to build

    rc = bdf.main(["--store", str(store.directory), "--cap-dir", str(cap_dir), "--out", str(out),
                  "--min-coverage", "0.0"])
    assert rc == 0
    assert set(DailyFieldStore(out).fields) == {"FloatCap", "TotalCap"}
    # the operator's own value is left exactly as it was for the rest of the process
    assert os.environ["ALPHACUBE_DAILY_FIELDS"] == str(out)


def test_an_unset_alphacube_daily_fields_is_not_resurrected_by_the_open(tmp_path, monkeypatch):
    """Mirror image: the neutralising must restore *absence* as absence, not
    plant an empty string that a later open_cube_store would try to use."""
    store, cap_dir = _tiny_build_inputs(tmp_path)
    monkeypatch.delenv("ALPHACUBE_DAILY_FIELDS", raising=False)
    assert bdf.main(["--store", str(store.directory), "--cap-dir", str(cap_dir),
                     "--out", str(tmp_path / "daily_fields"), "--min-coverage", "0.0"]) == 0
    assert "ALPHACUBE_DAILY_FIELDS" not in os.environ


def test_a_stale_daily_store_in_the_environment_is_never_attached(tmp_path, monkeypatch):
    """Design step 1: the store is opened "read-only purely for its canonical
    symbols/dates". An existing (stale) store named by the env var must not be
    attached and mixed into contract_hash behind the operator's back."""
    store, cap_dir = _tiny_build_inputs(tmp_path)
    stale = tmp_path / "stale_daily"
    assert bdf.main(["--store", str(store.directory), "--cap-dir", str(cap_dir), "--out", str(stale),
                     "--min-coverage", "0.0"]) == 0
    assert (stale / "meta.json").is_file()                     # a real, openable daily store

    monkeypatch.setenv("ALPHACUBE_DAILY_FIELDS", str(stale))
    opened = bdf.open_store_for_labels(Path(store.directory))
    assert opened.daily is None                                # not attached ...
    from alphacube.store import CubeStore
    assert opened.contract_hash == CubeStore(store.directory).contract_hash   # ... nor fingerprinted

    # and a second build still runs end to end with that stale store exported
    assert bdf.main(["--store", str(store.directory), "--cap-dir", str(cap_dir),
                     "--out", str(tmp_path / "daily_fields"), "--min-coverage", "0.0"]) == 0


def test_the_board_rule_refusal_survives_a_set_alphacube_daily_fields(tmp_path, monkeypatch):
    """With the env var honoured, CubeStore.__init__ ran require_board_rule
    itself -- ahead of this script's own call -- so a bad symbol spelling
    escaped as a raw ValueError instead of main()'s SystemExit."""
    bad_store = _bad_symbol_store(tmp_path)
    monkeypatch.setenv("ALPHACUBE_DAILY_FIELDS", str(tmp_path / "not_built_yet"))
    with pytest.raises(SystemExit, match="STAR.*GEM.*BSE|digit"):
        bdf.main(["--store", str(bad_store.directory), "--cap-dir", str(tmp_path / "daily_capital"),
                 "--out", str(tmp_path / "daily_fields")])
