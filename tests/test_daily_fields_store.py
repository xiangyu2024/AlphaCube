"""WP3 -- ``CubeView.daily()``, store wiring (``CubeStore``/``SegmentedCubeStore``/
``open_cube_store``'s ``daily_root``, ``contract_hash``), the identifiable
synthetic daily-field planes (``synthetic.make_synthetic_cube``'s
``daily_fields`` kwarg, ``write_daily_fields``), and the new fixture symbol
spelling (``synthetic._synthetic_symbols``). See
docs/alphacube_steering_design.md, section "Where the data meets a view:
CubeView.daily(name)" / "Store wiring" / "must_fix #2", and work package
"### WP3".

DEVIATION, measured (not asserted-in-prose): the design's own precision claim
in section "The field set" -- that ``ts_delta(LogFloatCap, 1)`` recovers a
day-over-day move MORE precisely than ``ts_delta(FloatCap, 1)`` at a
1e9-1e13 cap magnitude -- is false. ``dailyfields.py``'s own module docstring
(written in WP2) already corrects this with measurements; this file pins the
OPPOSITE (measured-correct) direction through CubeView.daily(), not the
design's stated one. See test_ts_delta_style_precision_favors_floatcap_not_
logfloatcap below.
"""

from __future__ import annotations

import numpy as np
import pytest

import alphacube.dailyfields as dailyfields
from alphacube.research import warmed_view
from alphacube.store import CubeStore, DAY_STATUS_CODES, SegmentedCubeStore, build_cube_store, open_cube_store, write_segment
from alphacube.synthetic import _synthetic_symbols, make_synthetic_cube, write_daily_fields, write_synthetic_store
from alphacube.tradability import board_limit


# ---------------------------------------------------------------------------
# synthetic.py: the new symbol spelling and the daily_fields kwarg
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 12, 23])
def test_synthetic_symbols_are_unique_and_the_right_count_at_every_size(n):
    syms = _synthetic_symbols(n)
    assert len(syms) == n
    assert len(set(syms)) == n


def test_synthetic_symbols_put_every_board_rule_within_reach():
    """must_fix #2's fix itself: on the OLD 'SH.6xxxxx' spelling no board rule
    ever fired (every symbol's first character is a letter). The new spelling
    must put a digit first for every symbol, and for n_symbols >= 3 must
    cover all three special boards."""
    for n in (3, 4, 10, 12):
        syms = _synthetic_symbols(n)
        assert all(s[:1].isdigit() for s in syms)
        star = any(s.startswith(("688", "689")) for s in syms)
        gem = any(s.startswith(("300", "301", "302")) for s in syms)
        bse = any(s.startswith(("4", "8", "92")) for s in syms)
        assert star and gem and bse, syms


def test_make_synthetic_cube_default_daily_fields_is_empty_and_arity_is_five():
    out = make_synthetic_cube(n_symbols=4, n_days=5, n_minutes=2, seed=0)
    assert len(out) == 5
    assert out[4] == {}


def test_daily_fields_kwarg_does_not_perturb_the_rng_stream_or_the_first_four_values():
    """Backward compatibility, the hard requirement: asking for the identifiable
    planes is plain index arithmetic, not ``rng``, so it must not move the RNG
    stream or change anything about cubes/symbols/dates/status."""
    a = make_synthetic_cube(n_symbols=6, n_days=7, n_minutes=3, seed=5, suspended_frac=0.1, unlisted_days=2)
    b = make_synthetic_cube(n_symbols=6, n_days=7, n_minutes=3, seed=5, suspended_frac=0.1, unlisted_days=2,
                            daily_fields=("FloatCap", "TotalCap", "FloatShares"))
    for name in a[0]:
        assert np.array_equal(a[0][name], b[0][name], equal_nan=True)
    assert a[1] == b[1] and a[2] == b[2]
    np.testing.assert_array_equal(a[3], b[3])
    assert a[4] == {}
    assert set(b[4]) == {"FloatCap", "TotalCap", "FloatShares"}


def test_identifiable_planes_formula_and_relationships():
    _, symbols, dates, _, planes = make_synthetic_cube(n_symbols=4, n_days=3, n_minutes=2, seed=0,
                                                        daily_fields=("FloatCap", "TotalCap", "FloatShares"))
    s, d = np.arange(4)[:, None], np.arange(3)[None, :]
    expect_fc = 1e9 * (s + 1.0) + d
    np.testing.assert_array_equal(planes["FloatCap"], expect_fc)
    np.testing.assert_allclose(planes["TotalCap"], expect_fc / 0.6)
    np.testing.assert_allclose(planes["FloatShares"], expect_fc / 10.0)
    for name, arr in planes.items():
        assert arr.dtype == np.float64
        assert arr.shape == (4, 3)


def test_daily_fields_kwarg_only_returns_the_requested_names():
    _, _, _, _, planes = make_synthetic_cube(n_symbols=3, n_days=2, n_minutes=2, daily_fields=("TotalCap",))
    assert set(planes) == {"TotalCap"}


def test_daily_fields_kwarg_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="FloatCap"):
        make_synthetic_cube(n_symbols=3, n_days=2, n_minutes=2, daily_fields=("Nonsense",))


def test_write_synthetic_store_still_returns_a_plain_cube_store_when_daily_fields_is_forwarded(tmp_path):
    """The exact hazard the design's own 'Judges' section quotes against a
    REJECTED sibling design: a keyword-gated 5th element that write_synthetic_
    store's fixed-arity unpack cannot survive. Here the unpack is fixed at
    five unconditionally, so forwarding daily_fields through **kwargs must not
    raise, and write_synthetic_store's own return type is unaffected."""
    st = write_synthetic_store(tmp_path / "cube", n_symbols=4, n_days=5, n_minutes=2, seed=1,
                               daily_fields=("FloatCap", "TotalCap"))
    assert isinstance(st, CubeStore)
    assert st.daily is None          # write_synthetic_store never attaches the planes itself


def test_write_daily_fields_rejects_a_plane_with_the_wrong_shape(tmp_path):
    with pytest.raises(ValueError, match="shape"):
        write_daily_fields(tmp_path / "bad", {"FloatCap": np.zeros((2, 3))}, ["a"], ["1", "2", "3"])


def test_write_daily_fields_writes_exactly_what_daily_field_store_reads(tmp_path):
    planes = {"FloatCap": np.array([[1.0, 2.0], [3.0, 4.0]])}
    root = write_daily_fields(tmp_path / "d", planes, ["A", "B"], ["20200101", "20200102"])
    dfs = dailyfields.DailyFieldStore(root)
    assert dfs.fields == ("FloatCap",) and dfs.symbols == ["A", "B"] and dfs.dates == ["20200101", "20200102"]
    np.testing.assert_array_equal(np.asarray(dfs.plane("FloatCap")), planes["FloatCap"])


# ---------------------------------------------------------------------------
# store.py: daily_root wiring on CubeStore / SegmentedCubeStore / open_cube_store
# ---------------------------------------------------------------------------

def test_cube_store_has_no_daily_source_by_default(store):
    assert store.daily is None


def test_cube_store_attaches_a_daily_field_store_when_given_a_root(store, daily_root):
    reopened = CubeStore(store.directory, daily_root=daily_root)
    assert isinstance(reopened.daily, dailyfields.DailyFieldStore)
    assert set(reopened.daily.fields) == {"FloatCap", "TotalCap", "FloatShares"}


def test_contract_hash_is_byte_identical_without_a_daily_source(store):
    """Backward compatibility (design item 7): building the SAME store twice
    with no daily_root must give the identical hash, and must equal the
    original fixture's own hash."""
    reopened = CubeStore(store.directory)
    assert reopened.contract_hash == store.contract_hash


def test_contract_hash_changes_with_a_daily_source_attached_cube_store(store, daily_root):
    with_daily = CubeStore(store.directory, daily_root=daily_root)
    assert with_daily.contract_hash != store.contract_hash
    # deterministic: attaching the SAME daily_root twice gives the SAME hash
    again = CubeStore(store.directory, daily_root=daily_root)
    assert again.contract_hash == with_daily.contract_hash


def test_daily_attach_runs_require_board_rule_over_the_full_store_symbols(tmp_path, daily_root):
    """dailyfields.require_board_rule's own docstring names its call site:
    'the daily-source attach on CubeStore / SegmentedCubeStore (over
    store.symbols)'. A store whose FULL symbol list cannot ever match a board
    rule (the pre-WP3 'SH.6xxxxx' spelling, or any other non-prefix-first
    one) must fail LOUDLY at attach time, not ship a silently-constant
    LimitWidth -- must_fix #2's composition half, closed at construction."""
    bad_store = build_cube_store(
        tmp_path / "bad_cube",
        {"Close": np.ones((2, 3, 2)), "Open": np.ones((2, 3, 2)), "High": np.ones((2, 3, 2)),
         "Low": np.ones((2, 3, 2)), "Volume": np.ones((2, 3, 2)), "Vwap": np.ones((2, 3, 2)),
         "Ret": np.ones((2, 3, 2)), "Amt": np.ones((2, 3, 2))},
        ["SH.600000", "SH.600001"], ["20200101", "20200102", "20200103"],
    )
    with pytest.raises(ValueError, match="STAR.*GEM.*BSE|digit"):
        CubeStore(bad_store.directory, daily_root=daily_root)
    # a store with NO daily_root is completely unaffected (no behavior change
    # for the overwhelming majority of existing callers, which never pass one)
    assert CubeStore(bad_store.directory).daily is None


def _two_segment_store(tmp_path, daily_fields=()):
    cubes, symbols, dates, status, planes = make_synthetic_cube(n_symbols=4, n_days=16, n_minutes=2, seed=2,
                                                                 daily_fields=daily_fields)
    comp = tmp_path / "comp"
    build_cube_store(comp / "minute", {k: v[:, :10] for k, v in cubes.items()}, symbols, dates[:10], status[:, :10])
    write_segment(comp, "seg1", {k: v[:, 10:] for k, v in cubes.items()}, symbols, dates[10:], status[:, 10:])
    return comp, symbols, dates, planes


def test_contract_hash_unchanged_and_changed_for_a_two_segment_segmented_store(tmp_path):
    comp, symbols, dates, planes = _two_segment_store(tmp_path, daily_fields=("FloatCap",))
    plain_a, plain_b = SegmentedCubeStore(comp), SegmentedCubeStore(comp)
    assert plain_a.daily is None
    assert plain_a.contract_hash == plain_b.contract_hash         # deterministic, no daily source

    droot = write_daily_fields(tmp_path / "daily", planes, symbols, dates)
    with_daily = SegmentedCubeStore(comp, daily_root=droot)
    assert with_daily.daily is not None
    assert with_daily.contract_hash != plain_a.contract_hash      # changes once attached


def test_open_cube_store_forwards_daily_root_to_either_class(tmp_path, daily_root):
    """open_cube_store picks SegmentedCubeStore for a compute root, CubeStore for
    a bare minute directory -- daily_root must reach whichever it returns."""
    comp, _, _, _ = _two_segment_store(tmp_path)
    seg = open_cube_store(comp, daily_root=daily_root)
    assert isinstance(seg, SegmentedCubeStore) and seg.daily is not None

    mono = write_synthetic_store(tmp_path / "mono", n_symbols=4, n_days=5, n_minutes=2, seed=3)
    bare = open_cube_store(mono.directory, daily_root=daily_root)
    assert isinstance(bare, CubeStore) and bare.daily is not None


def test_open_cube_store_env_var_default_and_direct_construction_ignores_it(monkeypatch, tmp_path, daily_root):
    """design step 3: only open_cube_store consults $ALPHACUBE_DAILY_FIELDS.
    CubeStore/SegmentedCubeStore built directly (as virtually every test in
    this suite does) must stay deterministic regardless of the caller's
    environment."""
    mono = write_synthetic_store(tmp_path / "mono2", n_symbols=4, n_days=5, n_minutes=2, seed=3)
    monkeypatch.setenv("ALPHACUBE_DAILY_FIELDS", str(daily_root))
    via_open = open_cube_store(mono.directory)
    assert via_open.daily is not None

    direct = CubeStore(mono.directory)
    assert direct.daily is None, "CubeStore's own constructor must not read the environment"

    # an explicit daily_root argument still wins over a (deliberately wrong)
    # environment value
    monkeypatch.setenv("ALPHACUBE_DAILY_FIELDS", "/does/not/exist")
    via_open_explicit = open_cube_store(mono.directory, daily_root=daily_root)
    assert via_open_explicit.daily is not None


# ---------------------------------------------------------------------------
# CubeView.daily_fields / daily() / daily_s() / daily_d()
# ---------------------------------------------------------------------------

def test_daily_fields_property_derived_always_stored_and_log_only_with_a_store(store, daily_view):
    assert set(store.view().daily_fields) == {"LimitWidth", "BoardCode", "ListAge"}
    assert set(daily_view.daily_fields) == {
        "FloatCap", "TotalCap", "FloatShares", "LimitWidth", "BoardCode", "ListAge",
        "LogFloatCap", "LogTotalCap", "LogFloatShares",
    }


def test_daily_resolves_once_per_view_then_serves_from_cache(daily_view, monkeypatch):
    calls = []
    real_resolve = dailyfields.resolve

    def counting(*a, **kw):
        calls.append(a[0])
        return real_resolve(*a, **kw)

    monkeypatch.setattr(dailyfields, "resolve", counting)
    fresh_view = daily_view.store.view()           # a view dailyfields.resolve has not seen yet
    fresh_view.daily("FloatCap")
    fresh_view.daily("FloatCap")
    fresh_view.daily("FloatCap")
    fresh_view.daily("TotalCap")
    assert calls == ["FloatCap", "TotalCap"]


def test_daily_s_and_daily_d_are_thin_slices_of_the_same_cached_array(daily_view, monkeypatch):
    calls = []
    real_resolve = dailyfields.resolve
    monkeypatch.setattr(dailyfields, "resolve", lambda *a, **kw: (calls.append(a[0]), real_resolve(*a, **kw))[1])

    fresh_view = daily_view.store.view()
    full = fresh_view.daily("FloatCap")
    s_slice = fresh_view.daily_s("FloatCap", 1, 3)
    d_slice = fresh_view.daily_d("FloatCap", 2, 5)
    assert calls == ["FloatCap"]                   # daily_s/daily_d must not trigger a second resolve
    np.testing.assert_array_equal(s_slice, full[1:3, :])
    np.testing.assert_array_equal(d_slice, full[:, 2:5])


def test_daily_view_float_cap_matches_the_identifiable_plane_exactly(daily_view, store):
    fc = daily_view.daily("FloatCap")
    assert fc.dtype == np.float32
    s, d = np.arange(store.n_symbols)[:, None], np.arange(store.n_days)[None, :]
    expect = (1e9 * (s + 1.0) + d).astype(np.float32)
    np.testing.assert_array_equal(fc, expect)


def test_daily_returns_s_d_orientation_not_d_s(daily_view, store):
    assert daily_view.daily("FloatCap").shape == (store.n_symbols, store.n_days)


def test_derived_fields_resolve_on_a_store_with_no_daily_root_attached(view):
    """design step 7 / test bullet: LimitWidth/BoardCode/ListAge need no build
    step and must work even though `store` (hence `view`) has no daily source
    attached at all."""
    assert view.store.daily is None
    for name in ("LimitWidth", "BoardCode", "ListAge"):
        plane = view.daily(name)
        assert plane.shape == (view.n_symbols, view.n_days)
        assert plane.dtype == np.float32


# ---------------------------------------------------------------------------
# A non-ascending symbol subset -- store.view's fancy-index path (store.py
# _contiguous=False) must still align daily() by LABEL, not by position.
# ---------------------------------------------------------------------------

def test_daily_aligns_by_label_through_a_non_ascending_symbol_subset(store, daily_root):
    st = CubeStore(store.directory, daily_root=daily_root)
    full = st.view().daily("FloatCap")
    picks = [store.symbols[7], store.symbols[2], store.symbols[9]]     # deliberately non-ascending
    sub = st.view(symbols=picks)
    assert sub._contiguous is False
    got = sub.daily("FloatCap")
    expect = full[[7, 2, 9], :]
    np.testing.assert_array_equal(got, expect)
    assert sub.symbols == picks


def test_daily_nan_fills_exactly_the_symbols_and_dates_the_daily_store_does_not_cover(tmp_path):
    """Extends test_daily_leaf_typing.py's DailyFieldStore.align coverage (WP2,
    a file WP3 does not own) up through CubeStore/CubeView: a daily source
    that only partially overlaps the minute store's symbols/dates must
    NaN-fill exactly the uncovered rows/columns, not raise and not silently
    mis-align."""
    st = write_synthetic_store(tmp_path / "cube", n_symbols=5, n_days=6, n_minutes=2, seed=4)
    partial_syms = [st.symbols[1], st.symbols[3]]
    partial_dates = st.dates[2:]
    planes = {"FloatCap": np.array([[1e9, 2e9, 3e9, 4e9], [5e9, 6e9, 7e9, 8e9]], dtype=np.float64)}
    droot = write_daily_fields(tmp_path / "daily", planes, partial_syms, partial_dates)

    view = CubeStore(st.directory, daily_root=droot).view()
    with pytest.warns(RuntimeWarning):          # this IS the near-empty-plane case the warning is for
        fc = view.daily("FloatCap")
    finite_rows = [i for i in range(5) if np.isfinite(fc[i]).any()]
    assert finite_rows == [1, 3]
    assert np.isnan(fc[:, :2]).all()
    np.testing.assert_array_equal(fc[1, 2:], np.array([1e9, 2e9, 3e9, 4e9], dtype=np.float32))
    np.testing.assert_array_equal(fc[3, 2:], np.array([5e9, 6e9, 7e9, 8e9], dtype=np.float32))


# ---------------------------------------------------------------------------
# View independence: a symbol subset AND a warmed_view-padded date range must
# both line up with the full view's own plane, for free (design step 4/5).
# ---------------------------------------------------------------------------

def test_view_independence_symbol_subset_and_warmed_view_padding(store, daily_root):
    st = CubeStore(store.directory, daily_root=daily_root)
    full = st.view().daily("FloatCap")
    subset_syms = [store.symbols[7], store.symbols[1], store.symbols[4]]    # non-ascending
    padded, pad = warmed_view(st, start=store.dates[20], end=store.dates[40], warmup=10, symbols=subset_syms)
    assert pad == 10 and padded.d0 == 10          # plenty of room before day 20; padding is exact
    got = padded.daily("FloatCap")
    idx = [store.symbols.index(s) for s in subset_syms]
    np.testing.assert_array_equal(got, full[idx][:, padded.d0:padded.d1])


# ---------------------------------------------------------------------------
# derive('LimitWidth') / derive('BoardCode') on the NEW fixture symbol
# spelling, including the 2020-08-24 GEM registration-system switch.
# ---------------------------------------------------------------------------

def test_limit_width_matches_board_limit_on_the_new_fixture_symbols_incl_gem_switch(view):
    dates = ["20200101", "20200824", "20200825", "20210101"]      # straddles the switch
    lw = dailyfields.derive("LimitWidth", view.symbols, dates)
    np.testing.assert_array_equal(lw, board_limit(view.symbols, dates).T)
    # every rule must have fired on this fixture -- not the pre-WP3 uniform plane
    assert set(np.unique(lw).tolist()) == {0.10, 0.20, 0.30}
    gem_row = lw[view.symbols.index("300001.SZ")]
    assert gem_row.tolist() == [0.10, 0.20, 0.20, 0.20]           # 0.10 -> 0.20 exactly at 20200824
    bse_row = lw[view.symbols.index("830001.BJ")]
    assert set(bse_row.tolist()) == {0.30}
    star_row = lw[view.symbols.index("688001.SH")]
    assert set(star_row.tolist()) == {0.20}


def test_board_code_matches_the_same_prefix_rule_on_the_new_fixture_symbols(view):
    bc = dailyfields.derive("BoardCode", view.symbols, view.dates)
    assert bc[view.symbols.index("688001.SH"), 0] == dailyfields.BOARD_STAR
    assert bc[view.symbols.index("300001.SZ"), 0] == dailyfields.BOARD_GEM
    assert bc[view.symbols.index("830001.BJ"), 0] == dailyfields.BOARD_BSE
    assert bc[view.symbols.index("600000.SH"), 0] == dailyfields.BOARD_MAIN


# ---------------------------------------------------------------------------
# derive('ListAge') on the unlisted_days=8 fixture.
# ---------------------------------------------------------------------------

def test_list_age_sentinel_and_sticky_through_suspension_on_the_fixture(daily_view, store):
    """Deterministic despite the fixture's RANDOM suspended_frac: suspended_frac
    only ever sets SUSPENDED, never OUTSIDE_LIFECYCLE, so every symbol except
    the deliberately-unlisted last one is "in lifecycle" on EVERY day of this
    fixture regardless of which days happen to be suspended."""
    age = daily_view.daily("ListAge")
    assert age.shape == (store.n_symbols, store.n_days)

    expect_listed_from_day_zero = 10000.0 + np.arange(store.n_days)
    for s in range(store.n_symbols - 1):
        np.testing.assert_array_equal(age[s], expect_listed_from_day_zero)

    last = age[-1]                                 # make_synthetic_cube's unlisted_days=8 target
    assert np.isnan(last[:8]).all()
    np.testing.assert_array_equal(last[8:], np.arange(store.n_days - 8, dtype=np.float64))

    # non-vacuous: this fixture really does contain a suspension, and the
    # +1-per-day property holds straight through it for every listed symbol.
    assert (store.day_status() == DAY_STATUS_CODES["SUSPENDED"]).any()
    assert np.all(np.diff(age[:-1], axis=1) == 1.0)


# ---------------------------------------------------------------------------
# LogFloatCap: exact log-of-float64-plane, and the (corrected) precision
# direction at cap magnitude -- see module docstring DEVIATION.
# ---------------------------------------------------------------------------

def test_log_float_cap_equals_log_of_the_float64_stored_plane(daily_view, daily_root):
    raw = np.load(daily_root / "daily_FloatCap.npy")
    assert raw.dtype == np.float64
    expect = np.log(raw).astype(np.float32)
    got = daily_view.daily("LogFloatCap")
    np.testing.assert_array_equal(got, expect)


def test_ts_delta_style_precision_favors_floatcap_not_logfloatcap(tmp_path):
    """See module docstring DEVIATION. Built on a dedicated tiny store (not the
    conftest identifiable-ramp fixture, whose day-over-day delta is a fixed
    +1.0 absolute -- too small a RELATIVE move at 1e9-scale to be the
    interesting case) so the magnitude and the planted move match the
    design's own example exactly: a 1e12-scale cap, a 1e-3 relative
    day-over-day move."""
    st = write_synthetic_store(tmp_path / "cube", n_symbols=2, n_days=2, n_minutes=2, seed=0)
    cap0, true_move = 1e12, 1e-3
    float_cap = np.array([[cap0, cap0 * (1.0 + true_move)], [cap0, cap0]], dtype=np.float64)
    droot = write_daily_fields(tmp_path / "daily", {"FloatCap": float_cap}, st.symbols, st.dates)
    view = CubeStore(st.directory, daily_root=droot).view()

    fc = view.daily("FloatCap").astype(np.float64)
    lfc = view.daily("LogFloatCap").astype(np.float64)
    direct_delta = (fc[0, 1] - fc[0, 0]) / fc[0, 0]
    log_delta = lfc[0, 1] - lfc[0, 0]
    true_log_move = np.log1p(true_move)

    direct_rel_err = abs(direct_delta - true_move) / true_move
    log_rel_err = abs(log_delta - true_log_move) / true_log_move
    assert direct_rel_err < 1e-4                  # "recovers... to better than" -- FloatCap's side
    assert log_rel_err > direct_rel_err            # the MEASURED direction, opposite the design's text


# ---------------------------------------------------------------------------
# contract_hash reaching far enough to matter: a changed daily source gives a
# changed hash even when nothing about the minute cube moved (sanity for the
# Partura-side cache-staleness risk the design's "risks" section names).
#
# NOTE (worth knowing, not a WP3 defect): fingerprint() hashes the DailyField
# Store's meta.json -- symbols/dates/fields -- not the .npy array bytes (see
# dailyfields.fingerprint's own docstring). Two roots with the SAME labels but
# different array VALUES therefore give the SAME fingerprint; only a change
# that moves meta.json's content (a different date range below, or formula
# changes via DERIVED_VERSION) is detected. Both properties are pinned here.
# ---------------------------------------------------------------------------

def test_two_daily_roots_with_different_meta_give_different_contract_hashes(store, tmp_path):
    _, _, _, _, planes_a = make_synthetic_cube(n_symbols=store.n_symbols, n_days=store.n_days,
                                               daily_fields=("FloatCap",))
    root_a = write_daily_fields(tmp_path / "a", planes_a, store.symbols, store.dates)
    # one fewer date changes meta.json's content -- and hence its sha256.
    root_b = write_daily_fields(tmp_path / "b", {"FloatCap": planes_a["FloatCap"][:, :-1]},
                                store.symbols, store.dates[:-1])
    hash_a = CubeStore(store.directory, daily_root=root_a).contract_hash
    hash_b = CubeStore(store.directory, daily_root=root_b).contract_hash
    assert hash_a != hash_b


def test_two_daily_roots_with_the_same_meta_but_different_values_give_the_same_fingerprint(store, tmp_path):
    """The flip side of the note above, pinned so it is a documented property,
    not a surprise: fingerprint()/contract_hash track the daily store's LABELS
    (meta.json), not its array contents."""
    _, _, _, _, planes_a = make_synthetic_cube(n_symbols=store.n_symbols, n_days=store.n_days,
                                               daily_fields=("FloatCap",))
    root_a = write_daily_fields(tmp_path / "a", planes_a, store.symbols, store.dates)
    root_b = write_daily_fields(tmp_path / "b", {"FloatCap": planes_a["FloatCap"] + 1.0},
                                store.symbols, store.dates)
    hash_a = CubeStore(store.directory, daily_root=root_a).contract_hash
    hash_b = CubeStore(store.directory, daily_root=root_b).contract_hash
    assert hash_a == hash_b
