"""WP2 -- the ``DailyFeature`` leaf (expression.py), its parse branch
(parser.py), and the (S, D) field resolver (dailyfields.py). See
docs/alphacube_steering_design.md, section "The leaf" / "Operator semantics"
/ "is_valid_root" / "The field set" / "New module dailyfields.py" /
"must_fix #2", and work package "### WP2".

DEVIATION from the design's own test-plan text: the example operator
``cs_bucket_rank`` (and ``cs_winsor``) named in the design's WP2 test bullet
does not exist in AlphaCube's own registry -- it is a Partura-plugin-only ND
kernel (``partura/ops_ext.py``'s ``NEW_OPERATORS``), and "AlphaCube never
imports Partura" (design Overview). ``cs_regres`` is used instead: the only
ND-face, arity-2, non-boolean operator AlphaCube ships, which exercises the
exact same thing the design's example was for -- an ND op fed one operand
that is COLLAPSED via a day_* reduction and one that is a bare DailyFeature.
"""

import warnings

import numpy as np
import pytest

from alphacube.catalog import (
    CUBE_FIELDS,
    DAILY_FIELDS,
    DERIVED_DAILY_FIELDS,
    LOG_DAILY_FIELDS,
    STORED_DAILY_FIELDS,
    Registry,
    build_registry,
)
from alphacube.expression import COLLAPSED, FULL, WIDE_DEGENERATE, DailyFeature, Feature, is_valid_root
from alphacube.parser import (
    DAILY_FIELD_ALIASES,
    DAILY_MACROS,
    DAILY_NAME_ALIASES,
    FIELD_ALIASES,
    ParseError,
    parse,
    parse_legacy,
)
from alphacube.fullcatalog import OPERATOR_ARITY_ALIASES, OPERATOR_NAME_ALIASES, default_full_catalog

from alphacube import dailyfields as df


# ---------------------------------------------------------------------------
# The leaf itself
# ---------------------------------------------------------------------------

def test_shape_is_the_collapsed_constant_by_identity():
    assert DailyFeature("FloatCap").shape is COLLAPSED


def test_key_is_tagged_and_does_not_collide_with_a_same_named_feature():
    daily = DailyFeature("Close")
    minute = Feature("Close")
    assert daily.key() == "DF:Close"
    assert minute.key() == "F:Close"
    assert daily.key() != minute.key()
    assert daily != minute
    assert hash(daily) != hash(minute)
    assert repr(daily) == "Close"                 # _repr is just the field name, like Feature


# ---------------------------------------------------------------------------
# Shape / is_valid_root over trees containing a daily leaf
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("src, shape, root", [
    ("cs_rank(FloatCap)", COLLAPSED, True),
    ("div(day_sum(Amt), FloatCap)", COLLAPSED, True),
    ("cs_regres(day_max(Ret), FloatCap)", COLLAPSED, True),          # see module docstring DEVIATION
    ("mul(Ret, LimitWidth)", FULL, False),
    ("day_mean(mul(Ret, LimitWidth))", COLLAPSED, True),
    ("ts_delta(LogFloatCap, 5)", COLLAPSED, True),
    ("cross_mean(FloatCap, 5)", COLLAPSED, True),
    ("FloatCap", COLLAPSED, True),
])
def test_shape_typing_with_daily_leaves(src, shape, root):
    e = parse(src)
    assert e.shape == shape
    assert is_valid_root(e) is root


@pytest.mark.parametrize("src", [
    "day_sum(FloatCap)",               # dt/day, needs_full_m
    "in_mean_rolling(FloatCap, 5)",    # dt/in, needs_full_m
    "nt_ts_mean(FloatCap, 5)",         # nt, needs_full_m
])
def test_bare_daily_leaf_rejected_by_minute_ops_in_both_modes(src):
    with pytest.raises(ParseError, match="daily-only field"):
        parse(src)                              # strict
    with pytest.raises(ParseError, match="daily-only field"):
        parse_legacy(src)                        # legacy: the new branch is unconditional


def test_in_expand_over_a_bare_daily_leaf_is_legal_and_wide_degenerate():
    e = parse("in_expand(FloatCap, 241)")          # restores_m branch is untouched by WP2
    assert e.shape == WIDE_DEGENERATE
    assert is_valid_root(e)


def test_in_mean_rolling_over_that_in_expand_still_raises_both_modes():
    src = "in_mean_rolling(in_expand(FloatCap, 241), 5)"
    with pytest.raises(ParseError):
        parse(src)
    with pytest.raises(ParseError):
        parse_legacy(src)


def test_why_not_message_names_the_daily_leaf_specifically():
    """The new _why_not clause is distinguishable from the pre-existing
    'minute-degenerate operand' message (test_in_mean_rolling_over_that_in_expand_
    still_raises_both_modes hits the OLD message instead, for contrast)."""
    with pytest.raises(ParseError, match="FloatCap.*daily-only field with no minute axis"):
        parse("day_sum(FloatCap)")


# ---------------------------------------------------------------------------
# The derived-operand carve-out: pinned, not fixed (design "risks" section)
# ---------------------------------------------------------------------------

def test_day_sum_of_log_floatcap_is_the_pinned_width_one_identity():
    """log(FloatCap) is COLLAPSED (collapsed=True), so the PRE-EXISTING width-1
    carve-out (unrelated to WP2) accepts it under legacy and rejects it under
    strict -- exactly as for any other COLLAPSED operand, daily-leaf-derived or
    not."""
    legacy = parse_legacy("day_sum(log(FloatCap))")
    assert legacy.shape == COLLAPSED
    assert repr(legacy) == "day_sum(log(FloatCap))"
    with pytest.raises(ParseError):
        parse("day_sum(log(FloatCap))")


def test_day_sum_of_div_floatcap_constant_raises_in_both_modes():
    """div(FloatCap, 2.0) is WIDE_DEGENERATE (collapsed=False), not COLLAPSED --
    measured, matching the design's correction of must_fix #8 -- so the width-1
    carve-out's `sh.collapsed` test excludes it regardless of strict."""
    assert parse_legacy("div(FloatCap, 2.0)").shape == WIDE_DEGENERATE
    with pytest.raises(ParseError):
        parse("day_sum(div(FloatCap, 2.0))")
    with pytest.raises(ParseError):
        parse_legacy("day_sum(div(FloatCap, 2.0))")


# ---------------------------------------------------------------------------
# is_valid_root: a bare daily leaf (or a tree with no Feature at all) is a root
# ---------------------------------------------------------------------------

def test_is_valid_root_accepts_a_pure_daily_tree():
    assert is_valid_root(parse("cs_rank(FloatCap)"))
    assert is_valid_root(DailyFeature("FloatCap"))


def test_is_valid_root_still_rejects_the_constant_only_tree():
    """Unaffected pre-existing behavior (design backward_compat #3): adding
    DailyFeature to the isinstance tuple can only flip False -> True, and only
    for a tree that actually contains one."""
    from alphacube.expression import Constant
    assert not is_valid_root(Constant(5.0))


# ---------------------------------------------------------------------------
# Parsing: aliases, precedence, coerce_daily non-rewrite, unknown-field message
# ---------------------------------------------------------------------------

def test_minute_field_precedence_is_unaffected():
    assert repr(parse("close")) == "Close"
    assert repr(parse("close_1d")) == "day_last(Close)"


@pytest.mark.parametrize("alias, canonical", [
    ("float_cap", "FloatCap"), ("floatcap", "FloatCap"), ("circ_mv", "FloatCap"),
    ("total_cap", "TotalCap"), ("totalcap", "TotalCap"), ("total_mv", "TotalCap"),
    ("float_shares", "FloatShares"), ("floatshares", "FloatShares"),
    ("limit_width", "LimitWidth"), ("limitwidth", "LimitWidth"),
    ("board", "BoardCode"), ("boardcode", "BoardCode"),
    ("list_age", "ListAge"), ("listage", "ListAge"),
    ("log_float_cap", "LogFloatCap"), ("logfloatcap", "LogFloatCap"),
    ("log_total_cap", "LogTotalCap"), ("logtotalcap", "LogTotalCap"),
    ("log_float_shares", "LogFloatShares"), ("logfloatshares", "LogFloatShares"),
])
def test_daily_field_aliases_resolve_to_the_canonical_name(alias, canonical):
    e = parse(alias)
    assert isinstance(e, DailyFeature)
    assert e.field == canonical
    assert repr(e) == canonical


def test_no_bare_age_alias():
    """Design step 8: deliberately no bare 'age' alias for ListAge."""
    assert "age" not in DAILY_FIELD_ALIASES
    with pytest.raises(ParseError):
        parse("age")


def test_coerce_daily_does_not_rewrite_a_bare_daily_leaf():
    """coerce_bare_fields (the renamed local flag) governs ONLY bare minute
    fields; the daily-leaf branch never consults it (design step 9)."""
    legacy = parse_legacy("cs_rank(FloatCap)")
    strict = parse("cs_rank(FloatCap)")
    assert repr(legacy) == repr(strict) == "cs_rank(FloatCap)"
    assert legacy.key() == strict.key() == "O:cs_rank(DF:FloatCap)"


def test_unknown_field_error_names_both_namespaces():
    with pytest.raises(ParseError) as excinfo:
        parse("Nonsense")
    msg = str(excinfo.value)
    assert "cube fields are" in msg and "daily fields are" in msg
    for f in CUBE_FIELDS:
        assert f in msg
    for f in DAILY_FIELDS:
        assert f in msg


def test_daily_field_alias_outside_a_restricted_registry_is_rejected():
    """Registry.daily_fields can legitimately be a strict subset of
    catalog.DAILY_FIELDS (tests/test_catalog.py::test_build_registry_daily_
    fields_override already exercises this on the Registry side). The parser
    must re-check an ALIAS-resolved name against THIS registry's own
    daily_fields, not just against DAILY_FIELD_ALIASES' value set -- otherwise
    a restricted Registry would still let an alias leak in a name it does not
    carry."""
    reg = build_registry(daily_fields=("FloatCap",))
    assert repr(parse("float_cap", reg)) == "FloatCap"
    assert repr(parse("FloatCap", reg)) == "FloatCap"
    with pytest.raises(ParseError):
        parse("total_cap", reg)                 # alias resolves to TotalCap, not in this registry
    with pytest.raises(ParseError):
        parse("TotalCap", reg)


def test_disjointness_of_daily_field_alias_table():
    """Extends tests/test_real_valued_flag.py::test_daily_fields_disjoint_
    from_existing_namespaces (WP1, a file WP2 does not own) with the one table
    WP2 adds: DAILY_FIELD_ALIASES. Self-contained here rather than depending on
    that other file's assertions."""
    reg = build_registry()
    names = set(DAILY_FIELDS)
    alias_keys, alias_vals = set(DAILY_FIELD_ALIASES), set(DAILY_FIELD_ALIASES.values())
    pools = (
        set(CUBE_FIELDS),
        {s.name for s in reg.list_all()},
        set(FIELD_ALIASES), set(FIELD_ALIASES.values()),
        set(DAILY_MACROS),
        set(DAILY_NAME_ALIASES), set(DAILY_NAME_ALIASES.values()),
        set(OPERATOR_NAME_ALIASES), set(OPERATOR_NAME_ALIASES.values()),
        set(OPERATOR_ARITY_ALIASES),
        alias_keys,
    )
    for pool in pools:
        assert not (names & pool), names & pool
    # Every alias VALUE must actually be one of the 9 canonical names, and every
    # alias KEY must not already mean something else in any existing table.
    assert alias_vals <= names
    for pool in pools[:-1]:
        assert not (alias_keys & pool), alias_keys & pool
    full_catalog = default_full_catalog()
    assert all(full_catalog.get(n) is None for n in names | alias_keys)


# ---------------------------------------------------------------------------
# dailyfields.derive(): hand-computed arrays
# ---------------------------------------------------------------------------

# A symbol set where every one of STAR / GEM / BSE fires at least once, in the
# "prefix-first" spelling board_limit actually keys off (tradability.py).
_SYMBOLS = ["600000.SH", "688001.SH", "300001.SZ", "000001.SZ", "831010.BJ"]
_DATES = ["20200101", "20200824", "20200825", "20210101"]       # straddles the GEM 2020-08-24 switch


def test_derive_limit_width_matches_board_limit_transposed():
    from alphacube.tradability import board_limit
    lw = df.derive("LimitWidth", _SYMBOLS, _DATES)
    assert lw.shape == (len(_SYMBOLS), len(_DATES))
    assert lw.dtype == np.float64
    assert lw.flags["C_CONTIGUOUS"]              # transpose-then-contiguous, not the other order
    np.testing.assert_array_equal(lw, board_limit(_SYMBOLS, _DATES).T)
    # hand-computed: main board 10%, STAR 20%, GEM 10%->20% after 20200824, BSE 30%
    np.testing.assert_array_equal(lw[0], [0.10, 0.10, 0.10, 0.10])     # 600000.SH, main
    np.testing.assert_array_equal(lw[1], [0.20, 0.20, 0.20, 0.20])     # 688001.SH, STAR
    np.testing.assert_array_equal(lw[2], [0.10, 0.20, 0.20, 0.20])     # 300001.SZ, GEM switch
    np.testing.assert_array_equal(lw[3], [0.10, 0.10, 0.10, 0.10])     # 000001.SZ, main
    np.testing.assert_array_equal(lw[4], [0.30, 0.30, 0.30, 0.30])     # 831010.BJ, BSE


def test_derive_board_code_matches_the_same_prefix_rule():
    bc = df.derive("BoardCode", _SYMBOLS, _DATES)
    assert bc.shape == (len(_SYMBOLS), len(_DATES))
    expect_row = [df.BOARD_MAIN, df.BOARD_STAR, df.BOARD_GEM, df.BOARD_MAIN, df.BOARD_BSE]
    np.testing.assert_array_equal(bc[:, 0], expect_row)
    # BoardCode is a date-INDEPENDENT board classification (board membership does
    # not change on 2020-08-24, only GEM's limit percentage does), unlike LimitWidth.
    for d in range(len(_DATES)):
        np.testing.assert_array_equal(bc[:, d], expect_row)
    assert len({df.BOARD_MAIN, df.BOARD_STAR, df.BOARD_GEM, df.BOARD_BSE}) == 4


@pytest.mark.parametrize("field", ["LimitWidth", "BoardCode"])
def test_derive_raises_on_the_old_prefix_free_symbol_spelling(field):
    """must_fix #2's FORMAT guard, the half that stays a hard raise per call.
    DEVIATION (documented, not a file-ownership change): the design's own text
    states the MUST-raise requirement for LimitWidth; this module applies the
    identical guard to BoardCode too, since an undetected all-MAIN BoardCode is
    exactly the other half of the same silent failure the design's
    Overview/Risks sections call out."""
    bad_symbols = ["SH.600000", "SH.600001", "SH.600002"]
    with pytest.raises(ValueError, match="STAR.*GEM.*BSE"):
        df.derive(field, bad_symbols, _DATES)


def test_require_board_rule_error_names_the_first_five_symbols():
    bad_symbols = [f"SH.{600000+i}" for i in range(7)]
    with pytest.raises(ValueError) as excinfo:
        df.derive("LimitWidth", bad_symbols, _DATES)
    msg = str(excinfo.value)
    for s in bad_symbols[:5]:
        assert repr(s) in msg
    assert repr(bad_symbols[6]) not in msg          # only the first five are named
    assert "7 symbols" in msg


# --- must_fix #2, the format/composition split (regression) ------------------
# A board-COMPOSITION check cannot run per call: store.view(max_symbols=k)
# takes sym_idx[:k] (store.py:215) and warmed_view(symbols=[...]) takes the
# caller's own list (research.py:84), so a legitimately main-board-only symbol
# list is a normal input. Raising there would kill every factor touching
# LimitWidth / BoardCode on such a view once WP3's CubeView.daily() calls
# resolve() with self.symbols. Per call we check FORMAT (raise) and warn about
# composition; the strict composition assertion is require_board_rule(), which
# a caller holding the FULL universe runs once.

# WP3 step 6's planned synthetic spelling: the special boards are necessarily a
# minority, so any short prefix of this list is main-board-only.
_WP3_SYMBOLS = [f"{600000 + i}.SH" for i in range(12)] + ["688001.SH", "300001.SZ", "830001.BJ"]
# a hand-picked real-world main-board-only selection (warmed_view(symbols=[...]))
_MAIN_ONLY = ["600000.SH", "600036.SH", "601398.SH", "000001.SZ"]


@pytest.mark.parametrize("field", ["LimitWidth", "BoardCode"])
@pytest.mark.parametrize("symbols", [_WP3_SYMBOLS[:4], _WP3_SYMBOLS[:8], _WP3_SYMBOLS[:12], _MAIN_ONLY],
                         ids=["max_symbols=4", "max_symbols=8", "max_symbols=12", "hand-picked"])
def test_derive_accepts_a_main_board_only_symbol_subset(field, symbols):
    """The regression itself: a correctly spelled main-board-only subset must
    EVALUATE, not raise -- and give the main-board answer."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # the composition warning below
        got = df.derive(field, symbols, _DATES)
    assert got.shape == (len(symbols), len(_DATES))
    expect = 0.10 if field == "LimitWidth" else df.BOARD_MAIN
    np.testing.assert_array_equal(got, np.full(got.shape, expect))
    # and the same through resolve(), which is what CubeView.daily() will call
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert df.resolve(field, symbols, _DATES).shape == (len(symbols), len(_DATES))


@pytest.mark.parametrize("field", ["LimitWidth", "BoardCode"])
def test_derive_warns_rather_than_raises_on_a_main_board_only_subset(field):
    """Degrade loudly, not fatally: the composition case is a warning per call."""
    with pytest.warns(RuntimeWarning, match="no STAR.*GEM.*BSE"):
        df.derive(field, _MAIN_ONLY, _DATES)
    # a subset that DOES contain a special board stays silent
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df.derive(field, _SYMBOLS, _DATES)


def test_require_board_rule_is_the_strict_full_universe_check():
    """The composition assertion keeps its teeth where the full symbol list is
    in hand (WP3's daily-source attach, WP9's build script)."""
    assert df.require_board_rule(_WP3_SYMBOLS) is None       # the full universe is fine
    with pytest.raises(ValueError, match="no STAR.*GEM.*BSE"):
        df.require_board_rule(_MAIN_ONLY)                     # main-board-only universe: impossible
    with pytest.raises(ValueError, match="starts with a digit"):
        df.require_board_rule(["SH.600000", "SH.600001"])     # and the format half still fires


def test_derive_list_age_sentinel_day_zero_and_suspension_monotonicity():
    """Hand-built status planes (store.DAY_STATUS_CODES's own convention:
    0=OUTSIDE_LIFECYCLE, 1=TRADING, 3=SUSPENDED), at the plain-array level WP2
    owns -- the fuller version against a real synthetic store (unlisted_days=8)
    is WP3's tests/test_daily_fields_store.py."""
    from alphacube.store import DAY_STATUS_CODES
    T, S, O = DAY_STATUS_CODES["TRADING"], DAY_STATUS_CODES["SUSPENDED"], DAY_STATUS_CODES["OUTSIDE_LIFECYCLE"]
    symbols = ["A", "B", "C"]
    dates = ["20200101", "20200102", "20200103", "20200104"]
    status = np.array([
        [T, T, T, T],            # already listed on day 0 -> sentinel throughout
        [O, O, T, T],            # listed day 2 (index 2) -> 0, 1 there
        [T, S, S, T],            # listed day 0, suspended days 1-2 -> sticky, no reset
    ], dtype=np.int8)
    age = df.derive("ListAge", symbols, dates, status=status)
    np.testing.assert_array_equal(age[0], [10000.0, 10001.0, 10002.0, 10003.0])
    assert np.isnan(age[1, 0]) and np.isnan(age[1, 1])
    np.testing.assert_array_equal(age[1, 2:], [0.0, 1.0])
    np.testing.assert_array_equal(age[2], [10000.0, 10001.0, 10002.0, 10003.0])  # suspension does not reset it


def test_derive_list_age_with_no_signal_defaults_to_in_lifecycle_everywhere():
    symbols, dates = ["A", "B"], ["20200101", "20200102", "20200103"]
    age = df.derive("ListAge", symbols, dates)
    np.testing.assert_array_equal(age, [[10000.0, 10001.0, 10002.0]] * 2)


def test_derive_list_age_prefers_status_over_tradable():
    """status is the richer signal and must win when BOTH are given.

    The pair below DISAGREES, which is the whole point: with agreeing inputs
    the assertion holds under either priority, so it pins nothing (verified by
    mutation -- swapping _lifecycle_mask to consult tradable first left the
    agreeing-inputs version of this test green). WP3's CubeView.daily() is
    exactly the caller that has both view.tradable() and view.day_status() in
    hand, so the priority has to be nailed down here."""
    from alphacube.store import DAY_STATUS_CODES
    T, S, O = DAY_STATUS_CODES["TRADING"], DAY_STATUS_CODES["SUSPENDED"], DAY_STATUS_CODES["OUTSIDE_LIFECYCLE"]
    symbols, dates = ["A"], ["20200101", "20200102", "20200103"]
    status = np.array([[O, T, T]], dtype=np.int8)       # not listed on day 0
    tradable = np.array([[True, True, True]])           # DISAGREES: claims day 0 is live
    both = df.derive("ListAge", symbols, dates, status=status, tradable=tradable)
    # status wins: day 0 is pre-listing, so NaN there and the count starts at 0
    # on day 1. The inverted priority would read tradable's "listed on day 0"
    # and return [10000, 10001, 10002] instead -- a listing date off by one day
    # and a sentinel where there should be none.
    assert np.isnan(both[0, 0])
    np.testing.assert_array_equal(both[0, 1:], [0.0, 1.0])
    # each signal alone still drives its own branch, and they differ here
    np.testing.assert_array_equal(df.derive("ListAge", symbols, dates, status=status)[0, 1:],
                                  [0.0, 1.0])
    np.testing.assert_array_equal(df.derive("ListAge", symbols, dates, tradable=tradable)[0],
                                  [10000.0, 10001.0, 10002.0])
    # and with inputs that AGREE (a SUSPENDED day is in lifecycle but not
    # tradable, so these two genuinely agree only on the sentinel row) the two
    # paths coincide, as before
    agree_status = np.array([[T, S, T]], dtype=np.int8)
    agree_tradable = np.array([[True, False, True]])
    np.testing.assert_array_equal(df.derive("ListAge", symbols, dates, status=agree_status),
                                  df.derive("ListAge", symbols, dates, tradable=agree_tradable))


def test_derive_unknown_field_raises():
    with pytest.raises(ValueError, match="not a derived field"):
        df.derive("Nonsense", _SYMBOLS, _DATES)


# ---------------------------------------------------------------------------
# dailyfields module tuples: must agree with catalog.py's independently-
# defined copy (design gives each module its OWN literal tuples; nothing
# machine-checks they still mean the same 9 names except this test).
# ---------------------------------------------------------------------------

def test_dailyfields_tuples_agree_with_catalog_py():
    assert df.STORED_FIELDS == STORED_DAILY_FIELDS
    assert df.DERIVED_FIELDS == DERIVED_DAILY_FIELDS
    assert tuple(df.LOG_OF) == LOG_DAILY_FIELDS
    assert df.STORED_FIELDS + df.DERIVED_FIELDS + tuple(df.LOG_OF) == DAILY_FIELDS
    assert set(df.LOG_OF.values()) == set(STORED_DAILY_FIELDS)


# ---------------------------------------------------------------------------
# DailyFieldStore / resolve / available / fingerprint -- no store.py import,
# so these are built from a tiny hand-written directory, exactly what the
# module docstring claims is possible.
# ---------------------------------------------------------------------------

@pytest.fixture
def daily_store(tmp_path):
    import json

    symbols = ["600000.SH", "688001.SH", "300001.SZ"]
    dates = ["20200101", "20200102", "20200103", "20200104"]
    S, D = len(symbols), len(dates)
    planes = {}
    for i, name in enumerate(("FloatCap", "TotalCap")):
        arr = (1e9 * (np.arange(S)[:, None] + 1) + np.arange(D)[None, :] + i * 1e12).astype(np.float64)
        arr[1, 0] = np.nan                               # one missing cell
        np.save(tmp_path / f"daily_{name}.npy", arr)
        planes[name] = arr
    (tmp_path / "meta.json").write_text(
        json.dumps({"fields": ["FloatCap", "TotalCap"], "symbols": symbols, "dates": dates}), encoding="utf-8")
    return tmp_path, symbols, dates, planes


def test_daily_field_store_reads_back_what_was_written(daily_store):
    root, symbols, dates, planes = daily_store
    store = df.DailyFieldStore(root)
    assert store.fields == ("FloatCap", "TotalCap")
    assert store.symbols == symbols and store.dates == dates
    np.testing.assert_array_equal(np.asarray(store.plane("FloatCap")), planes["FloatCap"])
    with pytest.raises(KeyError):
        store.plane("Nonsense")


def test_daily_field_store_align_by_label(daily_store):
    root, symbols, dates, planes = daily_store
    store = df.DailyFieldStore(root)

    # superset: an unknown symbol on either side comes back all-NaN
    superset = ["ZZZZZZ"] + symbols + ["AAAAAA"]
    out = store.align("FloatCap", superset, dates)
    assert out.shape == (len(superset), len(dates))
    assert np.all(np.isnan(out[0])) and np.all(np.isnan(out[-1]))
    np.testing.assert_array_equal(out[1:-1], planes["FloatCap"])

    # permutation (non-ascending): rows must follow the CALLER's order, not the store's
    perm = [symbols[2], symbols[0]]
    out2 = store.align("FloatCap", perm, dates)
    np.testing.assert_array_equal(out2[0], planes["FloatCap"][2])
    np.testing.assert_array_equal(out2[1], planes["FloatCap"][0])

    # a missing date: NaN exactly in that column
    out3 = store.align("FloatCap", symbols, [dates[0], "20209999"])
    np.testing.assert_array_equal(out3[:, 0], planes["FloatCap"][:, 0])
    assert np.all(np.isnan(out3[:, 1]))


def test_resolve_stored_field_needs_a_store(daily_store):
    root, symbols, dates, planes = daily_store
    store = df.DailyFieldStore(root)
    got = df.resolve("FloatCap", symbols, dates, store=store)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, planes["FloatCap"].astype(np.float32), equal_nan=True)
    with pytest.raises(ValueError, match="daily store"):
        df.resolve("FloatCap", symbols, dates, store=None)
    with pytest.raises(ValueError, match="daily store"):
        df.resolve("FloatShares", symbols, dates, store=store)      # store has no FloatShares


def test_resolve_log_field_logs_the_float64_plane_before_casting(daily_store):
    root, symbols, dates, planes = daily_store
    store = df.DailyFieldStore(root)
    got = df.resolve("LogFloatCap", symbols, dates, store=store)
    assert got.dtype == np.float32
    expect = np.log(planes["FloatCap"]).astype(np.float32)
    np.testing.assert_allclose(got, expect, equal_nan=True)
    with pytest.raises(ValueError):
        df.resolve("LogFloatShares", symbols, dates, store=store)   # no FloatShares to log


def test_resolve_derived_field_needs_no_store():
    got = df.resolve("LimitWidth", _SYMBOLS, _DATES, store=None)
    assert got.dtype == np.float32
    assert got.shape == (len(_SYMBOLS), len(_DATES))


def test_resolve_unknown_field_names_available(daily_store):
    root, symbols, dates, _ = daily_store
    store = df.DailyFieldStore(root)
    with pytest.raises(ValueError, match="FloatCap"):
        df.resolve("Nonsense", symbols, dates, store=store)


def test_available_lists_derived_always_and_stored_or_log_only_with_a_store(daily_store):
    root, symbols, dates, _ = daily_store
    store = df.DailyFieldStore(root)
    assert df.available(None) == DERIVED_DAILY_FIELDS
    avail = df.available(store)
    assert set(avail) == {"FloatCap", "TotalCap", "LogFloatCap", "LogTotalCap", *DERIVED_DAILY_FIELDS}
    assert "FloatShares" not in avail and "LogFloatShares" not in avail


def test_fingerprint_none_without_a_store_and_stable_with_one(daily_store):
    root, symbols, dates, _ = daily_store
    assert df.fingerprint(None) is None
    store_a, store_b = df.DailyFieldStore(root), df.DailyFieldStore(root)
    fp_a, fp_b = df.fingerprint(store_a), df.fingerprint(store_b)
    assert fp_a == fp_b
    assert fp_a.startswith(f"{df.DERIVED_VERSION}:")


def test_warn_once_message_fires_only_when_the_field_starts_late():
    dates = ["20200101", "20200102", "20200103"]
    late = np.array([[np.nan, 1.0, 1.0]])
    with pytest.warns(RuntimeWarning, match="no finite value before"):
        df._warn_once_if_late_start("Field", late, dates)

    on_time = np.array([[1.0, 1.0, 1.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")                   # any warning here fails the test
        df._warn_once_if_late_start("Field", on_time, dates)


# --- the all-NaN / near-empty plane (regression) -----------------------------
# The late-start check above provably CANNOT fire on a 100% NaN plane: with no
# finite cell there is no first finite date to compare against dates[0]. That
# is precisely the case the guard was written for -- a daily store whose symbol
# or date spelling does not match the view's, which align() answers with a NaN
# fill by design. Two more checks cover it; see _warn_once_if_late_start.

def test_warn_fires_on_an_all_nan_plane():
    dates = ["20200101", "20200102", "20200103"]
    with pytest.warns(RuntimeWarning, match="no finite value ANYWHERE"):
        df._warn_once_if_late_start("Field", np.full((4, 3), np.nan), dates)


def test_warn_fires_below_the_coverage_floor_and_not_above_it():
    dates = [f"202001{d:02d}" for d in range(1, 11)]
    thin = np.full((40, 10), np.nan)
    thin[0, 0] = 1.0                                     # 1 of 400 = 0.25%, below the 1% floor
    assert float(np.isfinite(thin).mean()) < df.LOW_COVERAGE_FLOOR
    with pytest.warns(RuntimeWarning, match="finite in only"):
        df._warn_once_if_late_start("Field", thin, dates)

    thick = np.full((40, 10), np.nan)
    thick[:, 0] = 1.0                                    # 40 of 400 = 10%, above the floor
    assert float(np.isfinite(thick).mean()) > df.LOW_COVERAGE_FLOOR
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df._warn_once_if_late_start("Field", thick, dates)


def _write_daily_store(root, symbols, dates, arr, field="FloatCap"):
    import json

    np.save(root / f"daily_{field}.npy", np.asarray(arr, dtype=np.float64))
    (root / "meta.json").write_text(
        json.dumps({"fields": [field], "symbols": list(symbols), "dates": list(dates)}),
        encoding="utf-8")
    return df.DailyFieldStore(root)


@pytest.mark.parametrize("mismatch", ["symbols", "dates"])
def test_resolve_warns_when_a_label_mismatch_nans_every_cell(tmp_path, mismatch):
    """must_fix #2's failure mode applied to the STORED half: the store spells
    its labels one way, the view asks another, every cell comes back NaN. It
    used to return that plane silently."""
    store_syms = ["SH.600000", "SH.600001", "SH.688001"]
    store_dates = ["20200101", "20200102", "20200103"]
    arr = 1e9 * (np.arange(3)[:, None] + 1) + np.arange(3)[None, :]
    store = _write_daily_store(tmp_path, store_syms, store_dates, arr)

    ask_syms = ["600000.SH", "600001.SH", "688001.SH"] if mismatch == "symbols" else store_syms
    ask_dates = store_dates if mismatch == "symbols" else ["2020-01-01", "2020-01-02", "2020-01-03"]
    with pytest.warns(RuntimeWarning, match="no finite value ANYWHERE"):
        plane = df.resolve("FloatCap", ask_syms, ask_dates, store=store)
    assert not np.isfinite(plane).any()                   # the plane itself is unchanged


def test_resolve_warns_when_the_store_covers_almost_none_of_the_view(tmp_path):
    """The partial mismatch one finite cell is enough to hide from the
    late-start check: a store overlapping the view on 1 of 200 symbols."""
    dates = ["20200101", "20200102", "20200103"]
    store = _write_daily_store(tmp_path, ["600000.SH"], dates, np.full((1, 3), 1e9))
    view_syms = ["600000.SH"] + [f"{600001 + i}.SH" for i in range(199)]
    with pytest.warns(RuntimeWarning, match="finite in only"):
        plane = df.resolve("FloatCap", view_syms, dates, store=store)
    assert np.isfinite(plane).sum() == 3


def test_resolve_warns_when_a_stored_field_starts_after_the_requested_range(daily_store):
    root, symbols, dates, _ = daily_store
    store = df.DailyFieldStore(root)
    # the fixture's FloatCap is NaN at [symbol 1, date 0] only -- not a field-wide
    # late start (some OTHER symbol is finite on date 0) -- so no warning here;
    # a genuinely late-starting field is covered by test_warn_once_message_fires_
    # only_when_the_field_starts_late at the lower level.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df.resolve("FloatCap", symbols, dates, store=store)
