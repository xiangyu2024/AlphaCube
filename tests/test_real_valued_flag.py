"""WP1 -- the real_valued declaration (operators/common.op + catalog._make_spec)
and the daily-field name tuples it ships alongside. See
docs/alphacube_steering_design.md, section "5. The boolean self-check" and
work package "### WP1".
"""

import pytest

from alphacube.catalog import (
    DAILY_FIELDS,
    DERIVED_DAILY_FIELDS,
    LOG_DAILY_FIELDS,
    REAL_VALUED_OVERRIDES,
    STORED_DAILY_FIELDS,
    build_registry,
)
from alphacube.operators.common import OPERATOR_REGISTRY, op

# Measured on the current tree (design doc §5, "must_fix #5"): exactly these 29
# names come back `boolean_output`, and exactly these 22 of them also
# `trivial_boolean` (0/1 output that some plausible input collapses to a single
# constant). Nothing in the suite pinned either set before WP1, which is why
# both mis-flag families below were silent.
BOOLEAN_OUTPUT_NAMES = frozenset({
    "cross_outlier_33", "cross_sustain", "cs_quantile_q25", "cs_quantile_q50",
    "cs_quantile_q75", "eq", "greater", "in_is_neighbor",
    "in_left_one_distance_or_idx", "in_left_one_distance_zero", "in_no_neighbor",
    "in_sustain", "in_sustain_mean", "in_sustain_product", "in_sustain_stddev",
    "in_top_quantile_flag_q10", "is_gt0", "is_lt0", "less", "logical_and",
    "logical_not", "logical_or", "logical_xor", "nt_cs_quantile_q25",
    "nt_cs_quantile_q50", "nt_cs_quantile_q75", "nt_ts_outlier_33", "sign",
    "ts_outlier_33",
})

TRIVIAL_BOOLEAN_NAMES = frozenset({
    "cross_outlier_33", "cross_sustain", "eq", "greater", "in_is_neighbor",
    "in_left_one_distance_or_idx", "in_left_one_distance_zero", "in_no_neighbor",
    "in_sustain", "in_sustain_mean", "in_sustain_product", "in_sustain_stddev",
    "is_gt0", "is_lt0", "less", "logical_and", "logical_not", "logical_or",
    "logical_xor", "nt_ts_outlier_33", "sign", "ts_outlier_33",
})

# in_sustain_mean / in_sustain_stddev / in_sustain_product are MIS-flagged: each
# averages/stddevs/products a 0/1 mask over a window, which is genuinely
# real-valued, unlike in_sustain itself (a true 0/1 mask, correctly flagged).
# Declaring the three with real_valued=True would move
# searchable(include_boolean=False) 218 -> 221 and the generator's default
# vocabulary (also include_resource_risk=False) 214 -> 217 -- see
# test_searchable_counts_pinned. WP1 ships the real_valued mechanism and
# `_make_spec` applies it to NOTHING by default (REAL_VALUED_OVERRIDES is
# empty and no @op(...) call site passes it); that declaration is a plugin's
# decision (the Partura plugin, WP11), not AlphaCube's default.
MISFLAGGED_IN_SUSTAIN = frozenset({"in_sustain_mean", "in_sustain_stddev", "in_sustain_product"})


def test_boolean_and_trivial_sets_pinned(registry):
    boolean = {s.name for s in registry.list_all() if s.boolean_output}
    trivial = {s.name for s in registry.list_all() if s.trivial_boolean}
    coarse = {s.name for s in registry.list_all() if s.coarse_output}
    assert boolean == BOOLEAN_OUTPUT_NAMES
    assert trivial == TRIVIAL_BOOLEAN_NAMES
    assert coarse == set()
    # in_sustain is a true 0/1 mask and must stay flagged; its three statistics
    # are the ones a future real_valued declaration is meant to clear.
    assert "in_sustain" in boolean and "in_sustain" in trivial
    assert MISFLAGGED_IN_SUSTAIN <= boolean and MISFLAGGED_IN_SUSTAIN <= trivial


def test_searchable_counts_pinned(registry):
    # WP1 applies real_valued to nothing, so today's counts must not move.
    assert len(registry.searchable(include_boolean=False)) == 218
    assert len(registry.searchable(include_boolean=False, include_resource_risk=False)) == 214
    # (declaring MISFLAGGED_IN_SUSTAIN real_valued would give 221 / 217 -- a
    # plugin's choice, exercised by the synthetic-kernel tests below instead of
    # on these three real kernels.)


def test_two_independent_builds_give_identical_flags():
    """The probes behind boolean_output/trivial_boolean/coarse_output must stay
    deterministic -- each uses its own fixed-seed torch.Generator() (design
    doc §5), never a shared RNG, so two builds cannot disagree."""
    a, b = build_registry(), build_registry()
    flagged = lambda r, attr: {s.name for s in r.list_all() if getattr(s, attr)}  # noqa: E731
    for attr in ("boolean_output", "trivial_boolean", "coarse_output"):
        assert flagged(a, attr) == flagged(b, attr), attr


@pytest.fixture
def fake_constant_capable_kernel():
    """Registers a throwaway unary kernel -- ``(x > 0).float()`` -- directly
    into OPERATOR_REGISTRY, exactly as a real ``@op(...)``-decorated function
    would. On the random-normal input ``_probe_distinct_values`` uses it is
    0/1 (boolean_output); on the all-positive / all-negative families
    ``_probe_constant_capable`` uses it collapses to a single constant
    (trivial_boolean). So without ``real_valued=True`` it is flagged both
    ways; with it, neither.

    Yields a ``register(real_valued) -> name`` callable so a test can flip the
    flag and rebuild. Removed from OPERATOR_REGISTRY on teardown (even if the
    test raises) so it can never leak into another test's build_registry() --
    in particular the session-scoped ``registry`` fixture built once from
    default_registry().
    """
    name = "_wp1_test_fake_constant_capable"

    def register(real_valued: bool) -> str:
        def fn(x):
            return (x > 0).float()
        fn.__name__ = name
        op("shared", "elementwise", arity=1, real_valued=real_valued)(fn)
        return name

    try:
        yield register
    finally:
        OPERATOR_REGISTRY.pop(name, None)
        REAL_VALUED_OVERRIDES.discard(name)


def test_real_valued_flag_skips_the_probes(fake_constant_capable_kernel):
    name = fake_constant_capable_kernel(real_valued=False)
    flagged = build_registry().get(name)
    assert flagged.boolean_output and flagged.trivial_boolean and not flagged.coarse_output
    assert name not in {s.name for s in build_registry().searchable(include_boolean=False)}

    name = fake_constant_capable_kernel(real_valued=True)       # re-registers the same name
    clean = build_registry().get(name)
    assert not clean.boolean_output and not clean.trivial_boolean and not clean.coarse_output
    assert name in {s.name for s in build_registry().searchable(include_boolean=False)}


def test_real_valued_overrides_by_name(fake_constant_capable_kernel):
    """REAL_VALUED_OVERRIDES reaches the same outcome by NAME, for a kernel the
    caller does not own and so cannot decorate with real_valued=True itself
    (the Partura plugin's use, per the design doc). Clearing the override
    restores today's flags on the next rebuild."""
    name = fake_constant_capable_kernel(real_valued=False)
    assert name not in REAL_VALUED_OVERRIDES

    flagged = build_registry().get(name)
    assert flagged.boolean_output and flagged.trivial_boolean

    REAL_VALUED_OVERRIDES.add(name)
    try:
        overridden = build_registry().get(name)
        assert not overridden.boolean_output and not overridden.trivial_boolean
    finally:
        REAL_VALUED_OVERRIDES.discard(name)

    restored = build_registry().get(name)
    assert restored.boolean_output and restored.trivial_boolean


def test_daily_field_namespace_tuples():
    """catalog.py's own copy of the 9-name daily-field namespace (WP1 ships the
    name set only; WP2 adds the DailyFeature leaf and the data resolver that
    make these names parseable/resolvable)."""
    assert STORED_DAILY_FIELDS == ("FloatCap", "TotalCap", "FloatShares")
    assert DERIVED_DAILY_FIELDS == ("LimitWidth", "BoardCode", "ListAge")
    assert LOG_DAILY_FIELDS == ("LogFloatCap", "LogTotalCap", "LogFloatShares")
    assert DAILY_FIELDS == STORED_DAILY_FIELDS + DERIVED_DAILY_FIELDS + LOG_DAILY_FIELDS
    assert len(DAILY_FIELDS) == 9
    assert len(set(DAILY_FIELDS)) == 9                      # no internal duplicates


def test_daily_fields_disjoint_from_existing_namespaces(registry):
    """Not one of WP1's own steps (no parser branch exists yet to make these
    names reachable), but a precondition WP2 relies on and is cheap to check
    now against everything that already exists: the 9 daily names must not
    collide with the minute fields, any registered operator name (244 today),
    or any alias table parser.py / fullcatalog.py already define."""
    from alphacube.fullcatalog import OPERATOR_ARITY_ALIASES, OPERATOR_NAME_ALIASES, default_full_catalog
    from alphacube.parser import DAILY_MACROS, DAILY_NAME_ALIASES, FIELD_ALIASES

    names = set(DAILY_FIELDS)
    pools = (
        set(registry.fields),
        {s.name for s in registry.list_all()},
        set(FIELD_ALIASES), set(FIELD_ALIASES.values()),
        set(DAILY_MACROS),
        set(DAILY_NAME_ALIASES), set(DAILY_NAME_ALIASES.values()),
        set(OPERATOR_NAME_ALIASES), set(OPERATOR_NAME_ALIASES.values()),
        set(OPERATOR_ARITY_ALIASES),
    )
    for pool in pools:
        assert not (names & pool), names & pool
    full_catalog = default_full_catalog()
    assert all(full_catalog.get(n) is None for n in names)
