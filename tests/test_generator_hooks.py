"""WP5 -- the generator's steering hooks: operator weights (``op_weight``), the
accept hook (``accept`` / ``accept_hook`` / ``hook_rng``), the two mutation-side
ban/hook bypasses (``mutate_window`` / ``mutate_constant``, and their
``mutation_checks_bans`` escape hatch), ``generate()``'s fallback (now routed
through ``acceptable()``, with ``on_exhausted`` and ``GenerationExhausted``), and
daily leaves drawn into the search (``daily_fields`` / ``daily_leaf_prob`` /
``leaf_slots``). See docs/alphacube_steering_design.md, section "2. Generator"
under "乙 Steering hooks", and work package "### WP5".

DEVIATION from the design's own test-plan text: its ``leaf_slots`` worked example,
``leaf_slots={"cs_bucket_rank": {1: ("FloatCap",)}}``, names an operator that does
not exist in AlphaCube's own registry -- ``cs_bucket_rank`` is a Partura-plugin-only
ND kernel (``partura/ops_ext.py``'s ``NEW_OPERATORS``; "AlphaCube never imports
Partura", design Overview). ``tests/test_daily_leaf_typing.py`` hit the identical
gap for WP2 and documented the same substitute: ``cs_regres``, the only other
ND-face, arity-2, non-boolean operator AlphaCube ships. It exercises exactly the
same mechanism -- an nd-face binary whose operand 1 becomes a bare ``DailyFeature``
instead of a recursed subtree.
"""

import random

import pytest

from alphacube.expression import DailyFeature, is_valid_root
from alphacube.mining.generator import GenerationExhausted, TypedTreeGenerator, one_layer_keys
from alphacube.parser import parse

# ---------------------------------------------------------------------------
# Stream identity: the core backward-compat guarantee, pinned two ways.
# ---------------------------------------------------------------------------


def test_generate_population_identical_with_all_defaults_made_explicit(registry):
    """(i) Same seed, one plain generator and one with every WP5 keyword passed
    explicitly at its default: the populations must be equal key for key."""
    plain = TypedTreeGenerator(registry, n_minutes=24, seed=11, max_depth=4)
    explicit = TypedTreeGenerator(
        registry, n_minutes=24, seed=11, max_depth=4,
        op_weight=None, accept=None, daily_fields=(), daily_leaf_prob=0.0,
        leaf_slots=None, on_exhausted="fallback", mutation_checks_bans=True,
    )
    assert [e.key() for e in plain.generate_population(40)] == \
        [e.key() for e in explicit.generate_population(40)]


class _RecordingRandom(random.Random):
    """Logs the name of every RNG method called, so two code paths can be compared
    on which primitives they drew and in what order -- not merely on whether their
    outputs happened to agree (two different draw sequences can coincide)."""

    def __init__(self, seed):
        super().__init__(seed)
        self.calls = []

    def random(self):
        self.calls.append("random")
        return super().random()

    def choice(self, seq):
        self.calls.append("choice")
        return super().choice(seq)

    def choices(self, population, weights=None, *, cum_weights=None, k=1):
        self.calls.append("choices")
        return super().choices(population, weights=weights, cum_weights=cum_weights, k=k)

    def randint(self, a, b):
        self.calls.append("randint")
        return super().randint(a, b)


def _legacy_pick(rng, specs):
    """A frozen copy of ``TypedTreeGenerator._pick`` exactly as it read before
    WP5 (generator.py:274, the uniform branch): two ``choice`` calls, in this
    order, with no other draw in between."""
    faces = {}
    for s in specs:
        faces.setdefault(s.face, []).append(s)
    return rng.choice(faces[rng.choice(sorted(faces))])


def test_pick_consumes_the_rng_identically_to_the_pre_wp5_version(registry):
    """(ii) An RNG-call-sequence recorder, compared against a local copy of the
    pre-change ``_pick``: proves the stream was CONSUMED identically, not just
    that the chosen operator happened to match."""
    specs = TypedTreeGenerator(registry, n_minutes=24).specs

    r_new = _RecordingRandom(123)
    new_pick = TypedTreeGenerator(registry, n_minutes=24, rng=r_new)._pick(specs)

    r_old = _RecordingRandom(123)
    old_pick = _legacy_pick(r_old, specs)

    assert r_new.calls == r_old.calls
    assert new_pick is old_pick


# ---------------------------------------------------------------------------
# op_weight
# ---------------------------------------------------------------------------


def test_op_weight_zero_for_a_whole_face_excludes_it(registry):
    gen = TypedTreeGenerator(registry, n_minutes=24, seed=5,
                              op_weight=lambda s: 0.0 if s.face == "nd" else 1.0)
    draws = [gen._pick(gen.specs) for _ in range(300)]
    assert draws and not any(s.face == "nd" for s in draws)


def test_op_weight_100x_on_one_operator_raises_its_frequency(registry):
    baseline = TypedTreeGenerator(registry, n_minutes=24, seed=6)
    base_freq = sum(baseline._pick(baseline.specs).name == "day_mean" for _ in range(500)) / 500

    weighted = TypedTreeGenerator(registry, n_minutes=24, seed=6,
                                   op_weight=lambda s: 100.0 if s.name == "day_mean" else 1.0)
    weighted_freq = sum(weighted._pick(weighted.specs).name == "day_mean" for _ in range(500)) / 500

    assert weighted_freq > max(base_freq * 10, 0.05), (base_freq, weighted_freq)


def test_op_weight_accepts_a_plain_mapping_too(registry):
    """``op_weight`` need not be a callable: a ``{name: weight}`` mapping is
    validated against the registry and wrapped into one internally (see
    ``_resolve_op_weight``). An operator the mapping does not mention keeps the
    uniform weight 1.0."""
    gen = TypedTreeGenerator(registry, n_minutes=24, seed=6, op_weight={"day_mean": 100.0})
    freq = sum(gen._pick(gen.specs).name == "day_mean" for _ in range(500)) / 500
    assert freq > 0.05


def test_op_weight_all_zero_for_one_draw_still_returns_an_operator(registry):
    """Not the construction-time-rejected case (an ``op_weight`` that is zero for
    literally EVERY operator also empties the 26-spec reduce pool ``generate()``'s
    fallback needs, and ``__init__`` raises for that -- see the construction-time
    tests below). This is ``_pick_weighted``'s own robustness when some specific
    draw's operators are all zero-weighted (e.g. ``mutate_operator``'s narrow
    per-node ``alts``, which construction cannot see ahead of time): it must fall
    back to the uniform path rather than crash."""
    gen = TypedTreeGenerator(registry, n_minutes=24, seed=7, op_weight=lambda s: 1.0)
    faces = {"dt": [registry.get("day_mean")], "nd": [registry.get("cs_regres")]}
    gen.op_weight = lambda s: 0.0   # reassigned post-construction; __init__'s check already ran
    picked = [gen._pick_weighted(faces) for _ in range(30)]
    assert all(p.name in ("day_mean", "cs_regres") for p in picked)


def test_op_weight_uniformly_zero_draw_matches_the_plain_uniform_path(registry):
    """When every weight _pick sees in a given call is 0, _pick_weighted's `live`
    list is empty and it falls back to exactly the no-op_weight branch -- so the
    two should consume the same draws and return the same sequence of operators."""
    specs = TypedTreeGenerator(registry, n_minutes=24).specs
    plain = TypedTreeGenerator(registry, n_minutes=24, rng=random.Random(42))
    zeroed = TypedTreeGenerator(registry, n_minutes=24, rng=random.Random(42), op_weight=lambda s: 1.0)
    zeroed.op_weight = lambda s: 0.0

    plain_names = [plain._pick(specs).name for _ in range(200)]
    zero_names = [zeroed._pick(specs).name for _ in range(200)]
    assert plain_names == zero_names


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------


def test_construction_rejects_an_unknown_operator_name_in_op_weight(registry):
    with pytest.raises(ValueError, match="unknown operator"):
        TypedTreeGenerator(registry, op_weight={"not_a_real_operator_xyz": 2.0})


def test_construction_rejects_a_negative_or_nan_weight(registry):
    with pytest.raises(ValueError, match="finite float >= 0"):
        TypedTreeGenerator(registry, op_weight={"day_mean": -1.0})
    with pytest.raises(ValueError, match="finite float >= 0"):
        TypedTreeGenerator(registry, op_weight=lambda s: float("nan"))


def test_construction_rejects_an_op_weight_that_is_neither_callable_nor_a_mapping(registry):
    with pytest.raises(ValueError):
        TypedTreeGenerator(registry, op_weight=42)


def test_construction_rejects_a_two_argument_accept(registry):
    with pytest.raises(ValueError, match="positional parameter"):
        TypedTreeGenerator(registry, accept=lambda tree, u: True)
    # An optional second parameter is fine: accept is always called with exactly
    # one positional argument, so a default-valued extra is harmless.
    TypedTreeGenerator(registry, accept=lambda tree, u=0.5: True)


def test_construction_rejects_a_non_callable_accept(registry):
    with pytest.raises(ValueError, match="callable"):
        TypedTreeGenerator(registry, accept="not callable")


def test_construction_rejects_an_op_weight_that_empties_the_reduce_pool(registry):
    pool_names = {s.name for s in TypedTreeGenerator(registry)._root_ops_of("reduce")}
    assert len(pool_names) == 26, "measured in the design; a drift here changes this test's meaning"
    with pytest.raises(ValueError, match="root-reducer"):
        TypedTreeGenerator(registry, op_weight=lambda s: 0.0 if s.name in pool_names else 1.0)


# ---------------------------------------------------------------------------
# accept hook: honoured everywhere, including generate()'s own fallback
# ---------------------------------------------------------------------------


def test_accept_hook_honoured_from_generate_and_every_mutation_and_crossover(registry):
    """Mirrors test_mining.py::test_banned_substructures_cannot_enter_by_any_route,
    for the accept hook instead of the ban set: generate(), every mutate_*, and
    crossover must all refuse the hook's victim."""
    free = TypedTreeGenerator(registry, n_minutes=24, seed=4, max_depth=4)
    pop = free.generate_population(200)
    victim = max((k for e in pop for k in one_layer_keys(e)),
                 key=lambda k: sum(k in one_layer_keys(e) for e in pop))

    def reject_victim(tree):
        return victim not in one_layer_keys(tree)

    gen = TypedTreeGenerator(registry, n_minutes=24, seed=4, max_depth=4, accept=reject_victim)
    fresh = gen.generate_population(200)
    assert not any(victim in one_layer_keys(e) for e in fresh)

    carrier = next(e for e in pop if victim in one_layer_keys(e))
    for _ in range(40):
        for fn in (gen.mutate_subtree, gen.mutate_operator, gen.mutate_hoist,
                   gen.mutate_window, gen.mutate_constant):
            child = fn(carrier)
            assert child is None or victim not in one_layer_keys(child), fn.__name__
        crossed = gen.crossover(carrier, fresh[0])
        assert crossed is None or victim not in one_layer_keys(crossed)


def test_generate_fallback_raises_when_hook_exhausted_and_on_exhausted_is_raise(registry):
    """Before WP5, generate()'s fallback bypassed acceptable() entirely: measured,
    an always-False hook still got day_cov(Close, Volume) back after 64
    rejections. With on_exhausted="raise" it must now raise instead."""
    gen = TypedTreeGenerator(registry, n_minutes=24, seed=8, min_depth=2, max_depth=2,
                              accept=lambda tree: False, on_exhausted="raise")
    with pytest.raises(GenerationExhausted):
        gen.generate()


def test_generate_fallback_returns_ban_respecting_tree_by_default_and_counts_it(registry):
    gen = TypedTreeGenerator(registry, n_minutes=24, seed=8, min_depth=2, max_depth=2,
                              accept=lambda tree: False)
    assert gen.unaccepted_fallbacks == 0
    tree = gen.generate()
    assert is_valid_root(tree)
    assert gen.unaccepted_fallbacks == 1


def test_generate_fallback_relaxed_tree_still_avoids_an_active_ban_set(registry):
    """The hook rejects everything, so the fallback can never return early via
    acceptable(); `relaxed` must still track only a tree that is ban_ok (not just
    any minimal collapse), proving the fallback discriminates on the ban set and
    not merely on is_valid_root."""
    probe = TypedTreeGenerator(registry, n_minutes=24, seed=2, min_depth=2, max_depth=2)
    pool = probe._root_ops_of("reduce")
    victim_tree = probe._collapse_leaf(pool[0])
    victim = next(iter(one_layer_keys(victim_tree)))

    gen = TypedTreeGenerator(registry, n_minutes=24, seed=2, min_depth=2, max_depth=2,
                              accept=lambda tree: False, banned_substructures={victim})
    tree = gen.generate()
    assert victim not in one_layer_keys(tree)
    assert gen.unaccepted_fallbacks == 1


def test_generation_exhausted_is_exported_and_is_a_value_error():
    from alphacube.mining import generator as generator_module

    assert "GenerationExhausted" in generator_module.__all__
    assert issubclass(GenerationExhausted, ValueError)


# ---------------------------------------------------------------------------
# The mutate_window / mutate_constant bypass, and its escape hatch
# ---------------------------------------------------------------------------


def test_mutate_window_now_respects_the_ban_set_and_the_escape_hatch_restores_it(registry):
    """Measured leak before WP5: with O:day_argmax(F:Low) banned, mutating the
    window of cross_quantile_q25(day_argmax(Low), 3) still returned a child,
    because mutate_window checked only is_valid_root. The window swap never
    touches the inner day_argmax(Low) node, so the banned key survives into every
    child regardless of which window is drawn."""
    tree = parse("cross_quantile_q25(day_argmax(Low), 3)")
    assert one_layer_keys(tree) == {"O:day_argmax(F:Low)"}

    gen = TypedTreeGenerator(registry, n_minutes=24, seed=0,
                              banned_substructures={"O:day_argmax(F:Low)"})
    assert all(gen.mutate_window(tree) is None for _ in range(40))

    legacy = TypedTreeGenerator(registry, n_minutes=24, seed=0,
                                 banned_substructures={"O:day_argmax(F:Low)"},
                                 mutation_checks_bans=False)
    children = [legacy.mutate_window(tree) for _ in range(40)]
    assert all(c is not None for c in children)
    assert all(one_layer_keys(c) == {"O:day_argmax(F:Low)"} for c in children)


def test_mutate_constant_now_respects_the_ban_set_and_the_escape_hatch_restores_it(registry):
    """Same hole, same fix: mul(day_cov(Volume, Low), 2.0) with
    O:day_cov(F:Volume,F:Low) banned. mutate_constant only ever touches the
    Constant node, never the day_cov one, so the banned key survives whatever
    constant value is drawn."""
    tree = parse("mul(day_cov(Volume, Low), 2.0)")
    assert one_layer_keys(tree) == {"O:day_cov(F:Volume,F:Low)"}

    gen = TypedTreeGenerator(registry, n_minutes=24, seed=1,
                              banned_substructures={"O:day_cov(F:Volume,F:Low)"})
    assert all(gen.mutate_constant(tree) is None for _ in range(40))

    legacy = TypedTreeGenerator(registry, n_minutes=24, seed=1,
                                 banned_substructures={"O:day_cov(F:Volume,F:Low)"},
                                 mutation_checks_bans=False)
    children = [legacy.mutate_constant(tree) for _ in range(40)]
    assert all(c is not None for c in children)
    assert all(one_layer_keys(c) == {"O:day_cov(F:Volume,F:Low)"} for c in children)


# ---------------------------------------------------------------------------
# hook_rng
# ---------------------------------------------------------------------------


def test_hook_rng_is_a_distinct_deterministic_instance(registry):
    a = TypedTreeGenerator(registry, n_minutes=24, seed=99)
    b = TypedTreeGenerator(registry, n_minutes=24, seed=99)
    assert a.hook_rng is not a._rng
    assert a.hook_rng.random() == b.hook_rng.random()


def test_hook_draining_hook_rng_does_not_move_generate_population(registry):
    """Emulates Partura's guidance hook, which must draw from hook_rng instead of
    gen._rng (the whole point of giving it a separate stream): draining it a
    data-dependent number of times must not perturb the tree-generation stream."""
    calls = []

    def hook(tree):
        calls.append(gen.hook_rng.random())
        return True

    gen = TypedTreeGenerator(registry, n_minutes=24, seed=17, max_depth=4, accept=hook)
    keys_with_hook = [e.key() for e in gen.generate_population(40)]
    assert calls, "the hook should have been consulted at least once"

    plain = TypedTreeGenerator(registry, n_minutes=24, seed=17, max_depth=4)
    keys_plain = [e.key() for e in plain.generate_population(40)]
    assert keys_with_hook == keys_plain


# ---------------------------------------------------------------------------
# Daily leaves in the search: daily_fields / daily_leaf_prob / leaf_slots
# ---------------------------------------------------------------------------


def test_daily_leaf_prob_one_always_draws_a_daily_feature_at_depth_one(registry):
    gen = TypedTreeGenerator(registry, n_minutes=24, seed=3,
                              daily_fields=("FloatCap",), daily_leaf_prob=1.0)
    for _ in range(50):
        leaf = gen._gen("daily", 1, full=False)
        assert isinstance(leaf, DailyFeature) and leaf.field == "FloatCap"
    for _ in range(50):
        leaf = gen._gen("collapsed", 1, full=False)
        assert isinstance(leaf, DailyFeature) and leaf.field == "FloatCap"


def test_daily_leaf_prob_zero_never_draws_a_daily_feature(registry):
    """daily_leaf_prob defaults to 0.0, and the guard is `self.daily_fields and
    self._rng.random() < self.daily_leaf_prob`, so even with daily_fields set,
    prob=0.0 must never produce one."""
    gen = TypedTreeGenerator(registry, n_minutes=24, seed=3, daily_fields=("FloatCap",))
    for _ in range(50):
        assert not isinstance(gen._gen("daily", 1, full=False), DailyFeature)


def test_daily_fields_integrate_into_whole_trees(registry):
    gen = TypedTreeGenerator(registry, n_minutes=24, seed=3, max_depth=4,
                              daily_fields=("FloatCap",), daily_leaf_prob=1.0)
    pop = gen.generate_population(30)
    assert all(is_valid_root(e) for e in pop)
    assert any(any(isinstance(n, DailyFeature) for n in e.nodes()) for e in pop)


def test_leaf_slots_fills_only_the_named_slot(registry):
    """leaf_slots={"cs_regres": {1: ("FloatCap",)}} must fill operand 1 with a
    DailyFeature and leave operand 0 to the normal recursive build -- the
    positional hint _operand_reqs cannot express on its own. See the module
    docstring for why cs_regres stands in for the design's cs_bucket_rank."""
    spec = registry.get("cs_regres")
    gen = TypedTreeGenerator(registry, n_minutes=24, seed=5,
                              leaf_slots={"cs_regres": {1: ("FloatCap",)}})
    for _ in range(30):
        tree = gen._build(spec, "daily", 3, False)
        assert isinstance(tree.operands[1], DailyFeature) and tree.operands[1].field == "FloatCap"
        assert not isinstance(tree.operands[0], DailyFeature)


def test_leaf_slots_and_daily_fields_both_empty_match_the_plain_population(registry):
    """Redundant with the explicit-defaults test above, kept as a direct pin on
    these two keywords specifically: both empty (the default) must not perturb
    generate_population's keys."""
    plain = TypedTreeGenerator(registry, n_minutes=24, seed=13, max_depth=4)
    same = TypedTreeGenerator(registry, n_minutes=24, seed=13, max_depth=4,
                               daily_fields=(), leaf_slots=None)
    assert [e.key() for e in plain.generate_population(40)] == \
        [e.key() for e in same.generate_population(40)]
