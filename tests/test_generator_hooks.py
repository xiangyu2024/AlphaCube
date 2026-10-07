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

DEVIATION from the design's literal ``hook_rng`` formula: it writes
``random.Random((seed * 2654435761) & 0xFFFFFFFF)``, which at the one in-tree
construction site (``gp.py:241`` passes ``rng=`` and never ``seed=``) is
``random.Random(0)`` in every mining run, and at the default ``seed=0`` is
``self._rng`` itself. ``generator.py``'s ``_hook_seed`` derives the hook seed from
the generator stream's state instead -- no draw consumed, so the tree stream is
still bit-identical -- and the three ``test_hook_rng_*`` tests below pin both
properties the formula was for. The alternative, keeping the formula and adding
``seed=seed`` at ``gp.py:241``, needs a file WP5 does not own and a step WP7 does
not have today.
"""

import itertools
import random
from collections import Counter

import pytest

from alphacube.expression import DailyFeature, Feature, is_valid_root, make_node
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


class _ChoicesRecorder(random.Random):
    """Captures the ``(population, weights)`` of every ``choices`` call, so a face
    draw's masses can be asserted directly instead of inferred from a sample."""

    def __init__(self, seed):
        super().__init__(seed)
        self.choices_calls = []

    def choices(self, population, weights=None, *, cum_weights=None, k=1):
        self.choices_calls.append((list(population), None if weights is None else list(weights)))
        return super().choices(population, weights=weights, cum_weights=cum_weights, k=k)


def _two_face_specs(registry):
    """Ten dt-face operators against two nd-face ones -- a deliberately lopsided
    pair of faces, so face-balancing is visible in a few thousand draws."""
    specs = TypedTreeGenerator(registry, n_minutes=24).specs
    dt = [s for s in specs if s.face == "dt"][:10]
    nd = [s for s in specs if s.face == "nd"][:2]
    assert len(dt) == 10 and len(nd) == 2
    return dt, nd


def test_unit_weights_leave_every_face_mass_equal(registry):
    """``op_weight=lambda s: 1.0`` must be the uniform path in distribution: a
    face's mass is its operators' MEAN weight, so ten equal weights and one equal
    weight are the same mass. A plain sum instead would make the masses the face
    sizes (measured 108 dt / 46 nd / 44 nt / 16 shared), i.e. the 108-operator dt
    face would swallow half the draws -- which is exactly what the face-balancing in
    ``_pick``'s docstring exists to prevent, and which no test caught."""
    rec = _ChoicesRecorder(0)
    gen = TypedTreeGenerator(registry, n_minutes=24, rng=rec, op_weight=lambda s: 1.0)
    assert not rec.choices_calls, "construction must not draw"
    gen._pick(gen.specs)
    live, masses = rec.choices_calls[0]              # the face draw, before the operator draw
    assert sorted(live) == sorted({s.face for s in gen.specs})
    assert len(set(masses)) == 1, dict(zip(live, masses))


def test_zeroing_most_of_a_face_does_not_shrink_that_faces_mass(registry):
    """Weight 0 is how "drop this operator from the vocabulary without shifting the
    draws" is expressed, so a face's mean must be taken over the operators that
    SURVIVE -- Partura's ``pick`` filters ``if w > 0`` before it groups by face
    (scripts/idea_search.py:225-238). Dividing by the face's full operator count
    instead scales its probability by the surviving fraction: measured, with only
    ``day_mean`` alive in the dt face, P(dt) falls from 0.25 to 0.003."""
    alive = lambda s: 1.0 if (s.face != "dt" or s.name == "day_mean") else 0.0   # noqa: E731
    rec = _ChoicesRecorder(0)
    gen = TypedTreeGenerator(registry, n_minutes=24, rng=rec, op_weight=alive)
    gen._pick(gen.specs)
    live, masses = rec.choices_calls[0]
    assert "dt" in live, "day_mean survives, so the dt face must stay in the draw"
    assert len(set(masses)) == 1, dict(zip(live, masses))


def test_unit_weights_reproduce_the_uniform_path_face_shares(registry):
    """The same property end to end through ``_pick``, on two lopsided faces: with
    every weight 1.0 the dt share must sit at the uniform path's 1/2, not at its
    10/12 share of the operators."""
    dt, nd = _two_face_specs(registry)
    specs = dt + nd
    n = 4000
    plain = TypedTreeGenerator(registry, n_minutes=24, rng=random.Random(0))
    unit = TypedTreeGenerator(registry, n_minutes=24, rng=random.Random(0), op_weight=lambda s: 1.0)
    p = sum(plain._pick(specs).face == "dt" for _ in range(n)) / n
    u = sum(unit._pick(specs).face == "dt" for _ in range(n)) / n
    assert abs(p - 0.5) < 0.04 and abs(u - 0.5) < 0.04, (p, u)   # sum-not-mean gives u = 10/12


def test_zeroing_nine_of_ten_operators_keeps_the_faces_draw_share(registry):
    """Nine of the ten dt operators weigh 0: the dt face keeps mass 1.0 (the mean
    over its one survivor) and so keeps its 1/2 share, and only the survivor is ever
    drawn from it. Dividing by the full ten would put the share at 0.1/1.1 = 9%."""
    dt, nd = _two_face_specs(registry)
    specs = dt + nd
    dead = {s.name for s in dt[1:]}
    gen = TypedTreeGenerator(registry, n_minutes=24, rng=random.Random(0),
                              op_weight=lambda s: 0.0 if s.name in dead else 1.0)
    n = 4000
    picks = [gen._pick(specs) for _ in range(n)]
    share = sum(p.face == "dt" for p in picks) / n
    assert abs(share - 0.5) < 0.04, share
    assert {p.name for p in picks if p.face == "dt"} == {dt[0].name}


def _partura_pick(gen, specs, w_of):
    """A frozen copy of Partura's ``pick``, the semantics ``_pick_weighted`` is
    specified to reproduce (``/Users/wxy/Desktop/Partura/scripts/idea_search.py``,
    the ``pick`` closure installed by ``install()``). Note where the ``if w > 0``
    filter sits: BEFORE the grouping, so ``len(faces[n])`` is the surviving count."""
    faces = {}
    for s in specs:
        w = w_of(s)
        if w > 0:
            faces.setdefault(s.face, []).append((s, w))
    if not faces:
        return _legacy_pick(gen._rng, specs)
    names = sorted(faces)
    f = gen._rng.choices(names, weights=[sum(w for _, w in faces[n]) / len(faces[n])
                                         for n in names])[0]
    return gen._rng.choices([s for s, _ in faces[f]], weights=[w for _, w in faces[f]])[0]


@pytest.mark.parametrize("label, w_of", [
    ("one operator boosted", lambda s: 4.0 if s.name == "day_mean" else 1.0),
    ("a whole face banned", lambda s: 0.0 if s.face == "nt" else 1.0),
    ("most of the dt face banned", lambda s: 1.0 if (s.face != "dt" or s.name == "day_mean") else 0.0),
    ("mixed boosts and bans", lambda s: {"day_mean": 4.0, "cs_rank": 0.25, "day_std": 0.0}.get(s.name, 1.0)),
    ("every weight zero", lambda s: 0.0),
])
def test_pick_weighted_is_parturas_pick_draw_for_draw(registry, label, w_of):
    """"This reproduces Partura's current ``pick`` semantics exactly, so the
    migration is behaviour-preserving" (WP5 step 7) -- asserted the only way that
    claim can be, against a local copy of that ``pick`` on a shared seed, so the two
    must agree draw for draw and not merely in distribution. This is what the
    denominator bug broke: dividing a face's weight sum by its FULL operator count
    instead of by its surviving count made the migration silently reweight both
    arms."""
    specs = TypedTreeGenerator(registry, n_minutes=24).specs
    mine = TypedTreeGenerator(registry, n_minutes=24, rng=random.Random(3), op_weight=lambda s: 1.0)
    mine.op_weight = w_of                   # post-construction: the all-zero case must be reachable
    theirs = TypedTreeGenerator(registry, n_minutes=24, rng=random.Random(3))
    assert [mine._pick(specs).name for _ in range(400)] == \
        [_partura_pick(theirs, specs, w_of).name for _ in range(400)], label


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


def _reachable_minimal_keys(gen):
    """Every depth-1 key ``generate()``'s fallback can possibly draw, enumerated
    exhaustively rather than sampled: ``_collapse_leaf`` builds one root reducer over
    ``1 if category in ('unary','rolling') else 2`` field leaves, so the reachable set
    is that product over ``gen.fields``. (Sampling it, as 400 draws per spec, misses a
    binary's rarer field pairs often enough to make a count assertion flaky: 2 of the
    26 reducers are binary, i.e. 64 ordered pairs each.) Windows do not enter
    ``key()``, verified by the count below matching an exhaustive sample."""
    keys = set()
    for spec in gen._root_ops_of("reduce"):
        n = 1 if spec.category in ("unary", "rolling") else 2
        for combo in itertools.product(gen.fields, repeat=n):
            leaves = tuple(Feature(f) for f in combo)
            keys |= one_layer_keys(make_node(spec, leaves, gen._window_for(spec)))
    return keys


def test_generate_fallback_never_returns_a_banned_tree_when_the_ban_covers_the_draws(registry):
    """The deliberate behaviour change the WP5 fallback fix exists for: a fallback
    tree carrying a banned key is no longer returned at all.

    The ban set has to COVER what the fallback can draw for this to be testable.
    Banning a single key tests nothing -- 320 minimal keys are reachable, so the
    chance the first fallback draw is that one key is 0.3%, and a ``generate()``
    that had lost its ``_ban_ok`` check entirely would still look clean. Here all
    320 but one ``spare`` are banned and the hook refuses everything, so the only
    two legal outcomes are GenerationExhausted or the spare. Measured against a copy
    of ``generate()`` with ``self._ban_ok(tree) and`` deleted from the ``relaxed``
    test: that version returns a BANNED tree in 40 of 40 seeds, this one in 0 of 40.
    """
    probe = TypedTreeGenerator(registry, n_minutes=24, seed=2, min_depth=2, max_depth=2)
    keys = _reachable_minimal_keys(probe)
    assert len(keys) == 320, (
        f"measured: 8 fields x (24 unary + 2 binary) root reducers; "
        f"drift here changes this test's meaning (got {len(keys)})")
    spare = sorted(keys)[0]
    ban = keys - {spare}

    outcomes = Counter()
    for seed in range(20):
        gen = TypedTreeGenerator(registry, n_minutes=24, seed=seed, min_depth=2, max_depth=2,
                                  accept=lambda tree: False, banned_substructures=ban)
        try:
            tree = gen.generate()
        except GenerationExhausted:
            outcomes["raised"] += 1
            continue
        assert not (ban & one_layer_keys(tree)), (seed, repr(tree))
        assert one_layer_keys(tree) == {spare}, (seed, repr(tree))
        assert gen.unaccepted_fallbacks == 1
        outcomes["spare"] += 1
    # Both branches must actually occur, or the assertions above never ran on a
    # returned tree and the test would be as vacuous as the one it replaces.
    assert outcomes["spare"] and outcomes["raised"], outcomes


def test_generate_fallback_relaxed_tree_still_avoids_a_single_banned_key(registry):
    """Cheap smoke test of the same route with one key banned. Kept deliberately
    separate from the test above, which is the one that pins the property: with 320
    reachable keys this one passes even if the fallback stops consulting the ban set
    altogether, so nothing may rest on it."""
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


def _built_like_gp(registry, seed):
    """Exactly how the only in-tree caller builds the generator: ``gp.py:241`` passes
    ``rng=self._rng`` (which is ``random.Random(seed)``) and never ``seed=``."""
    return TypedTreeGenerator(registry, n_minutes=24, rng=random.Random(seed))


def test_hook_rng_varies_with_the_seed_when_the_rng_is_supplied(registry):
    """The whole point of hook_rng is that "the steer/control comparison is
    reproducible from the seed". Seeding it from the ``seed`` keyword alone fails at
    the one place it is constructed: gp.py passes ``rng=`` only, so ``seed`` stayed
    at 0 and run seeds 0, 7 and 12345 all got the identical hook stream
    [0.844422, 0.757954, 0.420572] (measured) -- N seeds of the steer arm sharing one
    thinning realisation."""
    s7 = [_built_like_gp(registry, 7).hook_rng.random() for _ in range(3)]
    s12345 = [_built_like_gp(registry, 12345).hook_rng.random() for _ in range(3)]
    assert s7 != s12345
    assert [_built_like_gp(registry, 7).hook_rng.random() for _ in range(3)] == s7


def test_hook_rng_is_never_the_generators_own_sequence(registry):
    """"A second, independent stream" has to mean a different sequence, not just a
    different object: ``random.Random((0 * 2654435761) & 0xFFFFFFFF)`` is
    ``random.Random(0)``, which at the DEFAULT seed is ``self._rng`` byte for byte, so
    a hook thinning on ``hook_rng.random()`` would have been reading the generator's
    own draws."""
    for gen in (TypedTreeGenerator(registry, n_minutes=24),
                TypedTreeGenerator(registry, n_minutes=24, seed=0),
                _built_like_gp(registry, 0)):
        assert [gen.hook_rng.random() for _ in range(3)] != [gen._rng.random() for _ in range(3)]


def test_constructing_the_generator_consumes_no_draw_from_the_supplied_rng(registry):
    """The constraint deriving the hook seed from the rng has to respect:
    ``getstate()`` does not advance the stream, so construction may not move the
    caller's rng by a single draw -- GPMiner shares that rng with the generator, and
    one extra draw diverges every trajectory from generation 1."""
    probe = random.Random(5)
    TypedTreeGenerator(registry, n_minutes=24, rng=probe)
    assert probe.getstate() == random.Random(5).getstate()


def test_hook_rng_falls_back_to_the_seed_for_an_rng_with_no_state(registry):
    """``random.SystemRandom.getstate()`` raises; the hook stream must still exist."""
    gen = TypedTreeGenerator(registry, n_minutes=24, seed=4, rng=random.SystemRandom())
    assert isinstance(gen.hook_rng.random(), float)
    assert gen.hook_rng.random() != gen.hook_rng.random()


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
