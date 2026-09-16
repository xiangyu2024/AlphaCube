import numpy as np
import pytest

from alphacube.expression import contains_lookahead, is_valid_root
from alphacube.label import Label
from alphacube.mining.generator import TypedTreeGenerator, shape_req
from alphacube.mining.gp import GPMiner
from alphacube.mining.pool import DuplicateFactorError, LinearPool
from alphacube.parser import parse


def test_generator_sweep(registry):
    gen = TypedTreeGenerator(registry, n_minutes=24, seed=3)
    pop = gen.generate_population(300)
    faces = set()
    for e in pop:
        assert is_valid_root(e), e
        assert not contains_lookahead(e), e
        assert parse(repr(e)).key() == e.key(), e
        faces |= {n.spec.face for n in e.nodes() if n.spec}
        for n in e.nodes():
            if n.spec is not None and n.spec.restores_m:
                assert n.window == 24
    assert {"dt", "nd", "nt", "shared"} <= faces
    assert {e.key() for e in pop} and len({e.key() for e in pop}) > 200


def test_operators_stay_type_safe(registry):
    gen = TypedTreeGenerator(registry, n_minutes=24, seed=4)
    pop = gen.generate_population(60)
    ok = 0
    for i in range(120):
        a, b = pop[i % 60], pop[(i * 7) % 60]
        c = gen.crossover(a, b)
        if c is not None:
            ok += 1
            assert is_valid_root(c) and parse(repr(c)).key() == c.key()
        for m in (gen.mutate_subtree(a), gen.mutate_window(a), gen.mutate_constant(a)):
            if m is not None:
                assert is_valid_root(m)
    assert ok > 60
    assert shape_req(parse("day_mean(Close)")) == "collapsed" and shape_req(parse("Close")) == "full"
    assert shape_req(parse("mul(day_mean(Close), 2)")) == "daily"


def _panels(seed, D=80, S=20, k=4):
    rng = np.random.default_rng(seed)
    target = rng.normal(size=(D, S))
    panels = [target * w + rng.normal(size=(D, S)) for w in (1.0, 0.5, 0.2, 0.0)][:k]
    return target, panels


def test_pool_weights_prune_and_dedup():
    target, panels = _panels(0)
    pool = LinearPool(capacity=3, target=target)
    exprs = [parse(f"ts_mean(day_mean(Close), {w})") for w in (5, 10, 20, 30)]
    for e, p in zip(exprs, panels):
        ic = pool.try_add(e, p)
        assert np.isfinite(ic)
    assert pool.size == 3 and not np.allclose(pool.weights, pool.weights[0])
    assert exprs[3] not in pool.exprs() or exprs[2] not in pool.exprs()   # weakest pruned
    before = list(pool.exprs()), pool.weights.copy()
    with pytest.raises(DuplicateFactorError):
        pool.try_add(parse("ts_mean(day_mean(Close), 40)"), panels[0] * 2 + 1)   # affine copy
    assert pool.exprs() == before[0] and np.array_equal(pool.weights, before[1])   # transactional
    with pytest.raises(ValueError):
        pool.try_add(parse("ts_mean(day_mean(Close), 50)"), np.ones_like(target))  # no IC


def test_gp_populates_pool(registry, store):
    view = store.view(start=store.dates[5])
    target = Label("close", 1).compute(view)
    pool = LinearPool(capacity=4, target=target, dedup_threshold=0.99)
    miner = GPMiner(registry, view, target, pool, population_size=12, max_depth=3, seed=2)
    seen = []
    rep = miner.mine(48, on_generation=lambda g, n, s: seen.append((g, n)))
    assert seen == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert pool.size >= 1 and np.isfinite(pool.pool_ic())
    assert all(np.isfinite(m.ic) for m in pool.members)
    assert rep.archive_size >= pool.size
    assert rep.evaluated < 12 * 5                        # panel cache saved re-evaluations of elites/parents
    assert all(is_valid_root(e) for e in rep.added)


def test_trivial_boolean_ops_are_out_of_the_action_set_by_default(registry):
    """A strictly-0/1 operator cannot be a factor's root, but that rule is checked
    one node at a time, so arithmetic launders one through: ``pow_05(eq(a, b))``
    is a legal root whose panel is a near-constant indicator, and on the
    production store such a tree took the top of the population with a spurious
    |IC| of 0.58 while the pool recorded its IC as NaN.

    What gets dropped is only the booleans that can COLLAPSE -- predicates against
    a fixed threshold, which go all-true or all-false as soon as the data sits on
    one side of it. A predicate against the data's own distribution cannot
    collapse and stays in the vocabulary: ``cs_quantile_q50`` splits its input in
    half whatever the input is."""
    from alphacube.mining.tokens import Vocabulary

    trivial = {s.name for s in registry.searchable() if s.trivial_boolean}
    kept = {s.name for s in registry.searchable() if s.boolean_output and not s.trivial_boolean}
    assert {"eq", "greater", "less", "is_gt0", "is_lt0", "logical_and", "logical_or",
            "logical_not", "sign"} <= trivial, "fixed-threshold predicates must be droppable"
    assert {"cs_quantile_q50", "cs_quantile_q25", "in_top_quantile_flag_q10"} <= kept, \
        "distribution-relative flags carry information and must survive"
    assert trivial.isdisjoint(kept)

    boolean = trivial
    assert {s.name for s in registry.searchable(include_boolean=False)}.isdisjoint(boolean)
    assert len(registry.searchable(include_boolean=False)) == len(registry.searchable()) - len(boolean)

    for allow in (False, True):
        gen = TypedTreeGenerator(registry, n_minutes=24, seed=5, allow_boolean_ops=allow)
        vocab = Vocabulary(registry, n_minutes=24, allow_boolean_ops=allow)
        in_gen = {s.name for s in gen.specs} & boolean
        in_vocab = {s.name for s in vocab.operators} & boolean
        assert (in_gen == boolean) is allow and (in_vocab == boolean) is allow

    # and nothing boolean reaches a generated tree, at any depth, when it is off
    off = TypedTreeGenerator(registry, n_minutes=24, seed=6, allow_boolean_ops=False)
    pop = off.generate_population(200)
    assert not any(n.spec.name in boolean for e in pop for n in e.nodes() if n.spec is not None)
    on = TypedTreeGenerator(registry, n_minutes=24, seed=6, allow_boolean_ops=True)
    assert any(n.spec.name in boolean for e in on.generate_population(200)
               for n in e.nodes() if n.spec is not None), "the switch must actually let them back in"


def test_panel_cache_is_configurable_and_bounded(registry, view):
    """A cache miss re-reads every field the tree touches -- 9.1 GB each on the
    production cube -- so the cache size is the difference between a 4-minute
    generation and a 50-minute one. It was reachable only by constructing GPMiner
    directly; `cli mine --panel-cache` and train.py's gp_params now set it."""
    from alphacube.label import Label
    from alphacube.mining.pool import MsePool

    target = Label("close", 1).compute(view)
    miner = GPMiner(registry, view, target, MsePool(4, target), population_size=6,
                    max_depth=3, panel_cache_items=5, seed=1)
    miner.mine(24)
    assert len(miner._panel_cache) <= 5, "the LRU bound must hold"
    assert miner._panel_cache_items == 5

    big = GPMiner(registry, view, target, MsePool(4, target), population_size=6,
                  max_depth=3, panel_cache_items=500, seed=1)
    big.mine(24)
    # same seed, same answer -- the cache is a speed device, never a semantic one
    assert len(big._panel_cache) > len(miner._panel_cache)
    assert big.pool.size == miner.pool.size


def test_resource_risk_ops_are_out_of_the_action_set_by_default(registry):
    """calculate.py has always refused to compute a factor containing ts_pearson,
    nt_ts_pearson, nt_ts_spearman or in_multireg without --allow-resource-risk,
    because their cost grows super-linearly in the stock count -- the NT pair
    kernels correlate every stock against every other, once per day, which on the
    production cube is 5219^2 pairs x 1942 days.

    The search vocabulary did not have that guard, so GP was free to breed one.
    On a full-market run it did, in generation 14, and that generation was still
    running 78 minutes later against a 4-minute norm. Both paths now read the same
    set, so they cannot drift apart again."""
    from alphacube.calculate import DEFERRED_RESOURCE_OPERATORS
    from alphacube.catalog import RESOURCE_RISK_KERNELS
    from alphacube.mining.tokens import Vocabulary

    assert DEFERRED_RESOURCE_OPERATORS is RESOURCE_RISK_KERNELS, "one definition, not two"
    risky = {s.name for s in registry.list_all() if s.resource_risk}
    assert risky and risky <= RESOURCE_RISK_KERNELS

    # Registry.searchable() itself answers "what exists"; the opting out happens at
    # the miner, the same way include_boolean does, so that a caller asking the
    # registry a question is never silently given a filtered answer.
    gated = {s.name for s in registry.searchable(include_boolean=False, include_resource_risk=False)}
    assert gated.isdisjoint(risky)
    assert risky <= {s.name for s in registry.searchable()}

    for flag in (False, True):
        gen = TypedTreeGenerator(registry, n_minutes=24, seed=7, max_depth=5, allow_resource_risk=flag)
        vocab = Vocabulary(registry, n_minutes=24, allow_resource_risk=flag)
        assert (({s.name for s in gen.specs} & risky) == risky) is flag
        assert (({s.name for s in vocab.operators} & risky) == risky) is flag

    off = TypedTreeGenerator(registry, n_minutes=24, seed=8, max_depth=5, allow_resource_risk=False)
    assert not any(n.spec.name in risky for e in off.generate_population(300)
                   for n in e.nodes() if n.spec is not None)
