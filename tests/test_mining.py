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


def test_only_the_nt_face_pair_kernels_leave_the_action_set(registry):
    """``ts_pearson`` and friends correlate every stock against every other, which
    is O(windows x d x N^2) -- 27M pairs per window at N=5219. That cost is the
    factor's definition, not a slow implementation: the pairwise observation count
    is taken per pair, because stocks suspend and list on different days, so the
    bilinear identity that would collapse agg="mean" to O(N) does not hold.

    What decides affordability is the face. The ND entry is evaluated once over
    the whole (D, S) panel; its NT twin is evaluated once per day, 1942 times,
    which is two orders of magnitude more. So the ND ones stay searchable and only
    the NT twins are gated -- a blanket ban would throw away a whole family of
    factors to avoid a cost that only one face actually carries.

    A full-market run bred an NT one into generation 14, which then ran for 196
    minutes against a 4-minute norm."""
    from alphacube.calculate import DEFERRED_RESOURCE_OPERATORS
    from alphacube.catalog import RESOURCE_RISK_KERNELS, SEARCH_FORBIDDEN_KERNELS
    from alphacube.mining.tokens import Vocabulary

    assert DEFERRED_RESOURCE_OPERATORS is RESOURCE_RISK_KERNELS, "one definition, not two"
    assert SEARCH_FORBIDDEN_KERNELS < RESOURCE_RISK_KERNELS, "the search bans a subset, not the lot"

    gated = {s.name for s in registry.searchable(include_boolean=False, include_resource_risk=False)}
    assert {"ts_pearson", "ts_spearman"} <= gated, "the ND face is affordable and must stay"
    assert gated.isdisjoint({"nt_ts_pearson", "nt_ts_spearman"}), "the NT twins cost ~241x"

    opened = {s.name for s in registry.searchable(include_boolean=False, include_resource_risk=True)}
    assert {"nt_ts_pearson", "nt_ts_spearman"} <= opened, "the switch must let them back in"

    risky = {s.name for s in registry.list_all() if s.resource_risk}
    for flag in (False, True):
        gen = TypedTreeGenerator(registry, n_minutes=24, seed=7, max_depth=5, allow_resource_risk=flag)
        vocab = Vocabulary(registry, n_minutes=24, allow_resource_risk=flag)
        assert (({s.name for s in gen.specs} & risky) == risky) is flag
        assert (({s.name for s in vocab.operators} & risky) == risky) is flag

    off = TypedTreeGenerator(registry, n_minutes=24, seed=8, max_depth=5, allow_resource_risk=False)
    pop = off.generate_population(300)
    assert not any(n.spec.name in risky for e in pop for n in e.nodes() if n.spec is not None)
    assert any(n.spec.name == "ts_pearson" for e in off.generate_population(600)
               for n in e.nodes() if n.spec is not None), "the ND entry must still be reachable"


def test_archive_checkpoints_survive_an_interrupted_run(registry, view, tmp_path):
    """The archive lives in memory and only reaches disk in _commit(), so a
    full-market run killed at hour ten leaves no expressions at all -- which is
    exactly what happened. checkpoint_every writes the ranked archive as it goes."""
    from alphacube.label import Label
    from alphacube.mining.pool import MsePool

    target = Label("close", 1).compute(view)
    ckpt = tmp_path / "checkpoints"
    miner = GPMiner(registry, view, target, MsePool(4, target), population_size=6, max_depth=3,
                    checkpoint_dir=ckpt, checkpoint_every=1, seed=2)
    miner.mine(24)

    files = sorted(ckpt.glob("archive_gen*.csv"))
    assert len(files) >= 3, files
    assert not list(ckpt.glob("*.tmp")), "the write is atomic; no partial file may survive"

    import csv
    rows = list(csv.DictReader(files[-1].open(encoding="utf-8")))
    assert rows, "a checkpoint must carry the archive, not just a header"
    fits = [float(r["fitness"]) for r in rows]
    assert fits == sorted(fits, reverse=True), "ranked best first"
    assert all(parse(r["expression"]).key() for r in rows), "every row must parse back"
    assert len(rows) == len(miner.archive())

    # and the archive only grows, so a later checkpoint is never poorer than an earlier one
    early = list(csv.DictReader(files[0].open(encoding="utf-8")))
    assert len(rows) >= len(early)


def test_skip_kernels_reproduces_the_trajectory_until_the_first_refusal(registry, view):
    """Refusing a tree at EVALUATION time is not the same as removing its operator
    from the vocabulary, and the difference is the whole point.

    The generator draws operators from the vocabulary using the miner's RNG, so
    dropping one shifts every subsequent draw: a re-run diverges from generation 1
    and the factors found before the problem are gone with it. Keeping the
    vocabulary intact and scoring the offending tree -inf unevaluated leaves the
    draw sequence untouched, so every generation up to the first refusal is
    reproduced exactly.

    That is what lets a run poisoned at generation 14 by an expensive operator be
    replayed: generations 1-13 come back identical, and only from 14 on does
    selection differ -- this time without burning hours on the tree."""
    from alphacube.label import Label
    from alphacube.mining.pool import MsePool

    target = Label("close", 1).compute(view)

    def run(skip, dropped):
        reg_specs = {s.name for s in registry.searchable()}
        assert not dropped or dropped <= reg_specs
        m = GPMiner(registry, view, target, MsePool(4, target), population_size=8, max_depth=3,
                    min_depth=2, skip_kernels=skip, seed=5)
        pops = []
        orig = m.fitness

        def spy(pop):
            pops.append([e.key() for e in pop])
            return orig(pop)

        m.fitness = spy
        m.mine(40)
        return pops, m

    base_pops, base = run(frozenset(), set())
    assert base.skipped == 0

    # Refuse something the search actually uses, so there IS a first refusal.
    used = [k for pop in base_pops for k in pop]
    victim = next(n for n in ("day_mean", "day_std", "cs_rank", "ts_mean")
                  if any(f"O:{n}(" in k for k in used))
    skip_pops, skipped = run(frozenset({victim}), {victim})
    assert skipped.skipped > 0, "the victim must actually have been bred and refused"

    first_hit = next(i for i, pop in enumerate(base_pops)
                     if any(f"O:{victim}(" in k for k in pop))
    assert base_pops[:first_hit + 1] == skip_pops[:first_hit + 1], \
        "every generation up to and including the first refusal must be identical"

    # ... and dropping the operator from the vocabulary instead diverges immediately
    dropped = GPMiner(registry, view, target, MsePool(4, target), population_size=8, max_depth=3,
                      min_depth=2, seed=5)
    dropped.generator.specs = [s for s in dropped.generator.specs if s.name != victim]
    for key in list(dropped.generator._producers):
        dropped.generator._producers[key] = [s for s in dropped.generator._producers[key]
                                             if s.name != victim]
    first = [e.key() for e in dropped.generator.generate_population(8)]
    assert first != base_pops[0], "removing it from the vocabulary must shift the very first draw"
