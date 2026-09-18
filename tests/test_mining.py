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

    # the top rows carry real metrics, computed from panels already in the cache,
    # so a checkpoint is readable mid-run instead of being a list of bare numbers
    for col in ("ic_mean", "icir", "rank_ic", "coverage", "sharpe", "max_drawdown"):
        assert col in rows[0], col
    scored = [r for r in rows if r["sharpe"]]
    assert scored, "expected the leaders to be scored from cached panels"
    assert all(float(r["max_drawdown"]) <= 0 for r in scored)
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


def test_scored_days_gate_rejects_a_panel_that_separates_nothing(registry, view):
    """coverage asks whether a cell is finite; this asks whether the day produced an
    IC at all. A panel that is identically zero passes the first and fails the
    second: finite everywhere, varying nowhere, so cs_pearson returns NaN for the
    day and the |IC| ends up drawn from whatever handful of days did vary.

    On the production cube that is not hypothetical --
    ``signedpower(pow(0.5, ts_quantile_q75(day_close(Volume), 50)))`` came top of a
    generation with |IC| 0.0741 and a Sharpe of 0.53, because 0.5**1e5 underflows to
    exactly 0 in float32 and its IC came from 329 days out of 1893."""
    import numpy as np

    from alphacube.label import Label
    from alphacube.mining.pool import MsePool

    target = Label("close", 1).compute(view)
    D, S = target.shape

    rng = np.random.default_rng(0)
    real = np.where(np.isfinite(target), rng.normal(size=(D, S)), np.nan)

    # A panel that is degenerate on EVERY day was already refused -- its mean IC is
    # NaN. The one that got through is the partial case: constant on most days, with
    # a spread on a few, so the mean IC is finite and comes from those few. That is
    # the real factor's shape, 329 scored days out of 1893.
    with pytest.raises(ValueError, match="no finite IC"):
        MsePool(4, target, min_scored_days=0.5).try_add(parse("day_mean(Close)"), np.zeros((D, S)))

    sparse = np.where(np.arange(D)[:, None] % 10 == 0, real, 0.0)
    with pytest.raises(ValueError, match="scored on"):
        MsePool(4, target, min_scored_days=0.5).try_add(parse("day_max(High)"), sparse)

    # the threshold is what decides, not some incidental property of the data
    MsePool(4, target, min_scored_days=0.05).try_add(parse("day_max(High)"), sparse)

    pool = MsePool(4, target, min_scored_days=0.5)
    pool.try_add(parse("day_std(Ret)"), real)     # a panel that varies every day gets in
    assert pool.size == 1


def test_pow_exponent_is_baked_never_searched(registry):
    """A binary pow lets the search put a tensor in the exponent, and in float32
    0.5**x underflows to exactly 0 above x=126 -- so pow(0.5, volume) is a panel of
    zeros wearing a healthy coverage number. Exponents are baked variants instead;
    the binary form stays reachable by name for research and the legacy corpus."""
    searchable = {s.name for s in registry.searchable(include_boolean=False,
                                                      include_resource_risk=False)}
    everything = {s.name for s in registry.list_all()}

    assert "pow" not in searchable, "a searchable binary pow invites pow(const, tensor)"
    assert "pow" in everything, "research and the legacy corpus still need it by name"
    assert {"pow_05", "pow_1_5", "pow_2", "pow_2_5", "pow_3"} <= searchable

    import torch
    binary = registry.get("pow").fn
    assert binary(torch.tensor([[0.5]]), torch.tensor([[1e5]])).item() == 0.0, \
        "this is the underflow the gate exists for"
    for name, exp in (("pow_05", 0.5), ("pow_2", 2.0), ("pow_3", 3.0)):
        got = registry.get(name).fn(torch.tensor([[4.0]])).item()
        assert abs(got - 4.0 ** exp) < 1e-4, name


def test_one_layer_keys_are_the_primitives_not_the_wrappers(registry):
    from alphacube.mining.generator import one_layer_keys

    assert one_layer_keys(parse("ts_decay_linear(sigmoid(pow_2(day_ar1_coefficient(Volume))), 5)")) == \
        {"O:day_ar1_coefficient(F:Volume)"}, "wrappers are not primitives; the leaf-fed node is"
    assert one_layer_keys(parse("ts_min(ts_reg_fvalue(day_std(High), day_cov(Volume, Low), 40), 5)")) == \
        {"O:day_std(F:High)", "O:day_cov(F:Volume,F:Low)"}, "a tree can rest on more than one"
    assert one_layer_keys(parse("day_mean(Close)")) == {"O:day_mean(F:Close)"}


def test_banned_substructures_cannot_enter_by_any_route(registry):
    """Banning has to hold at every acceptance point -- fresh trees, mutations and
    crossovers alike -- or the primitive walks back in through breeding."""
    from alphacube.mining.generator import TypedTreeGenerator, one_layer_keys

    free = TypedTreeGenerator(registry, n_minutes=24, seed=4, max_depth=4)
    pop = free.generate_population(200)
    victim = max((k for e in pop for k in one_layer_keys(e)),
                 key=lambda k: sum(k in one_layer_keys(e) for e in pop))

    gen = TypedTreeGenerator(registry, n_minutes=24, seed=4, max_depth=4,
                             banned_substructures={victim})
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


def test_initial_population_does_not_repeat_itself(registry):
    from alphacube.mining.generator import TypedTreeGenerator

    gen = TypedTreeGenerator(registry, n_minutes=24, seed=6, max_depth=4)
    pop = gen.generate_population(64)
    assert len({e.key() for e in pop}) >= 60, "a population of 64 should not be 40 distinct trees"


def test_rolling_restart_bans_what_the_block_leaned_on(registry, view):
    """A long run converges on one primitive and spends the rest of the budget
    re-wrapping it. Rolling restarts end each block by banning the primitives that
    block used most, so the next one has to look elsewhere."""
    from alphacube.label import Label
    from alphacube.mining.generator import one_layer_keys
    from alphacube.mining.pool import MsePool

    target = Label("close", 1).compute(view)
    m = GPMiner(registry, view, target, MsePool(5, target), population_size=12, max_depth=4,
                rolling_block=3, ban_top_n=2, seed=3)
    lines = []
    m.mine(12 * 9, on_generation=lambda g, n, info: lines.append(info))

    # the rebuilt population after a roll is evaluated like any generation, and its
    # line says how it did -- it used to print "valid 0" and NaN placeholders
    rolled = [x for x in lines if "rolled" in x]
    assert rolled and all(x["valid"] > 0 and np.isfinite(x["gen_best"]) for x in rolled), rolled

    assert len(m.blocks) >= 2, "nine generations in blocks of three must roll at least twice"
    for b in m.blocks:
        assert len(b["banned"]) <= 2 and b["counts"], b
    banned = m.generator.banned_substructures
    assert banned, "each roll should have banned something"
    assert not any(banned & one_layer_keys(e) for e in m.generator.generate_population(150))

    # off by default: same seed, no rolls, and the old three mutation kinds still work
    plain = GPMiner(registry, view, target, MsePool(5, target), population_size=12, max_depth=4, seed=3)
    plain.mine(12 * 9)
    assert plain.blocks == [] and not plain.generator.banned_substructures


def test_structures_can_be_banned_from_the_first_generation(registry, view):
    """--ban carries an earlier run's exhausted structures into a fresh one: they
    are refused from generation 1, and a key that names nothing is an error rather
    than a ban that silently bans nothing."""
    from alphacube.cli import build_parser
    from alphacube.label import Label
    from alphacube.mining.generator import one_layer_keys
    from alphacube.mining.pool import MsePool

    args = build_parser().parse_args(["mine", "--store", "x", "--ban", "O:log(F:Amt)",
                                      "--ban", "O:day_cov(F:Ret,F:Amt)"])
    assert args.ban == ["O:log(F:Amt)", "O:day_cov(F:Ret,F:Amt)"]

    target = Label("close", 1).compute(view)
    keys = ["O:log(F:Amt)", "O:nt_ts_argmax(F:High)|w3", "O:day_ar1_coefficient(F:High)", "O:day_cov(F:Ret,F:Amt)"]
    m = GPMiner(registry, view, target, MsePool(4, target), population_size=12, max_depth=3, seed=3,
                banned_substructures=keys)
    assert set(keys) <= m.generator.banned_substructures
    assert not any(set(keys) & one_layer_keys(e) for e in m.generator.generate_population(300))
    for bad in ("log(Amt)", "O:no_such_op(F:Amt)", "O:log(F:NoSuchField)", "O:log(O:abs_(F:Amt))"):
        with pytest.raises(ValueError):
            GPMiner(registry, view, target, MsePool(4, target), banned_substructures=[bad])


def test_rank_ban_refuses_monotone_rewrites(registry, view):
    """A structural ban was walked around by rewriting: with log(Amt) banned, gpu7
    came back with log(pow_1_5(Amt)) and pow_05(pow_05(day_max(Amt))) -- the same
    daily ranks as day_max(log(Amt)), so the same backtest. Ranked panels score
    every monotone rewrite (a sign flip included) at exactly 1.0; an unrelated
    factor passes."""
    from alphacube.label import Label
    from alphacube.mining.pool import MsePool

    target = Label("close", 1).compute(view)
    rewrites = [parse(s) for s in ("pow_05(pow_05(day_max(Amt)))", "day_max(pow_1_5(log(pow_1_5(Amt))))",
                                   "sub(0.5, day_max(Amt))")]
    other = parse("day_std(Ret)")
    plain = GPMiner(registry, view, target, MsePool(4, target), seed=0)
    banned = GPMiner(registry, view, target, MsePool(4, target), seed=0, ban_rank_corr=0.95,
                     ban_exemplars=[parse("day_max(log(Amt))")])
    assert banned._add_exemplars(banned._pending_exemplars) == 1          # what mine() does first
    f0, f1 = plain.fitness(rewrites + [other]), banned.fitness(rewrites + [other])
    assert np.isfinite(f0).all(), f0
    assert np.isneginf(f1[:3]).all() and np.isfinite(f1[3]), f1
    assert banned.similar_refused == 3
    # a rewrite of an exemplar is not stored again: they rank identically
    assert not banned._rank_ban.add(rewrites[0], banned._panels([rewrites[0]])[0])


def test_rolling_blocks_turn_banned_families_into_rank_exemplars(registry, view):
    from alphacube.label import Label
    from alphacube.mining.generator import one_layer_keys
    from alphacube.mining.pool import MsePool

    target = Label("close", 1).compute(view)
    m = GPMiner(registry, view, target, MsePool(5, target), population_size=12, max_depth=4,
                rolling_block=3, ban_top_n=2, seed=3, ban_rank_corr=0.95)
    lines = []
    m.mine(12 * 7, on_generation=lambda g, n, info: lines.append(info))
    assert m.blocks and m.blocks[0]["exemplars_total"] > 0, m.blocks
    assert len(m._rank_ban) == m.blocks[-1]["exemplars_total"]
    assert all("similar" in x and "exemplars" in x for x in lines)
    # every stored exemplar is an archived factor that used a banned structure
    banned = m.generator.banned_substructures
    archived = {repr(e): e for e, _ in m.archive()}
    assert all(one_layer_keys(archived[r]) & banned for r in m._rank_ban.exemplars)


def test_ban_exemplars_come_from_an_earlier_archive(tmp_path):
    from alphacube.pipeline import ban_exemplars_from

    rows = [(0.03, "day_max(log(Amt))"), (0.02, "ts_mean(day_max(log(Amt)), 5)"), (0.05, "day_std(Ret)"),
            (0.01, "day_max(log(Amt))"), (0.04, "not_an_operator(Close)")]
    path = tmp_path / "archive_gen010.csv"
    path.write_text("rank,fitness,expression\n" + "\n".join(f'{i},{f},"{e}"' for i, (f, e) in enumerate(rows)))
    got = ban_exemplars_from([path], ["O:log(F:Amt)"])
    assert [repr(e) for e in got] == [repr(parse("day_max(log(Amt))")), repr(parse("ts_mean(day_max(log(Amt)), 5)"))]


def test_rank_ic_turnover_fitness_is_what_the_reports_measure(registry, view):
    """fitness_kind / turnover_penalty: the fitness must be built from the numbers
    the checkpoint and the reports print -- ic_summary's rank IC, backtest_metrics'
    decile-book turnover -- or the search optimises something nobody reads.
    The IC path's gates (coverage, scored days) hold in this path too."""
    from alphacube.label import Label
    from alphacube.metrics import backtest_metrics, cs_pearson, nanmean_quiet, rank_rows
    from alphacube.mining.pool import MsePool

    target = Label("close", 1).compute(view)
    exprs = [parse(s) for s in ("day_std(Ret)", "day_max(log(Amt))", "ts_mean(day_mean(Close), 5)",
                                "sub(0.5, day_max(Amt))")]
    rich = GPMiner(registry, view, target, MsePool(4, target), seed=0, fitness_kind="rank_ic",
                   turnover_penalty=0.14)
    ic_pen = GPMiner(registry, view, target, MsePool(4, target), seed=0, fitness_kind="ic", turnover_penalty=0.3)
    f_rank, f_ic = rich.fitness(exprs), ic_pen.fitness(exprs)
    for e, p, fr, fi in zip(exprs, rich._panels(exprs), f_rank, f_ic):
        pan = np.asarray(p, dtype=np.float64)
        ric = nanmean_quiet(cs_pearson(rank_rows(pan), rank_rows(target)))
        ic = nanmean_quiet(cs_pearson(pan, target))
        to = backtest_metrics(cs_pearson(pan, target), pan, target)["turnover"]
        assert abs(fr - (abs(ric) - 0.14 * to)) < 1e-9, (e, fr, ric, to)
        assert abs(fi - (abs(ic) - 0.3 * to)) < 1e-9, (e, fi, ic, to)
    # a decreasing rewrite trades the same book the other way round: same |rank IC|
    assert abs(f_rank[1] - f_rank[3]) < 0.02 * abs(f_rank[1]) + 1e-9
    with pytest.raises(ValueError):
        GPMiner(registry, view, target, MsePool(4, target), fitness_kind="sharpe")

    # the gates: a panel constant on 9 days in 10 has a finite mean rank IC from the
    # tenth, and must still score -inf (scored days), like it does in the IC path
    D, S = target.shape
    rng = np.random.default_rng(0)
    real = np.where(np.isfinite(target), rng.normal(size=(D, S)), np.nan)
    sparse = np.where(np.arange(D)[:, None] % 10 == 0, real, 0.0).astype(np.float32)
    e = parse("day_max(High)")
    for m in (rich, GPMiner(registry, view, target, MsePool(4, target), seed=0)):
        m._panel_cache[e.key()] = sparse
        assert np.isneginf(m.fitness([e])[0])


def test_archive_keeps_one_member_per_rank_family(registry, view):
    """archive_dedup: gpu9's first block filled its top 12 archive places with
    rewrites of day_max(Ret), and under a rank fitness every monotone rewrite scores
    EXACTLY the same. Trees whose daily ranks correlate >= the threshold are one
    family: the archive keeps its best (fitness, then the smaller tree) and, within
    a generation, only that one keeps a fitness to breed with. An exact copy is a
    member too."""
    from alphacube.label import Label
    from alphacube.mining.pool import MsePool

    target = Label("close", 1).compute(view)
    fam = [parse(s) for s in ("pow_05(pow_05(day_max(Amt)))", "day_max(log(Amt))",
                              "day_max(pow_1_5(log(pow_1_5(Amt))))", "sub(0.5, day_max(Amt))")]
    other = parse("day_std(Ret)")
    pop = fam + [other, fam[1]]
    kw = dict(seed=0, fitness_kind="rank_ic", turnover_penalty=0.14)
    plain = GPMiner(registry, view, target, MsePool(4, target), **kw)
    plain._archive_update(pop, plain.fitness(pop))
    assert len(plain.archive()) == 5

    m = GPMiner(registry, view, target, MsePool(4, target), archive_dedup=0.99, **kw)
    raw = m.fitness(pop)
    assert np.isfinite(raw).all()
    fit = m._archive_update(pop, raw)
    assert len(m.archive()) == 2 and m._gen_dup == 4 and m.dup_demoted == 4
    kept_fam = [i for i in (0, 1, 2, 3, 5) if np.isfinite(fit[i])]
    assert len(kept_fam) == 1 and np.isfinite(fit[4]), fit
    best = max((0, 1, 2, 3), key=lambda i: (raw[i], -len(pop[i].nodes())))
    rep = pop[kept_fam[0]]
    assert rep.key() == pop[best].key() and m._archive[rep.key()][1] == raw[best]
    assert m._absorbed[rep.key()] == {e.key() for e in fam} - {rep.key()}
    assert m._absorbed[other.key()] == set()
    assert m._similar_info() == {"dup": 4}


def test_a_better_family_member_takes_over_the_archive_place(registry, view):
    """Whichever order two members of a family arrive in, the archive ends up
    holding the better one, counting the other as a member."""
    from alphacube.label import Label
    from alphacube.mining.pool import MsePool

    target = Label("close", 1).compute(view)
    a, b = parse("pow_05(pow_05(day_max(Amt)))"), parse("day_max(log(Amt))")
    kw = dict(seed=0, fitness_kind="rank_ic", turnover_penalty=0.14, archive_dedup=0.99)
    ends = []
    for order in ((a, b), (b, a)):
        m = GPMiner(registry, view, target, MsePool(4, target), **kw)
        for e in order:                                   # one per generation
            fit = m._archive_update([e], m.fitness([e]))
            assert np.isfinite(fit).all(), "alone in its generation, a member keeps its fitness"
        (rep, _), = m.archive()
        ends.append((rep.key(), m._absorbed[rep.key()], len(m._families), sorted(m._slot_of)))
    assert ends[0] == ends[1], ends
    assert len(ends[0][1]) == 1 and ends[0][2] == 1


def test_mining_with_families_keeps_the_archive_a_set_of_families(registry, view, tmp_path):
    """A whole run -- rolling blocks, rank ban, families, checkpoints: the archive,
    the index and the slot map stay one-to-one, no two archived factors rank alike
    at the threshold, and every checkpoint row carries what its fitness was made of."""
    import csv

    from alphacube.label import Label
    from alphacube.metrics import rank_rows, cs_pearson, nanmean_quiet
    from alphacube.mining.pool import MsePool
    from alphacube.mining.rankban import CompactRankIndex

    target = Label("close", 1).compute(view)
    m = GPMiner(registry, view, target, MsePool(5, target), population_size=12, max_depth=4, seed=3,
                rolling_block=3, ban_top_n=2, ban_rank_corr=0.95, fitness_kind="rank_ic",
                turnover_penalty=0.14, archive_dedup=0.99, cost_per_side=0.0015,
                checkpoint_dir=tmp_path / "ck", checkpoint_every=1)
    lines = []
    rep = m.mine(12 * 7, on_generation=lambda g, n, info: lines.append(info))
    assert all("dup" in x and "similar" in x for x in lines)
    assert rep.dup_demoted == m.dup_demoted > 0
    assert len(m._archive) == len(m._slot_of) == len(m._families) == len(m._slot_key)
    assert set(m._slot_key) == set(m._archive)
    y = m._families._y[:, :len(m._families)]
    c = CompactRankIndex.pairwise(y)
    np.fill_diagonal(c, 0.0)
    assert np.nanmax(c) < 0.99 + 1e-6, "two archived factors rank alike"

    rows = list(csv.DictReader(sorted((tmp_path / "ck").glob("archive_gen*.csv"))[-1].open()))
    assert len(rows) == len(m.archive())
    assert all(r["score_rank_ic"] and r["score_turnover"] and r["dups"] != "" for r in rows)
    for r in rows:
        assert abs(float(r["fitness"]) - (abs(float(r["score_rank_ic"])) - 0.14 * float(r["score_turnover"]))) < 2e-6
    top = [r for r in rows if r["sharpe"]]
    assert top and all(r["sharpe_net"] and r["turnover"] for r in top)
    for r in top:
        assert abs(float(r["turnover"]) - float(r["score_turnover"])) < 2e-6
        assert abs(float(r["rank_ic"]) - float(r["score_rank_ic"])) < 2e-6


def test_a_family_is_rank_banned_when_any_member_used_the_banned_structure(registry, view):
    """With families on, a banned structure can sit in a lesser member only -- the
    family of day_max(Amt) may well be represented by day_max(log(Amt)). The block
    end must still make the family a rank exemplar, or the next block breeds it
    back under its representative's name. The ban counts come from the raw
    fitness, so the demoted member's structure is counted like before families."""
    from alphacube.label import Label
    from alphacube.mining.generator import one_layer_keys
    from alphacube.mining.pool import MsePool

    target = Label("close", 1).compute(view)
    m = GPMiner(registry, view, target, MsePool(4, target), seed=0, fitness_kind="rank_ic",
                turnover_penalty=0.14, archive_dedup=0.99, ban_rank_corr=0.95, rolling_block=5, ban_top_n=1)
    pair = [parse("day_max(log(Amt))"), parse("pow_05(pow_05(day_max(Amt)))")]
    raw = m.fitness(pair)
    fit = m._archive_update(pair, raw)
    (rep, _), = m.archive()
    member = next(e for e in pair if e.key() != rep.key())
    assert np.isneginf(fit[pair.index(member)]) and np.isfinite(raw).all()
    only_member = one_layer_keys(member) - one_layer_keys(rep)
    assert only_member
    order = [member, rep]                                  # the member's structure is counted first
    banned = m._roll(order, m.fitness(order), 5)
    assert set(banned) == only_member
    assert m.blocks[-1]["exemplars_added"] == 1 and m._rank_ban.exemplars == [repr(rep)]


def test_scoring_threads_follow_the_container_quota(tmp_path):
    """torch sizes its pool from the host's cores (104 threads on the 208-core
    server) while the container gets 25 CPUs; the fitness sort ran 4x slower
    oversubscribed. The quota comes from cgroup v2 cpu.max; the pool size is
    restored after scoring so the engine's own setting is left alone."""
    import torch

    from alphacube.mining.gp import cpu_quota, torch_threads

    f = tmp_path / "cpu.max"
    f.write_text("2500000 100000\n")
    import os
    assert cpu_quota(str(f)) == min(25, os.cpu_count())
    f.write_text("max 100000\n")
    assert cpu_quota(str(f)) is None
    assert cpu_quota(str(tmp_path / "missing")) is None
    before = torch.get_num_threads()
    with torch_threads(1):
        assert torch.get_num_threads() == 1
    assert torch.get_num_threads() == before
    with torch_threads(None):
        assert torch.get_num_threads() == before
