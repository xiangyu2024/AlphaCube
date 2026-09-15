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
