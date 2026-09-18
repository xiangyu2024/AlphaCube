"""MsePool's refit through the Gram cache: same least-squares answer as the
direct solve, same insertion outcomes, and never the full design again."""

import numpy as np
import pytest

from alphacube.metrics import zscore_rows
from alphacube.mining.pool import DuplicateFactorError, Member, MsePool, _GramCache, _lstsq_weights
from alphacube.parser import parse


def _panels(rng, D, S, k, target):
    """k candidate panels with NaN holes: some carry the target, some are near-copies."""
    out = []
    for i in range(k):
        if i % 5 == 4 and out:
            p = out[-1] + 0.05 * rng.normal(size=(D, S))          # near-duplicate of the previous one
        else:
            p = rng.normal(size=(D, S)) + (0.3 * i / k) * np.nan_to_num(target)
        p[rng.random((D, S)) < 0.1] = np.nan
        out.append(p)
    return out


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    D, S = 60, 80
    target = rng.normal(size=(D, S))
    target[rng.random((D, S)) < 0.2] = np.nan
    return rng, target, _panels(rng, D, S, 25, target)


class _Direct(MsePool):
    """The refit MsePool used to do: stack every member and solve from scratch."""

    def refit(self):
        self.weights = _lstsq_weights(self.z_columns(), self.target) if self.members else np.zeros(0)
        return self.weights


def test_gram_weights_match_the_direct_solve(data):
    rng, target, panels = data
    members = [Member(parse(f"add(day_mean(Close), {i})"), zscore_rows(p), 0.0) for i, p in enumerate(panels[:12])]
    cache = _GramCache(target)
    for k in (1, 2, 5, 12):
        got = cache.weights(members[:k])
        ref = _lstsq_weights([m.z for m in members[:k]], target)
        np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)
    # drop members from the middle and re-add them in another order: the cache must
    # follow the member list, not the insertion history
    order = [members[i] for i in (7, 0, 11, 3)]
    np.testing.assert_allclose(cache.weights(order), _lstsq_weights([m.z for m in order], target),
                               rtol=1e-9, atol=1e-12)
    # an exact copy makes the design rank-deficient: both answers are the minimum-norm one
    twin = Member(members[0].expr, members[0].z.copy(), 0.0)
    both = [members[0], twin, members[1]]
    np.testing.assert_allclose(cache.weights(both), _lstsq_weights([m.z for m in both], target),
                               rtol=1e-6, atol=1e-9)


def test_insertions_match_the_direct_solve(data):
    """Same candidates, same order, capacity small enough to force pruning and a
    dedup threshold that rejects the near-copies: every accept / reject, every
    pruned member and the final weights agree with the direct solve."""
    rng, target, panels = data
    fast, slow = MsePool(6, target, dedup_threshold=0.9), _Direct(6, target, dedup_threshold=0.9)
    outcomes = []
    for i, p in enumerate(panels):
        e = parse(f"add(day_mean(Close), {i})")
        got = []
        for pool in (fast, slow):
            try:
                got.append(("ok", pool.try_add(e, p)))
            except DuplicateFactorError:
                got.append(("dup", None))
        assert got[0][0] == got[1][0], f"candidate {i}: {got}"
        if got[0][0] == "ok":
            assert got[0][1] == pytest.approx(got[1][1], rel=1e-9)
        outcomes.append(got[0][0])
        assert [repr(m.expr) for m in fast.members] == [repr(m.expr) for m in slow.members], f"after {i}"
        np.testing.assert_allclose(fast.weights, slow.weights, rtol=1e-9, atol=1e-12)
    assert "dup" in outcomes and outcomes.count("ok") > fast.capacity     # both paths were exercised


def test_refit_never_solves_the_full_design(data, monkeypatch):
    """The slow path this replaces: every try_add stacked all members into an
    (n_cells, k) design and ran lstsq on it -- 22 s per solve at k = 31 on the
    production cube, twice per insertion once the pool was full. Now the only
    solve is on the k x k normal equations."""
    rng, target, panels = data
    shapes = []
    real = np.linalg.lstsq

    def spy(a, b, *args, **kw):
        shapes.append(np.shape(a))
        return real(a, b, *args, **kw)

    monkeypatch.setattr(np.linalg, "lstsq", spy)
    pool = MsePool(6, target, dedup_threshold=0.9)
    for i, p in enumerate(panels):
        try:
            pool.try_add(parse(f"add(day_mean(Close), {i})"), p)
        except DuplicateFactorError:
            pass
    assert shapes, "no solve happened"
    assert max(s[0] for s in shapes) <= pool.capacity + 1, shapes


def test_pool_ic_from_the_cache_is_the_combined_ic(data):
    """pool_ic on the cached vectors does the combination's own per-cell arithmetic
    (0 + w1*z1 + w2*z2 ..., member order) on the cells cs_pearson keeps, so it must
    equal the combined()-based answer exactly, not approximately."""
    from alphacube.mining.pool import PoolBase

    rng, target, panels = data
    pool = MsePool(6, target, dedup_threshold=0.9)
    checked = 0
    for i, p in enumerate(panels):
        try:
            pool.try_add(parse(f"add(day_mean(Close), {i})"), p)
        except DuplicateFactorError:
            continue
        assert pool.pool_ic() == PoolBase.pool_ic(pool), f"after candidate {i}"
        checked += 1
    assert checked > pool.capacity
