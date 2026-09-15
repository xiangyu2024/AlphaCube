import numpy as np

from alphacube.engine.engine import CubeEngine
from alphacube.mining.generator import TypedTreeGenerator
from alphacube.parser import parse


def test_mask_policies_agree(registry, view):
    """'boundaries' masks only where a fake finite value could leak; it must equal 'all' bit for bit."""
    pop = TypedTreeGenerator(registry, n_minutes=view.n_minutes, seed=21, max_depth=4).generate_population(120)
    pop += [parse(s) for s in ("day_sum(is_gt0(Ret))", "cross_mean(day_sum(is_gt0(Ret)), 3)",
                               "day_mean(nt_cs_rank(is_gt0(Ret)))", "day_last(nt_ts_mean(is_gt0(Ret), 3))",
                               "cs_rank(day_sum(logical_not(is_gt0(Ret))))")]
    a = CubeEngine(view, chunk_symbols=3, chunk_days=7, mask_policy="all").evaluate(pop, tolerant=True)
    b = CubeEngine(view, chunk_symbols=3, chunk_days=7, mask_policy="boundaries").evaluate(pop, tolerant=True)
    assert set(a.errors) == set(b.errors)
    dead = ~view.tradable().T
    for e, x, y in zip(pop, a.panels, b.panels):
        if x is None:
            continue
        assert np.array_equal(x, y, equal_nan=True), e
        assert np.isnan(y[dead]).all(), e
