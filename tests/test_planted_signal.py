"""A signal that is really in the data must come out: the research entry
measures the planted factor's IC, and a small GP run finds a pool whose
combined out-of-sample IC is clearly positive."""

import numpy as np

from alphacube.catalog import default_registry
from alphacube.label import Label
from alphacube.metrics import cs_pearson, nanmean_quiet
from alphacube.mining.gp import GPMiner
from alphacube.mining.pool import LinearPool
from alphacube.research import evaluate_factors
from alphacube.synthetic import write_synthetic_store


def test_planted_signal_recovered(tmp_path):
    store = write_synthetic_store(tmp_path / "cube", n_symbols=40, n_days=160, n_minutes=30,
                                  seed=21, suspended_frac=0.02, plant_beta=1.0)
    table = evaluate_factors(store, {"planted": "day_std(Ret)", "noise": "day_mean(nt_cs_rank(Volume))"},
                             label=Label("close", 1), quiet=True)
    assert table.loc["planted", "ic_mean"] > 0.25
    assert abs(table.loc["noise", "ic_mean"]) < 0.15

    # GP: fit on the first 110 days, judge the pool on the last 50 (never seen by the search)
    train = store.view(end=store.dates[109])
    test = store.view(start=store.dates[110])
    target_train = Label("close", 1).compute(train)
    pool = LinearPool(capacity=4, target=target_train)
    miner = GPMiner(default_registry(), train, target_train, pool, population_size=24, max_depth=3, seed=7)
    rep = miner.mine(240)
    assert pool.size >= 1 and rep.best_fitness > 0.2
    target_test = Label("close", 1).compute(test)
    from alphacube.engine.engine import CubeEngine
    from alphacube.metrics import zscore_rows
    panels = CubeEngine(test).evaluate(pool.exprs()).panels
    combined = sum(w * np.nan_to_num(zscore_rows(p), nan=0.0) for w, p in zip(pool.weights, panels))
    oos = nanmean_quiet(cs_pearson(combined, target_test))
    assert oos > 0.15, (oos, pool.summary())
