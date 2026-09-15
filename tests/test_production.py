"""Production calculation: resumable batches, status catalog, deferred-resource
policy, Parquet export; Feature Boosting pre-squeeze; pipeline.run_mining;
composition of a two-axis factor through face switching."""

import json

import numpy as np
import pandas as pd
import pytest

from alphacube import operators as ops
from alphacube.calculate import deferred_resource_factors, export_run_to_parquet, run_factor_calculation
from alphacube.engine.engine import CubeEngine
from alphacube.label import Label
from alphacube.mining.boosting import PresqueezeConfig, feature_boosting_presqueeze, seed_pool_from_presqueeze
from alphacube.mining.pool import MsePool
from alphacube.parser import parse
from alphacube.pipeline import run_mining
from alphacube.synthetic import write_synthetic_store

EXPRS = {
    "a": "day_std(in_mean_rolling(Close, 5))",
    "b": "cs_rank(ts_mean(day_std(Ret), 5))",
    "c": "day_mean(nt_cs_rank(Close))",
    "risky": "ts_pearson(day_last(Close), 5)",
    "uses_risky": "ts_mean(risky, 3)",
    "bad": "in_mean_rolling(day_mean(Close), 5)",
    "wide": "in_mean_rolling(Close, 5)",
    "d": "div(day_max(Close), close_1d)",
}


def test_deferred_policy():
    d = deferred_resource_factors(EXPRS)
    assert set(d) == {"risky", "uses_risky"} and "ts_pearson" in d["uses_risky"]


def test_calculation_resumes_and_matches(store, tmp_path):
    root = tmp_path / "runs"
    partial = run_factor_calculation(EXPRS, store, save_root=root, run_id="r", factor_batch_size=2,
                                     max_batches=1, quiet=True, start=store.dates[10])
    state = json.loads((root / "r" / "factor_results.json").read_text())
    assert state["run_status"] == "IN_PROGRESS" and not (root / "r" / "meta.json").exists()
    done_first = {n for n, r in state["rows"].items() if r["status"] == "SUCCESS"}
    assert len(done_first) == 2
    full = run_factor_calculation(EXPRS, store, save_root=root, run_id="r", factor_batch_size=2, quiet=True,
                                  start=store.dates[10])
    assert full.loc["a", "status"] == "SUCCESS" and full.loc["bad", "status"] == "PARSE_FAILED"
    assert full.loc["wide", "status"] == "NOT_DAILY" and full.loc["risky", "status"] == "DEFERRED_RESOURCE_RISK"
    assert full.loc["uses_risky", "status"] == "DEFERRED_RESOURCE_RISK"
    meta = json.loads((root / "r" / "meta.json").read_text())
    assert meta["run_status"] == "COMPLETE" and meta["status_counts"]["SUCCESS"] == 4
    fresh = run_factor_calculation(EXPRS, store, save_root=root, run_id="fresh", factor_batch_size=8, quiet=True,
                                   start=store.dates[10])
    for name in ("a", "b", "c", "d"):
        x = np.load(root / "r" / "values" / full.loc[name, "file"])
        y = np.load(root / "fresh" / "values" / fresh.loc[name, "file"])
        assert np.array_equal(x, y, equal_nan=True)                 # resumed run == uninterrupted run
        assert x.shape == (store.n_days - 10, store.n_symbols)
    allowed = run_factor_calculation({"risky": EXPRS["risky"]}, store, save_root=root, run_id="ok",
                                     defer_resource_risk=False, quiet=True)
    assert allowed.loc["risky", "status"] == "SUCCESS"
    info = export_run_to_parquet(root / "r", tmp_path / "pq", store=store)
    assert info["factors_completed"] == 4 and info["files_written"] == 4 * (store.n_days - 10)
    day = meta["dates"][5]
    frame = pd.read_parquet(tmp_path / "pq" / day / "a.parquet")
    assert list(frame.columns) == ["date", "symbol", "a"] and frame["date"].iloc[0] == f"{day[:4]}-{day[4:6]}-{day[6:]}"
    status = store.read_status(slice(0, store.n_symbols), 0, store.n_days)
    assert len(frame) == int((status[:, 5 + 10] != 0).sum())        # outside-lifecycle rows dropped
    again = export_run_to_parquet(root / "r", tmp_path / "pq", store=store)
    assert again["files_written"] == info["files_written"]           # manifest: nothing re-exported


def test_feature_boosting_finds_planted_signal(registry, tmp_path):
    st = write_synthetic_store(tmp_path / "cube", n_symbols=30, n_days=120, n_minutes=24, seed=5, plant_beta=1.0)
    view = st.view()
    target = Label("close", 1).compute(view)
    engine = CubeEngine(view)
    # steer the O(n) pairing at the planted field group and the day-reducers so the
    # test is about scoring, not about the random draw of groups
    cfg = PresqueezeConfig(n_iterations=3, top_k_per_iteration=4, candidates_per_iteration_cap=80, seed=1,
                           field_groups={"return": ["Ret"]}, pairing="heuristic",
                           heuristic_affinities={"return": ["dt_day", "dt_in"]})
    res = feature_boosting_presqueeze(registry, engine, target, cfg)
    assert res.iterations_run == 3 and res.candidates_evaluated_total > 0 and res.survivors
    assert all(e.shape.degenerate for e in res.survivors) and len(res.importances) == len(res.survivors)
    assert any(repr(e) in ("day_std(Ret)", "day_var(Ret)") for e in res.survivors), [repr(e) for e in res.survivors]
    pool = MsePool(4, target)
    rewards = seed_pool_from_presqueeze(pool, res, engine)
    assert pool.size >= 1 and pool.pool_ic() > 0.15
    assert res.per_iteration_log[0]["operator_group"] in ("dt_day", "dt_in")


def test_run_mining_pipeline(tmp_path):
    st = write_synthetic_store(tmp_path / "cube", n_symbols=20, n_days=90, n_minutes=24, seed=8, plant_beta=1.0)
    logs = []
    r = run_mining(st, start=st.dates[5], label="close", miner="gp", budget=36, pool_capacity=4,
                   gp_params=dict(population_size=12, max_depth=3), combine_methods=("equal", "ic"),
                   out_root=tmp_path / "runs", run_id="p", log=logs.append)
    assert r.pool_size >= 1 and r.run_dir == tmp_path / "runs" / "p" and (r.run_dir / "members.csv").exists()
    assert r.combination is not None and {"equal", "ic"} <= set(r.combination.index)
    assert any("pool:" in s for s in logs)
    r2 = run_mining(st, start=st.dates[5], label="volatility", horizon=5, miner="alphasage", budget=16,
                    pool="dynamic", pool_capacity=3, rl_params=dict(trajectories_per_update=8), max_expr_len=12,
                    combine_methods=(), log=None)
    assert r2.summary.startswith("16 episodes")


def test_two_axis_factor_by_composition(view, raw):
    """Intraday market beta: stock minute returns regressed on the cross-sectional
    mean at each minute -- two axes (S then M) in ONE D-round, no new kernel."""
    cubes, trad = raw
    S, D, M = view.n_symbols, view.n_days, view.n_minutes
    e = parse("day_last(in_regbeta(Ret, nt_cs_mean(Ret), 10))")
    eng = CubeEngine(view, chunk_days=7)
    plan = eng.plan([e])
    assert [r.mode for r in plan.rounds] == ["D"] and not plan.boundaries
    res = eng.evaluate([e])
    import torch
    x = torch.from_numpy(np.ascontiguousarray(cubes["Ret"]))
    mkt = torch.empty_like(x)
    for d in range(D):
        mkt[:, d, :] = ops.nd_cs.cs_mean(x[:, d, :].T.contiguous()).T
    m = torch.from_numpy(trad).unsqueeze(-1)
    nan = torch.full((), float("nan"))
    mkt = torch.where(m, mkt, nan)
    beta = ops.dt_in.in_regbeta(x.reshape(S * D, M), mkt.reshape(S * D, M), 10).reshape(S, D, M)
    last = ops.dt_day.day_last(torch.where(m, beta, nan).reshape(S * D, M)).reshape(S, D)
    ref = np.where(trad, last.numpy(), np.nan).T
    assert np.array_equal(ref, res.panels[0], equal_nan=True)
