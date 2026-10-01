"""带股票池的挖掘:掩码按名字对齐、引擎面板池外置空、端到端只在池子里挖。"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from alphacube.pipeline import UniverseEngine, align_universe, make_label, run_mining
from alphacube.synthetic import write_synthetic_store


def test_align_universe_by_name_not_position():
    mask = np.array([[True, False, True], [False, True, True]])
    out = align_universe(mask, ["d1", "d2"], ["a", "b", "c"], ["d0", "d2", "d1"], ["c", "a", "zz"])
    # 视图日 d0 不在掩码里 -> 全池外;股票 zz 不在掩码里 -> 池外
    np.testing.assert_array_equal(out, [[False, False, False], [True, False, False], [True, True, False]])


def test_universe_engine_masks_panels_and_forwards_everything_else():
    member = np.array([[True, False], [False, True]])

    class Fake:
        view = "the-view"

        def evaluate(self, exprs, tolerant=False):
            return SimpleNamespace(panels=[np.ones((2, 2), np.float32), None], errors={})

    e = UniverseEngine(Fake(), member)
    res = e.evaluate(["x", "y"], tolerant=True)
    np.testing.assert_array_equal(np.isnan(res.panels[0]), ~member)
    assert res.panels[0].dtype == np.float32 and res.panels[1] is None
    assert e.view == "the-view"


def test_buyable_label_kind():
    assert "limit-up" in make_label("open_vwap15_buyable", 1).describe()


def test_mining_runs_inside_the_universe(tmp_path):
    st = write_synthetic_store(tmp_path / "cube", n_symbols=20, n_days=90, n_minutes=24, seed=8, plant_beta=1.0)
    mask = np.zeros((len(st.dates), len(st.symbols)), dtype=bool)
    mask[:, :8] = True
    mask[45:, 8] = True                                     # 第 8 只后半段才进池子
    np.save(tmp_path / "u.npy", mask)
    (tmp_path / "u.json").write_text(json.dumps({"dates": list(st.dates), "symbols": list(st.symbols)}))
    logs = []
    r = run_mining(st, start=st.dates[5], label="close", miner="gp", budget=24, pool_capacity=3,
                   gp_params=dict(population_size=12, max_depth=3), combine_methods=("equal",),
                   universe=tmp_path / "u.npy", log=logs.append)
    line = [s for s in logs if s.startswith("universe ")]
    assert line and "9 symbols ever in" in line[0], logs[:3]
    assert any(" x 9 symbols" in s for s in logs)            # 引擎只在 9 只票上算
    assert r.pool_size >= 0
    # 只挖前半段:第 8 只后半段才进池子,不该出现在引擎里
    logs = []
    run_mining(st, start=st.dates[5], end=st.dates[40], label="close", miner="gp", budget=12, pool_capacity=2,
               gp_params=dict(population_size=12, max_depth=3), combine_methods=("equal",),
               universe=tmp_path / "u.npy", log=logs.append)
    assert any("8 symbols ever in" in s for s in logs), logs[:3]
