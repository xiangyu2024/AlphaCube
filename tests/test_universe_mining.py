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


def test_cpu_quota_respects_the_thread_cap(tmp_path, monkeypatch):
    from alphacube.mining.gp import cpu_quota
    f = tmp_path / "cpu.max"
    f.write_text("2500000 100000\n")
    monkeypatch.delenv("ALPHACUBE_CPU_THREADS", raising=False)
    full = cpu_quota(str(f))
    monkeypatch.setenv("ALPHACUBE_CPU_THREADS", "3")
    assert cpu_quota(str(f)) == min(3, full)
    f.write_text("max 100000\n")
    assert cpu_quota(str(f)) == 3                            # 没有配额时就用这个上限
    monkeypatch.setenv("ALPHACUBE_CPU_THREADS", "junk")
    assert cpu_quota(str(f)) is None


def test_gp_panels_are_masked_to_the_universe(tmp_path):
    """GPMiner 自己建引擎:这条锁住 run_mining 把它也套上股票池那层(冒烟测试只看了只数,没发现)。"""
    st = write_synthetic_store(tmp_path / "cube", n_symbols=20, n_days=90, n_minutes=24, seed=8, plant_beta=1.0)
    mask = np.zeros((len(st.dates), len(st.symbols)), dtype=bool)
    mask[:, :8] = True
    mask[45:, 8] = True
    mask[60:, 0] = False                                     # 第 0 只后来出池
    np.save(tmp_path / "u.npy", mask)
    (tmp_path / "u.json").write_text(json.dumps({"dates": list(st.dates), "symbols": list(st.symbols)}))
    r = run_mining(st, start=st.dates[5], label="close", miner="gp", budget=24, pool_capacity=3,
                   gp_params=dict(population_size=12, max_depth=3), combine_methods=("equal", "ridge"),
                   universe=tmp_path / "u.npy", out_root=tmp_path / "runs", run_id="g", log=lambda s: None)
    member = mask[5:, :9]                                    # 视图:第 5 天起,进过池子的 9 只
    vals = sorted((r.run_dir / "values").glob("*.npy"))
    assert vals, "no factor values saved"
    for f in vals:
        v = np.load(f)
        assert v.shape == member.shape, (f.name, v.shape)
        assert not np.isfinite(v[~member]).any(), f"{f.name} has values outside the universe"
    comb = np.load(r.run_dir / "combined.npy")
    assert not np.isfinite(comb[~member]).any()


def test_pool_combined_is_nan_where_no_member_has_a_value():
    from alphacube.mining.pool import MsePool
    target = np.random.default_rng(0).normal(size=(30, 6))
    p = MsePool(3, target, dedup_threshold=None)
    z = np.random.default_rng(1).normal(size=(30, 6))
    z[:, 5] = np.nan
    p.members = [SimpleNamespace(z=z, expr="x")]
    p.weights = np.array([1.0])
    out = p.combined()
    assert np.isnan(out[:, 5]).all() and np.isfinite(out[:, :5]).all()


def test_model_combinations_leave_cells_without_any_member_empty():
    from alphacube.combine.post_combine import combine
    rng = np.random.default_rng(0)
    D, S = 120, 30
    cols = [rng.normal(size=(D, S)) for _ in range(3)]
    for c in cols:
        c[:, -5:] = np.nan                                   # 最后 5 只:没有任何成员有值(像池外)
    target = 0.1 * cols[0] + rng.normal(0, 1, (D, S))
    res = combine("ridge", cols, target, target, slice(0, 80), slice(80, D))
    assert np.isnan(res.combined[:, -5:]).all() and np.isfinite(res.combined[:, :-5]).any()


def test_pool_reward_is_never_minus_inf_for_an_admitted_member():
    """合成信号的覆盖率按 z 分数算,会低于成员的原始覆盖率:池子奖励不能因此拿 -inf。"""
    from alphacube.mining.pool import MsePool
    from alphacube.mining.scoring import PanelScorer
    rng = np.random.default_rng(0)
    D, S = 200, 60
    panel = rng.normal(size=(D, S))
    panel[:, :24] = np.nan                                   # 40% 的票没值
    panel[150:] = 1.0                                        # 最后 1/4 的天截面没有离散度
    target = 0.3 * np.nan_to_num(panel) + rng.normal(size=(D, S))
    pool = MsePool(10, target, dedup_threshold=None)
    pool.try_add("only", panel)
    scorer = PanelScorer(target, kind="rank_ic", turnover_penalty=0.14, min_coverage=0.5)
    assert scorer.score(pool.combined())["fit"] == -np.inf    # 门槛照旧拒掉(这是给单个候选的)
    assert np.isfinite(scorer.score(pool.combined(), gated=False)["fit"])


def test_universe_exemplar_cache_never_touches_the_full_market_root(tmp_path):
    from alphacube.parser import parse
    st = write_synthetic_store(tmp_path / "cube", n_symbols=20, n_days=90, n_minutes=24, seed=8, plant_beta=1.0)
    mask = np.zeros((len(st.dates), len(st.symbols)), dtype=bool)
    mask[:, :10] = True
    np.save(tmp_path / "u.npy", mask)
    (tmp_path / "u.json").write_text(json.dumps({"dates": list(st.dates), "symbols": list(st.symbols)}))
    root = tmp_path / "cache"
    (root / "FULL_MARKET_VIEW").mkdir(parents=True)
    for miner in ("gp", "alphagen"):
        run_mining(st, start=st.dates[5], label="close", miner=miner, budget=12, pool_capacity=2,
                   gp_params=dict(population_size=12, max_depth=3, ban_rank_corr=0.95,
                                  ban_exemplars=[parse("day_mean(Close)", n_minutes=st.n_minutes)],
                                  exemplar_cache=str(root)),
                   combine_methods=("equal",), universe=tmp_path / "u.npy", log=lambda s: None)
        assert (root / "FULL_MARKET_VIEW").is_dir(), miner
