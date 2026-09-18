"""挖完之后的组合层:方向对齐 + 多方法样本外比较。

针对两个真实缺陷:
1. GP 用 |IC| 筛选(`fitness_signed=False`,明说"负 IC 是可反转信号"),但池组合
   时从不反转 —— 正负因子互相抵消,池 IC 会比单因子还差。
2. 上一版 `MseAlphaPool._fit_weights` 的 OLS 要求所有成员在同一格子同时有限,
   只要有一条常数/布尔因子就塌成 `1/n` 等权兜底。
"""

from __future__ import annotations

import numpy as np
import pytest

from alphacube.combine.post_combine import align_signs, compare, leaderboard

T, N, K = 400, 60, 6


@pytest.fixture(scope="module")
def planted():
    """一半因子故意反向 —— 正是 |IC| 筛选会留在池里的那种组合。"""
    rng = np.random.default_rng(0)
    future = rng.normal(0, 0.02, (T, N))
    cols = []
    for k in range(K):
        sign = 1.0 if k % 2 == 0 else -1.0
        cols.append(sign * future * 0.08 + rng.normal(0, 0.05, (T, N)))
    return cols, future


def test_signs_are_recovered(planted):
    cols, future = planted
    _aligned, signs = align_signs(cols, future, days=slice(0, int(T * 0.7)))
    assert list(signs) == [1.0 if k % 2 == 0 else -1.0 for k in range(K)]


def test_alignment_is_what_rescues_the_combination(planted):
    """不对齐时组合比单因子还差 —— 这就是用户看到的症状。"""
    cols, future = planted
    aligned, _ = compare(cols, future, future, methods=("equal",), train_frac=0.7, align=True)
    raw, _ = compare(cols, future, future, methods=("equal",), train_frac=0.7, align=False)
    assert aligned[0].metrics["ic_mean"] > 0.05
    assert abs(raw[0].metrics["ic_mean"]) < 0.02      # 抵消掉了
    assert aligned[0].metrics["icir"] > raw[0].metrics["icir"]


def test_degenerate_factor_does_not_collapse_everything(planted):
    """常数因子会让池的 OLS 塌成等权兜底;这里它只应拿到 0 权重。"""
    cols, future = planted
    cols = list(cols) + [np.full((T, N), 3.0)]
    results, signs = compare(cols, future, future, methods=("ic", "icir"), train_frac=0.7)
    assert len(signs) == K + 1
    for result in results:
        assert result.weights is not None
        assert result.weights[-1] == pytest.approx(0.0)   # 退化因子权重为 0
        assert np.isfinite(result.metrics["ic_mean"])


def test_metrics_are_out_of_sample(planted):
    """权重/模型只用训练段;换掉测试段的目标,指标必须跟着变。"""
    cols, future = planted
    split = int(T * 0.7)
    tampered = future.copy()
    tampered[split:] = np.random.default_rng(9).normal(0, 0.02, (T - split, N))
    base, _ = compare(cols, future, future, methods=("ic",), train_frac=0.7)
    shifted, _ = compare(cols, tampered, tampered, methods=("ic",), train_frac=0.7)
    assert base[0].metrics["ic_mean"] > 0.05
    assert abs(shifted[0].metrics["ic_mean"]) < 0.02


def test_model_method_runs_and_reports(planted):
    cols, future = planted
    results, _ = compare(cols, future, future, methods=("gbdt",),
                         train_frac=0.7, max_fit_rows=50_000)
    assert "error" not in results[0].metrics, results[0].metrics.get("error")
    assert results[0].weights is None          # 模型没有线性权重
    assert np.isfinite(results[0].metrics["ic_mean"])


def test_missing_optional_dependency_is_reported_not_raised(planted):
    cols, future = planted
    results, _ = compare(cols, future, future, methods=("definitely_not_installed",))
    assert "error" in results[0].metrics


def test_leaderboard_sorts_by_icir(planted):
    cols, future = planted
    results, _ = compare(cols, future, future, methods=("equal", "ic", "lstsq"), train_frac=0.7)
    table = leaderboard(results)
    icirs = table["icir"].to_numpy()
    assert np.all(np.diff(icirs) <= 1e-12)


# --------------------------------------------------------------------------
# factorgcl:超图 GNN 作为另一种组合方案
# --------------------------------------------------------------------------

def test_factorgcl_runs_and_is_out_of_sample(planted):
    cols, future = planted
    results, _ = compare(cols, future, future, methods=("factorgcl",), train_frac=0.7,
                         factorgcl_kwargs={"epochs": 3, "max_days_per_epoch": 20})
    result = results[0]
    assert "error" not in result.metrics, result.metrics.get("error")
    assert result.weights is None                 # 学出来的是非线性组合,没有线性权重
    assert np.isfinite(result.metrics["ic_mean"])
    assert result.combined.shape == future.shape


def test_factorgcl_only_trains_on_the_training_slice(planted):
    """训练段之外的天必须没被优化器看到 —— 否则样本外指标是假的。"""
    from alphacube.combine.factorgcl import FactorGCLCombiner

    cols, future = planted
    T = future.shape[0]
    split = int(T * 0.7)

    seen: list = []

    class _RecordingRng:
        """包住真 Generator —— Generator.choice 本身是只读属性,改不了。"""

        def __init__(self, inner):
            self._inner = inner

        def choice(self, days, **kwargs):
            picked = self._inner.choice(days, **kwargs)
            seen.extend(int(d) for d in np.atleast_1d(picked))
            return picked

        def __getattr__(self, name):
            return getattr(self._inner, name)

    combiner = FactorGCLCombiner(seed=0)
    combiner._rng = _RecordingRng(combiner._rng)
    combiner.fit(cols, future, train_days=slice(0, split), epochs=3)

    assert seen, "没有采样到任何训练日"
    assert max(seen) < split, f"优化器看到了第 {max(seen)} 天,训练段只到 {split}"


def test_build_prior_incidence_is_the_zscore_stack():
    """组合器的输入就是调用方手里的列逐日 z-score 后堆起来 —— 没有第二套语义。

    上一版项目里这条测试锁的是"传 prior 与走 (pool, data) 训练在同一份输入上"
    (`build_prior_incidence` 自己逐条求值成员表达式再 z-score)。现在表达式由
    立方体引擎在上游求值,组合器只认 (D, S) 面板,锁住的仍是同一件事:helper
    不做 z-score 堆叠之外的任何事。
    """
    from alphacube.combine.factorgcl.hypergraph import build_prior_incidence
    from alphacube.daily import synthetic_daily
    from alphacube.metrics import zscore_rows

    data = synthetic_daily(n_days=60, n_instruments=12, seed=3)
    panels = [data.get_field("Close"), data.get_field("Volume")]
    from_helper = build_prior_incidence(panels)
    from_columns = np.stack([zscore_rows(p) for p in panels], axis=-1)
    assert from_helper.shape == (60, 12, 2)
    np.testing.assert_allclose(from_helper, from_columns, rtol=0, atol=0, equal_nan=True)
    with pytest.raises(ValueError):
        build_prior_incidence([])


def test_costs_reach_every_combination(planted):
    """`compare` used to call backtest_metrics without costs, so every combination
    table -- the pipeline's final one and each rolling-block report -- carried only
    the no-cost Sharpe, whatever --cost-per-side the run was launched with. On
    gpu5's first block the model methods turned over 45-67% of the book a day
    against 18-19% for the simple weightings, which is the ranking costs overturn."""
    from alphacube.metrics import backtest_metrics, cross_sectional_pearson

    cols, future = planted
    methods = ("equal", "ic", "lstsq", "gbdt")
    gross, _ = compare(cols, future, future, methods=methods, train_frac=0.7)
    net, _ = compare(cols, future, future, methods=methods, train_frac=0.7,
                     cost_per_side=0.0015, turnover_cap=0.30)
    test = slice(int(T * 0.7), T)
    for g, r in zip(gross, net):
        assert "sharpe_net" not in g.metrics                        # no costs asked, none charged
        assert r.metrics["sharpe"] == g.metrics["sharpe"]           # the no-cost headline is unchanged
        ref = backtest_metrics(cross_sectional_pearson(r.combined[test], future[test]), r.combined[test],
                               future[test], cost_per_side=0.0015, turnover_cap=0.30)
        for k in ("sharpe_net", "annualized_return_net", "sharpe_capped", "turnover_capped"):
            assert r.metrics[k] == pytest.approx(ref[k], rel=1e-12), (r.method, k)
        assert r.metrics["sharpe_net"] < r.metrics["sharpe"], r.method
    table = leaderboard(net)
    assert {"sharpe_net", "sharpe_capped", "turnover"} <= set(table.columns)


def test_model_combinations_are_reproducible(planted):
    """Same inputs, same seed, two calls: the same combination. "gbdt" is
    HistGradientBoosting, which enables early stopping above 10k rows and draws the
    validation split at random; unseeded, the two calls gave out-of-sample Sharpe
    5.0 and 3.0 on this fixture."""
    cols, future = planted
    for method in ("gbdt", "ridge"):
        a, _ = compare(cols, future, future, methods=(method,), train_frac=0.7, seed=0)
        b, _ = compare(cols, future, future, methods=(method,), train_frac=0.7, seed=0)
        assert np.array_equal(a[0].combined, b[0].combined, equal_nan=True), method


def test_missing_members_are_neutral_not_contagious():
    """With many members nearly every cell has one missing, and the weighted sum
    used to be NaN there: all four weight methods came back NaN on gpu5's first
    block (145 members). A missing member counts as 0 -- the pool's own rule --
    and a cell is missing only when every member is. lstsq also stops needing a
    row where every member is present, which with many members is no row at all,
    and silently meant 1/n."""
    rng = np.random.default_rng(1)
    T2, N2, K2 = 300, 50, 12
    future = rng.normal(0, 0.02, (T2, N2))
    cols = []
    for k in range(K2):
        c = (future if k == 0 else 0.0) + rng.normal(0, 0.05, (T2, N2))
        c[rng.random((T2, N2)) < 0.5] = np.nan         # 0.5**12: ~3 train rows with every member present
        cols.append(c)
    all_missing = np.all([~np.isfinite(c) for c in cols], axis=0)
    results, _ = compare(cols, future, future, methods=("equal", "ic", "icir", "lstsq"), train_frac=0.7)
    test = slice(int(T2 * 0.7), T2)
    for r in results:
        assert np.isfinite(r.metrics["sharpe"]) and np.isfinite(r.metrics["ic_mean"]), r.method
        assert np.array_equal(np.isnan(r.combined), all_missing), r.method
        assert r.metrics["coverage"] == pytest.approx(1 - all_missing[test].mean())
    lstsq = next(r for r in results if r.method == "lstsq")
    assert not np.allclose(lstsq.weights, 1.0 / K2), "fell back to equal weights"
    assert abs(lstsq.weights[0]) > 3 * np.abs(lstsq.weights[1:]).max(), lstsq.weights   # found the planted one
