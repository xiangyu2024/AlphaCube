"""scripts/style_exposure.py:风格面板是因果的、方向是"值越大越是预期做多的那一头";逐日秩相关和 scipy 对得上;
把因子对风格回归取残差以后,纯风格因子什么都不剩、风格以外的信号还在。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import long_report as lr  # noqa: E402
import style_exposure as se  # noqa: E402
from alphacube.mining.scoring import LongScorer  # noqa: E402


def _market(D=90, S=14, seed=0, holes=0.08):
    """三份输入:复权收盘价(随机游走)、不复权收盘价、开盘 15 分钟成交额;都带 NaN 洞(停牌)。"""
    rng = np.random.default_rng(seed)
    adj = 10 * np.exp(np.cumsum(0.02 * rng.normal(size=(D, S)), axis=0))
    raw = adj / rng.uniform(1.0, 6.0, size=S)
    amt = np.exp(rng.normal(16, 1, size=(D, S)))
    hole = rng.random((D, S)) < holes
    return np.where(hole, np.nan, adj), np.where(hole, np.nan, raw), np.where(hole, np.nan, amt)


def test_rolling_mean_std_matches_pandas_with_nan_and_min_periods():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(1)
    x = rng.normal(size=(70, 9))
    x[rng.random(x.shape) < 0.25] = np.nan
    x[:, 0] = np.nan                                         # 整列没值
    x[30:55, 1] = np.nan                                     # 一段连续缺失(窗口里有效值掉到 10 个以下)
    mean, std = se.rolling_mean_std(x, 20, 10)
    ref = pd.DataFrame(x).rolling(20, min_periods=10)
    np.testing.assert_allclose(mean, ref.mean().to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(std, ref.std().to_numpy(), rtol=1e-8, atol=1e-12, equal_nan=True)
    assert np.isnan(mean[:, 0]).all() and np.isnan(mean[45:55, 1]).all()


def test_style_values_signs_and_the_ten_valid_days_rule():
    D = 14
    t = np.arange(D)
    # 两只票:A 成交额小、价格低、收益来回大幅波动;B 成交额大、价格高、收益很平
    amt = np.stack([np.full(D, 100.0), np.full(D, 400.0)], axis=1)
    raw = np.stack([np.full(D, 5.0), np.full(D, 50.0)], axis=1)
    adj = np.stack([10 * (1.05 ** (t % 2)) * (1 - 0.0 * t), 10 * 1.001 ** t], axis=1)
    out = se.style_panels(adj, raw, amt, window=20, min_valid=10)
    assert set(out) == {"illiq", "lowvol", "lowprice"} and all(v.dtype == np.float32 for v in out.values())
    assert np.isnan(out["illiq"][:9]).all() and np.isfinite(out["illiq"][9:]).all()      # 第 10 个有效日起才有值
    assert np.isnan(out["lowvol"][:10]).all() and np.isfinite(out["lowvol"][10:]).all()  # 第一天没有收益,所以晚一天
    np.testing.assert_allclose(out["illiq"][12], [-np.log(100.0), -np.log(400.0)], rtol=1e-6)
    np.testing.assert_allclose(out["lowprice"][3], [-np.log(5.0), -np.log(50.0)], rtol=1e-6)
    assert (out["illiq"][12, 0] > out["illiq"][12, 1]) and (out["lowprice"][12, 0] > out["lowprice"][12, 1])
    assert out["lowvol"][12, 1] > out["lowvol"][12, 0]                                   # 平的那只"低波动"这一头
    r = np.diff(adj[:, 0]) / adj[:-1, 0]                                                 # r[k] 是第 k+1 行的收益
    assert out["lowvol"][12, 0] == pytest.approx(-r[:12].std(ddof=1), rel=1e-5)          # 第 12 行的窗口含当天,共 13 行、12 个收益


def test_nan_and_non_positive_inputs_are_dropped_not_averaged():
    D = 25
    amt = np.full((D, 1), 100.0)
    amt[3, 0], amt[4, 0], amt[5, 0] = 0.0, -5.0, np.nan             # 非正、负、缺失
    amt[6, 0] = np.inf
    raw = np.full((D, 1), 10.0)
    raw[20, 0] = -1.0
    adj = np.full((D, 1), 10.0)
    adj[12, 0] = np.nan
    out = se.style_panels(adj, raw, amt, window=20, min_valid=10)
    assert out["illiq"][24, 0] == pytest.approx(-np.log(100.0), rel=1e-6)    # 坏值不参与均值,不是当 0 平均
    assert np.isnan(out["lowprice"][20, 0]) and np.isfinite(out["lowprice"][21, 0])
    lv = se.daily_returns(adj)
    assert np.isnan(lv[12, 0]) and np.isnan(lv[13, 0]) and np.isfinite(lv[14, 0])      # 缺一天,前后两个收益都没有(不跨停牌)


@pytest.mark.parametrize("seed", [0, 1])
def test_style_panels_are_causal(seed):
    adj, raw, amt = _market(seed=seed)
    base = se.style_panels(adj, raw, amt)
    t0 = 50
    rng = np.random.default_rng(99)
    adj2, raw2, amt2 = (a.copy() for a in (adj, raw, amt))
    for a in (adj2, raw2, amt2):
        a[t0 + 1:] = np.exp(rng.normal(size=a[t0 + 1:].shape))        # 第 t0 天以后的数全换掉
        a[t0 + 5:t0 + 9] = np.nan
    new = se.style_panels(adj2, raw2, amt2)
    for k in base:
        np.testing.assert_array_equal(base[k][:t0 + 1], new[k][:t0 + 1])     # 到第 t0 天为止一个数都不许变
        assert not np.array_equal(base[k][t0 + 1:], new[k][t0 + 1:], equal_nan=True)
    assert np.isfinite(base["illiq"][t0 + 1:]).any()


def test_close_is_the_adjusted_price_and_raw_close_is_close_over_fac():
    rng = np.random.default_rng(3)
    D, S = 30, 6
    fac = np.exp(rng.uniform(0, 3, size=S)) * np.ones((D, 1))
    raw_vwap = rng.uniform(5, 50, size=(D, S))
    adj_vwap = raw_vwap * fac
    close = (adj_vwap * rng.uniform(0.97, 1.03, size=(D, S))).astype(np.float32)
    vol = rng.uniform(1e5, 1e6, size=(D, S)).astype(np.float32)
    adj_close, raw_close, amount = se.style_inputs(close, fac, raw_vwap.astype(np.float32), vol)
    np.testing.assert_allclose(adj_close, close.astype(np.float64))                  # 复权收盘价就是 close,不再乘 fac
    np.testing.assert_allclose(raw_close, close / fac, rtol=1e-12)
    np.testing.assert_allclose(amount, vol.astype(np.float64) * raw_vwap.astype(np.float32), rtol=1e-12)
    ok = se.check_price_space(close, adj_vwap, raw_vwap, fac)
    assert abs(ok["close_over_adj"] - 1) < 0.05 and ok["close_over_raw"] > 1.5
    with pytest.raises(SystemExit):                                                  # close 其实是不复权价:挡住
        se.check_price_space((raw_vwap * 1.01).astype(np.float32), adj_vwap, raw_vwap, fac)


def test_rank_avg_and_daily_rank_corr_match_scipy():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(4)
    v = np.round(rng.normal(size=200), 1)                                             # 大量并列
    np.testing.assert_allclose(se.rank_avg(v), stats.rankdata(v))
    assert se.rank_avg(np.array([])).size == 0
    D, S = 25, 60
    a = np.round(rng.normal(size=(D, S)), 1)
    b = 0.5 * a + np.round(rng.normal(size=(D, S)), 1)
    a[rng.random(a.shape) < 0.2] = np.nan
    b[rng.random(b.shape) < 0.2] = np.nan
    a[3, 20:] = np.nan                                                                # 共同的票 < 30 只
    a[4, :] = 1.0                                                                     # 一边没有离散度
    out = se.daily_rank_corr(a, b, min_common=30)
    for t in range(D):
        m = np.isfinite(a[t]) & np.isfinite(b[t])
        if t == 3:
            assert np.isnan(out[t]) and m.sum() < 30
        elif t == 4:
            assert np.isnan(out[t])
        else:
            assert out[t] == pytest.approx(stats.spearmanr(a[t][m], b[t][m]).statistic, abs=1e-12)
    assert se.daily_rank_corr(a, a, 30)[0] == pytest.approx(1.0)


def test_period_means_split_and_all_nan_parts():
    v = np.array([1.0, 3.0, np.nan, 10.0, np.nan, 20.0])
    assert se.period_means(v, 3) == (2.0, 15.0)
    tr, te = se.period_means(np.array([np.nan, np.nan, 1.0]), 2)
    assert np.isnan(tr) and te == 1.0


def _styles(D=40, S=70, seed=5, holes=0.1):
    rng = np.random.default_rng(seed)
    st = [rng.normal(size=(D, S)) for _ in range(3)]
    for x in st:
        x[rng.random(x.shape) < holes] = np.nan
    return st


def test_ols_residual_of_an_exact_combination_is_zero():
    rng = np.random.default_rng(6)
    n = 80
    cols = [se._zrank(rng.normal(size=n)) for _ in range(3)]
    y = 0.4 + 2.0 * cols[0] - 3.0 * cols[1] + 0.1 * cols[2]
    X = np.column_stack([np.ones(n)] + cols)
    e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.abs(e).max() < 1e-10


def test_neutralize_a_pure_style_factor_leaves_nothing():
    st = _styles()
    rng = np.random.default_rng(7)
    fac = np.exp(st[0] / 2)                                  # 风格 0 的单调变换:秩一模一样
    fac[rng.random(fac.shape) < 0.1] = np.nan                # 因子自己再多几个洞
    res, r2, used = se.neutralize(fac, st, min_common=30)
    done = np.isfinite(r2)
    assert done.all() and (used > 30).all()
    assert np.nanmax(np.abs(res)) < 1e-9 and np.nanmin(r2) > 1 - 1e-9
    assert np.isnan(res[~(np.isfinite(fac) & np.isfinite(st[0]) & np.isfinite(st[1]) & np.isfinite(st[2]))]).all()


def test_neutralize_residual_is_orthogonal_to_the_style_ranks_and_keeps_the_rest():
    st = _styles(seed=8)
    rng = np.random.default_rng(9)
    noise = rng.normal(size=st[0].shape)
    fac = st[0] + 0.5 * st[1] + noise                       # 风格的组合 + 风格以外的东西
    res, r2, used = se.neutralize(fac, st, min_common=30)
    for t in range(fac.shape[0]):
        m = np.isfinite(fac[t]) & np.isfinite(st[0][t]) & np.isfinite(st[1][t]) & np.isfinite(st[2][t])
        assert np.isfinite(res[t][m]).all() and np.isnan(res[t][~m]).all()
        for s in st:
            assert abs(np.dot(res[t][m], se._zrank(s[t][m]))) / m.sum() < 1e-9            # 和每个风格的秩正交
        assert abs(res[t][m].mean()) < 1e-9                                               # 带截距
        assert se.daily_rank_corr(res[t:t + 1], noise[t:t + 1], 30)[0] > 0.5              # 风格以外的部分还在
    assert 0.3 < np.nanmean(r2) < 0.9
    short = se.neutralize(fac[:, :25], [s[:, :25] for s in st], min_common=30)             # 共同的票太少:整行跳过
    assert np.isnan(short[0]).all() and np.isnan(short[1]).all() and (short[2] == 0).all()


def test_residual_of_a_pure_style_factor_scores_nothing_but_an_independent_signal_survives():
    D, S, split = 300, 150, 210
    rng = np.random.default_rng(10)

    def ar():
        x = np.zeros((D, S))
        for t in range(1, D):
            x[t] = 0.9 * x[t - 1] + rng.normal(size=S)
        return x

    st, other = [ar(), ar(), ar()], ar()
    target = 0.01 * st[0] + 0.01 * other + 0.02 * rng.normal(size=(D, S))
    train, test = LongScorer(target[:split], min_coverage=0.8), LongScorer(target[split:], min_coverage=0.0)
    pure, mixed = np.exp(st[0] / 4), st[0] + other + rng.normal(size=(D, S))
    ev_pure, ev_mixed = (lr.evaluate(f, train, test, split) for f in (pure, mixed))
    assert ev_pure["test"]["long_sharpe"] > 1.0 and ev_mixed["test"]["long_sharpe"] > 1.0       # 两个原因子都有用
    res_pure, _, _ = se.neutralize(pure, st, 30)
    res_mixed, _, _ = se.neutralize(mixed, st, 30)
    assert np.nanmax(np.abs(res_pure)) < 1e-9                                                  # 纯风格:什么都不剩
    ev_p = lr.evaluate(res_pure, train, test, split)
    assert ev_p["gated_out"] or ev_p["test"]["long_sharpe"] < 1.0                              # 剩下的是浮点噪声,打不出分
    ev_m = lr.evaluate(res_mixed, train, test, split)
    assert not ev_m["gated_out"] and ev_m["sign"] == 1.0 and ev_m["test"]["long_sharpe"] > 0.5  # 风格以外的信号还在


def test_select_top_and_summarize_group():
    def row(i, fit, lib="L", gated=False, hit=None):
        return {"lib": lib, "i": i, "gated_out": gated, "fit_train": fit, "hit_fit_train": hit}

    rows = [row(0, 0.02), row(1, 0.05), row(2, 0.05), row(3, float("-inf"), gated=True), row(4, 0.0), row(5, 0.9, lib="X"),
            row(6, 0.03, hit=0.5), row(7, 0.04, hit=0.7), row(8, 0.01, hit=None)]
    assert [r["i"] for r in se.select_top(rows, "L", "fit_train", 3)] == [1, 2, 7]          # 并列按 i 小的在前;门槛挡掉的、别的库的不要
    assert [r["i"] for r in se.select_top(rows, "L", "fit_train", 10)][-1] == 4              # 0.0 是有效分数,不是缺失
    assert [r["i"] for r in se.select_top(rows, "L", "hit_fit_train", 5)] == [7, 6]          # 没有这个分数的不要

    def fr(c, o_raw, o_sm, r_raw, r_sm, ru):
        z = {s: {"train": c, "test": c / 2} for s in se.STYLES}
        return {"corr": z, "resid_exposure": {s: {"train": 0.0, "test": 0.0} for s in se.STYLES}, "r2": {"train": 0.4, "test": 0.2},
                "orig": {"test_sharpe": o_raw, "test_sharpe_smooth": o_sm},
                "resid": {"test_sharpe": r_raw, "test_sharpe_smooth": r_sm, "gated_out": r_raw is None} if r_raw is not None
                else {"gated_out": True}, "resid_used": {"test_sharpe": ru}}

    out = se.summarize_group([fr(0.2, 1.0, 2.0, 0.5, 1.0, 0.0), fr(0.4, 2.0, 3.0, -1.0, 2.0, -0.5), fr(0.6, 3.0, 4.0, None, None, 0.7)])
    assert out["n"] == 3 and out["corr"]["illiq"] == {"train": pytest.approx(0.4), "test": pytest.approx(0.2)}
    assert out["med_orig_raw"] == 2.0 and out["pos_orig_smooth"] == 3
    assert out["med_resid_raw"] == pytest.approx(-0.25) and out["pos_resid_raw"] == 1 and out["valid_resid_raw"] == 2   # 被挡掉的不进中位数
    assert out["med_resid_smooth"] == pytest.approx(1.5) and out["pos_resid_smooth"] == 2
    assert out["med_residused_raw"] == 0.0 and out["pos_residused_raw"] == 1 and out["r2"] == {"train": 0.4, "test": 0.2}
