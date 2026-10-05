"""scripts/long_report.py:测试段的方向用训练段定的、不再挑;汇总按训练段的分数排。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import long_report as lr  # noqa: E402
from alphacube.mining.scoring import LongScorer  # noqa: E402


def _data(D=160, S=150, seed=0, flip_at=None):
    rng = np.random.default_rng(seed)
    x = np.zeros((D, S))
    for t in range(1, D):
        x[t] = 0.9 * x[t - 1] + rng.normal(size=S)
    beta = np.full(D, 0.01)
    if flip_at is not None:
        beta[flip_at:] = -0.01                              # 测试段关系反过来
    target = beta[:, None] * x + 0.02 * rng.normal(size=(D, S))
    return x + rng.normal(size=(D, S)), target


def _scorers(target, split):
    return (LongScorer(target[:split], min_coverage=0.8), LongScorer(target[split:], min_coverage=0.0))


def test_test_period_uses_the_train_orientation_and_never_reselects():
    split = 110
    sig, target = _data(flip_at=split)
    train, test = _scorers(target, split)
    r = lr.evaluate(sig, train, test, split)
    assert r["sign"] == 1.0 and r["train"]["long_sharpe"] > 0
    assert r["test"]["long_sharpe"] < 0, "测试段关系反了:方向不许跟着翻"
    assert test.score(sig[split:])["long_sign"] == -1.0      # 要是在测试段重新挑方向,会挑到另一头
    neg = lr.evaluate(-sig, train, test, split)              # 反着写的因子:训练段挑到 -1,成绩和正着写一样
    assert neg["sign"] == -1.0
    assert neg["test"]["long_sharpe"] == pytest.approx(r["test"]["long_sharpe"], rel=1e-9)
    assert neg["train_smooth"]["long_sharpe"] == pytest.approx(r["train_smooth"]["long_sharpe"], rel=1e-9)


def test_train_numbers_do_not_depend_on_the_test_period_and_smoothing_is_causal():
    split = 110
    sig, target = _data(seed=1)
    train, test = _scorers(target, split)
    a = lr.evaluate(sig, train, test, split)
    sig2 = sig.copy()
    sig2[split:] = np.random.default_rng(9).normal(size=sig2[split:].shape)
    b = lr.evaluate(sig2, train, test, split)
    for k in ("fit_train", "old_fit_train", "sign"):
        assert a[k] == b[k]
    assert a["train"] == b["train"] and a["train_smooth"] == b["train_smooth"]
    assert a["test"] != b["test"]
    assert a["old_fit_train"] == pytest.approx(abs(a["rank_ic_train"]) - 0.14 * a["ls_turnover_train"])


def test_gated_out_factors_carry_no_test_numbers_and_summary_ranks_by_train():
    split = 110
    sig, target = _data(seed=2)
    train, test = _scorers(target, split)
    const = lr.evaluate(np.ones_like(sig), train, test, split)
    assert const["gated_out"] and "test" not in const

    def row(fit, sharpe):
        part = {"long_sharpe": sharpe, "long_ann": sharpe / 10, "long_turnover": 0.1, "long_rank_ic": 0.01}
        return {"gated_out": False, "fit_train": fit, "old_fit_train": -fit, "train": part, "test": part,
                "train_smooth": part, "test_smooth": part}

    rows = [row(0.05, 2.0), row(0.04, 1.0), row(0.01, -1.0), {"gated_out": True, "fit_train": float("-inf")}]
    s = lr.summarize(rows, "fit_train", tops=(2,))
    assert s["前2"]["n"] == 2 and s["前2"]["test_sharpe"] == pytest.approx(1.5) and s["前2"]["test_pos"] == 1.0
    assert s["全部"]["n"] == 3 and s["全部"]["test_pos"] == pytest.approx(2 / 3)
    assert lr.summarize(rows, "old_fit_train", tops=(1,))["前1"]["test_sharpe"] == pytest.approx(-1.0)


def test_second_version_ranking_and_extra_test_columns():
    from alphacube.mining.scoring import LongHitScorer

    split = 110
    sig, target = _data(seed=3)
    train, test = _scorers(target, split)
    w = {k: 1.0 for k in lr.HIT_KEYS}
    hit_train = LongHitScorer(target[:split], min_coverage=0.8)
    hit_test = LongHitScorer(target[split:], min_coverage=0.0, weights=w)
    r = lr.evaluate(sig, train, test, split, hit_train, hit_test)
    assert r["hit_fit_train"] == pytest.approx(hit_train.score(sig[:split])["fit"]) and np.isfinite(r["hit_fit_train"])
    assert set(lr.HIT_KEYS) <= set(r["test_smooth"]) and not set(lr.HIT_KEYS) & set(r["test"])
    plain = lr.evaluate(sig, train, test, split)
    assert np.isnan(plain["hit_fit_train"]) and plain["test_smooth"]["long_sharpe"] == r["test_smooth"]["long_sharpe"]
    sig2 = sig.copy()
    sig2[split:] = np.random.default_rng(1).normal(size=sig2[split:].shape)      # 测试段换掉:训练段的排法不变
    assert lr.evaluate(sig2, train, test, split, hit_train, hit_test)["hit_fit_train"] == r["hit_fit_train"]
    s = lr.summarize([dict(r, gated_out=False)], "hit_fit_train", tops=(1,))
    assert "smooth_xs_top20" in s["全部"] and s["全部"]["n"] == 1
