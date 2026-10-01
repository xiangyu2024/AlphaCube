"""只为纯多头拟合族间权重:全截面要还原出埋进去的权重;条件拟合只用头部样本。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import long_side_fit as lf                                   # noqa: E402


def test_full_cross_section_recovers_planted_weights():
    rng = np.random.default_rng(0)
    fams = rng.normal(size=(4, 40, 80)).astype(np.float32)
    y = fams[0] * 0.5 + fams[1] * 0.2 + rng.normal(0, 0.1, (40, 80))
    w = lf.fit_between(fams, y, y, range(40))
    np.testing.assert_allclose(w, [0.5, 0.2, 0, 0], atol=0.02)


def test_conditional_fit_uses_only_the_top_of_the_base_ranking():
    rng = np.random.default_rng(1)
    fams = rng.normal(size=(2, 30, 100)).astype(np.float32)
    base = fams[0].astype(np.float64)                       # 头部 = 第 0 个族因子高的那些
    # 头部里只有第 1 个族因子有用;尾部里第 1 个族因子的方向反过来
    top = np.stack([lf.day_ranks(base[t]) <= 0.3 for t in range(30)])
    y = np.where(top, 1.0, -1.0) * fams[1] + rng.normal(0, 0.05, (30, 100))
    w_all = lf.fit_between(fams, y, base, range(30))
    w_top = lf.fit_between(fams, y, base, range(30), top=0.3)
    assert w_top[1] > 0.9 and abs(w_all[1]) < 0.5


def test_rank_z_is_increasing_and_keeps_nan():
    z = lf.rank_z(np.array([3.0, 1.0, np.nan, 2.0]))
    assert np.isnan(z[2]) and z[0] > z[3] > z[1]
    np.testing.assert_allclose(np.nanmean(z), 0.0, atol=1e-12)


def test_missing_label_counts_as_zero_like_fit_weights():
    fams = np.ones((1, 3, 20), dtype=np.float32)            # 每天至少 10 个样本才进回归
    y = np.array([[1.0, np.nan] * 10] * 3)
    w = lf.fit_between(fams, y, y, range(3))
    np.testing.assert_allclose(w, [0.5])                     # 缺失当 0:均值 0.5,不是 1


def test_two_stage_keeps_the_screen_on_top_and_reorders_inside():
    base = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
    inner = np.array([[0.0, 9.0, 1.0, 100.0, 100.0]])        # 头部外面的票内层分数再高也不进前排
    out = lf.two_stage(base, inner, top=0.6)                # 前 60% = 前 3 只
    order = np.argsort(-out[0])
    assert order.tolist() == [1, 2, 0, 3, 4]                # 头部 3 只按内层排,其余按 base
