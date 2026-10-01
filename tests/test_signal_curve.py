"""分桶曲线:单调的信号给单调的曲线;每天的桶超额相对当天可买股票的等权。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import signal_curve as sc                                    # noqa: E402


def test_planted_monotone_signal_gives_monotone_buckets():
    rng = np.random.default_rng(0)
    D, S = 80, 1000
    score = rng.normal(size=(D, S))
    ret = 0.001 * score + rng.normal(0, 0.002, (D, S))
    curve = sc.bucket_curve(score, ret, np.ones((D, S), dtype=bool), edges=(0, 0.1, 0.5, 1.0))
    means = [m for m, _ in sc.summarize(curve)]
    assert means[0] > means[1] > means[2]
    # 相对当天等权:按人数加权的桶均值之和为 0
    np.testing.assert_allclose(0.1 * curve[:, 0] + 0.4 * curve[:, 1] + 0.5 * curve[:, 2], 0, atol=1e-12)


def test_ineligible_names_are_left_out_before_ranking():
    D, S = 3, 200
    score = np.tile(np.arange(S, dtype=float), (D, 1))
    ret = np.zeros((D, S))
    ret[:, -1] = 1.0                                        # 信号最高那只收益很高,但买不进
    ok = np.ones((D, S), dtype=bool)
    ok[:, -1] = False
    curve = sc.bucket_curve(score, ret, ok, edges=(0, 0.01, 1.0))
    np.testing.assert_allclose(curve, 0.0, atol=1e-12)
