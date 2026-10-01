"""股票池:掩码形状检查、只看池子的训练段指纹、回测只在池子里选股。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_champion as bc                                   # noqa: E402
import tradable_backtest as tb                                # noqa: E402
from universe import load_universe                            # noqa: E402


def test_load_universe_checks_shape_and_dtype(tmp_path):
    p = tmp_path / "u.npy"
    np.save(p, np.ones((3, 4), dtype=bool))
    assert load_universe(str(p), (3, 4)).all()
    assert load_universe("", (3, 4)) is None
    with pytest.raises(SystemExit):
        load_universe(str(p), (3, 5))
    np.save(p, np.ones((3, 4)))
    with pytest.raises(SystemExit):
        load_universe(str(p), (3, 4))                         # 不是 bool


def test_universe_fingerprints_only_see_members(tmp_path):
    rng = np.random.default_rng(0)
    D, S = 12, 6
    (tmp_path / "panels").mkdir()
    a = rng.normal(size=(D, S))
    np.save(tmp_path / "panels" / "00000.npy", a.astype(np.float32))
    b = a.copy()
    b[:, 4:] = rng.normal(size=(D, 2)) * 100                  # 只在池外不一样
    np.save(tmp_path / "panels" / "00001.npy", b.astype(np.float32))
    u = np.zeros((D, S), dtype=bool)
    u[:, :4] = True
    fp = bc.universe_fingerprints(tmp_path / "panels", [0, 1], slice(2, 10), u, tmp_path / "fp.npy")
    assert fp.shape == (2, 8 * 4)                             # 列 = 窗口里进过池子的 4 只
    np.testing.assert_allclose(fp[0], fp[1], atol=1e-6)       # 池外的差别看不见


def test_backtest_never_buys_outside_the_universe():
    D, S = 6, 6
    m = tb.Market(dates=[f"202401{d + 1:02d}" for d in range(D)], adj=np.full((D, S), 10.0),
                  raw=np.full((D, S), 10.0), vol15=np.full((D, S), 1e9), up_lock=np.zeros((D, S), bool),
                  down_lock=np.zeros((D, S), bool))
    score = np.tile(np.arange(S, 0, -1, dtype=float), (D, 1))  # 第 0 只信号最高
    u = np.zeros((D, S), dtype=bool)
    u[:, 3:] = True
    score[~u] = np.nan                                        # 池外 NaN = 不进排名
    res = tb.simulate(m, score, 0, 3, capital=10_000, cap=0.5, pool_frac=1.0, costs=tb.Costs("零", rate=0.0),
                      keep_units=True)
    assert (res["units"][:, :3] == 0).all() and (res["units"][:, 3:] > 0).any()
