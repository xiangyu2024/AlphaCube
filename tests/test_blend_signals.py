"""信号等权合成:逐日秩 z 分数平均,缺值不拖累另一个信号。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import blend_signals                                         # noqa: E402


def test_rank_z_average_and_missing_cells(tmp_path):
    a = np.array([[1.0, 2.0, 3.0, np.nan], [3.0, 2.0, 1.0, 0.0]])
    b = np.array([[10.0, 30.0, 20.0, 5.0], [np.nan, np.nan, np.nan, np.nan]])
    out = blend_signals.blend([a, b])
    assert out.shape == a.shape and np.isfinite(out).all()      # 只有一个信号有值的格子也给分
    assert np.argmax(out[0]) == 1 or np.argmax(out[0]) == 2
    assert np.argsort(out[1]).tolist() == [3, 2, 1, 0]           # 第 2 天只有 a,次序就是 a 的
    np.save(tmp_path / "a.npy", a)
    np.save(tmp_path / "b.npy", b)
    assert blend_signals.main(["--signal", str(tmp_path / "a.npy"), "--signal", str(tmp_path / "b.npy"),
                               "--out", str(tmp_path / "o.npy")]) == 0
    assert np.load(tmp_path / "o.npy").dtype == np.float32
