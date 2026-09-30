"""全因子评估的面板复用:序号会因为新 run 的加入整体错位。

面板文件名就是序号,`collect()` 按目录名字母序遍历(gpu2 排在 gpu17 后面),
所以加三个新 run 会把 gpu2~gpu9 的序号全推后。按序号判断"已经算过"就会张冠
李戴 —— 而重算 106 GB 的面板是十小时 GPU。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_all_factors import remap_existing        # noqa: E402


def _setup(tmp_path, old):
    (tmp_path / "panels").mkdir()
    for i, (_run, expr) in enumerate(old):
        np.save(tmp_path / "panels" / f"{i:05d}.npy", np.full((2, 2), i, dtype=np.float32))
    with open(tmp_path / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["i", "run", "expression"])
        for i, (run, expr) in enumerate(old):
            w.writerow([i, run, expr])


def _value(tmp_path, i):
    p = tmp_path / "panels" / f"{i:05d}.npy"
    return float(np.load(p)[0, 0]) if p.exists() else None


def test_panels_follow_their_expression_when_indices_shift(tmp_path):
    _setup(tmp_path, [("gpu2", "A"), ("gpu2", "B"), ("gpu9", "C")])
    new = [("gpu15", "D", None), ("gpu2", "A", None), ("gpu2", "B", None), ("gpu9", "C", None)]
    assert remap_existing(tmp_path, new) == 3
    assert [_value(tmp_path, i) for i in range(4)] == [None, 0.0, 1.0, 2.0]   # D 没算过,A/B/C 跟着走
    assert not (tmp_path / "panels_previous").exists()


def test_no_shift_is_a_no_op(tmp_path):
    _setup(tmp_path, [("gpu2", "A"), ("gpu2", "B")])
    same = [("gpu2", "A", None), ("gpu2", "B", None), ("gpu9", "C", None)]
    assert remap_existing(tmp_path, same) == 0
    assert [_value(tmp_path, i) for i in range(3)] == [0.0, 1.0, None]


def test_a_vanished_expression_keeps_its_panel_for_inspection(tmp_path):
    _setup(tmp_path, [("gpu2", "A"), ("gpu2", "GONE"), ("gpu9", "C")])
    new = [("gpu15", "D", None), ("gpu2", "A", None), ("gpu9", "C", None)]
    assert remap_existing(tmp_path, new) == 2
    left = list((tmp_path / "panels_previous").iterdir())
    assert [p.name for p in left] == ["00001.npy"]      # 不静默删,留着让人看
