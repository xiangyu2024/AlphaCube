"""sweep 汇总表:按训练段挑格,测试段只报数;另报全部格子的中位和范围。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sweep_table                                           # noqa: E402


def _cell(name, tr, te):
    return {"name": name, "halflife": 5.0, "kw": {},
            "train": {"ex_sharpe": tr}, "test": {"ex_sharpe": te, "ex_ann": te / 10, "fees_yr": 0.01}}


def test_picks_on_train_reports_test_and_spread(tmp_path, capsys):
    p = tmp_path / "sweep_x_ridge.json"
    p.write_text(json.dumps({"stage1": [_cell("a", 1.0, 0.9), _cell("b", 2.0, -0.5), _cell("c", 0.5, 0.1)]}),
                 encoding="utf-8")
    r = sweep_table.summarize(str(p))
    assert r["name"] == "x_ridge" and r["train"] == 2.0 and r["test"] == -0.5      # 训练段最好那格,不看测试段
    assert r["median"] == 0.1 and (r["lo"], r["hi"]) == (-0.5, 0.9)
    assert sweep_table.summarize(str(tmp_path / "missing.json")) is None
    assert sweep_table.main([str(p), str(tmp_path / "missing.json")]) == 0
    assert "(没有)" in capsys.readouterr().out


def test_pick_uses_train_only(tmp_path, capsys):
    a, b = tmp_path / "sweep_a.json", tmp_path / "sweep_b.json"
    a.write_text(json.dumps({"stage1": [_cell("x", 3.0, -1.0)]}), encoding="utf-8")      # 训练好、测试差
    b.write_text(json.dumps({"stage1": [_cell("y", 1.0, 2.0)]}), encoding="utf-8")       # 训练差、测试好
    assert sweep_table.pick([sweep_table.summarize(str(a)), sweep_table.summarize(str(b))])["name"] == "a"
    assert sweep_table.main(["--pick", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "按训练段挑出:a" in out and out.count("<- 按训练段挑出") == 1
