"""族代表 -> RL 池子种子 yaml:每族第一个、按族顺序,load_seed_library 读得回来。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import seed_yaml                                             # noqa: E402
from alphacube.legacy import load_seed_library               # noqa: E402


def test_one_representative_per_family_in_family_order(tmp_path):
    (tmp_path / "m.csv").write_text(
        "i,run,expression,name\n0,legacy,day_std(Ret),a\n1,legacy,day_mean(Ret),b\n2,legacy,day_max(Ret),c\n",
        encoding="utf-8")
    (tmp_path / "f.json").write_text(json.dumps([{"members": [2, 0]}, {"members": [1]}]), encoding="utf-8")
    out = tmp_path / "s.yaml"
    assert seed_yaml.main(["--families", str(tmp_path / "f.json"), "--manifest", str(tmp_path / "m.csv"),
                           "--top", "5", "--out", str(out)]) == 0
    seeds, errors = load_seed_library(out)
    assert not errors and [n for n, _ in seeds] == ["c", "b"]
    assert [repr(e) for _, e in seeds] == ["day_max(Ret)", "day_mean(Ret)"]
