"""原始手写因子库的求值脚本:因子清单(去重、日频根、重算子排最后)和合成立方体上的端到端 + 续跑。"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eval_legacy_factors as ev                             # noqa: E402
from alphacube.catalog import RESOURCE_RISK_KERNELS          # noqa: E402
from alphacube.synthetic import write_synthetic_store        # noqa: E402


def _risky(e):
    return any(n.spec is not None and n.spec.name.removesuffix("_legacy") in RESOURCE_RISK_KERNELS for n in e.nodes())


def test_items_are_daily_unique_and_heavy_last():
    items, failed = ev.legacy_items(241)
    assert 1400 <= len(items) <= 1429
    assert len(items) + len(failed) == 1498
    assert all(e.shape.degenerate for _, _, e in items)
    assert len({e.key() for _, _, e in items}) == len(items)
    flags = [_risky(e) for _, _, e in items]
    assert sum(flags) == 16 and flags == sorted(flags)     # 重算子全在尾巴上


def test_main_writes_panels_and_resumes(tmp_path):
    root = tmp_path / "cube"
    write_synthetic_store(root, n_symbols=5, n_days=45, n_minutes=241, seed=1)
    out = tmp_path / "legacy"
    args = [str(out), "--store", str(root), "--mirror-root", "", "--start", "", "--end", "", "--warmup", "0",
            "--device", "cpu", "--limit", "12", "--batch", "5"]
    assert ev.main(args) == 0
    rows = list(csv.DictReader(open(out / "manifest.csv", encoding="utf-8")))
    assert [int(r["i"]) for r in rows] == list(range(12)) and all(r["run"] == "legacy" for r in rows)
    panels = sorted((out / "panels").glob("*.npy"))
    assert len(panels) >= 10
    p = np.load(panels[0])
    assert p.dtype == np.float32 and p.shape == (45, 5)
    assert len(np.load(out / "dates.npy")) == 45
    mtimes = {q: q.stat().st_mtime_ns for q in panels}
    assert ev.main(args) == 0                                  # 续跑:已有的不重算
    assert all(q.stat().st_mtime_ns == t for q, t in mtimes.items())


def test_parity_mode_compares_devices_in_batches(tmp_path, monkeypatch, capsys):
    root = tmp_path / "cube"
    write_synthetic_store(root, n_symbols=5, n_days=45, n_minutes=241, seed=1)
    import alphacube.engine.engine as eng
    real = eng.CubeEngine
    monkeypatch.setattr(eng, "CubeEngine", lambda view, device="cpu", **kw: real(view, device="cpu", **kw))
    assert ev.main([str(tmp_path / "x"), "--store", str(root), "--mirror-root", "", "--parity", "9",
                    "--parity-start", "", "--parity-days", "30", "--parity-symbols", "4", "--batch", "4"]) == 0
    out = capsys.readouterr().out
    assert "cpu: " in out and "cuda: " in out and "不一致 0 条" in out
    assert not (tmp_path / "x").exists()                      # 对数模式不写面板
