"""scripts/compare_daily_store.py, including the ``--root`` branch that only
ever runs on the server.

The conventions themselves are what ``--self-check`` covers (derived fields vs
a numpy reference over the same cube). What is pinned here instead is the
plumbing that stands between the server and that comparison and that has no
other test: reading ``daily/meta.json``, lining a view up against a store whose
axes are much larger than it, refusing to compare when they do not line up, and
returning an exit code a caller can act on.

The V2 daily store written here is deliberately *wider* than the view in both
axes -- extra dates on both sides, extra instruments interleaved -- because
position-based slicing would pass on an exactly-matching fixture and silently
compare the wrong columns on the real one.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compare_daily_store import main, reference_daily  # noqa: E402

from alphacube.store import open_cube_store  # noqa: E402
from alphacube.synthetic import write_synthetic_store  # noqa: E402

FIELDS = ("Open", "High", "Low", "Close", "Volume", "Amount", "Vwap", "Return")


def _write_v2_daily(root: Path, view, fields, pad_days=3, pad_syms=2, corrupt=None):
    """A V2 ``daily/`` directory whose axes strictly contain the view's."""
    daily = root / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    dates = ([f"1999{i:04d}" for i in range(pad_days)] + list(view.dates)
             + [f"2099{i:04d}" for i in range(pad_days)])
    # interleave the extra instruments so the view's columns are not a contiguous block
    insts, cols = [], []
    for i, s in enumerate(view.symbols):
        for k in range(pad_syms):
            insts.append(f"EXTRA{i}_{k}")
        cols.append(len(insts))
        insts.append(s)
    insts += [f"EXTRA_TAIL{k}" for k in range(pad_syms)]
    (daily / "meta.json").write_text(json.dumps({"dates": dates, "instruments": insts}), encoding="utf-8")
    for name in FIELDS:
        arr = np.full((len(dates), len(insts)), np.nan, dtype=np.float32)
        block = np.asarray(fields[name], dtype=np.float32)
        if corrupt == name:
            block = block + 1.0
        arr[pad_days:pad_days + view.n_days, cols] = block
        np.save(daily / f"field_{name}.npy", arr)
    return daily


@pytest.fixture(scope="module")
def compute_root(tmp_path_factory):
    """A compute root shaped like the server's: ``minute/`` plus ``daily/``."""
    root = tmp_path_factory.mktemp("compute_root")
    write_synthetic_store(root / "minute", n_symbols=8, n_days=30, n_minutes=241,
                          seed=5, suspended_frac=0.05)
    view = open_cube_store(root).view()
    return root, view, reference_daily(view)


def test_self_check_passes():
    assert main.__module__  # the module imported cleanly with its script-relative sys.path
    assert subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_daily_store.py"),
                           "--self-check"], cwd=ROOT, env={"PYTHONPATH": "src", "PATH": ""},
                          capture_output=True).returncode == 0


def test_root_branch_aligns_by_label_not_position(compute_root, monkeypatch, capsys):
    root, view, ref = compute_root
    _write_v2_daily(root, view, ref)
    monkeypatch.setattr(sys, "argv", ["compare_daily_store.py", "--root", str(root)])
    assert main() == 0
    out = capsys.readouterr().out
    assert "derived (engine) vs V2 daily store" in out
    assert "comparison OK" in out
    for name in FIELDS:                              # every field was actually reached
        assert name in out


def test_root_branch_reports_a_corrupted_field(compute_root, monkeypatch, capsys):
    root, view, ref = compute_root
    _write_v2_daily(root, view, ref, corrupt="Close")
    monkeypatch.setattr(sys, "argv", ["compare_daily_store.py", "--root", str(root)])
    assert main() == 1
    assert "MISMATCH" in capsys.readouterr().out


def test_root_branch_refuses_when_the_axes_do_not_cover_the_view(compute_root, monkeypatch, capsys):
    root, view, ref = compute_root
    daily = _write_v2_daily(root, view, ref)
    meta = json.loads((daily / "meta.json").read_text(encoding="utf-8"))
    meta["instruments"] = [s for s in meta["instruments"] if s != view.symbols[0]]
    (daily / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["compare_daily_store.py", "--root", str(root)])
    assert main() == 1
    assert "do not cover the view" in capsys.readouterr().out


def test_root_branch_without_a_daily_store(tmp_path, monkeypatch, capsys):
    root = tmp_path / "minute_only"
    write_synthetic_store(root / "minute", n_symbols=4, n_days=12, n_minutes=241, seed=6)
    monkeypatch.setattr(sys, "argv", ["compare_daily_store.py", "--root", str(root)])
    assert main() == 0
    assert "no daily store" in capsys.readouterr().out
