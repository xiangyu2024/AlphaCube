"""多空书报告:和 build_champion 同口径;两条腿加起来 x 0.5 = 毛多空;半衰期只在训练段选。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ls_report                                             # noqa: E402


def _data(D=200, S=60, seed=0):
    rng = np.random.default_rng(seed)
    alpha = rng.normal(size=(D, S))
    target = 0.01 * alpha + rng.normal(0, 0.02, (D, S))
    return alpha, target


def test_legs_add_up_to_the_gross_book():
    from alphacube.metrics import long_short_weights, portfolio_returns

    sig, target = _data()
    long_r, short_r = ls_report.leg_returns(sig, target)
    gross = portfolio_returns(long_short_weights(sig, 0.1), target, 0.0)[1]
    assert np.allclose(0.5 * (long_r + short_r), gross, atol=1e-12)
    assert np.nanmean(long_r) > 0 and np.nanmean(short_r) > 0


def test_report_matches_build_champion_and_writes_json(tmp_path):
    from build_champion import evaluate

    sig, target = _data()
    split = 140
    r = ls_report.report(sig, target, split)
    ev = evaluate(sig, target, split, target)
    assert np.isclose(r["raw"]["test"]["sharpe"], ev["ls_net"])
    assert r["halflife"] == ev["ls_hl"] and np.isclose(r["smoothed"]["test"]["sharpe"], ev["ls_sm_net"])
    np.save(tmp_path / "s.npy", sig)
    np.save(tmp_path / "t.npy", target)
    np.save(tmp_path / "d.npy", np.array([f"2020{i:04d}" for i in range(200)]))
    out = tmp_path / "o.json"
    assert ls_report.main(["--signal", f"x={tmp_path / 's.npy'}", "--target", str(tmp_path / "t.npy"),
                           "--dates", str(tmp_path / "d.npy"), "--out", str(out)]) == 0
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["split"] == 140 and len(d["signals"]["x"]["smoothed"]["net"]) == 200


def test_plot_mode_writes_a_png(tmp_path):
    import pytest
    pytest.importorskip("matplotlib")
    sig, target = _data()
    np.save(tmp_path / "s.npy", sig)
    np.save(tmp_path / "t.npy", target)
    import datetime as dt
    np.save(tmp_path / "d.npy", np.array([(dt.date(2020, 1, 1) + dt.timedelta(days=i)).strftime("%Y%m%d")
                                          for i in range(200)]))
    j = tmp_path / "o.json"
    ls_report.main(["--signal", f"x={tmp_path / 's.npy'}", "--target", str(tmp_path / "t.npy"),
                    "--dates", str(tmp_path / "d.npy"), "--out", str(j)])
    png = tmp_path / "o.png"
    assert ls_report.main(["plot", "--json", str(j), "--out", str(png)]) == 0 and png.stat().st_size > 1000


def _json_with(tmp_path, names):
    import datetime as dt
    sig, target = _data()
    rng = np.random.default_rng(1)
    paths = []
    for i, n in enumerate(names):
        s = sig + rng.normal(0, 0.5 + i, sig.shape)
        np.save(tmp_path / f"{i}.npy", s)
        paths.append(f"{n}={tmp_path / f'{i}.npy'}")
    np.save(tmp_path / "t.npy", target)
    np.save(tmp_path / "d.npy", np.array([(dt.date(2020, 1, 1) + dt.timedelta(days=i)).strftime("%Y%m%d")
                                          for i in range(200)]))
    j = tmp_path / "o.json"
    args = [a for p in paths for a in ("--signal", p)]
    assert ls_report.main(args + ["--target", str(tmp_path / "t.npy"), "--dates", str(tmp_path / "d.npy"),
                                  "--out", str(j)]) == 0
    return j


def test_tearsheet_style_one_two_three_signals(tmp_path):
    import pytest
    pytest.importorskip("matplotlib")
    j = _json_with(tmp_path, ["a", "b", "c"])
    for only in ("a", "b,a", ""):                             # 1 个、2 个(第一个减第二个)、全部 3 个
        png = tmp_path / f"t_{only or 'all'}.png"
        args = ["plot", "--style", "tearsheet", "--json", str(j), "--out", str(png)]
        if only:
            args += ["--only", only]
        assert ls_report.main(args + ["--label", "a=甲,全市场"]) == 0 and png.stat().st_size > 1000


def test_plot_arguments_are_validated(tmp_path):
    import pytest
    j = _json_with(tmp_path, ["a", "b"])
    base = ["plot", "--json", str(j), "--out", str(tmp_path / "x.png")]
    with pytest.raises(SystemExit):
        ls_report.main(base + ["--style", "tearsheet", "--label", "a"])          # 没有 =
    with pytest.raises(SystemExit):
        ls_report.main(base + ["--style", "tearsheet", "--label", "zz=甲"])      # 不存在的名字
    with pytest.raises(SystemExit):
        ls_report.main(base + ["--only", "a"])                                   # 简版不认 --only
    with pytest.raises(SystemExit):
        ls_report.plot_tearsheet(str(j), str(tmp_path / "y.png"), only=["a", "nope"])
