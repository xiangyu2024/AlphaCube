import json

import numpy as np

from alphacube.label import Label
from alphacube.research import evaluate_factors, safe_names


def test_evaluate_factors_end_to_end(store, tmp_path):
    exprs = {
        "dt_vol": "day_std(in_mean_rolling(Close, 5))",
        "mixed": "cs_rank(ts_mean(day_std(Ret), 5))",
        "nt": "day_mean(nt_cs_rank(Close))",
        "波动/率": "div(day_max(Close), close_1d)",
        "bad": "in_mean_rolling(day_mean(Close), 5)",
    }
    table = evaluate_factors(store, exprs, start=store.dates[20], end=store.dates[50],
                             label=Label("close", 1), out_dir=tmp_path / "run", quiet=True)
    assert list(table.index) == list(exprs)
    assert np.isnan(table.loc["bad", "ic_mean"])
    for n in ("dt_vol", "mixed", "nt", "波动/率"):
        assert np.isfinite(table.loc[n, "ic_mean"]) and table.loc[n, "coverage"] > 0.8, n
    meta = json.loads((tmp_path / "run" / "meta.json").read_text(encoding="utf-8"))
    assert meta["dates"][0] == store.dates[20] and len(meta["dates"]) == 31
    assert "bad" in meta["errors"] and "波动/率" in meta["value_files"]
    v = np.load(tmp_path / "run" / "values" / meta["value_files"]["mixed"])
    assert v.shape == (31, store.n_symbols) and v.dtype == np.float32
    assert np.isfinite(v[:5]).mean() > 0.8            # warm-up rows were read before `start`
    assert (tmp_path / "run" / "metrics.csv").exists()


def test_safe_names_unique_and_unicode():
    m = safe_names(["a b", "a_b", "波动率", "a/b", "a_b"])
    assert len(set(m.values())) == 4 and m["波动率"] == "波动率"
