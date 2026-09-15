import numpy as np
import torch

from alphacube.label import Label, daily_close, forward_return, open_vwap15
from alphacube.metrics import MIN_PAIRS, batched_ic, cs_pearson, ic_summary, rank_rows, zscore_rows
from alphacube.store import DAY_STATUS_CODES, build_cube_store


def _tiny_store(tmp_path):
    S, D, M = 3, 4, 20
    rng = np.random.default_rng(0)
    price = 10 + rng.random((S, D, M))
    vol = np.full((S, D, M), 100.0)
    amt = vol * price
    high, low = price + 0.5, price - 0.5
    # stock 0 day 1: a missing bar in the window -> NaN
    vol[0, 1, 5] = np.nan
    # stock 1 day 2: zero volume in the whole window -> NaN
    vol[1, 2, 1:16] = 0.0
    amt[1, 2, 1:16] = 0.0
    # stock 2 day 3: vwap outside [low, high] -> NaN
    amt[2, 3, 1:16] = vol[2, 3, 1:16] * 50.0
    cubes = {"Open": price, "High": high, "Low": low, "Close": price, "Volume": vol, "Vwap": amt / vol, "Ret": price * 0, "Amt": amt}
    status = np.full((S, D), DAY_STATUS_CODES["TRADING"], dtype=np.int8)
    status[2, 0] = DAY_STATUS_CODES["SUSPENDED"]
    return build_cube_store(tmp_path / "tiny", cubes, [f"s{i}" for i in range(S)],
                            [f"2024010{i+1}" for i in range(D)], day_status=status), price


def test_open_vwap15_rules(tmp_path):
    store, price = _tiny_store(tmp_path)
    v = open_vwap15(store.view())                       # (D, S)
    assert np.isnan(v[1, 0]) and np.isnan(v[2, 1]) and np.isnan(v[3, 2]) and np.isnan(v[0, 2])
    expected = price[0, 0, 1:16].mean()                  # equal volumes -> plain mean of bar prices
    assert np.isclose(v[0, 0], expected)
    dc = daily_close(store.view())
    assert np.isclose(dc[0, 0], price[0, 0, -1]) and np.isnan(dc[0, 2])


def test_forward_return_alignment():
    p = np.arange(1, 7, dtype=float)[:, None] * np.ones((1, 2))
    t = forward_return(p, horizon=1, entry_lag=1)
    assert np.isclose(t[0, 0], 3 / 2 - 1) and np.isnan(t[-1, 0]) and np.isnan(t[-2, 0])
    t0 = forward_return(p, horizon=2, entry_lag=0)
    assert np.isclose(t0[0, 0], 3 / 1 - 1)
    assert Label("open_vwap15", 1).describe().startswith("OpenVwap15[t+1+1]")


def test_cs_pearson_semantics():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(5, 12)), rng.normal(size=(5, 12))
    ic = cs_pearson(a, b)
    for d in range(5):
        assert np.isclose(ic[d], np.corrcoef(a[d], b[d])[0, 1])
    a2 = a.copy()
    a2[0, :] = 1.0                        # zero variance
    a2[1, 3] = np.inf                     # inf is not a value
    a2[2, MIN_PAIRS - 1:] = np.nan        # too few pairs
    ic2 = cs_pearson(a2, b)
    assert np.isnan(ic2[0]) and np.isfinite(ic2[1]) and np.isnan(ic2[2])
    assert np.isclose(ic2[1], np.corrcoef(np.delete(a2[1], 3), np.delete(b[1], 3))[0, 1])
    t = batched_ic([torch.from_numpy(a2).float()], torch.from_numpy(b).float())[0].item()
    assert np.isclose(t, np.nanmean(ic2), rtol=1e-4)


def test_summary_and_helpers():
    rng = np.random.default_rng(2)
    target = rng.normal(size=(30, 15))
    panel = target + rng.normal(size=(30, 15))
    s = ic_summary(panel, target)
    assert s["ic_mean"] > 0.3 and s["n_days"] == 30 and 0 < s["coverage"] <= 1
    r = rank_rows(np.array([[3.0, np.nan, 1.0]]))
    assert r[0, 0] == 2 and np.isnan(r[0, 1]) and r[0, 2] == 1
    z = zscore_rows(np.array([[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan], [2.0, 2.0, 2.0]]))
    assert np.isclose(z[0].std(), 1.0) and np.isnan(z[1]).all() and np.isnan(z[2]).all()
