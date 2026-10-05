"""scripts/pick_single_factors.py:只按训练段的书排序,在平滑后的信号上贪心去相关,整面板的数算得对。"""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pick_single_factors as psf  # noqa: E402


def _row(i, ric=0.05, expr=None, **kw):
    return {"i": str(i), "rank_ic_train": str(ric), "expression": expr or f"f{i}", **kw}


def _score(sharpe, cov=1.0, distinct=1.0, ls=0.0):
    return {"long_sharpe_train": sharpe, "ls_sharpe_train": ls, "coverage_train": cov, "distinct_train": distinct}


def _signal(D=160, S=120, seed=0, persistence=0.9, noise=3.0):
    rng = np.random.default_rng(seed)
    x = np.zeros((D, S))
    for t in range(1, D):
        x[t] = persistence * x[t - 1] + rng.normal(size=S)
    target = 0.01 * x + 0.02 * rng.normal(size=(D, S))
    noisy = x + noise * rng.normal(size=(D, S))             # 噪声大:平滑以后才好
    noisy[rng.random((D, S)) < 0.03] = np.nan
    return noisy, target


def test_greedy_skips_correlated_and_respects_limit():
    e = np.eye(4)
    vecs = {0: e[0], 1: psf.unit(e[0] + 0.5 * e[1]), 2: e[1], 3: e[2], 4: None, 5: -e[2]}
    assert psf.greedy([0, 1, 4, 2, 5, 3], vecs.get, 0.5, 10) == [0, 2, 5]   # 1 和 0 相关 0.89;4 没指纹;3 和 5 是 -1
    assert psf.greedy([0, 1, 2, 3], vecs.get, 0.5, 1) == [0]
    assert psf.greedy([], vecs.get, 0.5, 3) == []


def test_greedy_threshold_is_strict():
    a = np.array([1.0, 0.0])
    b = psf.unit(np.array([0.5, np.sqrt(0.75)]))            # 相关正好 0.5
    assert psf.greedy([0, 1], {0: a, 1: b}.get, 0.5, 5) == [0]
    assert psf.greedy([0, 1], {0: a, 1: b}.get, 0.51, 5) == [0, 1]


def test_eligible_order_gates_and_sorts_by_the_chosen_train_sharpe():
    rows = [_row(i, lib="x") for i in range(6)]
    scores = {("x", 0): _score(2.0, ls=5.0), ("x", 1): _score(3.0, cov=0.5), ("x", 2): _score(2.5, distinct=0.01),
              ("x", 3): _score(0.9, ls=9.0), ("x", 4): _score(2.7, ls=1.5), ("x", 5): _score(np.nan)}
    assert psf.eligible_order(rows, scores, 0.95, 0.5, 1.0) == [4, 0]
    assert psf.eligible_order(rows, scores, 0.95, 0.5, 1.0, "ls_sharpe_train") == [3, 0, 4]
    assert psf.eligible_order(rows, scores, 0.0, 0.0, -9.0) == [1, 4, 2, 0, 3]
    del scores[("x", 4)]                                    # 没打过分的不收
    assert psf.eligible_order(rows, scores, 0.95, 0.5, 1.0) == [0]


def test_dedupe_keeps_first():
    rows = [_row(0, expr="a", lib="x"), _row(1, expr="b", lib="x"), _row(0, expr="a", lib="y")]
    assert [(r["lib"], r["expression"]) for r in psf.dedupe(rows)] == [("x", "a"), ("x", "b")]


def test_distinct_fraction():
    rng = np.random.default_rng(0)
    cont = rng.normal(size=(20, 100))
    assert psf.distinct_fraction(cont) == pytest.approx(1.0)
    disc = rng.integers(0, 4, size=(20, 100)).astype(float)
    assert psf.distinct_fraction(disc) == pytest.approx(0.04)
    cont[:, 80:] = np.nan
    assert psf.distinct_fraction(cont) == pytest.approx(1.0)       # 分母是有值的票
    assert psf.distinct_fraction(np.full((20, 100), np.nan)) == 0.0


def test_long_book_by_hand():
    # 20 只票,前 10% = 2 只。第 0 天持 18、19;第 1 天换成 17、19;第 2 天不变
    sig = np.tile(np.arange(20.0), (3, 1))
    sig[1, 17], sig[1, 18] = 100.0, -1.0
    sig[2] = sig[1]
    target = np.zeros((3, 20))
    target[0, 19], target[0, 18] = 0.04, 0.02               # 多头 +3%,等权 0.3%
    target[1, 17] = np.nan                                  # 买不进:这只记 0,等权分母少一只
    target[1, 19] = 0.02
    from alphacube.metrics import long_short_weights

    net, gross, turn = psf.long_book(long_short_weights(sig, 0.1), sig, target, cost=0.001)
    assert gross[0] == pytest.approx(0.03 - 0.06 / 20)
    assert gross[1] == pytest.approx(0.01 - 0.02 / 19)
    assert turn.tolist() == pytest.approx([0.5, 0.5, 0.0])  # 建仓算半边,换一只 = 卖一半买一半的一半
    assert net[0] == pytest.approx(gross[0] - 0.001 * 1.0) and net[1] == pytest.approx(gross[1] - 0.001 * 1.0)
    assert net[2] == pytest.approx(gross[2]) and gross[2] == pytest.approx(0.0)


def test_long_book_gross_matches_ls_report_long_leg():
    from alphacube.metrics import long_short_weights
    from ls_report import leg_returns

    sig, target = _signal(seed=4)
    target[np.random.default_rng(5).random(target.shape) < 0.05] = np.nan
    _, gross, _ = psf.long_book(long_short_weights(sig, 0.1), sig, target)
    assert np.allclose(gross, leg_returns(sig, target)[0], equal_nan=True)


def test_train_books_ls_side_matches_ls_report_and_long_side_is_the_argmax():
    from alphacube.metrics import ema_rows, long_short_weights
    from long_only_grid import stats
    from ls_report import HALFLIVES, report

    sig, target = _signal()
    split = 110
    got = psf.train_books(sig[:split], target[:split])
    want = report(sig, target, split)
    assert got["hl_ls"] == want["halflife"] and got["hl_ls"] > 0
    assert got["ls_sharpe_train"] == pytest.approx(want["smoothed"]["train"]["sharpe"])
    by_hl = {}
    for hl in HALFLIVES:
        sm = ema_rows(sig[:split], hl) if hl else sig[:split]
        by_hl[hl] = stats(psf.long_book(long_short_weights(sm, 0.1), sm, target[:split])[0])["sharpe"]
    assert got["hl_long"] == max(HALFLIVES, key=lambda h: by_hl[h])
    assert got["long_sharpe_train"] == pytest.approx(by_hl[got["hl_long"]])


def test_train_books_ignore_test_period():
    sig, target = _signal(seed=1)
    split = 110
    a = psf.train_books(sig[:split], target[:split])
    sig2, target2 = sig.copy(), target.copy()
    sig2[split:], target2[split:] = -sig[split:], 5 * target[split:]
    assert psf.train_books(sig2[:split], target2[:split]) == a


def test_pick_report_train_numbers_match_train_books_and_test_book_starts_empty():
    from alphacube.metrics import ema_rows

    sig, target = _signal(seed=6)
    split = 110
    tb = psf.train_books(sig[:split], target[:split])
    sm = ema_rows(sig, tb["hl_long"]) if tb["hl_long"] else sig
    rep, series = psf.pick_report(sm, target, split)
    assert rep["train"]["long_sharpe"] == pytest.approx(tb["long_sharpe_train"])
    assert rep["train"]["long_ann"] == pytest.approx(tb["long_ann_train"])
    assert rep["train"]["long_turnover"] == pytest.approx(tb["long_turnover_train"])
    n_test = sig.shape[0] - split                                    # 测试段单独拿出来从空仓建书:应当一样
    alone, s2 = psf.pick_report(sm[split:], target[split:], n_test - 1)
    assert np.allclose(series[split:-1], s2[:-1], equal_nan=True)
    assert alone["train"]["long_turnover"] > 0 and series.shape == (sig.shape[0],)
    cont, _ = psf.pick_report(sm, target, 1)                         # 不重新建仓的话第一天不付建仓费,数不一样
    assert not np.isclose(series[split], _[split])


def test_smoothed_fingerprint_restricting_columns_is_exact_and_uses_last_days():
    from alphacube.metrics import ema_rows, rank_rows, zscore_rows

    sig, _ = _signal(D=80, S=60, seed=7)
    cols = np.array([0, 7, 8, 30, 59])
    got = psf.smoothed_fingerprint(sig, -1.0, 5, cols, days=20)
    full = ema_rows(-sig, 5)[-20:][:, cols]
    want = np.nan_to_num(zscore_rows(rank_rows(full)), nan=0.0).reshape(-1)
    assert got.shape == (20 * 5,) and np.allclose(got, want)
    raw = psf.smoothed_fingerprint(sig, 1.0, 0, cols, days=20)
    assert np.allclose(raw, np.nan_to_num(zscore_rows(rank_rows(sig[-20:][:, cols])), nan=0.0).reshape(-1))


def test_smoothing_exposes_shared_slow_component():
    """两个日频因子原始面板上低相关,各自平滑以后高相关 —— 贪心必须在平滑后的信号上比。"""
    rng = np.random.default_rng(8)
    D, S = 300, 200
    slow = np.cumsum(rng.normal(size=(D, S)) * 0.05, axis=0) + rng.normal(size=S)
    a, b = slow + 3 * rng.normal(size=(D, S)), slow + 3 * rng.normal(size=(D, S))
    cols = np.arange(S)

    def corr(hl):
        return float(psf.unit(psf.smoothed_fingerprint(a, 1.0, hl, cols)) @ psf.unit(psf.smoothed_fingerprint(b, 1.0, hl, cols)))

    assert corr(0) < 0.3 < 0.6 < corr(20)
    vecs = {hl: {0: psf.unit(psf.smoothed_fingerprint(a, 1.0, hl, cols)), 1: psf.unit(psf.smoothed_fingerprint(b, 1.0, hl, cols))}
            for hl in (0, 20)}
    assert psf.greedy([0, 1], vecs[0].get, 0.3, 5) == [0, 1] and psf.greedy([0, 1], vecs[20].get, 0.3, 5) == [0]


def _library(tmp_path, D=60, S=90, split=40):
    sig, target = _signal(D=D, S=S, seed=2)
    (tmp_path / "panels").mkdir()
    np.save(tmp_path / "_target.npy", target)
    np.save(tmp_path / "panels" / "00000.npy", sig.astype(np.float32))
    half = -sig.copy()
    half[:, ::2] = np.nan
    np.save(tmp_path / "panels" / "00003.npy", half.astype(np.float32))
    rows = [_row(0, 0.05, lib="x", dir=str(tmp_path)), _row(3, -0.05, lib="x", dir=str(tmp_path))]
    return rows, sig, target, split


def test_score_candidates_caches_and_slices_train_only(tmp_path):
    rows, sig, target, split = _library(tmp_path)
    cache = tmp_path / "s" / "scores.csv"
    got = psf.score_candidates(rows, str(tmp_path / "_target.npy"), split, cache, workers=2)
    assert set(got) == {("x", 0), ("x", 3)}
    assert got[("x", 3)]["sign"] == -1.0 and got[("x", 0)]["sign"] == 1.0
    assert got[("x", 3)]["coverage_train"] < 0.55 < 0.9 < got[("x", 0)]["coverage_train"]
    want = psf.train_books(np.load(tmp_path / "panels" / "00000.npy")[:split].astype(np.float64), target[:split])
    assert got[("x", 0)]["long_sharpe_train"] == pytest.approx(want["long_sharpe_train"])
    assert got[("x", 0)]["ls_sharpe_train"] == pytest.approx(want["ls_sharpe_train"])
    np.save(tmp_path / "panels" / "00000.npy", np.zeros_like(sig, dtype=np.float32))   # 缓存命中:不重算
    again = psf.score_candidates(rows, str(tmp_path / "_target.npy"), split, cache, workers=2)
    assert again[("x", 0)]["long_sharpe_train"] == pytest.approx(want["long_sharpe_train"])
    assert len(list(csv.DictReader(open(cache)))) == 2


def test_score_cache_refuses_other_settings(tmp_path):
    rows, _, _, split = _library(tmp_path)
    cache = tmp_path / "scores.csv"
    psf.score_candidates(rows, str(tmp_path / "_target.npy"), split, cache, workers=1)
    with pytest.raises(SystemExit):                          # 换了切分点
        psf.score_candidates(rows, str(tmp_path / "_target.npy"), split - 5, cache, workers=1)
    with pytest.raises(SystemExit):                          # 换了费率
        psf.score_candidates(rows, str(tmp_path / "_target.npy"), split, cache, workers=1, cost=0.003)
    moved = [dict(r, dir=str(tmp_path / "elsewhere")) for r in rows]
    with pytest.raises(SystemExit):                          # 同名的库指到了别的目录
        psf.score_candidates(moved, str(tmp_path / "_target.npy"), split, cache, workers=1)
    (tmp_path / "scores.csv.meta.json").unlink()
    with pytest.raises(SystemExit):                          # 没有 meta 的旧缓存不认
        psf.score_candidates(rows, str(tmp_path / "_target.npy"), split, cache, workers=1)


def test_pairwise_rank_corr_matches_spearman_on_common_names():
    from scipy.stats import spearmanr

    rng = np.random.default_rng(0)
    D, S = 6, 80
    a = rng.normal(size=(D, S))
    b = 0.6 * a + 0.8 * rng.normal(size=(D, S))
    c = -a ** 3                                              # 单调变换:秩相关 -1
    b[:, :10] = np.nan                                       # 覆盖不同
    got = psf.pairwise_rank_corr([a, b, c], slice(0, D))
    assert got[0, 2] == pytest.approx(-1.0)
    assert np.allclose(np.diag(got), 1.0) and np.allclose(got, got.T)
    want = np.mean([spearmanr(a[t, 10:], b[t, 10:])[0] for t in range(D)])
    # b 的秩是在自己 70 只上取的,a 的是在 80 只上取的:只在共同的 70 只上算相关,和 Spearman 差一点点
    assert got[0, 1] == pytest.approx(want, abs=0.02)
    half = psf.pairwise_rank_corr([a, b, c], slice(0, 3))
    assert half[0, 1] != pytest.approx(got[0, 1], abs=1e-6)  # days 真的起作用


def test_pairwise_rank_corr_skips_thin_days():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(3, 50)), rng.normal(size=(3, 50))
    b[0, psf.MIN_COMMON - 1:] = np.nan                       # 第 0 天共同只有 29 只
    got = psf.pairwise_rank_corr([a, b], slice(0, 3))
    want = psf.pairwise_rank_corr([a[1:], b[1:]], slice(0, 2))
    assert got[0, 1] == pytest.approx(want[0, 1])


def test_return_corr_uses_common_days():
    rng = np.random.default_rng(2)
    a = rng.normal(size=100)
    b = 0.5 * a + rng.normal(size=100)
    b[:20] = np.nan
    got = psf.return_corr([a, b, -a])
    assert got[0, 1] == pytest.approx(np.corrcoef(a[20:], b[20:])[0, 1]) and got[0, 2] == pytest.approx(-1.0)
    assert np.allclose(got, got.T) and np.allclose(np.diag(got), 1.0)
    assert np.isnan(psf.return_corr([a, np.full(100, np.nan)])[0, 1])


def test_compare_panels_tolerates_boundary_noise_but_not_a_different_factor():
    rng = np.random.default_rng(3)
    old = rng.normal(size=(30, 400)).astype(np.float32)
    new = old * (1 + 1e-6)
    new[0, :5] = np.nan                                      # 5 / 12000 格 NaN 不一致
    assert psf.compare_panels(new, old)["ok"]
    assert psf.compare_panels(old.copy(), old) == {"ok": True, "mask_mismatch": 0.0, "rank_corr_mean": pytest.approx(1.0),
                                                   "rank_corr_min": pytest.approx(1.0)}
    other = rng.normal(size=old.shape).astype(np.float32)
    assert not psf.compare_panels(other, old)["ok"]
    holes = old.copy()
    holes[:, :40] = np.nan                                   # 10% 的格子没了
    assert not psf.compare_panels(holes, old)["ok"]
    assert not psf.compare_panels(old[:, :10], old)["ok"]


def test_load_library_does_not_use_the_healthy_column(tmp_path):
    cols = ["i", "rank_ic_train", "rank_ic_test", "healthy", "expression"]
    with open(tmp_path / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerow({"i": 0, "rank_ic_train": 0.05, "rank_ic_test": 0.01, "healthy": "True", "expression": "f0"})
        w.writerow({"i": 1, "rank_ic_train": -0.04, "rank_ic_test": "", "healthy": "False", "expression": "f1"})
        w.writerow({"i": 2, "rank_ic_train": "nan", "rank_ic_test": 0.2, "healthy": "True", "expression": "f2"})
        w.writerow({"i": 3, "rank_ic_train": "", "rank_ic_test": 0.2, "healthy": "True", "expression": "f3"})
    with open(tmp_path / "manifest.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([["i", "run", "expression", "name"], [0, "legacy", "f0", "alpha_a"],
                                  [1, "legacy", "f1", "b"], [2, "legacy", "f2", "c"], [3, "legacy", "f3", "d"]])
    rows = psf.load_library(tmp_path, "手写")
    assert [(r["i"], r["name"], r["lib"]) for r in rows] == [("0", "alpha_a", "手写"), ("1", "b", "手写")]
