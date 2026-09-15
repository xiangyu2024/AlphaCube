"""Hand-verified tests for alphacube.portfolio.construct.

Expected values for `transform_signal` variants are computed independently
in each test (straightforward pandas/numpy arithmetic mirroring the spec),
not by re-invoking the module's private helpers, so a bug in the
implementation can't also be baked into the "expected" value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphacube.portfolio.construct import (
    KNOWN_VARIANTS,
    CappedWeights,
    blend_signals_by_rolling_icir,
    cap_and_renormalize_weights,
    is_supported_variant,
    top_n_from_variant,
    transform_signal,
)

# 7 names, at least one NaN (CCC) and one negative (BBB, FFF); GGG is an
# explicit zero (inactive but not NaN); the rest are ordinary positives.
NAMES = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]
RAW = pd.Series([1.0, -0.5, np.nan, 0.2, 0.8, -2.0, 0.0], index=NAMES)
# The "cleaned" signal every variant is expected to start from:
CLEANED = pd.Series([1.0, -0.5, 0.0, 0.2, 0.8, -2.0, 0.0], index=NAMES)


def test_raw_continuous_clips_and_cleans():
    out = transform_signal(RAW, "raw_continuous")
    expected = pd.Series([1.0, -0.5, 0.0, 0.2, 0.8, -1.0, 0.0], index=NAMES)
    pd.testing.assert_series_equal(out, expected, check_names=False)


def test_long_short_is_alias_of_raw_continuous():
    pd.testing.assert_series_equal(
        transform_signal(RAW, "long_short"), transform_signal(RAW, "raw_continuous")
    )


def test_long_only_drops_negatives():
    out = transform_signal(RAW, "long_only")
    expected = pd.Series([1.0, 0.0, 0.0, 0.2, 0.8, 0.0, 0.0], index=NAMES)
    pd.testing.assert_series_equal(out, expected, check_names=False)


def test_shift_to_long_subtracts_min():
    out = transform_signal(RAW, "shift_to_long")
    # min of CLEANED is -2.0 (FFF)
    expected = CLEANED - (-2.0)
    pd.testing.assert_series_equal(out, expected, check_names=False)


def test_shift_to_long_degenerate_day_is_all_zero():
    flat = pd.Series([0.5, 0.5, 0.5, 0.5], index=list("ABCD"))
    out = transform_signal(flat, "shift_to_long")
    pd.testing.assert_series_equal(out, pd.Series(0.0, index=list("ABCD")), check_names=False)


def test_demean_shift_long():
    active = CLEANED[CLEANED != 0.0]  # AAA, BBB, DDD, EEE, FFF
    mean_active = active.mean()
    x = CLEANED - mean_active
    shifted = x[active.index] - x[active.index].min()
    expected = pd.Series(0.0, index=NAMES)
    expected.loc[active.index] = shifted

    out = transform_signal(RAW, "demean_shift_long")
    pd.testing.assert_series_equal(out, expected, check_names=False)
    # inactive names are exactly zero
    assert out["CCC"] == 0.0
    assert out["GGG"] == 0.0


def test_rank_long():
    active = CLEANED[CLEANED != 0.0]
    r = active.rank(pct=True, method="average")
    expected = pd.Series(0.0, index=NAMES)
    expected.loc[active.index] = r**2

    out = transform_signal(RAW, "rank_long")
    pd.testing.assert_series_equal(out, expected, check_names=False)
    # hand check the extremes: FFF is the lowest active value -> rank 1/5 -> 0.04
    assert out["FFF"] == pytest.approx(0.04)
    # AAA is the highest active value -> rank 5/5 -> 1.0
    assert out["AAA"] == pytest.approx(1.0)


def test_softmax_long_default_temperature():
    active = CLEANED[CLEANED != 0.0]
    mean_active = active.mean()
    std_active = active.std(ddof=0)
    z = ((active - mean_active) / std_active).clip(-8.0, 8.0) / 1.0
    numerator = np.exp(z - z.max())
    expected = pd.Series(0.0, index=NAMES)
    expected.loc[active.index] = numerator

    out = transform_signal(RAW, "softmax_long")
    pd.testing.assert_series_equal(out, expected, check_names=False)
    # the max-active-value name always gets numerator == 1.0 exactly
    assert out["AAA"] == pytest.approx(1.0)
    assert out["CCC"] == 0.0
    assert out["GGG"] == 0.0


def test_softmax_long_temperature_flattens_distribution():
    out_cold = transform_signal(RAW, "softmax_long", temperature=1.0)
    out_hot = transform_signal(RAW, "softmax_long", temperature=10.0)
    # higher temperature -> softmax numerators move closer together (less peaked)
    spread_cold = out_cold[out_cold > 0].max() - out_cold[out_cold > 0].min()
    spread_hot = out_hot[out_hot > 0].max() - out_hot[out_hot > 0].min()
    assert spread_hot < spread_cold


def test_softmax_long_degenerate_std_falls_back_to_clip():
    flat_active = pd.Series([2.0, 2.0, 2.0, 0.0], index=list("ABCD"))
    out = transform_signal(flat_active, "softmax_long")
    pd.testing.assert_series_equal(out, flat_active.clip(lower=0.0), check_names=False)


def test_top_positive_long():
    positive = CLEANED[CLEANED > 0.0]  # AAA=1.0, DDD=0.2, EEE=0.8
    threshold = positive.quantile(0.80)
    keep = positive[positive > threshold]
    expected = pd.Series(0.0, index=NAMES)
    expected.loc[keep.index] = keep

    out = transform_signal(RAW, "top_positive_long")
    pd.testing.assert_series_equal(out, expected, check_names=False)
    # hand check: sorted positives are [0.2, 0.8, 1.0]; pandas' default
    # linear-interpolation 0.80-quantile is 0.8 + 0.6*(1.0-0.8) = 0.92,
    # so only AAA (1.0) survives "> threshold"
    assert threshold == pytest.approx(0.92)
    assert out["AAA"] == pytest.approx(1.0)
    assert out["DDD"] == 0.0
    assert out["EEE"] == 0.0


def test_top_rank_long():
    active = CLEANED[CLEANED != 0.0]
    r = active.rank(pct=True, method="average")
    # AAA (rank 5/5=1.0) and EEE (rank 4/5=0.8) both clear the >=0.80 bar
    keep = r[r >= 0.80]
    expected = pd.Series(0.0, index=NAMES)
    expected.loc[keep.index] = keep**2

    out = transform_signal(RAW, "top_rank_long")
    pd.testing.assert_series_equal(out, expected, check_names=False)
    assert out["AAA"] == pytest.approx(1.0)
    assert out["DDD"] == 0.0  # rank 3/5=0.6, below the bar


def test_top_rank_long_boundary_is_inclusive():
    # EEE has rank 4/5 = 0.80 exactly, which is >= top_quantile=0.80 -> kept.
    active = CLEANED[CLEANED != 0.0]
    r = active.rank(pct=True, method="average")
    assert r["EEE"] == pytest.approx(0.8)
    out = transform_signal(RAW, "top_rank_long")
    assert out["EEE"] == pytest.approx(0.8**2)


def test_topn_long():
    n = 3
    selected_idx = CLEANED.sort_values(ascending=False).index[:n]  # AAA, EEE, DDD
    assert list(selected_idx) == ["AAA", "EEE", "DDD"]
    r = CLEANED.loc[selected_idx].rank(pct=True, method="average")
    expected = pd.Series(0.0, index=NAMES)
    expected.loc[selected_idx] = r**2

    out = transform_signal(RAW, "top3_long")
    pd.testing.assert_series_equal(out, expected, check_names=False)
    assert out["AAA"] == pytest.approx(1.0)  # highest of the 3 -> rank 1.0
    assert out["DDD"] == pytest.approx((1 / 3) ** 2)  # lowest of the 3 -> rank 1/3


def test_unsupported_variant_raises():
    with pytest.raises(ValueError):
        transform_signal(RAW, "not_a_real_variant")


# ---------------------------------------------------------------------------
# top_n_from_variant / is_supported_variant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "variant,expected_n",
    [
        ("top100_long", 100),
        ("top500_long", 500),
        ("top1_long", 1),
        ("top0_long", None),  # degenerate N=0 rejected
        ("top_long", None),  # no digits
        ("top-5_long", None),  # negative not matched by \d+
        ("top100_short", None),  # wrong suffix
        ("raw_continuous", None),
        ("Top100_long", None),  # case sensitive
    ],
)
def test_top_n_from_variant(variant, expected_n):
    assert top_n_from_variant(variant) == expected_n


@pytest.mark.parametrize("variant", list(KNOWN_VARIANTS) + ["top100_long", "top7_long"])
def test_is_supported_variant_true_cases(variant):
    assert is_supported_variant(variant) is True


@pytest.mark.parametrize("variant", ["top0_long", "bogus", "top_long", "TOP100_LONG", ""])
def test_is_supported_variant_false_cases(variant):
    assert is_supported_variant(variant) is False


# ---------------------------------------------------------------------------
# cap_and_renormalize_weights
# ---------------------------------------------------------------------------


def test_cap_and_renormalize_redistributes_excess_and_hits_target():
    raw = pd.Series([0.6, 0.1, 0.1, 0.1, 0.1], index=list("ABCDE"))
    result = cap_and_renormalize_weights(
        raw, target_exposure=1.0, max_single_name_weight=0.3, allow_short=False
    )
    assert isinstance(result, CappedWeights)
    assert result.converged is True
    # no single name exceeds the cap
    assert (result.weights.abs() <= 0.3 + 1e-9).all()
    # total exposure still hits target (redistribution worked)
    assert result.achieved_exposure == pytest.approx(1.0)
    assert result.weights.sum() == pytest.approx(1.0)
    # A is pinned exactly at the cap
    assert result.weights["A"] == pytest.approx(0.3)
    # remaining 0.7 of exposure split proportionally (equally, since B..E
    # were equal in raw weight) across B, C, D, E
    for name in "BCDE":
        assert result.weights[name] == pytest.approx(0.7 / 4)


def test_cap_and_renormalize_reports_unreachable_target_honestly():
    raw = pd.Series([0.6, 0.1, 0.1, 0.1, 0.1], index=list("ABCDE"))
    # 5 names * cap 0.1 = 0.5 max achievable gross, but target asks for 1.0
    result = cap_and_renormalize_weights(
        raw, target_exposure=1.0, max_single_name_weight=0.1, allow_short=False
    )
    assert result.converged is False
    assert result.achieved_exposure < 1.0
    assert result.achieved_exposure == pytest.approx(0.5)
    # cap is still respected even though the target wasn't reachable
    assert (result.weights.abs() <= 0.1 + 1e-9).all()
    # every name ends up pinned at the cap (best effort)
    assert np.allclose(result.weights.to_numpy(), 0.1)


def test_cap_and_renormalize_allow_short_uses_gross_exposure():
    raw = pd.Series([0.5, -0.5, 0.2, -0.2], index=list("ABCD"))
    result = cap_and_renormalize_weights(
        raw, target_exposure=1.0, max_single_name_weight=0.6, allow_short=True
    )
    assert result.converged is True
    assert result.weights.abs().sum() == pytest.approx(1.0)
    assert (result.weights.abs() <= 0.6 + 1e-9).all()


# ---------------------------------------------------------------------------
# blend_signals_by_rolling_icir
# ---------------------------------------------------------------------------


def _make_blend_fixture(n_days=150, n_inst=12, seed=42, tail_lag=5):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-02", periods=n_days, freq="B")
    cols = [f"I{i:02d}" for i in range(n_inst)]

    common_factor = rng.normal(size=(n_days, n_inst))
    target_vals = 0.9 * common_factor + 1.2 * rng.normal(size=(n_days, n_inst))
    good_vals = common_factor + rng.normal(scale=0.3, size=(n_days, n_inst))
    bad_vals = rng.normal(size=(n_days, n_inst))  # pure noise, ~uncorrelated w/ target

    target_df = pd.DataFrame(target_vals, index=dates, columns=cols)
    # Simulate a properly pre-lagged target: the most recent `tail_lag` days
    # haven't "realized" yet (not knowable at those signal dates). Guarded
    # because `iloc[-0:]` means "all rows", not "zero rows".
    if tail_lag:
        target_df.iloc[-tail_lag:] = np.nan

    good_df = pd.DataFrame(good_vals, index=dates, columns=cols)
    bad_df = pd.DataFrame(bad_vals, index=dates, columns=cols)
    return good_df, bad_df, target_df


def test_blend_warmup_period_is_nan():
    good_df, bad_df, target_df = _make_blend_fixture()
    lookback, min_obs = 40, 15
    blended = blend_signals_by_rolling_icir(
        {"good": good_df, "bad": bad_df}, target_df, lookback=lookback, min_obs=min_obs
    )
    warmup_days = lookback + min_obs
    assert blended.iloc[: warmup_days - 1].isna().all().all()
    # and there IS real (non-NaN) output once past warmup and clear of the
    # tail lag window
    assert blended.iloc[warmup_days:-5].notna().all().all()


def test_blend_favors_the_better_correlated_signal():
    good_df, bad_df, target_df = _make_blend_fixture()
    lookback, min_obs = 40, 15
    blended = blend_signals_by_rolling_icir(
        {"good": good_df, "bad": bad_df}, target_df, lookback=lookback, min_obs=min_obs
    )
    warmup_days = lookback + min_obs
    post_warmup = blended.iloc[warmup_days:]
    good_post = good_df.iloc[warmup_days:]
    bad_post = bad_df.iloc[warmup_days:]

    # Per-day cross-sectional correlation between the blend and each raw
    # signal: if the blend weights favor "good", the blend should track
    # "good" much more closely than it tracks "bad".
    corr_good = [
        post_warmup.loc[d].corr(good_post.loc[d])
        for d in post_warmup.index
        if post_warmup.loc[d].notna().any()
    ]
    corr_bad = [
        post_warmup.loc[d].corr(bad_post.loc[d])
        for d in post_warmup.index
        if post_warmup.loc[d].notna().any()
    ]
    mean_corr_good = float(np.nanmean(corr_good))
    mean_corr_bad = float(np.nanmean(corr_bad))

    assert mean_corr_good > 0.5
    assert mean_corr_good > mean_corr_bad + 0.3


def test_blend_requires_aligned_panels():
    good_df, bad_df, target_df = _make_blend_fixture()
    misaligned = good_df.iloc[:-1]  # drop last row -> index mismatch
    with pytest.raises(ValueError):
        blend_signals_by_rolling_icir({"good": misaligned, "bad": bad_df}, target_df)


def test_blend_warns_when_target_looks_unlagged():
    good_df, bad_df, target_df = _make_blend_fixture(tail_lag=0)  # no lag at all
    with pytest.warns(UserWarning, match="embargo"):
        blend_signals_by_rolling_icir({"good": good_df, "bad": bad_df}, target_df)
