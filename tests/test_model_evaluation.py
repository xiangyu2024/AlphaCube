import numpy as np
import pytest

from alphacube.models.evaluation import (
    EvaluationResult,
    cross_sectional_ic,
    evaluate_target,
    leaderboard,
    rolling_single_stock_ic,
)

N_DAYS = 150
N_INSTRUMENTS = 20
TRUE_SLOPE = 0.5


@pytest.fixture(scope="module")
def correlated_prediction_and_target():
    rng = np.random.default_rng(42)
    prediction = rng.normal(0.0, 1.0, size=(N_DAYS, N_INSTRUMENTS))
    noise = rng.normal(0.0, 0.1, size=(N_DAYS, N_INSTRUMENTS))
    target = TRUE_SLOPE * prediction + noise
    return prediction, target


def test_cross_sectional_ic_is_meaningfully_positive(correlated_prediction_and_target):
    prediction, target = correlated_prediction_and_target
    metrics = cross_sectional_ic(prediction, target)
    assert metrics["mean_ic"] > 0.5
    assert metrics["mean_rank_ic"] > 0.5
    assert metrics["icir"] > 0
    assert metrics["rank_icir"] > 0


def test_cross_sectional_ic_skips_sparse_days():
    # 3 days x 10 instruments; day 0 has only 3 finite pairs (< 5), should
    # be skipped and not contribute a (possibly spurious) IC value.
    prediction = np.full((3, 10), np.nan)
    target = np.full((3, 10), np.nan)
    prediction[0, :3] = [1.0, 2.0, 3.0]
    target[0, :3] = [1.0, 2.0, 3.0]
    prediction[1, :6] = np.arange(6)
    target[1, :6] = np.arange(6) * 2.0
    prediction[2, :6] = np.arange(6)
    target[2, :6] = -np.arange(6)

    metrics = cross_sectional_ic(prediction, target)
    # day 0 skipped (3 pairs < 5); days 1 and 2 have perfect +1/-1 correlation
    assert metrics["mean_ic"] == pytest.approx(0.0, abs=1e-9)


def test_rolling_single_stock_ic_shape_and_summary(correlated_prediction_and_target):
    prediction, target = correlated_prediction_and_target
    panel, summary = rolling_single_stock_ic(prediction, target, window=30)
    assert panel.shape == (N_DAYS, N_INSTRUMENTS)
    for key in ("mean_of_stock_ic", "median_of_stock_ic", "std_of_stock_ic"):
        assert key in summary
        assert np.isfinite(summary[key])
    # Same underlying correlated relationship per-instrument over time, so
    # the per-stock time-series IC should also come out clearly positive.
    assert summary["mean_of_stock_ic"] > 0.5


def test_evaluate_target_end_to_end_calibration_and_sharpe(correlated_prediction_and_target):
    prediction, target = correlated_prediction_and_target
    # Use the (small-magnitude) target itself as the realized-return panel
    # for P&L purposes -- keeps the long-short legs correlated with the
    # signal so Sharpe isn't degenerate, while exercising the separate
    # "returns_panel_for_pnl vs realized_target_panel" plumbing.
    returns_for_pnl = target * 0.01

    result = evaluate_target(
        prediction_panel=prediction,
        realized_target_panel=target,
        returns_panel_for_pnl=returns_for_pnl,
        target_name="synthetic_ret",
        horizon=1,
        rolling_window=30,
    )
    assert isinstance(result, EvaluationResult)

    # IC metrics
    assert result.mean_ic > 0.5
    assert result.icir > 0

    # Rolling per-stock IC
    assert result.rolling_ic_mean_of_stock > 0.5

    # Long-short Sharpe metrics: computed and finite
    assert np.isfinite(result.long_short_sharpe)
    assert np.isfinite(result.long_short_decile_sharpe)

    # Magnitude / calibration metrics: slope should recover the true 0.5
    # scaling factor used to construct the synthetic relationship.
    assert result.calibration_slope == pytest.approx(TRUE_SLOPE, abs=0.05)
    assert abs(result.calibration_intercept) < 0.05
    assert result.rmse > 0
    assert np.isfinite(result.mean_bias)
    assert result.n_valid_pairs == N_DAYS * N_INSTRUMENTS


def test_evaluate_target_uses_returns_panel_for_pnl_not_target():
    # Construct a case where realized_target and returns-for-pnl disagree in
    # sign, so the Sharpe metrics must reflect returns_panel_for_pnl, not
    # realized_target_panel -- proving the two are not conflated internally.
    rng = np.random.default_rng(7)
    n_days, n_instruments = 60, 10
    prediction = rng.normal(0.0, 1.0, size=(n_days, n_instruments))
    target_up = prediction.copy()  # perfectly IC-aligned target
    returns_down = -prediction.copy() * 0.01  # inversely aligned P&L returns

    result = evaluate_target(
        prediction_panel=prediction,
        realized_target_panel=target_up,
        returns_panel_for_pnl=returns_down,
        target_name="sign_flip",
        horizon=1,
    )
    # IC (uses target_up) should be strongly positive...
    assert result.mean_ic > 0.9
    # ...but Sharpe (uses returns_down, inversely aligned) should be negative.
    assert result.long_short_sharpe < 0
    assert result.long_short_decile_sharpe < 0


def test_leaderboard_tabulates_and_sorts_by_icir(correlated_prediction_and_target):
    prediction, target = correlated_prediction_and_target
    good = evaluate_target(prediction, target, target * 0.01, target_name="good", horizon=1)
    noise_only = evaluate_target(
        prediction,
        np.random.default_rng(1).normal(size=prediction.shape),
        target * 0.01,
        target_name="noisy",
        horizon=1,
    )
    df = leaderboard([noise_only, good])
    assert list(df.columns[:2]) == ["target_name", "horizon"]
    assert len(df) == 2
    # sorted by icir descending -- the genuinely correlated one should rank first
    assert df.iloc[0]["target_name"] == "good"
