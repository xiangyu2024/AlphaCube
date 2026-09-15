import numpy as np
import pytest

from alphacube.daily import synthetic_daily
from alphacube.models.trainer import MultiTargetTrainer, TargetSpec, _to_day_indices, _validate_embargo


N_DAYS = 90
N_INSTRUMENTS = 6


@pytest.fixture(scope="module")
def returns_panel():
    # The previous project's version wrote a synthetic OHLCV CSV and loaded it
    # back; `synthetic_daily` is the same random-walk panel without the CSV.
    data = synthetic_daily(n_days=N_DAYS, n_instruments=N_INSTRUMENTS, seed=11)
    close = data.get_field("Close")
    returns = np.full_like(close, np.nan)
    returns[1:] = close[1:] / close[:-1] - 1.0
    return returns


@pytest.fixture(scope="module")
def feature_panels(returns_panel):
    # "lagged return" and "rolling mean of returns" -- simple, deterministic
    # stand-ins for evaluated alpha-factor exposure panels.
    lagged_return = np.full_like(returns_panel, np.nan)
    lagged_return[1:] = returns_panel[:-1]

    n_days = returns_panel.shape[0]
    window = 5
    rolling_mean = np.full_like(returns_panel, np.nan)
    for t in range(window, n_days):
        rolling_mean[t] = np.nanmean(returns_panel[t - window : t], axis=0)

    return {"lagged_return": lagged_return, "rolling_mean_5": rolling_mean}


def test_to_day_indices_supports_bool_mask_and_slice():
    mask = np.zeros(10, dtype=bool)
    mask[2:5] = True
    idx = _to_day_indices(mask, 10)
    np.testing.assert_array_equal(idx, [2, 3, 4])

    idx_slice = _to_day_indices(slice(0, 5), 10)
    np.testing.assert_array_equal(idx_slice, [0, 1, 2, 3, 4])


def test_validate_embargo_raises_on_too_small_gap():
    train_days = np.arange(0, 50)
    holdout_days = np.arange(52, 60)  # gap = 52-49 = 3, horizon=5 -> not enough
    with pytest.raises(ValueError):
        _validate_embargo(train_days, holdout_days, horizon=5)


def test_validate_embargo_accepts_sufficient_gap():
    train_days = np.arange(0, 50)
    holdout_days = np.arange(56, 60)  # gap = 56-49 = 7 > horizon=5
    _validate_embargo(train_days, holdout_days, horizon=5)  # should not raise


def test_trainer_raises_on_embargo_violating_split(feature_panels, returns_panel):
    trainer = MultiTargetTrainer(feature_panels, returns_panel)
    specs = [TargetSpec(name="ret_h5", kind="forward_return", horizon=5, estimator="ridge")]

    train_dates = np.arange(0, 60)
    # Holdout starts right after train ends -- far too close for horizon=5.
    holdout_dates = np.arange(60, 70)

    with pytest.raises(ValueError):
        trainer.run(specs, train_dates, holdout_dates)


def test_trainer_valid_split_produces_full_panel_train_only_predictions(feature_panels, returns_panel):
    trainer = MultiTargetTrainer(feature_panels, returns_panel)

    horizon = 5
    train_dates = np.arange(0, 55)
    holdout_dates = np.arange(65, N_DAYS)  # gap = 65-54 = 11 > horizon=5

    specs = [
        TargetSpec(name="ret_h5", kind="forward_return", horizon=horizon, estimator="ridge"),
        TargetSpec(name="vol_h5", kind="forward_volatility", horizon=horizon, estimator="hgb"),
        TargetSpec(
            name="cls_h5",
            kind="forward_return_class",
            horizon=horizon,
            estimator="hgb_classifier",
            params={"top_frac": 0.3, "bottom_frac": 0.3},
        ),
    ]

    results = trainer.run(specs, train_dates, holdout_dates)
    assert set(results) == {"ret_h5", "vol_h5", "cls_h5"}

    train_day_mask = np.zeros(N_DAYS, dtype=bool)
    train_day_mask[train_dates] = True

    for name, trained in results.items():
        assert trained.predictions.shape == (N_DAYS, N_INSTRUMENTS)
        assert trained.target_panel.shape == (N_DAYS, N_INSTRUMENTS)
        assert trained.feature_names == ["lagged_return", "rolling_mean_5"]

        # Predictions exist (finite) on holdout days too -- full-panel, not
        # train-days-only.
        holdout_preds = trained.predictions[holdout_dates]
        assert np.isfinite(holdout_preds).any(), f"{name}: no finite holdout predictions"

        # Independently recompute the expected training-row count (finite
        # label on a train day, at least one finite feature) and check it
        # matches n_train_rows exactly -- this is the crux check that only
        # train rows (never holdout rows) were used to fit.
        target_panel = trained.target_panel
        feature_valid = np.any(
            np.isfinite(np.stack([feature_panels["lagged_return"], feature_panels["rolling_mean_5"]], axis=-1)),
            axis=-1,
        )
        expected_train_mask = train_day_mask[:, None] & feature_valid & np.isfinite(target_panel)
        assert trained.n_train_rows == int(expected_train_mask.sum())
        assert trained.n_train_rows > 0


def test_trainer_rejects_mismatched_feature_shape(returns_panel):
    bad_features = {"bad": np.ones((returns_panel.shape[0] + 1, returns_panel.shape[1]))}
    with pytest.raises(ValueError):
        MultiTargetTrainer(bad_features, returns_panel)


def test_trainer_rejects_unknown_estimator_name():
    with pytest.raises(ValueError):
        TargetSpec(name="x", kind="forward_return", horizon=1, estimator="not_a_real_estimator")


def test_trainer_rejects_unknown_target_kind():
    with pytest.raises(ValueError):
        TargetSpec(name="x", kind="not_a_real_kind", horizon=1, estimator="ridge")
