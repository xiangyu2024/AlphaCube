import numpy as np
import pytest

from alphacube.models import targets as T
from alphacube.models.estimators import ESTIMATOR_REGISTRY
from alphacube.models.horizon_mixture import (
    HorizonMixtureEstimator,
    build_candidate_horizon_labels,
)


# ---------------------------------------------------------------------------
# Synthetic "label horizon paradox" panel.
#
# Story: an observable per-(day, instrument) alpha feature drives a
# *decaying* impulse response on daily returns over the next 5 days only
# (`_DECAY`, length 5) -- i.e. the *signal* component of a forward-return
# label saturates once horizon >= 5, because no more alpha-driven drift is
# added past day 5. Idiosyncratic daily noise, by contrast, is iid and keeps
# accumulating in the compounded forward return as horizon grows. On top of
# that, `_DECAY`-weighted contributions from *other* (temporally
# uncorrelated) alpha draws also keep entering the window as horizon grows,
# which -- from the point of view of a model that only observes today's
# alpha value as a feature -- behaves like additional unexplained variance.
# Net effect: horizon-5/10 labels have a much cleaner alpha<->label
# relationship than horizon-40's own (very noisy) label, even though the
# *evaluation* target everyone actually cares about is horizon 40. A handful
# of pure-noise "junk" feature columns are included alongside the real alpha
# feature so a flexible tree model (HGB) has room to overfit the noisier
# long-horizon label, which is exactly the failure mode the paper's argument
# (and this module) is about.
# ---------------------------------------------------------------------------

_DECAY = np.array([0.5, 0.4, 0.3, 0.2, 0.1])  # impulse response, lags 1..5
_SIGNAL_SCALE = 0.02
_NOISE_SIGMA = 0.01
_N_JUNK_FEATURES = 5


def _build_synthetic_panel(n_days: int, n_instruments: int, seed: int):
    rng = np.random.default_rng(seed)
    alpha = rng.normal(0.0, 1.0, size=(n_days, n_instruments))
    noise = rng.normal(0.0, _NOISE_SIGMA, size=(n_days, n_instruments))

    daily_returns = noise.copy()
    for k, w in enumerate(_DECAY, start=1):
        # alpha[t - k] contributes to daily_returns[t] via lag k.
        daily_returns[k:] += _SIGNAL_SCALE * w * alpha[:-k]

    junk = rng.normal(0.0, 1.0, size=(n_days, n_instruments, _N_JUNK_FEATURES))
    # Feature stack: [alpha, junk_0, ..., junk_4] -> (n_days, n_instruments, 1 + n_junk)
    features = np.concatenate([alpha[..., None], junk], axis=-1)
    return daily_returns, features


def _flatten_days(arr_2d_or_3d: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Select `days` along axis 0, then flatten the (day, instrument, ...)
    leading axes into one row axis, day-major (matches how every candidate
    horizon's label and the feature matrix are flattened consistently)."""
    sub = arr_2d_or_3d[days]
    if sub.ndim == 2:
        return sub.reshape(-1)
    return sub.reshape(-1, sub.shape[-1])


@pytest.fixture(scope="module")
def synthetic_panel():
    n_days, n_instruments = 520, 40
    daily_returns, features = _build_synthetic_panel(n_days, n_instruments, seed=7)
    candidate_horizons = [5, 10, 20, 40]
    target_horizon = 40
    y_by_horizon_full = build_candidate_horizon_labels(daily_returns, candidate_horizons)
    # target_horizon (40) is already included since it's also a candidate here.
    return {
        "n_days": n_days,
        "n_instruments": n_instruments,
        "daily_returns": daily_returns,
        "features": features,
        "candidate_horizons": candidate_horizons,
        "target_horizon": target_horizon,
        "y_by_horizon_full": y_by_horizon_full,
    }


def test_core_empirical_claim_mixture_beats_single_horizon_baseline(synthetic_panel):
    n_days = synthetic_panel["n_days"]
    features = synthetic_panel["features"]
    candidate_horizons = synthetic_panel["candidate_horizons"]
    target_horizon = synthetic_panel["target_horizon"]
    y_by_horizon_full = synthetic_panel["y_by_horizon_full"]

    max_horizon = max(candidate_horizons)
    # Train block, then an embargo gap > max_horizon, then a fully disjoint
    # test block the estimator never sees (not even via its internal
    # fit/val split, which only ever touches `train_days`).
    train_days = np.arange(0, 350)
    test_days = np.arange(350 + max_horizon + 1, n_days)
    assert test_days.size > 1000 // synthetic_panel["n_instruments"]  # sanity: nonempty

    X_train = _flatten_days(features, train_days)
    y_by_horizon_train = {h: _flatten_days(y_by_horizon_full[h], train_days) for h in candidate_horizons}

    X_test = _flatten_days(features, test_days)
    y_test_true_horizon = _flatten_days(y_by_horizon_full[target_horizon], test_days)

    # --- Fit the mixture estimator ---
    mixture = HorizonMixtureEstimator(
        candidate_horizons=candidate_horizons,
        target_horizon=target_horizon,
        base_estimator="hgb",
        base_estimator_params={"max_iter": 50, "random_state": 0},
        mixture_val_frac=0.25,
        random_state=0,
    )
    mixture.fit(X_train, y_by_horizon_train)

    # (a) runs and produces finite predictions
    mixture_preds = mixture.predict(X_test)
    assert np.isfinite(mixture_preds).all()

    # (b) meaningful, non-trivial weight on the shorter, more-informative
    # horizons -- not everything dumped on a single arbitrary horizon.
    weights = mixture.mixture_weights_
    assert set(weights) == set(candidate_horizons)
    assert all(w >= 0.0 for w in weights.values())
    assert np.isclose(sum(weights.values()), 1.0, atol=1e-6)
    short_weight = weights[5] + weights[10]
    long_weight = weights[20] + weights[40]
    assert short_weight > long_weight, (
        f"expected short-horizon (5,10) proxies to dominate the mixture weights, got {weights}"
    )
    # Genuinely non-trivial: more than one horizon carries real weight.
    n_meaningful = sum(1 for w in weights.values() if w > 0.05)
    assert n_meaningful >= 2, f"expected weight spread across >=2 horizons, got {weights}"

    # --- Baseline: a single hgb model trained directly on the noisy
    # horizon-40 label alone, same train rows, same base hyperparameters. ---
    finite_mask = np.isfinite(y_by_horizon_train[target_horizon])
    baseline = ESTIMATOR_REGISTRY["hgb"](max_iter=50, random_state=0)
    baseline.fit(X_train[finite_mask], y_by_horizon_train[target_horizon][finite_mask])
    baseline_preds = baseline.predict(X_test)

    # (c) IC comparison on a held-out test set disjoint from fit AND
    # internal val fold.
    test_finite = np.isfinite(y_test_true_horizon)
    mixture_ic = np.corrcoef(mixture_preds[test_finite], y_test_true_horizon[test_finite])[0, 1]
    baseline_ic = np.corrcoef(baseline_preds[test_finite], y_test_true_horizon[test_finite])[0, 1]

    assert np.isfinite(mixture_ic) and np.isfinite(baseline_ic)
    print(f"\n[horizon_mixture] mixture_weights_={weights}")
    print(f"[horizon_mixture] mixture_ic={mixture_ic:.4f} baseline_ic={baseline_ic:.4f}")

    assert mixture_ic >= baseline_ic - 1e-9
    assert mixture_ic > baseline_ic  # demonstrate a real, not just tied, improvement


def test_row_alignment_with_different_nan_patterns_per_horizon():
    rng = np.random.default_rng(3)
    n_rows, n_features = 400, 3
    X = rng.normal(size=(n_rows, n_features))

    base_signal = X[:, 0]
    y5 = 2.0 * base_signal + rng.normal(0.0, 0.1, size=n_rows)
    y10 = 2.0 * base_signal + rng.normal(0.0, 0.3, size=n_rows)
    y40 = 2.0 * base_signal + rng.normal(0.0, 1.0, size=n_rows)

    # Different, deliberately staggered trailing-NaN tails per horizon (as
    # forward_return would produce: longer horizon -> more trailing NaN).
    y5 = y5.copy()
    y5[-5:] = np.nan
    y10 = y10.copy()
    y10[-15:] = np.nan
    y40 = y40.copy()
    y40[-40:] = np.nan

    y_by_horizon = {5: y5, 10: y10, 40: y40}

    est = HorizonMixtureEstimator(
        candidate_horizons=[5, 10, 40],
        target_horizon=40,
        base_estimator="ridge",
        mixture_val_frac=0.2,
        random_state=1,
    )
    est.fit(X, y_by_horizon)

    weights = est.mixture_weights_
    assert set(weights) == {5, 10, 40}
    assert all(np.isfinite(w) for w in weights.values())
    assert any(w > 0.0 for w in weights.values())
    assert np.isclose(sum(weights.values()), 1.0, atol=1e-6)

    preds = est.predict(X)
    assert preds.shape == (n_rows,)
    assert np.isfinite(preds).all()


def test_missing_target_horizon_raises_value_error():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    y_by_horizon = {5: rng.normal(size=100), 10: rng.normal(size=100)}

    est = HorizonMixtureEstimator(candidate_horizons=[5, 10], target_horizon=20)
    with pytest.raises(ValueError, match="target_horizon"):
        est.fit(X, y_by_horizon)


def test_missing_candidate_horizon_raises_value_error():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    # target_horizon (20) present, but candidate horizon 10 is missing.
    y_by_horizon = {5: rng.normal(size=100), 20: rng.normal(size=100)}

    est = HorizonMixtureEstimator(candidate_horizons=[5, 10], target_horizon=20)
    with pytest.raises(ValueError, match="candidate_horizon"):
        est.fit(X, y_by_horizon)


def test_build_candidate_horizon_labels_matches_manual_forward_return():
    rng = np.random.default_rng(5)
    returns_panel = rng.normal(0.0, 0.02, size=(60, 8))
    candidate_horizons = [3, 7, 15]

    result = build_candidate_horizon_labels(returns_panel, candidate_horizons)

    assert set(result) == set(candidate_horizons)
    for h in candidate_horizons:
        expected = T.forward_return(returns_panel, h, from_returns=True)
        np.testing.assert_array_equal(result[h], expected)
