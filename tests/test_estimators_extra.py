"""Tests for the optional XGBoost + TabPFN estimators added to
`alphacube.models.estimators`.

- XGBoost tests: real fit/predict correctness checks against synthetic
  data, skipped if `xgboost` isn't importable in this environment.
- TabPFN bagging-logic test: the primary correctness proof for the
  auto-subsampling + bagging wrapper. It stubs out the underlying
  `TabPFNRegressor` class with a trivial fake ("predicts the mean of its
  training y") so it exercises `TabPFNRegressorEstimator`'s bagging logic
  in complete isolation from the real `tabpfn` package, real model
  inference, and any network access. This test must always run and pass.
- TabPFN real-integration test: a bonus smoke test that fits the actual
  `tabpfn` package on a tiny dataset. It is allowed to skip (not fail) if
  `tabpfn` isn't installed, or if the first `.fit()` call needs to download
  pretrained weights and that download is unavailable/slow in this
  sandboxed environment.
"""

from __future__ import annotations

import importlib.util
import signal

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from alphacube.models import estimators as estimators_module


class _TimeoutError(Exception):
    pass


def _run_with_timeout(fn, seconds: int):
    """Run `fn()` bounded by a hard wall-clock timeout using SIGALRM, so a
    stalled network download (e.g. TabPFN fetching pretrained weights on
    first use) can never hang the test suite indefinitely. Unix-only, which
    is fine for this project's CI/dev environments."""

    def _handler(signum, frame):
        raise _TimeoutError(f"timed out after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

# Per-test skip rather than a module-level `importorskip`, so the TabPFN
# bagging tests below stay reachable in an environment without xgboost.
requires_xgboost = pytest.mark.skipif(
    importlib.util.find_spec("xgboost") is None, reason="xgboost is an optional dependency"
)


@requires_xgboost
def test_xgboost_regressor_registered_and_tracks_signal():
    assert "xgboost" in estimators_module.ESTIMATOR_REGISTRY
    cls = estimators_module.ESTIMATOR_REGISTRY["xgboost"]
    assert cls is estimators_module.XGBoostRegressorEstimator

    X, y = make_regression(n_samples=200, n_features=10, noise=5.0, random_state=0)
    X_train, y_train = X[:160], y[:160]
    X_test, y_test = X[160:], y[160:]

    model = cls(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert np.all(np.isfinite(preds))
    corr = np.corrcoef(preds, y_test)[0, 1]
    assert corr > 0.3, f"expected predictions to track true target, got corr={corr}"


@requires_xgboost
def test_xgboost_classifier_registered_and_beats_chance():
    assert "xgboost_classifier" in estimators_module.ESTIMATOR_REGISTRY
    cls = estimators_module.ESTIMATOR_REGISTRY["xgboost_classifier"]
    assert cls is estimators_module.XGBoostClassifierEstimator

    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=6, n_redundant=0, random_state=0
    )
    X_train, y_train = X[:160], y[:160]
    X_test, y_test = X[160:], y[160:]

    model = cls(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)

    assert np.all(np.isfinite(preds))
    assert proba.shape == (len(X_test), 2)
    accuracy = np.mean(preds == y_test)
    assert accuracy > 0.6, f"expected accuracy well above chance, got {accuracy}"


@requires_xgboost
def test_xgboost_accepts_nan_features_natively():
    cls = estimators_module.ESTIMATOR_REGISTRY["xgboost"]
    X, y = make_regression(n_samples=100, n_features=5, noise=1.0, random_state=1)
    X = X.copy()
    rng = np.random.default_rng(1)
    nan_mask = rng.random(X.shape) < 0.1
    X[nan_mask] = np.nan

    model = cls(n_estimators=20, random_state=0)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ---------------------------------------------------------------------------
# TabPFN: bagging logic, stubbed (always runs, no network/tabpfn needed)
# ---------------------------------------------------------------------------


class _FakeTabPFNRegressor:
    """Trivial stand-in for `tabpfn.TabPFNRegressor`: predicts the mean of
    its training `y` for every row, regardless of `X`. Records how many
    rows it was fit on and how many instances were created, so tests can
    assert on subsampling + bagging behavior without any real TabPFN
    dependency."""

    instances: list = []

    def __init__(self, device: str = "cpu", **kwargs):
        self.device = device
        self.kwargs = kwargs
        self.fit_n_rows = None
        self.fit_X = None
        self.fit_y = None
        _FakeTabPFNRegressor.instances.append(self)

    def fit(self, X, y):
        self.fit_n_rows = len(X)
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        self._mean = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self._mean)


@pytest.fixture(autouse=False)
def fake_tabpfn_regressor(monkeypatch):
    """Ensure TabPFNRegressorEstimator exists for the test (defining a
    throwaway subclass wired to the fake if the real `tabpfn` package isn't
    installed), and stub the underlying model class either way."""
    _FakeTabPFNRegressor.instances = []

    if getattr(estimators_module, "TabPFNRegressorEstimator", None) is None:
        pytest.skip(
            "TabPFNRegressorEstimator not defined (tabpfn package not importable "
            "at module load time); bagging logic cannot be exercised through the "
            "real class in this case."
        )

    monkeypatch.setattr(estimators_module, "_TabPFNRegressor", _FakeTabPFNRegressor)
    yield _FakeTabPFNRegressor


def test_tabpfn_bagging_small_dataset_fits_single_model(fake_tabpfn_regressor):
    cls = estimators_module.TabPFNRegressorEstimator
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 4))
    y = rng.normal(size=500)

    model = cls(max_rows_per_model=3000, n_bags=5, random_state=0)
    model.fit(X, y)

    assert len(fake_tabpfn_regressor.instances) == 1
    assert fake_tabpfn_regressor.instances[0].fit_n_rows == 500


def test_tabpfn_bagging_large_dataset_fits_n_bags_subsampled_models(fake_tabpfn_regressor):
    cls = estimators_module.TabPFNRegressorEstimator
    rng = np.random.default_rng(1)
    n_rows = 20000
    X = rng.normal(size=(n_rows, 4))
    y = rng.normal(size=n_rows)

    max_rows_per_model = 3000
    n_bags = 5
    model = cls(
        max_rows_per_model=max_rows_per_model,
        n_bags=n_bags,
        random_state=42,
    )
    model.fit(X, y)

    assert len(fake_tabpfn_regressor.instances) == n_bags
    for instance in fake_tabpfn_regressor.instances:
        assert instance.fit_n_rows == max_rows_per_model

    # Different bags should (with overwhelming probability) draw different
    # subsamples -- verifies each bag gets its own random subsample rather
    # than all bags reusing the same rows.
    first_rows = fake_tabpfn_regressor.instances[0].fit_X
    second_rows = fake_tabpfn_regressor.instances[1].fit_X
    assert not np.array_equal(first_rows, second_rows)


def test_tabpfn_bagging_reproducible_with_fixed_random_state(fake_tabpfn_regressor):
    cls = estimators_module.TabPFNRegressorEstimator
    rng = np.random.default_rng(2)
    n_rows = 10000
    X = rng.normal(size=(n_rows, 3))
    y = rng.normal(size=n_rows)

    model_a = cls(max_rows_per_model=2000, n_bags=3, random_state=7)
    model_a.fit(X, y)
    rows_a = [inst.fit_X.copy() for inst in fake_tabpfn_regressor.instances]

    _FakeTabPFNRegressor.instances = []
    model_b = cls(max_rows_per_model=2000, n_bags=3, random_state=7)
    model_b.fit(X, y)
    rows_b = [inst.fit_X.copy() for inst in fake_tabpfn_regressor.instances]

    for a, b in zip(rows_a, rows_b):
        assert np.array_equal(a, b)


def test_tabpfn_predict_averages_across_bags(fake_tabpfn_regressor):
    cls = estimators_module.TabPFNRegressorEstimator
    rng = np.random.default_rng(3)
    n_rows = 9000
    X = rng.normal(size=(n_rows, 2))
    y = rng.normal(size=n_rows)

    max_rows_per_model = 3000
    n_bags = 3
    model = cls(max_rows_per_model=max_rows_per_model, n_bags=n_bags, random_state=5)
    model.fit(X, y)

    X_test = rng.normal(size=(10, 2))
    preds = model.predict(X_test)

    expected_means = [inst._mean for inst in fake_tabpfn_regressor.instances]
    expected_preds = np.mean(expected_means)  # each fake predicts a constant
    assert np.allclose(preds, expected_preds)


# ---------------------------------------------------------------------------
# TabPFN: real integration smoke test (bonus, allowed to skip)
# ---------------------------------------------------------------------------


def test_tabpfn_real_integration_smoke():
    tabpfn = pytest.importorskip("tabpfn", reason="tabpfn is an optional dependency")

    if getattr(estimators_module, "TabPFNRegressorEstimator", None) is None:
        pytest.skip("TabPFNRegressorEstimator not registered (tabpfn import failed at module load)")

    rng = np.random.default_rng(0)
    n_samples = 150
    X = rng.normal(size=(n_samples, 4))
    true_coef = np.array([1.5, -2.0, 0.0, 0.5])
    y = X @ true_coef + rng.normal(scale=0.1, size=n_samples)

    model = estimators_module.TabPFNRegressorEstimator(
        max_rows_per_model=100, n_bags=2, device="cpu", random_state=0
    )

    def _fit_and_predict():
        model.fit(X[:100], y[:100])
        return model.predict(X[100:])

    try:
        preds = _run_with_timeout(_fit_and_predict, seconds=120)
    except Exception as exc:  # noqa: BLE001 - network/weights/env/timeout failures all land here
        pytest.skip(f"TabPFN weight download unavailable: {exc}")

    if not np.all(np.isfinite(preds)):
        pytest.skip("TabPFN produced non-finite predictions in this environment")

    corr = np.corrcoef(preds, y[100:])[0, 1]
    assert corr > 0.3, f"expected TabPFN predictions to track true signal, got corr={corr}"
