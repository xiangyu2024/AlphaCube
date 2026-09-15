"""`ModelEstimator` ABC + sklearn-backed concrete estimators, plus a small
name -> class registry for config-driven selection (`ESTIMATOR_REGISTRY`).

`X` is always a flattened `(n_samples, n_features)` matrix — one row per
`(day, instrument)` pair with a valid label (see `trainer.py` for the
flattening). Factor-exposure feature panels routinely contain NaN (warm-up
windows, missing data), so the default gradient-boosting estimators use
`sklearn`'s `HistGradientBoosting*`, which accepts NaN features natively;
the linear baselines impute/scale via a `Pipeline` first.

`lightgbm`, `xgboost`, and `tabpfn` are all optional extras: each is
imported defensively at module load, and its estimator classes + registry
entries only appear if the import succeeded. `scikit-learn` is the only new
*hard* dependency this module adds to the project.

  - `pip install xgboost` (>=2.0) unlocks `XGBoostRegressorEstimator` /
    `XGBoostClassifierEstimator` (registry keys `"xgboost"` /
    `"xgboost_classifier"`). Like `HistGradientBoosting*`, XGBoost's tree
    models accept NaN features natively, so no imputation pipeline is used.
  - `pip install tabpfn` unlocks `TabPFNRegressorEstimator` /
    `TabPFNClassifierEstimator` (registry keys `"tabpfn"` /
    `"tabpfn_classifier"`) — a tabular foundation model (in-context-learning
    transformer, no gradient training at predict-time). See those classes'
    docstrings for the row-count caveats and first-use network download.
"""

from __future__ import annotations

import os

# `lightgbm`/`xgboost` (and `tabpfn`'s own torch-adjacent stack) each bundle
# their own OpenMP runtime. On macOS, loading more than one `libomp.dylib`
# into the same process (e.g. alongside the one `torch` already loaded) is a
# known source of a duplicate-init crash -- non-deterministic segfaults, with
# the crash traceback often pointing at unrelated code because it's memory
# corruption, not a fault in the library actually at fault.
# `KMP_DUPLICATE_LIB_OK=TRUE` alone is the commonly-cited workaround but was
# empirically *not* sufficient here to stop the segfault once `torch` had
# already initialized its OpenMP runtime; `OMP_NUM_THREADS=1` (which sidesteps
# the conflicting runtimes' thread-pool init entirely) was additionally
# required to reliably prevent the crash in testing. Both are set only as
# defaults so an explicit environment setting from the caller is never
# overridden. Must happen before the `import lightgbm` / `import xgboost`
# below.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as _lightgbm
except ImportError:  # pragma: no cover - optional dependency
    _lightgbm = None

try:
    import xgboost as _xgboost
except ImportError:  # pragma: no cover - optional dependency
    _xgboost = None

try:
    from tabpfn import TabPFNClassifier as _TabPFNClassifier
    from tabpfn import TabPFNRegressor as _TabPFNRegressor
except ImportError:  # pragma: no cover - optional dependency
    _TabPFNRegressor = None
    _TabPFNClassifier = None


class ModelEstimator(ABC):
    """Common interface every concrete estimator below implements."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ModelEstimator":
        """Fit on flattened `(n_samples, n_features)` X and `(n_samples,)` y."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on flattened `(n_samples, n_features)` X."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities; only meaningful for classifiers."""
        raise NotImplementedError(f"{type(self).__name__} does not support predict_proba")


def _linear_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


class RidgeRegressorEstimator(ModelEstimator):
    """Linear baseline: mean-impute + standardize + `sklearn.linear_model.Ridge`."""

    def __init__(self, alpha: float = 1.0, **kwargs: Any) -> None:
        self._pipeline = _linear_pipeline(Ridge(alpha=alpha, **kwargs))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressorEstimator":
        self._pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict(X)


class GradientBoostingRegressorEstimator(ModelEstimator):
    """Default gradient-boosting regressor: `HistGradientBoostingRegressor`
    (native NaN support, no imputation needed — matters for factor panels)."""

    def __init__(self, **kwargs: Any) -> None:
        self._model = HistGradientBoostingRegressor(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressorEstimator":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


class LogisticRegressionEstimator(ModelEstimator):
    """Classification baseline: mean-impute + standardize + `LogisticRegression`."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("max_iter", 1000)
        self._pipeline = _linear_pipeline(LogisticRegression(**kwargs))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionEstimator":
        self._pipeline.fit(X, np.asarray(y).astype(int))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict_proba(X)


class GradientBoostingClassifierEstimator(ModelEstimator):
    """Default gradient-boosting classifier: `HistGradientBoostingClassifier`."""

    def __init__(self, **kwargs: Any) -> None:
        self._model = HistGradientBoostingClassifier(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingClassifierEstimator":
        self._model.fit(X, np.asarray(y).astype(int))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)


ESTIMATOR_REGISTRY: Dict[str, Type[ModelEstimator]] = {
    "ridge": RidgeRegressorEstimator,
    "hgb": GradientBoostingRegressorEstimator,
    "logistic": LogisticRegressionEstimator,
    "hgb_classifier": GradientBoostingClassifierEstimator,
}


if _lightgbm is not None:  # pragma: no cover - exercised only when installed

    class LightGBMRegressorEstimator(ModelEstimator):
        """Optional: `lightgbm.LGBMRegressor`, only registered if `lightgbm`
        is importable. Not a hard dependency of this project."""

        def __init__(self, **kwargs: Any) -> None:
            kwargs.setdefault("verbosity", -1)
            self._model = _lightgbm.LGBMRegressor(**kwargs)

        def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMRegressorEstimator":
            self._model.fit(X, y)
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            return self._model.predict(X)

    class LightGBMClassifierEstimator(ModelEstimator):
        """Optional: `lightgbm.LGBMClassifier`, only registered if `lightgbm`
        is importable. Not a hard dependency of this project."""

        def __init__(self, **kwargs: Any) -> None:
            kwargs.setdefault("verbosity", -1)
            self._model = _lightgbm.LGBMClassifier(**kwargs)

        def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMClassifierEstimator":
            self._model.fit(X, np.asarray(y).astype(int))
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            return self._model.predict(X)

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return self._model.predict_proba(X)

    ESTIMATOR_REGISTRY["lightgbm"] = LightGBMRegressorEstimator
    ESTIMATOR_REGISTRY["lightgbm_classifier"] = LightGBMClassifierEstimator


if _xgboost is not None:  # pragma: no cover - exercised only when installed

    class XGBoostRegressorEstimator(ModelEstimator):
        """Optional: `xgboost.XGBRegressor`, only registered if `xgboost`
        (>=2.0) is importable. Not a hard dependency of this project. Like
        `HistGradientBoostingRegressor`, XGBoost's tree models accept NaN
        features natively, so no imputation pipeline is needed here."""

        def __init__(self, **kwargs: Any) -> None:
            kwargs.setdefault("verbosity", 0)
            self._model = _xgboost.XGBRegressor(**kwargs)

        def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostRegressorEstimator":
            self._model.fit(X, y)
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            return self._model.predict(X)

    class XGBoostClassifierEstimator(ModelEstimator):
        """Optional: `xgboost.XGBClassifier`, only registered if `xgboost`
        (>=2.0) is importable. Not a hard dependency of this project."""

        def __init__(self, **kwargs: Any) -> None:
            kwargs.setdefault("verbosity", 0)
            self._model = _xgboost.XGBClassifier(**kwargs)

        def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostClassifierEstimator":
            self._model.fit(X, np.asarray(y).astype(int))
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            return self._model.predict(X)

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return self._model.predict_proba(X)

    ESTIMATOR_REGISTRY["xgboost"] = XGBoostRegressorEstimator
    ESTIMATOR_REGISTRY["xgboost_classifier"] = XGBoostClassifierEstimator


if _TabPFNRegressor is not None and _TabPFNClassifier is not None:  # pragma: no cover - exercised only when installed

    def _tabpfn_bag_seeds(random_state: Optional[int], n_bags: int) -> List[Optional[int]]:
        """Derive one reproducible per-bag seed from a base `random_state`.

        `None` (no fixed seed requested) is propagated as `None` for every
        bag so behavior without an explicit seed stays non-deterministic,
        matching sklearn's own `random_state=None` convention.
        """
        if random_state is None:
            return [None] * n_bags
        return [int(random_state) + i for i in range(n_bags)]

    class TabPFNRegressorEstimator(ModelEstimator):
        """Optional: wraps `tabpfn.TabPFNRegressor` (a tabular foundation
        model / in-context-learning transformer — no gradient training at
        predict-time) for interface consistency with the rest of this file.
        Only registered if `tabpfn` is importable; not a hard dependency of
        this project (`pip install tabpfn`).

        Two practical caveats, both handled automatically by this wrapper:

        1. **Row-count limit.** TabPFN's cost scales with training-row count
           (attention over the whole context), and CPU inference becomes
           impractically slow beyond roughly 1,000-5,000 rows; the package
           can raise rather than silently truncate above its supported
           limit. `MultiTargetTrainer`'s flattened `(day, instrument)`
           training sets routinely have tens of thousands of rows, so a
           naive single `.fit(X, y)` call is often too slow or fails
           outright. This wrapper auto-subsamples: when
           `len(X) > max_rows_per_model` it fits `n_bags` independent
           TabPFN instances, each on its own random subsample of
           `max_rows_per_model` rows (seeded off `random_state` for
           reproducibility), and `.predict` averages their outputs. This
           "bag independent in-context models, each within the native
           context limit, and average" pattern is the standard workaround
           used across the TabPFN ecosystem for datasets larger than its
           context limit — see https://github.com/PriorLabs/TabPFN — it is
           not something invented for this project.
        2. **First-use network download.** The underlying `tabpfn` package
           may download pretrained weights from the internet the first time
           a model actually calls `.fit()`/`.predict()`. That download is
           not triggered by importing this module or constructing this
           estimator — only by the first real fit — so callers in offline
           or sandboxed environments should expect that first call to fail
           or hang unless the weights are already cached locally.
        """

        def __init__(
            self,
            max_rows_per_model: int = 3000,
            n_bags: int = 5,
            device: str = "cpu",
            random_state: Optional[int] = None,
            **kwargs: Any,
        ) -> None:
            self.max_rows_per_model = max_rows_per_model
            self.n_bags = n_bags
            self.device = device
            self.random_state = random_state
            self._kwargs = kwargs
            self._models: List[Any] = []

        def fit(self, X: np.ndarray, y: np.ndarray) -> "TabPFNRegressorEstimator":
            X = np.asarray(X)
            y = np.asarray(y)
            self._models = []
            if len(X) <= self.max_rows_per_model:
                model = _TabPFNRegressor(device=self.device, **self._kwargs)
                model.fit(X, y)
                self._models.append(model)
                return self

            seeds = _tabpfn_bag_seeds(self.random_state, self.n_bags)
            n_rows = len(X)
            for seed in seeds:
                rng = np.random.default_rng(seed)
                idx = rng.choice(n_rows, size=self.max_rows_per_model, replace=False)
                model = _TabPFNRegressor(device=self.device, **self._kwargs)
                model.fit(X[idx], y[idx])
                self._models.append(model)
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            preds = np.stack([model.predict(X) for model in self._models], axis=0)
            return preds.mean(axis=0)

    class TabPFNClassifierEstimator(ModelEstimator):
        """Optional: wraps `tabpfn.TabPFNClassifier` with the same automatic
        subsampling + bagging as `TabPFNRegressorEstimator` (see its
        docstring for the row-count and first-use network-download caveats,
        and the citation of this bagging workaround as standard TabPFN
        practice, not a project-specific invention). `.predict` averages
        `predict_proba` across bags and takes the argmax."""

        def __init__(
            self,
            max_rows_per_model: int = 3000,
            n_bags: int = 5,
            device: str = "cpu",
            random_state: Optional[int] = None,
            **kwargs: Any,
        ) -> None:
            self.max_rows_per_model = max_rows_per_model
            self.n_bags = n_bags
            self.device = device
            self.random_state = random_state
            self._kwargs = kwargs
            self._models: List[Any] = []
            self._classes: Optional[np.ndarray] = None

        def fit(self, X: np.ndarray, y: np.ndarray) -> "TabPFNClassifierEstimator":
            X = np.asarray(X)
            y = np.asarray(y).astype(int)
            self._classes = np.unique(y)
            self._models = []
            if len(X) <= self.max_rows_per_model:
                model = _TabPFNClassifier(device=self.device, **self._kwargs)
                model.fit(X, y)
                self._models.append(model)
                return self

            seeds = _tabpfn_bag_seeds(self.random_state, self.n_bags)
            n_rows = len(X)
            for seed in seeds:
                rng = np.random.default_rng(seed)
                idx = rng.choice(n_rows, size=self.max_rows_per_model, replace=False)
                model = _TabPFNClassifier(device=self.device, **self._kwargs)
                model.fit(X[idx], y[idx])
                self._models.append(model)
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            proba = self.predict_proba(X)
            classes = self._classes if self._classes is not None else np.arange(proba.shape[1])
            return classes[np.argmax(proba, axis=1)]

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            probas = np.stack([model.predict_proba(X) for model in self._models], axis=0)
            return probas.mean(axis=0)

    ESTIMATOR_REGISTRY["tabpfn"] = TabPFNRegressorEstimator
    ESTIMATOR_REGISTRY["tabpfn_classifier"] = TabPFNClassifierEstimator
