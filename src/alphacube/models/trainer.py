"""`MultiTargetTrainer`: fits one estimator per `TargetSpec` on flattened,
leakage-safe train rows and returns full-panel predictions.

Feature panels here stand in for "mined alpha factor exposures" — in a real
pipeline these come from evaluating a pool member's expression through the
cube engine into a daily `(D, S)` panel (see `mining/pool.py`), but this
module accepts the already-evaluated `dict[str, np.ndarray]` directly,
decoupled from that machinery.

**Row-selection design.** Every `(day, instrument)` pair is a candidate row.
A row is *feature-usable* if at least one feature is finite for it (an
all-NaN row, e.g. an instrument not yet listed, has nothing to predict
from). Training rows are further restricted to `train_dates` days with a
finite label. Individual NaN feature values are passed straight through to
the estimator: `HistGradientBoosting*` (the default) handles NaN natively,
and the linear estimators impute via their internal `Pipeline` — so a row
does *not* need every feature finite to be used, only not-all-NaN.
Prediction rows are feature-usable rows across *all* days (train + holdout),
so a holdout-only prediction panel always exists even though holdout labels
are never used for fitting.

**Embargo.** A label for day `t` at horizon `h` is derived from data up to
day `t + h` (see `targets.py`). So if the holdout window starts at day
`first_holdout`, the last train day usable for training must satisfy
`last_train + h < first_holdout` — otherwise a "train" row's label would
have been computed using data that overlaps the holdout period, which is
look-ahead leakage across the split, not just within a panel. This is
checked explicitly per `TargetSpec` (each may have a different horizon) and
raises `ValueError` rather than trusting the caller's split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from . import targets as T
from .estimators import ESTIMATOR_REGISTRY, ModelEstimator

DayIndexLike = Union[np.ndarray, Sequence[int], Sequence[bool], slice]

CLASSIFICATION_KINDS = frozenset({"forward_return_class", "spike", "crash"})
_VALID_KINDS = frozenset(
    {
        "forward_return",
        "forward_volatility",
        "forward_downside_volatility",
        "forward_return_rank",
        "forward_return_class",
        "spike",
        "crash",
    }
)


@dataclass
class TargetSpec:
    """One (target kind, horizon, estimator) row to train.

    `name` is an arbitrary label used as the results-dict key (e.g.
    `"ret_h5"`). `kind` selects which `targets.py` function builds the label
    panel; `spike`/`crash` both call `forward_event_label`. `params` are
    passed through to that target function (e.g. `mar`, `top_frac`,
    `bottom_frac`, `pct`, `threshold`, `quantile`). `estimator` is a key into
    `ESTIMATOR_REGISTRY`; `estimator_params` are passed to its constructor.
    `task` ("regression"/"classification") is auto-derived from `kind` if
    left `None`.
    """

    name: str
    kind: str
    horizon: int
    estimator: str
    params: Dict[str, Any] = field(default_factory=dict)
    estimator_params: Dict[str, Any] = field(default_factory=dict)
    task: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"Unknown target kind '{self.kind}'. Valid kinds: {sorted(_VALID_KINDS)}")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.estimator not in ESTIMATOR_REGISTRY:
            raise ValueError(
                f"Unknown estimator '{self.estimator}'. Registered: {sorted(ESTIMATOR_REGISTRY)}"
            )
        if self.task is None:
            self.task = "classification" if self.kind in CLASSIFICATION_KINDS else "regression"


@dataclass
class TrainedTarget:
    """Result of training one `TargetSpec`."""

    spec: TargetSpec
    estimator: ModelEstimator
    predictions: np.ndarray  # (n_days, n_instruments), NaN where not feature-usable
    target_panel: np.ndarray  # realized label panel, (n_days, n_instruments)
    feature_names: List[str]
    n_train_rows: int


def compute_target(returns_panel: np.ndarray, spec: TargetSpec) -> np.ndarray:
    """Dispatch to the right `targets.py` function for `spec.kind`."""
    if spec.kind == "forward_return":
        return T.forward_return(returns_panel, spec.horizon, from_returns=True)
    if spec.kind == "forward_volatility":
        return T.forward_volatility(returns_panel, spec.horizon)
    if spec.kind == "forward_downside_volatility":
        return T.forward_downside_volatility(
            returns_panel, spec.horizon, mar=spec.params.get("mar", 0.0)
        )
    if spec.kind == "forward_return_rank":
        return T.forward_return_rank(returns_panel, spec.horizon, pct=spec.params.get("pct", True))
    if spec.kind == "forward_return_class":
        return T.forward_return_class(
            returns_panel,
            spec.horizon,
            top_frac=spec.params.get("top_frac", 0.3),
            bottom_frac=spec.params.get("bottom_frac", 0.3),
        )
    if spec.kind in ("spike", "crash"):
        return T.forward_event_label(
            returns_panel,
            spec.horizon,
            kind=spec.kind,
            threshold=spec.params.get("threshold"),
            quantile=spec.params.get("quantile"),
        )
    raise ValueError(f"Unknown target kind '{spec.kind}'")  # pragma: no cover - guarded in __post_init__


def _to_day_indices(days: DayIndexLike, n_days: int) -> np.ndarray:
    """Normalize a boolean mask / integer index list / slice into a sorted,
    de-duplicated array of day indices in `[0, n_days)`."""
    if isinstance(days, slice):
        idx = np.arange(n_days)[days]
    else:
        arr = np.asarray(days)
        if arr.dtype == bool:
            if arr.shape[0] != n_days:
                raise ValueError(f"boolean day mask has length {arr.shape[0]}, expected {n_days}")
            idx = np.flatnonzero(arr)
        else:
            idx = arr.astype(int)
    if idx.size and (idx.min() < 0 or idx.max() >= n_days):
        raise ValueError("day index out of range [0, n_days)")
    return np.unique(idx)


def _validate_embargo(train_days: np.ndarray, holdout_days: np.ndarray, horizon: int) -> None:
    """Raise if the train/holdout split doesn't leave a `horizon`-day embargo.

    A train-day label at day `t` is derived from data through `t + horizon`.
    For that label to never touch the holdout window, the last train day
    `last_train` and the first holdout day `first_holdout` must satisfy
    `last_train + horizon < first_holdout`, i.e. an (exclusive) gap strictly
    greater than `horizon` days.
    """
    if train_days.size == 0 or holdout_days.size == 0:
        return
    last_train = int(train_days.max())
    first_holdout = int(holdout_days.min())
    if first_holdout <= last_train:
        raise ValueError(
            f"holdout must start strictly after train ends: last_train={last_train}, "
            f"first_holdout={first_holdout}"
        )
    gap = first_holdout - last_train
    if gap <= horizon:
        raise ValueError(
            f"embargo gap too small for horizon={horizon}: last_train={last_train}, "
            f"first_holdout={first_holdout} (gap={gap} days). Need gap > horizon, i.e. "
            f"first_holdout - last_train >= horizon + 1, since a train label at "
            f"last_train depends on data through last_train + horizon."
        )


def stack_features(feature_panels: Dict[str, np.ndarray]) -> "tuple[np.ndarray, List[str]]":
    """Stack a `{name: (n_days, n_instruments)}` dict into one
    `(n_days, n_instruments, n_features)` array, preserving dict insertion
    order for `feature_names` (deterministic given a deterministic caller)."""
    if not feature_panels:
        raise ValueError("feature_panels must be non-empty")
    names = list(feature_panels)
    shape = np.asarray(feature_panels[names[0]]).shape
    arrays = []
    for name in names:
        arr = np.asarray(feature_panels[name], dtype=np.float64)
        if arr.shape != shape:
            raise ValueError(
                f"feature '{name}' has shape {arr.shape}, expected {shape} (all feature "
                "panels and the returns panel must share shape)"
            )
        arrays.append(arr)
    stacked = np.stack(arrays, axis=-1)
    return stacked, names


class MultiTargetTrainer:
    """Trains one estimator per `TargetSpec` against a shared set of feature
    panels and a returns panel, producing full-panel predictions."""

    def __init__(self, feature_panels: Dict[str, np.ndarray], returns_panel: np.ndarray) -> None:
        self.returns_panel = np.asarray(returns_panel, dtype=np.float64)
        if self.returns_panel.ndim != 2:
            raise ValueError("returns_panel must be 2D (n_days, n_instruments)")
        self.n_days, self.n_instruments = self.returns_panel.shape

        self._feature_stack, self.feature_names = stack_features(feature_panels)
        if self._feature_stack.shape[:2] != self.returns_panel.shape:
            raise ValueError(
                f"feature panels have shape {self._feature_stack.shape[:2]}, expected "
                f"{self.returns_panel.shape} to match returns_panel"
            )

        # Row-usable mask: at least one feature finite for that (day, instrument).
        self._feature_valid = np.any(np.isfinite(self._feature_stack), axis=-1)

    def run(
        self,
        specs: Sequence[TargetSpec],
        train_dates: DayIndexLike,
        holdout_dates: DayIndexLike,
    ) -> Dict[str, TrainedTarget]:
        train_days = _to_day_indices(train_dates, self.n_days)
        holdout_days = _to_day_indices(holdout_dates, self.n_days)
        if train_days.size == 0:
            raise ValueError("train_dates selects zero days")

        train_day_mask = np.zeros(self.n_days, dtype=bool)
        train_day_mask[train_days] = True

        n_features = len(self.feature_names)
        X_flat = self._feature_stack.reshape(self.n_days * self.n_instruments, n_features)
        feature_valid_flat = self._feature_valid.reshape(-1)

        results: Dict[str, TrainedTarget] = {}
        for spec in specs:
            _validate_embargo(train_days, holdout_days, spec.horizon)

            target_panel = compute_target(self.returns_panel, spec)
            target_flat = target_panel.reshape(-1)
            label_finite_flat = np.isfinite(target_flat)

            train_row_mask = (
                np.repeat(train_day_mask, self.n_instruments) & feature_valid_flat & label_finite_flat
            )
            n_train_rows = int(train_row_mask.sum())
            if n_train_rows == 0:
                raise ValueError(
                    f"target '{spec.name}': zero training rows (no day/instrument pair in "
                    "train_dates has both a finite label and a usable feature row)"
                )

            X_train = X_flat[train_row_mask]
            y_train = target_flat[train_row_mask]
            if spec.task == "classification":
                y_train = y_train.astype(int)

            estimator_cls = ESTIMATOR_REGISTRY[spec.estimator]
            estimator = estimator_cls(**spec.estimator_params)
            estimator.fit(X_train, y_train)

            # Predict on ALL feature-usable rows (train + holdout + anything
            # else), using only the train-fitted model. No retraining here.
            predictions_flat = np.full(self.n_days * self.n_instruments, np.nan, dtype=np.float64)
            if feature_valid_flat.any():
                predictions_flat[feature_valid_flat] = estimator.predict(X_flat[feature_valid_flat])
            predictions = predictions_flat.reshape(self.n_days, self.n_instruments)

            results[spec.name] = TrainedTarget(
                spec=spec,
                estimator=estimator,
                predictions=predictions,
                target_panel=target_panel,
                feature_names=self.feature_names,
                n_train_rows=n_train_rows,
            )
        return results
