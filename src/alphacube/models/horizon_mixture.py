"""`HorizonMixtureEstimator`: a held-out-optimized stacking ensemble across
*label horizons*, inspired by (and an explicitly simplified, sklearn-shaped
approximation of) "The Label Horizon Paradox: Rethinking Supervision Targets
in Financial Forecasting" (arXiv:2602.03395).

**The paper's argument.** If the deployment/evaluation horizon is `Delta`
(e.g. 20 days), training directly on the horizon-`Delta` label can be
*suboptimal* supervision, because a forward return's "signal" component
(information getting priced in) saturates as horizon grows, while its
"noise" component (idiosyncratic variance) keeps growing roughly like a
random walk. So a shorter proxy horizon `delta* < Delta` can have a better
signal-to-noise ratio and, counterintuitively, be a better *training* target
even though the thing you actually want to predict is the horizon-`Delta`
return.

**What the paper actually proposes**, and what this module does **not**
implement: a fully differentiable bi-level meta-learning scheme — a
learnable per-horizon weight vector optimized via nested gradient-through-
gradient updates (an inner loop fits base models, an outer loop
differentiates through the inner solution to update the horizon weights).
That requires a custom autodiff training loop; this project's stage-3
architecture (`estimators.py`'s `ModelEstimator` ABC, `trainer.py`'s
`MultiTargetTrainer`) is built entirely around plain sklearn-style
`.fit(X, y)` / `.predict(X)` estimators with no such loop, so a literal port
isn't compatible with the rest of the codebase (cf. the same kind of
"this project's simplification" note in `combine/factorgcl/model.py`,
where pragmatic gap-filling is done and documented rather than silently
glossed over).

**This project's simplification.** Instead of differentiable bi-level
optimization, `HorizonMixtureEstimator` trains one ordinary base estimator
per *candidate* horizon (each on that horizon's own label, computed via
`targets.py`/the same machinery `trainer.py` already uses), then learns
fixed, non-negative, sum-to-one mixture weights over their predictions by
fitting a small constrained regression (`scipy.optimize.nnls`) on a
held-out fold, *against the true target-horizon label* — a held-out-
optimized stacking ensemble across horizons. This captures the paper's core
empirical claim (some horizons are more informative *proxies* than others,
and that can be learned from out-of-sample performance) without any nested
gradients: everything below is a two-stage `.fit(X, y)` procedure.

**A real, documented limitation this simplification introduces**: because
the mixture weights are fit by regressing candidate-horizon predictions
against the *true* target-horizon label on a held-out fold, this is a
supervised meta-learner that *requires* `y_by_horizon[target_horizon]` to be
present and at least partially finite. The paper's differentiable version
can, in principle, still learn useful horizon weighting from proxy signal
alone (e.g. via a meta-objective that doesn't require true-horizon labels at
every step); this module cannot — see `HorizonMixtureEstimator.fit`'s
docstring.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import nnls
from sklearn.model_selection import train_test_split

from . import targets as T
from .estimators import ESTIMATOR_REGISTRY, ModelEstimator

logger = logging.getLogger(__name__)

# Minimum number of rows with a finite label across *every* candidate
# horizon and the target horizon, required to even attempt a fit/val split
# for the mixture-weight-fitting step. Below this, `train_test_split` can
# degenerate to an empty val fold (or raise outright), and a mixture fit on
# a handful of points is not meaningfully "held-out optimized" anyway.
_MIN_ALL_FINITE_ROWS = 5


def build_candidate_horizon_labels(
    returns_panel: np.ndarray, candidate_horizons: List[int]
) -> Dict[int, np.ndarray]:
    """Thin convenience wrapper: `{h: targets.forward_return(returns_panel, h,
    from_returns=True) for h in candidate_horizons}`.

    Saves callers from hand-writing this loop -- it's the natural glue
    between `targets.py` and `HorizonMixtureEstimator.fit`'s `y_by_horizon`
    argument, mirroring how `trainer.py`'s `MultiTargetTrainer` computes one
    label panel per `TargetSpec` via `compute_target`.
    """
    return {h: T.forward_return(returns_panel, h, from_returns=True) for h in candidate_horizons}


class HorizonMixtureEstimator(ModelEstimator):
    """Held-out-optimized stacking ensemble over candidate label horizons.

    Trains one base estimator per horizon in `candidate_horizons` (each on
    its own label), then learns non-negative, sum-to-one mixture weights
    over their predictions on a held-out fold, regressed against the *true*
    `target_horizon` label. `target_horizon` does not need to be a member of
    `candidate_horizons` -- the whole point is that candidate horizons are
    *proxies* -- though including it as one of the candidates is a sensible
    default caller pattern; this class does not force it.

    `base_estimator` is a key into `estimators.ESTIMATOR_REGISTRY`, used to
    construct one fresh instance per candidate horizon (all candidates
    share the same estimator type/hyperparameters -- only the label
    differs).

    **Deviation from `ModelEstimator.fit(X, y)`.** This class fits several
    differently-labeled base models rather than a single `(X, y)` pair, so
    it cannot satisfy the plain ABC signature the way a normal estimator
    does. Its `.fit` instead takes `X` (shared features) and
    `y_by_horizon: Dict[int, np.ndarray]` (one label array per candidate
    horizon, same rows/features as `X`, each computed independently at that
    horizon -- e.g. via `build_candidate_horizon_labels` or by hand-calling
    `targets.py` per horizon, exactly as `trainer.py`'s `MultiTargetTrainer`
    does per `TargetSpec`). See `.fit`'s docstring for the row-alignment and
    true-label-requirement details.
    """

    def __init__(
        self,
        candidate_horizons: List[int],
        target_horizon: int,
        base_estimator: str = "hgb",
        base_estimator_params: Optional[dict] = None,
        mixture_val_frac: float = 0.2,
        random_state: int = 0,
    ) -> None:
        if not candidate_horizons:
            raise ValueError("candidate_horizons must be non-empty")
        if base_estimator not in ESTIMATOR_REGISTRY:
            raise ValueError(
                f"Unknown base_estimator '{base_estimator}'. Registered: {sorted(ESTIMATOR_REGISTRY)}"
            )
        if not (0.0 < mixture_val_frac < 1.0):
            raise ValueError("mixture_val_frac must be in (0, 1)")

        self.candidate_horizons: List[int] = list(candidate_horizons)
        self.target_horizon = target_horizon
        self.base_estimator = base_estimator
        self.base_estimator_params: dict = dict(base_estimator_params or {})
        self.mixture_val_frac = mixture_val_frac
        self.random_state = random_state

        self.models_: Dict[int, ModelEstimator] = {}
        self.mixture_weights_: Dict[int, float] = {}
        self._fitted = False

    def fit(self, X: np.ndarray, y_by_horizon: Dict[int, np.ndarray]) -> "HorizonMixtureEstimator":
        """Fit one base model per candidate horizon, then learn mixture
        weights on a held-out fold against the true `target_horizon` label.

        `y_by_horizon` must contain an entry for every `candidate_horizons`
        member (each that horizon's own label array, same length as `X`)
        **and** for `target_horizon` -- if `target_horizon` is missing, a
        `ValueError` is raised: this is a supervised meta-learner that needs
        the real target to learn to weight the proxies against (unlike the
        paper's differentiable version, which can in principle work from
        proxy signal alone -- a real, documented limitation of this
        simplified version; see the module docstring).

        **Row-selection / alignment.** Different horizons' labels have
        different NaN patterns (longer horizons have more trailing NaN rows
        near the panel's tail, per `targets.py`'s no-lookahead contract).
        Two different row sets are used, deliberately:

        1. *Mixture-weight validation set*: rows where **every** candidate
           horizon's label **and** the target horizon's label are finite
           (`all_finite_mask`). A fixed-size fraction (`mixture_val_frac`)
           of these rows, chosen via `sklearn.model_selection.train_test_split`
           with `random_state`, is held out as the *val fold*. Restricting
           the val fold to `all_finite_mask` rows guarantees every candidate
           base model can be evaluated (features are shared, so predicting
           doesn't need a finite label, but scoring the mixture against the
           true label requires a finite `target_horizon` label on that row,
           and comparing candidates head-to-head on the *same* row set
           requires all of them to have had a finite label available to
           train on somewhere in the panel) and that the val-fold rows are
           never used for anything else.
        2. *Per-horizon base-model training set*: every row **not** in the
           val fold (i.e. all rows outside `all_finite_mask` are always
           eligible, plus the `all_finite_mask` rows not drawn into the val
           fold), further restricted **independently per horizon** to rows
           where *that horizon's own* label is finite. This is the "each
           base model can still be fit on its own horizon's full
           finite-label set" behavior: a horizon-5 base model gets to use
           tail rows that a horizon-40 label can't (because horizon-40's
           NaN tail is longer), rather than being needlessly restricted to
           the intersection-of-all-horizons set.

        The same row *never* appears in both a horizon's training set and
        the val fold (the val fold is carved out of `all_finite_mask` before
        any per-horizon training set is formed), so there is no leakage
        between base-model fitting and mixture-weight fitting.

        Raises `ValueError` if `target_horizon` is missing from
        `y_by_horizon`, if any `candidate_horizons` entry is missing, if any
        label array's length doesn't match `X`, or if there are too few
        rows with a finite label across every candidate horizon and the
        target horizon to form a meaningful val fold.
        """
        X = np.asarray(X, dtype=np.float64)
        n_rows = X.shape[0]

        if self.target_horizon not in y_by_horizon:
            raise ValueError(
                f"y_by_horizon is missing target_horizon={self.target_horizon}: fitting mixture "
                "weights requires the true target-horizon label as ground truth (this simplified, "
                "sklearn-compatible version is a supervised meta-learner over proxy horizons, unlike "
                "the paper's differentiable bi-level scheme, which can in principle use proxy signal "
                "alone -- see the module docstring)."
            )

        finite_masks: Dict[int, np.ndarray] = {}
        for h in self.candidate_horizons:
            if h not in y_by_horizon:
                raise ValueError(
                    f"y_by_horizon is missing candidate_horizon={h}: every candidate horizon needs "
                    "its own label array to train that horizon's base model."
                )
            arr = np.asarray(y_by_horizon[h], dtype=np.float64)
            if arr.shape[0] != n_rows:
                raise ValueError(
                    f"y_by_horizon[{h}] has {arr.shape[0]} rows, expected {n_rows} to match X"
                )
            finite_masks[h] = np.isfinite(arr)

        target_y = np.asarray(y_by_horizon[self.target_horizon], dtype=np.float64)
        if target_y.shape[0] != n_rows:
            raise ValueError(
                f"y_by_horizon[{self.target_horizon}] has {target_y.shape[0]} rows, expected "
                f"{n_rows} to match X"
            )
        target_finite = np.isfinite(target_y)

        # Rows usable for the mixture-weight-fitting step: finite label for
        # EVERY candidate horizon AND the target horizon. Note `target_horizon`
        # may or may not also be a candidate; either way its own finiteness
        # is required here.
        all_finite_mask = target_finite.copy()
        for h in self.candidate_horizons:
            all_finite_mask &= finite_masks[h]

        all_finite_idx = np.flatnonzero(all_finite_mask)
        if all_finite_idx.size < _MIN_ALL_FINITE_ROWS:
            raise ValueError(
                f"only {all_finite_idx.size} row(s) have a finite label across every candidate "
                f"horizon {self.candidate_horizons} and target_horizon={self.target_horizon}; need "
                f"at least {_MIN_ALL_FINITE_ROWS} to form a meaningful held-out mixture-weight fit. "
                "This usually means the candidate horizons are too spread out relative to the panel "
                "length (longer horizons NaN out more trailing rows -- see targets.py)."
            )

        # Carve the val fold out of the all-finite rows only. Every row NOT
        # selected for validation (whether or not it was even eligible for
        # validation) is available for per-horizon base-model training.
        _, val_idx = train_test_split(
            all_finite_idx, test_size=self.mixture_val_frac, random_state=self.random_state
        )
        val_mask = np.zeros(n_rows, dtype=bool)
        val_mask[val_idx] = True
        fit_fold_mask = ~val_mask

        models: Dict[int, ModelEstimator] = {}
        for h in self.candidate_horizons:
            train_mask = fit_fold_mask & finite_masks[h]
            if not train_mask.any():
                raise ValueError(
                    f"candidate_horizon={h}: zero fit-fold rows with a finite label (after holding "
                    "out the mixture validation fold)"
                )
            estimator_cls = ESTIMATOR_REGISTRY[self.base_estimator]
            model = estimator_cls(**self.base_estimator_params)
            model.fit(X[train_mask], np.asarray(y_by_horizon[h], dtype=np.float64)[train_mask])
            models[h] = model

        X_val = X[val_mask]
        y_val = target_y[val_mask]
        pred_matrix = np.column_stack([models[h].predict(X_val) for h in self.candidate_horizons])

        weights, _residual = nnls(pred_matrix, y_val)
        weight_sum = weights.sum()
        if weight_sum <= 0.0 or not np.isfinite(weight_sum):
            logger.warning(
                "HorizonMixtureEstimator: nnls returned all-zero (or non-finite) weights for "
                "candidate horizons %s against target_horizon=%s -- falling back to equal weighting "
                "rather than dividing by zero.",
                self.candidate_horizons,
                self.target_horizon,
            )
            weights = np.full(len(self.candidate_horizons), 1.0 / len(self.candidate_horizons))
        else:
            weights = weights / weight_sum

        self.models_ = models
        self.mixture_weights_ = {h: float(w) for h, w in zip(self.candidate_horizons, weights)}
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted sum of every candidate horizon's base-model predictions,
        using the learned `mixture_weights_`."""
        if not self._fitted:
            raise RuntimeError("HorizonMixtureEstimator.predict called before fit")
        X = np.asarray(X, dtype=np.float64)
        out = np.zeros(X.shape[0], dtype=np.float64)
        for h in self.candidate_horizons:
            out += self.mixture_weights_[h] * self.models_[h].predict(X)
        return out
