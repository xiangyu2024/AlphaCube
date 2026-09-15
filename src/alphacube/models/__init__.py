"""Model-aggregate layer: trains ML models on top of mined-factor exposure
panels to predict several kinds of forward-looking targets (return,
volatility, downside volatility, cross-sectional rank, 3-class direction,
spike/crash event probability) at multiple horizons.

Sits between the alpha-mining layer (`engine/`, `mining/` — produces
expressions and their evaluated daily `(D, S)` factor panels) and portfolio
construction: `targets.py` turns a returns panel into label panels,
`estimators.py` wraps sklearn (optionally lightgbm) regressors/classifiers
behind a common interface, `trainer.py` flattens feature/label panels into
training rows with a leakage-safe embargoed train/holdout split and fits one
model per target, and `evaluation.py` scores the resulting prediction panels
(IC, rolling per-stock IC, long-short Sharpe, magnitude/calibration).

Nothing in this package imports `engine/` or `mining/` — it consumes
already-evaluated `dict[str, np.ndarray]` feature panels, decoupled from the
mining machinery that (in a real pipeline) produced them.
"""

from __future__ import annotations

from .estimators import ESTIMATOR_REGISTRY, ModelEstimator
from .evaluation import EvaluationResult, evaluate_target, leaderboard
from .trainer import MultiTargetTrainer, TargetSpec, TrainedTarget

__all__ = [
    "ESTIMATOR_REGISTRY",
    "ModelEstimator",
    "EvaluationResult",
    "evaluate_target",
    "leaderboard",
    "MultiTargetTrainer",
    "TargetSpec",
    "TrainedTarget",
]
