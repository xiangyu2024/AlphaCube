"""Feature Boosting O(n) pre-squeeze (the user's design note, "Feature Boosting
预压榨"), on the cube engine.

Each iteration pairs ONE field group with ONE operator group, builds a
bounded local set of candidate expressions from the carried-forward operands,
evaluates every candidate that is already a daily factor in one engine call,
ranks them by GBDT permutation importance against the label, and carries the
survivors forward as operands for the next iteration. The cost is
``n_iterations`` local products, never the full N_op x N_feat product.

Adapted from the previous project (same grouping idea, same GBDT filter):
operator groups are the registry's (face, family) pairs; candidates that are
not yet one-value-per-stock-day (a live minute axis) cannot be scored, so a
capped random sample of them is carried forward unscored, which is how a
``day_std(in_mean_rolling(Ret, 5))`` composite gets built across iterations.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..catalog import Registry
from ..engine.engine import CubeEngine
from ..expression import Constant, Expression, Feature, ShapeError, is_valid_root, make_node


def default_field_groups(fields: Sequence[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {"price": [], "volume": [], "return": []}
    for f in fields:
        low = f.lower()
        if low in ("volume", "vol", "amt", "amount"):
            groups["volume"].append(f)
        elif low in ("ret", "return", "returns"):
            groups["return"].append(f)
        else:
            groups["price"].append(f)
    return {k: v for k, v in groups.items() if v}


def default_operator_groups(registry: Registry) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for s in registry.searchable():
        groups.setdefault(f"{s.face}_{s.family}", []).append(s.name)
    return groups


@dataclass(frozen=True)
class PresqueezeConfig:
    n_iterations: int = 4
    candidates_per_iteration_cap: int = 200
    top_k_per_iteration: int = 8
    carry_unscored: int = 12            # full-width candidates carried forward as operands
    seed: int = 0
    pairing: str = "random"             # "random" | "heuristic"
    heuristic_affinities: Dict[str, Sequence[str]] = field(default_factory=dict)
    field_groups: Optional[Dict[str, List[str]]] = None
    operator_groups: Optional[Dict[str, List[str]]] = None
    permutation_repeats: int = 5
    min_finite_rows: int = 30
    gbdt_max_iter: int = 50
    max_fit_rows: int = 200_000


@dataclass
class PresqueezeResult:
    survivors: List[Expression]
    importances: List[float]
    iterations_run: int
    candidates_generated_total: int
    candidates_evaluated_total: int
    field_groups: Dict[str, List[str]]
    operator_groups: Dict[str, List[str]]
    per_iteration_log: List[dict]


def _candidates(registry: Registry, op_names: Sequence[str], operands: Sequence[Expression],
                windows_default: Sequence[int], n_minutes: int, cap: int, rng: random.Random) -> List[Expression]:
    out: List[Expression] = []
    seen = set()
    for name in op_names:
        spec = registry.get(name)
        n = 1 if spec.category in ("unary", "rolling") else 2
        window = None
        if spec.requires_window:
            window = n_minutes if spec.restores_m else int(rng.choice(list(spec.windows) or list(windows_default)))
        pairs = [(o,) for o in operands] if n == 1 else [(a, b) for a in operands for b in operands if a is not b]
        for ops in pairs:
            try:
                e = make_node(spec, ops, window)
            except ShapeError:
                continue
            if e.key() in seen:
                continue
            seen.add(e.key())
            out.append(e)
            if len(out) >= cap:
                return out
    return out


def _rank(candidates: List[Expression], panels: List[np.ndarray], target: np.ndarray,
          config: PresqueezeConfig, seed: int) -> Tuple[List[Expression], List[float]]:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance

    if not candidates:
        return [], []
    X = np.column_stack([np.where(np.isfinite(p), p, np.nan).reshape(-1) for p in panels])
    y = target.reshape(-1)
    keep = np.isfinite(y)
    X, y = X[keep], y[keep]
    if X.shape[0] < config.min_finite_rows:
        return [], []
    if X.shape[0] > config.max_fit_rows:
        idx = np.random.default_rng(seed).choice(X.shape[0], config.max_fit_rows, replace=False)
        X, y = X[idx], y[idx]
    model = HistGradientBoostingRegressor(max_iter=config.gbdt_max_iter, random_state=seed)
    model.fit(X, y)
    imp = permutation_importance(model, X, y, n_repeats=config.permutation_repeats, random_state=seed, n_jobs=1).importances_mean
    order = np.argsort(imp)[::-1]
    top = [i for i in order[: config.top_k_per_iteration] if imp[i] > 0]
    return [candidates[i] for i in top], [float(imp[i]) for i in top]


def feature_boosting_presqueeze(registry: Registry, engine: CubeEngine, target: np.ndarray,
                                config: Optional[PresqueezeConfig] = None) -> PresqueezeResult:
    config = config or PresqueezeConfig()
    fields = list(registry.fields)
    field_groups = config.field_groups or default_field_groups(fields)
    operator_groups = config.operator_groups or default_operator_groups(registry)
    if not field_groups or not operator_groups:
        raise ValueError("no field/operator groups to pre-squeeze")
    rng = random.Random(config.seed)
    windows_default = sorted({w for s in registry.searchable() for w in s.windows}) or [10]
    carried: List[Expression] = [Feature(f) for f in fields]
    survivors: List[Expression] = []
    importances: List[float] = []
    log: List[dict] = []
    generated = evaluated = 0
    target = np.asarray(target, dtype=np.float64)
    for it in range(config.n_iterations):
        fkey = rng.choice(list(field_groups))
        if config.pairing == "heuristic" and fkey in config.heuristic_affinities:
            allowed = [k for k in config.heuristic_affinities[fkey] if k in operator_groups]
            okey = rng.choice(allowed) if allowed else rng.choice(list(operator_groups))
        else:
            okey = rng.choice(list(operator_groups))
        group_fields = set(field_groups[fkey])
        operands = [e for e in carried if not isinstance(e, Feature) or e.field in group_fields]
        if not operands:
            continue
        cands = _candidates(registry, operator_groups[okey], operands, windows_default, engine.view.n_minutes,
                            config.candidates_per_iteration_cap, rng)
        generated += len(cands)
        daily = [c for c in cands if is_valid_root(c)]
        wide = [c for c in cands if not c.shape.degenerate]
        panels: List[np.ndarray] = []
        scored: List[Expression] = []
        if daily:
            res = engine.evaluate(daily, tolerant=True)
            for c, p in zip(daily, res.panels):
                if p is not None and np.isfinite(p).any():
                    scored.append(c)
                    panels.append(p.astype(np.float64))
        evaluated += len(scored)
        round_surv, round_imp = _rank(scored, panels, target, config, config.seed + it)
        log.append({"iteration": it, "field_group": fkey, "operator_group": okey, "candidates": len(cands),
                    "daily": len(daily), "scored": len(scored), "survivors": len(round_surv)})
        survivors.extend(round_surv)
        importances.extend(round_imp)
        carried.extend(round_surv)
        carried.extend(rng.sample(wide, min(len(wide), config.carry_unscored)))
    return PresqueezeResult(survivors, importances, config.n_iterations, generated, evaluated,
                            field_groups, operator_groups, log)


def seed_pool_from_presqueeze(pool, result: PresqueezeResult, engine: CubeEngine) -> List[float]:
    from ..legacy.seed_library import seed_pool
    return seed_pool(pool, result.survivors, engine)


__all__ = ["PresqueezeConfig", "PresqueezeResult", "feature_boosting_presqueeze", "seed_pool_from_presqueeze",
           "default_field_groups", "default_operator_groups"]
