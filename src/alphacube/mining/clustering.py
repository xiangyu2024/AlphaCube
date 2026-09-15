"""Structural clustering of a GP population + the cluster-culling rule
(ported from the previous project's GP).

``skeleton_key`` is the structural key made leaf-blind: every Feature is
``F``, every Constant is its class (``C:pct`` for |v| in (0, 1], ``C:scl``
otherwise); operator names and windows are kept, so ``add(Close, Open)`` and
``add(High, Low)`` share a skeleton while ``ts_mean(x,5)`` / ``ts_mean(x,10)``
do not. Singleton clusters merge into the nearest cluster by operator-family
histogram cosine (> ``merge_threshold``). ``cull_clusters`` keeps, in a
cluster whose mean fitness is more than ``rel_threshold`` below the best
cluster's mean, only members that individually beat that best mean.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np

from ..expression import Constant, Expression, Feature


def constant_class(value: float) -> str:
    v = abs(float(value))
    return "percent" if 0.0 < v <= 1.0 else "scalar"


def skeleton_key(expr: Expression) -> str:
    if isinstance(expr, Feature):
        return "F"
    if isinstance(expr, Constant):
        return "C:pct" if constant_class(expr.value) == "percent" else "C:scl"
    inner = ",".join(skeleton_key(o) for o in expr.operands)
    w = "" if expr.window is None else f"@{expr.window}"
    return f"O:{expr.spec.name}({inner}){w}"


def _family(expr: Expression) -> str:
    return f"{expr.spec.face}_{expr.spec.family}"


def _hist(expr: Expression) -> Dict[str, int]:
    h: Dict[str, int] = {}
    for n in expr.nodes():
        if n.spec is not None:
            f = _family(n)
            h[f] = h.get(f, 0) + 1
    return h


def _cosine(a: Dict[str, int], b: Dict[str, int]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def cluster_population(exprs: Sequence[Expression], merge_threshold: float = 0.8) -> List[List[int]]:
    groups: Dict[str, List[int]] = {}
    for i, e in enumerate(exprs):
        groups.setdefault(skeleton_key(e), []).append(i)
    clusters = [list(v) for v in groups.values()]
    hists = []
    for members in clusters:
        h: Dict[str, int] = {}
        for i in members:
            for k, v in _hist(exprs[i]).items():
                h[k] = h.get(k, 0) + v
        hists.append(h)
    absorbed = [False] * len(clusters)
    for si, members in enumerate(clusters):
        if len(members) != 1 or absorbed[si]:
            continue
        best_j, best = -1, float(merge_threshold)
        for j in range(len(clusters)):
            if j == si or absorbed[j]:
                continue
            sim = _cosine(hists[si], hists[j])
            if sim > best:
                best, best_j = sim, j
        if best_j >= 0:
            clusters[best_j] = clusters[best_j] + clusters[si]
            for k, v in hists[si].items():
                hists[best_j][k] = hists[best_j].get(k, 0) + v
            absorbed[si] = True
    return [sorted(c) for k, c in enumerate(clusters) if not absorbed[k]]


def cull_clusters(clusters: Sequence[Sequence[int]], fitness: np.ndarray, rel_threshold: float = 0.5) -> List[List[int]]:
    fitness = np.asarray(fitness, dtype=float)
    means = []
    for members in clusters:
        v = fitness[list(members)]
        f = v[np.isfinite(v)]
        means.append(float(f.mean()) if f.size else float("-inf"))
    best = max(means) if means else float("-inf")
    if not np.isfinite(best):
        return [list(c) for c in clusters]
    cutoff = best - float(rel_threshold) * abs(best)
    out = []
    for members, mean in zip(clusters, means):
        if mean >= cutoff:
            out.append(list(members))
        else:
            out.append([i for i in members if np.isfinite(fitness[i]) and fitness[i] > best])
    return out


__all__ = ["skeleton_key", "cluster_population", "cull_clusters", "constant_class"]
