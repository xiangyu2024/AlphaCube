"""Typed random trees over the unified registry.

Generation is requirement-driven (top-down), not rejection-sampled: every
slot carries a requirement on the minute-axis type of what fills it --

- ``"daily"``: one value per stock-day (root slots, ND-face operands);
- ``"full"``: genuine per-minute variation (``in_*``/``nt_*``/``day_*`` operands);
- ``"collapsed"``: literal width 1 (``in_expand`` operands).

Operator choice per slot is drawn from the operators whose output can satisfy
the requirement given some choice of operand requirements, so the tree is
valid by construction; the ``Expression`` constructors re-check every node.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

from ..axes import OpSpec
from ..catalog import Registry
from ..expression import (Constant, Expression, Feature, ShapeError, is_valid_root, make_node,
                          replace)

REQS = ("daily", "full", "collapsed")


def shape_req(expr: Expression) -> str:
    """The requirement a subtree currently satisfies (for like-for-like replacement)."""
    sh = expr.shape
    if sh.collapsed:
        return "collapsed"
    if sh.degenerate:
        return "daily"
    return "full"


def collect(expr: Expression) -> List[Tuple[Expression, Optional[Expression]]]:
    """(node, parent) pairs in pre-order."""
    out: List[Tuple[Expression, Optional[Expression]]] = []

    def walk(n: Expression, parent: Optional[Expression]) -> None:
        out.append((n, parent))
        for c in n.operands:
            walk(c, n)

    walk(expr, None)
    return out


class TypedTreeGenerator:
    def __init__(self, registry: Registry, fields: Optional[Sequence[str]] = None,
                 constants: Optional[Sequence[float]] = None, n_minutes: int = 241,
                 min_depth: int = 2, max_depth: int = 4, feature_leaf_prob: float = 0.8,
                 grow_leaf_prob: float = 0.3, max_retries: int = 16,
                 include_lookahead: bool = False, allow_boolean_ops: bool = False,
                 rng: Optional[random.Random] = None, seed: int = 0) -> None:
        self.registry = registry
        self.fields = list(fields) if fields is not None else list(registry.fields)
        self.constants = list(constants) if constants is not None else list(registry.constants)
        self.n_minutes = int(n_minutes)
        self.min_depth = max(2, int(min_depth))
        self.max_depth = max(self.min_depth, int(max_depth))
        self.feature_leaf_prob = float(feature_leaf_prob)
        self.grow_leaf_prob = float(grow_leaf_prob)
        self.max_retries = int(max_retries)
        self._rng = rng if rng is not None else random.Random(seed)
        specs = (registry.list_all() if include_lookahead
                 else registry.searchable(include_boolean=allow_boolean_ops))
        self.allow_boolean_ops = bool(allow_boolean_ops)
        self.specs: List[OpSpec] = specs
        # operators grouped by what they can produce
        self._producers: Dict[str, List[OpSpec]] = {
            "daily": [s for s in specs if s.reduces_m or s.face == "nd" or s.family == "elementwise"
                      or (s.face == "dt" and s.family == "cross")],
            "full": [s for s in specs if s.needs_full_m and not s.reduces_m or s.family == "elementwise"
                     or (s.face == "dt" and s.family == "cross")],
            "collapsed": [s for s in specs if s.reduces_m or s.face == "nd"],
        }
        self._root_ops = [s for s in self._producers["daily"] if not s.non_root]

    # -- public --------------------------------------------------------------------
    def generate(self, depth: Optional[int] = None) -> Expression:
        d = depth if depth is not None else self._rng.randint(self.min_depth, self.max_depth)
        for _ in range(self.max_retries * 4):
            try:
                tree = self._gen("daily", d, full=self._rng.random() < 0.5, root=True)
            except ShapeError:
                continue
            if is_valid_root(tree):
                return tree
        return self._collapse_leaf(self._pick(self._root_ops_of("reduce")))

    def generate_population(self, n: int) -> List[Expression]:
        depths = list(range(self.min_depth, self.max_depth + 1))
        return [self.generate(depths[(i // 2) % len(depths)]) for i in range(n)]

    def subtree(self, req: str, depth: Optional[int] = None) -> Expression:
        d = depth if depth is not None else self._rng.randint(1, self.max_depth - 1)
        for _ in range(self.max_retries):
            try:
                return self._gen(req, d, full=False)
            except ShapeError:
                continue
        return self._gen(req, 2, full=True)

    def mutate_subtree(self, expr: Expression) -> Optional[Expression]:
        nodes = collect(expr)
        for _ in range(self.max_retries):
            node, _parent = self._rng.choice(nodes)
            new = self.subtree(shape_req(node), max(1, node.depth()))
            try:
                child = replace(expr, node, new)
            except ShapeError:
                continue
            if is_valid_root(child):
                return child
        return None

    def mutate_window(self, expr: Expression) -> Optional[Expression]:
        cands = [n for n, _ in collect(expr) if n.window is not None and not n.spec.restores_m]
        if not cands:
            return None
        node = self._rng.choice(cands)
        pool = [w for w in node.spec.windows if w != node.window]
        if not pool:
            return None
        new = make_node(node.spec, node.operands, self._rng.choice(pool))
        try:
            child = replace(expr, node, new)
        except ShapeError:
            return None
        return child if is_valid_root(child) else None

    def mutate_constant(self, expr: Expression) -> Optional[Expression]:
        cands = [n for n, _ in collect(expr) if isinstance(n, Constant)]
        if not cands:
            return None
        node = self._rng.choice(cands)
        pool = [c for c in self.constants if c != node.value] or [node.value * 2.0]
        try:
            child = replace(expr, node, Constant(self._rng.choice(pool)))
        except ShapeError:
            return None
        return child if is_valid_root(child) else None

    def crossover(self, a: Expression, b: Expression) -> Optional[Expression]:
        nodes_a, nodes_b = collect(a), collect(b)
        for _ in range(self.max_retries):
            na, pa = self._rng.choice(nodes_a)
            nb, _ = self._rng.choice(nodes_b)
            if pa is None:
                continue
            if shape_req(na) != shape_req(nb):
                continue
            try:
                child = replace(a, na, nb)
            except ShapeError:
                continue
            if is_valid_root(child):
                return child
        return None

    # -- internals -----------------------------------------------------------------
    def _pick(self, specs: List[OpSpec]) -> OpSpec:
        """Face-balanced: draw a face uniformly among those present, then an operator
        within it -- otherwise the 65 ``in_*`` operators swamp the 16 ``nt_cs_*``."""
        faces: Dict[str, List[OpSpec]] = {}
        for s in specs:
            faces.setdefault(s.face, []).append(s)
        return self._rng.choice(faces[self._rng.choice(sorted(faces))])

    def _root_ops_of(self, kind: str) -> List[OpSpec]:
        return [s for s in self.specs if s.reduces_m and not s.non_root]

    def _collapse_leaf(self, spec: OpSpec) -> Expression:
        """Minimal daily tree: a day_* reducer over raw field leaves."""
        n = 1 if spec.category in ("unary", "rolling") else 2
        leaves = tuple(Feature(self._rng.choice(self.fields)) for _ in range(n))
        return make_node(spec, leaves, self._window_for(spec))

    def _window_for(self, spec: OpSpec) -> Optional[int]:
        if not spec.requires_window:
            return None
        if spec.restores_m:
            return self.n_minutes
        return int(self._rng.choice(spec.windows))

    def _leaf(self, req: str) -> Expression:
        if req == "full":
            return Feature(self._rng.choice(self.fields))
        if req == "daily":
            return Constant(self._rng.choice(self.constants))
        raise ShapeError("no leaf satisfies 'collapsed'")

    def _gen(self, req: str, depth: int, full: bool, root: bool = False) -> Expression:
        if depth <= 1:
            if req == "full":
                return self._leaf("full")
            # daily/collapsed cannot be a leaf: minimal collapse of a field
            spec = self._pick([s for s in self.specs if s.reduces_m and not (root and s.non_root)])
            return self._collapse_leaf(spec)
        if req == "full" and not full and self._rng.random() < self.grow_leaf_prob:
            return self._leaf("full")
        pool = self._root_ops if root else self._producers[req]
        for _ in range(self.max_retries):
            spec = self._pick(pool)
            try:
                return self._build(spec, req, depth, full)
            except ShapeError:
                continue
        raise ShapeError(f"could not fill a {req!r} slot at depth {depth}")

    def _operand_reqs(self, spec: OpSpec, req: str) -> Tuple[str, ...]:
        n = 1 if spec.category in ("unary", "rolling") else 2
        if spec.reduces_m:
            return ("full",) * n
        if spec.restores_m:
            return ("collapsed",)
        if spec.face == "nd":
            return ("daily",) * n
        if spec.needs_full_m:                     # in_* / nt_*
            return ("full",) * n
        if spec.face == "dt" and spec.family == "cross":
            # pass-through: operands carry the required type
            return (req,) * n if req != "collapsed" else ("collapsed",) * n
        # elementwise
        if req == "full":
            if n == 1:
                return ("full",)
            other = "full" if self._rng.random() < 0.6 else "daily"
            return ("full", other) if self._rng.random() < 0.5 else (other, "full")
        if req == "daily":
            return ("daily",) * n
        return ("collapsed",) * n

    def _build(self, spec: OpSpec, req: str, depth: int, full: bool) -> Expression:
        reqs = self._operand_reqs(spec, req)
        operands = []
        const_slot = -1
        if spec.family == "elementwise" and len(reqs) == 2 and self._rng.random() < (1 - self.feature_leaf_prob):
            const_slot = self._rng.choice([i for i, r in enumerate(reqs) if r == "daily"] or [-1])
        for i, r in enumerate(reqs):
            if i == const_slot:
                operands.append(Constant(self._rng.choice(self.constants)))
            else:
                operands.append(self._gen(r, depth - 1, full))
        return make_node(spec, tuple(operands), self._window_for(spec))


__all__ = ["TypedTreeGenerator", "shape_req", "collect", "REQS"]
