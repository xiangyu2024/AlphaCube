"""Expression list -> DAG with common-subexpression elimination on ``key()``."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from ..axes import OpSpec
from ..expression import Constant, Expression, Feature, Shape


class Node:
    __slots__ = ("expr", "key", "children", "parents", "root_indices", "mode", "round_id",
                 "boundary", "bound_layout", "panel_out", "mask_out", "failed", "error")

    def __init__(self, expr: Expression) -> None:
        self.expr = expr
        self.key = expr.key()
        self.children: List["Node"] = []
        self.parents: List["Node"] = []
        self.root_indices: List[int] = []
        self.mode: Optional[str] = None
        self.round_id: int = -1
        self.boundary = False       # full-width output needed by another cube mode -> disk
        self.bound_layout = "S"     # its file's axis order: "S" = (S, D, M), "D" = (D, S, M)
        self.panel_out = False      # degenerate output needed as a (S, D) panel -> memory
        self.mask_out = True        # re-apply the tradability mask at this node's output
        self.failed = False
        self.error: Optional[str] = None

    @property
    def spec(self) -> Optional[OpSpec]:
        return self.expr.spec

    @property
    def shape(self) -> Shape:
        return self.expr.shape

    @property
    def is_leaf(self) -> bool:
        return isinstance(self.expr, (Feature, Constant))

    @property
    def is_root(self) -> bool:
        return bool(self.root_indices)

    def __repr__(self) -> str:
        return f"Node({self.key}, mode={self.mode}, round={self.round_id})"


class Dag:
    def __init__(self, roots: Sequence[Expression]) -> None:
        self.by_key: Dict[str, Node] = {}
        self.topo: List[Node] = []          # children before parents
        self.roots: List[Node] = []
        for i, expr in enumerate(roots):
            node = self._add(expr)
            node.root_indices.append(i)
            self.roots.append(node)

    def _add(self, expr: Expression) -> Node:
        key = expr.key()
        node = self.by_key.get(key)
        if node is not None:
            return node
        node = Node(expr)
        for child_expr in expr.operands:
            child = self._add(child_expr)
            node.children.append(child)
            child.parents.append(node)
        self.by_key[key] = node
        self.topo.append(node)
        return node

    @property
    def computed(self) -> List[Node]:
        return [n for n in self.topo if not n.is_leaf]


__all__ = ["Node", "Dag"]
