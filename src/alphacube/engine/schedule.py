"""Chunk-phase scheduling.

Every computed node gets a **mode**:

- ``P``: whole ``(D, S)`` panel in memory. ND-face operators, and elementwise
  nodes whose operands are all panels / minute-degenerate.
- ``S``: cube chunked over stocks ``(C, D, M)``. Required by D-axis operators
  (``cross_*``); allowed for M-axis and elementwise ones.
- ``D``: cube chunked over days ``(S, C, M)``. Required by S-axis operators
  (``nt_cs_*``); allowed for M-axis and elementwise ones.

Free nodes inherit the mode of their computed children, then of their
parents, defaulting to ``S``. Any edge whose two ends are evaluated in
different rounds is a hand-off: a full-width child is a **boundary**
(materialised to disk in canonical ``(S, D, M)`` order and re-read in the
consumer's orientation); a minute-degenerate child (every ``P`` node
included) is a **panel hand-off** (column 0 is exact, so it costs one
``(S, D)`` array). Same-mode nodes separated by a round of the other mode
(``S -> D -> S``) hand off exactly like different modes do.

Rounds are formed greedily from ready sets: at each step take every node of
one mode whose dependencies are done (transitively within the mode); ``P``
first when possible (free), otherwise the mode with the larger closure.
Leaves are never scheduled -- fields are re-read from the store in whatever
orientation a round needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

from ..axes import MODE_D, MODE_P, MODE_S
from .dag import Dag, Node


@dataclass
class Round:
    mode: str
    nodes: List[Node] = field(default_factory=list)   # topological order

    def __repr__(self) -> str:
        return f"Round({self.mode}, {len(self.nodes)} nodes)"


@dataclass
class Plan:
    rounds: List[Round]
    boundaries: List[Node]
    panels: List[Node]

    def summary(self) -> str:
        r = " -> ".join(f"{x.mode}[{len(x.nodes)}]" for x in self.rounds)
        return f"rounds: {r}; boundaries: {len(self.boundaries)}; panel hand-offs: {len(self.panels)}"


def _p_eligible(node: Node) -> bool:
    spec = node.spec
    if spec.face == "nd":
        return True
    if spec.family == "elementwise":
        return all(c.is_leaf and c.shape.degenerate or (not c.is_leaf and (c.mode == MODE_P or c.shape.degenerate))
                   for c in node.children)
    return False


def assign_modes(dag: Dag) -> None:
    computed = dag.computed
    # 1. hard requirements + P eligibility (children first, so P propagates upward)
    for n in computed:
        spec = n.spec
        if _p_eligible(n):
            n.mode = MODE_P
        elif spec.allowed_modes == (MODE_S,):
            n.mode = MODE_S
        elif spec.allowed_modes == (MODE_D,):
            n.mode = MODE_D
        else:
            n.mode = None
    # 2. free nodes: inherit from computed cube children (bottom-up)
    for n in computed:
        if n.mode is None:
            for c in n.children:
                if not c.is_leaf and c.mode in (MODE_S, MODE_D):
                    n.mode = c.mode
                    break
    # 3. then from parents (top-down), else default S
    for n in reversed(computed):
        if n.mode is None:
            for p in n.parents:
                if p.mode in (MODE_S, MODE_D):
                    n.mode = p.mode
                    break
            if n.mode is None:
                n.mode = MODE_S


def plan_rounds(dag: Dag) -> Plan:
    assign_modes(dag)
    remaining: Set[str] = {n.key for n in dag.computed}
    done: Set[str] = set()
    order: Dict[str, int] = {n.key: i for i, n in enumerate(dag.topo)}
    rounds: List[Round] = []

    def closure(mode: str) -> List[Node]:
        picked: Set[str] = set()
        changed = True
        while changed:
            changed = False
            for n in dag.computed:
                if n.key in picked or n.key not in remaining or n.mode != mode:
                    continue
                if all(c.is_leaf or c.key in done or c.key in picked for c in n.children):
                    picked.add(n.key)
                    changed = True
        return sorted((dag.by_key[k] for k in picked), key=lambda n: order[n.key])

    while remaining:
        p = closure(MODE_P)
        if p:
            chosen, mode = p, MODE_P
        else:
            s, d = closure(MODE_S), closure(MODE_D)
            if not s and not d:
                raise RuntimeError("scheduler deadlock (cyclic dependency?)")
            chosen, mode = (s, MODE_S) if len(s) >= len(d) else (d, MODE_D)
        rnd = Round(mode, chosen)
        for n in chosen:
            n.round_id = len(rounds)
            remaining.discard(n.key)
            done.add(n.key)
        rounds.append(rnd)
    # Hand-offs are decided by ROUND, not by mode: a value consumed in any later
    # round -- even one of the same mode, e.g. S -> D -> S -- has left the chunk
    # loop that produced it. Degenerate values (one number per stock-day,
    # including every P node) are kept as (S, D) panels; full-width values go
    # to disk in canonical (S, D, M) order.
    for n in dag.computed:
        n.boundary = n.panel_out = False
        if n.is_root and n.mode != MODE_P:
            n.panel_out = True
        for p in n.parents:
            if p.round_id != n.round_id:
                if n.shape.degenerate:
                    n.panel_out = True
                else:
                    n.boundary = True
        # Where the tradability mask must be re-applied under the "boundaries"
        # policy: an intraday (M-axis / elementwise) node can only produce a fake
        # finite value inside its own stock-day, so masking is only needed where a
        # value can leak to another day (input of a D-axis op), to another stock
        # (input of an NT op), or leaves the round (root / panel / disk).
        n.mask_out = n.is_root or n.panel_out or n.boundary or any(
            p.spec.compute_axis in ("D", "S") or p.spec.face == "nt" for p in n.parents)
    return Plan(rounds=rounds,
                boundaries=[n for n in dag.computed if n.boundary],
                panels=[n for n in dag.computed if n.panel_out])


__all__ = ["Round", "Plan", "assign_modes", "plan_rounds"]
