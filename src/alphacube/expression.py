"""Expression trees over the cube, with static shape typing.

Two properties are tracked per node, exactly the pair the previous project
found it had to keep separate (width is not information):

- ``collapsed``: literal minute width is 1 (a ``day_*`` result, or an ND
  panel). Gates ``in_expand``, whose torch ``.expand()`` needs width 1.
- ``degenerate``: no genuine per-minute variation, whatever the width. This
  is the real "one value per stock-day" property: a factor is usable as a
  daily signal iff its root is degenerate, and an ``in_*``/``nt_*``/``day_*``
  operator applied to a degenerate operand is structurally meaningless (the
  kernels do not fail, they return identity/constant/clamped-zero).

ND-face operators consume degenerate nodes (the engine hands them the
``(D, S)`` panel), NT/in/day operators consume non-degenerate nodes,
``cross_*`` and elementwise operators pass shape through.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .axes import OpSpec


@dataclass(frozen=True)
class Shape:
    collapsed: bool
    degenerate: bool
    width: Optional[int] = None        # literal minute width when narrower than the panel (1 = collapsed)

    @property
    def daily(self) -> bool:
        return self.degenerate


FULL = Shape(collapsed=False, degenerate=False)
COLLAPSED = Shape(collapsed=True, degenerate=True, width=1)
WIDE_DEGENERATE = Shape(collapsed=False, degenerate=True)


class ShapeError(TypeError):
    """An operator was applied to an operand of the wrong minute-axis type."""


class Expression(ABC):
    spec: Optional[OpSpec] = None
    __slots__ = ("_key", "_shape")

    # -- structure ---------------------------------------------------------
    @property
    def operands(self) -> Tuple["Expression", ...]:
        return ()

    @property
    def window(self) -> Optional[int]:
        return None

    @property
    def name(self) -> str:
        return self.spec.name if self.spec is not None else type(self).__name__

    def key(self) -> str:
        """Structural identity for CSE: type-tagged, constants exact, windows included."""
        k = getattr(self, "_key", None)
        if k is None:
            k = self._make_key()
            object.__setattr__(self, "_key", k)
        return k

    def _make_key(self) -> str:
        raise NotImplementedError

    @property
    def shape(self) -> Shape:
        s = getattr(self, "_shape", None)
        if s is None:
            s = self._infer_shape()
            object.__setattr__(self, "_shape", s)
        return s

    def _infer_shape(self) -> Shape:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self._repr()

    def _repr(self) -> str:
        raise NotImplementedError

    def __eq__(self, other) -> bool:
        return isinstance(other, Expression) and self.key() == other.key()

    def __hash__(self) -> int:
        return hash(self.key())

    def nodes(self) -> List["Expression"]:
        """Pre-order list of all nodes (with duplicates for shared subtrees)."""
        out = [self]
        for c in self.operands:
            out.extend(c.nodes())
        return out

    def depth(self) -> int:
        return 1 + max((c.depth() for c in self.operands), default=0)


class Feature(Expression):
    __slots__ = ("field",)

    def __init__(self, field: str) -> None:
        self.field = field

    def _make_key(self) -> str:
        return f"F:{self.field}"

    def _infer_shape(self) -> Shape:
        return FULL

    def _repr(self) -> str:
        return self.field


class Constant(Expression):
    __slots__ = ("value",)

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def _make_key(self) -> str:
        return f"C:{self.value!r}"

    def _infer_shape(self) -> Shape:
        # Same value at every minute: materialized full width, no variation.
        return WIDE_DEGENERATE

    def _repr(self) -> str:
        v = self.value
        return repr(int(v)) if v.is_integer() and abs(v) < 1e15 else repr(v)


def operand_ok(spec: OpSpec, operand: Expression, strict: bool = True) -> bool:
    """May ``operand`` be fed to an operator described by ``spec``?

    ``strict=False`` is the research/legacy relaxation (never used by the
    search grammar): ``in_expand`` accepts any minute-degenerate operand (the
    engine hands it column 0), and a ``day_*`` reducer accepts a literal
    width-1 operand (the legacy framework silently computed the identity /
    zero there, so historical factors depend on it).
    """
    sh = operand.shape
    if spec.restores_m:
        return sh.collapsed if strict else sh.degenerate
    if spec.needs_full_m:
        if not sh.degenerate:
            return True
        return (not strict) and spec.reduces_m and sh.collapsed
    if spec.needs_daily:
        return sh.degenerate
    return True


def _why_not(spec: OpSpec, operand: Expression) -> str:
    sh = operand.shape
    if spec.restores_m:
        return f"{spec.name} needs a width-1 (day-collapsed) operand, got {operand!r}"
    if spec.needs_full_m:
        return (f"{spec.name} ({spec.face}/{spec.family}) needs genuine per-minute variation, "
                f"got {'collapsed' if sh.collapsed else 'minute-degenerate'} operand {operand!r}")
    return f"{spec.name} (ND face) needs a one-value-per-stock-day operand, got {operand!r}"


def _passthrough(shapes: Tuple[Shape, ...]) -> Shape:
    """cross_* and elementwise: broadcast of the operands' minute types."""
    if not shapes:
        return WIDE_DEGENERATE
    widths = {s.width for s in shapes if s.width is not None and s.width != 1}
    width: Optional[int] = next(iter(widths)) if len(widths) == 1 else None
    collapsed = all(s.collapsed for s in shapes)
    return Shape(collapsed=collapsed, degenerate=all(s.degenerate for s in shapes),
                 width=1 if collapsed else width)


def _output_shape(spec: OpSpec, shapes: Tuple[Shape, ...]) -> Shape:
    if spec.reduces_m:
        return COLLAPSED
    if spec.restores_m:
        return WIDE_DEGENERATE
    if spec.face == "nd":
        return COLLAPSED
    if spec.needs_full_m:            # in_*, nt_*: width of the (widest) operand
        w = _passthrough(shapes).width
        return Shape(collapsed=False, degenerate=False, width=w)
    return _passthrough(shapes)


class _OpNode(Expression):
    __slots__ = ("spec", "_operands", "_window")

    def __init__(self, spec: OpSpec, operands: Tuple[Expression, ...], window: Optional[int] = None,
                 strict: bool = True) -> None:
        self.spec = spec
        self._operands = tuple(operands)
        self._window = None if window is None else int(window)
        expected = 1 if spec.category in ("unary", "rolling") else 2
        if len(self._operands) != expected:
            raise ShapeError(f"{spec.name} takes {expected} operand(s), got {len(self._operands)}")
        if spec.requires_window and self._window is None:
            raise ShapeError(f"{spec.name} requires a window")
        if not spec.requires_window and self._window is not None:
            raise ShapeError(f"{spec.name} takes no window")
        for o in self._operands:
            if not operand_ok(spec, o, strict):
                raise ShapeError(_why_not(spec, o))

    @property
    def operands(self) -> Tuple[Expression, ...]:
        return self._operands

    @property
    def window(self) -> Optional[int]:
        return self._window

    def _make_key(self) -> str:
        inner = ",".join(o.key() for o in self._operands)
        w = "" if self._window is None else f"|w{self._window}"
        return f"O:{self.spec.name}({inner}){w}"

    def _infer_shape(self) -> Shape:
        return _output_shape(self.spec, tuple(o.shape for o in self._operands))

    def _repr(self) -> str:
        args = [repr(o) for o in self._operands]
        if self._window is not None:
            args.append(str(self._window))
        return f"{self.spec.name}({', '.join(args)})"


class Unary(_OpNode):
    __slots__ = ()

    def __init__(self, spec: OpSpec, operand: Expression, strict: bool = True) -> None:
        super().__init__(spec, (operand,), strict=strict)


class Binary(_OpNode):
    __slots__ = ()

    def __init__(self, spec: OpSpec, lhs: Expression, rhs: Expression, strict: bool = True) -> None:
        super().__init__(spec, (lhs, rhs), strict=strict)


class Rolling(_OpNode):
    __slots__ = ()

    def __init__(self, spec: OpSpec, operand: Expression, window: int, strict: bool = True) -> None:
        super().__init__(spec, (operand,), window, strict=strict)


class PairRolling(_OpNode):
    __slots__ = ()

    def __init__(self, spec: OpSpec, lhs: Expression, rhs: Expression, window: int, strict: bool = True) -> None:
        super().__init__(spec, (lhs, rhs), window, strict=strict)


def make_node(spec: OpSpec, operands: Tuple[Expression, ...], window: Optional[int] = None,
              strict: bool = True) -> Expression:
    if spec.category == "unary":
        return Unary(spec, operands[0], strict=strict)
    if spec.category == "binary":
        return Binary(spec, operands[0], operands[1], strict=strict)
    if spec.category == "rolling":
        return Rolling(spec, operands[0], window, strict=strict)
    if spec.category == "pair_rolling":
        return PairRolling(spec, operands[0], operands[1], window, strict=strict)
    raise ShapeError(f"{spec.name}: category {spec.category!r} cannot be built by make_node")


LITERAL_TYPES = (bool, int, float, str, type(None))


class GeneralCall(Expression):
    """A call to any function of the kernel library that the four-category
    grammar cannot express: arbitrary arity, raw literal arguments
    (``in_slice(x, 10, 229)``, ``day_pdf_tail_sum_m(x, f, 50, False, 'rank')``),
    tensor-list arguments (``in_stddev_rolling_multi([o, c, h, l], 5)``),
    three-tensor calls (``select_where(cond, x, y)``), width-changing calls.

    Never produced by the search grammar: only the parser builds it, for
    research / legacy expressions. ``spec`` is an ``OpSpec`` with category
    ``"general"`` (see ``fullcatalog``), so the scheduler and the engine treat
    it by face / compute axis exactly like a registry operator. ``exempt``
    lists tensor-argument positions that are legitimately day-collapsed
    indices (``day_locate(x, day_argmax(y))``) and skip the operand check.
    """

    __slots__ = ("spec", "args", "_operands", "exempt")

    def __init__(self, spec: OpSpec, args: tuple, strict: bool = True,
                 exempt: frozenset = frozenset()) -> None:
        ops = []
        for i, a in enumerate(args):
            if isinstance(a, tuple):
                if not a or not all(isinstance(e, Expression) for e in a):
                    raise ShapeError(f"{spec.name}: argument {i} must be a non-empty tuple of expressions")
                ops.extend(a)
            elif isinstance(a, Expression):
                ops.append(a)
            elif not isinstance(a, LITERAL_TYPES):
                raise ShapeError(f"{spec.name}: argument {i} of type {type(a).__name__} is not supported")
        self.spec = spec
        self.args = tuple(args)
        self._operands = tuple(ops)
        self.exempt = frozenset(exempt)
        for i, o in enumerate(self._operands):
            if i in self.exempt:
                continue
            if not operand_ok(spec, o, strict):
                raise ShapeError(_why_not(spec, o))

    @property
    def operands(self) -> Tuple[Expression, ...]:
        return self._operands

    def literal_width(self) -> Optional[int]:
        if self.spec.name == "in_slice" and len(self.args) == 3:
            _, a, b = self.args
            if isinstance(a, int) and isinstance(b, int) and not isinstance(a, bool) and not isinstance(b, bool):
                return b - a + 1
        return None

    def _make_key(self) -> str:
        def rend(a):
            if isinstance(a, tuple):
                return "[" + ",".join(e.key() for e in a) + "]"
            if isinstance(a, Expression):
                return a.key()
            return f"L:{a!r}"
        return f"G:{self.spec.name}({','.join(rend(a) for a in self.args)})"

    def _infer_shape(self) -> Shape:
        spec = self.spec
        if spec.reduces_m:
            return COLLAPSED
        if spec.restores_m:
            return WIDE_DEGENERATE
        if spec.face == "nd":
            return COLLAPSED
        w = self.literal_width()
        shapes = tuple(o.shape for o in self._operands)
        if w is not None:
            # A width-1 slice is a REAL selected minute, but has no minute-axis
            # variation left, so it behaves as collapsed for everything downstream.
            if w == 1:
                return COLLAPSED
            return Shape(collapsed=False, degenerate=all(s.degenerate for s in shapes), width=w)
        if spec.needs_full_m:
            return Shape(collapsed=False, degenerate=False, width=_passthrough(shapes).width)
        return _passthrough(shapes)

    def _repr(self) -> str:
        def rend(a):
            if isinstance(a, tuple):
                return "[" + ", ".join(repr(e) for e in a) + "]"
            return repr(a)
        return f"{self.spec.name}({', '.join(rend(a) for a in self.args)})"

    def with_operands(self, new_ops: Tuple[Expression, ...]) -> "GeneralCall":
        it = iter(new_ops)
        args = []
        for a in self.args:
            if isinstance(a, tuple):
                args.append(tuple(next(it) for _ in a))
            elif isinstance(a, Expression):
                args.append(next(it))
            else:
                args.append(a)
        return GeneralCall(self.spec, tuple(args), strict=False, exempt=self.exempt)


def replace(expr: Expression, target: Expression, replacement: Expression) -> Expression:
    """Copy of ``expr`` with the node ``target`` (by identity) replaced; may raise ShapeError."""
    if expr is target:
        return replacement
    if not expr.operands:
        return expr
    new_ops = tuple(replace(o, target, replacement) for o in expr.operands)
    if all(n is o for n, o in zip(new_ops, expr.operands)):
        return expr
    if isinstance(expr, GeneralCall):
        return expr.with_operands(new_ops)
    return make_node(expr.spec, new_ops, expr.window)


def is_valid_root(expr: Expression) -> bool:
    """One value per stock-day, not a boolean/near-constant indicator, and reads at least one field."""
    if not expr.shape.degenerate:
        return False
    if expr.spec is not None and expr.spec.non_root:
        return False
    return any(isinstance(n, Feature) for n in expr.nodes())


def contains_lookahead(expr: Expression) -> bool:
    return any(n.spec is not None and n.spec.lookahead for n in expr.nodes())


def max_window(expr: Expression, axis: str) -> int:
    """Largest rolling window along ``axis`` ("D" or "M"); the D value is the warm-up need.
    For a GeneralCall along that axis the largest integer literal argument is taken."""
    best = 0
    for n in expr.nodes():
        if n.spec is None or n.spec.compute_axis != axis or n.spec.restores_m:
            continue
        if n.window is not None:
            best = max(best, n.window)
        elif isinstance(n, GeneralCall):
            ints = [a for a in n.args if isinstance(a, int) and not isinstance(a, bool)]
            if ints:
                best = max(best, max(ints))
    return best


__all__ = [
    "Shape", "FULL", "COLLAPSED", "WIDE_DEGENERATE", "ShapeError", "Expression", "Feature",
    "Constant", "Unary", "Binary", "Rolling", "PairRolling", "GeneralCall", "LITERAL_TYPES",
    "make_node", "replace", "operand_ok", "is_valid_root", "contains_lookahead", "max_window",
]
