"""Token vocabulary and incremental RPN builder: the shared action space of
every RL / generative miner, over the three-axis registry.

RPN convention (unchanged from the previous project):
    unary:        operand, Op                  -> Unary
    binary:       lhs, rhs, Op                 -> Binary
    rolling:      operand, Window(w), Op       -> Rolling
    pair_rolling: lhs, rhs, Window(w), Op      -> PairRolling

What the three-axis type system adds to the grammar mask:

- operand compatibility is ``expression.operand_ok`` (DT/ND/NT rules: an
  ``in_*``/``nt_*``/``day_*`` needs genuine minute variation, an ND-face
  operator needs a one-value-per-stock-day operand, ``in_expand`` needs
  literal width 1);
- windows are shared tokens (the union of every face's window pool plus the
  panel's real minute count), so the mask on the *operator* checks that the
  pending window belongs to that operator's pool -- ``in_expand`` is valid
  only under the real minute count;
- ``SEP`` is offered only when the single stack item is a valid root: one
  value per stock-day, not a boolean/near-constant operator, reads a field.
- forward-looking operators are not in the vocabulary at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from ..axes import OpSpec
from ..catalog import Registry
from ..expression import (Constant, DailyFeature, Expression, Feature, ShapeError, is_valid_root, make_node,
                          operand_ok)


@dataclass(frozen=True)
class BegToken:
    pass


@dataclass(frozen=True)
class SepToken:
    pass


@dataclass(frozen=True)
class FeatureToken:
    field: str


@dataclass(frozen=True)
class DailyFeatureToken:
    field: str


@dataclass(frozen=True)
class OperatorToken:
    name: str


@dataclass(frozen=True)
class ConstantToken:
    value: float


@dataclass(frozen=True)
class WindowToken:
    window: int


Token = Union[BegToken, SepToken, FeatureToken, DailyFeatureToken, OperatorToken, ConstantToken, WindowToken]
BEG = BegToken()
SEP = SepToken()
StackItem = Union[Expression, int]


class Vocabulary:
    """The fixed, finite token set with stable indices."""

    def __init__(self, registry: Registry, fields: Optional[Sequence[str]] = None,
                 constants: Optional[Sequence[float]] = None, windows: Optional[Sequence[int]] = None,
                 n_minutes: int = 241, include_lookahead: bool = False,
                 allow_boolean_ops: bool = False, allow_resource_risk: bool = False,
                 daily_fields: Sequence[str] = ()) -> None:
        self.registry = registry
        self.n_minutes = int(n_minutes)
        self.fields = list(fields) if fields is not None else list(registry.fields)
        # A namespace PARALLEL to `fields` (mirrors `Registry.daily_fields`), empty
        # by default: with no daily fields named, no DailyFeatureToken is appended
        # below and every existing token index is unchanged (WP10's backward-compat
        # requirement -- a saved policy's action indices must still line up).
        self.daily_fields = list(daily_fields)
        self.constants = [float(c) for c in (constants if constants is not None else registry.constants)]
        specs = (registry.list_all() if include_lookahead
                 else registry.searchable(include_boolean=allow_boolean_ops,
                                          include_resource_risk=allow_resource_risk))
        self.allow_boolean_ops = bool(allow_boolean_ops)
        self.operators: List[OpSpec] = sorted(specs, key=lambda s: s.name)
        if windows is None:
            pool = set()
            for s in self.operators:
                pool.update(int(w) for w in s.windows)
            pool.add(self.n_minutes)
            windows = sorted(pool)
        self.windows = [int(w) for w in windows]
        tokens: List[Token] = [BEG, SEP]
        tokens += [FeatureToken(f) for f in self.fields]
        tokens += [OperatorToken(s.name) for s in self.operators]
        tokens += [ConstantToken(c) for c in self.constants]
        tokens += [WindowToken(w) for w in self.windows]
        # Appended AFTER every existing group: with the empty default this adds
        # nothing, so a vocabulary built before WP10 and one built after it (both
        # with no daily_fields) are byte-identical, and a non-empty daily_fields
        # only ever grows the index space upward, never renumbers a token below.
        tokens += [DailyFeatureToken(f) for f in self.daily_fields]
        self.token_to_index: Dict[Token, int] = {t: i for i, t in enumerate(tokens)}
        self.index_to_token: Dict[int, Token] = {i: t for t, i in self.token_to_index.items()}
        self._spec_by_name = {s.name: s for s in self.operators}

    @property
    def size(self) -> int:
        return len(self.token_to_index)

    def index_of(self, token: Token) -> int:
        try:
            return self.token_to_index[token]
        except KeyError:
            raise KeyError(f"token {token!r} is not in the vocabulary") from None

    @property
    def sep_index(self) -> int:
        return self.index_of(SEP)

    @property
    def beg_index(self) -> int:
        return self.index_of(BEG)

    def spec(self, name: str) -> OpSpec:
        return self._spec_by_name[name]

    def window_ok(self, spec: OpSpec, window: int) -> bool:
        if spec.restores_m:
            return window == self.n_minutes
        return window in spec.windows


def operator_logit_bias(vocab: Vocabulary, op_weight: Optional[Callable[[OpSpec], float]],
                         tiny: float = 1e-9) -> np.ndarray:
    """Per-token additive bias for a policy's pre-``action_mask`` logits, from the
    same ``op_weight`` a GP run passes to ``TypedTreeGenerator`` (resolved to a
    plain ``spec -> float`` callable -- see ``mining.generator._resolve_op_weight``,
    which a ``{name: weight}`` mapping also goes through before reaching either
    search). ``log(max(w, tiny))`` at every ``OperatorToken`` index, 0.0
    everywhere else (``BEG``/``SEP``/feature/daily-feature/constant/window
    tokens). Added to the policy logits BEFORE the grammar mask, a weight of 0
    (or any w <= 0) becomes an effective -inf-ish mask once softmax runs, and the
    one ``op_weight`` a caller builds steers both the GP and the RL search.

    ``op_weight=None`` -- the GP-side default too -- returns an all-zero array:
    no operator is biased.

    A non-finite weight is REJECTED (``ValueError``, naming the operator), the
    same bound ``generator._check_weight`` enforces on the GP side: ``max(w, tiny)``
    only floors the low end -- ``max(nan, 1e-9)`` is ``nan`` and ``max(inf, 1e-9)``
    is ``inf`` -- and one such entry poisons the whole logit row, so
    ``Categorical(logits=...)`` would later fail with no operator named (the
    expected caller derives weights from research output, where a mean over an
    empty group is NaN). A *negative* finite weight is floored to ``log(tiny)``
    instead of rejected, matching ``generator._pick_weighted``'s use-time
    ``max(0.0, w)`` clamp: it already means "never draw this operator", which is
    exactly what ``log(tiny)`` encodes.
    """
    bias = np.zeros(vocab.size, dtype=np.float64)
    if op_weight is None:
        return bias
    if not callable(op_weight):
        # The same ``{name: weight}`` mapping shape the GP side accepts, run through
        # the same resolver -- so the unknown-operator check and the ND/NT twin
        # inheritance apply identically on both sides, which is the whole of "one
        # shared op_weight steers both searches". Lazy import: ``generator`` does not
        # import ``tokens``, and this keeps that one-directional.
        from .generator import _resolve_op_weight
        op_weight = _resolve_op_weight(op_weight, vocab.registry)
    for spec in vocab.operators:
        w = float(op_weight(spec))
        if not math.isfinite(w):
            raise ValueError(f"op_weight for {spec.name!r} must be a finite float, got {w!r}")
        bias[vocab.index_of(OperatorToken(spec.name))] = math.log(max(w, tiny))
    return bias


class ExpressionBuilderError(ValueError):
    pass


class ExpressionBuilder:
    """Stack machine consuming RPN tokens into an Expression, with a grammar mask."""

    def __init__(self, vocabulary: Vocabulary) -> None:
        self.vocabulary = vocabulary
        self.stack: List[StackItem] = []
        self._began = False
        self._complete = False

    # -- mutation ---------------------------------------------------------------------
    def push(self, token: Token) -> None:
        if self._complete:
            raise ExpressionBuilderError("builder is already complete")
        if isinstance(token, BegToken):
            if self._began:
                raise ExpressionBuilderError("BEG already pushed")
            self._began = True
            return
        if not self._began:
            raise ExpressionBuilderError("push BEG first")
        if isinstance(token, SepToken):
            if not (len(self.stack) == 1 and isinstance(self.stack[0], Expression)):
                raise ExpressionBuilderError(f"cannot terminate: stack has {len(self.stack)} items")
            if not is_valid_root(self.stack[0]):
                raise ExpressionBuilderError(f"cannot terminate: {self.stack[0]!r} is not a valid factor root")
            self._complete = True
            return
        if isinstance(token, FeatureToken):
            if token.field not in self.vocabulary.fields:
                raise ExpressionBuilderError(f"unknown field {token.field!r}")
            self.stack.append(Feature(token.field))
            return
        if isinstance(token, DailyFeatureToken):
            if token.field not in self.vocabulary.daily_fields:
                raise ExpressionBuilderError(f"unknown daily field {token.field!r}")
            # No operand-type check here: a DailyFeature leaf is COLLAPSED/degenerate
            # by construction (expression.py), so the same static shape system that
            # already gates every operator in `_apply` is what will accept or refuse
            # it as an operand -- nothing daily-specific to re-check at push time.
            self.stack.append(DailyFeature(token.field))
            return
        if isinstance(token, ConstantToken):
            self.stack.append(Constant(token.value))
            return
        if isinstance(token, WindowToken):
            if not (self.stack and isinstance(self.stack[-1], Expression)):
                raise ExpressionBuilderError("a window must follow an expression operand")
            self.stack.append(int(token.window))
            return
        if isinstance(token, OperatorToken):
            self._apply(self.vocabulary.spec(token.name))
            return
        raise ExpressionBuilderError(f"unknown token type {type(token).__name__}")

    def _apply(self, spec: OpSpec) -> None:
        n = 1 if spec.category in ("unary", "rolling") else 2
        window = None
        need = n
        if spec.requires_window:
            if not (self.stack and isinstance(self.stack[-1], int)):
                raise ExpressionBuilderError(f"{spec.name} needs a window on top of the stack")
            window = self.stack[-1]
            if not self.vocabulary.window_ok(spec, window):
                raise ExpressionBuilderError(f"window {window} is not in {spec.name}'s pool")
            need += 1
        if len(self.stack) < need:
            raise ExpressionBuilderError(f"{spec.name} needs {need} stack items")
        operands = self.stack[-need: -1] if window is not None else self.stack[-need:]
        if not all(isinstance(o, Expression) for o in operands):
            raise ExpressionBuilderError(f"{spec.name} needs expression operands")
        try:
            node = make_node(spec, tuple(operands), window)
        except ShapeError as exc:
            raise ExpressionBuilderError(str(exc)) from None
        del self.stack[-need:]
        self.stack.append(node)

    # -- inspection -------------------------------------------------------------------
    def is_complete(self) -> bool:
        return self._complete

    def build(self) -> Expression:
        if not self._complete:
            raise ExpressionBuilderError("builder is not complete")
        return self.stack[0]

    def valid_next_tokens(self) -> List[int]:
        v = self.vocabulary
        if self._complete:
            return []
        if not self._began:
            return [v.beg_index]
        out: List[Token] = ([FeatureToken(f) for f in v.fields] + [DailyFeatureToken(f) for f in v.daily_fields]
                             + [ConstantToken(c) for c in v.constants])
        st = self.stack
        top_expr = bool(st) and isinstance(st[-1], Expression)
        if top_expr:
            out += [WindowToken(w) for w in v.windows]
        for spec in v.operators:
            n = 1 if spec.category in ("unary", "rolling") else 2
            if spec.requires_window:
                if not (st and isinstance(st[-1], int)) or len(st) < n + 1:
                    continue
                if not v.window_ok(spec, st[-1]):
                    continue
                operands = st[-n - 1: -1]
            else:
                if len(st) < n:
                    continue
                operands = st[-n:]
            if all(isinstance(o, Expression) and operand_ok(spec, o) for o in operands):
                out.append(OperatorToken(spec.name))
        if len(st) == 1 and isinstance(st[0], Expression) and is_valid_root(st[0]):
            out.append(SEP)
        return [v.index_of(t) for t in out]


__all__ = ["BegToken", "SepToken", "FeatureToken", "DailyFeatureToken", "OperatorToken", "ConstantToken",
           "WindowToken", "Token", "BEG", "SEP", "Vocabulary", "ExpressionBuilder", "ExpressionBuilderError",
           "operator_logit_bias"]
