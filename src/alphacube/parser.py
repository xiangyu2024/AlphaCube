"""String -> Expression.

One namespace: fields are cube leaves (``Close`` ...), operators are the
search registry (``day_mean``, ``ts_mean`` = ND face, ``nt_ts_mean`` = NT
face) plus, as a fallback, the full kernel catalog (``GeneralCall``) for
anything the four-category grammar cannot express.

Legacy spellings are accepted: lowercase minute fields, daily ``*_1d``
macros (``close_1d`` -> ``day_last(Close)``), Python operators ``+ - * / **``,
``|`` / ``&`` (logical or/and), comparisons ``== > < >= <=``, ``x.float()``,
``abs`` / ``sum`` / ``rank`` / ``_max`` / ``_min`` / ``_pow`` aliases,
list arguments for the multi-tensor family, ``in_expand(x)`` without a width.

Two research-only relaxations, off for the search grammar (``strict=True``):

- ``strict=False``: ``day_*`` over a literal width-1 operand, ``in_expand``
  over any minute-degenerate operand (see ``expression.operand_ok``);
- ``coerce_daily=True``: if the expression uses NO minute-level operator at
  all (only ND-face and elementwise ones), every bare field means its DAILY
  aggregate -- ``ts_corr(open, vol, 10)`` is daily open vs daily volume,
  ``select_where(is_lt(ts_mean(vol, 20), vol), ...)`` compares daily
  volumes -- exactly what the old ND-mode parser did. In an expression that
  does use minute operators, bare fields are minute fields and only the
  direct operands of ND-face operators are coerced when they are raw-field
  subtrees.
"""

from __future__ import annotations

import ast
import inspect
from typing import Dict, Optional, Sequence, Tuple

from .axes import OpSpec
from .catalog import Registry, default_registry
from .expression import Constant, DailyFeature, Expression, Feature, GeneralCall, ShapeError, make_node, operand_ok
from .fullcatalog import COLLAPSED_INDEX_OPERANDS, OPERATOR_ARITY_ALIASES, OPERATOR_NAME_ALIASES, default_full_catalog

FIELD_ALIASES: Dict[str, str] = {
    "close": "Close", "open": "Open", "high": "High", "low": "Low",
    "vol": "Volume", "volume": "Volume", "vwap": "Vwap", "ret": "Ret", "amt": "Amt", "returns": "Ret",
}

# Lowercase/legacy spellings of the daily-field namespace (Registry.daily_fields,
# catalog.DAILY_FIELDS), parallel to FIELD_ALIASES. Deliberately NO bare "age": it
# is too easily confused with a window-length argument spelled the same way.
DAILY_FIELD_ALIASES: Dict[str, str] = {
    "floatcap": "FloatCap", "float_cap": "FloatCap", "circ_mv": "FloatCap",
    "totalcap": "TotalCap", "total_cap": "TotalCap", "total_mv": "TotalCap",
    "floatshares": "FloatShares", "float_shares": "FloatShares",
    "limitwidth": "LimitWidth", "limit_width": "LimitWidth",
    "boardcode": "BoardCode", "board": "BoardCode",
    "listage": "ListAge", "list_age": "ListAge",
    "logfloatcap": "LogFloatCap", "log_float_cap": "LogFloatCap",
    "logtotalcap": "LogTotalCap", "log_total_cap": "LogTotalCap",
    "logfloatshares": "LogFloatShares", "log_float_shares": "LogFloatShares",
}

# Daily fields as cube expressions (parsed recursively). Also the daily
# aggregate used when an ND-face operator receives a raw minute field.
DAILY_MACROS: Dict[str, str] = {
    "close_1d": "day_last(Close)",
    "open_1d": "day_first(Open)",
    "high_1d": "day_max(High)",
    "low_1d": "day_min(Low)",
    "vol_1d": "day_sum(Volume)",
    "volume_1d": "day_sum(Volume)",
    "amt_1d": "day_sum(Amt)",
    "vwap_1d": "div(day_sum(Amt), day_sum(Volume))",
    "ret_1d": "sub(div(day_last(Close), ts_delay(day_last(Close), 1)), 1)",
}
# Daily-only names of the previous project's ND registry (seed libraries, old reprs).
DAILY_NAME_ALIASES: Dict[str, str] = {"Return": "ret_1d", "Amount": "amt_1d", "AdjClose": "close_1d"}
DAILY_OF_FIELD: Dict[str, str] = {
    "Close": "close_1d", "Open": "open_1d", "High": "high_1d", "Low": "low_1d",
    "Volume": "vol_1d", "Amt": "amt_1d", "Vwap": "vwap_1d", "Ret": "ret_1d",
}

_BINOPS = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div", ast.Pow: "pow",
           ast.BitOr: "logical_or", ast.BitAnd: "logical_and"}
_CMPOPS = {ast.Gt: "greater", ast.Lt: "less", ast.Eq: "eq", ast.GtE: "greater_equal", ast.LtE: "less_equal"}
_CATEGORY_ARITY = {"unary": 1, "binary": 2, "rolling": 2, "pair_rolling": 3}


class ParseError(ValueError):
    pass


class _Parser:
    def __init__(self, registry: Registry, strict: bool, coerce_daily: bool, n_minutes: int,
                 allow_general: bool) -> None:
        self.reg = registry
        self.strict = strict
        self.coerce_daily = coerce_daily
        self.n_minutes = int(n_minutes)
        self.full = default_full_catalog() if allow_general else None
        # Whether a BARE minute field means its daily aggregate (coerce_daily's own
        # mechanism) -- named away from "daily_fields" (the Registry's namespace of
        # DailyFeature names, e.g. FloatCap) so the two do not read as the same thing
        # two lines apart.
        self.coerce_bare_fields = False

    # -- entry --------------------------------------------------------------------------
    def parse(self, text: str) -> Expression:
        try:
            tree = ast.parse(text.strip(), mode="eval")
        except SyntaxError as exc:
            raise ParseError(f"not a valid expression: {text!r} ({exc})") from None
        if self.coerce_daily and not self.uses_minute_operators(tree):
            self.coerce_bare_fields = True
        return self.convert(tree.body)

    def uses_minute_operators(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                name = node.func.id
                spec = None
                resolved = self.registry_name(name, len(node.args))
                if resolved is not None:
                    spec = self.reg.get(resolved)
                elif self.full is not None:
                    spec = self.full.get(name)
                if spec is not None and spec.face in ("dt", "nt"):
                    return True
            if isinstance(node, ast.Name) and node.id in DAILY_MACROS:
                continue
        return False

    # -- nodes ----------------------------------------------------------------------------
    def convert(self, node: ast.AST) -> Expression:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return Constant(float(node.value))
        if isinstance(node, ast.Name):
            return self.name(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = self.convert(node.operand)
            if isinstance(inner, Constant):
                return Constant(-inner.value)
            return self.apply("neg", (inner,))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
            return self.convert(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return self.apply(_BINOPS[type(node.op)], (self.convert(node.left), self.convert(node.right)))
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or type(node.ops[0]) not in _CMPOPS:
                raise ParseError("only single comparisons with ==, >, <, >=, <= are supported")
            return self.apply(_CMPOPS[type(node.ops[0])], (self.convert(node.left), self.convert(node.comparators[0])))
        if isinstance(node, ast.Call):
            return self.call(node)
        raise ParseError(f"unsupported syntax: {ast.dump(node)[:80]}")

    def name(self, ident: str) -> Expression:
        field = ident if ident in self.reg.fields else FIELD_ALIASES.get(ident)
        if field is not None:
            if self.coerce_bare_fields:
                return self.macro(DAILY_OF_FIELD[field])
            return Feature(field)
        if ident in DAILY_MACROS:
            return self.macro(ident)
        if ident in DAILY_NAME_ALIASES:
            return self.macro(DAILY_NAME_ALIASES[ident])
        # The daily-leaf namespace (Registry.daily_fields, e.g. FloatCap): checked
        # LAST, after every minute-field spelling and macro above, and deliberately
        # NOT routed through coerce_bare_fields -- a daily leaf has no minute-level
        # definition, so there is no day_* macro to rewrite it into.
        if ident in self.reg.daily_fields:
            return DailyFeature(ident)
        dfield = DAILY_FIELD_ALIASES.get(ident)
        if dfield is not None and dfield in self.reg.daily_fields:
            return DailyFeature(dfield)
        raise ParseError(f"unknown field {ident!r}; cube fields are {self.reg.fields}, "
                         f"daily fields are {self.reg.daily_fields}")

    def macro(self, name: str) -> Expression:
        saved = self.coerce_bare_fields
        self.coerce_bare_fields = False                  # macro bodies are written in minute terms
        try:
            return self.convert(ast.parse(DAILY_MACROS[name], mode="eval").body)
        finally:
            self.coerce_bare_fields = saved

    def call(self, node: ast.Call) -> Expression:
        if isinstance(node.func, ast.Attribute) and not node.args and node.func.attr in ("float", "double"):
            return self.convert(node.func.value)            # legacy dtype casts are no-ops here
        if not isinstance(node.func, ast.Name):
            raise ParseError("only plain function calls are supported")
        if node.keywords:
            raise ParseError(f"{node.func.id}: keyword arguments are not supported")
        name, args = node.func.id, list(node.args)
        n = len(args)
        # 1. four-category registry (after aliases), arity must match
        resolved = self.registry_name(name, n)
        if resolved is not None:
            spec = self.reg.get(resolved)
            if _CATEGORY_ARITY[spec.category] == n:
                window = None
                if spec.requires_window:
                    window = self.window_literal(args[-1], name)
                    args = args[:-1]
                operands = tuple(self.convert(a) for a in args)
                return self.apply(resolved, operands, window)
            if spec.restores_m and n == 1:                     # legacy in_expand(x): full panel width
                return self.apply(resolved, (self.convert(args[0]),), self.n_minutes)
        # 2. full catalog
        if self.full is None:
            raise ParseError(f"unknown operator {name!r}")
        gspec = self.full.get(name)
        if gspec is None:
            if name in self.full.ambiguous:
                raise ParseError(f"operator {name!r} is defined with different bodies in more than one library module")
            raise ParseError(f"unknown operator {name!r}")
        lo, hi = _positional_arity(gspec.fn)
        if not (lo <= n <= hi):
            span = f"{lo}" if lo == hi else f"{lo} to {hi}"
            raise ParseError(f"{name} expects {span} argument(s), got {n}")
        converted = []
        for a in args:
            if isinstance(a, ast.Constant) and (isinstance(a.value, (bool, int, float, str)) or a.value is None):
                converted.append(a.value)
            elif isinstance(a, (ast.List, ast.Tuple)):
                converted.append(tuple(self.convert(e) for e in a.elts))
            elif isinstance(a, ast.UnaryOp) and isinstance(a.op, ast.USub) and isinstance(a.operand, ast.Constant) \
                    and isinstance(a.operand.value, (int, float)) and not isinstance(a.operand.value, bool):
                converted.append(-a.operand.value)
            else:
                converted.append(self.convert(a))
        exempt = COLLAPSED_INDEX_OPERANDS.get(gspec.name, frozenset())
        if gspec.face == "nd" and self.coerce_daily:
            converted = [self.coerce(x, gspec) if isinstance(x, Expression) else
                         (tuple(self.coerce(e, gspec) for e in x) if isinstance(x, tuple) else x)
                         for x in converted]
        try:
            return GeneralCall(gspec, tuple(converted), strict=self.strict, exempt=exempt)
        except ShapeError as exc:
            raise ParseError(str(exc)) from None

    def registry_name(self, name: str, n: int) -> Optional[str]:
        if name in self.reg:
            return name
        arity = OPERATOR_ARITY_ALIASES.get(name)
        if arity and n in arity and arity[n] in self.reg:
            return arity[n]
        alias = OPERATOR_NAME_ALIASES.get(name)
        if alias and alias in self.reg:
            return alias
        return None

    def window_literal(self, node: ast.AST, op: str) -> int:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return int(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            return -int(node.operand.value)
        raise ParseError(f"{op}: last argument must be an integer window")

    # -- building ----------------------------------------------------------------------------
    def apply(self, name: str, operands: Tuple[Expression, ...], window: Optional[int] = None) -> Expression:
        spec = self.reg.get(name) if name in self.reg else None
        if spec is None:
            # comparison sugar that only exists in the full catalog (greater_equal / less_equal)
            gspec = self.full.get(name) if self.full is not None else None
            if gspec is None:
                raise ParseError(f"unknown operator {name!r}")
            try:
                return GeneralCall(gspec, tuple(operands), strict=self.strict)
            except ShapeError as exc:
                raise ParseError(str(exc)) from None
        if spec.face == "nd" and self.coerce_daily:
            operands = tuple(self.coerce(o, spec) for o in operands)
        try:
            return make_node(spec, operands, window, strict=self.strict)
        except ShapeError as exc:
            raise ParseError(str(exc)) from None

    def coerce(self, operand: Expression, spec: OpSpec) -> Expression:
        """Daily aggregate of a raw-field subtree fed to an ND-face operator."""
        if operand_ok(spec, operand, self.strict):
            return operand
        return self.daily_of(operand)

    def daily_of(self, expr: Expression) -> Expression:
        if expr.shape.degenerate:
            return expr
        if isinstance(expr, Feature):
            macro = DAILY_OF_FIELD.get(expr.field)
            if macro is None:
                raise ParseError(f"no daily definition for field {expr.field!r}")
            return self.macro(macro)
        if expr.spec is not None and expr.spec.face == "shared":
            if isinstance(expr, GeneralCall):
                return expr.with_operands(tuple(self.daily_of(o) for o in expr.operands))
            return make_node(expr.spec, tuple(self.daily_of(o) for o in expr.operands), expr.window, strict=self.strict)
        raise ParseError(f"an ND-face operator needs a one-value-per-stock-day operand, got {expr!r} "
                         "(wrap the minute-level subtree in day_*)")


def _positional_arity(fn) -> Tuple[int, int]:
    """(min, max) positional argument counts of a kernel; max is unbounded for *args."""
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return 0, 10**6
    lo = hi = 0
    for prm in params:
        if prm.kind in (prm.POSITIONAL_ONLY, prm.POSITIONAL_OR_KEYWORD):
            hi += 1
            if prm.default is prm.empty:
                lo += 1
        elif prm.kind == prm.VAR_POSITIONAL:
            return lo, 10**6
    return lo, hi


def parse(text: str, registry: Optional[Registry] = None, *, strict: bool = True, coerce_daily: bool = False,
          n_minutes: int = 241, allow_general: bool = True) -> Expression:
    """Strict by default (the search grammar's rules, no silent daily coercion);
    ``parse_legacy`` / ``research.parse_many`` turn the research relaxations on."""
    return _Parser(registry or default_registry(), strict, coerce_daily, n_minutes, allow_general).parse(text)


def parse_legacy(text: str, registry: Optional[Registry] = None, n_minutes: int = 241) -> Expression:
    """Research/legacy settings: lenient operand rules, daily coercion, full catalog."""
    return parse(text, registry, strict=False, coerce_daily=True, n_minutes=n_minutes, allow_general=True)


__all__ = ["parse", "parse_legacy", "ParseError", "FIELD_ALIASES", "DAILY_MACROS", "DAILY_OF_FIELD",
           "DAILY_NAME_ALIASES", "DAILY_FIELD_ALIASES"]
