"""Per-expression migration from the legacy 240-bar minute grid to the V2
241-bar grid (09:30 auction bar at index 0): explicit ``in_slice`` clock
positions shift by one, literal full-day windows 240 become 241, and the
grid-aware operators are counted for the audit. Ported unchanged."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Tuple

FULL_DAY_WINDOW_CALLS = frozenset({"in_roll_labels", "in_argmax", "in_argmin", "in_multireg"})
GRID_AWARE_OPERATOR_CALLS = frozenset({"in_expand", "in_half", "delete_open_close"})


@dataclass(frozen=True)
class FactorGridMigration:
    factor: str
    changed: bool
    slice_calls_shifted: int
    full_day_windows_changed: int
    grid_aware_operator_calls: int
    policy: str

    def to_dict(self) -> dict:
        return asdict(self)


class _Transformer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.slice_calls_shifted = 0
        self.full_day_windows_changed = 0
        self.grid_aware_operator_calls = 0

    def visit_Call(self, node: ast.Call):  # noqa: N802
        self.generic_visit(node)
        name = node.func.id if isinstance(node.func, ast.Name) else None
        if name in GRID_AWARE_OPERATOR_CALLS:
            self.grid_aware_operator_calls += 1
        if name == "in_slice" and len(node.args) >= 3:
            start, end = node.args[1], node.args[2]
            if (isinstance(start, ast.Constant) and isinstance(start.value, int) and isinstance(end, ast.Constant)
                    and isinstance(end.value, int) and 0 <= start.value <= end.value <= 239):
                start.value += 1
                end.value += 1
                self.slice_calls_shifted += 1
        if name in FULL_DAY_WINDOW_CALLS:
            for a in node.args[1:]:
                if isinstance(a, ast.Constant) and a.value == 240:
                    a.value = 241
                    self.full_day_windows_changed += 1
        return node


def migrate_factor_expression(name: str, source: str) -> Tuple[str, FactorGridMigration]:
    tree = ast.parse(source, mode="eval")
    t = _Transformer()
    migrated = ast.fix_missing_locations(t.visit(tree))
    expr_changed = bool(t.slice_calls_shifted or t.full_day_windows_changed)
    changed = bool(expr_changed or t.grid_aware_operator_calls)
    output = ast.unparse(migrated) if expr_changed else source
    if expr_changed and t.grid_aware_operator_calls:
        policy = "rewrite_expression_and_use_241_aware_operator"
    elif expr_changed:
        policy = "shift_explicit_clock_positions_and_expand_full_day_window"
    elif t.grid_aware_operator_calls:
        policy = "use_241_aware_operator_without_expression_rewrite"
    else:
        policy = "no_240_grid_dependency"
    return output, FactorGridMigration(name, changed, t.slice_calls_shifted, t.full_day_windows_changed,
                                       t.grid_aware_operator_calls, policy)


def migrate_factor_dict(factors: Dict[str, str]) -> Tuple[Dict[str, str], List[FactorGridMigration]]:
    out, audits = {}, []
    for name, src in factors.items():
        out[name], audit = migrate_factor_expression(name, src)
        audits.append(audit)
    return out, audits


def migration_rows(audits: Iterable[FactorGridMigration]) -> List[dict]:
    return [a.to_dict() for a in audits]


__all__ = ["FactorGridMigration", "migrate_factor_expression", "migrate_factor_dict", "migration_rows"]
