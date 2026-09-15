"""The FULL callable surface of the kernel library, for research / legacy
expressions -- a strict superset of the four-category search registry.

The search registry answers "what may a miner generate?" (four grammar
shapes, no literals). A hand-written factor calls three-tensor functions
(``select_where``), functions with literal parameters (``in_slice``,
``day_pdf_tail_sum_m(x, f, 50, False, 'rank')``), tensor-list functions
(``in_multireg``), and undecorated helpers of the legacy framework
(``create_like``, ``delete_open_close``). Every one of them is a working
kernel; ``GeneralCall`` is the node that carries such a call and this module
classifies each kernel onto a face / compute axis so the scheduler and the
engine can place it like any registry operator.

Entries come from (1) every ``@op``-decorated function (its mode/category is
authoritative) and (2) a scan of public module-level functions in the
library's modules, classified by module and name prefix. Names defined with
different bodies in two scanned modules are *ambiguous* and never resolved
silently. ND-face entries also resolve under an ``nt_`` prefix as their NT
twin, exactly like the search registry.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import operators as real_ops
from .axes import OpSpec, compute_axis_for
from .catalog import LOOKAHEAD_KERNELS, RESTORES_M_NAMES

# Legacy spellings -> library names (all verified renames, see operators/__init__.py).
OPERATOR_NAME_ALIASES: Dict[str, str] = {
    "cross_zscore_rolling": "in_zscore_rolling",
    "abs": "abs_",
    "_max": "pairwise_max",
    "_min": "pairwise_min",
    "_pow": "signed_pow",
    "sum": "add",
    "rank": "cs_rank",
}
# Names that exist in the catalog but under the NEW API; legacy call sites go elsewhere.
GENERAL_NAME_OVERRIDES: Dict[str, str] = {"in_multireg": "in_multireg_legacy"}
# Overloaded legacy names disambiguated by call-site arity.
OPERATOR_ARITY_ALIASES: Dict[str, Dict[int, str]] = {"in_stddev": {1: "in_stddev_full", 2: "in_stddev_rolling"}}
# Tensor-argument positions that are legitimately day-collapsed indices.
COLLAPSED_INDEX_OPERANDS: Dict[str, frozenset] = {
    "day_locate": frozenset({1}),
    "in_slice_from_start": frozenset({1}),
    "in_slice_to_end": frozenset({1}),
    "day_tide_speed": frozenset({1, 2, 3, 4}),
    "day_tide_change": frozenset({1, 2, 3, 4}),
}

_MODULE_SOURCES: Tuple[Tuple[str, str, str], ...] = (
    ("common", "shared", "elementwise"),
    ("nd_cs", "nd", "cs"),
    ("nd_ts", "nd", "ts"),
    ("dt_in", "dt", "in"),
    ("dt_cross", "dt", "cross"),
    ("dt_day", "dt", "day"),
    ("legacy_specialized_nd", "nd", "ts"),
    ("legacy_specialized_dt", "dt", "in"),
    ("dt_spectral", "dt", "in"),
    ("legacy_framework_compat", "shared", "elementwise"),
)
_PREFIX_CLASSIFIED = frozenset({"legacy_specialized_nd", "legacy_specialized_dt", "dt_spectral",
                                "legacy_framework_compat"})
_CATEGORY_PREFIXES = (("day_", "day"), ("cross_", "cross"), ("in_", "in"), ("cs_", "cs"), ("ts_", "ts"))
_MODE_BY_CATEGORY = {"in": "dt", "cross": "dt", "day": "dt", "cs": "nd", "ts": "nd"}


def _general_spec(name: str, face: str, family: str, fn, kernel_id: str) -> OpSpec:
    return OpSpec(
        name=name, face=face, family=family, category="general", arity=-1, requires_window=False,
        fn=fn, kernel_id=kernel_id, compute_axis=compute_axis_for(face, family),
        reduces_m=(family == "day"), restores_m=(name in RESTORES_M_NAMES),
        lookahead=(kernel_id.split(".")[-1] in LOOKAHEAD_KERNELS),
    )


@dataclass
class FullCatalog:
    entries: Dict[str, OpSpec] = field(default_factory=dict)
    ambiguous: Dict[str, List[OpSpec]] = field(default_factory=dict)

    def get(self, name: str) -> Optional[OpSpec]:
        """Resolve ``name`` (aliases applied, ``nt_`` twins derived); None if unknown/ambiguous."""
        # overrides first: the legacy spelling may collide with the library's NEW api of the same name
        for cand in (GENERAL_NAME_OVERRIDES.get(name), name, OPERATOR_NAME_ALIASES.get(name)):
            if cand and cand in self.entries:
                return self.entries[cand]
        if name.startswith("nt_"):
            base = self.get(name[3:])
            if base is not None and base.face == "nd":
                return OpSpec(**{**base.__dict__, "name": name, "face": "nt",
                                 "compute_axis": compute_axis_for("nt", base.family)})
        return None

    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None


def _category_for(module_key: str, default: str, fn_name: str) -> str:
    if module_key not in _PREFIX_CLASSIFIED:
        return default
    for prefix, cat in _CATEGORY_PREFIXES:
        if fn_name.startswith(prefix):
            return cat
    return default


def build_full_catalog() -> FullCatalog:
    cat = FullCatalog()
    for name, meta in real_ops.OPERATOR_REGISTRY.items():
        mode, family = meta["mode"], meta["category"]
        face = "shared" if mode == "shared" else mode
        fam = "elementwise" if mode == "shared" else family
        kernel_id = f"{meta['fn'].__module__.split('.')[-1]}.{name}"
        cat.entries[name] = _general_spec(name, face, fam, meta["fn"], kernel_id)
    for module_key, mode, default_family in _MODULE_SOURCES:
        try:
            module = importlib.import_module(f"{real_ops.__name__}.{module_key}")
        except ImportError:
            continue
        explicit = getattr(module, "EXPLICIT_CATEGORIES", {})
        for fn_name, obj in vars(module).items():
            if fn_name.startswith("_") or not inspect.isfunction(obj) or obj.__module__ != module.__name__:
                continue
            family = explicit.get(fn_name, _category_for(module_key, default_family, fn_name))
            face = mode
            if module_key == "legacy_framework_compat":
                face = _MODE_BY_CATEGORY.get(family, "shared")
            if face == "shared":
                family = "elementwise"
            entry = _general_spec(fn_name, face, family, obj, f"{module_key}.{fn_name}")
            if fn_name in cat.ambiguous:
                cat.ambiguous[fn_name].append(entry)
                continue
            existing = cat.entries.get(fn_name)
            if existing is None:
                cat.entries[fn_name] = entry
            elif existing.kernel_id.split(".")[-1] == fn_name and existing.fn is not obj and \
                    existing.kernel_id.split(".")[0] != module_key and existing.category == "general" and \
                    fn_name not in real_ops.OPERATOR_REGISTRY:
                cat.ambiguous[fn_name] = [existing, entry]
                del cat.entries[fn_name]
    return cat


_FULL: Optional[FullCatalog] = None


def default_full_catalog() -> FullCatalog:
    global _FULL
    if _FULL is None:
        _FULL = build_full_catalog()
    return _FULL


__all__ = ["FullCatalog", "build_full_catalog", "default_full_catalog", "OPERATOR_NAME_ALIASES",
           "GENERAL_NAME_OVERRIDES", "OPERATOR_ARITY_ALIASES", "COLLAPSED_INDEX_OPERANDS"]
