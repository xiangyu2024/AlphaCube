"""The three-axis type system.

Every field is a cube ``(S, D, M)`` = (stock, day, minute). Every operator is a
two-axis kernel from the vendored library, and what makes it unambiguous on
the cube is a declared **face** plus a declared **compute axis**:

====== ============================== ============ =====================================
face   kernel sees (rows, cols)       compute axis independent axes (= legal chunk axes)
====== ============================== ============ =====================================
dt/in     (D, M) at fixed s           M            S, D
dt/day    (D, M) -> (D, 1)            M (reduce)   S, D
dt/cross  (D, M) at fixed s           D            S, M
nd/cs     (D, S) with M collapsed     S            D          (whole panel, no chunking)
nd/ts     (D, S) with M collapsed     D            S          (whole panel, no chunking)
nt/cs     (M, S) at fixed d           S            D          (per-day kernel call)
nt/ts     (M, S) at fixed d           M            D          (per-day kernel call)
shared    elementwise                 none         all
====== ============================== ============ =====================================

NT operators are evaluated exactly as their definition says: the ND kernel
is called once per day on that day's ``(M, S)`` face, in D-mode. Batching
several days' columns into one call would be faster but is not the same
computation: two ND kernels (``ts_pearson``/``ts_spearman``) mix columns by
design, and every reduction's float32 order depends on tensor geometry, so
only the per-day call is both semantically exact and chunk-invariant.

The ND and NT rows share the same kernel functions. They are nevertheless two
distinct registry entries each (``ts_mean`` on the ND face, ``nt_ts_mean`` on
the NT face): feeding a ``(T, N)`` kernel stock x day means one thing and
stock x minute means another, and the only way to make a three-axis
expression unambiguous is to say which one in the operator's name.

Chunking follows from the table: an operator may be evaluated on a slice of
the cube along any axis it is independent of. ``cross_*``/``ts_*`` need the
full day axis, so their phase is chunked over S; ``nt_cs_*`` needs the full
stock axis, so its phase is chunked over D; M-axis operators are happy in
either. ND-face operators act on ``(D, S)`` panels, which are small enough
(full market, 10 years: 58 MB) never to be chunked at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

# axes
S, D, M = "S", "D", "M"

# execution modes of the engine
MODE_S = "S"        # cube chunked over stocks: tensors (C, D, M)
MODE_D = "D"        # cube chunked over days:   tensors (S, C, M)
MODE_P = "P"        # whole (D, S) panel in memory, minute axis already collapsed

FACES = ("dt", "nd", "nt", "shared")
FAMILIES = ("in", "day", "cross", "cs", "ts", "elementwise")
CATEGORIES = ("unary", "binary", "rolling", "pair_rolling")


@dataclass(frozen=True)
class OpSpec:
    """One operator of the unified registry.

    ``fn`` is the raw torch kernel with its face's two-axis calling
    convention; the engine is responsible for presenting the right 2-D view of
    a cube chunk (see engine/kernels.py). ``kernel_id`` records provenance
    (``nd_ts.ts_mean``) so the ND/NT twins are visibly the same code.
    """

    name: str
    face: str
    family: str
    category: str
    arity: int
    requires_window: bool
    fn: Callable
    kernel_id: str
    compute_axis: Optional[str]          # "M" | "D" | "S" | None
    reduces_m: bool = False              # day_*: (.., M) -> (.., 1)
    restores_m: bool = False             # in_expand: (.., 1) -> (.., M)
    lookahead: bool = False              # uses data after t: never in a search vocabulary
    boolean_output: bool = False         # strictly 0/1 output: never a factor root
    trivial_boolean: bool = False        # 0/1 output that some input collapses to one constant
    coarse_output: bool = False          # <= 3 distinct values: never a factor root
    windows: Tuple[int, ...] = field(default_factory=tuple)

    # ---- static typing helpers -------------------------------------------------
    @property
    def needs_full_m(self) -> bool:
        """Operand must carry genuine per-minute variation."""
        return (self.face == "dt" and self.family in ("in", "day")) or self.face == "nt"

    @property
    def needs_daily(self) -> bool:
        """Operand must be one value per stock-day (ND face)."""
        return self.face == "nd"

    @property
    def allowed_modes(self) -> Tuple[str, ...]:
        if self.face == "nd":
            return (MODE_P,)
        if self.face == "nt":
            return (MODE_D,)
        if self.compute_axis == M:
            return (MODE_S, MODE_D)
        if self.compute_axis == D:
            return (MODE_S,)
        if self.compute_axis == S:
            return (MODE_D,)
        return (MODE_S, MODE_D, MODE_P)     # elementwise

    @property
    def non_root(self) -> bool:
        return self.boolean_output or self.coarse_output


def compute_axis_for(face: str, family: str) -> Optional[str]:
    if family == "elementwise":
        return None
    if face == "dt":
        return {"in": M, "day": M, "cross": D}[family]
    if face == "nd":
        return {"cs": S, "ts": D}[family]
    if face == "nt":
        return {"cs": S, "ts": M}[family]
    raise ValueError(f"unknown face/family {face}/{family}")


__all__ = [
    "S", "D", "M", "MODE_S", "MODE_D", "MODE_P", "FACES", "FAMILIES", "CATEGORIES",
    "OpSpec", "compute_axis_for",
]
