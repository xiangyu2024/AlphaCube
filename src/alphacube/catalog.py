"""Build the unified operator registry from the vendored kernel library.

Adapted from the previous project's ``ops_real_backend`` (same smoke-test
validation, same parameter-baked variants) with one structural difference:
every ``mode == "nd"`` kernel is registered **twice**, once on the ND face
under its own name and once on the NT face under ``nt_<name>``. Same
function object, different declared face -- that duplication is what makes
``(stock x day)`` vs ``(stock x minute)`` explicit instead of implied by
whatever tensor happened to be passed in.

Also computed here, by probing rather than by hand:

- ``boolean_output`` / ``coarse_output``: feed every operator a random panel
  and count distinct finite output values (<= 2 -> boolean, <= 3 -> coarse).
- ``lookahead``: the three kernels documented as forward-looking
  (``ts_shift``, ``cross_shift``, ``in_local_mean``) plus the NT twin of
  ``ts_shift``; verified by a perturbation test in tests/test_catalog.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch

from . import operators as real_ops
from .axes import CATEGORIES, OpSpec, compute_axis_for

# ---- default window pools per face ------------------------------------------
# DT rolling windows run along minutes (in_*) or days (cross_*); 241 is the
# full A-share trading day on the V2 grid (09:30 auction + 240 bars) so that
# in_expand can restore a collapsed operand to the panel's real width.
WINDOWS = {
    "dt": (3, 5, 10, 15, 30, 60, 241),
    "nd": (1, 5, 10, 20, 30, 40, 50),
    "nt": (3, 5, 10, 15, 30, 60),
}
CONSTANTS = (-5.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 5.0)

LOOKAHEAD_KERNELS = frozenset({"ts_shift", "cross_shift", "in_local_mean"})
# Kernels that correlate every stock against every other, so their time is
# O(windows x d x N^2) -- 5219^2 = 27M pairs per window on the production cube.
# That is the factor's definition, not an implementation detail: the pairwise
# count is taken per pair (stocks are suspended and listed on different days), so
# the bilinear shortcut that would collapse agg="mean" to O(N) does not apply.
#
# What decides whether it is affordable is the FACE it runs on, and the two
# differ by two orders of magnitude:
#
#   ts_pearson     ND (D, S)  evaluated once over the whole panel      1x
#   nt_ts_pearson  NT (M, S)  evaluated once PER DAY, 1942 of them  ~241x
#
# So the ND ones stay in the search vocabulary and the NT twins do not. The whole
# set is still what calculate.py defers, which is a different (memory) question.
RESOURCE_RISK_KERNELS = frozenset({"ts_pearson", "nt_ts_pearson", "nt_ts_spearman", "in_multireg"})
# Operators whose operand positions are not all interchangeable: the search has no
# way to say "a tensor belongs here but a literal belongs there", so a binary
# ``pow`` invites pow(constant, tensor) and its float32 underflow (see _VARIANT_DEFS).
# The baked unary variants cover every exponent worth searching. Still reachable by
# name for research and the legacy corpus -- only the SEARCH vocabulary drops it.
POSITIONALLY_UNSAFE_KERNELS = frozenset({"pow"})
# Of those, the ones a full-market search must not breed: same kernels, NT face.
SEARCH_FORBIDDEN_KERNELS = frozenset({"nt_ts_pearson", "nt_ts_spearman", "in_multireg"})
# Kernels deliberately left out of the registry (recorded in Registry.skipped):
EXCLUDED_KERNELS = {
    "rank": "duplicate of cs_rank (same percentile rank across stocks); keep one name",
    "indneutralize": "needs an industry-code tensor as its second operand; no such field in the cube",
}
RESTORES_M_NAMES = frozenset({"in_expand"})


# ---- classification + smoke test (from ops_real_backend) --------------------

def classify(meta: dict) -> Optional[str]:
    arity, window, pair = meta["arity"], meta["requires_window"], meta["pair"]
    if arity == 1 and not window:
        return "unary"
    if arity == 2 and not window:
        return "binary"
    if arity == 1 and window:
        return "rolling"
    if arity == 2 and window and pair:
        return "pair_rolling"
    return None


def _sample(rows: int = 7, cols: int = 5, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand((rows, cols), generator=g) + 0.5


def _smoke_call(fn: Callable, category: str):
    x = _sample(seed=1)
    if category == "unary":
        return fn(x), x
    if category == "binary":
        return fn(x, _sample(seed=2)), x
    if category == "rolling":
        return fn(x, 3), x
    return fn(x, _sample(seed=2), 3), x


def _validate(name: str, meta: dict) -> Tuple[Optional[str], Optional[str]]:
    if name in RESTORES_M_NAMES:
        x = _sample(rows=7, cols=1, seed=1)
        try:
            out = meta["fn"](x, 3)
        except Exception as exc:  # noqa: BLE001
            return None, f"restores_m smoke test raised {type(exc).__name__}: {exc}"
        if not isinstance(out, torch.Tensor) or tuple(out.shape) != (7, 3):
            return None, "restores_m smoke test did not return a (7, 3) tensor"
        return "rolling", None
    category = classify(meta)
    if category is None:
        return None, (
            f"(arity={meta['arity']}, requires_window={meta['requires_window']}, "
            f"pair={meta['pair']}) is not one of {CATEGORIES}"
        )
    try:
        out, x = _smoke_call(meta["fn"], category)
    except Exception as exc:  # noqa: BLE001
        return None, f"raised {type(exc).__name__} when smoke-tested as {category}: {exc}"
    if not isinstance(out, torch.Tensor):
        return None, f"returned {type(out).__name__}, not a single tensor"
    if meta["category"] == "day":
        if tuple(out.shape[:-1]) != tuple(x.shape[:-1]) or out.shape[-1] != 1:
            return None, f"day kernel returned shape {tuple(out.shape)} for input {tuple(x.shape)}"
    elif tuple(out.shape) != tuple(x.shape):
        return None, f"output shape {tuple(out.shape)} != input shape {tuple(x.shape)}"
    return category, None


# ---- parameter-baked variants (unchanged from the previous project) ----------
_VARIANT_DEFS: Tuple[Tuple[str, str, Callable], ...] = (
    # Exponents are BAKED, never searched as an operand. A binary pow lets the
    # search put a tensor in the exponent -- pow(0.5, volume) -- and in float32
    # 0.5**x underflows to exactly 0 for x > 126, so a daily volume of 1e5 gives a
    # panel that is identically zero. It is finite everywhere (coverage looks
    # healthy) but has no cross-sectional variance, so cs_pearson returns NaN on
    # almost every day and the |IC| is computed from the handful that survive. That
    # is how such a tree reached rank 1 of a full-market generation with |IC| 0.0741
    # and a Sharpe of 0.53. The exponent is not a thing a factor searches over.
    ("pow_05", "pow_", lambda f: (lambda x: f(x, 0.5))),
    ("pow_1_5", "pow_", lambda f: (lambda x: f(x, 1.5))),
    ("pow_2", "pow_", lambda f: (lambda x: f(x, 2.0))),
    ("pow_2_5", "pow_", lambda f: (lambda x: f(x, 2.5))),
    ("pow_3", "pow_", lambda f: (lambda x: f(x, 3.0))),
    ("is_gt0", "is_gt", lambda f: (lambda x: f(x, 0.0))),
    ("is_lt0", "is_lt", lambda f: (lambda x: f(x, 0.0))),
    ("cs_quantile_q25", "cs_quantile", lambda f: (lambda x: f(x, 0.25))),
    ("cs_quantile_q50", "cs_quantile", lambda f: (lambda x: f(x, 0.50))),
    ("cs_quantile_q75", "cs_quantile", lambda f: (lambda x: f(x, 0.75))),
    ("ts_quantile_q25", "ts_quantile", lambda f: (lambda x, d: f(x, 0.25, d))),
    ("ts_quantile_q50", "ts_quantile", lambda f: (lambda x, d: f(x, 0.50, d))),
    ("ts_quantile_q75", "ts_quantile", lambda f: (lambda x, d: f(x, 0.75, d))),
    ("cross_quantile_q25", "cross_quantile", lambda f: (lambda x, d: f(x, 0.25, d))),
    ("cross_quantile_q50", "cross_quantile", lambda f: (lambda x, d: f(x, 0.50, d))),
    ("cross_quantile_q75", "cross_quantile", lambda f: (lambda x, d: f(x, 0.75, d))),
    ("in_quantile_q25", "in_quantile_rolling", lambda f: (lambda x, d: f(x, 0.25, d))),
    ("in_quantile_q50", "in_quantile_rolling", lambda f: (lambda x, d: f(x, 0.50, d))),
    ("in_quantile_q75", "in_quantile_rolling", lambda f: (lambda x, d: f(x, 0.75, d))),
    ("ts_autocorr_s1", "ts_autocorr", lambda f: (lambda x, d: f(x, d, 1))),
    ("cross_autocorr_s1", "cross_autocorr", lambda f: (lambda x, d: f(x, d, 1))),
    ("in_autocorr_s1", "in_autocorr", lambda f: (lambda x, d: f(x, d, 1))),
    ("ts_outlier_33", "ts_outlier", lambda f: (lambda x, d: f(x, d, 3.0, 3.0))),
    ("cross_outlier_33", "cross_outlier", lambda f: (lambda x, d: f(x, d, 3.0, 3.0))),
    ("day_autocorr_n1", "day_autocorr", lambda f: (lambda x: f(x, 1))),
    ("day_quantile_q25", "day_quantile", lambda f: (lambda x: f(x, 0.25))),
    ("day_quantile_q75", "day_quantile", lambda f: (lambda x: f(x, 0.75))),
    ("in_top_quantile_flag_q10", "in_top_quantile_flag", lambda f: (lambda x: f(x, 0.1))),
    ("in_topk_masked_p5", "in_topk_masked", lambda f: (lambda x, y: f(x, y, 5))),
    ("in_topk_masked_exact_k5", "in_topk_masked_exact", lambda f: (lambda x, y: f(x, y, 5))),
    ("sliding_top_mean_masked_p5", "sliding_top_mean_masked", lambda f: (lambda x, y, d: f(x, y, d, 5))),
    ("sliding_top_sum_masked_p5", "sliding_top_sum_masked", lambda f: (lambda x, y, d: f(x, y, d, 5))),
    ("sliding_bottom_mean_masked_p5", "sliding_bottom_mean_masked", lambda f: (lambda x, y, d: f(x, y, d, 5))),
    ("sliding_bottom_sum_masked_p5", "sliding_bottom_sum_masked", lambda f: (lambda x, y, d: f(x, y, d, 5))),
)


# ---- output-cardinality probe ------------------------------------------------

def _probe_constant_capable(fn: Callable, category: str, restores: bool) -> bool:
    """Can some plausible input make this operator's output a single constant?

    That is the difference between a boolean worth searching and one that is not.
    A predicate against a FIXED threshold collapses whenever the data sits on one
    side of it -- ``is_gt0`` on an all-positive panel is all ones, ``eq`` on two
    unrelated panels is all zeros, ``logical_and`` on positive data is all ones --
    and a constant panel carries no cross-sectional information, yet scores a
    spurious IC once arithmetic launders it into a legal root. A predicate against
    the data's OWN distribution cannot collapse: ``cs_quantile_q50`` splits its
    input in half whatever the input is, ``in_top_quantile_flag_q10`` always flags
    a tenth. So the question is asked empirically, over input families chosen to
    put the data wholly on one side of any fixed threshold."""
    g = torch.Generator().manual_seed(11)
    base = torch.rand((40, 9), generator=g) + 0.5           # strictly positive
    other = torch.rand((40, 9), generator=g) + 0.5
    families = [(base, other), (-base, -other), (base, base),
                (torch.randn((40, 9), generator=g), torch.randn((40, 9), generator=g))]
    for x, y in families:
        if restores:
            x = x[:, :1]
        try:
            with torch.no_grad():
                if category == "unary":
                    out = fn(x)
                elif category == "binary":
                    out = fn(x, y)
                elif category == "rolling":
                    out = fn(x, 5)
                else:
                    out = fn(x, y, 5)
            flat = out.float().reshape(-1)
            flat = flat[torch.isfinite(flat)]
        except Exception:                                    # noqa: BLE001 - a family it cannot take
            continue
        if flat.numel() == 0 or int(torch.unique(flat).numel()) <= 1:
            return True
    return False


def _probe_per_element_python(fn: Callable, category: str, restores: bool) -> bool:
    """Does the kernel do Python-level work per element of its input?

    Counted, not timed: the Python lines executed inside ``alphacube.operators``
    while the kernel runs on a 16 x 48 input, then on 4x the rows and on 4x the
    columns. A vectorised kernel runs the same lines whatever the size; one that
    loops in Python runs ~4x more. Deterministic and machine-independent, so the
    search vocabulary cannot drift with the host's speed.

    Why it matters: a DT kernel sees ~8M stock-day-minute cells per chunk on the
    production cube. ``in_mask_nearby_ones`` -- written for 1/NaN masks -- loops
    over every finite cell with two ``.item()`` syncs, so on a float panel one
    chunk took ~7 minutes instead of ~1 second; one tree using it froze gpu8's
    generation 9 at 2% after 89 minutes, ETA 77 hours. Such kernels stay
    callable by name (research, the legacy corpus) but leave the search -- except
    on the ND face, whose kernels run once per panel (see ``_make_spec``)."""
    import sys

    def lines(rows: int, cols: int) -> int:
        g = torch.Generator().manual_seed(5)
        x = torch.rand((rows, cols), generator=g) + 0.5
        y = torch.rand((rows, cols), generator=g) + 0.5
        if restores:
            x = x[:, :1]
        n = [0]

        def tracer(frame, event, arg):
            if "alphacube" in frame.f_code.co_filename and "operators" in frame.f_code.co_filename:
                if event == "line":
                    n[0] += 1
                return tracer
            return tracer if event == "call" else None

        previous = sys.gettrace()
        sys.settrace(tracer)
        try:
            with torch.no_grad():
                if category == "unary":
                    fn(x)
                elif category == "binary":
                    fn(x, y)
                elif category == "rolling":
                    fn(x, 5)
                else:
                    fn(x, y, 5)
        except Exception:                                    # noqa: BLE001 - counted either way
            pass
        finally:
            sys.settrace(previous)
        return n[0]

    small = lines(16, 48)
    return lines(64, 48) > 2 * small + 20 or lines(16, 192) > 2 * small + 20


def _probe_distinct_values(fn: Callable, category: str, reduces: bool, restores: bool) -> int:
    """Distinct finite output values on a random normal panel (rows 40, cols 9)."""
    g = torch.Generator().manual_seed(7)
    x = torch.randn((40, 9), generator=g)
    y = torch.randn((40, 9), generator=g)
    if restores:
        x = x[:, :1]
    with torch.no_grad():
        if category == "unary":
            out = fn(x)
        elif category == "binary":
            out = fn(x, y)
        elif category == "rolling":
            out = fn(x, 5)
        else:
            out = fn(x, y, 5)
    out = out.float().reshape(-1)
    out = out[torch.isfinite(out)]
    return int(torch.unique(out).numel())


# ---- manual binary ops (Expression.__pow__ etc.) -----------------------------

def _manual_pow(x, y):
    return torch.pow(x, y)


def _manual_greater(x, y):
    return (x > y).float()


def _manual_less(x, y):
    return (x < y).float()


@dataclass(frozen=True)
class SkipRecord:
    name: str
    mode: str
    family: str
    reason: str


class Registry:
    """Name -> OpSpec, plus the field list and constant pool."""

    def __init__(self, fields: Tuple[str, ...], constants: Tuple[float, ...] = CONSTANTS):
        self._ops: Dict[str, OpSpec] = {}
        self.fields: Tuple[str, ...] = tuple(fields)
        self.constants: Tuple[float, ...] = tuple(constants)
        self.skipped: List[SkipRecord] = []

    def register(self, spec: OpSpec) -> None:
        if spec.name in self._ops:
            raise ValueError(f"duplicate operator name {spec.name!r}")
        self._ops[spec.name] = spec

    def get(self, name: str) -> OpSpec:
        try:
            return self._ops[name]
        except KeyError:
            raise KeyError(f"unknown operator {name!r}") from None

    def __contains__(self, name: str) -> bool:
        return name in self._ops

    def __len__(self) -> int:
        return len(self._ops)

    def list_all(self) -> List[OpSpec]:
        return list(self._ops.values())

    def by_face(self, face: str) -> List[OpSpec]:
        return [s for s in self._ops.values() if s.face == face]

    def searchable(self, include_boolean: bool = True,
                   include_resource_risk: bool = True) -> List[OpSpec]:
        """Operators a search vocabulary may use: everything not forward-looking.

        ``include_boolean=False`` also drops the operators flagged
        ``trivial_boolean`` -- 0/1 outputs that some input collapses to a single
        constant (``eq``, ``is_gt0``, ``is_lt0``, ``logical_and/or``, ...). They
        are already barred from being a factor's root, but that rule is checked
        one node at a time, so a search launders one through any arithmetic
        operator: ``pow_05(eq(a, b))`` is a legal root made of a near-constant
        indicator, and it took the top of the population on the production store
        with a spurious |IC| of 0.58 while the pool recorded its IC as NaN.

        Booleans that split the data by its own distribution (``cs_quantile_q50``,
        ``in_top_quantile_flag_q10``, the outlier flags) are NOT dropped: they
        cannot collapse, so they carry real cross-sectional information and are
        worth searching."""
        return [s for s in self._ops.values() if not s.lookahead
                and (include_boolean or not s.trivial_boolean)
                and (include_resource_risk or not s.resource_risk)]


CUBE_FIELDS = ("Open", "High", "Low", "Close", "Volume", "Vwap", "Ret", "Amt")


def _make_spec(name: str, face: str, family: str, category: str, meta: dict,
               fn: Callable, kernel_id: str) -> OpSpec:
    reduces = family == "day"
    restores = name in RESTORES_M_NAMES
    try:
        distinct = _probe_distinct_values(fn, category, reduces, restores)
    except Exception:  # noqa: BLE001 - a probe failure just means "unknown", not boolean
        distinct = 10**9
    return OpSpec(
        name=name, face=face, family=family, category=category,
        arity=meta["arity"], requires_window=(True if restores else meta["requires_window"]),
        fn=fn, kernel_id=kernel_id,
        compute_axis=compute_axis_for(face, family),
        reduces_m=reduces, restores_m=restores,
        lookahead=(kernel_id.split(".")[-1] in LOOKAHEAD_KERNELS),
        boolean_output=(distinct <= 2), coarse_output=(2 < distinct <= 3),
        trivial_boolean=(distinct <= 2 and _probe_constant_capable(fn, category, restores)),
        # ND kernels run once over the whole (D, S) panel, so a Python loop over its
        # 5219 stocks costs seconds (ts_spearman, kept on purpose -- see
        # test_only_the_nt_face_pair_kernels_leave_the_action_set). Every other face
        # runs per chunk of millions of stock-day-minutes or once per day.
        resource_risk=(name in SEARCH_FORBIDDEN_KERNELS or name in POSITIONALLY_UNSAFE_KERNELS
                       or (face != "nd" and _probe_per_element_python(fn, category, restores))),
        windows=WINDOWS.get(face, ()),
    )


def build_registry(fields: Tuple[str, ...] = CUBE_FIELDS) -> Registry:
    reg = Registry(fields)
    accepted: List[Tuple[str, dict, Callable, str]] = []   # (name, meta, fn, kernel_id)
    for name, meta in real_ops.OPERATOR_REGISTRY.items():
        if name in EXCLUDED_KERNELS:
            reg.skipped.append(SkipRecord(name, meta["mode"], meta["category"], EXCLUDED_KERNELS[name]))
            continue
        category, reason = _validate(name, meta)
        if category is None:
            reg.skipped.append(SkipRecord(name, meta["mode"], meta["category"], reason))
            continue
        accepted.append((name, {**meta, "grammar_category": category}, meta["fn"],
                         f"{meta['fn'].__module__.split('.')[-1]}.{name}"))
    for vname, base_name, factory in _VARIANT_DEFS:
        base = real_ops.OPERATOR_REGISTRY.get(base_name)
        if base is None:
            continue
        fn = factory(base["fn"])
        category, reason = _validate(vname, {**base, "fn": fn})
        if category is None:
            reg.skipped.append(SkipRecord(vname, base["mode"], base["category"], reason))
            continue
        accepted.append((vname, {**base, "grammar_category": category}, fn,
                         f"{base['fn'].__module__.split('.')[-1]}.{base_name}"))

    for name, meta, fn, kernel_id in accepted:
        mode, family, category = meta["mode"], meta["category"], meta["grammar_category"]
        if mode == "shared":
            reg.register(_make_spec(name, "shared", "elementwise", category, meta, fn, kernel_id))
        elif mode == "dt":
            reg.register(_make_spec(name, "dt", family, category, meta, fn, kernel_id))
        elif mode == "nd":
            # Twice: ND face under the kernel's own name, NT face under nt_<name>.
            reg.register(_make_spec(name, "nd", family, category, meta, fn, kernel_id))
            reg.register(_make_spec(f"nt_{name}", "nt", family, category, meta, fn, kernel_id))
    # manual binaries behind Python operator syntax
    for name, fn in (("pow", _manual_pow), ("greater", _manual_greater), ("less", _manual_less)):
        meta = {"arity": 2, "requires_window": False}
        reg.register(_make_spec(name, "shared", "elementwise", "binary", meta, fn, f"manual.{name}"))
    return reg


_DEFAULT: Optional[Registry] = None


def default_registry() -> Registry:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = build_registry()
    return _DEFAULT


__all__ = ["Registry", "SkipRecord", "build_registry", "default_registry", "CUBE_FIELDS",
           "WINDOWS", "CONSTANTS", "LOOKAHEAD_KERNELS", "EXCLUDED_KERNELS"]
