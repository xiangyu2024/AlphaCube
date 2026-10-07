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

import inspect
import math
import random
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from ..axes import OpSpec
from ..catalog import Registry
from ..expression import (Constant, DailyFeature, Expression, Feature, ShapeError, is_valid_root,
                          make_node, replace)

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


def one_layer_keys(expr: Expression) -> set:
    """Keys of ``expr``'s depth-1 subtrees -- an operator whose operands are all
    leaves, e.g. ``day_ar1_coefficient(Volume)`` or ``day_cov(Volume, Low)``.

    This is the unit a search actually latches onto. A run that finds one good
    primitive will spend the rest of its budget wrapping it: the same
    ``day_ar1_coefficient(Volume)`` appeared in 179 of 364 archived expressions
    under sixteen different jackets, all scoring within noise of each other,
    because the wrappers were monotone and barely moved the ranks. Counting these
    keys is how a later block knows what the previous one exhausted.
    """
    out = set()
    for n in expr.nodes():
        if n.spec is not None and all(getattr(x, "spec", None) is None for x in n.operands):
            out.add(n.key())
    return out


class GenerationExhausted(ValueError):
    """``generate()`` found no tree, even in its minimal fallback, that survives the
    ban set -- a configuration error, not a transient miss -- or the caller opted
    into ``on_exhausted="raise"``. Hook exhaustion alone (the ban set admits
    something, but the accept hook refuses every try) does NOT raise by default:
    it returns a ban-respecting tree and increments ``unaccepted_fallbacks``
    instead, because a tight guidance hook may have no minimal tree it would ever
    accept, and raising there would crash the very steer arm this is for."""


GENERATE_FALLBACK_TRIES = 64   # tries generate()'s fallback gives a hook/ban set before giving up


def _check_weight(name: str, w) -> float:
    """A construction-time ``op_weight`` value must be a finite float >= 0 -- the
    same bound ``_pick_weighted`` clamps to at use time, checked eagerly here so a
    mistake (a negative weight, a NaN from some upstream computation) fails at
    ``__init__`` instead of silently thinning a face for an entire run."""
    try:
        wf = float(w)
    except (TypeError, ValueError):
        raise ValueError(f"op_weight for {name!r} must be a finite float >= 0, got {w!r}") from None
    if not math.isfinite(wf) or wf < 0.0:
        raise ValueError(f"op_weight for {name!r} must be a finite float >= 0, got {w!r}")
    return wf


def _check_accept_hook(accept) -> None:
    """``accept`` is always invoked as ``self.accept_hook(tree)`` from ``acceptable()``,
    so it must be callable with exactly one positional parameter."""
    if not callable(accept):
        raise ValueError(f"accept must be callable, got {accept!r}")
    try:
        inspect.signature(accept).bind(object())
    except TypeError as exc:
        raise ValueError(f"accept must take exactly one positional parameter: {exc}") from exc
    except ValueError:
        pass    # signature unavailable (some builtins) -- nothing more to check here


def _resolve_op_weight(op_weight, registry: Registry):
    """``op_weight`` is always called as ``self.op_weight(spec) -> float`` (see
    ``_pick_weighted``); this accepts that callable directly, or the more common,
    more ergonomic shape -- a ``{operator name: weight}`` mapping -- and wraps the
    latter into exactly such a callable. A mapping's keys are checked against the registry
    up front (a typo'd name would otherwise sit silently inert, never matched
    against any real ``OpSpec``) and an operator the mapping does not mention
    defaults to weight 1.0, the uniform baseline."""
    if op_weight is None or callable(op_weight):
        return op_weight
    try:
        table = dict(op_weight)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"op_weight must be callable or a {{name: weight}} mapping: {exc}") from exc
    for name, w in table.items():
        if name not in registry:
            raise ValueError(f"op_weight names an unknown operator: {name!r}")
        _check_weight(name, w)
    return lambda s, _table=table: _table.get(s.name, 1.0)


class TypedTreeGenerator:
    def __init__(self, registry: Registry, fields: Optional[Sequence[str]] = None,
                 constants: Optional[Sequence[float]] = None, n_minutes: int = 241,
                 min_depth: int = 2, max_depth: int = 4, feature_leaf_prob: float = 0.8,
                 grow_leaf_prob: float = 0.3, max_retries: int = 16,
                 include_lookahead: bool = False, allow_boolean_ops: bool = False,
                 allow_resource_risk: bool = False, banned_substructures: frozenset = frozenset(),
                 rng: Optional[random.Random] = None, seed: int = 0,
                 op_weight: Optional[Union[Mapping[str, float], Callable[[OpSpec], float]]] = None,
                 accept: Optional[Callable[[Expression], bool]] = None,
                 daily_fields: Sequence[str] = (), daily_leaf_prob: float = 0.0,
                 leaf_slots: Optional[Mapping[str, Mapping[int, Sequence[str]]]] = None,
                 on_exhausted: str = "fallback", mutation_checks_bans: bool = True) -> None:
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
        # A second, independent stream for accept hooks: Partura's guidance hook used
        # to draw from gen._rng, which made its own trajectory depend on how many
        # candidates were thinned -- not reproducible from the seed. A hook MUST draw
        # from this one instead, never from self._rng.
        self.hook_rng = random.Random((seed * 2654435761) & 0xFFFFFFFF)
        specs = (registry.list_all() if include_lookahead
                 else registry.searchable(include_boolean=allow_boolean_ops,
                                          include_resource_risk=allow_resource_risk))
        self.allow_boolean_ops = bool(allow_boolean_ops)
        # Depth-1 subtrees this generator refuses to build (see one_layer_keys).
        # Checked at every acceptance point -- initial trees, mutations, crossovers
        # -- so a banned primitive cannot re-enter through breeding either.
        self.banned_substructures = frozenset(banned_substructures)
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

        # -- steering hooks (WP5): every default below selects today's code path
        # verbatim, so a caller that sets none of these reproduces the random
        # stream and the results bit-identically -- see docs/alphacube_steering_
        # design.md, "Backward compatibility" items 11-14.
        self.op_weight = _resolve_op_weight(op_weight, registry)
        self.accept_hook = accept
        if accept is not None:
            _check_accept_hook(accept)
        self.daily_fields = list(daily_fields)
        self.daily_leaf_prob = float(daily_leaf_prob)
        self.leaf_slots = leaf_slots
        self.on_exhausted = on_exhausted
        self.mutation_checks_bans = bool(mutation_checks_bans)
        self.unaccepted_fallbacks = 0
        if self.op_weight is not None:
            weights: Dict[str, float] = {}
            for s in self.specs:
                try:
                    w = self.op_weight(s)
                except Exception as exc:  # noqa: BLE001 - surfaced as a clear construction-time error
                    raise ValueError(f"op_weight raised for operator {s.name!r}: {exc}") from exc
                weights[s.name] = _check_weight(s.name, w)
            pool = self._root_ops_of("reduce")
            if pool and not any(weights[s.name] > 0.0 for s in pool):
                raise ValueError(
                    f"op_weight assigns weight 0 to every one of the {len(pool)} root-reducer "
                    "operators generate()'s fallback needs; at least one must stay > 0")

    # -- public --------------------------------------------------------------------
    def _ban_ok(self, tree: Expression) -> bool:
        return not (self.banned_substructures and (self.banned_substructures & one_layer_keys(tree)))

    def acceptable(self, tree: Expression) -> bool:
        """A legal root, carrying no banned primitive, that the accept hook (if any)
        is willing to let through. Same two checks and the same short-circuit as
        before the hook existed -- ``one_layer_keys`` is still computed only when a
        ban set exists -- so this one acceptance point now covers fresh trees,
        mutations, crossovers, and generate()'s own fallback alike."""
        if not is_valid_root(tree):
            return False
        if not self._ban_ok(tree):
            return False
        return self.accept_hook is None or bool(self.accept_hook(tree))

    def ban(self, keys) -> None:
        self.banned_substructures = self.banned_substructures | frozenset(keys)

    def generate(self, depth: Optional[int] = None) -> Expression:
        d = depth if depth is not None else self._rng.randint(self.min_depth, self.max_depth)
        for _ in range(self.max_retries * 4):
            try:
                tree = self._gen("daily", d, full=self._rng.random() < 0.5, root=True)
            except ShapeError:
                continue
            if self.acceptable(tree):
                return tree
        # The first loop's retries ran out. Before WP5 this returned the raw minimal
        # collapse unconditionally -- bypassing acceptable() entirely, so an
        # always-False accept hook or an exhausting ban set got silently overridden.
        # With no hook and no ban, tries==1 below and the first candidate is always
        # acceptable (is_valid_root over a day_*(Feature) tree already satisfies it),
        # so the draws and the result stay byte-identical to before this existed.
        pool = self._root_ops_of("reduce")
        tries = 1 if (self.accept_hook is None and not self.banned_substructures) else GENERATE_FALLBACK_TRIES
        relaxed: Optional[Expression] = None
        for _ in range(tries):
            tree = self._collapse_leaf(self._pick(pool))
            if self.acceptable(tree):
                return tree
            if relaxed is None and self._ban_ok(tree) and is_valid_root(tree):
                relaxed = tree
        if relaxed is None or self.on_exhausted == "raise":
            raise GenerationExhausted(
                f"no minimal day_* tree over {len(self.fields)} fields survives the ban set "
                f"({len(self.banned_substructures)} keys)"
                f"{' and the accept hook' if self.accept_hook is not None else ''}")
        self.unaccepted_fallbacks += 1
        return relaxed

    def generate_population(self, n: int, unique: bool = True, tries_per_slot: int = 6) -> List[Expression]:
        """Ramped half-and-half with de-duplication.

        The old version walked the depth ladder two slots at a time and kept
        whatever came back, so a population could carry the same tree several
        times over and start the search narrower than its size suggests.
        Retrying a slot until its key is new costs nothing and buys real
        diversity; after ``tries_per_slot`` misses it accepts a duplicate rather
        than loop forever on a small grammar."""
        depths = list(range(self.min_depth, self.max_depth + 1))
        out: List[Expression] = []
        seen: set = set()
        for i in range(n):
            d = depths[i % len(depths)]
            best = None
            for _ in range(tries_per_slot if unique else 1):
                cand = self.generate(d)
                best = best or cand
                if not unique or cand.key() not in seen:
                    best = cand
                    break
            seen.add(best.key())
            out.append(best)
        return out

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
            if node.shape.width not in (None, 1):
                # 只有种子(手写因子的 in_slice 之类)才有这种窄于整天的节点;随机子树总是整天宽,
                # 换进去和兄弟节点对不齐,引擎会报错。随机树永远走不到这里,轨迹不变。
                continue
            new = self.subtree(shape_req(node), max(1, node.depth()))
            try:
                child = replace(expr, node, new)
            except ShapeError:
                continue
            if self.acceptable(child):
                return child
        return None

    def mutate_operator(self, expr: Expression) -> Optional[Expression]:
        """Swap one node's operator for another that accepts the same operands.

        Subtree mutation throws a whole branch away; this keeps the branch and
        changes only what is done to it, which is the small move a search needs to
        refine a structure rather than restart it."""
        nodes = [n for n, _ in collect(expr) if n.spec is not None]
        if not nodes:
            return None
        for _ in range(self.max_retries):
            node = self._rng.choice(nodes)
            alts = [s for s in self.specs
                    if s.name != node.spec.name and s.category == node.spec.category
                    and s.arity == node.spec.arity and s.face == node.spec.face
                    and s.family == node.spec.family]
            if not alts:
                continue
            spec = self._pick(alts)
            w = node.window
            if spec.requires_window and (w is None or (spec.windows and w not in spec.windows)):
                w = self._rng.choice(list(spec.windows)) if spec.windows else w
            try:
                swapped = make_node(spec, tuple(node.operands), w)
                child = replace(expr, node, swapped)
            except ShapeError:
                continue
            if self.acceptable(child):
                return child
        return None

    def mutate_hoist(self, expr: Expression) -> Optional[Expression]:
        """Promote one of the tree's own subtrees to be the whole tree.

        The counterweight to growth: without it trees only ever get deeper, and a
        population of sixteen-layer wrappers around one primitive is what that
        looks like."""
        cands = [n for n, parent in collect(expr) if parent is not None and n.spec is not None]
        self._rng.shuffle(cands)
        for node in cands[: self.max_retries]:
            if self.acceptable(node):
                return node
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
        # Before WP5 this checked only is_valid_root, so a banned depth-1 subtree
        # elsewhere in the tree (untouched by the window swap) walked back through
        # unchecked -- e.g. with O:day_argmax(F:Low) banned, mutating the window of
        # cross_quantile_q25(day_argmax(Low), 3) still returned the child. acceptable()
        # closes that (and lets the accept hook see this route too); the escape hatch
        # restores the old expression exactly, for bit-reproducing a historical --ban
        # run that relied on the hole.
        return child if (self.acceptable(child) if self.mutation_checks_bans else is_valid_root(child)) else None

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
        # Same hole, same fix -- see mutate_window above.
        return child if (self.acceptable(child) if self.mutation_checks_bans else is_valid_root(child)) else None

    def crossover(self, a: Expression, b: Expression) -> Optional[Expression]:
        nodes_a, nodes_b = collect(a), collect(b)
        for _ in range(self.max_retries):
            na, pa = self._rng.choice(nodes_a)
            nb, _ = self._rng.choice(nodes_b)
            if pa is None:
                continue
            if shape_req(na) != shape_req(nb):
                continue
            if shape_req(na) == "full" and na.shape.width != nb.shape.width:
                continue                        # 种子的窄分钟段(in_slice)只和同宽的换,同上
            try:
                child = replace(a, na, nb)
            except ShapeError:
                continue
            if self.acceptable(child):
                return child
        return None

    # -- internals -----------------------------------------------------------------
    def _pick(self, specs: List[OpSpec]) -> OpSpec:
        """Face-balanced: draw a face uniformly among those present, then an operator
        within it -- otherwise the 65 ``in_*`` operators swamp the 16 ``nt_cs_*``.

        With ``op_weight`` unset this is exactly two ``_rng.choice`` calls in this
        order -- byte-identical to every run before ``op_weight`` existed, which is
        the single most important guarantee in the design: any change to this draw
        count diverges every trajectory from generation 1. With it set, both draws
        are tilted instead; see ``_pick_weighted``."""
        faces: Dict[str, List[OpSpec]] = {}
        for s in specs:
            faces.setdefault(s.face, []).append(s)
        if self.op_weight is None:
            return self._rng.choice(faces[self._rng.choice(sorted(faces))])
        return self._pick_weighted(faces)

    def _pick_weighted(self, faces: Dict[str, List[OpSpec]]) -> OpSpec:
        """Weighted counterpart of the uniform branch above, reproducing Partura's
        current ``pick`` semantics exactly: a face's mass is its operators' mean
        weight (so a face with more operators in it is not automatically louder),
        then an operator within the chosen face in proportion to its own weight. A
        face whose every operator weighs 0 drops out of the draw; if that drops
        every face, the uniform path above runs instead -- a weight function must
        not be able to make generation impossible."""
        weights: Dict[str, List[float]] = {
            face: [max(0.0, float(self.op_weight(s))) for s in specs] for face, specs in faces.items()
        }
        live = [face for face in sorted(faces) if sum(weights[face]) > 0.0]
        if not live:
            return self._rng.choice(faces[self._rng.choice(sorted(faces))])
        masses = [sum(weights[face]) / len(weights[face]) for face in live]
        face = self._rng.choices(live, weights=masses)[0]
        return self._rng.choices(faces[face], weights=weights[face])[0]

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
            # daily/collapsed cannot be a leaf: minimal collapse of a field -- unless
            # a daily leaf is drawn instead. Off by default (self.daily_fields empty),
            # so no draw is consumed and the stream is unchanged; `_leaf("daily")` is
            # dead code either way, since this is the only depth-1 daily/collapsed
            # slot and it never calls `_leaf`.
            if self.daily_fields and self._rng.random() < self.daily_leaf_prob:
                return DailyFeature(self._rng.choice(self.daily_fields))
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
        # The positional hint `_operand_reqs` cannot express: an nd-face binary gets a
        # "daily" requirement in BOTH slots, so without this a search cannot know
        # which operand is e.g. a bucketing kernel's own conditioning variable. Off by
        # default (self.leaf_slots empty/None) -- no draw added.
        slots = self.leaf_slots.get(spec.name) if self.leaf_slots else None
        for i, r in enumerate(reqs):
            if i == const_slot:
                operands.append(Constant(self._rng.choice(self.constants)))
            elif slots and i in slots:
                operands.append(DailyFeature(self._rng.choice(slots[i])))
            else:
                operands.append(self._gen(r, depth - 1, full))
        return make_node(spec, tuple(operands), self._window_for(spec))


__all__ = ["TypedTreeGenerator", "shape_req", "collect", "REQS", "GenerationExhausted"]
