"""Plugin loading: how a caller (Partura, or anyone else) reaches the search
without AlphaCube importing them back.

AlphaCube never imports a plugin on its own and never names one in its source
-- the caller always supplies the module name itself (a CLI's ``--plugin`` /
``ALPHACUBE_PLUGINS``, or a direct library call to :func:`load_plugins`). A
plugin is an ordinary importable module that may do two things:

- register new operators at IMPORT time: an ``@op(...)``-decorated kernel
  lands in ``operators.common.OPERATOR_REGISTRY`` the moment the module body
  runs, with no further call needed;
- expose a ``register()`` function for whatever must happen BEFORE the
  registry is built from that table -- :func:`declare_real_valued`,
  ``mining.scoring.register_scorer``, and populating the legacy / full-catalog
  name tables (``operators.operator_dict_DT`` / ``operator_dict_ND``) so they
  see the new kernels too.

A plugin may also expose ``search_hooks() -> SearchHooks`` for whatever
steering opinion it has on the generator itself (operator weights, an accept
hook, daily leaves) -- see :func:`search_hooks_from` and ``mining/gp.py``'s
``GPMiner(hooks=...)``.

See docs/alphacube_steering_design.md, "乙 Steering hooks" section 3 ("Reaching
the hooks") and work package "### WP7".
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import catalog
from .axes import OpSpec
from .expression import Expression


@dataclass(frozen=True)
class SearchHooks:
    """One bundle carrying everything a plugin wants the search to see,
    forwarded whole into ``GPMiner(hooks=...)`` -> ``TypedTreeGenerator(
    op_weight=..., accept=..., daily_fields=..., daily_leaf_prob=...,
    leaf_slots=...)`` (``mining/gp.py``, WP7).

    Every field's default is exactly the value ``TypedTreeGenerator`` itself
    defaults to, so ``hooks=None`` and ``hooks=SearchHooks()`` both leave the
    generator -- and therefore the random stream -- byte-identical to a run
    with no plugin at all; see docs/alphacube_steering_design.md, "Backward
    compatibility" item 15."""

    op_weight: Optional[Callable[[OpSpec], float]] = None
    accept: Optional[Callable[[Expression], bool]] = None
    daily_fields: Tuple[str, ...] = ()
    daily_leaf_prob: float = 0.0
    leaf_slots: Dict[str, Dict[int, Tuple[str, ...]]] = field(default_factory=dict)


def load_plugins(names: Sequence[str]) -> List[str]:
    """Import each of ``names`` (``importlib.import_module``) and call its
    ``register()`` if it has one and it is callable, then -- once, and only if
    ``names`` is non-empty -- drop the cached default registry
    (``catalog._DEFAULT = None``) so the next ``default_registry()`` call
    rebuilds it and sees whatever the plugins just registered. Returns
    ``names`` back as a plain list, in the order loaded, for the caller to
    report ("loaded plugins: ...").

    Importing alone is already enough for anything that registers at import
    time: an ``@op(...)``-decorated kernel lands in
    ``operators.common.OPERATOR_REGISTRY`` the moment the module body runs.
    ``register()`` is the hook for what must happen BEFORE ``build_registry()``
    runs instead -- :func:`declare_real_valued`, ``mining.scoring.
    register_scorer``, and populating ``operators.operator_dict_DT`` /
    ``operator_dict_ND`` so the legacy / full-catalog name tables see the new
    kernels too.

    The reset is unconditional -- not guarded by "raise if the registry is
    already built". That guard is order-sensitive inside a pytest session (or
    any process that happened to call ``default_registry()`` before loading a
    plugin) and would fail its own positive test there. What that costs: a
    caller holding a ``Registry`` object returned *before* this call still
    sees the short vocabulary -- the object itself does not change, only what
    ``default_registry()`` returns next -- and the symptom is ``ParseError:
    unknown operator``, which reads like a typo rather than a stale registry.
    Call ``load_plugins`` before the first ``default_registry()`` of a run to
    avoid it.

    AlphaCube never imports a plugin on its own and never names one in its
    source -- ``names`` always comes from the caller (a CLI's ``--plugin`` /
    ``ALPHACUBE_PLUGINS``, or a direct library call)."""
    loaded: List[str] = []
    for name in names:
        module = importlib.import_module(name)
        register = getattr(module, "register", None)
        if callable(register):
            register()
        loaded.append(name)
    if loaded:
        catalog._DEFAULT = None
    return loaded


def declare_real_valued(*names: str) -> None:
    """Add ``names`` to ``catalog.REAL_VALUED_OVERRIDES`` -- the by-name escape
    hatch for a kernel a plugin does not own (so it cannot pass
    ``real_valued=True`` at the ``@op(...)`` definition site itself) but knows
    is genuinely real-valued, never boolean/trivial on any input worth
    searching. Mutates the existing set in place (``.update``, never a
    reassignment of ``catalog.REAL_VALUED_OVERRIDES`` itself), because
    ``catalog._make_spec`` and anything that already did ``from
    alphacube.catalog import REAL_VALUED_OVERRIDES`` must see the same object
    change. Takes effect on the next registry build -- call it, like
    ``register()`` itself, before ``default_registry()`` runs (see
    :func:`load_plugins`'s caveat about a registry already returned)."""
    catalog.REAL_VALUED_OVERRIDES.update(names)


def search_hooks_from(names: Sequence[str]) -> Optional[SearchHooks]:
    """The :class:`SearchHooks` a run assembled from ``names`` should use.

    Imports each of ``names`` (``importlib.import_module`` -- idempotent via
    ``sys.modules``, so this is safe to call whether or not
    :func:`load_plugins` already imported them, and in either order) and
    returns the result of calling ``search_hooks()`` on the LAST one that
    exposes a callable by that name, with no arguments. ``None`` when no
    module in ``names`` exposes one -- which is what lets
    ``GPMiner(hooks=search_hooks_from(names))`` reproduce today's generator
    exactly for a run with no plugin, or with a plugin that only registers
    operators / daily fields / scorer kinds and has no steering opinion of its
    own."""
    hooks: Optional[SearchHooks] = None
    for name in names:
        module = importlib.import_module(name)
        factory = getattr(module, "search_hooks", None)
        if callable(factory):
            hooks = factory()
    return hooks


__all__ = ["SearchHooks", "load_plugins", "declare_real_valued", "search_hooks_from"]
