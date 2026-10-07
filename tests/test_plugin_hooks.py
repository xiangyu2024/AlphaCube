"""WP7 -- the plugin loader (``plugins.py``: ``SearchHooks`` / ``load_plugins`` /
``declare_real_valued`` / ``search_hooks_from``), ``GPMiner``'s plumbing for it
(``hooks=`` / ``warmup=`` / ``wall_seconds=``), the scorer-registry-aware
``fitness_kind`` guard, the wall clock, the ``check_ban_key`` ``DF:`` branch, and
``view_cache_key``'s daily-fingerprint keyword. See
docs/alphacube_steering_design.md, "乙 Steering hooks" section 3 ("Reaching the
hooks") and section 4 ("GPMiner plumbing"), and work package "### WP7".
"""

from __future__ import annotations

import hashlib
import importlib

import pytest

from alphacube.label import Label
from alphacube.mining.gp import GPMiner, check_ban_key
from alphacube.mining.pool import MsePool
from alphacube.mining.rankban import ExemplarCache, view_cache_key
from alphacube.mining.scoring import BOOK_TOP_FRAC, SCORER_FACTORIES, PanelScorer, register_scorer, scorer_kinds
from alphacube.plugins import SearchHooks, declare_real_valued, load_plugins, search_hooks_from
from alphacube.store import CubeStore
from alphacube.synthetic import make_synthetic_cube, write_daily_fields


@pytest.fixture(autouse=True)
def _clean_scorer_factories():
    """``SCORER_FACTORIES`` is a module-level dict shared by every test file (and,
    eventually, every plugin). Snapshot/restore around each test in THIS file so a
    registration here can never leak into test_scorer_registry.py's own
    assertions (e.g. its ``test_registry_is_empty_by_default``), or the reverse,
    regardless of collection order -- mirrors that file's own ``_clean_registry``
    fixture."""
    before = dict(SCORER_FACTORIES)
    try:
        yield
    finally:
        SCORER_FACTORIES.clear()
        SCORER_FACTORIES.update(before)


@pytest.fixture
def mining_view(store):
    return store.view(start=store.dates[5])


@pytest.fixture
def mining_target(mining_view):
    return Label("close", 1).compute(mining_view)


# ---------------------------------------------------------------------------
# GPMiner(hooks=...): byte-identical with no hooks, forwarded with one
# ---------------------------------------------------------------------------


def test_gpminer_hooks_none_matches_omitting_hooks_entirely(registry, mining_view, mining_target):
    """hooks=None (and hooks=SearchHooks(), every field at its own default) must
    reproduce exactly what omitting the keyword does -- not just the same final
    archive, but the same BRED population every generation, for three
    generations. ``_next_generation`` is wrapped per-instance (a plain
    monkey-patch of a bound method on one object) to record what each call
    returns, since GPMiner does not otherwise expose a generation's population."""

    def _generations(hooks_kw):
        pool = MsePool(4, mining_target)
        miner = GPMiner(registry, mining_view, mining_target, pool, population_size=10, max_depth=3,
                        seed=5, **hooks_kw)
        seen = []
        orig = miner._next_generation

        def wrapped(pop, fit):
            nxt = orig(pop, fit)
            seen.append([e.key() for e in nxt])
            return nxt

        miner._next_generation = wrapped
        miner.mine(10 * 3)
        return seen

    omitted = _generations({})
    explicit_none = _generations({"hooks": None})
    explicit_empty = _generations({"hooks": SearchHooks()})
    assert len(omitted) == 3           # exactly three generations, as asked
    assert omitted == explicit_none
    assert omitted == explicit_empty


# ---------------------------------------------------------------------------
# load_plugins
# ---------------------------------------------------------------------------


def _write_module(tmp_path, monkeypatch, name: str, source: str):
    (tmp_path / f"{name}.py").write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()


def test_load_plugins_imports_registers_and_resets_default_registry(tmp_path, monkeypatch, registry):
    """``registry`` (session-scoped) has ALREADY forced ``catalog._DEFAULT`` to be
    built by the time this test runs -- load_plugins must not guard on that (a
    raise-if-already-built guard would fail its OWN positive test for exactly
    this reason, which is why the design rejects it). Cleans up afterward
    (``OPERATOR_REGISTRY.pop`` + restoring ``catalog._DEFAULT``) exactly as
    ``tests/test_real_valued_flag.py``'s ``fake_constant_capable_kernel`` fixture
    does, so a throwaway operator registered here cannot leak into some other
    test file's ``build_registry()``."""
    import alphacube.catalog as catalog_mod
    from alphacube.operators.common import OPERATOR_REGISTRY

    assert catalog_mod._DEFAULT is not None
    original_default = catalog_mod._DEFAULT
    mod_name = "_wp7_plugin_register_probe"
    _write_module(tmp_path, monkeypatch, mod_name,
        "from alphacube.operators.common import op\n"
        "REGISTER_CALLS = []\n"
        '@op("shared", "elementwise", arity=1, real_valued=True)\n'
        "def wp7_plugin_probe_op(x):\n"
        "    return x * 2.0\n"
        "def register():\n"
        "    REGISTER_CALLS.append(1)\n")

    try:
        loaded = load_plugins([mod_name])
        assert loaded == [mod_name]
        assert catalog_mod._DEFAULT is None                      # the reset

        plugin_module = importlib.import_module(mod_name)
        assert plugin_module.REGISTER_CALLS == [1]                # register() really ran

        fresh = catalog_mod.default_registry()
        assert "wp7_plugin_probe_op" in fresh
        assert fresh is not original_default
    finally:
        OPERATOR_REGISTRY.pop("wp7_plugin_probe_op", None)
        catalog_mod._DEFAULT = original_default


def test_load_plugins_with_no_names_is_a_no_op(registry):
    import alphacube.catalog as catalog_mod

    before = catalog_mod._DEFAULT
    assert load_plugins(()) == []
    assert catalog_mod._DEFAULT is before                         # untouched, not even reset-then-rebuilt


def test_load_plugins_skips_modules_without_a_callable_register(tmp_path, monkeypatch, registry):
    """A module with no ``register`` at all, and one whose ``register`` is present
    but not callable, must both load cleanly (no ``TypeError``) and still count
    as loaded for the ``catalog._DEFAULT`` reset."""
    import alphacube.catalog as catalog_mod

    original_default = catalog_mod._DEFAULT
    mod_a, mod_b = "_wp7_plugin_no_register_attr", "_wp7_plugin_register_not_callable"
    _write_module(tmp_path, monkeypatch, mod_a, "X = 1\n")
    _write_module(tmp_path, monkeypatch, mod_b, "register = 42\n")

    try:
        assert load_plugins([mod_a]) == [mod_a]
        assert catalog_mod._DEFAULT is None
        catalog_mod._DEFAULT = original_default                   # reset between the two sub-cases

        assert load_plugins([mod_b]) == [mod_b]                   # must not raise TypeError
        assert catalog_mod._DEFAULT is None
    finally:
        catalog_mod._DEFAULT = original_default


# ---------------------------------------------------------------------------
# declare_real_valued
# ---------------------------------------------------------------------------


def test_declare_real_valued_updates_the_shared_overrides_set_in_place():
    import alphacube.catalog as catalog_mod
    from alphacube.catalog import REAL_VALUED_OVERRIDES as imported_overrides

    name_a, name_b = "_wp7_declare_real_valued_probe_a", "_wp7_declare_real_valued_probe_b"
    assert name_a not in catalog_mod.REAL_VALUED_OVERRIDES
    try:
        declare_real_valued(name_a, name_b)
        assert {name_a, name_b} <= catalog_mod.REAL_VALUED_OVERRIDES
        # mutated in place, not reassigned: a name already bound via
        # `from alphacube.catalog import REAL_VALUED_OVERRIDES` must see it too.
        assert imported_overrides is catalog_mod.REAL_VALUED_OVERRIDES
        assert name_a in imported_overrides
    finally:
        catalog_mod.REAL_VALUED_OVERRIDES.discard(name_a)
        catalog_mod.REAL_VALUED_OVERRIDES.discard(name_b)


# ---------------------------------------------------------------------------
# search_hooks_from
# ---------------------------------------------------------------------------


def test_search_hooks_from_last_module_wins_and_none_without_one(tmp_path, monkeypatch):
    mod_a, mod_b, mod_c = "_wp7_hooks_mod_a", "_wp7_hooks_mod_b", "_wp7_hooks_mod_c"
    _write_module(tmp_path, monkeypatch, mod_a,
        "from alphacube.plugins import SearchHooks\n"
        "def search_hooks():\n"
        "    return SearchHooks(daily_leaf_prob=0.1)\n")
    _write_module(tmp_path, monkeypatch, mod_b, "X = 1\n")                   # no search_hooks at all
    _write_module(tmp_path, monkeypatch, mod_c,
        "from alphacube.plugins import SearchHooks\n"
        "def search_hooks():\n"
        "    return SearchHooks(daily_leaf_prob=0.9)\n")

    assert search_hooks_from([mod_a, mod_b, mod_c]) == SearchHooks(daily_leaf_prob=0.9)
    assert search_hooks_from([mod_c, mod_b, mod_a]) == SearchHooks(daily_leaf_prob=0.1)
    assert search_hooks_from([mod_b]) is None
    assert search_hooks_from([]) is None


# ---------------------------------------------------------------------------
# SearchHooks(op_weight=...) reaches _pick through GPMiner(hooks=...)
# ---------------------------------------------------------------------------


def test_search_hooks_op_weight_reaches_pick_via_gpminer(registry, mining_view, mining_target):
    pool = MsePool(4, mining_target)
    hooks = SearchHooks(op_weight=lambda s: 0.0 if s.face == "nd" else 1.0)
    miner = GPMiner(registry, mining_view, mining_target, pool, population_size=8, max_depth=3, seed=5,
                    hooks=hooks)
    draws = [miner.generator._pick(miner.generator.specs) for _ in range(300)]
    assert draws and not any(s.face == "nd" for s in draws)

    plain_pool = MsePool(4, mining_target)
    plain = GPMiner(registry, mining_view, mining_target, plain_pool, population_size=8, max_depth=3, seed=5)
    plain_draws = [plain.generator._pick(plain.generator.specs) for _ in range(300)]
    assert any(s.face == "nd" for s in plain_draws)   # reachable without the hook: the check above is not vacuous


def test_search_hooks_daily_fields_reach_the_generator_via_gpminer(registry, mining_view, mining_target):
    """The other four SearchHooks fields forward through the identical
    ``hooks.X if hooks is not None else <default>`` line op_weight does; this
    checks daily_fields / daily_leaf_prob specifically, since a DailyFeature
    leaf is generatable (if not evaluatable) without a daily store attached --
    daily fields are always PARSEABLE, per WP2's design note; whether data for
    them exists is a view-time question."""
    from alphacube.expression import DailyFeature

    pool = MsePool(4, mining_target)
    hooks = SearchHooks(daily_fields=("FloatCap",), daily_leaf_prob=1.0)
    miner = GPMiner(registry, mining_view, mining_target, pool, population_size=8, max_depth=4, seed=6,
                    hooks=hooks)
    pop = miner.generator.generate_population(30)
    assert any(isinstance(n, DailyFeature) for e in pop for n in e.nodes())

    plain_pool = MsePool(4, mining_target)
    plain = GPMiner(registry, mining_view, mining_target, plain_pool, population_size=8, max_depth=4, seed=6)
    plain_pop = plain.generator.generate_population(30)
    assert not any(isinstance(n, DailyFeature) for e in plain_pop for n in e.nodes())


# ---------------------------------------------------------------------------
# GPMiner(fitness_kind=<registered>): constructs, mines, forwards dates/warmup
# ---------------------------------------------------------------------------


def test_gpminer_registered_fitness_kind_mines_and_forwards_dates_and_warmup(registry, mining_view, mining_target):
    captured = {}

    def factory(target, *, dates=None, warmup=0, turnover_penalty=0.0, signed=False, cost_per_side=0.0015,
               top_frac=BOOK_TOP_FRAC, min_coverage=0.5, min_scored_days=0.5):
        captured["dates"], captured["warmup"] = dates, warmup
        return PanelScorer(target, kind="rank_ic", turnover_penalty=turnover_penalty, signed=signed,
                           top_frac=top_frac, min_coverage=min_coverage, min_scored_days=min_scored_days)

    register_scorer("wp7_plugin_fitness", factory)
    pool = MsePool(4, mining_target)
    miner = GPMiner(registry, mining_view, mining_target, pool, population_size=8, max_depth=3, seed=1,
                    fitness_kind="wp7_plugin_fitness", warmup=3)
    assert captured["dates"] == list(mining_view.dates)   # forwarded at construction (make_scorer call)
    assert captured["warmup"] == 3

    rep = miner.mine(16)
    assert rep.generations >= 1

    with pytest.raises(ValueError) as exc:
        GPMiner(registry, mining_view, mining_target, pool, fitness_kind="not_a_real_kind_xyz")
    assert str(exc.value) == f"fitness_kind must be one of {scorer_kinds()}, not 'not_a_real_kind_xyz'"
    assert "wp7_plugin_fitness" in str(exc.value)


# ---------------------------------------------------------------------------
# wall_seconds
# ---------------------------------------------------------------------------


def test_wall_seconds_zero_runs_the_full_budget(registry, mining_view, mining_target):
    pool = MsePool(4, mining_target)
    miner = GPMiner(registry, mining_view, mining_target, pool, population_size=6, max_depth=3, seed=2,
                    wall_seconds=0.0)
    rep = miner.mine(6 * 5)
    assert rep.generations == 5
    assert rep.stopped_early == ""


def test_wall_seconds_zero_never_evaluates_the_new_checks_second_operand(
        registry, mining_view, mining_target, monkeypatch):
    """Isolates the NEW ``if ws and time.perf_counter() - t0 >= ws`` check from
    every OTHER perf_counter() call already in the pipeline -- ``engine/
    engine.py`` times its own ``evaluate()`` calls through the very same global
    ``time`` module, so a plain call-count assertion on a single run would be
    measuring the engine's instrumentation, not the new check. Instead, the
    SAME seed/budget is mined twice -- once with ``wall_seconds=0.0`` (must
    short-circuit before the comparison) and once with a value large enough to
    never trip (so every generation still runs, identically, and the ONLY
    difference between the two runs is whether the check's second operand is
    ever evaluated) -- the call-count gap between them must be exactly one per
    generation actually run (``rolling_block`` is 0 here, so the check site is
    reached once per generation, not twice)."""
    import alphacube.mining.gp as gp_module

    real_perf_counter = gp_module.time.perf_counter
    calls: list = []

    def counting():
        calls.append(1)
        return real_perf_counter()

    monkeypatch.setattr(gp_module.time, "perf_counter", counting)

    pool0 = MsePool(4, mining_target)
    m0 = GPMiner(registry, mining_view, mining_target, pool0, population_size=6, max_depth=3, seed=2,
                wall_seconds=0.0)
    calls.clear()
    rep0 = m0.mine(6 * 4)
    calls_at_zero = len(calls)

    pool1 = MsePool(4, mining_target)
    m1 = GPMiner(registry, mining_view, mining_target, pool1, population_size=6, max_depth=3, seed=2,
                wall_seconds=1e9)                      # truthy, but never actually trips
    calls.clear()
    rep1 = m1.mine(6 * 4)
    calls_at_never_trips = len(calls)

    assert rep0.generations == rep1.generations == 4   # identical work: same seed, same budget
    assert rep0.stopped_early == rep1.stopped_early == ""
    assert calls_at_never_trips - calls_at_zero == rep0.generations


def test_wall_seconds_stops_mine_early_and_the_mine_level_argument_overrides_the_constructors(
        registry, mining_view, mining_target):
    """No fake clock needed (and, per the test above, a fake clock on the global
    ``time`` module would also move every engine-internal timing call, which
    cannot be predicted without depending on cache-hit internals): an
    effectively-zero but truthy ``wall_seconds`` (1e-9) is already over budget
    by the very first check, right after generation 1's own archive_update, on
    any real machine -- generating and scoring a population takes far longer
    than a nanosecond. So exactly 1 of the 5 planned generations runs,
    deterministically regardless of machine speed."""
    pool1 = MsePool(4, mining_target)
    m1 = GPMiner(registry, mining_view, mining_target, pool1, population_size=6, max_depth=3, seed=2,
                wall_seconds=1e-9)
    rep1 = m1.mine(6 * 5)
    assert rep1.generations == 1
    assert rep1.stopped_early == "wall clock"

    # the override passed to mine() itself must be what actually governs --
    # the constructor's own wall_seconds=1000 here would never trip.
    pool2 = MsePool(4, mining_target)
    m2 = GPMiner(registry, mining_view, mining_target, pool2, population_size=6, max_depth=3, seed=2,
                wall_seconds=1000.0)
    rep2 = m2.mine(6 * 5, wall_seconds=1e-9)
    assert rep2.generations == 1
    assert rep2.stopped_early == "wall clock"


# ---------------------------------------------------------------------------
# check_ban_key: the DF: branch
# ---------------------------------------------------------------------------


def test_check_ban_key_accepts_a_daily_leaf_and_rejects_an_unknown_one(registry):
    fields = list(registry.fields)
    check_ban_key("O:cs_rank(DF:FloatCap)", registry, fields)         # must not raise
    with pytest.raises(ValueError, match="no daily field named 'Nonsense'"):
        check_ban_key("O:cs_rank(DF:Nonsense)", registry, fields)


def test_check_ban_key_still_handles_field_and_constant_leaves_as_before(registry):
    """Regression pin for the two pre-existing branches, unmoved by adding DF:."""
    fields = list(registry.fields)
    check_ban_key("O:log(F:Amt)", registry, fields)
    check_ban_key("O:nt_ts_argmax(F:High)|w3", registry, fields)
    with pytest.raises(ValueError, match="no field named 'NotAField'"):
        check_ban_key("O:log(F:NotAField)", registry, fields)
    with pytest.raises(ValueError, match="not a leaf"):
        check_ban_key("O:log(X:Amt)", registry, fields)


# ---------------------------------------------------------------------------
# view_cache_key's daily= keyword
# ---------------------------------------------------------------------------


def test_view_cache_key_daily_default_is_byte_identical_to_a_hand_written_hash(store):
    view = store.view()
    dates = [str(x) for x in view.dates]
    raw = "|".join([dates[0], dates[-1], str(len(dates)), str(view.n_symbols), "250", "512"])
    expected = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    assert view_cache_key(view, 250, 512) == expected
    assert view_cache_key(view, 250, 512, daily=None) == expected    # explicit None: still unchanged


def test_view_cache_key_differs_with_a_fingerprint_and_two_roots_give_two_cache_dirs(store, daily_root, tmp_path):
    from alphacube import dailyfields

    plain = view_cache_key(store.view(), 250, 512)

    view_a = CubeStore(store.directory, daily_root=daily_root).view()
    fp_a = dailyfields.fingerprint(view_a.store.daily)
    assert fp_a is not None
    key_a = view_cache_key(view_a, 250, 512, daily=fp_a)
    assert key_a != plain
    assert key_a == view_cache_key(view_a, 250, 512, daily=fp_a)      # deterministic, not merely "different"

    # A second daily root with a DIFFERENT field set: a different meta.json ->
    # a different meta_sha256 -> a different fingerprint (meta_sha256 hashes the
    # fields/symbols/dates manifest, not the plane VALUES, so a second root with
    # the SAME field set would alias the first -- see dailyfields.DailyFieldStore).
    _, _, _, _, planes_b = make_synthetic_cube(n_symbols=store.n_symbols, n_days=store.n_days,
                                               daily_fields=("FloatCap",))
    root_b = write_daily_fields(tmp_path / "daily_b", planes_b, store.symbols, store.dates)
    view_b = CubeStore(store.directory, daily_root=root_b).view()
    fp_b = dailyfields.fingerprint(view_b.store.daily)
    assert fp_b is not None and fp_b != fp_a
    key_b = view_cache_key(view_b, 250, 512, daily=fp_b)
    assert key_b != key_a

    cache_a = ExemplarCache(tmp_path / "exemplars", key_a)
    cache_b = ExemplarCache(tmp_path / "exemplars", key_b)
    assert cache_a.dir == tmp_path / "exemplars" / key_a
    assert cache_b.dir == tmp_path / "exemplars" / key_b
    assert cache_a.dir != cache_b.dir


def test_gpminer_exemplar_cache_uses_the_daily_fingerprint(registry, store, daily_root, tmp_path):
    """End-to-end through GPMiner itself (gp.py step 8): a run over a daily-
    aware view must land its ExemplarCache under the fingerprinted key, not the
    bare one -- otherwise a rank-ban fingerprint cached before a daily-field
    rebuild would be silently reused as if still valid."""
    from alphacube import dailyfields

    view = CubeStore(store.directory, daily_root=daily_root).view(start=store.dates[5])
    target = Label("close", 1).compute(view)
    pool = MsePool(4, target)
    miner = GPMiner(registry, view, target, pool, population_size=6, max_depth=3, seed=0,
                    ban_rank_corr=0.95, exemplar_cache=str(tmp_path), exemplar_cache_gb=1.0)
    expected_key = view_cache_key(view, miner._rank_ban.index.days.size, miner._rank_ban.index.cols.size,
                                  daily=dailyfields.fingerprint(view.store.daily))
    assert miner._exemplar_cache.view_key == expected_key
    assert miner._exemplar_cache.dir == tmp_path / expected_key
