"""WP8 -- pipeline.run_mining's new ``daily_root`` keyword and the RL-branch
parity fixes (dates/warmup forwarded to make_scorer, banned_substructures
validated like the GP branch, a plugin's ``accept`` hook reaching
BatchRewarder), plus the CLI layer that reaches all of it: ``--daily-fields``
/ ``--plugin`` / ``--wall-seconds`` on ``cli.py``'s shared parser, ``main()``
loading plugins before ``args.fn(args)`` runs, and ``--fitness`` losing its
static ``choices=`` in favour of a ``scorer_kinds()`` check inside
``cmd_mine``. ``research.evaluate_factors`` / ``calculate.
run_factor_calculation`` each gain the same ``daily_root`` keyword. See
docs/alphacube_steering_design.md, work package "### WP8" and its
"Backward compatibility" item 16.
"""

from __future__ import annotations

import importlib
import json

import pytest

import alphacube.catalog as catalog_mod
import alphacube.mining as mining_mod
from alphacube import cli
from alphacube.engine.engine import CubeEngine
from alphacube.label import Label
from alphacube.mining.env import BatchRewarder
from alphacube.mining.gp import GPMiner as RealGPMiner
from alphacube.mining.pool import MsePool
from alphacube.mining.scoring import SCORER_FACTORIES, register_scorer, scorer_kinds
from alphacube.parser import parse
from alphacube.pipeline import run_mining
from alphacube.plugins import SearchHooks
from alphacube.store import CubeStore


def _write_module(tmp_path, monkeypatch, name: str, source: str) -> None:
    """Mirrors tests/test_plugin_hooks.py's own helper (WP7) -- duplicated rather
    than imported, since WP8 owns this file alone and that one belongs to WP7."""
    (tmp_path / f"{name}.py").write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()


def _fail():
    raise AssertionError("engine should not be called")


@pytest.fixture(autouse=True)
def _clean_global_registries():
    """``cli.main`` -> ``load_plugins`` mutates two process-global singletons:
    ``mining.scoring.SCORER_FACTORIES`` and ``catalog._DEFAULT``. Snapshot/restore
    around every test in this file, mirroring test_plugin_hooks.py's own cleanup
    fixtures, so a plugin loaded here can never leak into test_scorer_registry.py
    or test_plugin_hooks.py's own assertions regardless of collection order."""
    factories_before = dict(SCORER_FACTORIES)
    default_before = catalog_mod._DEFAULT
    try:
        yield
    finally:
        SCORER_FACTORIES.clear()
        SCORER_FACTORIES.update(factories_before)
        catalog_mod._DEFAULT = default_before


# A plugin registering a --fitness kind: the minimal shape mining.scoring.
# make_scorer calls a registered factory with (see register_scorer's docstring).
_SCORER_PLUGIN_SOURCE = (
    "from alphacube.mining.scoring import PanelScorer, register_scorer\n"
    "CAPTURED = []\n"
    "def _factory(target, *, dates=None, warmup=0, turnover_penalty=0.0, signed=False,\n"
    "             cost_per_side=0.0015, top_frac=0.1, min_coverage=0.5, min_scored_days=0.5):\n"
    "    CAPTURED.append((dates, warmup))\n"
    "    return PanelScorer(target, kind='rank_ic', turnover_penalty=turnover_penalty, signed=signed,\n"
    "                       top_frac=top_frac, min_coverage=min_coverage, min_scored_days=min_scored_days)\n"
    "def register():\n"
    "    register_scorer('wp8_plugin_kind', _factory)\n"
)


@pytest.fixture(scope="module")
def mine_store(tmp_path_factory):
    """Sized exactly like tests/test_long_scorer.py's / test_long_hit_scorer.py's
    own run_mining end-to-end fixture (40 symbols, 70 days, seed=3) -- that is the
    proven-reliable recipe for a tiny synthetic run that actually fills the pool,
    which this file's full CLI integration test needs too."""
    from alphacube.synthetic import write_synthetic_store

    root = tmp_path_factory.mktemp("wp8_mine_cube")
    return write_synthetic_store(root, n_symbols=40, n_days=70, n_minutes=24, seed=3,
                                 suspended_frac=0.03, unlisted_days=4)


@pytest.fixture(scope="module")
def mine_daily_root(tmp_path_factory, mine_store):
    """A dailyfields.DailyFieldStore directory over the SAME symbols/dates as
    `mine_store`, built the same way tests/conftest.py's own `daily_root` fixture
    is (identifiable, not random, planes -- see synthetic.make_synthetic_cube)."""
    from alphacube.synthetic import make_synthetic_cube, write_daily_fields

    _, _, _, _, planes = make_synthetic_cube(n_symbols=mine_store.n_symbols, n_days=mine_store.n_days,
                                              daily_fields=("FloatCap", "TotalCap", "FloatShares"))
    root = tmp_path_factory.mktemp("wp8_daily")
    return write_daily_fields(root, planes, mine_store.symbols, mine_store.dates)


# ---------------------------------------------------------------------------
# cli.main(["mine", ...]): --daily-fields, --plugin, a plugin fitness kind
# ---------------------------------------------------------------------------


def test_cli_mine_with_daily_fields_and_plugin_fitness_returns_0_and_records_contract_hash(
        tmp_path, monkeypatch, mine_store, mine_daily_root):
    mod_name = "_wp8_mine_scorer_plugin"
    _write_module(tmp_path, monkeypatch, mod_name, _SCORER_PLUGIN_SOURCE)
    out = tmp_path / "runs"

    rc = cli.main(["mine", "--store", str(mine_store.directory), "--daily-fields", str(mine_daily_root),
                  "--plugin", mod_name, "--fitness", "wp8_plugin_kind",
                  "--start", mine_store.dates[5], "--label", "close",
                  "--budget", "24", "--population", "12", "--max-depth", "3",
                  "--min-coverage", "0.8", "--cost-per-side", "0.0015",
                  "--pool-capacity", "3", "--combine", "",
                  "--out", str(out), "--run-id", "r1"])
    assert rc == 0

    meta = json.loads((out / "r1" / "meta.json").read_text(encoding="utf-8"))
    plain = CubeStore(mine_store.directory)
    daily = CubeStore(mine_store.directory, daily_root=mine_daily_root)
    assert meta["store_contract_hash"] == daily.contract_hash
    assert meta["store_contract_hash"] != plain.contract_hash   # daily-aware, not the plain store's hash

    plugin_mod = importlib.import_module(mod_name)
    assert plugin_mod.CAPTURED, "the plugin's registered scorer factory must actually have been called"


def test_cli_mine_rejects_unknown_fitness_naming_scorer_kinds():
    """argparse itself must no longer reject this at parse time (no choices= any
    more) -- cmd_mine's own scorer_kinds() check does, and before anything else
    runs, so a bogus --store never even gets opened."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["mine", "--store", "/does/not/exist", "--fitness", "totally_bogus_kind"])
    msg = str(exc.value)
    assert "totally_bogus_kind" in msg
    for k in scorer_kinds():
        assert k in msg, msg


def test_cli_mine_defaults_have_no_daily_store_and_no_hooks(tmp_path, monkeypatch, capsys, store):
    """With neither --plugin nor --daily-fields: the say() line must report no
    daily store attached, and the constructed GPMiner must receive no `hooks`
    keyword at all -- not even hooks=None spelled out, just plain omission,
    which is what reproduces today's construction exactly (WP7's own
    test_gpminer_hooks_none_matches_omitting_hooks_entirely)."""
    captured = {}

    class _RecordingGPMiner(RealGPMiner):
        def __init__(self, *a, **kw):
            captured["hooks"] = kw.get("hooks")
            super().__init__(*a, **kw)

    monkeypatch.setattr(mining_mod, "GPMiner", _RecordingGPMiner)
    out = tmp_path / "runs"
    rc = cli.main(["mine", "--store", str(store.directory), "--label", "close",
                  "--start", store.dates[5], "--budget", "12", "--population", "6", "--max-depth", "3",
                  "--combine", "", "--out", str(out), "--run-id", "r2"])
    assert rc in (0, 1)              # a tiny 10-symbol store may or may not fill the pool; must not crash
    assert "hooks" in captured and captured["hooks"] is None

    stdout = capsys.readouterr().out
    assert "daily fields: none attached" in stdout


# ---------------------------------------------------------------------------
# cli.main(["research"/"calculate", ..., "--daily-fields", ...])
# ---------------------------------------------------------------------------


def test_cli_research_with_daily_fields_evaluates_floatcap(tmp_path, store, daily_root):
    out = tmp_path / "research_cli"
    rc = cli.main(["research", "--store", str(store.directory), "--daily-fields", str(daily_root),
                  "--expr", "cap=FloatCap", "--start", store.dates[5], "--end", store.dates[-1],
                  "--label", "close", "--out", str(out), "--run-id", "r1"])
    assert rc == 0
    meta = json.loads((out / "r1" / "meta.json").read_text(encoding="utf-8"))
    assert "cap" not in meta.get("errors", {}), meta.get("errors")
    metrics_csv = (out / "r1" / "metrics.csv").read_text(encoding="utf-8")
    assert "cap" in metrics_csv


def test_cli_calculate_with_daily_fields_evaluates_floatcap(tmp_path, store, daily_root):
    out = tmp_path / "calc_cli"
    rc = cli.main(["calculate", "--store", str(store.directory), "--daily-fields", str(daily_root),
                  "--expr", "cap=FloatCap", "--start", store.dates[5], "--end", store.dates[-1],
                  "--out", str(out), "--run-id", "c1"])
    assert rc == 0
    results = json.loads((out / "c1" / "factor_results.json").read_text(encoding="utf-8"))
    assert results["rows"]["cap"]["status"] == "SUCCESS", results["rows"]["cap"]


def test_evaluate_factors_daily_root_kwarg_resolves_floatcap(store, daily_root, tmp_path):
    """Library-level pin of evaluate_factors' own new keyword (not only through the
    CLI): it is forwarded to the `CubeStore(store, daily_root=...)` construction,
    which only fires when `store` is given as a path, not an already-built
    CubeStore -- so this passes the store's directory, not the `store` fixture
    object itself."""
    from alphacube.research import evaluate_factors

    out = tmp_path / "research_lib"
    table = evaluate_factors(str(store.directory), {"cap": "FloatCap"}, start=store.dates[5], end=store.dates[-1],
                             label=Label("close", 1), daily_root=str(daily_root), out_dir=out, quiet=True)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert "cap" not in meta["errors"], meta["errors"]
    assert table.loc["cap", "coverage"] > 0.5


def test_run_factor_calculation_daily_root_kwarg_resolves_floatcap(store, daily_root, tmp_path):
    from alphacube.calculate import run_factor_calculation

    catalog_df = run_factor_calculation({"cap": "FloatCap"}, str(store.directory), save_root=tmp_path / "calc_lib",
                                        run_id="c1", start=store.dates[5], end=store.dates[-1],
                                        daily_root=str(daily_root), quiet=True)
    assert catalog_df.loc["cap", "status"] == "SUCCESS", catalog_df.loc["cap"].to_dict()


# ---------------------------------------------------------------------------
# pipeline.run_mining's RL branch: banned_substructures validation, dates/
# warmup forwarding, and hooks.accept reaching BatchRewarder
# ---------------------------------------------------------------------------


def test_run_mining_rl_branch_validates_banned_substructures(store):
    """Today only the GP path validates (gp.py's GPMiner.__init__ calls
    check_ban_key on construction); pipeline.py's RL branch used to hand the
    tuple straight to BatchRewarder, so a malformed key banned nothing, silently.
    Must raise before any mining happens."""
    with pytest.raises(ValueError, match="is not a depth-1 structure"):
        run_mining(store, start=store.dates[5], label="close", miner="alphasage", budget=4,
                  pool_capacity=2, max_expr_len=10, combine_methods=(),
                  gp_params=dict(banned_substructures=("not-a-valid-key",)))

    with pytest.raises(ValueError, match="no field named 'NotAField'"):
        run_mining(store, start=store.dates[5], label="close", miner="alphasage", budget=4,
                  pool_capacity=2, max_expr_len=10, combine_methods=(),
                  gp_params=dict(banned_substructures=("O:log(F:NotAField)",)))


def test_run_mining_rl_branch_accepts_a_valid_ban_key_and_actually_bans_it(store, monkeypatch):
    """Positive counterpart: a real key must not raise, and the ban really reaches
    BatchRewarder (mirrors test_rl_miners.py::test_banned_structure_is_refused_
    before_the_engine_runs, through the RL path of run_mining this time, by
    patching BatchRewarder to capture what it was constructed with)."""
    captured = {}

    class _RecordingRewarder(BatchRewarder):
        def __init__(self, *a, **kw):
            captured["banned_substructures"] = kw.get("banned_substructures")
            super().__init__(*a, **kw)

    monkeypatch.setattr(mining_mod, "BatchRewarder", _RecordingRewarder)
    run_mining(store, start=store.dates[5], label="close", miner="alphasage", budget=4,
              pool_capacity=2, max_expr_len=10, combine_methods=(),
              gp_params=dict(banned_substructures=("O:log(F:Amt)",)))
    assert captured["banned_substructures"] == ("O:log(F:Amt)",)


def test_run_mining_rl_branch_forwards_dates_and_warmup_to_make_scorer(store):
    """WP8 step 3: dates=list(view.dates), warmup=trim must reach a registered
    scorer factory on the RL path exactly as gp.py already does for GP (WP7's
    test_gpminer_registered_fitness_kind_mines_and_forwards_dates_and_warmup)."""
    captured = {}

    def factory(target, *, dates=None, warmup=0, **kw):
        captured["dates"], captured["warmup"] = dates, warmup
        from alphacube.mining.scoring import PanelScorer
        return PanelScorer(target, kind="rank_ic", min_coverage=kw.get("min_coverage", 0.5),
                           min_scored_days=kw.get("min_scored_days", 0.5))

    register_scorer("wp8_rl_dates_probe", factory)
    run_mining(store, start=store.dates[5], label="close", miner="alphasage", budget=4, warmup=3,
              pool_capacity=2, max_expr_len=10, combine_methods=(),
              rl_params=dict(trajectories_per_update=4),
              gp_params=dict(fitness_kind="wp8_rl_dates_probe"))
    assert captured["dates"] is not None and len(captured["dates"]) > 0
    assert captured["warmup"] == 3           # start=dates[5], warmup=3 <= 5, so trim == warmup exactly


def test_run_mining_rl_branch_forwards_hooks_accept_into_batch_rewarder(store, monkeypatch):
    """WP8 step 5: cfg.get("hooks").accept must reach BatchRewarder(accept=...);
    with no "hooks" key in gp_params at all, BatchRewarder must get accept=None,
    exactly like omitting the keyword."""
    captured = {}

    class _RecordingRewarder(BatchRewarder):
        def __init__(self, *a, **kw):
            captured["accept"] = kw.get("accept")
            super().__init__(*a, **kw)

    monkeypatch.setattr(mining_mod, "BatchRewarder", _RecordingRewarder)

    def reject_everything(expr):
        return False

    run_mining(store, start=store.dates[5], label="close", miner="alphasage", budget=4,
              pool_capacity=2, max_expr_len=10, combine_methods=(),
              rl_params=dict(trajectories_per_update=4),
              gp_params=dict(hooks=SearchHooks(accept=reject_everything)))
    assert captured["accept"] is reject_everything

    captured.clear()
    run_mining(store, start=store.dates[5], label="close", miner="alphasage", budget=4,
              pool_capacity=2, max_expr_len=10, combine_methods=(),
              rl_params=dict(trajectories_per_update=4))
    assert captured["accept"] is None


# ---------------------------------------------------------------------------
# mining.env.BatchRewarder's own accept= keyword
# ---------------------------------------------------------------------------


def test_batch_rewarder_accept_defaults_to_none_and_refuses_nothing_extra(store):
    view = store.view(start=store.dates[5])
    target = Label("close", 1).compute(view)
    rew = BatchRewarder(CubeEngine(view), MsePool(capacity=4, target=target))
    assert rew.accept is None and rew.refused_accept == 0
    out = rew.rewards([parse("day_mean(Amt)")])
    assert out[0].reason == "added" and rew.refused_accept == 0


def test_batch_rewarder_accept_hook_refuses_before_the_engine_runs(store):
    """Mirrors tests/test_rl_miners.py::test_banned_structure_is_refused_before_
    the_engine_runs exactly, for the NEW accept= hook instead of banned_
    substructures: the engine must never even be called for a candidate the hook
    rejects, and the refusal gets its own counter and reason string, distinct
    from "banned substructure"."""
    view = store.view(start=store.dates[5])
    target = Label("close", 1).compute(view)
    pool = MsePool(capacity=4, target=target)

    def reject_log_amt(expr):
        return "O:log(F:Amt)" not in expr.key()

    rew = BatchRewarder(CubeEngine(view), pool, accept=reject_log_amt)
    expr = parse("day_mean(log(Amt))")
    calls = []
    rew.engine.evaluate = lambda exprs, **kw: calls.append(list(exprs)) or _fail()   # noqa: E731

    out = rew.rewards([expr])
    assert out[0].reason == "rejected by accept hook" and out[0].reward == rew.penalty
    assert rew.refused_accept == 1 and rew.refused_structure == 0
    assert calls == [], "a candidate the accept hook refuses must never reach the engine"

    # a candidate the SAME hook accepts must still reach the engine and the pool normally
    pool2 = MsePool(capacity=4, target=target)
    rew2 = BatchRewarder(CubeEngine(view), pool2, accept=reject_log_amt)
    ok = rew2.rewards([parse("day_mean(Amt)")])
    assert ok[0].reason == "added" and ok[0].added


def test_batch_rewarder_accept_and_banned_substructures_each_count_their_own_refusals(store):
    """Both checks live in the same `if` (design: `(banned) or (hook rejects)`) --
    pin that each still gets its OWN counter/reason rather than collapsing into
    one, and that a banned structure short-circuits before the hook is even
    asked (so a hook that raises on anything unexpected still works for a
    banned candidate)."""
    view = store.view(start=store.dates[5])
    target = Label("close", 1).compute(view)
    pool = MsePool(capacity=4, target=target)

    def explode(expr):
        raise AssertionError("the accept hook must not be consulted for an already-banned candidate")

    rew = BatchRewarder(CubeEngine(view), pool, accept=explode, banned_substructures=("O:log(F:Amt)",))
    out = rew.rewards([parse("day_mean(log(Amt))")])
    assert out[0].reason == "banned substructure"
    assert rew.refused_structure == 1 and rew.refused_accept == 0


# ---------------------------------------------------------------------------
# argparse-level pins: --wall-seconds / --plugin / --daily-fields / --fitness
# ---------------------------------------------------------------------------


def test_mine_parser_has_wall_seconds_plugin_daily_fields_and_no_fitness_choices():
    args = cli.build_parser().parse_args(
        ["mine", "--store", "x", "--fitness", "anything_goes", "--wall-seconds", "12.5",
         "--plugin", "a.b", "--plugin", "c.d", "--daily-fields", "/some/dir"])
    assert args.fitness == "anything_goes"      # no choices=: argparse itself never rejects this
    assert args.wall_seconds == 12.5
    assert args.plugin == ["a.b", "c.d"]
    assert args.daily_fields == "/some/dir"

    defaults = cli.build_parser().parse_args(["mine", "--store", "x"])
    assert defaults.wall_seconds == 0.0 and defaults.plugin == [] and defaults.daily_fields is None
    assert defaults.fitness == "ic"


def test_research_and_calculate_parsers_also_accept_plugin_and_daily_fields():
    for sub in ("research", "calculate"):
        args = cli.build_parser().parse_args([sub, "--store", "x", "--plugin", "m", "--daily-fields", "/d"])
        assert args.plugin == ["m"] and args.daily_fields == "/d"


# ---------------------------------------------------------------------------
# main(): plugin collection from --plugin / $ALPHACUBE_PLUGINS, for every
# subcommand including one (synth) that has no --plugin argument at all
# ---------------------------------------------------------------------------


def test_main_handles_synth_subcommand_with_no_plugin_attribute(tmp_path):
    """`synth`'s own subparser does not inherit `common`, so `args` here has no
    `.plugin` attribute at all -- main()'s plugin collection must use getattr's
    default rather than crash with AttributeError."""
    rc = cli.main(["synth", "--out", str(tmp_path / "cube"), "--symbols", "4", "--days", "10", "--minutes", "6"])
    assert rc == 0


def test_main_loads_plugins_from_env_var_when_no_plugin_flag_given(tmp_path, monkeypatch, capsys):
    mod_name = "_wp8_env_plugin_probe"
    _write_module(tmp_path, monkeypatch, mod_name, "LOADED = []\ndef register():\n    LOADED.append(1)\n")
    monkeypatch.setenv("ALPHACUBE_PLUGINS", mod_name)

    rc = cli.main(["synth", "--out", str(tmp_path / "cube2"), "--symbols", "4", "--days", "10", "--minutes", "6"])
    assert rc == 0
    assert f"plugins: {mod_name}" in capsys.readouterr().out

    plugin_mod = importlib.import_module(mod_name)
    assert plugin_mod.LOADED == [1]


def test_main_prefers_explicit_plugin_flag_over_the_env_var(tmp_path, monkeypatch, capsys):
    flag_mod, env_mod = "_wp8_flag_plugin_probe", "_wp8_env_plugin_probe_2"
    _write_module(tmp_path, monkeypatch, flag_mod, "LOADED = []\ndef register():\n    LOADED.append('flag')\n")
    _write_module(tmp_path, monkeypatch, env_mod, "LOADED = []\ndef register():\n    LOADED.append('env')\n")
    monkeypatch.setenv("ALPHACUBE_PLUGINS", env_mod)

    # cmd_mine itself goes on to fail opening a nonexistent store -- irrelevant
    # here, main() has already loaded plugins and printed which ones by the time
    # args.fn(args) raises.
    with pytest.raises(FileNotFoundError):
        cli.main(["mine", "--store", "/does/not/matter", "--plugin", flag_mod, "--fitness", "ic"])
    out = capsys.readouterr().out
    assert f"plugins: {flag_mod}" in out
    assert env_mod not in out
    assert importlib.import_module(flag_mod).LOADED == ["flag"]
    assert not importlib.import_module(env_mod).LOADED
