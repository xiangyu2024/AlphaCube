"""The default path writes the baseline's output, line for line and key for key.

``run_mining`` with no daily store attached and no search hooks must print the
log lines, and write the ``meta.json`` keys, that the pre-steering ``run_mining``
(commit 517fd09) printed and wrote. The identical-defaults promise is about
bytes, and a run's log and its ``meta.json`` are bytes a caller reads: a script
that greps the first log line, counts lines, or diffs two runs' ``meta.json``
files changes behaviour if an unconditional extra line or key shows up.

Three things used to break that:

1. ``daily fields: none attached`` as a new first log line on every run;
2. ``store_contract_hash`` as a new ``meta.json`` key on every run;
3. ``unresolvable_daily_fields`` as a new ``meta.json`` key on every run.

The first two were found by running both trees side by side
(``Partura/tmp/impl/verify_defaults``) and diffing the output directories. The
third came in afterwards, with the commit that wired the hooks into the RL
branch (e4b258a), so that first comparison could not have seen it; it was found
by reading that commit and is pinned here by the tests failing on it.

Each is now written only when there is something to say: the log line and the
hash key when a daily store IS attached, the unresolvable-fields key when the
run was GIVEN search hooks. Everything else here pins the other deliberate
behaviour change on this path, the RL branch validating ``banned_substructures``
like the GP branch always did: ``check_ban_key`` may only ever reject a key
that no search could produce, never one it could (see
``docs/steering_and_daily_fields.md`` section 7).
"""

from __future__ import annotations

import json
import random
import re

import numpy as np
import pytest

from alphacube.mining.env import TokenEnv
from alphacube.mining.generator import TypedTreeGenerator, one_layer_keys
from alphacube.mining.gp import check_ban_key
from alphacube.mining.tokens import Vocabulary
from alphacube.pipeline import run_mining
from alphacube.plugins import SearchHooks
from alphacube.store import CubeStore

# Read off run_mining's meta.json dict at 517fd09 (`git show 517fd09:src/alphacube/
# pipeline.py`), in that order. A literal on purpose: the point is to pin what the
# BASELINE wrote, not to restate whatever the current code writes.
BASELINE_META_KEYS = ["store", "start", "end", "dates", "symbols", "label", "miner", "pool", "budget", "seed",
                      "pool_metrics", "rejected", "seconds", "engine_seconds"]

# The shape of every log line the baseline's run_mining printed for the tiny
# configurations below (GP, and one RL miner; no universe, no seed library, no
# exemplars, no combination). A line that matches none of these is a NEW kind of
# line on the default path.
BASELINE_LOG_SHAPES = (
    r"store ",                       # the first line: the store, date range, label, tradable share
    r"\s+\d+/\d+\s",                 # a progress line: "   1/2  best ..." (GP) / "   8/16  pool ..." (RL)
    r"done ",                        # "done 1.2s (engine 0.4s): ..."
    r"\s+x\d+\s",                   # the tally of why candidates were rejected: "   x1    solo IC not finite"
    r"pool: ",                       # the pool's metrics
    r"\s+\[\s*\d+\] ",               # one pool member
    r"wrote ",                       # the run directory
)

MINERS = ("gp", "alphasage")


def _kwargs(miner: str, gp_params):
    """A tiny, fast run_mining configuration: GP, or one RL miner. The RL budget is
    the smallest that leaves a non-empty pool on the 10-symbol test store for every
    seed tried (0, 1, 2): an empty pool makes run_mining return before it writes a
    run directory -- the baseline does that too -- and there would be no meta.json."""
    if miner == "gp":
        return dict(budget=8, gp_params=dict(population_size=6, max_depth=3, **(gp_params or {})))
    return dict(budget=32, max_expr_len=12, rl_params=dict(trajectories_per_update=8), gp_params=gp_params)


def _mine(store, tmp_path, miner, *, source=None, gp_params=None, **kw):
    """One run that writes its run directory; returns (log lines, parsed meta.json)."""
    lines = []
    res = run_mining(store if source is None else source, start=store.dates[5], label="close", miner=miner,
                     pool_capacity=2, combine_methods=(), out_root=str(tmp_path / "runs"), run_id="r",
                     log=lines.append, **_kwargs(miner, gp_params), **kw)
    assert res.run_dir is not None, f"{miner}: empty pool, so nothing was written -- {lines[-3:]}"
    return lines, json.loads((res.run_dir / "meta.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1 + 2: no daily store attached -> the baseline's lines and the baseline's keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("miner", MINERS)
def test_default_run_prints_only_the_kinds_of_line_the_baseline_printed(store, tmp_path, miner):
    lines, _ = _mine(store, tmp_path, miner)
    assert lines[0].startswith("store "), lines[0]          # the baseline's first line, not a daily-fields one
    stray = [ln for ln in lines if not any(re.match(p, ln) for p in BASELINE_LOG_SHAPES)]
    assert not stray, stray
    assert not [ln for ln in lines if "daily field" in ln.lower()], lines


@pytest.mark.parametrize("miner", MINERS)
def test_default_run_meta_json_is_exactly_the_baseline_key_list_in_order(store, tmp_path, miner):
    _, meta = _mine(store, tmp_path, miner)
    assert list(meta) == BASELINE_META_KEYS


# ---------------------------------------------------------------------------
# a daily store attached -> the line and the hash come back, and nothing else moves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("miner", MINERS)
def test_attached_daily_store_is_named_first_and_its_hash_recorded(store, daily_root, tmp_path, miner):
    # Through the path form, so the new `daily_root=` keyword is what attaches it.
    lines, meta = _mine(store, tmp_path, miner, source=str(store.directory), daily_root=str(daily_root))
    attached = CubeStore(store.directory, daily_root=daily_root)
    plain = CubeStore(store.directory)

    assert lines[0].startswith("daily fields "), lines[0]
    assert str(attached.daily.root) in lines[0] and "FloatCap" in lines[0], lines[0]
    assert lines[1].startswith("store "), lines[1]            # the baseline's first line, one line later

    assert meta["store_contract_hash"] == attached.contract_hash
    assert meta["store_contract_hash"] != plain.contract_hash  # the daily-aware hash, not the minute store's
    assert list(meta)[:2] == ["store", "store_contract_hash"]
    assert [k for k in meta if k != "store_contract_hash"] == BASELINE_META_KEYS   # the ONLY addition


# ---------------------------------------------------------------------------
# 3: unresolvable_daily_fields follows the hooks, not the default path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("miner", MINERS)
def test_unresolvable_daily_fields_is_written_only_for_a_run_given_hooks(store, tmp_path, miner):
    with_key = BASELINE_META_KEYS[:-2] + ["unresolvable_daily_fields"] + BASELINE_META_KEYS[-2:]

    # no hooks at all: nothing to be unable to resolve, and no key
    _, unsteered = _mine(store, tmp_path / "a", miner)
    assert list(unsteered) == BASELINE_META_KEYS

    # hooks given, and nothing in them to resolve: the key is there, empty -- "checked, clean"
    _, clean = _mine(store, tmp_path / "b", miner, gp_params=dict(hooks=SearchHooks()))
    assert list(clean) == with_key and clean["unresolvable_daily_fields"] == []

    # hooks naming a field this view has no data for (no daily store attached): named
    _, steered = _mine(store, tmp_path / "c", miner,
                       gp_params=dict(hooks=SearchHooks(daily_fields=("FloatCap",), daily_leaf_prob=0.3)))
    assert list(steered) == with_key and steered["unresolvable_daily_fields"] == ["FloatCap"]


# ---------------------------------------------------------------------------
# the RL branch validates banned_substructures like the GP branch always did
# ---------------------------------------------------------------------------

# Every one of these is rejected by gp.check_ban_key, hence by GPMiner.__init__ at the
# baseline already; the RL branch used to hand them to BatchRewarder unchecked, where
# they could never equal a key one_layer_keys writes -- a ban that silently banned nothing.
NEVER_MATCHING_KEYS = (
    "not-a-valid-key",                  # not O:<op>(<leaves>)[|w<n>]
    "O:log(F:NotAField)",               # a minute field the registry does not have
    "O:no_such_op(F:Amt)",              # an operator the registry does not have
    "O:log(Amt)",                       # an operand that is not spelt as a leaf
    "O:cs_rank(DF:NotADailyField)",     # a daily field the registry does not have
)
# And these three are well-formed on both paths now. The last one is the one place the
# GP side gets MORE permissive: before the DF: branch it raised "operand 'DF:FloatCap'
# is not a leaf".
MATCHABLE_KEYS = ("O:log(F:Amt)", "O:ts_mean(F:Amt)|w3", "O:cs_rank(DF:FloatCap)")


def _bare_run(store, miner, key):
    return run_mining(store, start=store.dates[5], label="close", miner=miner, pool_capacity=2,
                      combine_methods=(), log=None,
                      **_kwargs(miner, dict(banned_substructures=(key,))))


@pytest.mark.parametrize("key", NEVER_MATCHING_KEYS)
def test_a_ban_key_that_could_never_match_raises_identically_on_gp_and_rl(store, key):
    messages = []
    for miner in MINERS:
        with pytest.raises(ValueError) as exc:
            _bare_run(store, miner, key)
        messages.append(str(exc.value))
    assert messages[0] == messages[1]            # one function, one message, both paths
    assert key in messages[0]                    # and it names the offending key


@pytest.mark.parametrize("key", MATCHABLE_KEYS)
def test_a_well_formed_ban_key_runs_on_both_paths(store, key):
    for miner in MINERS:
        _bare_run(store, miner, key)             # must not raise


def test_every_registry_operator_name_fits_the_ban_key_grammar(registry):
    """check_ban_key's operator-name class is [A-Za-z0-9_]+. One registered name outside
    it would make a key the search really emits unwritable as a ban."""
    bad = [s.name for s in registry.list_all() if not re.fullmatch(r"[A-Za-z0-9_]+", s.name)]
    assert not bad, bad


def test_check_ban_key_accepts_every_key_either_search_can_emit(registry):
    """The raise above is only safe if it can never hit a key that could have matched.
    Collect every depth-1 key a GP generator and a random legal walk through the RL
    grammar produce -- minute leaves, constants, windows and daily leaves -- and
    require check_ban_key to take all of them, against the registry's own field list
    (what pipeline.py's RL branch passes)."""
    daily = ("FloatCap", "LimitWidth")
    gp_keys, rl_keys = set(), set()
    for seed in range(3):
        gen = TypedTreeGenerator(registry, n_minutes=24, rng=random.Random(seed), daily_fields=daily,
                                 daily_leaf_prob=0.5)
        for _ in range(150):
            gp_keys |= one_layer_keys(gen.generate())

    env = TokenEnv(Vocabulary(registry, n_minutes=24, daily_fields=daily), 20)
    rng = np.random.default_rng(0)
    for _ in range(1000):
        step = env.reset()
        while not step.done and step.mask.any():
            step = env.step(int(rng.choice(np.flatnonzero(step.mask))))
        if step.expr is not None:
            rl_keys |= one_layer_keys(step.expr)

    keys = gp_keys | rl_keys
    assert len(gp_keys) > 100 and len(rl_keys) >= 10, (len(gp_keys), len(rl_keys))
    # not vacuous: every leaf spelling and the window suffix are among what was checked
    assert any("F:" in k for k in keys) and any("DF:" in k for k in keys)
    assert any("C:" in k for k in keys) and any("|w" in k for k in keys)

    rejected = []
    for k in sorted(keys):
        try:
            check_ban_key(k, registry, registry.fields)
        except ValueError as exc:
            rejected.append((k, str(exc)))
    assert not rejected, rejected[:5]
