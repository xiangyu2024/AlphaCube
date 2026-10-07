"""WP4 -- Engine: the DAG leaf predicate (``engine/dag.py``), the three leaf
readers (``engine/engine.py``'s ``_leaf_panel``, ``_load_leaf_cube`` and
``_run_panel_round``'s ``operand()``, plus the per-``evaluate()`` ``_daily_t``
cache), and the two structural-key readers that would otherwise crash on a
``DailyFeature`` (``mining/clustering.py``'s ``skeleton_key``,
``mining/alphasage.py``'s ``_unroll``). See
docs/alphacube_steering_design.md, sections "The five engine insertion
points" / "Two structural-key readers that would crash", and work package
"### WP4".

DEVIATION from the design's own test-plan text: the bullet
"``skeleton_key(DailyFeature('FloatCap'))`` and ``cluster_population([
cs_rank(FloatCap), cs_rank(Close)])`` do not raise" cannot be built as
written. ``cs_rank`` is an ND-face operator (``needs_daily``), and
``expression.operand_ok``'s ``needs_daily`` branch requires
``operand.shape.degenerate`` UNCONDITIONALLY on strict/legacy (unlike the
``needs_full_m`` branch, it takes no ``strict`` parameter at all); a bare
``Close`` is FULL (not degenerate), so ``parse("cs_rank(Close)")`` raises
``ParseError`` regardless of how the tree is built (``parse()`` or
``make_node()``, any ``strict`` value) -- measured, and unrelated to WP4
(this predates the daily-leaf work entirely). Substituted with
``cs_rank(day_last(Close))``, which exercises exactly what the bullet was
for: a population mixing a ``DailyFeature``-rooted tree and a plain
``Feature``-rooted tree through ``skeleton_key``/``cluster_population``
without raising.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from alphacube.axes import MODE_D, MODE_P, MODE_S
from alphacube.engine.dag import Dag
from alphacube.engine.engine import CubeEngine
from alphacube.engine.schedule import plan_rounds
from alphacube.expression import DailyFeature
from alphacube.mining.alphasage import _unroll
from alphacube.mining.clustering import cluster_population, skeleton_key
from alphacube.mining.tokens import FeatureToken, Vocabulary
from alphacube.parser import parse


def _six_daily_roots():
    """The exact six roots /Users/wxy/Desktop/Partura/tmp/design/
    proto_daily_leaf.py measured (e1..e6 there), re-parsed fresh each call so
    no Expression instance is shared across engines. Together they touch all
    three leaf readers and the two round-structure traps named in the
    Overview: all six schedule as ``P[2] -> S[3] -> P[2]`` (pinned below)."""
    return [parse("cs_rank(FloatCap)"), parse("div(day_sum(Amt), FloatCap)"),
            parse("cs_demean(div(day_sum(Amt), FloatCap))"), parse("day_mean(mul(Ret, LimitWidth))"),
            parse("ts_delta(FloatCap, 5)"), parse("FloatCap")]


# ---------------------------------------------------------------------------
# Numeric correctness against hand-written numpy
# ---------------------------------------------------------------------------

def test_div_day_sum_amt_floatcap_matches_hand_written_numpy(daily_view, raw):
    """The turnover-factor example: div (elementwise) in MODE_P over
    day_sum(Amt) (MODE_S) and the bare daily leaf -- see
    test_round_structure_for_the_two_measured_examples for the schedule."""
    cubes, trad = raw
    e = parse("div(day_sum(Amt), FloatCap)")
    res = CubeEngine(daily_view).evaluate([e])
    cap = daily_view.daily("FloatCap").astype(np.float64)                      # (S, D), unmasked
    amt = cubes["Amt"].astype(np.float64)                                      # (S, D, M), already masked
    with np.errstate(invalid="ignore"):
        day_sum = np.where(np.isfinite(amt).any(axis=2), np.nansum(amt, axis=2), np.nan)
        want = np.where(trad, day_sum / np.where(trad, cap, np.nan), np.nan).T  # (D, S)
    assert np.array_equal(np.isfinite(want), np.isfinite(res.panels[0]))
    assert np.nanmax(np.abs(want - res.panels[0])) < 1e-6


def test_day_mean_mul_ret_limitwidth_matches_hand_written_numpy(daily_view, raw):
    """mul(Ret, LimitWidth) broadcasts the daily leaf's single value across
    all M minutes (apply_cube's elementwise short-circuit); day_mean then
    reduces it -- the mul/day_mean pair runs as one MODE_S round."""
    cubes, trad = raw
    e = parse("day_mean(mul(Ret, LimitWidth))")
    res = CubeEngine(daily_view).evaluate([e])
    lw = daily_view.daily("LimitWidth").astype(np.float64)   # (S, D), unmasked (derived, no store needed)
    ret = cubes["Ret"].astype(np.float64)                     # (S, D, M), already masked
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)       # "Mean of empty slice" on a fully-masked stock-day
        want = np.where(trad, np.nanmean(ret * lw[:, :, None], axis=2), np.nan).T
    assert np.array_equal(np.isfinite(want), np.isfinite(res.panels[0]))
    assert np.nanmax(np.abs(want - res.panels[0])) < 1e-6


# ---------------------------------------------------------------------------
# Chunk invariance -- the load-bearing test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [{}, {"chunk_symbols": 3}, {"chunk_symbols": 1},
                                {"chunk_days": 7}, {"chunk_days": 1}],
                         ids=["default", "chunk_symbols=3", "chunk_symbols=1", "chunk_days=7", "chunk_days=1"])
def test_chunk_invariance_over_six_daily_roots(daily_view, kw):
    """Mirrors test_engine.py::test_chunk_invariance_over_random_population's
    pattern: every one of the six roots must be bit-identical under any
    chunking, including the S -> P hand-off of the turnover factor."""
    exprs = _six_daily_roots()
    a = CubeEngine(daily_view, **kw).evaluate(exprs)
    b = CubeEngine(daily_view).evaluate(exprs)
    for e, x, y in zip(exprs, a.panels, b.panels):
        assert np.array_equal(x, y, equal_nan=True), (kw, e)


# ---------------------------------------------------------------------------
# Round structure: the two assertions that fail if _run_panel_round.operand
# is not patched (the branch an implementer is most likely to skip)
# ---------------------------------------------------------------------------

def test_round_structure_for_the_two_measured_examples(daily_view):
    pair_plan = CubeEngine(daily_view).plan([parse("cs_rank(FloatCap)"), parse("ts_delta(FloatCap, 5)")])
    assert pair_plan.rounds[0].mode == MODE_P

    div_plan = CubeEngine(daily_view).plan([parse("div(day_sum(Amt), FloatCap)")])
    assert [r.mode for r in div_plan.rounds] == [MODE_S, MODE_P]
    div_node = next(n for r in div_plan.rounds for n in r.nodes if n.key.startswith("O:div"))
    assert div_node.mode == MODE_P


def test_all_six_roots_together_plan_matches_the_design_measurement(daily_view):
    plan = CubeEngine(daily_view).plan(_six_daily_roots())
    assert [r.mode for r in plan.rounds] == [MODE_P, MODE_S, MODE_P]


# ---------------------------------------------------------------------------
# dag.is_leaf regression
# ---------------------------------------------------------------------------

def test_dag_is_leaf_regression_plan_rounds_does_not_raise():
    """Measured without the dag.py fix: plan_rounds raises ``AttributeError:
    'NoneType' object has no attribute 'face'`` inside schedule._p_eligible,
    because the leaf lands in dag.computed and assign_modes dereferences
    node.spec.face on None (a leaf's .spec is always None)."""
    dag = Dag([parse("cs_rank(FloatCap)")])
    plan_rounds(dag)                      # must not raise
    leaf = dag.by_key["DF:FloatCap"]
    assert leaf.is_leaf
    assert leaf not in dag.computed


def test_dag_treats_a_bare_daily_feature_root_as_a_leaf_too():
    dag = Dag([DailyFeature("FloatCap")])
    assert dag.roots[0].is_leaf
    assert dag.computed == []


# ---------------------------------------------------------------------------
# Bare DailyFeature as a root -- the _leaf_panel path
# ---------------------------------------------------------------------------

def test_bare_daily_feature_as_root_uses_leaf_panel(daily_view):
    res = CubeEngine(daily_view).evaluate([DailyFeature("FloatCap")])
    want = daily_view.daily("FloatCap").astype(np.float32)
    want = np.where(daily_view.tradable(), want, np.nan).T
    assert np.array_equal(want, res.panels[0], equal_nan=True)


def test_leaf_panel_returns_a_writable_copy_not_the_views_own_cache(daily_view):
    """_leaf_panel's docstring claim: mutating the returned panel must not
    corrupt view.daily()'s own cache, which store.py documents as staying
    UNMASKED (the engine, not the view, is responsible for masking)."""
    eng = CubeEngine(daily_view)
    p = eng._leaf_panel(DailyFeature("FloatCap"))
    assert np.isnan(p).any(), "the fixture has suspended/unlisted days, so masking should have fired"
    p[:] = -999.0
    raw_again = daily_view.daily("FloatCap")
    assert not np.any(raw_again == -999.0)
    assert not np.isnan(raw_again).any()


# ---------------------------------------------------------------------------
# Masking: a suspended stock-day is NaN on every leaf reader
# ---------------------------------------------------------------------------

def test_suspended_stock_day_is_nan_on_every_leaf_reader(daily_view):
    trad = daily_view.tradable()
    assert not trad.all(), "fixture must have some suspended/unlisted stock-days"
    exprs = [parse("FloatCap"),                        # _leaf_panel
            parse("cs_rank(FloatCap)"),                # _run_panel_round.operand
            parse("day_mean(mul(Ret, LimitWidth))")]   # _load_leaf_cube (cube chunk) -> panel hand-off
    res = CubeEngine(daily_view).evaluate(exprs)
    for e, p in zip(exprs, res.panels):
        assert np.array_equal(np.isnan(p), ~trad.T), e


def test_mask_policies_agree_with_daily_leaves(daily_view):
    """Extends test_engine.py::test_mask_policies_agree to a population of
    daily-leaf trees."""
    exprs = _six_daily_roots() + [parse("cross_mean(FloatCap, 3)"), parse("ts_delta(LogFloatCap, 5)")]
    a = CubeEngine(daily_view, chunk_symbols=3, chunk_days=7, mask_policy="all").evaluate(exprs, tolerant=True)
    b = CubeEngine(daily_view, chunk_symbols=3, chunk_days=7, mask_policy="boundaries").evaluate(exprs, tolerant=True)
    assert set(a.errors) == set(b.errors)
    for e, x, y in zip(exprs, a.panels, b.panels):
        if x is not None:
            assert np.array_equal(x, y, equal_nan=True), e


def test_float64_dtype_agrees_with_float32_to_float32_tolerance(daily_view):
    exprs = _six_daily_roots()
    r32 = CubeEngine(daily_view, dtype="float32").evaluate(exprs)
    r64 = CubeEngine(daily_view, dtype="float64").evaluate(exprs)
    for e, x, y in zip(exprs, r32.panels, r64.panels):
        assert np.allclose(x, y.astype(np.float32), equal_nan=True, rtol=1e-5, atol=1e-5), e


# ---------------------------------------------------------------------------
# _load_leaf_cube: shapes for both chunk modes, and the shape-assertion graft
# ---------------------------------------------------------------------------

def test_load_leaf_cube_returns_correct_shapes_for_both_modes(daily_view):
    eng = CubeEngine(daily_view, apply_mask=False)   # raw values, no masking, for a direct comparison
    trad_s = eng._tradable[0:3, :]
    trad_d = eng._tradable[:, 0:7]
    cube_s = eng._load_leaf_cube(DailyFeature("FloatCap"), MODE_S, 0, 3, trad_s)
    cube_d = eng._load_leaf_cube(DailyFeature("FloatCap"), MODE_D, 0, 7, trad_d)
    assert cube_s.shape == (3, daily_view.n_days, 1)
    assert cube_d.shape == (daily_view.n_symbols, 7, 1)
    np.testing.assert_allclose(cube_s[:, :, 0].numpy(), daily_view.daily_s("FloatCap", 0, 3))
    np.testing.assert_allclose(cube_d[:, :, 0].numpy(), daily_view.daily_d("FloatCap", 0, 7))


def test_load_leaf_cube_shape_assertion_names_both_shapes_on_a_mismatch(daily_view, monkeypatch):
    """The one graft in WP4's engine changes: a lo:hi slip in daily_s/daily_d
    would otherwise silently produce a wrong-shaped chunk; this makes it a
    diagnosable error instead."""
    fresh = daily_view.store.view()
    monkeypatch.setattr(fresh, "daily_s", lambda name, lo, hi: np.zeros((1, 1), dtype=np.float32))
    eng = CubeEngine(fresh)
    trad_s = eng._tradable[0:3, :]
    with pytest.raises(ValueError) as excinfo:
        eng._load_leaf_cube(DailyFeature("LimitWidth"), MODE_S, 0, 3, trad_s)
    msg = str(excinfo.value)
    assert "LimitWidth" in msg and "expected" in msg and "(1, 1)" in msg


# ---------------------------------------------------------------------------
# The _daily_t cache (_run_panel_round.operand)
# ---------------------------------------------------------------------------

def test_daily_t_cache_one_view_daily_call_per_evaluate_for_three_parents(daily_view, monkeypatch):
    """A tree where 3 parents consume the SAME daily leaf in one P round
    calls view.daily() exactly once. Counts INVOCATIONS, not cost: view.daily
    already caches on the view itself (WP3), which would mask the absence of
    this engine-level cache -- the point of _daily_t is to dedup the
    numpy->torch copy and the mask, not the (already free) resolve."""
    fresh = daily_view.store.view()
    calls = []
    real_daily = fresh.daily

    def counting(name):
        calls.append(name)
        return real_daily(name)

    monkeypatch.setattr(fresh, "daily", counting)
    tree = [parse("cs_rank(FloatCap)"), parse("cs_demean(FloatCap)"), parse("ts_delta(FloatCap, 3)")]
    plan = CubeEngine(fresh).plan(tree)
    assert [r.mode for r in plan.rounds] == [MODE_P]          # all 3 in ONE P round sharing the leaf

    eng = CubeEngine(fresh)
    eng.evaluate(tree)
    assert calls == ["FloatCap"]
    assert eng._daily_t == {}, "cleared by evaluate()'s existing finally -- a per-call cache only"


def test_daily_t_cache_is_keyed_per_field_name(daily_view, monkeypatch):
    fresh = daily_view.store.view()
    calls = []
    real_daily = fresh.daily
    monkeypatch.setattr(fresh, "daily", lambda name: (calls.append(name), real_daily(name))[1])
    tree = [parse("cs_rank(FloatCap)"), parse("cs_rank(LogFloatCap)"), parse("ts_delta(FloatCap, 3)")]
    CubeEngine(fresh).evaluate(tree)
    assert sorted(calls) == ["FloatCap", "LogFloatCap"]


def test_daily_t_cache_stays_empty_without_any_daily_leaf(view):
    """Backward compatibility, directly: a tree with no DailyFeature never
    touches the new cache at all."""
    eng = CubeEngine(view)
    eng.evaluate([parse("cs_rank(day_mean(Close))")])
    assert eng._daily_t == {}


# ---------------------------------------------------------------------------
# The two structural-key readers: mining/clustering.py, mining/alphasage.py
# ---------------------------------------------------------------------------

def test_skeleton_key_tags_a_daily_leaf_and_is_leaf_blind_within_it():
    assert skeleton_key(DailyFeature("FloatCap")) == "DF"
    assert skeleton_key(DailyFeature("FloatCap")) == skeleton_key(DailyFeature("LogFloatCap"))
    assert skeleton_key(DailyFeature("FloatCap")) != skeleton_key(parse("Close"))


def test_cluster_population_does_not_raise_on_a_mix_of_daily_and_minute_leaves():
    # see module docstring DEVIATION: cs_rank(Close) does not parse, substituted
    # with cs_rank(day_last(Close)) -- same point, a population mixing a
    # DailyFeature-rooted tree and a Feature-rooted one.
    pop = [parse("cs_rank(FloatCap)"), parse("cs_rank(day_last(Close))"), parse("div(day_sum(Amt), FloatCap)")]
    clusters = cluster_population(pop)                    # must not raise
    assert sorted(i for c in clusters for i in c) == [0, 1, 2]


def test_alphasage_unroll_raises_vocab_keyerror_not_attributeerror(registry):
    """Pre-WP10 the RL vocabulary carries no daily tokens, so this cannot yet
    succeed -- but it must fail with the vocabulary's OWN KeyError (naming
    the missing token), not an AttributeError on expr.spec.name (the measured
    crash this package fixes: DailyFeature.spec is None, so the pre-existing
    fall-through `vocab.index_of(OperatorToken(expr.spec.name))` dies on
    None.name one line before ever reaching the vocabulary)."""
    vocab = Vocabulary(registry)
    assert FeatureToken("FloatCap") not in vocab.token_to_index
    with pytest.raises(KeyError, match="FloatCap"):
        _unroll(DailyFeature("FloatCap"), vocab, [], [])


def test_alphasage_unroll_resolves_once_a_vocabulary_carries_the_token(registry):
    """Forward-looking: _unroll's own logic is correct given ANY vocabulary
    that does carry a FeatureToken for the field (however WP10 arranges for
    that), not only that it currently raises cleanly."""
    vocab = Vocabulary(registry, fields=list(registry.fields) + ["FloatCap"])
    nodes, edges = [], []
    idx = _unroll(DailyFeature("FloatCap"), vocab, nodes, edges)
    assert nodes[idx] == vocab.index_of(FeatureToken("FloatCap"))
    assert edges == []
