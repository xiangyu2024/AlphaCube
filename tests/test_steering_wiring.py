"""Branch-verification regressions: the five places the steering hooks were
built but never actually wired, or wired only halfway.

Every test here was written against a reproduction of the defect it pins, and
each one fails on the parent commit:

1. ``operator_logit_bias`` had no caller anywhere in the tree and no miner could
   take one, so a sweep pointing one steer config at a GP arm and an RL arm got a
   steered GP arm and an **unsteered** RL arm, with nothing in stdout or
   meta.json telling them apart (design "6. RL parity", WP10 step 5).
2. ``pipeline.py``'s lone ``Vocabulary(...)`` passed no ``daily_fields``, so no
   CLI or library route could put a ``DailyFeatureToken`` in an RL action space
   -- a run paid the daily store's changed contract hash and then mined a grammar
   that could not reach the data, while the GP arm of the same config did.
3. A steer config naming a daily field the view cannot resolve burnt one
   evaluated candidate per tree that drew it, silently: the engine evaluates
   ``tolerant=True``, so the exception became a ``None`` panel, ``-inf`` fitness
   and a dropped tree, and nothing anywhere named the field.
4. ``op_weight``'s mapping form and ``leaf_slots`` both key on ``spec.name``,
   but every nd-face kernel is registered TWICE (ND under its own name, NT under
   ``nt_<name>``), so naming one face steered half the kernel -- and *raised* the
   twin's draw rate, because the mass taken off the ND half was redistributed.
5. ``daily_fields`` / ``leaf_slots`` field names were not checked against the
   registry's daily namespace at all, so a one-letter typo silently poisoned
   every candidate that drew a daily leaf.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest
import torch

import alphacube.mining as mining_mod
from alphacube.expression import DailyFeature
from alphacube.label import Label
from alphacube.mining.alphaforge import AlphaForgeMiner
from alphacube.mining.alphagen import AlphaGenMiner
from alphacube.mining.alphaqcm import AlphaQCMMiner
from alphacube.mining.alphasage import AlphaSAGEMiner
from alphacube.mining.env import BatchRewarder
from alphacube.mining.generator import TypedTreeGenerator, nd_twin_base
from alphacube.mining.gp import GPMiner, unresolvable_daily_fields
from alphacube.mining.pool import MsePool
from alphacube.mining.tokens import DailyFeatureToken, OperatorToken, Vocabulary, operator_logit_bias
from alphacube.mining.warmstart import PPOWarmStartQCMMiner
from alphacube.pipeline import run_mining
from alphacube.plugins import SearchHooks
from alphacube.store import CubeStore

MINER_CLASSES = (AlphaGenMiner, AlphaQCMMiner, AlphaForgeMiner, AlphaSAGEMiner, PPOWarmStartQCMMiner)


@pytest.fixture
def mining_view(store):
    return store.view(start=store.dates[5])


@pytest.fixture
def mining_target(mining_view):
    return Label("close", 1).compute(mining_view)


@pytest.fixture
def rewarder(mining_view, mining_target):
    from alphacube.engine.engine import CubeEngine
    pool = MsePool(4, mining_target)
    return BatchRewarder(CubeEngine(mining_view), pool)


# ---------------------------------------------------------------------------
# 1. op_weight reaches the RL search
# ---------------------------------------------------------------------------


def test_every_token_miner_accepts_a_logit_bias_and_keeps_it(registry, rewarder):
    """All five miners, not just the one the sweep happens to use: before this,
    ``inspect.signature(AlphaGenMiner.__init__)`` had no bias/weight parameter at
    all, so there was no route from ``op_weight`` into any RL policy."""
    vocab = Vocabulary(registry, n_minutes=24)
    bias = operator_logit_bias(vocab, {"day_mean": 0.0})
    for cls in MINER_CLASSES:
        m = cls(vocab, rewarder, max_expr_len=10, logit_bias=bias)
        assert m.logit_bias is not None, cls.__name__
        assert m.logit_bias.shape == (vocab.size,), cls.__name__
        np.testing.assert_allclose(m.logit_bias.numpy(), bias.astype(np.float32), rtol=0, atol=0)
        assert cls(vocab, rewarder, max_expr_len=10).logit_bias is None, cls.__name__


def test_each_miner_hands_the_bias_to_the_component_that_masks(registry, rewarder):
    """The bias is only a steer if it reaches the object that applies the grammar
    mask -- each miner masks in a different place (the ActorCritic, the Masker, the
    GFlowNetSampler, ``_select_action``), so each forward is its own hole."""
    vocab = Vocabulary(registry, n_minutes=24)
    bias = operator_logit_bias(vocab, {"day_mean": 0.0})

    ppo = AlphaGenMiner(vocab, rewarder, max_expr_len=10, logit_bias=bias)
    assert ppo.policy.logit_bias is not None
    assert "logit_bias" in dict(ppo.policy.named_buffers())        # moves with .to(device) / state_dict

    forge = AlphaForgeMiner(vocab, rewarder, max_expr_len=10, logit_bias=bias)
    assert forge.masker.logit_bias is forge.logit_bias

    sage = AlphaSAGEMiner(vocab, rewarder, max_expr_len=10, logit_bias=bias)
    assert sage.sampler.logit_bias is sage.logit_bias


def test_actor_critic_adds_the_bias_before_the_mask_not_after(registry, rewarder):
    """Order matters: added BEFORE the mask, a weight-0 operator keeps its -1e9
    when the mask would have allowed it, and an operator the mask forbids is still
    -1e9 either way. Checked on the logits themselves, because this is what makes
    weight 0 behave as a mask instead of as a mild preference."""
    vocab = Vocabulary(registry, n_minutes=24)
    i = vocab.index_of(OperatorToken("day_mean"))
    bias = operator_logit_bias(vocab, {"day_mean": 0.0})
    plain = AlphaGenMiner(vocab, rewarder, max_expr_len=10, seed=3)
    steered = AlphaGenMiner(vocab, rewarder, max_expr_len=10, seed=3, logit_bias=bias)

    res = plain.env.reset()
    tokens = torch.from_numpy(res.state).unsqueeze(0)
    lengths = torch.tensor([res.length])
    allow_all = torch.ones(1, vocab.size, dtype=torch.bool)
    with torch.no_grad():
        lo_plain, _ = plain.policy(tokens, lengths, allow_all)
        lo_steer, _ = steered.policy(tokens, lengths, allow_all)
    # Same seed -> same weights, so the gap IS the bias.
    assert float(lo_steer[0, i] - lo_plain[0, i]) == pytest.approx(np.log(1e-9), abs=1e-3)
    probs = torch.softmax(lo_steer[0], dim=-1)
    assert float(probs[i]) < 1e-7                                  # an effective mask, not a nudge
    others = [j for j in range(vocab.size) if j != i]
    np.testing.assert_allclose((lo_steer[0, others] - lo_plain[0, others]).numpy(), 0.0, atol=1e-6)


def test_alphaqcm_epsilon_exploration_cannot_re_admit_a_zero_weight_operator(registry, rewarder):
    """QCM picks a uniformly random VALID action with probability epsilon (0.3 at
    the start of a run), which bypasses the scores the bias is added to -- so
    biasing only the greedy branch would leave a weight-0 operator drawn at
    epsilon/|valid| forever. Measured here at epsilon=1.0, where the unbiased
    control draws it and the steered one never does."""
    vocab = Vocabulary(registry, n_minutes=24)
    name = "day_mean"
    i = vocab.index_of(OperatorToken(name))
    bias = operator_logit_bias(vocab, {name: 0.0})
    mask = np.ones(vocab.size, dtype=bool)

    plain = AlphaQCMMiner(vocab, rewarder, max_expr_len=10, seed=1)
    steered = AlphaQCMMiner(vocab, rewarder, max_expr_len=10, seed=1, logit_bias=bias)
    res = plain.env.reset()
    n_plain = sum(plain._select_action(res.state, res.length, mask, 1.0) == i for _ in range(600))
    n_steer = sum(steered._select_action(res.state, res.length, mask, 1.0) == i for _ in range(600))
    assert n_plain > 0, "control never drew the operator; the test would be vacuous"
    assert n_steer == 0


def test_a_logit_bias_of_the_wrong_length_or_a_non_finite_one_is_refused(registry, rewarder):
    """A length-1 array would broadcast over every token silently, and one NaN
    poisons the whole row so ``Categorical(logits=...)`` fails later naming
    nothing. A vocabulary built with ``daily_fields`` is longer than one without,
    which is exactly the mismatch a reused bias hits."""
    vocab = Vocabulary(registry, n_minutes=24)
    wide = Vocabulary(registry, n_minutes=24, daily_fields=("FloatCap",))
    with pytest.raises(ValueError, match="one value per token"):
        AlphaGenMiner(vocab, rewarder, max_expr_len=10, logit_bias=np.zeros(1))
    with pytest.raises(ValueError, match="one value per token"):
        AlphaGenMiner(vocab, rewarder, max_expr_len=10, logit_bias=np.zeros(wide.size))
    bad = np.zeros(vocab.size)
    bad[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        AlphaGenMiner(vocab, rewarder, max_expr_len=10, logit_bias=bad)


def test_logit_bias_none_is_byte_identical_to_omitting_it(registry, rewarder):
    """The backward-compat half: ``logit_bias=None`` must not move a single draw,
    so every run without steering reproduces exactly what it did before the
    keyword existed."""
    vocab = Vocabulary(registry, n_minutes=24)

    def sampled(**kw):
        miner = AlphaSAGEMiner(vocab, rewarder, max_expr_len=12, seed=4, **kw)
        torch.manual_seed(4)           # sampling draws from the GLOBAL stream, so re-seed per run
        return [None if t.expr is None else t.expr.key() for t in (miner.sampler.sample() for _ in range(12))]

    a, b = sampled(), sampled(logit_bias=None)
    assert a == b
    assert any(k is not None for k in a)                           # non-vacuous
    # And a bias really does move it, so the comparison above is not trivially true.
    # A broad one: zeroing a single operator out of 214 often leaves the inverse-CDF
    # draw on the same token, which would make this assertion flaky rather than wrong.
    wide = operator_logit_bias(vocab, lambda spec: 0.0 if spec.name.startswith("in_") else 1.0)
    assert (wide != 0).sum() > 40
    assert sampled(logit_bias=wide) != a


def test_operator_logit_bias_takes_the_mapping_form_the_gp_side_takes(registry):
    """"One shared op_weight steers both searches" only holds if both sides accept
    the same shapes: the GP side resolves a ``{name: weight}`` mapping through
    ``_resolve_op_weight``, so this helper must too -- including that resolver's
    unknown-operator check and its ND/NT twin inheritance."""
    vocab = Vocabulary(registry, n_minutes=24)
    bias = operator_logit_bias(vocab, {"cs_mean": 0.0})
    assert bias[vocab.index_of(OperatorToken("cs_mean"))] == pytest.approx(np.log(1e-9))
    assert bias[vocab.index_of(OperatorToken("nt_cs_mean"))] == pytest.approx(np.log(1e-9))
    assert bias[vocab.index_of(OperatorToken("day_mean"))] == 0.0
    with pytest.raises(ValueError, match="unknown operator"):
        operator_logit_bias(vocab, {"not_a_real_operator_xyz": 0.0})


def test_pipeline_rl_branch_derives_the_logit_bias_from_hooks_op_weight(monkeypatch, store, capsys):
    """The wiring itself: ``run_mining`` must turn ``hooks.op_weight`` into the
    miner's ``logit_bias``, and must say so -- before this, a steered RL arm and
    an unsteered one were indistinguishable in stdout and in meta.json."""
    seen = {}
    real = mining_mod.AlphaSAGEMiner

    class _Recording(real):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            seen["bias"] = None if self.logit_bias is None else self.logit_bias.numpy().copy()
            seen["vocab"] = self.vocabulary

    monkeypatch.setattr(mining_mod, "AlphaSAGEMiner", _Recording)
    kw = dict(start=store.dates[5], label="close", miner="alphasage", budget=4, pool_capacity=2,
              max_expr_len=10, combine_methods=(), rl_params=dict(trajectories_per_update=4))

    run_mining(store, **kw, gp_params=dict(hooks=SearchHooks(op_weight={"day_mean": 0.0})))
    bias, vocab = seen["bias"], seen["vocab"]
    assert bias is not None, "hooks.op_weight never reached the RL miner"
    expected = operator_logit_bias(vocab, {"day_mean": 0.0})
    np.testing.assert_allclose(bias, expected.astype(np.float32), rtol=0, atol=0)
    assert "logit bias" in capsys.readouterr().out

    seen.clear()
    run_mining(store, **kw)                                        # no hooks: unchanged
    assert seen["bias"] is None


# ---------------------------------------------------------------------------
# 2. hooks.daily_fields reach an RL run
# ---------------------------------------------------------------------------


def test_pipeline_rl_branch_puts_hooks_daily_fields_in_the_vocabulary(monkeypatch, store, daily_root):
    """``pipeline.py``'s is the only ``Vocabulary(...)`` construction in the tree,
    and it passed only n_minutes / the two allow flags -- ``rl_params`` reach the
    miner class, not the vocabulary, so no route could populate ``daily_fields``
    and no daily leaf could appear in an RL run's output."""
    seen = {}
    real = mining_mod.Vocabulary

    class _Recording(real):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            seen["daily"] = [t.field for t in self.token_to_index if isinstance(t, DailyFeatureToken)]

    monkeypatch.setattr(mining_mod, "Vocabulary", _Recording)
    dstore = CubeStore(store.directory, daily_root=daily_root)
    kw = dict(start=store.dates[5], label="close", miner="alphasage", budget=4, pool_capacity=2,
              max_expr_len=10, combine_methods=(), rl_params=dict(trajectories_per_update=4))

    run_mining(dstore, **kw, gp_params=dict(hooks=SearchHooks(daily_fields=("FloatCap", "LogFloatCap"),
                                                             daily_leaf_prob=0.5)))
    assert seen["daily"] == ["FloatCap", "LogFloatCap"]

    seen.clear()
    run_mining(dstore, **kw)                                       # no hooks: no daily tokens, same indices
    assert seen["daily"] == []


def test_pipeline_rl_branch_rejects_a_mistyped_daily_field_name(store, daily_root):
    """``Vocabulary`` mints a ``DailyFeatureToken`` for any string handed to it, and
    the RL branch builds no generator -- so without this check the RL half would
    accept a name even the parser rejects, and defect 5's typo would come back on
    exactly the route defect 2 just opened."""
    dstore = CubeStore(store.directory, daily_root=daily_root)
    with pytest.raises(ValueError, match="unknown daily field"):
        run_mining(dstore, start=store.dates[5], label="close", miner="alphasage", budget=4,
                   pool_capacity=2, max_expr_len=10, combine_methods=(),
                   rl_params=dict(trajectories_per_update=4),
                   gp_params=dict(hooks=SearchHooks(daily_fields=("FlotCap",), daily_leaf_prob=0.5)))


# ---------------------------------------------------------------------------
# 3. a daily field this view cannot resolve is named, not swallowed
# ---------------------------------------------------------------------------


def test_unresolvable_daily_fields_covers_both_hook_fields(mining_view):
    assert unresolvable_daily_fields(None, mining_view) == ()
    assert unresolvable_daily_fields(SearchHooks(), mining_view) == ()
    # The three DERIVED fields need no store at all, so they are never flagged.
    assert unresolvable_daily_fields(SearchHooks(daily_fields=("LimitWidth",)), mining_view) == ()
    hooks = SearchHooks(daily_fields=("FloatCap",), leaf_slots={"cs_regres": {1: ("LogTotalCap",)}})
    assert unresolvable_daily_fields(hooks, mining_view) == ("FloatCap", "LogTotalCap")


def test_gpminer_warns_naming_a_daily_field_the_view_cannot_resolve(registry, mining_view, mining_target):
    """Measured on the pre-fix tree: stdout's only daily line was "daily fields:
    none attached", meta.json said nothing, the archive was non-empty and carried
    no such expression, and 5 of 70 evaluated candidates (7%) at
    ``daily_leaf_prob=0.35`` went on trees that cannot be evaluated at all."""
    hooks = SearchHooks(daily_fields=("FloatCap", "LogFloatCap"), daily_leaf_prob=0.35)
    with pytest.warns(RuntimeWarning, match="cannot resolve") as rec:
        miner = GPMiner(registry, mining_view, mining_target, MsePool(4, mining_target),
                        population_size=8, max_depth=3, seed=6, hooks=hooks)
    assert miner.unresolvable_daily_fields == ("FloatCap", "LogFloatCap")
    msg = str(rec[0].message)
    assert "FloatCap" in msg and "LogFloatCap" in msg
    assert "LimitWidth" in msg, "the warning must also say what IS resolvable here"


def test_gpminer_is_silent_when_every_named_daily_field_resolves(registry, daily_view, recwarn):
    """The other half, so the warning cannot be a blanket one: with the store
    attached the same hooks are clean, and the attribute is empty."""
    view = daily_view
    target = Label("close", 1).compute(view)
    miner = GPMiner(registry, view, target, MsePool(4, target), population_size=8, max_depth=3, seed=6,
                    hooks=SearchHooks(daily_fields=("FloatCap", "LogFloatCap"), daily_leaf_prob=0.35))
    assert miner.unresolvable_daily_fields == ()
    assert not [w for w in recwarn if "cannot resolve" in str(w.message)]
    plain = GPMiner(registry, view, target, MsePool(4, target), population_size=8, max_depth=3, seed=6)
    assert plain.unresolvable_daily_fields == ()                   # hooks omitted entirely


def test_pipeline_says_and_records_the_unresolvable_daily_fields(tmp_path, store):
    """Run level, both branches: the say() line is what a sweep's log shows, and
    meta.json is what lets an archived arm be told apart from a steered one
    afterwards."""
    hooks = SearchHooks(daily_fields=("FloatCap",), daily_leaf_prob=0.3)
    lines = []
    run_mining(store, start=store.dates[5], label="close", miner="gp", budget=12, pool_capacity=2,
               combine_methods=(), out_root=str(tmp_path / "runs"), run_id="r1", log=lines.append,
               gp_params=dict(population_size=6, max_depth=3, hooks=hooks))
    said = [ln for ln in lines if "cannot resolve" in ln]
    assert len(said) == 1 and "FloatCap" in said[0]
    meta = json.loads((tmp_path / "runs" / "r1" / "meta.json").read_text(encoding="utf-8"))
    assert meta["unresolvable_daily_fields"] == ["FloatCap"]

    lines.clear()
    run_mining(store, start=store.dates[5], label="close", miner="gp", budget=12, pool_capacity=2,
               combine_methods=(), out_root=str(tmp_path / "runs2"), run_id="r2", log=lines.append,
               gp_params=dict(population_size=6, max_depth=3))
    assert not [ln for ln in lines if "cannot resolve" in ln]
    meta2 = json.loads((tmp_path / "runs2" / "r2" / "meta.json").read_text(encoding="utf-8"))
    assert meta2["unresolvable_daily_fields"] == []


# ---------------------------------------------------------------------------
# 4. the ND/NT twin a half-written steer config used to miss
# ---------------------------------------------------------------------------


def test_nd_twin_base_only_maps_the_nt_face(registry):
    assert nd_twin_base(registry.get("nt_cs_mean")) == "cs_mean"
    assert nd_twin_base(registry.get("cs_mean")) is None           # the ND face itself
    assert nd_twin_base(registry.get("day_mean")) is None          # dt face: no twin exists


def _operator_counts(registry, op_weight, seeds=40, n=12):
    c: Counter = Counter()
    for s in range(seeds):
        gen = TypedTreeGenerator(registry, n_minutes=24, seed=s, op_weight=op_weight)
        for e in gen.generate_population(n):
            for node in e.nodes():
                spec = getattr(node, "spec", None)
                if spec is not None:
                    c[spec.name] += 1
    return c


def test_op_weight_zero_on_an_nd_name_also_zeroes_its_nt_twin(registry):
    """Weight 0 is defined as "take this operator out of the vocabulary", and an
    nd-face kernel has two names, so a mapping naming one of them must take both.
    Measured on the pre-fix tree over 400 seeds x generate_population(25),
    ``{"cs_mean": 0.0}`` took ``cs_mean`` 148 -> 0 and left ``nt_cs_mean`` at
    17 -> 17: the same kernel, half of it still being drawn.

    A CALLABLE op_weight is the caller's own business and is handed through
    untouched -- which is also how the pre-fix behaviour is reproduced here, so
    this test pins the mapping form against the exact thing it used to do."""
    base = _operator_counts(registry, None)
    assert base["cs_mean"] > 0 and base["nt_cs_mean"] > 0, "baseline draws both faces; test is non-vacuous"
    nd_only = _operator_counts(registry, lambda s: 0.0 if s.name == "cs_mean" else 1.0)
    assert nd_only["cs_mean"] == 0 and nd_only["nt_cs_mean"] > 0, "the callable form stays the caller's own"
    zeroed = _operator_counts(registry, {"cs_mean": 0.0})
    assert zeroed["cs_mean"] == 0
    assert zeroed["nt_cs_mean"] == 0


def test_an_explicit_nt_entry_still_overrides_the_inherited_one(registry):
    """Inheritance must not cost the ability to weight the two faces apart on
    purpose: an explicit ``nt_<name>`` entry always wins over the ND name's."""
    split = _operator_counts(registry, {"cs_mean": 0.0, "nt_cs_mean": 1.0})
    assert split["cs_mean"] == 0
    assert split["nt_cs_mean"] > 0


def test_leaf_slots_nd_key_also_covers_the_nt_twin_and_an_empty_entry_opts_out(registry):
    """``leaf_slots`` reads ``spec.name`` in the same way, and had the same half
    effect (measured: 1 of 40 seeds left ``nt_cs_regres``'s slot 1 unfilled; 0 once
    both keys were written). Resolved at LOOKUP time, so the mapping stays the
    caller's own object and a plugin mutating it between blocks still steers."""
    slots = {"cs_regres": {1: ("FloatCap",)}}
    gen = TypedTreeGenerator(registry, n_minutes=24, seed=5, daily_fields=("FloatCap",), leaf_slots=slots)
    assert gen.leaf_slots is slots                                 # not a normalised copy
    assert gen._slots_for(registry.get("cs_regres")) == {1: ("FloatCap",)}
    assert gen._slots_for(registry.get("nt_cs_regres")) == {1: ("FloatCap",)}
    assert gen._slots_for(registry.get("day_mean")) is None

    explicit = {"cs_regres": {1: ("FloatCap",)}, "nt_cs_regres": {0: ("LimitWidth",)}}
    gen2 = TypedTreeGenerator(registry, n_minutes=24, seed=5, daily_fields=("FloatCap", "LimitWidth"),
                              leaf_slots=explicit)
    assert gen2._slots_for(registry.get("nt_cs_regres")) == {0: ("LimitWidth",)}

    opted_out = {"cs_regres": {1: ("FloatCap",)}, "nt_cs_regres": {}}
    gen3 = TypedTreeGenerator(registry, n_minutes=24, seed=5, daily_fields=("FloatCap",), leaf_slots=opted_out)
    assert gen3._slots_for(registry.get("nt_cs_regres")) == {}     # the documented opt-out
    spec = registry.get("cs_regres")
    tree = gen3._build(spec, "daily", 3, False)
    assert isinstance(tree.operands[1], DailyFeature)              # the ND face is unaffected


# ---------------------------------------------------------------------------
# 5. a mistyped daily field name fails at construction, not silently at use
# ---------------------------------------------------------------------------


def test_generator_rejects_an_unknown_daily_field_name(registry):
    """Pre-fix: ``daily_fields=("NoSuchField",), daily_leaf_prob=1.0`` constructed
    fine and put the leaf in 34 of 40 trees; ``DailyFeature``'s shape looks at the
    node type, not the field name, so the typo survived the shape system and
    ``is_valid_root`` and only surfaced inside ``dailyfields.resolve`` at
    evaluation time -- where ``GPMiner._panels``' ``tolerant=True`` filed it under
    ``res.errors`` instead of raising. Symmetric with the existing ``op_weight``
    check."""
    with pytest.raises(ValueError, match="unknown daily field") as exc:
        TypedTreeGenerator(registry, n_minutes=24, seed=1, daily_fields=("NoSuchField",), daily_leaf_prob=1.0)
    msg = str(exc.value)
    assert "NoSuchField" in msg
    assert "FloatCap" in msg, "the error must list what IS available"
    assert "--daily-fields" in msg, "a legal name still needs data; the message must say so"

    with pytest.raises(ValueError, match="unknown daily field"):
        TypedTreeGenerator(registry, n_minutes=24, seed=1, leaf_slots={"cs_regres": {1: ("NoSuchField",)}})
    # The near-miss the design's must_fix #2 is about: one letter.
    with pytest.raises(ValueError, match="FlotCap"):
        TypedTreeGenerator(registry, n_minutes=24, seed=1, daily_fields=("FlotCap",))


def test_every_real_daily_field_name_still_constructs(registry):
    """The check must not be tighter than the namespace: all 9 names, in both
    keywords at once, and the empty default."""
    gen = TypedTreeGenerator(registry, n_minutes=24, seed=1, daily_fields=tuple(registry.daily_fields),
                             leaf_slots={"cs_regres": {1: tuple(registry.daily_fields)}})
    assert gen.daily_fields == list(registry.daily_fields)
    assert TypedTreeGenerator(registry, n_minutes=24, seed=1).daily_fields == []
