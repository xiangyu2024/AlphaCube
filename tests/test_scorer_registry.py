"""WP6 -- the scorer factory registry (register_scorer / scorer_kinds / SCORER_FACTORIES)
and make_scorer's dates/warmup passthrough. See docs/alphacube_steering_design.md,
section "1. Scorer registry" under "乙 Steering hooks", and work package "### WP6".
"""

import numpy as np
import pytest
import torch

from alphacube.mining.scoring import (FITNESS_KINDS, SCORER_FACTORIES, LongHitScorer, LongScorer, PanelScorer,
                                      make_scorer, register_scorer, scorer_kinds)


def _target(D=24, S=18, seed=0):
    return np.random.default_rng(seed).normal(size=(D, S))


class _DummyScorerA:
    """Minimal stand-in satisfying the scorer protocol from register_scorer's docstring --
    just enough to check that make_scorer forwards dates/warmup (and everything else) to a
    registered factory, without doing any real scoring."""

    def __init__(self, target, *, dates=None, warmup=0, **kw):
        self.target = target
        self.dates = dates
        self.warmup = warmup
        self.kw = kw
        self.target_t64 = torch.zeros(1, dtype=torch.float64)
        self.target_ranks_t = torch.zeros(1, dtype=torch.float64)
        self.columns = ("dummy_a",)

    def score(self, panel, gated=True):
        return {"fit": 0.0}

    def score_combined(self, panel):
        return {"fit": 0.0}


class _DummyScorerB(_DummyScorerA):
    """A second, distinct factory -- only used to tell apart which one `replace` installed."""

    def __init__(self, target, *, dates=None, warmup=0, **kw):
        super().__init__(target, dates=dates, warmup=warmup, **kw)
        self.columns = ("dummy_b",)


@pytest.fixture(autouse=True)
def _clean_registry():
    """SCORER_FACTORIES is a module-level dict shared by every test (and, eventually, every
    plugin). Snapshot/restore around each test so nothing registered here leaks into another
    test file, regardless of collection order."""
    before = dict(SCORER_FACTORIES)
    try:
        yield
    finally:
        SCORER_FACTORIES.clear()
        SCORER_FACTORIES.update(before)


def test_fitness_kinds_untouched():
    # Pins backward_compat item 10's premise: WP6 must not touch FITNESS_KINDS itself.
    assert FITNESS_KINDS == ("ic", "rank_ic", "long", "long_hit")


def test_registry_is_empty_by_default():
    assert SCORER_FACTORIES == {}
    assert scorer_kinds() == FITNESS_KINDS


def test_builtin_kinds_return_same_class_and_ignore_dates_warmup():
    """With an empty registry, make_scorer for each built-in kind returns the same class with
    the same configured thresholds whether or not dates/warmup are passed."""
    target = _target()
    dates = [f"d{i:03d}" for i in range(target.shape[0])]
    common = dict(turnover_penalty=0.2, signed=True, cost_per_side=0.002, top_frac=0.08,
                  min_coverage=0.6, min_scored_days=0.4)
    for kind in FITNESS_KINDS:
        plain = make_scorer(target, kind, **common)
        dated = make_scorer(target, kind, dates=dates, warmup=5, **common)
        assert type(plain) is type(dated)
        if kind in ("ic", "rank_ic"):
            assert isinstance(plain, PanelScorer)
            attrs = ("kind", "turnover_penalty", "signed", "top_frac", "min_coverage", "min_scored_days")
        else:
            assert isinstance(plain, LongScorer if kind == "long" else LongHitScorer)
            attrs = ("cost_per_side", "top_frac", "side_frac", "min_coverage", "min_scored_days",
                     "min_distinct", "max_boundary_ties", "max_code_order", "unit", "weights", "scales")
        for attr in attrs:
            assert getattr(plain, attr) == getattr(dated, attr), attr


def test_unknown_kind_raises_the_same_error_as_panelscorer_directly():
    target = _target()
    with pytest.raises(ValueError) as from_factory:
        make_scorer(target, "bogus")
    with pytest.raises(ValueError) as direct:
        PanelScorer(target, kind="bogus")
    assert str(from_factory.value) == str(direct.value)
    assert "kind must be ic or rank_ic" in str(from_factory.value)


def test_registered_factory_receives_dates_and_warmup():
    target = _target()
    dates = [f"d{i:03d}" for i in range(target.shape[0])]
    register_scorer("dummy_wp6", _DummyScorerA)

    got = make_scorer(target, "dummy_wp6", dates=dates, warmup=5, cost_per_side=0.002)
    assert isinstance(got, _DummyScorerA)
    assert got.dates is dates
    assert got.warmup == 5
    assert dates[got.warmup] == "d005"            # dates[warmup] is the first evaluated day
    assert got.kw["cost_per_side"] == 0.002
    assert "dummy_wp6" in scorer_kinds()

    # the default (no dates/warmup passed at all) still reaches the factory, as None / 0
    default = make_scorer(target, "dummy_wp6")
    assert default.dates is None and default.warmup == 0


def test_scorer_kinds_sorts_registered_names_after_the_builtins():
    register_scorer("zz_wp6", _DummyScorerA)
    register_scorer("aa_wp6", _DummyScorerA)
    kinds = scorer_kinds()
    assert kinds[:len(FITNESS_KINDS)] == FITNESS_KINDS
    assert kinds[len(FITNESS_KINDS):] == ("aa_wp6", "zz_wp6")


def test_duplicate_registration_raises_unless_replace():
    register_scorer("dummy_wp6", _DummyScorerA)
    with pytest.raises(ValueError, match="already registered"):
        register_scorer("dummy_wp6", _DummyScorerB)
    assert isinstance(make_scorer(_target(), "dummy_wp6"), _DummyScorerA)   # unchanged

    register_scorer("dummy_wp6", _DummyScorerB, replace=True)
    assert isinstance(make_scorer(_target(), "dummy_wp6"), _DummyScorerB)  # now swapped


def test_registering_over_a_builtin_name_raises():
    for kind in FITNESS_KINDS:
        with pytest.raises(ValueError, match="built-in"):
            register_scorer(kind, _DummyScorerA)
        assert kind not in SCORER_FACTORIES


def test_builtin_protection_is_not_bypassed_by_replace():
    # replace=True only governs collisions with another REGISTERED kind, never a built-in one.
    with pytest.raises(ValueError, match="built-in"):
        register_scorer("rank_ic", _DummyScorerA, replace=True)


def test_new_names_are_exported():
    import alphacube.mining.scoring as scoring_mod
    for name in ("SCORER_FACTORIES", "register_scorer", "scorer_kinds"):
        assert name in scoring_mod.__all__
