"""WP10: RL token parity for daily-feature leaves, plus the operator logit-bias
helper that lets a GP-side ``op_weight`` steer an RL policy too.

No cube/store fixtures needed -- `Vocabulary` and `ExpressionBuilder` are pure
functions of the registry, so every test here uses only the session-scoped
`registry` fixture (tests/conftest.py).
"""

import math

import numpy as np
import pytest

from alphacube.expression import DailyFeature, is_valid_root
from alphacube.mining.tokens import (BEG, SEP, ConstantToken, DailyFeatureToken, ExpressionBuilder,
                                     ExpressionBuilderError, FeatureToken, OperatorToken, Vocabulary,
                                     WindowToken, operator_logit_bias)


# ---------------------------------------------------------------------------
# Backward compatibility: empty daily_fields reproduces the pre-WP10 vocabulary
# ---------------------------------------------------------------------------

def test_empty_daily_fields_matches_a_hand_written_token_order(registry):
    """`Vocabulary(reg)` (no `daily_fields` argument at all, exactly every
    call site today) must enumerate tokens in precisely [BEG, SEP, Feature*,
    Operator*, Constant*, Window*] with nothing appended -- the order tokens.py
    had before WP10. `fields`/`constants`/`windows` are pinned to a small,
    literal list here (rather than the full 218-operator registry default) so
    the expected mapping below is actually hand-written, not a restatement of
    production code; `vocab.operators`'s own sort order is reused as-is since
    sorting by name predates WP10 and is not part of this package."""
    vocab = Vocabulary(registry, fields=["Close", "Ret"], constants=[0.0, 1.0], windows=[5, 10],
                       n_minutes=24)
    expected = [BEG, SEP, FeatureToken("Close"), FeatureToken("Ret")]
    expected += [OperatorToken(s.name) for s in vocab.operators]
    expected += [ConstantToken(0.0), ConstantToken(1.0)]
    expected += [WindowToken(5), WindowToken(10)]
    assert vocab.token_to_index == {t: i for i, t in enumerate(expected)}
    assert vocab.daily_fields == []
    assert not any(isinstance(t, DailyFeatureToken) for t in vocab.token_to_index)


def test_daily_fields_append_after_existing_tokens_without_moving_them(registry):
    """With real daily-field names (`registry.daily_fields`), the new tokens
    land strictly after every pre-existing one, and not a single existing
    token's index moves."""
    base = Vocabulary(registry, n_minutes=24)
    assert "FloatCap" in registry.daily_fields and "LimitWidth" in registry.daily_fields
    with_daily = Vocabulary(registry, n_minutes=24, daily_fields=("FloatCap", "LimitWidth"))

    assert with_daily.daily_fields == ["FloatCap", "LimitWidth"]
    assert with_daily.size == base.size + 2
    for token, idx in base.token_to_index.items():
        assert with_daily.token_to_index[token] == idx                   # no existing index moves
    assert with_daily.token_to_index[DailyFeatureToken("FloatCap")] == base.size
    assert with_daily.token_to_index[DailyFeatureToken("LimitWidth")] == base.size + 1
    assert all(not isinstance(t, DailyFeatureToken) for t in base.token_to_index)


# ---------------------------------------------------------------------------
# ExpressionBuilder: DailyFeatureToken -> DailyFeature
# ---------------------------------------------------------------------------

def test_expression_builder_builds_cs_rank_of_a_daily_feature(registry):
    vocab = Vocabulary(registry, n_minutes=24, daily_fields=("FloatCap",))
    b = ExpressionBuilder(vocab)
    b.push(BEG)
    assert vocab.index_of(DailyFeatureToken("FloatCap")) in b.valid_next_tokens()
    b.push(DailyFeatureToken("FloatCap"))
    assert isinstance(b.stack[0], DailyFeature) and b.stack[0].field == "FloatCap"
    assert vocab.index_of(OperatorToken("cs_rank")) in b.valid_next_tokens()
    b.push(OperatorToken("cs_rank"))
    assert vocab.sep_index in b.valid_next_tokens()
    b.push(SEP)
    assert b.is_complete()
    assert b.build().key() == "O:cs_rank(DF:FloatCap)"


def test_expression_builder_accepts_a_bare_daily_feature_as_a_root(registry):
    """A `DailyFeature` leaf is already COLLAPSED/degenerate (expression.py), so
    -- like a bare `day_mean(Close)` would be -- it is itself a valid factor
    root with no operator wrapped around it."""
    vocab = Vocabulary(registry, n_minutes=24, daily_fields=("FloatCap",))
    b = ExpressionBuilder(vocab)
    b.push(BEG)
    b.push(DailyFeatureToken("FloatCap"))
    assert is_valid_root(b.stack[0])
    assert vocab.sep_index in b.valid_next_tokens()
    b.push(SEP)
    assert b.is_complete() and b.build().key() == "DF:FloatCap"


def test_daily_feature_token_rejects_a_field_outside_the_vocabulary(registry):
    """"TotalCap" is a real daily field (`registry.daily_fields`), but this
    vocabulary only names "FloatCap" -- the same unknown-field guard
    `FeatureToken` already has."""
    vocab = Vocabulary(registry, n_minutes=24, daily_fields=("FloatCap",))
    assert "TotalCap" in registry.daily_fields
    b = ExpressionBuilder(vocab)
    b.push(BEG)
    with pytest.raises(ExpressionBuilderError):
        b.push(DailyFeatureToken("TotalCap"))


def test_valid_next_tokens_offers_daily_feature_tokens_only_when_configured(registry):
    empty_vocab = Vocabulary(registry, n_minutes=24)
    b_empty = ExpressionBuilder(empty_vocab)
    b_empty.push(BEG)
    assert not any(isinstance(empty_vocab.index_to_token[i], DailyFeatureToken)
                    for i in b_empty.valid_next_tokens())
    with pytest.raises(KeyError):
        empty_vocab.index_of(DailyFeatureToken("FloatCap"))

    vocab = Vocabulary(registry, n_minutes=24, daily_fields=("FloatCap",))
    b = ExpressionBuilder(vocab)
    b.push(BEG)
    assert vocab.index_of(DailyFeatureToken("FloatCap")) in b.valid_next_tokens()


# ---------------------------------------------------------------------------
# operator_logit_bias
# ---------------------------------------------------------------------------

def test_operator_logit_bias_is_all_zero_when_op_weight_is_none(registry):
    vocab = Vocabulary(registry, n_minutes=24)
    bias = operator_logit_bias(vocab, None)
    assert bias.shape == (vocab.size,)
    assert np.all(bias == 0.0)


def test_operator_logit_bias_masks_a_zeroed_operator(registry):
    """A zero `op_weight` on one operator gives that OperatorToken a
    `log(tiny)` bias and leaves every other token at 0.0; added to flat base
    logits, the zeroed operator never wins greedily and is (for all practical
    purposes) never drawn by a sampled policy either."""
    vocab = Vocabulary(registry, n_minutes=24)
    tiny = 1e-9

    def op_weight(spec):
        return 0.0 if spec.name == "day_mean" else 1.0

    bias = operator_logit_bias(vocab, op_weight, tiny=tiny)
    assert bias.shape == (vocab.size,)
    zeroed_idx = vocab.index_of(OperatorToken("day_mean"))
    other_idx = vocab.index_of(OperatorToken("day_sum"))
    assert bias[zeroed_idx] == pytest.approx(math.log(tiny))
    assert bias[other_idx] == pytest.approx(0.0)
    # every non-operator token is untouched
    non_op_tokens = (BEG, SEP, FeatureToken("Close"), ConstantToken(vocab.constants[0]),
                      WindowToken(vocab.windows[0]))
    for tok in non_op_tokens:
        assert bias[vocab.index_of(tok)] == 0.0

    # greedy: among operators, the zeroed one never has the top logit
    op_indices = [vocab.index_of(OperatorToken(s.name)) for s in vocab.operators]
    best = op_indices[int(np.argmax(bias[op_indices]))]
    assert best != zeroed_idx

    # sampled: softmax over just the two operators above makes the zeroed one
    # draw with probability ~tiny -- effectively never, in 20000 draws
    logits = np.array([bias[zeroed_idx], bias[other_idx]])
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    assert probs[0] < 1e-6
    samples = np.random.default_rng(0).choice(2, size=20000, p=probs)
    assert not np.any(samples == 0)


def test_operator_logit_bias_floors_a_negative_weight_at_tiny_same_as_zero(registry):
    vocab = Vocabulary(registry, n_minutes=24)
    tiny = 1e-9
    bias = operator_logit_bias(vocab, lambda spec: -5.0 if spec.name == "day_mean" else 1.0, tiny=tiny)
    assert bias[vocab.index_of(OperatorToken("day_mean"))] == pytest.approx(math.log(tiny))
