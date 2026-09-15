import copy
import random

import numpy as np
import pytest

from alphacube.expression import contains_lookahead, is_valid_root
from alphacube.mining.env import INVALID_EXPR_PENALTY, BatchRewarder, TokenEnv
from alphacube.mining.pool import MsePool
from alphacube.mining.tokens import (BEG, SEP, ConstantToken, ExpressionBuilder, ExpressionBuilderError,
                                     FeatureToken, OperatorToken, Vocabulary, WindowToken)
from alphacube.parser import parse


@pytest.fixture(scope="module")
def vocab(registry):
    return Vocabulary(registry, n_minutes=24)


def test_vocabulary_layout(vocab, registry):
    assert vocab.size == 2 + len(vocab.fields) + len(vocab.operators) + len(vocab.constants) + len(vocab.windows)
    assert 24 in vocab.windows and 241 in vocab.windows
    assert not any(s.lookahead for s in vocab.operators)
    assert vocab.index_of(SEP) == vocab.sep_index and vocab.index_to_token[vocab.beg_index] == BEG


def test_rpn_replay_matches_parser(vocab):
    b = ExpressionBuilder(vocab)
    b.push(BEG)
    for t in (FeatureToken("Close"), WindowToken(5), OperatorToken("in_mean_rolling"), OperatorToken("day_std"),
              WindowToken(5), OperatorToken("ts_mean"), OperatorToken("cs_rank")):
        assert vocab.index_of(t) in b.valid_next_tokens()
        b.push(t)
    assert vocab.sep_index in b.valid_next_tokens()
    b.push(SEP)
    assert b.build().key() == parse("cs_rank(ts_mean(day_std(in_mean_rolling(Close, 5)), 5))").key()


def test_grammar_rejections(vocab):
    def fresh(*tokens):
        b = ExpressionBuilder(vocab)
        b.push(BEG)
        for t in tokens:
            b.push(t)
        return b

    with pytest.raises(ExpressionBuilderError):          # ND op on a live minute axis
        fresh(FeatureToken("Close"), OperatorToken("cs_rank"))
    with pytest.raises(ExpressionBuilderError):          # in_* on collapsed
        fresh(FeatureToken("Close"), OperatorToken("day_mean"), WindowToken(5), OperatorToken("in_mean_rolling"))
    with pytest.raises(ExpressionBuilderError):          # window not in the operator's pool (ND windows only)
        fresh(FeatureToken("Close"), OperatorToken("day_mean"), WindowToken(241), OperatorToken("ts_mean"))
    with pytest.raises(ExpressionBuilderError):          # in_expand needs the real minute count
        fresh(FeatureToken("Close"), OperatorToken("day_mean"), WindowToken(5), OperatorToken("in_expand"))
    b = fresh(FeatureToken("Close"), OperatorToken("day_mean"), WindowToken(24), OperatorToken("in_expand"))
    assert b.stack[0].shape.degenerate and not b.stack[0].shape.collapsed
    with pytest.raises(ExpressionBuilderError):          # boolean root cannot terminate
        fresh(FeatureToken("Ret"), OperatorToken("day_mean"), OperatorToken("is_gt0"), SEP)
    with pytest.raises(ExpressionBuilderError):          # constant-only tree cannot terminate
        fresh(ConstantToken(1.0), OperatorToken("sigmoid"), SEP)
    with pytest.raises(ExpressionBuilderError):          # full-width root cannot terminate
        fresh(FeatureToken("Close"), SEP)


def test_mask_and_push_agree(vocab):
    rng = random.Random(1)
    checked = 0
    for _ in range(40):
        b = ExpressionBuilder(vocab)
        b.push(BEG)
        for _ in range(rng.randint(1, 8)):
            valid = b.valid_next_tokens()
            if not valid or b.is_complete():
                break
            b.push(vocab.index_to_token[rng.choice(valid)])
        if b.is_complete():
            continue
        valid = set(b.valid_next_tokens())
        for i in range(vocab.size):
            c = copy.deepcopy(b)
            try:
                c.push(vocab.index_to_token[i])
                pushed = True
            except ExpressionBuilderError:
                pushed = False
            assert pushed == (i in valid), (b.stack, vocab.index_to_token[i])
            checked += 1
    assert checked > 1000


def test_random_walks_terminate_only_on_valid_roots(vocab):
    rng = random.Random(2)
    done = 0
    for _ in range(300):
        b = ExpressionBuilder(vocab)
        b.push(BEG)
        for _ in range(30):
            valid = b.valid_next_tokens()
            idx = vocab.sep_index if (vocab.sep_index in valid and rng.random() < 0.5) else rng.choice(valid)
            b.push(vocab.index_to_token[idx])
            if b.is_complete():
                e = b.build()
                assert is_valid_root(e) and not contains_lookahead(e)
                assert parse(repr(e)).key() == e.key()
                done += 1
                break
    assert done > 40


def test_env_defers_reward_and_penalises_invalid(vocab, view):
    env = TokenEnv(vocab, max_len=12)
    r = env.reset()
    assert r.reward == 0.0 and not r.done and r.mask[vocab.index_of(FeatureToken("Close"))]
    r = env.step(vocab.index_of(FeatureToken("Close")))
    r = env.step(vocab.index_of(OperatorToken("day_mean")))
    r = env.step(vocab.sep_index)
    assert r.done and r.reward is None and repr(r.expr) == "day_mean(Close)"
    env.reset()
    r = env.step(vocab.sep_index)                       # SEP on an empty stack: immediate penalty
    assert r.done and r.reward == INVALID_EXPR_PENALTY and r.expr is None
    env.reset()
    for _ in range(12):
        r = env.step(vocab.index_of(FeatureToken("Close")))
        if r.done:
            break
    assert r.done and r.reward == INVALID_EXPR_PENALTY   # length overrun


def test_batch_rewarder(view):
    from alphacube.engine.engine import CubeEngine
    from alphacube.label import Label

    target = Label("close", 1).compute(view)
    pool = MsePool(capacity=3, target=target)
    rew = BatchRewarder(CubeEngine(view), pool)
    exprs = [parse("day_mean(Close)"), None, parse("day_std(Ret)"), parse("mul(day_mean(Close), 2)"), parse("add(1, 2)")]
    out = rew.rewards(exprs)
    assert out[1].reward == INVALID_EXPR_PENALTY and out[1].reason == "no expression"
    assert out[0].added and np.isfinite(out[0].reward) and out[0].reward == pytest.approx(pool.pool_ic(), abs=1e-9) or out[2].added
    assert not out[3].added and "Duplicate" in out[3].reason          # affine copy of member 0
    assert not out[4].added and "solo IC" in out[4].reason            # constant panel
    assert rew.n_evaluated == 4 and pool.size == 2
    again = rew.rewards([parse("day_mean(Close)")])                    # cached panel, no new evaluation
    assert rew.n_evaluated == 4 and not again[0].added
