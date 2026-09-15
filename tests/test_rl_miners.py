import numpy as np
import pytest

from alphacube.engine.engine import CubeEngine
from alphacube.expression import is_valid_root
from alphacube.label import Label
from alphacube.mining import (AlphaForgeMiner, AlphaGenMiner, AlphaQCMMiner, AlphaSAGEMiner, BatchRewarder,
                              MsePool, PPOWarmStartQCMMiner, Vocabulary)


@pytest.fixture(scope="module")
def setup(registry, store):
    view = store.view(start=store.dates[5])
    target = Label("close", 1).compute(view)
    return view, target, Vocabulary(registry, n_minutes=view.n_minutes)


def _run(cls, setup, budget=32, **kw):
    view, target, vocab = setup
    pool = MsePool(capacity=4, target=target)
    rew = BatchRewarder(CubeEngine(view), pool)
    miner = cls(vocab, rew, max_expr_len=14, seed=3, **kw)
    seen = []
    rep = miner.mine(budget, on_progress=lambda n, b, info: seen.append(n))
    assert rep.episodes == budget and seen and seen[-1] == budget
    assert rep.evaluated <= rep.completed
    assert all(is_valid_root(e) for e in rep.added)
    assert len(rep.added) >= pool.size or pool.size <= rep.completed
    return miner, pool, rep


def test_alphagen_backfills_rewards(setup):
    miner, pool, rep = _run(AlphaGenMiner, setup, episodes_per_update=8, record_transitions=True)
    terminal = [t for t in miner.recorded if t.done]
    assert terminal and all(np.isfinite(t.reward) for t in terminal)
    completed = [t for t in terminal if t.reward != -1.0]
    assert len(completed) == sum(1 for r in [rep] for _ in range(len(rep.added)))  # every added expr has a real reward
    assert pool.size >= 1


def test_alphaqcm_runs(setup):
    miner, pool, rep = _run(AlphaQCMMiner, setup, episodes_per_batch=8, min_buffer_before_update=8, batch_size=8)
    assert len(miner.buffer) > 0


def test_alphaforge_runs_and_combine_dynamic(setup):
    miner, pool, rep = _run(AlphaForgeMiner, setup, batch_size=8)
    if pool.size:
        dyn = miner.combine_dynamic(window=20)
        assert dyn.size == pool.size


def test_alphasage_novelty_reward(setup):
    miner, pool, rep = _run(AlphaSAGEMiner, setup, trajectories_per_update=8)
    assert miner.rewarder.with_novelty


def test_warmstart_seeds_replay(setup):
    miner, pool, rep = _run(PPOWarmStartQCMMiner, setup, budget=24, ppo_fraction=0.5,
                            qcm_kwargs=dict(episodes_per_batch=4, min_buffer_before_update=8, batch_size=8))
    s = miner.last_stats
    assert s.ppo_episodes == 12 and s.qcm_episodes == 12 and s.ppo_transitions_recorded > 0
    assert s.replay_buffer_seeded_with == min(s.ppo_transitions_recorded, 4000)
