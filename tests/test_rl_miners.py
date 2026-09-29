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


# --- RL 的奖励口径:和 GP 用同一个目标、同一套封禁 --------------------------
# 以前 RL 的奖励是 Pearson 池 IC,GP 从 gpu10 起用的是 |rank IC| - 0.14 x 换手。
# 在跨版本 2775 个因子上量过:|IC| 对扣费净夏普的 Spearman +0.14,这个组合式
# +0.89。口径不统一,两条搜索线挖出来的东西没法比,RL 还会一路挖高换手因子。

def _rewarder(setup, **kw):
    view, target, _ = setup
    pool = MsePool(capacity=4, target=target)
    return BatchRewarder(CubeEngine(view), pool, **kw), pool, target


def test_banned_structure_is_refused_before_the_engine_runs(setup):
    """一次引擎调用是几秒 GPU,而这些结构是前面十几轮已经挖穿的。"""
    from alphacube.parser import parse

    rew, _pool, _t = _rewarder(setup, banned_substructures=("O:log(F:Amt)",))
    expr = parse("day_mean(log(Amt))")
    calls = []
    rew.engine.evaluate = lambda exprs, **kw: calls.append(list(exprs)) or _fail()   # noqa: E731

    out = rew.rewards([expr])
    assert out[0].reason == "banned substructure" and out[0].reward == rew.penalty
    assert rew.refused_structure == 1
    assert calls == [], "被封的结构根本不该送进引擎"


def _fail():
    raise AssertionError("engine should not be called")


def test_turnover_is_charged_the_way_gp_charges_it(setup):
    """同样的 rank IC,换手高的那个奖励必须更低 —— 不然 RL 又会挖出 40~70% 换手。"""
    from alphacube.mining.scoring import PanelScorer

    from alphacube.metrics import ema_rows

    _rew, _pool, target = _rewarder(setup)
    rng = np.random.default_rng(0)
    D, S = target.shape
    churny = np.where(np.isfinite(target), target, 0.0) + 0.3 * rng.normal(size=(D, S))
    calm = ema_rows(churny, 5.0)          # 同一个信号,平滑过 —— 换手塌下来,信号还在

    free = PanelScorer(target, kind="rank_ic", turnover_penalty=0.0, min_coverage=0.0, min_scored_days=0.0)
    charged = PanelScorer(target, kind="rank_ic", turnover_penalty=0.14, min_coverage=0.0, min_scored_days=0.0)
    a, b = charged.score(churny), charged.score(calm)
    assert abs(a["fit"] - (abs(a["rank_ic"]) - 0.14 * a["turnover"])) < 1e-12   # 就是这个式子
    assert free.score(churny)["fit"] > a["fit"] + 0.1                          # 不收费时高 0.14x0.86
    assert b["turnover"] < a["turnover"] - 0.2                                 # 平滑之后换手塌了
    assert b["fit"] > a["fit"], "换手低的那个必须拿到更高的奖励"


def test_scorer_rejects_instead_of_scoring_low(setup):
    """覆盖率不够是拒绝(-inf -> 惩罚),不是给个小分数,也不该进池。"""
    from alphacube.mining.scoring import PanelScorer
    from alphacube.parser import parse

    rew, pool, target = _rewarder(setup)
    scorer = PanelScorer(target, kind="rank_ic", turnover_penalty=0.14, min_coverage=0.9)
    dead = np.full_like(target, np.nan)
    dead[:, :2] = 1.0                       # 只有两只股票有值
    assert scorer.score(dead)["fit"] == -np.inf

    class Reject(PanelScorer):              # 打分器说不行,奖励层必须照办
        def score(self, panel):
            out = super().score(panel)
            return {**out, "fit": -np.inf}

    rew.scorer = Reject(target, kind="rank_ic", turnover_penalty=0.14)
    out = rew.rewards([parse("day_mean(Close)")])
    assert out[0].reward == rew.penalty and "rejected" in out[0].reason
    assert pool.size == 0


def test_rewarder_cache_honours_the_byte_cap(setup):
    from alphacube.parser import parse

    rew, _pool, _t = _rewarder(setup, panel_cache_items=500, panel_cache_bytes=1)
    rew.rewards([parse("day_mean(Close)"), parse("day_max(High)"), parse("day_min(Low)")])
    assert len(rew._cache) <= 1                       # 1 字节的上限 = 最多留一条
    assert rew._cache_bytes == sum(p.nbytes for p in rew._cache.values())

    big, _pool2, _t2 = _rewarder(setup, panel_cache_items=500, panel_cache_bytes=0)
    big.rewards([parse("day_mean(Close)"), parse("day_max(High)"), parse("day_min(Low)")])
    assert len(big._cache) == 3                       # 0 = 不按字节封顶
