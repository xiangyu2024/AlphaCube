"""PPO-explore-then-QCM-refine (the user's design note, "PPO 序列初始化
Q-Learning"): run AlphaGen for a share of the budget with transition
recording on, seed AlphaQCM's replay buffer with those transitions (final
rewards included -- they are back-filled after each rewarded batch), then
let QCM refine on the SAME pool.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .alphagen import AlphaGenMiner
from .alphaqcm import AlphaQCMMiner
from .base import MinerReport, ProgressFn, TokenMiner
from .env import BatchRewarder
from .tokens import Vocabulary


@dataclass
class WarmStartStats:
    ppo_episodes: int = 0
    ppo_transitions_recorded: int = 0
    ppo_expressions_added: int = 0
    qcm_episodes: int = 0
    qcm_expressions_added: int = 0
    replay_buffer_seeded_with: int = 0


class PPOWarmStartQCMMiner(TokenMiner):
    def __init__(self, vocabulary: Vocabulary, rewarder: BatchRewarder, max_expr_len: int = 20,
                 ppo_fraction: float = 0.4, ppo_kwargs: Optional[Dict[str, Any]] = None,
                 qcm_kwargs: Optional[Dict[str, Any]] = None, seed: int = 0) -> None:
        super().__init__(vocabulary, rewarder, max_expr_len)
        if not (0.0 < ppo_fraction < 1.0):
            raise ValueError(f"ppo_fraction must be in (0, 1), got {ppo_fraction}")
        self.ppo_fraction = float(ppo_fraction)
        self.ppo_kwargs, self.qcm_kwargs, self.seed = dict(ppo_kwargs or {}), dict(qcm_kwargs or {}), seed
        self.last_stats: Optional[WarmStartStats] = None

    def mine(self, budget: int, on_progress: Optional[ProgressFn] = None) -> MinerReport:
        t0 = time.perf_counter()
        if budget < 2:
            raise ValueError("budget must be >= 2")
        ppo_budget = max(1, min(budget - 1, round(budget * self.ppo_fraction)))
        qcm_budget = budget - ppo_budget
        ppo = AlphaGenMiner(self.vocabulary, self.rewarder, self.max_expr_len, record_transitions=True,
                            seed=self.seed, **self.ppo_kwargs)
        ppo_rep = ppo.mine(ppo_budget, None if on_progress is None else
                           (lambda n, b, info: on_progress(n, budget, {**info, "stage": "ppo"})))
        qcm = AlphaQCMMiner(self.vocabulary, self.rewarder, self.max_expr_len, seed=self.seed, **self.qcm_kwargs)
        for t in ppo.recorded:
            qcm.buffer.push(t)
        qcm_rep = qcm.mine(qcm_budget, None if on_progress is None else
                           (lambda n, b, info: on_progress(ppo_budget + n, budget, {**info, "stage": "qcm"})))
        self.last_stats = WarmStartStats(ppo_budget, len(ppo.recorded), len(ppo_rep.added), qcm_budget,
                                         len(qcm_rep.added), min(len(ppo.recorded), qcm.buffer.capacity))
        self.report = MinerReport(episodes=budget, completed=ppo_rep.completed + qcm_rep.completed,
                                  added=ppo_rep.added + qcm_rep.added,
                                  best_reward=max(ppo_rep.best_reward, qcm_rep.best_reward))
        for rep in (ppo_rep, qcm_rep):
            for k, v in rep.rejected.items():
                self.report.rejected[k] = self.report.rejected.get(k, 0) + v
        return self._finish(t0)


__all__ = ["PPOWarmStartQCMMiner", "WarmStartStats"]
