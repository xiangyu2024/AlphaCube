"""Shared pieces of the token-sequence miners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ..expression import Expression
from .env import BatchRewarder, RewardResult, TokenEnv
from .tokens import Vocabulary

ProgressFn = Callable[[int, int, dict], None]     # (episodes_done, budget, info)


@dataclass
class MinerReport:
    episodes: int = 0
    completed: int = 0                 # episodes that reached SEP (a candidate expression)
    added: List[Expression] = field(default_factory=list)
    rejected: Dict[str, int] = field(default_factory=dict)
    best_reward: float = float("-inf")
    evaluated: int = 0
    engine_seconds: float = 0.0
    seconds: float = 0.0

    def note(self, r: RewardResult) -> None:
        if r.added:
            self.best_reward = max(self.best_reward, r.reward)
        else:
            self.rejected[r.reason] = self.rejected.get(r.reason, 0) + 1


class TokenMiner(ABC):
    """A miner that builds expressions token by token in ``TokenEnv`` and scores
    them through a ``BatchRewarder`` (deferred, batched pool rewards)."""

    def __init__(self, vocabulary: Vocabulary, rewarder: BatchRewarder, max_expr_len: int = 20) -> None:
        self.vocabulary = vocabulary
        self.rewarder = rewarder
        self.max_expr_len = int(max_expr_len)
        self.env = TokenEnv(vocabulary, max_expr_len)
        self.report = MinerReport()

    @property
    def pool(self):
        return self.rewarder.pool

    @abstractmethod
    def mine(self, budget: int, on_progress: Optional[ProgressFn] = None) -> MinerReport:
        """Run ``budget`` episodes; return the report (also kept on ``self.report``)."""

    def _finish(self, t0: float) -> MinerReport:
        import time
        self.report.evaluated = self.rewarder.n_evaluated
        self.report.engine_seconds = self.rewarder.engine_seconds
        self.report.seconds = time.perf_counter() - t0
        return self.report


__all__ = ["TokenMiner", "MinerReport", "ProgressFn"]
