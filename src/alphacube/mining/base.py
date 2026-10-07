"""Shared pieces of the token-sequence miners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

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

    def __init__(self, vocabulary: Vocabulary, rewarder: BatchRewarder, max_expr_len: int = 20,
                 logit_bias: Optional["np.ndarray"] = None) -> None:
        self.vocabulary = vocabulary
        self.rewarder = rewarder
        self.max_expr_len = int(max_expr_len)
        self.env = TokenEnv(vocabulary, max_expr_len)
        self.report = MinerReport()
        # The GP side's `op_weight`, in this search's own currency: one additive
        # per-token number, log(max(w, tiny)) at every OperatorToken index and 0
        # elsewhere (`tokens.operator_logit_bias`). Every miner adds it to its raw
        # policy scores BEFORE the grammar mask -- see `biased` -- so weight 0 is an
        # effective mask once softmax/argmax runs, and ONE op_weight steers the GP
        # and the RL search alike (design "6. RL parity", WP10 step 5).
        # None is the default and keeps every miner byte-identical to before this
        # existed: `biased` returns its argument unchanged, no tensor is allocated,
        # and no draw moves.
        self.logit_bias = self._check_logit_bias(logit_bias, vocabulary)

    @staticmethod
    def _check_logit_bias(logit_bias, vocabulary: Vocabulary) -> Optional["torch.Tensor"]:
        """``logit_bias`` must be one finite float per token of THIS vocabulary.

        Checked here rather than per use: a wrong length would otherwise broadcast
        silently (a length-1 array biases every token equally) or blow up deep
        inside a policy, and a single non-finite entry poisons the whole logit row,
        so ``Categorical(logits=...)`` would later fail naming nothing. A vocabulary
        built with ``daily_fields`` is longer than one without, which is exactly the
        mismatch this catches when a bias is reused across two runs."""
        if logit_bias is None:
            return None
        t = torch.as_tensor(np.asarray(logit_bias, dtype=np.float32))
        if t.shape != (vocabulary.size,):
            raise ValueError(f"logit_bias must hold one value per token ({vocabulary.size} in this "
                             f"vocabulary), got shape {tuple(t.shape)}")
        if not bool(torch.isfinite(t).all()):
            raise ValueError("logit_bias must be finite everywhere; a NaN/inf entry poisons every "
                             "token's probability (see tokens.operator_logit_bias, which rejects a "
                             "non-finite op_weight for the same reason)")
        return t

    def biased(self, scores: "torch.Tensor") -> "torch.Tensor":
        """``scores`` plus the steering bias, to be called on RAW policy scores
        BEFORE the grammar mask is applied. Returns ``scores`` itself when there is
        no bias, so the default path allocates nothing and is bit-identical.

        Broadcasts over any leading batch dimension: the bias is ``(vocab,)``."""
        return scores if self.logit_bias is None else scores + self.logit_bias

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
