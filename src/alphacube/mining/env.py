"""Token-by-token construction environment with **deferred, batched** rewards.

The previous project's environment called the pool at every completed
expression. On the cube a candidate costs a full engine pass, so the reward
is deferred: ``TokenEnv.step`` returns ``done=True, expr=<Expression>,
reward=None`` on SEP, the miner collects the pending expressions of a batch
(PPO's ``episodes_per_update``, QCM's episode batch, SAGE's trajectories per
update, Forge's decode batch), and ``BatchRewarder.rewards`` evaluates them in
ONE engine call (CSE across the batch, one pass per chunk phase) before
offering them to the pool in order. Numerically this is the same reward the
old immediate path produced -- only the timing changed. Invalid pushes and
length overruns still terminate immediately with the penalty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..engine.engine import CubeEngine
from ..expression import Expression
from ..metrics import cs_pearson, nanmean_quiet, sampled_abs_corr
from .pool import PoolBase
from .tokens import BEG, ExpressionBuilder, Vocabulary

INVALID_EXPR_PENALTY = -1.0


@dataclass
class StepResult:
    state: np.ndarray            # (max_len,) padded token indices
    length: int
    mask: np.ndarray             # (vocab_size,) bool
    reward: Optional[float]      # None = pending batch evaluation
    done: bool
    expr: Optional[Expression] = None


@dataclass
class RewardResult:
    reward: float
    ic: float = float("nan")
    added: bool = False
    novelty: float = float("nan")
    reason: str = ""


class TokenEnv:
    def __init__(self, vocabulary: Vocabulary, max_len: int = 20) -> None:
        self.vocabulary = vocabulary
        self.max_len = int(max_len)
        self._builder = ExpressionBuilder(vocabulary)
        self._tokens: List[int] = []
        self.reset()

    def reset(self) -> StepResult:
        self._builder = ExpressionBuilder(self.vocabulary)
        self._builder.push(BEG)
        self._tokens = [self.vocabulary.beg_index]
        return self._result(reward=0.0, done=False)

    @property
    def builder(self) -> ExpressionBuilder:
        return self._builder

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.vocabulary.size, dtype=bool)
        valid = self._builder.valid_next_tokens()
        if valid:
            mask[valid] = True
        return mask

    def _result(self, reward: Optional[float], done: bool, expr: Optional[Expression] = None) -> StepResult:
        state = np.zeros(self.max_len, dtype=np.int64)
        n = min(len(self._tokens), self.max_len)
        state[:n] = self._tokens[:n]
        return StepResult(state=state, length=max(n, 1), mask=self.action_mask(), reward=reward, done=done, expr=expr)

    def step(self, action: int) -> StepResult:
        token = self.vocabulary.index_to_token[int(action)]
        try:
            self._builder.push(token)
        except Exception:  # noqa: BLE001 - any grammar violation is a terminal penalty
            return self._result(reward=INVALID_EXPR_PENALTY, done=True)
        self._tokens.append(int(action))
        if self._builder.is_complete():
            return self._result(reward=None, done=True, expr=self._builder.build())
        if len(self._tokens) >= self.max_len:
            return self._result(reward=INVALID_EXPR_PENALTY, done=True)
        return self._result(reward=0.0, done=False)


class BatchRewarder:
    """One engine call per batch of candidates, then sequential pool insertion."""

    def __init__(self, engine: CubeEngine, pool: PoolBase, penalty: float = INVALID_EXPR_PENALTY,
                 with_novelty: bool = False, panel_cache_items: int = 256) -> None:
        self.engine = engine
        self.pool = pool
        self.penalty = float(penalty)
        self.with_novelty = bool(with_novelty)
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_items = int(panel_cache_items)
        self.n_evaluated = 0
        self.engine_seconds = 0.0

    def panels(self, exprs: Sequence[Expression]) -> List[Optional[np.ndarray]]:
        out: List[Optional[np.ndarray]] = [None] * len(exprs)
        todo: Dict[str, List[int]] = {}
        for i, e in enumerate(exprs):
            k = e.key()
            if k in self._cache:
                out[i] = self._cache[k]
            else:
                todo.setdefault(k, []).append(i)
        if todo:
            batch = [exprs[idx[0]] for idx in todo.values()]
            res = self.engine.evaluate(batch, tolerant=True)
            self.engine_seconds += res.seconds
            self.n_evaluated += len(batch)
            for (k, idx), p in zip(todo.items(), res.panels):
                if p is None:
                    continue
                self._cache[k] = p
                for i in idx:
                    out[i] = p
            while len(self._cache) > self._cache_items:
                self._cache.pop(next(iter(self._cache)))
        return out

    def rewards(self, exprs: Sequence[Optional[Expression]]) -> List[RewardResult]:
        live = [i for i, e in enumerate(exprs) if e is not None]
        panels = self.panels([exprs[i] for i in live]) if live else []
        results = [RewardResult(self.penalty, reason="no expression") for _ in exprs]
        for i, panel in zip(live, panels):
            expr = exprs[i]
            if panel is None:
                results[i] = RewardResult(self.penalty, reason="evaluation failed")
                continue
            p64 = panel.astype(np.float64)
            ic = nanmean_quiet(cs_pearson(p64, self.pool.target))
            if not np.isfinite(ic):
                results[i] = RewardResult(self.penalty, ic=ic, reason="solo IC not finite")
                continue
            novelty = float("nan")
            if self.with_novelty:
                from ..metrics import zscore_rows
                z = zscore_rows(p64)
                sims = [sampled_abs_corr(z, m.z) for m in self.pool.members]
                sims = [s for s in sims if np.isfinite(s)]
                novelty = 1.0 - max(sims) if sims else 1.0
            try:
                pool_ic = self.pool.try_add(expr, p64)
            except Exception as exc:  # noqa: BLE001 - rejected candidates are a penalty, pool unchanged
                results[i] = RewardResult(self.penalty, ic=ic, novelty=novelty,
                                          reason=f"{type(exc).__name__}: {str(exc)[:60]}")
                continue
            results[i] = RewardResult(float(pool_ic), ic=ic, added=True, novelty=novelty, reason="added")
        return results


__all__ = ["TokenEnv", "StepResult", "BatchRewarder", "RewardResult", "INVALID_EXPR_PENALTY"]
