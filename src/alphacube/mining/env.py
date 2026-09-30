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
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..engine.engine import CubeEngine
from ..expression import Expression
from ..metrics import cs_pearson, nanmean_quiet, sampled_abs_corr
from .generator import one_layer_keys
from .pool import PoolBase
from .rankban import RankSimilarityBan
from .scoring import PanelScorer
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
                 with_novelty: bool = False, panel_cache_items: int = 256,
                 panel_cache_bytes: Optional[int] = 8 * 1024 ** 3,
                 scorer: Optional["PanelScorer"] = None, reward_kind: str = "pool",
                 banned_substructures: Sequence[str] = (),
                 rank_ban: Optional["RankSimilarityBan"] = None) -> None:
        self.engine = engine
        self.pool = pool
        self.penalty = float(penalty)
        self.with_novelty = bool(with_novelty)
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._cache_items = int(panel_cache_items)
        # 条数上限不认识一条面板多大:生产立方体上 256 条就是 9.7 GB。GP 那边
        # 因为这个在 gpu14 上把容器撑到 90 GiB 边缘,见 gp.GPMiner._evict_panels。
        self._cache_bytes_cap = int(panel_cache_bytes) if panel_cache_bytes else 0
        self._cache_bytes = 0
        if reward_kind not in ("pool", "solo"):
            raise ValueError("reward_kind 只能是 pool 或 solo")
        self.scorer = scorer
        self.reward_kind = reward_kind
        self.banned_substructures = frozenset(banned_substructures)
        self.rank_ban = rank_ban
        self.refused_structure = 0
        self.refused_similar = 0
        self.exemplar_cache = None        # 由 pipeline 塞进来,见 rankban.ExemplarCache
        self.n_evaluated = 0
        self.engine_seconds = 0.0

    def panels(self, exprs: Sequence[Expression], cache: bool = True) -> List[Optional[np.ndarray]]:
        out: List[Optional[np.ndarray]] = [None] * len(exprs)
        todo: Dict[str, List[int]] = {}
        for i, e in enumerate(exprs):
            k = e.key()
            if k in self._cache:
                out[i] = self._cache[k]
                self._cache.move_to_end(k)
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
                if cache:
                    old = self._cache.pop(k, None)
                    if old is not None:
                        self._cache_bytes -= old.nbytes
                    self._cache[k] = p
                    self._cache_bytes += p.nbytes
                for i in idx:
                    out[i] = p
            while self._cache and (len(self._cache) > self._cache_items
                                   or (self._cache_bytes_cap and self._cache_bytes > self._cache_bytes_cap)):
                _k, old = self._cache.popitem(last=False)
                self._cache_bytes -= old.nbytes
        return out

    def add_exemplars(self, exprs: Sequence[Expression], batch: int = 150) -> int:
        """把这些表达式的面板存成 rank 封禁的样本,返回新增了几个。

        和 `gp.GPMiner._add_exemplars` 同一个做法:分批(一次全评估会把 GPU 撑爆,
        而 tolerant 引擎遇到 CUDA OOM 是**静默丢树**,丢的是封禁样本),而且不进
        面板缓存 —— 一个样本只用一次,用完就该走。
        """
        if self.rank_ban is None or not exprs:
            return 0
        added = 0
        exprs = list(exprs)
        cache = self.exemplar_cache
        for lo in range(0, len(exprs), max(1, int(batch))):
            part = exprs[lo:lo + batch]
            rows = [cache.get(repr(e)) if cache is not None else None for e in part]
            todo = [i for i, r in enumerate(rows) if r is None]
            if todo:
                for i, panel in zip(todo, self.panels([part[i] for i in todo], cache=False)):
                    if panel is None:
                        continue
                    rows[i] = self.rank_ban.index.rows(panel.astype(np.float64))
                    if cache is not None:
                        cache.put(repr(part[i]), rows[i])
            for e, r in zip(part, rows):
                if r is not None and self.rank_ban.add_rows(e, r):
                    added += 1
        if cache is not None:
            cache.prune()
        return added

    def memory_line(self) -> str:
        g = 1024.0 ** 3
        return (f"cache {self._cache_bytes / g:.1f}G/{len(self._cache)} "
                f"exem {(self.rank_ban.index.nbytes / g) if self.rank_ban is not None else 0.0:.2f}G/"
                f"{len(self.rank_ban) if self.rank_ban is not None else 0}")

    def rewards(self, exprs: Sequence[Optional[Expression]]) -> List[RewardResult]:
        results = [RewardResult(self.penalty, reason="no expression") for _ in exprs]
        live = []
        for i, e in enumerate(exprs):
            if e is None:
                continue
            # 结构封禁在**评估之前**拦下来:一次引擎调用是几秒 GPU,而这些结构
            # 是前面十几轮已经挖穿的(gp_full_market.sh 里累计 46 个)。
            if self.banned_substructures and (one_layer_keys(e) & self.banned_substructures):
                self.refused_structure += 1
                results[i] = RewardResult(self.penalty, reason="banned substructure")
                continue
            live.append(i)
        panels = self.panels([exprs[i] for i in live]) if live else []
        for i, panel in zip(live, panels):
            expr = exprs[i]
            if panel is None:
                results[i] = RewardResult(self.penalty, reason="evaluation failed")
                continue
            p64 = panel.astype(np.float64)
            if self.scorer is not None:
                # GP 用的同一个目标:|rank IC| - λ x 十分位多空书日换手。
                # 老口径(纯 Pearson IC)留给没传 scorer 的调用方。
                s = self.scorer.score(p64)
                ic, solo = s["ic"], s["fit"]
                if not np.isfinite(solo):
                    results[i] = RewardResult(self.penalty, ic=ic, reason="solo fitness rejected")
                    continue
            else:
                ic = nanmean_quiet(cs_pearson(p64, self.pool.target))
                solo = abs(ic) if np.isfinite(ic) else float("nan")
                if not np.isfinite(ic):
                    results[i] = RewardResult(self.penalty, ic=ic, reason="solo IC not finite")
                    continue
            if self.rank_ban is not None and len(self.rank_ban) and self.rank_ban.match(p64) is not None:
                self.refused_similar += 1
                results[i] = RewardResult(self.penalty, ic=ic, reason="ranks like a banned family")
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
            if self.reward_kind == "solo":
                reward = float(solo)
            elif self.scorer is not None:
                # 池的奖励也换成同一种货币:合成信号自己的 |rank IC| - λ x 换手。
                # 只有真的进了池才算这一次(一次排序约 0.5 s),被拒的走惩罚。
                reward = float(self.scorer.score(self.pool.combined())["fit"])
            else:
                reward = float(pool_ic)
            results[i] = RewardResult(reward, ic=ic, added=True, novelty=novelty, reason="added")
        return results


__all__ = ["TokenEnv", "StepResult", "BatchRewarder", "RewardResult", "INVALID_EXPR_PENALTY"]
