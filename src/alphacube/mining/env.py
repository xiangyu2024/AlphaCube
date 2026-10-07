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
from typing import Callable, Dict, List, Optional, Sequence

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
                 min_solo: Optional[float] = None,
                 banned_substructures: Sequence[str] = (),
                 accept: Optional[Callable[[Expression], bool]] = None,
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
        # 进池的门槛:候选自己的适应度(已经扣了换手)低于这个数就不进池,奖励就是它自己的适应度。
        # 不设的话池子只按合成信号的 IC 收人 —— rl_long_1 的 30 个成员全是高换手因子,单独做多头都是亏的。
        self.min_solo = None if min_solo is None else float(min_solo)
        self.refused_solo = 0
        # 没进池的候选只在第一次出现时拿到自己的适应度,再出现就是惩罚。不这样的话"刚好不够格"的表达式是个
        # 可以反复领的稳定奖励(约 -0.02),而重复提交池内成员是 -1:rl_hit_1 的策略 800 局后就塌到一条表达式上,
        # 池子停在 17 个不动了。
        self._paid_outside: set = set()
        self.banned_substructures = frozenset(banned_substructures)
        # A plugin's reverse-guidance hook (plugins.SearchHooks.accept, forwarded by
        # pipeline.run_mining's RL branch from cfg.get("hooks")) -- the RL-side mirror
        # of TypedTreeGenerator.accept_hook (mining/generator.py) on the GP side. None
        # leaves the check in rewards() below off entirely, so a caller that never
        # passes this keyword sees exactly today's behaviour.
        self.accept = accept
        self.rank_ban = rank_ban
        self.refused_structure = 0
        self.refused_accept = 0
        self.refused_similar = 0
        self.exemplar_cache = None        # 由 pipeline 塞进来,见 rankban.ExemplarCache
        self._solo_scores: Dict[str, dict] = {}       # 表达式 key -> 打分器的结果(收敛的策略会反复吐同一条)
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

    def _outside(self, expr: Expression, solo: float, ic: float, reason: str) -> RewardResult:
        """没进池的候选:第一次给它自己的适应度(不低于半个惩罚),同一条再来就是惩罚。"""
        key = expr.key()
        if key in self._paid_outside:
            return RewardResult(self.penalty, ic=ic, reason=reason + " (repeat)")
        self._paid_outside.add(key)
        return RewardResult(max(float(solo), 0.5 * self.penalty), ic=ic, reason=reason)

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
            # 结构封禁和插件的 accept 钩子都在**评估之前**拦下来:一次引擎调用是几秒
            # GPU,而封禁结构是前面十几轮已经挖穿的(gp_full_market.sh 里累计 46 个),
            # accept 钩子则是插件的反向指引(plugins.SearchHooks.accept)—— 没必要为了
            # 一个钩子注定要拒的候选真的去跑引擎。两者各算各的计数器和理由,方便区分
            # 一次运行里到底是封禁结构挡得多,还是钩子挡得多。
            banned = bool(self.banned_substructures) and bool(one_layer_keys(e) & self.banned_substructures)
            if banned or (self.accept is not None and not self.accept(e)):
                if banned:
                    self.refused_structure += 1
                    results[i] = RewardResult(self.penalty, reason="banned substructure")
                else:
                    self.refused_accept += 1
                    results[i] = RewardResult(self.penalty, reason="rejected by accept hook")
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
                # GP 用的同一个目标(mining/scoring.py):|rank IC| - λ x 十分位多空书日换手,或纯多头口径。
                # 老口径(纯 Pearson IC)留给没传 scorer 的调用方。面板是表达式 key 的纯函数,分数按 key 缓存。
                key = expr.key()
                s = self._solo_scores.get(key)
                if s is None:
                    s = self._solo_scores[key] = self.scorer.score(p64)
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
            if self.scorer is not None and self.min_solo is not None and solo < self.min_solo:
                self.refused_solo += 1
                results[i] = self._outside(expr, solo, ic, "solo fitness below the pool's bar")
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
            if self.scorer is not None and self.min_solo is not None \
                    and not any(m.expr.key() == expr.key() for m in self.pool.members):
                # 池子满了、被挤掉的正是它自己:池子没变,不能当成"进了池"去领池子的奖励
                results[i] = self._outside(expr, solo, ic, "pruned on entry: pool unchanged")
                continue
            if self.reward_kind == "solo":
                reward = float(solo)
            elif self.scorer is not None:
                # 池的奖励也换成同一种货币:合成信号自己的适应度(老口径 |rank IC| - λ x 换手,或纯多头口径)。
                # 只有真的进了池才算这一次(一次排序约 0.5 s),被拒的走惩罚。
                reward = float(self.scorer.score_combined(self.pool.combined())["fit"])
                if not np.isfinite(reward):
                    # 合成信号一个能算的日子都没有(极端情形):退回候选自己的分数,
                    # 绝不能把 -inf 喂进 PPO / QCM 的更新 —— 一个就把优势函数整批变 NaN
                    reward = float(solo)
            else:
                reward = float(pool_ic)
            results[i] = RewardResult(reward, ic=ic, added=True, novelty=novelty, reason="added")
        return results


__all__ = ["TokenEnv", "StepResult", "BatchRewarder", "RewardResult", "INVALID_EXPR_PENALTY"]
