"""AlphaSAGE: trajectory-balance GFlowNet whose state is the RGCN-encoded
partial expression tree read off ``ExpressionBuilder.stack``; reward =
exp(clip(pool IC)) x novelty, novelty = 1 - max |corr| against current pool
members (value-based, the paper's KNN structural term is omitted). The batch
of ``trajectories_per_update`` expressions is rewarded in one engine call.
Backward policy fixed to uniform (the RPN state graph is a tree).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ..expression import Binary, Constant, DailyFeature, Expression, Feature, PairRolling, Rolling, Unary
from .base import MinerReport, ProgressFn, TokenMiner
from .env import BatchRewarder
from .tokens import (BEG, ConstantToken, DailyFeatureToken, ExpressionBuilder, FeatureToken, OperatorToken,
                     Vocabulary)

RELATIONS = ["unary_arg", "binary_lhs", "binary_rhs", "rolling_operand", "pair_rolling_lhs", "pair_rolling_rhs"]
REL = {r: i for i, r in enumerate(RELATIONS)}


@dataclass
class PartialTreeGraph:
    node_token_idx: List[int] = field(default_factory=list)
    edges: List[Tuple[int, int, int]] = field(default_factory=list)
    root_ids: List[int] = field(default_factory=list)
    n_pending_windows: int = 0


def _unroll(expr: Expression, vocab: Vocabulary, nodes: List[int], edges: List[Tuple[int, int, int]]) -> int:
    if isinstance(expr, Feature):
        nodes.append(vocab.index_of(FeatureToken(expr.field)))
        return len(nodes) - 1
    if isinstance(expr, DailyFeature):
        # Its OWN token family: WP10 gave the daily namespace `DailyFeatureToken`,
        # parallel to `fields`/`FeatureToken`, so a vocabulary built with
        # `daily_fields=(...)` carries DailyFeatureToken('FloatCap') and never a
        # FeatureToken for the same name. Looking up the minute token here is what
        # the builder's own inverse (tokens.ExpressionBuilder.push) would never do,
        # and would KeyError on every WP10 vocabulary -- `GFlowNetSampler.sample`
        # re-reads the whole stack through this function at every step, so the step
        # after the policy first draws a daily leaf would die.
        nodes.append(vocab.index_of(DailyFeatureToken(expr.field)))
        return len(nodes) - 1
    if isinstance(expr, Constant):
        try:
            idx = vocab.index_of(ConstantToken(expr.value))
        except KeyError:
            idx = vocab.index_of(ConstantToken(min(vocab.constants, key=lambda c: abs(c - expr.value))))
        nodes.append(idx)
        return len(nodes) - 1
    nodes.append(vocab.index_of(OperatorToken(expr.spec.name)))
    me = len(nodes) - 1
    if isinstance(expr, Unary):
        edges.append((me, _unroll(expr.operands[0], vocab, nodes, edges), REL["unary_arg"]))
    elif isinstance(expr, Rolling):
        edges.append((me, _unroll(expr.operands[0], vocab, nodes, edges), REL["rolling_operand"]))
    elif isinstance(expr, Binary):
        edges.append((me, _unroll(expr.operands[0], vocab, nodes, edges), REL["binary_lhs"]))
        edges.append((me, _unroll(expr.operands[1], vocab, nodes, edges), REL["binary_rhs"]))
    elif isinstance(expr, PairRolling):
        edges.append((me, _unroll(expr.operands[0], vocab, nodes, edges), REL["pair_rolling_lhs"]))
        edges.append((me, _unroll(expr.operands[1], vocab, nodes, edges), REL["pair_rolling_rhs"]))
    else:
        raise TypeError(f"unsupported node {type(expr).__name__}")
    return me


def build_partial_tree_graph(stack, vocab: Vocabulary) -> PartialTreeGraph:
    g = PartialTreeGraph()
    for item in stack:
        if isinstance(item, Expression):
            g.root_ids.append(_unroll(item, vocab, g.node_token_idx, g.edges))
        else:
            g.n_pending_windows += 1
    return g


class RGCNEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 64, n_relations: int = len(RELATIONS), n_layers: int = 2) -> None:
        super().__init__()
        self.hidden_dim, self.n_layers = hidden_dim, n_layers
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.self_loop = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.rel = nn.ModuleList([nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_relations)])
                                  for _ in range(n_layers)])
        self.pending_proj = nn.Linear(1, hidden_dim)

    def forward(self, g: PartialTreeGraph) -> torch.Tensor:
        n = len(g.node_token_idx)
        if n == 0:
            pooled = torch.zeros(self.hidden_dim)
        else:
            h = self.embed(torch.tensor(g.node_token_idx, dtype=torch.long))
            for layer in range(self.n_layers):
                agg = self.self_loop[layer](h)
                msgs = torch.zeros_like(h)
                counts = torch.zeros(n)
                for parent, child, rel in g.edges:
                    msgs[parent] = msgs[parent] + self.rel[layer][rel](h[child])
                    counts[parent] += 1
                h = F.relu(agg + msgs / counts.clamp(min=1).unsqueeze(-1))
            pooled = (h[g.root_ids] if g.root_ids else torch.zeros(1, self.hidden_dim)).mean(dim=0)
        return pooled + self.pending_proj(torch.tensor([float(g.n_pending_windows)]))


class ForwardPolicy(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int, sep_index: Optional[int] = None, sep_bias: float = 4.0) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, vocab_size))
        if sep_index is not None:
            with torch.no_grad():
                self.net[-1].bias[sep_index] = sep_bias

    def forward(self, x):
        return self.net(x)


@dataclass
class Trajectory:
    log_pf_sum: torch.Tensor
    expr: Optional[Expression]
    n_steps: int


class GFlowNetSampler:
    def __init__(self, vocabulary: Vocabulary, encoder: RGCNEncoder, policy: ForwardPolicy, max_len: int = 20,
                 logit_bias: Optional[torch.Tensor] = None) -> None:
        self.vocabulary, self.encoder, self.policy, self.max_len = vocabulary, encoder, policy, max_len
        # TokenMiner.logit_bias: added to the forward policy's raw logits BEFORE the
        # grammar mask, so weight 0 is an effective mask once softmax runs. log_pf is
        # then taken from the SAME biased distribution the action came from, which is
        # what keeps the trajectory-balance loss self-consistent.
        self.logit_bias = logit_bias

    def sample(self, greedy: bool = False) -> Trajectory:
        b = ExpressionBuilder(self.vocabulary)
        b.push(BEG)
        log_pf = torch.zeros(())
        steps = 0
        while not b.is_complete() and steps < self.max_len:
            state = self.encoder(build_partial_tree_graph(b.stack, self.vocabulary))
            mask = torch.zeros(self.vocabulary.size, dtype=torch.bool)
            valid = b.valid_next_tokens()
            if not valid:
                break
            mask[valid] = True
            raw = self.policy(state)
            if self.logit_bias is not None:
                raw = raw + self.logit_bias
            logits = raw.masked_fill(~mask, -1e9)
            dist = Categorical(logits=logits)
            action = torch.argmax(logits) if greedy else dist.sample()
            log_pf = log_pf + dist.log_prob(action)
            b.push(self.vocabulary.index_to_token[int(action.item())])
            steps += 1
        return Trajectory(log_pf, b.build() if b.is_complete() else None, steps)


class TrajectoryBalanceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_z = nn.Parameter(torch.zeros(()))

    def forward(self, log_pf_sum, log_reward):
        return (self.log_z + log_pf_sum - log_reward).pow(2)


class AlphaSAGEMiner(TokenMiner):
    def __init__(self, vocabulary: Vocabulary, rewarder: BatchRewarder, max_expr_len: int = 20,
                 hidden_dim: int = 64, trajectories_per_update: int = 8, lr: float = 1e-3,
                 reward_eps: float = 1e-3, seed: int = 0, logit_bias=None) -> None:
        super().__init__(vocabulary, rewarder, max_expr_len, logit_bias=logit_bias)
        if not rewarder.with_novelty:
            rewarder.with_novelty = True
        self.trajectories_per_update, self.reward_eps = int(trajectories_per_update), float(reward_eps)
        torch.manual_seed(seed)
        self.encoder = RGCNEncoder(vocabulary.size, hidden_dim)
        self.policy = ForwardPolicy(hidden_dim, vocabulary.size, sep_index=vocabulary.sep_index)
        self.tb = TrajectoryBalanceLoss()
        self.sampler = GFlowNetSampler(vocabulary, self.encoder, self.policy, max_expr_len,
                                       logit_bias=self.logit_bias)
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.policy.parameters())
                                          + list(self.tb.parameters()), lr=lr)

    def mine(self, budget: int, on_progress: Optional[ProgressFn] = None) -> MinerReport:
        t0 = time.perf_counter()
        n = 0
        while n < budget:
            k = min(self.trajectories_per_update, budget - n)
            trajs = [self.sampler.sample() for _ in range(k)]
            n += k
            self.report.episodes = n
            results = self.rewarder.rewards([t.expr for t in trajs])
            losses = []
            for t, r in zip(trajs, results):
                if t.expr is not None:
                    self.report.completed += 1
                    self.report.note(r)
                if r.added:
                    self.report.added.append(t.expr)
                    nov = r.novelty if np.isfinite(r.novelty) else 1.0
                    reward = max(float(np.exp(np.clip(r.reward, -5.0, 5.0))) * nov, self.reward_eps)
                else:
                    reward = self.reward_eps
                losses.append(self.tb(t.log_pf_sum, torch.log(torch.tensor(reward, dtype=torch.float32))))
            if losses:
                self.optimizer.zero_grad()
                torch.stack(losses).mean().backward()
                self.optimizer.step()
            if on_progress is not None:
                on_progress(n, budget, {"pool": self.pool.size, "best": self.report.best_reward,
                                        "completed": self.report.completed})
        return self._finish(t0)


__all__ = ["AlphaSAGEMiner", "GFlowNetSampler", "RGCNEncoder", "ForwardPolicy", "TrajectoryBalanceLoss",
           "build_partial_tree_graph", "PartialTreeGraph"]
