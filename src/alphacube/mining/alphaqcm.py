"""AlphaQCM: distributional (IQN) DQN over ``TokenEnv`` with tail-mean
(CVaR-style) action selection. Ported unchanged in its learning machinery;
the only structural change is reward timing: an episode's transitions are
held back until the batch of ``episodes_per_batch`` expressions has been
rewarded in one engine call, then pushed to the replay buffer with the final
terminal reward, and the same number of gradient updates the per-step
schedule would have made are run.

Simplification vs the paper (unchanged): the Quantile Cornish-Fisher moment
correction is replaced by the variance of the sampled quantiles as the
uncertainty bonus; ``tail_alpha`` biases the taus toward the upper tail at
action-selection time only (the regression target keeps uniform taus).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .alphagen import TokenLSTMEncoder
from .base import MinerReport, ProgressFn, TokenMiner
from .env import BatchRewarder
from .tokens import Vocabulary


@dataclass
class Transition:
    state: np.ndarray
    length: int
    mask: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_length: int
    next_mask: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 4000) -> None:
        self.capacity = int(capacity)
        self.buffer: List[Transition] = []
        self._pos = 0

    def push(self, t: Transition) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(t)
        else:
            self.buffer[self._pos] = t
            self._pos = (self._pos + 1) % self.capacity

    def sample(self, n: int, rng: random.Random) -> List[Transition]:
        return rng.sample(self.buffer, min(n, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class IQNNetwork(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_dim: int = 128, num_layers: int = 2,
                 n_cos: int = 64, sep_index: Optional[int] = None, sep_bias: float = 4.0) -> None:
        super().__init__()
        self.encoder = TokenLSTMEncoder(vocab_size, embed_dim, hidden_dim, num_layers)
        self.cos_embed = nn.Linear(n_cos, hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, vocab_size))
        self.register_buffer("i_pi", math.pi * torch.arange(n_cos, dtype=torch.float32))
        if sep_index is not None:
            with torch.no_grad():
                self.head[-1].bias[sep_index] = sep_bias

    def forward(self, tokens, lengths, taus):
        state = self.encoder(tokens, lengths)
        phi = torch.relu(self.cos_embed(torch.cos(taus.unsqueeze(-1) * self.i_pi)))
        return self.head(state.unsqueeze(1) * phi)                       # (B, N, vocab)

    def tail_mean_and_uncertainty(self, tokens, lengths, tau_lo: float, n_tau: int):
        if not (0.0 <= tau_lo < 1.0):
            raise ValueError(f"tau_lo must be in [0, 1), got {tau_lo}")
        taus = tau_lo + (1.0 - tau_lo) * torch.rand(tokens.shape[0], n_tau, device=tokens.device)
        q = self.forward(tokens, lengths, taus)
        return q.mean(dim=1), q.var(dim=1, unbiased=False)


def quantile_huber_loss(pred, taus, target, kappa: float = 1.0):
    td = target.unsqueeze(1) - pred.unsqueeze(2)                         # (B, N, N')
    huber = torch.where(td.abs() <= kappa, 0.5 * td.pow(2), kappa * (td.abs() - 0.5 * kappa))
    w = torch.abs(taus.unsqueeze(2) - (td.detach() < 0).float())
    return (w * huber / kappa).sum(dim=2).mean(dim=1).mean()


class AlphaQCMMiner(TokenMiner):
    def __init__(self, vocabulary: Vocabulary, rewarder: BatchRewarder, max_expr_len: int = 20,
                 episodes_per_batch: int = 8, n_tau: int = 8, n_tau_prime: int = 8,
                 uncertainty_lambda: float = 0.1, tail_alpha: float = 0.75, gamma: float = 0.99, lr: float = 1e-3,
                 batch_size: int = 32, buffer_capacity: int = 4000, target_update_every: int = 5,
                 min_buffer_before_update: int = 32, epsilon_start: float = 0.3, epsilon_end: float = 0.05,
                 seed: int = 0) -> None:
        super().__init__(vocabulary, rewarder, max_expr_len)
        self.episodes_per_batch = int(episodes_per_batch)
        self.n_tau, self.n_tau_prime = int(n_tau), int(n_tau_prime)
        self.uncertainty_lambda, self.tail_alpha, self.gamma = float(uncertainty_lambda), float(tail_alpha), float(gamma)
        self.batch_size, self.target_update_every = int(batch_size), int(target_update_every)
        self.min_buffer_before_update = int(min_buffer_before_update)
        self.epsilon_start, self.epsilon_end = float(epsilon_start), float(epsilon_end)
        torch.manual_seed(seed)
        self._rng = random.Random(seed)
        self.online_net = IQNNetwork(vocabulary.size, sep_index=vocabulary.sep_index)
        self.target_net = IQNNetwork(vocabulary.size, sep_index=vocabulary.sep_index)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

    def _select_action(self, state, length, mask, epsilon: float) -> int:
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            return 0
        if self._rng.random() < epsilon:
            return int(self._rng.choice(valid))
        with torch.no_grad():
            q, var = self.online_net.tail_mean_and_uncertainty(torch.from_numpy(state).unsqueeze(0),
                                                               torch.tensor([length]), self.tail_alpha, self.n_tau)
            scores = (q + self.uncertainty_lambda * var.clamp(min=0).sqrt()).squeeze(0)
            scores = scores.masked_fill(~torch.from_numpy(mask), -1e9)
            return int(torch.argmax(scores).item())

    def mine(self, budget: int, on_progress: Optional[ProgressFn] = None) -> MinerReport:
        t0 = time.perf_counter()
        episodes = 0
        held: List[Tuple[List[Transition], Optional[object]]] = []
        while episodes < budget:
            eps = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * (episodes / max(1, budget - 1))
            result = self.env.reset()
            transitions: List[Transition] = []
            expr = None
            while True:
                a = self._select_action(result.state, result.length, result.mask, eps)
                nxt = self.env.step(a)
                transitions.append(Transition(result.state, result.length, result.mask, a,
                                              0.0 if nxt.reward is None else nxt.reward,
                                              nxt.state, nxt.length, nxt.mask, nxt.done))
                if nxt.done:
                    expr = nxt.expr
                    break
                result = nxt
            held.append((transitions, expr))
            episodes += 1
            self.report.episodes = episodes
            if expr is not None:
                self.report.completed += 1
            if episodes % self.target_update_every == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())
            if len(held) >= self.episodes_per_batch or episodes == budget:
                self._flush(held)
                if on_progress is not None:
                    on_progress(episodes, budget, {"pool": self.pool.size, "best": self.report.best_reward,
                                                   "completed": self.report.completed, "buffer": len(self.buffer)})
                held = []
        return self._finish(t0)

    def _flush(self, held) -> None:
        results = self.rewarder.rewards([e for _, e in held])
        n_steps = 0
        for (transitions, expr), r in zip(held, results):
            if expr is not None:
                transitions[-1].reward = r.reward
                self.report.note(r)
                if r.added:
                    self.report.added.append(expr)
            for t in transitions:
                self.buffer.push(t)
            n_steps += len(transitions)
        if len(self.buffer) >= self.min_buffer_before_update:
            for _ in range(n_steps):
                self._update()

    def _update(self) -> None:
        batch = self.buffer.sample(self.batch_size, self._rng)
        if not batch:
            return
        states = torch.from_numpy(np.stack([t.state for t in batch]))
        lengths = torch.tensor([t.length for t in batch])
        actions = torch.tensor([t.action for t in batch], dtype=torch.long)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        next_states = torch.from_numpy(np.stack([t.next_state for t in batch]))
        next_lengths = torch.tensor([t.next_length for t in batch])
        next_masks = torch.from_numpy(np.stack([t.next_mask for t in batch]))
        dones = torch.tensor([float(t.done) for t in batch])
        B = len(batch)
        taus = torch.rand(B, self.n_tau)
        pred = self.online_net(states, lengths, taus).gather(2, actions.view(B, 1, 1).expand(B, self.n_tau, 1)).squeeze(-1)
        with torch.no_grad():
            tq, tv = self.target_net.tail_mean_and_uncertainty(next_states, next_lengths, self.tail_alpha, self.n_tau)
            scores = (tq + self.uncertainty_lambda * tv.clamp(min=0).sqrt()).masked_fill(~next_masks, -1e9)
            best = scores.argmax(dim=1)
            taus_next = torch.rand(B, self.n_tau_prime)
            tgt_q = self.target_net(next_states, next_lengths, taus_next).gather(
                2, best.view(B, 1, 1).expand(B, self.n_tau_prime, 1)).squeeze(-1)
            target = rewards.unsqueeze(1) + self.gamma * (1.0 - dones).unsqueeze(1) * tgt_q
        loss = quantile_huber_loss(pred, taus, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


__all__ = ["AlphaQCMMiner", "IQNNetwork", "Transition", "ReplayBuffer", "quantile_huber_loss"]
