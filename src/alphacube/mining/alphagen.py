"""AlphaGen (KDD'23) as a from-scratch PPO over ``TokenEnv``, with the reward
of every completed expression deferred to the rollout-batch boundary: the
``episodes_per_update`` episodes are collected first, their expressions are
evaluated in ONE engine call, the terminal rewards are back-filled into the
rollout buffer, and only then does the PPO update run. With ``gamma = 1``
and a single terminal reward per episode, GAE gives exactly the advantages
the immediate-reward version would.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .base import MinerReport, ProgressFn, TokenMiner
from .env import BatchRewarder
from .tokens import Vocabulary


class TokenLSTMEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_dim: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embed(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.clamp(min=1).cpu(), batch_first=True,
                                                   enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return h_n[-1]


class ActorCritic(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_dim: int = 128, num_layers: int = 2,
                 sep_index: Optional[int] = None, sep_bias: float = 4.0,
                 logit_bias: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.encoder = TokenLSTMEncoder(vocab_size, embed_dim, hidden_dim, num_layers)
        self.actor = nn.Linear(hidden_dim, vocab_size)
        self.critic = nn.Linear(hidden_dim, 1)
        if sep_index is not None:
            # Cold start: bias SEP so early episodes terminate often enough to
            # bootstrap the sparse pool reward (a practical addition, not PPO).
            with torch.no_grad():
                self.actor.bias[sep_index] = sep_bias
        # Steering bias (TokenMiner.logit_bias), a buffer rather than a plain
        # attribute so .to(device) / state_dict carry it with the rest of the policy.
        # It lives HERE, not in the miner, because forward() is the single place both
        # rollout (act) and the PPO update read logits -- biasing one and not the
        # other would make the ratio in the clipped objective wrong.
        self.register_buffer("logit_bias", None if logit_bias is None else logit_bias.detach().clone())

    def forward(self, tokens, lengths, mask):
        feat = self.encoder(tokens, lengths)
        raw = self.actor(feat)
        if self.logit_bias is not None:
            raw = raw + self.logit_bias                 # BEFORE the mask: weight 0 -> an effective mask
        logits = raw.masked_fill(~mask, -1e9)
        return logits, self.critic(feat).squeeze(-1)

    def act(self, tokens, lengths, mask):
        logits, value = self.forward(tokens, lengths, mask)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value, dist.entropy()


class _RolloutBuffer:
    def __init__(self) -> None:
        self.states: List[np.ndarray] = []
        self.lengths: List[int] = []
        self.masks: List[np.ndarray] = []
        self.actions: List[int] = []
        self.logprobs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.next_states: List[np.ndarray] = []
        self.next_lengths: List[int] = []
        self.next_masks: List[np.ndarray] = []
        self.dones: List[bool] = []
        self.episode_bounds: List[Tuple[int, int]] = []
        self._start = 0

    def add(self, state, length, mask, action, logprob, value) -> None:
        self.states.append(state); self.lengths.append(length); self.masks.append(mask)
        self.actions.append(action); self.logprobs.append(logprob); self.values.append(value)
        self.rewards.append(0.0)

    def add_next(self, state, length, mask, done) -> None:
        self.next_states.append(state); self.next_lengths.append(length)
        self.next_masks.append(mask); self.dones.append(done)

    def finish_episode(self, terminal_reward: float) -> int:
        """Close the episode; returns the index of its terminal transition."""
        self.rewards[-1] = terminal_reward
        self.episode_bounds.append((self._start, len(self.states)))
        self._start = len(self.states)
        return len(self.states) - 1

    def __len__(self) -> int:
        return len(self.states)


def _compute_gae(rewards, values, gamma, lam):
    T = len(rewards)
    adv = np.zeros(T)
    last, next_value = 0.0, 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
        next_value = values[t]
    return adv, adv + np.asarray(values, dtype=np.float64)


class AlphaGenMiner(TokenMiner):
    def __init__(self, vocabulary: Vocabulary, rewarder: BatchRewarder, max_expr_len: int = 20,
                 episodes_per_update: int = 8, ppo_epochs: int = 4, clip_eps: float = 0.2, gamma: float = 1.0,
                 gae_lambda: float = 0.95, entropy_coef: float = 0.01, value_coef: float = 0.5, lr: float = 3e-4,
                 record_transitions: bool = False, seed: int = 0, logit_bias=None) -> None:
        super().__init__(vocabulary, rewarder, max_expr_len, logit_bias=logit_bias)
        self.episodes_per_update = int(episodes_per_update)
        self.ppo_epochs, self.clip_eps, self.gamma = int(ppo_epochs), float(clip_eps), float(gamma)
        self.gae_lambda, self.entropy_coef, self.value_coef = float(gae_lambda), float(entropy_coef), float(value_coef)
        torch.manual_seed(seed)
        self.policy = ActorCritic(vocab_size=vocabulary.size, sep_index=vocabulary.sep_index,
                                  logit_bias=self.logit_bias)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        # Optional side channel for the PPO -> QCM warm start: every transition
        # with its FINAL reward (known only after the batch is rewarded).
        self.record_transitions = bool(record_transitions)
        self.recorded: List["Transition"] = []

    def mine(self, budget: int, on_progress: Optional[ProgressFn] = None) -> MinerReport:
        t0 = time.perf_counter()
        buf = _RolloutBuffer()
        pending: List[Tuple[int, object]] = []          # (terminal index in buf, expr)
        episodes = 0
        while episodes < budget:
            expr, term_idx = self._run_episode(buf)
            if expr is not None:
                pending.append((term_idx, expr))
                self.report.completed += 1
            episodes += 1
            self.report.episodes = episodes
            if len(buf.episode_bounds) >= self.episodes_per_update or episodes == budget:
                self._reward_pending(buf, pending)
                if len(buf) > 0:
                    self._ppo_update(buf)
                if self.record_transitions:
                    self._record(buf)
                if on_progress is not None:
                    on_progress(episodes, budget, {"pool": self.pool.size, "best": self.report.best_reward,
                                                   "completed": self.report.completed})
                buf, pending = _RolloutBuffer(), []
        return self._finish(t0)

    def _run_episode(self, buf: _RolloutBuffer):
        result = self.env.reset()
        while True:
            with torch.no_grad():
                a, lp, v, _ = self.policy.act(torch.from_numpy(result.state).unsqueeze(0),
                                              torch.tensor([result.length]),
                                              torch.from_numpy(result.mask).unsqueeze(0))
            buf.add(result.state.copy(), result.length, result.mask.copy(), int(a.item()), float(lp.item()), float(v.item()))
            result = self.env.step(int(a.item()))
            buf.add_next(result.state.copy(), result.length, result.mask.copy(), result.done)
            if result.done:
                idx = buf.finish_episode(0.0 if result.reward is None else result.reward)
                return result.expr, idx

    def _reward_pending(self, buf: _RolloutBuffer, pending) -> None:
        if not pending:
            return
        results = self.rewarder.rewards([e for _, e in pending])
        for (idx, _), r in zip(pending, results):
            buf.rewards[idx] = r.reward
            self.report.note(r)
            if r.added:
                self.report.added.append(pending[0][1] if False else _expr_of(pending, idx))

    def _record(self, buf: _RolloutBuffer) -> None:
        from .alphaqcm import Transition
        for i in range(len(buf)):
            self.recorded.append(Transition(buf.states[i], buf.lengths[i], buf.masks[i], buf.actions[i],
                                            buf.rewards[i], buf.next_states[i], buf.next_lengths[i],
                                            buf.next_masks[i], buf.dones[i]))

    def _ppo_update(self, buf: _RolloutBuffer) -> None:
        advantages = np.zeros(len(buf))
        returns = np.zeros(len(buf))
        for s, e in buf.episode_bounds:
            adv, ret = _compute_gae(buf.rewards[s:e], buf.values[s:e], self.gamma, self.gae_lambda)
            advantages[s:e], returns[s:e] = adv, ret
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        states = torch.from_numpy(np.stack(buf.states))
        lengths = torch.tensor(buf.lengths)
        masks = torch.from_numpy(np.stack(buf.masks))
        actions = torch.tensor(buf.actions, dtype=torch.long)
        old_lp = torch.tensor(buf.logprobs, dtype=torch.float32)
        adv_t = torch.tensor(advantages, dtype=torch.float32)
        ret_t = torch.tensor(returns, dtype=torch.float32)
        for _ in range(self.ppo_epochs):
            logits, values = self.policy(states, lengths, masks)
            dist = Categorical(logits=logits)
            new_lp = dist.log_prob(actions)
            ratio = torch.exp(new_lp - old_lp)
            surr = torch.min(ratio * adv_t, torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t)
            loss = -surr.mean() + self.value_coef * F.mse_loss(values, ret_t) - self.entropy_coef * dist.entropy().mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


def _expr_of(pending, idx):
    for i, e in pending:
        if i == idx:
            return e
    raise KeyError(idx)


__all__ = ["AlphaGenMiner", "ActorCritic", "TokenLSTMEncoder"]
