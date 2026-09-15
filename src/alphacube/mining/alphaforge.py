"""AlphaForge stage 1: a GAN-flavoured Generator / Masker / Predictor triad
over the shared grammar, ported unchanged except that the batch of decoded
expressions is rewarded in one engine call (it was already a batch).

Simplification vs the paper (unchanged): the Generator is non-autoregressive
(one logit vector per position from a single MLP pass); grammar validity is
enforced by the Masker walking ``ExpressionBuilder``; a straight-through
estimator lets the Predictor's score flow back into the Generator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..expression import Expression
from .base import MinerReport, ProgressFn, TokenMiner
from .env import BatchRewarder
from .pool import DynamicRollingPool
from .tokens import BEG, ExpressionBuilder, Vocabulary


class Generator(nn.Module):
    def __init__(self, latent_dim: int, vocab_size: int, max_len: int, hidden_dim: int = 128,
                 sep_index: Optional[int] = None, sep_bias: float = 4.0) -> None:
        super().__init__()
        self.max_len, self.vocab_size = max_len, vocab_size
        self.net = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(), nn.Linear(hidden_dim, max_len * vocab_size))
        if sep_index is not None:
            with torch.no_grad():
                for pos in range(max_len):
                    self.net[-1].bias[pos * vocab_size + sep_index] = sep_bias

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(z.shape[0], self.max_len, self.vocab_size)


@dataclass
class DecodedSample:
    token_indices: List[int]
    ste_embeddings: List[torch.Tensor]
    position_probs: List[torch.Tensor]
    expr: Optional[Expression]


class Masker:
    def __init__(self, vocabulary: Vocabulary, embedding: nn.Embedding) -> None:
        self.vocabulary, self.embedding = vocabulary, embedding

    def decode(self, raw_logits: torch.Tensor, sample: bool = False) -> DecodedSample:
        v = self.vocabulary
        b = ExpressionBuilder(v)
        b.push(BEG)
        idxs, stes, probs_l = [], [], []
        for pos in range(raw_logits.shape[0]):
            if b.is_complete():
                break
            mask = torch.zeros(v.size, dtype=torch.bool)
            valid = b.valid_next_tokens()
            if not valid:
                break
            mask[valid] = True
            masked = raw_logits[pos].masked_fill(~mask, -1e9)
            probs = F.softmax(masked, dim=-1)
            idx = int(torch.multinomial(probs, 1).item()) if sample else int(torch.argmax(masked).item())
            hard = F.one_hot(torch.tensor(idx), v.size).float()
            stes.append((hard + probs - probs.detach()) @ self.embedding.weight)
            probs_l.append(probs)
            idxs.append(idx)
            b.push(v.index_to_token[idx])
        return DecodedSample(idxs, stes, probs_l, b.build() if b.is_complete() else None)


class Predictor(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.net = nn.Sequential(nn.Linear(embed_dim * 2, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([seq.mean(dim=0), seq.max(dim=0).values], dim=-1)).squeeze(-1)


def diversity_penalty(batch_position_probs: List[List[torch.Tensor]]) -> torch.Tensor:
    max_len = max((len(p) for p in batch_position_probs), default=0)
    terms = []
    for pos in range(max_len):
        active = [p[pos] for p in batch_position_probs if len(p) > pos]
        if len(active) < 2:
            continue
        P = torch.stack(active, dim=0)
        sim = P @ P.T
        n = P.shape[0]
        terms.append((sim.sum() - sim.diag().sum()) / (n * (n - 1)))
    return torch.stack(terms).mean() if terms else torch.zeros(())


class AlphaForgeMiner(TokenMiner):
    def __init__(self, vocabulary: Vocabulary, rewarder: BatchRewarder, max_expr_len: int = 20,
                 latent_dim: int = 16, batch_size: int = 16, diversity_coef: float = 0.05,
                 gen_lr: float = 1e-3, pred_lr: float = 1e-3, sample: bool = False, seed: int = 0) -> None:
        super().__init__(vocabulary, rewarder, max_expr_len)
        self.latent_dim, self.batch_size, self.diversity_coef = int(latent_dim), int(batch_size), float(diversity_coef)
        self.sample = bool(sample)
        torch.manual_seed(seed)
        self.generator = Generator(latent_dim, vocabulary.size, max_expr_len, sep_index=vocabulary.sep_index)
        self.predictor = Predictor(vocabulary.size)
        self.masker = Masker(vocabulary, self.predictor.embedding)
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=gen_lr)
        self.pred_optimizer = torch.optim.Adam(self.predictor.parameters(), lr=pred_lr)

    def mine(self, budget: int, on_progress: Optional[ProgressFn] = None) -> MinerReport:
        t0 = time.perf_counter()
        bs = max(1, min(self.batch_size, budget))
        done = 0
        while done < budget:
            n = min(bs, budget - done)
            self._train_step(n)
            done += n
            self.report.episodes = done
            if on_progress is not None:
                on_progress(done, budget, {"pool": self.pool.size, "best": self.report.best_reward,
                                           "completed": self.report.completed})
        return self._finish(t0)

    def _train_step(self, batch_size: int) -> None:
        z = torch.randn(batch_size, self.latent_dim)
        raw = self.generator(z)
        decoded = [self.masker.decode(raw[i], sample=self.sample) for i in range(batch_size)]
        scores = [self.predictor(torch.stack(d.ste_embeddings)) for d in decoded if d.ste_embeddings]
        gen_term = torch.stack(scores).mean() if scores else torch.zeros(())
        loss_g = -gen_term + self.diversity_coef * diversity_penalty([d.position_probs for d in decoded])
        self.gen_optimizer.zero_grad()
        loss_g.backward()
        self.gen_optimizer.step()
        results = self.rewarder.rewards([d.expr for d in decoded])
        preds, targets = [], []
        for d, r in zip(decoded, results):
            if not d.ste_embeddings:
                continue
            preds.append(self.predictor(torch.stack(d.ste_embeddings).detach()))
            targets.append(float(r.reward))
            if d.expr is not None:
                self.report.completed += 1
                self.report.note(r)
                if r.added:
                    self.report.added.append(d.expr)
        if preds:
            loss_p = F.mse_loss(torch.stack(preds), torch.tensor(targets, dtype=torch.float32))
            self.pred_optimizer.zero_grad()
            loss_p.backward()
            self.pred_optimizer.step()

    def combine_dynamic(self, window: int = 60) -> DynamicRollingPool:
        """Stage 2: re-wrap the mined members in a rolling-window pool."""
        dyn = DynamicRollingPool(self.pool.capacity, self.pool.target, window=window,
                                 dedup_threshold=None, min_coverage=0.0)
        panels = self.rewarder.panels(self.pool.exprs())
        for m, p in zip(self.pool.members, panels):
            if p is not None:
                dyn.try_add(m.expr, p.astype(float))
        return dyn


__all__ = ["AlphaForgeMiner", "Generator", "Masker", "Predictor", "DecodedSample", "diversity_penalty"]
