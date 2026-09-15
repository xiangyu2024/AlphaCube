"""Compact hypergraph-conv net implementing FactorGCL's cascading-residual
design: prior beta -> hidden beta -> residual alpha, then a return-prediction
head on top of the accumulated signal.

Simplification: each stage is a single linear (or bilinear-similarity, for
the hidden hyperedges) propagation over one cross-section/day rather than a
deep multi-layer hypergraph GCN with multiple rounds of message passing --
appropriate given the small pool sizes and instrument counts this mines
against; the cascading structure (prior -> hidden -> residual) itself is
preserved, which is the paper's central architectural idea.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .hypergraph import HiddenFactorHyperedges


class FactorGCLModel(nn.Module):
    def __init__(self, n_prior_factors: int, embed_dim: int = 16, n_hidden_factors: int = 8) -> None:
        super().__init__()
        n_prior_factors = max(n_prior_factors, 1)
        self.instrument_proj = nn.Linear(n_prior_factors, embed_dim)
        self.prior_readout = nn.Linear(n_prior_factors, embed_dim)
        self.hidden_hyperedges = HiddenFactorHyperedges(embed_dim, n_hidden_factors)
        self.hidden_readout = nn.Linear(n_hidden_factors, embed_dim)
        self.residual_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.pred_head = nn.Linear(embed_dim, 1)

    def forward(self, prior_incidence_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """prior_incidence_t: (n_instruments, n_prior_factors), NaN-free (caller
        zero-fills warm-up NaNs before calling).

        Returns (predicted_return, residual_embedding, instrument_embedding),
        each first-dimension-aligned to `n_instruments`.
        """
        instrument_embed = torch.tanh(self.instrument_proj(prior_incidence_t))  # prior-beta stage
        prior_signal = self.prior_readout(prior_incidence_t)

        hidden_incidence = self.hidden_hyperedges(instrument_embed)
        hidden_signal = self.hidden_readout(hidden_incidence)  # hidden-beta stage

        residual = self.residual_mlp(instrument_embed + prior_signal + hidden_signal)  # residual-alpha stage
        pred = self.pred_head(prior_signal + hidden_signal + residual).squeeze(-1)
        return pred, residual, instrument_embed
