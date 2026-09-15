"""Stock x factor incidence for FactorGCL's cascading hypergraph.

Two kinds of hyperedges:
  - "prior factors": one hyperedge per already-mined factor panel (e.g. a
    pool member's `(n_days, n_instruments)` panel). Each instrument's
    incidence weight is that factor's cross-sectional z-score for the day.
    This is a *continuous* relaxation of the paper's discrete quantile/sign
    grouping (e.g. "top decile" / "bottom decile" bucket membership) -- using
    the continuous z-score directly is simpler, keeps the whole pipeline
    differentiable end-to-end, and preserves the same ranking information
    the discrete buckets would encode.
  - "hidden factors": a small number of learnable prototype vectors; an
    instrument's soft membership in hidden hyperedge k is
    `sigmoid(instrument_embedding . prototype_k)`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from ...metrics import zscore_rows


def build_prior_incidence(panels: Sequence[np.ndarray]) -> np.ndarray:
    """(n_days, n_instruments, n_prior_factors) continuous incidence matrix:
    every `(n_days, n_instruments)` factor panel z-scored per day, stacked.

    `panels` are already-evaluated daily factor values (a pool's
    `z_columns()`, or any `(D, S)` panels the cube engine produced). The
    previous project's version evaluated pool-member expressions here
    itself; that now happens upstream in the engine, so panels are the only
    input. Z-scoring is idempotent, so already-z-scored columns are fine.
    An empty sequence has no `(n_days, n_instruments)` to shape a zero-width
    incidence with and is rejected -- `FactorGCLCombiner` handles the
    no-factor case itself with a single all-zero column.
    """
    if len(panels) == 0:
        raise ValueError("build_prior_incidence needs at least one (n_days, n_instruments) panel")
    columns = []
    for i, panel in enumerate(panels):
        panel = np.asarray(panel, dtype=np.float64)
        if panel.ndim != 2:
            raise ValueError(f"panel {i} must be 2-D (n_days, n_instruments), got shape {panel.shape}")
        if columns and panel.shape != columns[0].shape:
            raise ValueError(f"panel {i} has shape {panel.shape}, expected {columns[0].shape}")
        columns.append(zscore_rows(panel))
    return np.stack(columns, axis=-1)


class HiddenFactorHyperedges(nn.Module):
    def __init__(self, embed_dim: int, n_hidden_factors: int = 8) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(n_hidden_factors, embed_dim) * 0.1)

    def forward(self, instrument_embed: torch.Tensor) -> torch.Tensor:
        """instrument_embed: (n_instruments, embed_dim) -> (n_instruments, n_hidden_factors)."""
        return torch.sigmoid(instrument_embed @ self.prototypes.T)
