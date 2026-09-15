"""Temporal residual InfoNCE: FactorGCL's contrastive regularizer on the
residual-alpha stage. Positive pair = the same instrument's residual
embedding at two different time windows; negatives = the other instruments
in the batch (a standard symmetric InfoNCE / SimCLR-style setup).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def temporal_residual_infonce(
    residual_a: torch.Tensor, residual_b: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """residual_a, residual_b: (n_instruments, embed_dim), index-aligned so row i
    in both is the SAME instrument at two different days/windows."""
    a = F.normalize(residual_a, dim=-1)
    b = F.normalize(residual_b, dim=-1)
    logits = (a @ b.T) / temperature
    targets = torch.arange(a.shape[0], device=a.device)
    loss_ab = F.cross_entropy(logits, targets)
    loss_ba = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss_ab + loss_ba)
