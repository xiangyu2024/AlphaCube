"""Factor pools over ``(D, S)`` panels.

A pool holds member expressions together with their per-day z-scored panels
and a weight vector, and answers one question for a search algorithm: what is
the combined IC after adding this candidate? Three weight-fitting styles, all
on the same base:

- ``MsePool``: ordinary least squares of the label on the z-scored members
  (missing z-scores count as neutral 0, like the combination itself).
- ``MeanStdPool``: Adam on an ICIR objective of the combination.
- ``DynamicRollingPool``: an inner fit restricted to a trailing window of days.

All insertions are transactional and pass the same gates: finite solo IC,
coverage >= ``min_coverage``, |corr| <= ``dedup_threshold`` against every
member, finite pool IC after the (possibly pruned) refit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Type

import numpy as np
import torch

from ..expression import Expression
from ..metrics import backtest_metrics, cs_pearson, nanmean_quiet, sampled_abs_corr, zscore_rows


class DuplicateFactorError(ValueError):
    pass


@dataclass
class Member:
    expr: Expression
    z: np.ndarray          # (D, S) per-day z-scored panel
    ic: float


class PoolBase(ABC):
    def __init__(self, capacity: int, target: np.ndarray, dedup_threshold: Optional[float] = 0.99,
                 dedup_sample_days: int = 250, min_coverage: float = 0.5,
                 realized_returns: Optional[np.ndarray] = None) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = int(capacity)
        self.target = np.asarray(target, dtype=np.float64)
        self.dedup_threshold = dedup_threshold
        self.dedup_sample_days = int(dedup_sample_days)
        self.min_coverage = float(min_coverage)
        # P&L leg for test_ensemble: the label itself when it is a return.
        self.realized = self.target if realized_returns is None else np.asarray(realized_returns, dtype=np.float64)
        self.members: List[Member] = []
        self.weights: np.ndarray = np.zeros(0)

    # -- introspection -----------------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self.members)

    def exprs(self) -> List[Expression]:
        return [m.expr for m in self.members]

    def __contains__(self, expr: Expression) -> bool:
        return any(m.expr == expr for m in self.members)

    def z_columns(self) -> List[np.ndarray]:
        return [m.z for m in self.members]

    def combined(self) -> np.ndarray:
        out = np.zeros_like(self.target)
        for m, w in zip(self.members, self.weights):
            out = out + w * np.nan_to_num(m.z, nan=0.0)
        return out

    def pool_ic(self) -> float:
        if not self.members:
            return float("nan")
        return nanmean_quiet(cs_pearson(self.combined(), self.target))

    def test_ensemble(self) -> Dict[str, float]:
        if not self.members:
            raise ValueError("pool is empty; nothing to test")
        combined = self.combined()
        return backtest_metrics(cs_pearson(combined, self.target), combined, self.realized)

    def summary(self) -> List[Dict[str, float]]:
        return [{"expression": repr(m.expr), "weight": float(w), "ic": m.ic}
                for m, w in zip(self.members, self.weights)]

    # -- fitting ----------------------------------------------------------------------
    @abstractmethod
    def _fit(self, z: List[np.ndarray], target: np.ndarray) -> np.ndarray:
        """Weights for z-scored member panels against ``target`` (both (D, S))."""

    def refit(self) -> np.ndarray:
        self.weights = self._fit(self.z_columns(), self.target) if self.members else np.zeros(0)
        return self.weights

    def try_add(self, expr: Expression, panel: np.ndarray) -> float:
        """Add, refit, prune the weakest if over capacity, refit; return the pool IC.
        Any failure restores the pool exactly and re-raises."""
        z = zscore_rows(panel)
        ic = nanmean_quiet(cs_pearson(panel, self.target))
        if not np.isfinite(ic):
            raise ValueError("candidate has no finite IC")
        cells = max(int(np.isfinite(self.target).sum()), 1)
        coverage = float((np.isfinite(panel) & np.isfinite(self.target)).sum() / cells)
        if coverage < self.min_coverage:
            raise ValueError(f"coverage {coverage:.2f} below min_coverage {self.min_coverage}")
        saved_members, saved_weights = list(self.members), self.weights
        try:
            if self.dedup_threshold is not None:
                for i, m in enumerate(self.members):
                    c = sampled_abs_corr(z, m.z, self.dedup_sample_days)
                    if np.isfinite(c) and c > self.dedup_threshold:
                        raise DuplicateFactorError(f"|corr| = {c:.4f} with member #{i} {m.expr!r}")
            self.members.append(Member(expr, z, ic))
            self.refit()
            if self.size > self.capacity:
                weakest = int(np.argmin(np.abs(self.weights)))
                del self.members[weakest]
                self.refit()
            pic = self.pool_ic()
            if not np.isfinite(pic):
                raise ValueError("pool IC not finite after insertion")
            return pic
        except Exception:
            self.members, self.weights = saved_members, saved_weights
            raise


def _lstsq_weights(z: List[np.ndarray], target: np.ndarray) -> np.ndarray:
    k = len(z)
    if k == 0:
        return np.zeros(0)
    X = np.stack([np.nan_to_num(c.reshape(-1), nan=0.0) for c in z], axis=1)
    y = target.reshape(-1)
    ok = np.isfinite(y)
    if ok.sum() < 10 * k:
        return np.full(k, 1.0 / k)
    w, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    return w if np.isfinite(w).all() else np.full(k, 1.0 / k)


class MsePool(PoolBase):
    def _fit(self, z: List[np.ndarray], target: np.ndarray) -> np.ndarray:
        return _lstsq_weights(z, target)


def _torch_daily_ic(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    count = mask.sum(dim=1)
    safe = count.clamp(min=2)
    xm = torch.where(mask, x, torch.zeros_like(x))
    ym = torch.where(mask, y, torch.zeros_like(y))
    xc = torch.where(mask, x - (xm.sum(1) / safe).unsqueeze(1), torch.zeros_like(x))
    yc = torch.where(mask, y - (ym.sum(1) / safe).unsqueeze(1), torch.zeros_like(y))
    cov = (xc * yc).sum(1)
    xv, yv = (xc ** 2).sum(1), (yc ** 2).sum(1)
    ic = cov / torch.sqrt(xv * yv).clamp(min=1e-8)
    return ic, (count >= 2) & (xv > 1e-12) & (yv > 1e-12)


class MeanStdPool(PoolBase):
    """Adam on -ICIR of the combination (the AlphaGen 'MeanStd' pool variant)."""

    def __init__(self, capacity: int, target: np.ndarray, lr: float = 0.05, n_steps: int = 200, **kw) -> None:
        super().__init__(capacity, target, **kw)
        self.lr, self.n_steps = float(lr), int(n_steps)

    def _fit(self, z: List[np.ndarray], target: np.ndarray) -> np.ndarray:
        k = len(z)
        if k == 0:
            return np.zeros(0)
        stacked = np.stack(z, axis=0)
        mask = np.isfinite(target) & np.all(np.isfinite(stacked), axis=0)
        x = torch.tensor(np.nan_to_num(stacked, nan=0.0), dtype=torch.float64)
        y = torch.tensor(np.nan_to_num(target, nan=0.0), dtype=torch.float64)
        m = torch.tensor(mask)
        w = torch.full((k,), 1.0 / k, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([w], lr=self.lr)
        for _ in range(self.n_steps):
            opt.zero_grad()
            ic, valid = _torch_daily_ic(torch.einsum("i,idn->dn", w, x), y, m)
            v = ic[valid]
            if v.numel() < 2:
                break
            (-(v.mean() / (v.std() + 1e-8))).backward()
            opt.step()
        return w.detach().numpy().astype(np.float64)


class DynamicRollingPool(PoolBase):
    """Weights fitted on the trailing ``window`` days only (regime-adaptive)."""

    def __init__(self, capacity: int, target: np.ndarray, window: int = 60,
                 inner: Type[PoolBase] = MsePool, **kw) -> None:
        super().__init__(capacity, target, **kw)
        self.window = int(window)
        self._inner = inner(capacity, target, dedup_threshold=None)
        self.as_of_day: Optional[int] = None

    def _fit(self, z: List[np.ndarray], target: np.ndarray) -> np.ndarray:
        end = target.shape[0] if self.as_of_day is None else int(self.as_of_day)
        start = max(0, end - self.window)
        return self._inner._fit([c[start:end] for c in z], target[start:end])

    def refit_as_of(self, as_of_day: int, window: Optional[int] = None) -> np.ndarray:
        """Re-fit on the trailing window ending at (exclusive) ``as_of_day``."""
        if window is not None:
            self.window = int(window)
        self.as_of_day = int(as_of_day)
        try:
            return self.refit()
        finally:
            self.as_of_day = None


LinearPool = MsePool     # name used by the first cut of this project

__all__ = ["PoolBase", "MsePool", "MeanStdPool", "DynamicRollingPool", "LinearPool",
           "DuplicateFactorError", "Member"]
