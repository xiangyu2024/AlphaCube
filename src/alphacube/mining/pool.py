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
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Type

import numpy as np
import torch

from ..expression import Expression
from ..metrics import (MIN_PAIRS, backtest_metrics, cs_pearson, nanmean_quiet, sampled_abs_corr,
                       zscore_rows)


class DuplicateFactorError(ValueError):
    pass


@dataclass
class Member:
    expr: Expression
    z: np.ndarray          # (D, S) per-day z-scored panel
    ic: float


class PoolBase(ABC):
    def __init__(self, capacity: Optional[int], target: np.ndarray, dedup_threshold: Optional[float] = 0.99,
                 dedup_sample_days: int = 250, min_coverage: float = 0.5,
                 min_scored_days: float = 0.5,
                 realized_returns: Optional[np.ndarray] = None) -> None:
        # None or 0: no cap -- every candidate that passes the gates stays.
        if capacity is not None and capacity < 0:
            raise ValueError("capacity must be >= 0 (0 or None = unlimited)")
        self.capacity: Optional[int] = int(capacity) if capacity else None
        self.target = np.asarray(target, dtype=np.float64)
        self.dedup_threshold = dedup_threshold
        self.dedup_sample_days = int(dedup_sample_days)
        self.min_coverage = float(min_coverage)
        # Share of the label's days on which the factor must actually produce an IC.
        # coverage asks "is the cell finite", which a degenerate panel passes: a
        # factor that is identically 0 everywhere is finite everywhere. What it
        # cannot do is vary across stocks, so cs_pearson returns NaN for the day and
        # the |IC| ends up computed from whatever handful of days did vary. On the
        # production cube `signedpower(pow(0.5, ts_quantile_q75(day_close(Volume), 50)))`
        # came top of a generation that way -- 0.5**1e5 underflows to exactly 0 in
        # float32 -- with coverage 0.81 and an IC drawn from 329 days out of 1893.
        # This gate asks the question coverage cannot: does it separate stocks, and
        # on how many days.
        self.min_scored_days = float(min_scored_days)
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

    def _screen(self, panel: np.ndarray) -> Tuple[np.ndarray, float]:
        """The gates a candidate passes on its own: finite IC, coverage, scored days.
        Returns (z-scored panel, IC); raises ValueError naming the gate."""
        z = zscore_rows(panel)
        ic_series = cs_pearson(panel, self.target)
        ic = nanmean_quiet(ic_series)
        if not np.isfinite(ic):
            raise ValueError("candidate has no finite IC")
        cells = max(int(np.isfinite(self.target).sum()), 1)
        coverage = float((np.isfinite(panel) & np.isfinite(self.target)).sum() / cells)
        if coverage < self.min_coverage:
            raise ValueError(f"coverage {coverage:.2f} below min_coverage {self.min_coverage}")
        usable = max(int((np.isfinite(self.target).sum(axis=1) >= MIN_PAIRS).sum()), 1)
        scored = float(np.isfinite(ic_series).sum() / usable)
        if scored < self.min_scored_days:
            raise ValueError(f"scored on {scored:.2f} of usable days, below min_scored_days "
                             f"{self.min_scored_days} (finite everywhere but no cross-sectional spread)")
        return z, ic

    def try_add(self, expr: Expression, panel: np.ndarray) -> float:
        """Add, refit, prune the weakest if over capacity, refit; return the pool IC.
        Any failure restores the pool exactly and re-raises."""
        z, ic = self._screen(panel)
        saved_members, saved_weights = list(self.members), self.weights
        try:
            if self.dedup_threshold is not None:
                for i, m in enumerate(self.members):
                    c = sampled_abs_corr(z, m.z, self.dedup_sample_days)
                    if np.isfinite(c) and c > self.dedup_threshold:
                        raise DuplicateFactorError(f"|corr| = {c:.4f} with member #{i} {m.expr!r}")
            self.members.append(Member(expr, z, ic))
            self.refit()
            if self.capacity is not None and self.size > self.capacity:
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

    def admit(self, candidates: Iterable[Tuple[Expression, Optional[np.ndarray]]]) -> "Admission":
        """Offer candidates in order: which joined, and why the rest did not.

        With a capacity this is ``try_add`` one at a time -- which member gets pruned
        depends on the weights after every insertion. Without one nothing is ever
        pruned, so a candidate only has to pass the gates and dedup, and the weights
        are fitted once at the end. Refitting after every insertion costs O(k) inner
        products over ~8M cells plus an O(k) combination for the pool IC, and dedup
        O(k) correlations, so with a few hundred members per-insertion refits were
        most of a block's scoring time. Dedup here runs against every member at once
        (``_DedupIndex``); same threshold, same first offending member."""
        out = Admission()
        items = list(candidates)
        if self.capacity is not None:
            for expr, panel in items:
                if panel is None:
                    out.reject("panel unavailable", expr)
                    continue
                try:
                    self.try_add(expr, np.asarray(panel, dtype=np.float64))
                    out.added.append(expr)
                except Exception as exc:  # noqa: BLE001 - every gate raises; callers want the tally
                    out.reject(f"{type(exc).__name__}: {str(exc)[:60]}", expr)
            return out
        saved_members, saved_weights = list(self.members), self.weights
        index = None
        if self.dedup_threshold is not None:
            index = _DedupIndex(self.target.shape, self.dedup_sample_days)
            for m in self.members:
                index.add(m.z)
        for expr, panel in items:
            if panel is None:
                out.reject("panel unavailable", expr)
                continue
            try:
                z, ic = self._screen(np.asarray(panel, dtype=np.float64))
                if index is not None and index.k:
                    c = index.abs_corr(z)
                    hit = np.flatnonzero(c > self.dedup_threshold)
                    if hit.size:
                        i = int(hit[0])
                        raise DuplicateFactorError(f"|corr| = {c[i]:.4f} with member #{i} {self.members[i].expr!r}")
            except Exception as exc:  # noqa: BLE001
                out.reject(f"{type(exc).__name__}: {str(exc)[:60]}", expr)
                continue
            self.members.append(Member(expr, z, ic))
            if index is not None:
                index.add(z)
            out.added.append(expr)
        if len(self.members) > len(saved_members):
            self.refit()
            if not np.isfinite(self.pool_ic()):
                # Never seen, but the per-insertion path would have refused whichever
                # candidate did this: redo it that way rather than keep a broken pool.
                self.members, self.weights = saved_members, saved_weights
                slow = Admission()
                for expr, panel in items:
                    if panel is None:
                        slow.reject("panel unavailable", expr)
                        continue
                    try:
                        self.try_add(expr, np.asarray(panel, dtype=np.float64))
                        slow.added.append(expr)
                    except Exception as exc:  # noqa: BLE001
                        slow.reject(f"{type(exc).__name__}: {str(exc)[:60]}", expr)
                return slow
        return out


@dataclass
class Admission:
    """What ``PoolBase.admit`` did: members added, rejections by reason, one example each."""
    added: List[Expression] = field(default_factory=list)
    reasons: Dict[str, int] = field(default_factory=dict)
    examples: Dict[str, str] = field(default_factory=dict)

    def reject(self, reason: str, expr: Expression) -> None:
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
        self.examples.setdefault(reason, repr(expr)[:100])


class _DedupIndex:
    """``sampled_abs_corr`` of one candidate against every member at once.

    Members are kept as their sampled rows (the days ``sampled_abs_corr`` takes),
    NaN-zeroed, with finite indicators and squares, so the pairwise-finite sums a
    correlation needs are six matrix-vector products instead of k calls that each
    copy and mask 1.3M cells (~25 ms apiece on the production cube). The one-pass
    formula can differ from ``np.corrcoef`` in the last bits, so a decision can only
    change when |corr| sits within ~1e-12 of the threshold."""

    def __init__(self, shape: Tuple[int, int], sample_days: int) -> None:
        D, S = shape
        self.idx = np.linspace(0, D - 1, num=min(int(sample_days), D)).astype(int)
        self.width = len(self.idx) * S
        self.k = 0
        self._x0 = np.empty((0, self.width))
        self._ind = np.empty((0, self.width))
        self._x2 = np.empty((0, self.width))

    def _rows(self, z: np.ndarray):
        x = np.asarray(z, dtype=np.float64)[self.idx].reshape(-1)
        fin = np.isfinite(x)
        x0 = np.where(fin, x, 0.0)
        return x0, fin.astype(np.float64), x0 * x0

    def add(self, z: np.ndarray) -> None:
        if self.k == self._x0.shape[0]:
            grow = max(16, self.k // 2)
            self._x0, self._ind, self._x2 = (np.concatenate([a, np.empty((grow, self.width))])
                                            for a in (self._x0, self._ind, self._x2))
        self._x0[self.k], self._ind[self.k], self._x2[self.k] = self._rows(z)
        self.k += 1

    def abs_corr(self, z: np.ndarray) -> np.ndarray:
        """|corr| against members 0..k-1, NaN where ``sampled_abs_corr`` gives NaN
        (fewer than two shared finite cells, or no variance on one side)."""
        x0, ix, x2 = self._rows(z)
        Y0, IY, Y2 = self._x0[:self.k], self._ind[:self.k], self._x2[:self.k]
        n = IY @ ix
        sx, sxx = IY @ x0, IY @ x2
        sy, syy = Y0 @ ix, Y2 @ ix
        sxy = Y0 @ x0
        with np.errstate(invalid="ignore", divide="ignore"):
            vx = sxx - sx * sx / n
            vy = syy - sy * sy / n
            c = np.abs((sxy - sx * sy / n) / np.sqrt(vx * vy))
        return np.where((n >= 2) & (vx > 0) & (vy > 0), c, np.nan)


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


class _GramCache:
    """``XᵀX`` and ``Xᵀy`` of a pool's members against its own target, kept across refits.

    ``_lstsq_weights`` stacks every member into an ``(n_cells, k)`` design and
    solves it from scratch, and ``try_add`` does that on every insertion -- twice
    once the pool is full. On the production cube that is ~8M rows: 22 s per solve
    at k = 31 (measured on the server), so scoring one rolling block (up to 120
    candidates) spent most of an hour refitting the pool. The normal equations need
    only the k x k Gram matrix and k inner products with the target: a new member
    costs one row of k + 1 inner products, a dropped member is a deleted row. Same
    least-squares solution up to rounding, not bit-identical to the SVD path.

    Keyed by ``id(member)`` (``Member`` is unhashable); each entry holds the member
    itself, so an id cannot be reused by a new object while it is cached, and stale
    ids are pruned on every solve."""

    def __init__(self, target: np.ndarray) -> None:
        self._target = np.asarray(target, dtype=np.float64)
        y = self._target.reshape(-1)
        self._ok = np.isfinite(y)
        self._y = y[self._ok]
        self.n_ok = int(self._ok.sum())
        self._vec: Dict[int, tuple] = {}            # id(member) -> (member, masked z, z.y)
        self._dot: Dict[tuple, float] = {}          # (id_a, id_b), id_a <= id_b -> z_a.z_b

    def _entry(self, m: Member) -> tuple:
        e = self._vec.get(id(m))
        if e is None:
            v = np.nan_to_num(np.asarray(m.z, dtype=np.float64).reshape(-1)[self._ok], nan=0.0)
            e = self._vec[id(m)] = (m, v, float(v @ self._y))
        return e

    def weights(self, members: Sequence[Member]) -> np.ndarray:
        k = len(members)
        if k == 0:
            return np.zeros(0)
        if self.n_ok < 10 * k:
            return np.full(k, 1.0 / k)
        ids = [id(m) for m in members]
        live = set(ids)
        for i in [i for i in self._vec if i not in live]:
            del self._vec[i]
        for p in [p for p in self._dot if p[0] not in live or p[1] not in live]:
            del self._dot[p]
        entries = [self._entry(m) for m in members]
        keys = [[(ids[a], ids[c]) if ids[a] <= ids[c] else (ids[c], ids[a]) for c in range(k)] for a in range(k)]
        missing = sum(keys[a][c] not in self._dot for a in range(k) for c in range(a, k))
        if missing > 2 * k:
            # Cold cache (a pool filled in one go by admit()): one blocked matrix
            # product instead of k^2/2 separate inner products over ~8M cells --
            # at k = 300 that is ~45k calls, minutes, against seconds for BLAS.
            full = np.zeros((k, k))
            step = max(1, (1 << 25) // k)                  # ~256 MB of stacked rows per block
            for lo in range(0, self.n_ok, step):
                V = np.stack([e[1][lo:lo + step] for e in entries])
                full += V @ V.T
            for a in range(k):
                for c in range(a, k):
                    self._dot[keys[a][c]] = float(full[a, c])
        G, b = np.empty((k, k)), np.empty(k)
        for a in range(k):
            b[a] = entries[a][2]
            for c in range(a, k):
                g = self._dot.get(keys[a][c])
                if g is None:
                    g = self._dot[keys[a][c]] = float(entries[a][1] @ entries[c][1])
                G[a, c] = G[c, a] = g
        w, *_ = np.linalg.lstsq(G, b, rcond=None)
        return w if np.isfinite(w).all() else np.full(k, 1.0 / k)

    def pool_ic(self, members: Sequence[Member], weights: np.ndarray) -> float:
        """``PoolBase.pool_ic`` on the cached vectors. The combination only matters
        where the target is finite (``cs_pearson`` drops the rest), and there the
        per-cell arithmetic -- ``0 + w1*z1 + w2*z2 + ...`` in member order -- is the
        one ``combined()`` does, without its three fresh full-size arrays per member
        (~5 s at k = 30 on the production cube)."""
        acc, tmp = np.zeros(self.n_ok), np.empty(self.n_ok)
        for m, w in zip(members, weights):
            np.multiply(self._entry(m)[1], w, out=tmp)
            acc += tmp
        panel = np.full(self._target.shape, np.nan)
        panel.reshape(-1)[self._ok] = acc
        return nanmean_quiet(cs_pearson(panel, self._target))


class MsePool(PoolBase):
    _gram: Optional[_GramCache] = None

    def _fit(self, z: List[np.ndarray], target: np.ndarray) -> np.ndarray:
        # Arbitrary columns and target (DynamicRollingPool passes a trailing window):
        # the direct solve. The pool's own refits go through the Gram cache below.
        return _lstsq_weights(z, target)

    def refit(self) -> np.ndarray:
        if self._gram is None:
            self._gram = _GramCache(self.target)
        self.weights = self._gram.weights(self.members)
        return self.weights

    def pool_ic(self) -> float:
        if self._gram is None or not self.members or len(self.weights) != len(self.members):
            return super().pool_ic()
        return self._gram.pool_ic(self.members, self.weights)


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
           "DuplicateFactorError", "Member", "Admission"]
