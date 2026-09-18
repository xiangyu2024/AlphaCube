"""Ban by what a factor ranks like, not only by how it is written.

A structural ban (``one_layer_keys``) forbids a depth-1 subtree such as
``log(Amt)``. The search walks straight around it with a monotone rewrite: with
``log(Amt)`` banned, gpu7's first block came back with ``log(pow_1_5(Amt))`` and
``pow_05(pow_05(day_max(Amt)))`` -- the same daily cross-sectional ranks as
``day_max(log(Amt))``, hence the identical backtest to the fourth decimal.

``RankSimilarityBan`` keeps panels of the banned families (exemplars) and refuses
any candidate whose panel ranks like one of them: each sampled day's
cross-section is ranked (ties averaged) and standardised, and the pooled
correlation over the sampled days -- pairwise-finite cells, as the pool's dedup
takes them -- is compared with ``threshold`` in absolute value (a sign flip
trades the same book). Ranks rather than values make every monotone rewrite of an
exemplar score exactly 1.0, which Pearson on values does not.

Exemplars that rank alike to ``exemplar_dedup`` are stored once: rewrites of one
factor are all 1.0 to each other, and keeping them all would only cost memory
(3 x 1.3M float64 per exemplar on the production cube).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..metrics import rank_rows, zscore_rows
from .pool import _DedupIndex


class _RankIndex(_DedupIndex):
    """``_DedupIndex`` on standardised daily ranks instead of values."""

    def _rows(self, panel: np.ndarray):
        r = zscore_rows(rank_rows(np.asarray(panel, dtype=np.float64)[self.idx]))
        x = r.reshape(-1)
        fin = np.isfinite(x)
        x0 = np.where(fin, x, 0.0)
        return x0, fin.astype(np.float64), x0 * x0


class RankSimilarityBan:
    def __init__(self, shape: Tuple[int, int], threshold: float, sample_days: int = 250,
                 exemplar_dedup: float = 0.99) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = float(threshold)
        self.exemplar_dedup = float(exemplar_dedup)
        self.index = _RankIndex(shape, sample_days)
        self.exemplars: List[str] = []          # repr of each stored exemplar, in index order
        self.version = 0                         # bumps whenever an exemplar is stored

    def __len__(self) -> int:
        return self.index.k

    def add(self, expr, panel: np.ndarray) -> bool:
        """Store ``panel`` as an exemplar unless one already ranks like it."""
        if self.index.k:
            c = self.index.abs_corr(panel)
            if np.any(c >= self.exemplar_dedup):
                return False
        self.index.add(panel)
        self.exemplars.append(repr(expr))
        self.version += 1
        return True

    def match(self, panel: np.ndarray) -> Optional[Tuple[str, float]]:
        """(exemplar, |rank corr|) of the first exemplar ``panel`` ranks like, or None."""
        if not self.index.k:
            return None
        c = self.index.abs_corr(panel)
        hit = np.flatnonzero(c >= self.threshold)
        if not hit.size:
            return None
        i = int(hit[0])
        return self.exemplars[i], float(c[i])


class CompactRankIndex:
    """Standardised daily ranks on a fixed sample of days AND symbols, stored as
    float32 -- one row per archived factor family (``GPMiner(archive_dedup=...)``).

    ``_RankIndex`` keeps every symbol of 250 days in float64: 31 MB a factor,
    affordable for a few hundred ban exemplars but not for an archive that grows
    by ~35 factors a generation. 250 days x 512 symbols in float32 is 1.5 MB, and
    at the 0.99 level this index decides the sampling error of the estimate is
    ~5e-5 -- far inside the gap between a rewrite (1.0) and a different factor.
    The symbols are ranked among themselves; a monotone rewrite still scores 1.0.

    Correlations are pairwise-finite, from the same six sums ``_DedupIndex`` uses,
    batched as matrix products over many candidates at once; the sums come out of
    float32 BLAS (~1e-5 relative), the final arithmetic is float64."""

    def __init__(self, shape: Tuple[int, int], sample_days: int = 250, sample_symbols: int = 512) -> None:
        D, S = shape
        self.days = np.linspace(0, D - 1, num=min(int(sample_days), D)).astype(int)
        self.cols = np.linspace(0, S - 1, num=min(int(sample_symbols), S)).astype(int)
        self.width = len(self.days) * len(self.cols)
        self.k = 0
        self._y = np.empty((3, 0, self.width), dtype=np.float32)      # x0, finite indicator, x0**2

    def __len__(self) -> int:
        return self.k

    def rows(self, panel: np.ndarray) -> np.ndarray:
        """``(3, width)`` float32: the panel's sampled z-scored ranks (NaN -> 0), the
        finite indicator, and the squares."""
        sub = np.asarray(panel)[np.ix_(self.days, self.cols)].astype(np.float64)   # sample, then widen
        z = zscore_rows(rank_rows(sub)).reshape(-1)
        fin = np.isfinite(z)
        x0 = np.where(fin, z, 0.0)
        return np.stack([x0, fin.astype(np.float64), x0 * x0]).astype(np.float32)

    def add(self, rows: np.ndarray) -> int:
        """Store ``rows``; returns its slot."""
        if self.k == self._y.shape[1]:
            grow = max(16, self.k // 2)
            self._y = np.concatenate([self._y, np.empty((3, grow, self.width), dtype=np.float32)], axis=1)
        self._y[:, self.k] = rows
        self.k += 1
        return self.k - 1

    def set(self, slot: int, rows: np.ndarray) -> None:
        self._y[:, slot] = rows

    @staticmethod
    def _corr(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """|corr| between every row of ``X`` (3, m, w) and of ``Y`` (3, k, w) -> (m, k);
        NaN with fewer than two shared finite cells or no variance on one side."""
        x0, ix, x2 = X
        y0, iy, y2 = Y
        f = lambda a, b: (a @ b.T).astype(np.float64)       # noqa: E731
        n, sx, sxx = f(ix, iy), f(x0, iy), f(x2, iy)
        sy, syy, sxy = f(ix, y0), f(ix, y2), f(x0, y0)
        with np.errstate(invalid="ignore", divide="ignore"):
            vx = sxx - sx * sx / n
            vy = syy - sy * sy / n
            c = np.abs((sxy - sx * sy / n) / np.sqrt(vx * vy))
        return np.where((n >= 2) & (vx > 0) & (vy > 0), c, np.nan)

    def corr(self, X: np.ndarray) -> np.ndarray:
        """|corr| of candidates ``X`` (3, m, w) against every stored row -> (m, k)."""
        return self._corr(X, self._y[:, :self.k])

    @classmethod
    def pairwise(cls, X: np.ndarray) -> np.ndarray:
        """|corr| among the candidates ``X`` (3, m, w) themselves -> (m, m)."""
        return cls._corr(X, X)


__all__ = ["RankSimilarityBan", "CompactRankIndex"]
