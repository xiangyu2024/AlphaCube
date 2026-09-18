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


__all__ = ["RankSimilarityBan"]
