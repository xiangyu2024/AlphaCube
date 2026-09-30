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


class RankSimilarityBan:
    def __init__(self, shape: Tuple[int, int], threshold: float, sample_days: int = 250,
                 exemplar_dedup: float = 0.99, sample_symbols: int = 1024) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = float(threshold)
        self.exemplar_dedup = float(exemplar_dedup)
        # Sampled symbols in float32 (`CompactRankIndex`), not every symbol in
        # float64: 3.1 MB an exemplar instead of 31 MB. gpu14 held 748 exemplars
        # -- 23 GB of a 90 GiB container, second only to its panel cache -- and
        # died at the block-3 boundary where the pool fit and four tree models
        # land on top of that. 1024 symbols is twice what the family index uses
        # at a stricter threshold; the decisions are the same (test_rankban).
        self.index = CompactRankIndex(shape, sample_days, sample_symbols)
        self.exemplars: List[str] = []          # repr of each stored exemplar, in index order
        self.version = 0                         # bumps whenever an exemplar is stored

    def __len__(self) -> int:
        return self.index.k

    def add(self, expr, panel: np.ndarray) -> bool:
        """Store ``panel`` as an exemplar unless one already ranks like it."""
        return self.add_rows(expr, self.index.rows(panel))

    def add_rows(self, expr, rows: np.ndarray) -> bool:
        """同上,但直接吃秩指纹 —— 缓存命中时面板根本没被算出来过。"""
        if self.index.k and np.any(self.index.corr(rows[:, None, :])[0] >= self.exemplar_dedup):
            return False
        self.index.add(rows)
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

    def abs_corr(self, panel: np.ndarray) -> np.ndarray:
        """|corr| of one panel against every stored row -> (k,)."""
        return self.corr(self.rows(panel)[:, None, :])[0]

    @property
    def nbytes(self) -> int:
        """Bytes the index holds, allocation included (it grows by k // 2)."""
        return int(self._y.nbytes)

    @classmethod
    def pairwise(cls, X: np.ndarray) -> np.ndarray:
        """|corr| among the candidates ``X`` (3, m, w) themselves -> (m, m)."""
        return cls._corr(X, X)


class ExemplarCache:
    """把 ban exemplar 的秩指纹落盘,下次直接读,不再重算面板。

    每轮开跑前都要把 ban-archive 里的表达式全评估一遍,只为了拿它们的秩指纹:
    gpu15 装 1049 条花了 1 小时,gpu17 装 1673 条花了 **2 小时 04 分** —— 一轮
    四个多小时里三分之一在重算前几轮已经算过的东西,而且 ban 列表每轮都在长。
    一条指纹是 3 x 250 x 1024 x 4 B = 3.07 MB,2500 条 7.7 GB,读回来是几秒。

    文件按"视图键 + 表达式哈希"存:指纹取决于哪些天、哪些股票、采样几天几只,
    视图一换,旧的全部作废 —— 所以**构造时就把其它视图键的目录删掉**,那些文件
    再也不会被用到。当前视图的目录超过 ``max_bytes`` 时按最近使用时间淘汰。
    """

    def __init__(self, root, view_key: str, max_bytes: int = 12 * 1024 ** 3) -> None:
        from pathlib import Path

        self.root = Path(root)
        self.view_key = str(view_key)
        self.max_bytes = int(max_bytes)
        self.dir = self.root / self.view_key
        self.hits = 0
        self.misses = 0
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dropped_views = self._drop_other_views()

    def _drop_other_views(self) -> int:
        """别的视图键的目录:那些指纹和当前这批数据对不上,永远用不到了。"""
        import shutil

        dropped = 0
        for child in self.root.iterdir():
            if child.is_dir() and child.name != self.view_key:
                shutil.rmtree(child, ignore_errors=True)
                dropped += 1
        return dropped

    @staticmethod
    def _name(expression: str) -> str:
        import hashlib

        return hashlib.sha1(expression.encode("utf-8")).hexdigest()[:24] + ".npy"

    def get(self, expression: str) -> Optional[np.ndarray]:
        path = self.dir / self._name(expression)
        try:
            rows = np.load(path)
        except (OSError, ValueError):
            self.misses += 1
            return None
        path.touch()                      # 最近使用时间,淘汰时看它
        self.hits += 1
        return np.ascontiguousarray(rows, dtype=np.float32)

    def put(self, expression: str, rows: np.ndarray) -> None:
        path = self.dir / self._name(expression)
        # 临时名也必须以 .npy 结尾:np.save 会自作主张补后缀,写出 xxx.tmp.npy,
        # 然后替换 xxx.tmp 找不到文件 —— 缓存会**静默地**一条都存不下来。
        tmp = path.with_name(path.name + ".tmp.npy")
        try:
            np.save(tmp, np.ascontiguousarray(rows, dtype=np.float32))
            tmp.replace(path)             # 原子替换:半个文件比没有文件更糟
        except OSError:
            tmp.unlink(missing_ok=True)

    def prune(self) -> int:
        """超过上限就按最近使用时间从旧到新删,返回删了几个。"""
        if self.max_bytes <= 0:
            return 0
        files = sorted(self.dir.glob("*.npy"), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        removed = 0
        for path in files:
            if total <= self.max_bytes:
                break
            total -= path.stat().st_size
            path.unlink(missing_ok=True)
            removed += 1
        return removed

    def summary(self) -> str:
        n = len(list(self.dir.glob("*.npy")))
        gb = sum(p.stat().st_size for p in self.dir.glob("*.npy")) / 1024 ** 3
        return f"exemplar 缓存 {n} 条 / {gb:.1f} GB(命中 {self.hits},未命中 {self.misses})"


def view_cache_key(view, sample_days: int, sample_symbols: int) -> str:
    """指纹取决于这些东西,任何一个变了,缓存就作废。"""
    import hashlib

    dates = [str(x) for x in view.dates]
    raw = "|".join([dates[0], dates[-1], str(len(dates)), str(view.n_symbols),
                    str(int(sample_days)), str(int(sample_symbols))])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


__all__ = ["RankSimilarityBan", "CompactRankIndex", "ExemplarCache", "view_cache_key"]
