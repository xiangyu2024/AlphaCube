"""The cube store: one float32 ``(S, D, M)`` mmap per field + a ``(S, D)``
stock-day status plane. The on-disk layout is the V2 minute store of the
previous project (``dt_meta.json`` + ``dtfield_<name>.npy`` +
``day_status.npy``), so a server store compiled by ``qmt_ingest`` opens here
unchanged. There is no separate daily store: daily fields are cube
expressions (``day_last(Close)``), see parser.DAILY_MACROS.

Three readers share one interface (``fields / symbols / dates / n_* /
read_cube(name, rows, d0, d1) / read_status(rows, d0, d1)``):

- ``CubeStore``: one directory. Optionally carries a **D-major mirror**
  (``dtfield_<name>.dmajor.npy``, shape ``(D, S, M)``) built by
  ``build_d_major_mirror``; a read of many symbols over few days is then a
  contiguous read instead of one stripe per symbol.
- ``SegmentedCubeStore``: the production layout -- an immutable history base
  (``<root>/minute``) plus append-only date segments listed in
  ``<root>/incremental/index.json`` (each ``<root>/incremental/<rel>/minute``),
  joined by date and symbol only when a slice is read. Same index schema and
  hash contract as the previous project's ``segmented_store``.
- ``open_cube_store(path)`` picks the right one.

``CubeView`` is what the engine consumes: a symbol subset and a day range,
read in either chunk orientation.
"""

from __future__ import annotations

import hashlib
import json
import mmap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

DAY_STATUS_CODES = {"OUTSIDE_LIFECYCLE": 0, "TRADING": 1, "RESUMED": 2, "SUSPENDED": 3}
TRADABLE_CODES = (DAY_STATUS_CODES["TRADING"], DAY_STATUS_CODES["RESUMED"])
META_FILE = "dt_meta.json"
INDEX_SCHEMA_VERSION = "qmt-v2-fast-factor-segment-index-1"


def sha256_file(path: PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


_ADVICE = {True: getattr(mmap, "MADV_RANDOM", None), False: getattr(mmap, "MADV_NORMAL", None)}


def advise_access(arr, random: bool) -> bool:
    """Tell the kernel how a memmap is about to be read, and whether it was told.

    A cube is ``(S, D, M)``, so a read of many symbols over few days -- the D
    round's pattern, and every boundary read in it -- takes ~1 KB out of every
    1.9 MB. The kernel's read-ahead then fetches whole windows around each of
    those crumbs: measured on the production store (9.8 GB field, 6 days x 5207
    symbols, cold ranges) it read 5649 MB for the 28.7 MB actually wanted --
    197x -- in 6.9 s, against 13 MB in 0.37 s with MADV_RANDOM. Extrapolated
    over 1942 days that is 37 minutes a field against 2. It is what made gpu10's
    first D round crawl at 3 days a minute.

    Contiguous reads must keep read-ahead: the same slice read at 47 MB/s under
    MADV_RANDOM, because every page then faults on its own.

    A no-op (returning False) where the platform, numpy or the mapping does not
    support it -- the advice is a performance hint, never a correctness one.
    """
    mm = getattr(arr, "_mmap", None)
    flag = _ADVICE[bool(random)]
    if mm is None or flag is None or not hasattr(mm, "madvise"):
        return False
    try:
        mm.madvise(flag)
    except (OSError, ValueError):                 # e.g. a mapping the kernel will not advise
        return False
    return True


class CubeStore:
    def __init__(self, directory: PathLike) -> None:
        self.directory = Path(directory)
        with open(self.directory / META_FILE, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.fields: List[str] = list(self.meta["fields"])
        self.n_symbols = int(self.meta["n_symbols"])
        self.n_days = int(self.meta["n_days"])
        self.n_minutes = int(self.meta["n_minutes"])
        self.symbols: List[str] = list(self.meta.get("symbols") or [f"s{i}" for i in range(self.n_symbols)])
        self.dates: List[str] = [str(d) for d in (self.meta.get("dates") or range(self.n_days))]
        self._codes = {**DAY_STATUS_CODES, **self.meta.get("day_status_codes", {})}
        self._cache: Dict[str, np.ndarray] = {}
        self._advised: Dict[str, bool] = {}          # field -> last access advice given
        self._mirror: Dict[str, np.ndarray] = {}
        self._status: Optional[np.ndarray] = None
        self.contract_hash = canonical_hash({"meta_sha256": sha256_file(self.directory / META_FILE)})

    # -- raw access ---------------------------------------------------------------------
    def field(self, name: str) -> np.ndarray:
        if name not in self.fields:
            raise KeyError(f"store has no field {name!r}; it has {self.fields}")
        if name not in self._cache:
            arr = np.load(self.directory / f"dtfield_{name}.npy", mmap_mode="r")
            if arr.shape != (self.n_symbols, self.n_days, self.n_minutes):
                raise ValueError(f"field {name}: shape {arr.shape} != meta")
            self._cache[name] = arr
        return self._cache[name]

    def mirror(self, name: str) -> Optional[np.ndarray]:
        """(D, S, M) D-major mirror of ``name`` if one was built, else None."""
        files = self.meta.get("d_major_files", {})
        if name not in files:
            return None
        if name not in self._mirror:
            arr = np.load(self.directory / files[name], mmap_mode="r")
            if arr.shape != (self.n_days, self.n_symbols, self.n_minutes):
                raise ValueError(f"mirror {name}: shape {arr.shape} != meta")
            self._mirror[name] = arr
        return self._mirror[name]

    def day_status(self) -> np.ndarray:
        if self._status is None:
            f = self.meta.get("day_status_file")
            if not f:
                raise FileNotFoundError("store has no day_status_file")
            st = np.load(self.directory / f, mmap_mode="r")
            if st.shape != (self.n_symbols, self.n_days):
                raise ValueError(f"day_status shape {st.shape} != ({self.n_symbols}, {self.n_days})")
            self._status = st
        return self._status

    def tradable(self) -> np.ndarray:
        return np.isin(np.asarray(self.day_status()), [self._codes["TRADING"], self._codes["RESUMED"]])

    # -- slice interface ---------------------------------------------------------------
    def read_cube(self, name: str, rows, d0: int, d1: int) -> np.ndarray:
        """(len(rows), d1 - d0, M) float32 copy; ``rows`` is a slice or an index array."""
        mirror = self.mirror(name)
        n_rows = (rows.stop - rows.start) if isinstance(rows, slice) else len(rows)
        strided = (d1 - d0) * 4 <= n_rows                           # few days, many symbols
        if mirror is not None and strided:                          # contiguous in the mirror
            block = mirror[d0:d1]                                     # (d, S, M)
            block = block[:, rows, :] if not isinstance(rows, slice) else block[:, rows, :]
            return np.ascontiguousarray(np.transpose(np.asarray(block, dtype=np.float32), (1, 0, 2)))
        arr = self.advised_field(name, strided)
        return np.array(arr[rows, d0:d1, :], dtype=np.float32, copy=True)

    def advised_field(self, name: str, strided: bool) -> np.ndarray:
        """``field(name)`` with the kernel told how it is about to be read.
        Read-ahead is worth its cost only on the contiguous pattern; on the strided
        one it multiplied the bytes off the disk by ~350 (``advise_access``). Every
        reader of a field must come through here -- ``SegmentedCubeStore`` reads
        its segments' fields directly, and for a day it did not."""
        arr = self.field(name)
        if self._advised.get(name) is not strided:
            advise_access(arr, random=strided)
            self._advised[name] = strided
        return arr

    def read_status(self, rows, d0: int, d1: int) -> np.ndarray:
        return np.array(self.day_status()[rows, d0:d1], dtype=np.int8, copy=True)

    # -- selection ------------------------------------------------------------------------
    def date_range(self, start: Optional[str], end: Optional[str]) -> Tuple[int, int]:
        """[d0, d1): first trading day >= start, last <= end (calendar bounds snap)."""
        d = np.asarray([int(x) for x in self.dates])
        d0 = int(np.searchsorted(d, int(start))) if start else 0
        d1 = int(np.searchsorted(d, int(end), side="right")) if end else len(d)
        if d0 >= d1:
            raise ValueError(f"no trading days in [{start}, {end}]")
        return d0, d1

    def view(self, symbols: Optional[Sequence[str]] = None, max_symbols: Optional[int] = None,
             start: Optional[str] = None, end: Optional[str] = None) -> "CubeView":
        if symbols is not None:
            index = {s: i for i, s in enumerate(self.symbols)}
            missing = [s for s in symbols if s not in index]
            if missing:
                raise KeyError(f"symbols not in store: {missing[:10]}")
            sym_idx = np.asarray([index[s] for s in symbols], dtype=np.int64)
        else:
            sym_idx = np.arange(self.n_symbols, dtype=np.int64)
        if max_symbols is not None:
            sym_idx = sym_idx[:max_symbols]
        if sym_idx.size == 0:
            raise ValueError("no symbols selected")
        d0, d1 = self.date_range(start, end)
        return CubeView(self, sym_idx, d0, d1)


# ---- segmented (base + incremental segments) ----------------------------------------------

@dataclass(frozen=True)
class _Source:
    root: Path
    store: CubeStore
    dates: Tuple[str, ...]
    symbols: Tuple[str, ...]
    meta_sha256: Optional[str]


def _load_index(compute_root: Path) -> Tuple[Path, dict]:
    index_path = compute_root / "incremental" / "index.json"
    if not index_path.is_file():
        return index_path, {"schema_version": INDEX_SCHEMA_VERSION, "run_status": "COMPLETE", "segments": []}
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise ValueError(f"unsupported incremental store index: {index_path}")
    if index.get("run_status") != "COMPLETE":
        raise ValueError(f"incremental store index is not COMPLETE: {index_path}")
    if not isinstance(index.get("segments"), list):
        raise ValueError(f"incremental store index has invalid segments: {index_path}")
    return index_path, index


def _run(idx: np.ndarray):
    """``idx`` as a slice when it is one ascending run of consecutive positions."""
    if idx.size and (idx.size == 1 or bool((np.diff(idx) == 1).all())):
        return slice(int(idx[0]), int(idx[-1]) + 1)
    return idx


def _outer(rows: np.ndarray, days: np.ndarray):
    """(rows x days) outer index. One array next to a slice already indexes as an
    outer product; ``np.ix_`` is needed only when neither axis is a run."""
    r, d = _run(rows), _run(days)
    if isinstance(r, slice) or isinstance(d, slice):
        return r, d
    return np.ix_(r, d)


class SegmentedCubeStore:
    """History base ``<root>/minute`` + segments ``<root>/incremental/<rel>/minute``."""

    def __init__(self, compute_root: PathLike, include_segments: bool = True) -> None:
        self.compute_root = Path(compute_root)
        base = CubeStore(self.compute_root / "minute")
        sources = [_Source(self.compute_root, base, tuple(base.dates), tuple(base.symbols), None)]
        index_path, index = _load_index(self.compute_root)
        seen = set(base.dates)
        if include_segments:
            for entry in index.get("segments", []):
                rel = entry.get("relative_path")
                if not isinstance(rel, str) or Path(rel).is_absolute():
                    raise ValueError("incremental segment relative_path is invalid")
                root = self.compute_root / "incremental" / rel
                meta_path = root / "meta.json"
                if not meta_path.is_file():
                    raise FileNotFoundError(f"segment meta is missing: {meta_path}")
                actual = sha256_file(meta_path)
                if actual != entry.get("meta_sha256"):
                    raise ValueError(f"segment meta hash mismatch: {meta_path}")
                if json.loads(meta_path.read_text(encoding="utf-8")).get("run_status") != "COMPLETE":
                    raise ValueError(f"segment is not COMPLETE: {meta_path}")
                seg = CubeStore(root / "minute")
                if tuple(seg.dates) != tuple(map(str, entry.get("dates", []))):
                    raise ValueError(f"segment dates differ from index: {root}")
                if seg.n_minutes != base.n_minutes or seg.fields != base.fields:
                    raise ValueError(f"segment axes/fields differ from base: {root}")
                overlap = seen.intersection(seg.dates)
                if overlap:
                    raise ValueError(f"duplicate dates across stores: {sorted(overlap)[:3]}")
                seen.update(seg.dates)
                sources.append(_Source(root, seg, tuple(seg.dates), tuple(seg.symbols), actual))
        self._sources = sources
        self.dates: List[str] = [d for s in sources for d in s.dates]
        if self.dates != sorted(self.dates):
            raise ValueError("base and incremental store dates must be increasing")
        base_symbols = list(base.symbols)
        extra = sorted(set().union(*(set(s.symbols) for s in sources[1:])) - set(base_symbols)) if len(sources) > 1 else []
        self.symbols: List[str] = base_symbols + extra
        self.fields = list(base.fields)
        self.n_symbols, self.n_days, self.n_minutes = len(self.symbols), len(self.dates), base.n_minutes
        global_symbol = {s: i for i, s in enumerate(self.symbols)}
        # Read maps, built once rather than per slice: the source holding each
        # global day and its position there, and each global symbol's row in each
        # source (-1 where the source does not list it).
        self._day_src = np.repeat(np.arange(len(sources)), [len(s.dates) for s in sources])
        self._day_pos = np.concatenate([np.arange(len(s.dates)) for s in sources])
        self._sym_row = []
        for src in sources:
            m = np.full(self.n_symbols, -1, dtype=np.int64)
            m[[global_symbol[s] for s in src.symbols]] = np.arange(len(src.symbols))
            self._sym_row.append(m)
        self._codes = dict(base._codes)
        self.meta = {**base.meta, "n_symbols": self.n_symbols, "n_days": self.n_days, "symbols": self.symbols,
                     "dates": self.dates, "segmented": True}
        self.contract = {
            "base_minute_meta_sha256": sha256_file(self.compute_root / "minute" / META_FILE),
            "index_sha256": sha256_file(index_path) if index_path.is_file() else None,
            "segment_meta_sha256": [s.meta_sha256 for s in sources[1:]],
        }
        self.contract_hash = canonical_hash(self.contract)
        self.directory = self.compute_root

    def _pieces(self, rows, d0: int, d1: int):
        """Per source that holds any of the cells: ``(store, out_index, src_index)``,
        each a (rows, days) outer index into the output and into that source.

        A run of consecutive positions comes back as a slice, which is what keeps
        a read a stripe copy. The previous version indexed all three axes with
        ``np.ix_`` -- the minute axis included -- which gathers element by
        element: 21-28x slower than a slice on the production store's chunk
        shapes (58 ms vs 2.1-2.7 ms for 15 symbols x 1942 days), and the leaf
        read was the engine's largest cost on the GPU."""
        rows_g = np.arange(self.n_symbols)[rows] if isinstance(rows, slice) else np.asarray(rows, dtype=np.int64)
        day_src = self._day_src[d0:d1]
        for k in np.unique(day_src):
            out_days = np.flatnonzero(day_src == k)
            src_rows = self._sym_row[k][rows_g]
            out_rows = np.flatnonzero(src_rows >= 0)
            if out_rows.size:
                yield (self._sources[k].store, _outer(out_rows, out_days),
                       _outer(src_rows[out_rows], self._day_pos[d0:d1][out_days]))

    def read_cube(self, name: str, rows, d0: int, d1: int) -> np.ndarray:
        if name not in self.fields:
            raise KeyError(name)
        n_rows = len(range(self.n_symbols)[rows]) if isinstance(rows, slice) else len(rows)
        out = np.empty((n_rows, d1 - d0, self.n_minutes), dtype=np.float32)
        covered = np.zeros((n_rows, d1 - d0), dtype=bool)
        # The same pattern test as CubeStore.read_cube. This reader used to take
        # store.field() directly and so skipped the advice altogether: gpu12's
        # third-block rebuild ran its D round at 5 days a minute, 987 MB/s off the
        # disk, with no "rr" on any of the eight field mappings.
        strided = (d1 - d0) * 4 <= n_rows
        for store, out_ix, src_ix in self._pieces(rows, d0, d1):
            out[out_ix] = store.advised_field(name, strided)[src_ix]
            covered[out_ix] = True
        if not covered.all():
            out[~covered] = np.nan
        return out

    def read_status(self, rows, d0: int, d1: int) -> np.ndarray:
        n_rows = len(range(self.n_symbols)[rows]) if isinstance(rows, slice) else len(rows)
        out = np.full((n_rows, d1 - d0), self._codes["OUTSIDE_LIFECYCLE"], dtype=np.int8)
        for store, out_ix, src_ix in self._pieces(rows, d0, d1):
            out[out_ix] = np.asarray(store.day_status())[src_ix]
        return out

    def day_status(self) -> np.ndarray:
        return self.read_status(slice(0, self.n_symbols), 0, self.n_days)

    def tradable(self) -> np.ndarray:
        return np.isin(self.day_status(), [self._codes["TRADING"], self._codes["RESUMED"]])

    date_range = CubeStore.date_range
    view = CubeStore.view


def open_cube_store(path: PathLike, include_segments: bool = True):
    """A compute root (``<root>/minute`` exists) opens as a SegmentedCubeStore, a
    bare minute directory as a CubeStore."""
    p = Path(path)
    if (p / "minute" / META_FILE).is_file():
        return SegmentedCubeStore(p, include_segments)
    return CubeStore(p)


class CubeView:
    """Symbol subset x day range of a store, readable in either chunk orientation."""

    def __init__(self, store, sym_idx: np.ndarray, d0: int, d1: int) -> None:
        self.store = store
        self.sym_idx = np.asarray(sym_idx, dtype=np.int64)
        self.d0, self.d1 = int(d0), int(d1)
        self.n_symbols = int(self.sym_idx.size)
        self.n_days = self.d1 - self.d0
        self.n_minutes = store.n_minutes
        self.symbols = [store.symbols[i] for i in self.sym_idx]
        self.dates = store.dates[self.d0:self.d1]
        self.fields = list(store.fields)
        self._contiguous = bool(np.all(np.diff(self.sym_idx) == 1)) if self.n_symbols > 1 else True
        self._tradable: Optional[np.ndarray] = None

    def _rows(self, s_lo: int, s_hi: int):
        if self._contiguous:
            return slice(int(self.sym_idx[s_lo]), int(self.sym_idx[s_hi - 1]) + 1)
        return self.sym_idx[s_lo:s_hi]

    def read_s(self, name: str, s_lo: int, s_hi: int) -> np.ndarray:
        """(C, D, M) float32 for symbols [s_lo, s_hi) of the view."""
        return self.store.read_cube(name, self._rows(s_lo, s_hi), self.d0, self.d1)

    def read_d(self, name: str, d_lo: int, d_hi: int) -> np.ndarray:
        """(S, C, M) float32 for view days [d_lo, d_hi)."""
        return self.store.read_cube(name, self._rows(0, self.n_symbols), self.d0 + d_lo, self.d0 + d_hi)

    def tradable(self) -> np.ndarray:
        """(S, D) bool for the whole view."""
        if self._tradable is None:
            st = self.store.read_status(self._rows(0, self.n_symbols), self.d0, self.d1)
            codes = getattr(self.store, "_codes", DAY_STATUS_CODES)
            self._tradable = np.isin(st, [codes["TRADING"], codes["RESUMED"]])
        return self._tradable

    def tradable_s(self, s_lo: int, s_hi: int) -> np.ndarray:
        return self.tradable()[s_lo:s_hi]

    def tradable_d(self, d_lo: int, d_hi: int) -> np.ndarray:
        return self.tradable()[:, d_lo:d_hi]


# ---- builders ------------------------------------------------------------------------------

def build_cube_store(out_dir: PathLike, cubes: Dict[str, np.ndarray], symbols: Sequence[str],
                     dates: Sequence[str], day_status: Optional[np.ndarray] = None,
                     extra_meta: Optional[dict] = None) -> CubeStore:
    """Write a store from in-memory (S, D, M) cubes (rounded to float32 once)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = sorted(cubes)
    shapes = {cubes[n].shape for n in names}
    if len(shapes) != 1:
        raise ValueError(f"all cubes must share one (S, D, M) shape, got {shapes}")
    S, D, M = next(iter(shapes))
    if len(symbols) != S or len(dates) != D:
        raise ValueError("symbols/dates length must match the cube axes")
    for n in names:
        np.save(out / f"dtfield_{n}.npy", np.ascontiguousarray(cubes[n], dtype=np.float32))
    if day_status is None:
        day_status = np.full((S, D), DAY_STATUS_CODES["TRADING"], dtype=np.int8)
    day_status = np.ascontiguousarray(day_status, dtype=np.int8)
    if day_status.shape != (S, D):
        raise ValueError(f"day_status shape {day_status.shape} != {(S, D)}")
    np.save(out / "day_status.npy", day_status)
    meta = {
        "fields": names, "n_symbols": int(S), "n_days": int(D), "n_minutes": int(M),
        "symbols": [str(s) for s in symbols], "dates": [str(d) for d in dates],
        "day_status_file": "day_status.npy", "day_status_codes": DAY_STATUS_CODES,
        "label_used": False,
    }
    meta.update(extra_meta or {})
    with open(out / META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    return CubeStore(out)


def build_d_major_mirror(store: CubeStore, fields: Optional[Sequence[str]] = None,
                         days_per_chunk: int = 64) -> CubeStore:
    """Write ``dtfield_<name>.dmajor.npy`` (D, S, M) for each field and register
    them in the meta; reads over many symbols and few days become contiguous.
    Built in day chunks so peak memory is one chunk, not the whole cube."""
    names = list(fields) if fields is not None else list(store.fields)
    files = dict(store.meta.get("d_major_files", {}))
    for name in names:
        src = store.field(name)
        path = store.directory / f"dtfield_{name}.dmajor.npy"
        out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                        shape=(store.n_days, store.n_symbols, store.n_minutes))
        for lo in range(0, store.n_days, days_per_chunk):
            hi = min(lo + days_per_chunk, store.n_days)
            out[lo:hi] = np.transpose(np.asarray(src[:, lo:hi, :]), (1, 0, 2))
        out.flush()
        del out
        files[name] = path.name
    store.meta["d_major_files"] = files
    with open(store.directory / META_FILE, "w", encoding="utf-8") as f:
        json.dump(store.meta, f, ensure_ascii=False)
    store._mirror.clear()
    return store


def write_segment(compute_root: PathLike, relative_path: str, cubes: Dict[str, np.ndarray],
                  symbols: Sequence[str], dates: Sequence[str], day_status: Optional[np.ndarray] = None) -> Path:
    """Append one date segment under ``<root>/incremental/<relative_path>`` and
    register it in ``index.json`` (the previous project's schema)."""
    root = Path(compute_root)
    seg_root = root / "incremental" / relative_path
    build_cube_store(seg_root / "minute", cubes, symbols, dates, day_status)
    meta_path = seg_root / "meta.json"
    meta_path.write_text(json.dumps({"run_status": "COMPLETE", "dates": [str(d) for d in dates],
                                     "symbols": [str(s) for s in symbols]}, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    index_path, index = _load_index(root)
    index.setdefault("segments", []).append({"relative_path": relative_path, "meta_sha256": sha256_file(meta_path),
                                             "dates": [str(d) for d in dates]})
    index["schema_version"], index["run_status"] = INDEX_SCHEMA_VERSION, "COMPLETE"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return seg_root


__all__ = ["CubeStore", "SegmentedCubeStore", "CubeView", "open_cube_store", "build_cube_store",
           "advise_access", "build_d_major_mirror", "write_segment", "DAY_STATUS_CODES", "TRADABLE_CODES", "INDEX_SCHEMA_VERSION",
           "sha256_file", "canonical_hash"]
