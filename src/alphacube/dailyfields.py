"""The (S, D) data side of the daily-field namespace (``catalog.DAILY_FIELDS``):
nine names in three kinds, all resolved by :func:`resolve`.

- **stored** (``FloatCap``, ``TotalCap``, ``FloatShares``): read from a
  ``DailyFieldStore`` directory -- ``daily_<name>.npy`` (float64, mmap'd) next to
  one ``meta.json`` naming the symbol/date labels those rows/columns are in.
- **derived** (``LimitWidth``, ``BoardCode``, ``ListAge``): computed by
  :func:`derive` from the symbol/date labels (and, for ``ListAge``, a per-stock-day
  lifecycle signal) alone -- no store, no build step.
- **log** (``LogFloatCap``, ``LogTotalCap``, ``LogFloatShares``): ``log`` of the
  matching stored plane, taken in float64 *before* the cast to float32, per the
  design (the structural contract this module implements -- see ``resolve``).
  CAUTION, measured rather than asserted (see this work package's ``deviations`` /
  notes for the numbers): logging before the cast does NOT give
  ``ts_delta(LogFloatCap, 1)`` better precision than ``ts_delta(FloatCap, 1)`` at
  this magnitude. For a 1e9-1e13 cap, ``log(cap)`` is only ~21-30, so float32's
  fixed ~6e-8 RELATIVE precision on that small logged number is a LARGER share of
  a ~1e-3-scale log-return than the same ~6e-8 relative precision on the raw cap
  is of the raw day-over-day move -- the log transform amplifies, not shrinks, the
  cancellation in a subsequent ``ts_delta``. Measured across cap in
  (1e9, 1e12, 1e13) and move in (1e-2 .. 1e-8): ``ts_delta(FloatCap, 1)`` recovers
  the true move with LOWER relative error than ``ts_delta(LogFloatCap, 1)`` in
  every case tried, by roughly a factor of ``log(cap)`` (~20-40x). The log fields
  stay useful for what logging a size variable is actually for -- a roughly
  symmetric, outlier-tamed distribution for ``cs_rank`` / linear-model use -- just
  not for the precision reason the design states.

This module imports nothing from ``store.py`` or ``engine/``, so it is
unit-testable on plain symbol/date label lists, independent of the rest of the
cube machinery. ``store.CubeView.daily(name)`` (the engine-facing side) is what
actually calls :func:`resolve` with a real view's labels and its own
``tradable()`` / ``day_status()`` planes.

Every entry point here is given a VIEW's labels, which may be any subset of the
universe, so must_fix #2's symbol-spelling guard is split in two:
:func:`derive` raises only on a spelling no board rule could ever match and
warns when the spelling is fine but no rule fired (normal for a main-board-only
subset), while :func:`require_board_rule` is the strict assertion for a caller
holding the FULL symbol list -- the store's daily-source attach and
``scripts/build_daily_fields.py``.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .tradability import board_limit

STORED_FIELDS: Tuple[str, ...] = ("FloatCap", "TotalCap", "FloatShares")
DERIVED_FIELDS: Tuple[str, ...] = ("LimitWidth", "BoardCode", "ListAge")
LOG_OF: Dict[str, str] = {"LogFloatCap": "FloatCap", "LogTotalCap": "TotalCap", "LogFloatShares": "FloatShares"}
# Already listed on the store's first day -- the data cannot say how old such a
# stock really is, so it is marked rather than guessed at. Partura's cross_age
# (ops_ext.py) uses the same constant for the same reason; derive()'s ListAge
# follows its exact convention (see _list_age), just transposed to (S, D).
LIST_AGE_SENTINEL = 10000.0
# Bumped when derive()'s formulas change, and mixed into fingerprint() so a stale
# on-disk ExemplarCache cannot silently serve rank fingerprints computed under an
# older definition of a derived field.
DERIVED_VERSION = 1

# BoardCode's four categories. Not pinned by any upstream convention -- this
# module is the one place that invents the mapping -- but exported so a caller
# never has to spell the magic numbers 0..3 itself.
BOARD_MAIN, BOARD_STAR, BOARD_GEM, BOARD_BSE = 0.0, 1.0, 2.0, 3.0

# The same STAR / GEM / BSE prefix rule as tradability.board_limit (688/689,
# 300/301/302, 4/8/92), duplicated here -- not imported -- because board_limit
# does not expose the three boolean masks it computes internally, and BoardCode
# needs exactly those masks, not the limit percentages, for must_fix #2's "two
# cannot disagree" guarantee (see _board_masks).
_STAR_PREFIXES = ("688", "689")
_GEM_PREFIXES = ("300", "301", "302")
_BSE_PREFIXES = ("4", "8", "92")

# store.DAY_STATUS_CODES["OUTSIDE_LIFECYCLE"], duplicated as a bare int rather
# than imported: this module does not import store.py (see the module
# docstring), and the convention -- 0 means "not yet listed / already
# delisted" -- is one that every status plane in the codebase already follows.
_OUTSIDE_LIFECYCLE = 0

# resolve() warns when fewer than this fraction of the requested (S, D) cells
# came back finite. A daily store that carries, say, 3 of 5000 symbols is a
# label mismatch, not data; see _warn_once_if_late_start.
LOW_COVERAGE_FLOOR = 0.01


def _board_masks(symbols: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(star, gem, bse) boolean arrays over ``symbols``, one entry per symbol."""
    star = np.array([s.startswith(_STAR_PREFIXES) for s in symbols], dtype=bool)
    gem = np.array([s.startswith(_GEM_PREFIXES) for s in symbols], dtype=bool)
    bse = np.array([s.startswith(_BSE_PREFIXES) for s in symbols], dtype=bool)
    return star, gem, bse


def _require_prefix_first(symbols: Sequence[str]) -> None:
    """must_fix #2's FORMAT half, cheap enough to run on every ``derive`` call:
    raise unless at least one symbol could possibly match a board rule, i.e.
    starts with a digit.

    Measured (must_fix #2): on a ``SH.600000``-style spelling every board rule
    misses, so ``board_limit`` quietly returns a uniform 0.10 plane and a
    same-prefix ``BoardCode`` would be one constant category at 100% finite
    coverage -- wrong, and nothing else catches it. ``'SH.600000'[:1]`` is not a
    digit, so this catches that spelling (and any other prefix-last one) without
    asking anything about which boards the symbols happen to be on.

    It deliberately does NOT ask "did a STAR/GEM/BSE rule actually fire": a
    correctly spelled main-board-only list -- ``store.view(max_symbols=4)``,
    ``warmed_view(symbols=['600000.SH', '600036.SH', ...])`` -- is a normal
    input, not a format error, and the composition question can only be answered
    over the FULL universe. That is :func:`require_board_rule`, which callers
    holding the whole symbol list run once; per call, ``derive`` only warns (see
    :func:`_warn_if_no_board_rule`)."""
    if any(str(s)[:1].isdigit() for s in symbols):
        return
    sample = ", ".join(repr(s) for s in list(symbols)[:5])
    raise ValueError(
        f"no board rule -- STAR (688/689), GEM (300/301/302) or BSE (4/8/92) -- can ever fire: "
        f"not one of the {len(symbols)} symbols even starts with a digit; first five: {sample}. "
        "This almost always means the symbol spelling does not put the market prefix first "
        "(e.g. 'SH.600000' instead of '600000.SH'); LimitWidth would otherwise ship as a "
        "constant 0.10 plane and BoardCode as a single category, both silently."
    )


def _warn_if_no_board_rule(name: str, symbols: Sequence[str], star: np.ndarray,
                           gem: np.ndarray, bse: np.ndarray) -> None:
    """must_fix #2's COMPOSITION half, downgraded to a warning for the per-call
    path: a prefix-first symbol list on which no STAR/GEM/BSE rule fires is
    either a genuine main-board-only subset (fine) or the whole universe
    mis-spelled in some way the format check did not catch (not fine), and this
    function cannot tell which, because it is not given the universe.

    Warns once per distinct message, the same way :func:`_warn_once_if_late_start`
    does. ``stacklevel=3`` points at ``derive``'s caller on the direct path and
    at ``resolve`` on the resolve path."""
    if star.any() or gem.any() or bse.any():
        return
    sample = ", ".join(repr(s) for s in list(symbols)[:5])
    warnings.warn(
        f"daily field {name!r}: no STAR (688/689), GEM (300/301/302) or BSE (4/8/92) prefix "
        f"matched any of {len(symbols)} symbols; first five: {sample}. Over a symbol SUBSET "
        "that is normal (a main-board-only selection); over the full universe it means the "
        "spelling is wrong and the field is a constant. Run dailyfields.require_board_rule() "
        "on the full symbol list to tell the two apart.",
        RuntimeWarning, stacklevel=3,
    )


def require_board_rule(symbols: Sequence[str]) -> None:
    """must_fix #2's composition assertion, for a caller holding the FULL
    universe: raise unless at least one STAR/GEM/BSE prefix matched.

    This is the strict form, and it is strict precisely because "no special
    board anywhere in the whole A-share universe" cannot be true. It is NOT run
    per ``derive`` call -- a subset view legitimately contains main-board
    symbols only, and raising there kills every factor touching ``LimitWidth``
    or ``BoardCode`` on a ``max_symbols`` / explicit-``symbols`` view. Call it
    where the full list is in hand: the daily-source attach on
    ``CubeStore`` / ``SegmentedCubeStore`` (over ``store.symbols``), and
    ``scripts/build_daily_fields.py`` next to its coverage floor."""
    _require_prefix_first(symbols)
    star, gem, bse = _board_masks(symbols)
    if star.any() or gem.any() or bse.any():
        return
    sample = ", ".join(repr(s) for s in list(symbols)[:5])
    raise ValueError(
        f"no STAR (688/689), GEM (300/301/302) or BSE (4/8/92) symbol prefix matched any of "
        f"{len(symbols)} symbols; first five: {sample}. On a full universe that cannot be "
        "right: LimitWidth would ship as a constant 0.10 plane and BoardCode as a single "
        "category, both silently and both at 100% finite coverage."
    )


def _lifecycle_mask(S: int, D: int, *, tradable: Optional[np.ndarray],
                    status: Optional[np.ndarray]) -> np.ndarray:
    """(S, D) bool, True on a stock-day that counts toward ``ListAge``.

    ``status`` (TRADING/RESUMED/SUSPENDED all count, only OUTSIDE_LIFECYCLE does
    not) is the precise signal and is preferred; ``tradable`` (TRADING/RESUMED
    only) is the fallback and reads a merely-suspended stock as not-yet-listed on
    that one day, which only matters if a stock's very first in-lifecycle day is
    also a suspension. With neither given, every requested day is treated as in
    lifecycle (so every symbol gets the sentinel, the safe default -- it says
    exactly "this plane cannot tell when these stocks were listed")."""
    if status is not None:
        status = np.asarray(status)
        if status.shape != (S, D):
            raise ValueError(f"ListAge: status shape {status.shape} != {(S, D)}")
        return status != _OUTSIDE_LIFECYCLE
    if tradable is not None:
        tradable = np.asarray(tradable, dtype=bool)
        if tradable.shape != (S, D):
            raise ValueError(f"ListAge: tradable shape {tradable.shape} != {(S, D)}")
        return tradable
    return np.ones((S, D), dtype=bool)


def _list_age(symbols: Sequence[str], dates: Sequence[str], *, tradable: Optional[np.ndarray],
              status: Optional[np.ndarray]) -> np.ndarray:
    """Trading days since the first in-lifecycle day: 0 there, +1 every
    subsequent trading day -- INCLUDING through a suspension, since lifecycle,
    once entered, is sticky -- and ``LIST_AGE_SENTINEL`` added throughout for a
    stock already in lifecycle on the window's first day (the data cannot say
    how old such a stock really is). Exactly Partura's ``cross_age`` formula
    (``ops_ext.py``: ``age = cumsum(seen) - 1``, ``old = seen[:1] * 10000``,
    ``where(seen > 0, age + old, NaN)``), with ``seen`` built from the lifecycle
    mask instead of ``isfinite(x)`` and the day axis transposed to ``axis=1``
    (this module's ``(S, D)``, not cross_age's ``(D, S)``)."""
    S, D = len(symbols), len(dates)
    in_lifecycle = _lifecycle_mask(S, D, tradable=tradable, status=status).astype(np.float64)
    seen = np.maximum.accumulate(in_lifecycle, axis=1)        # sticky: 1 forever once ever in lifecycle
    age = np.cumsum(seen, axis=1) - 1.0
    sentinel = seen[:, :1] * LIST_AGE_SENTINEL                 # already in lifecycle on day 0
    age = age + sentinel
    return np.where(seen > 0, age, np.nan)


def derive(name: str, symbols: Sequence[str], dates: Sequence[str], *,
           tradable: Optional[np.ndarray] = None, status: Optional[np.ndarray] = None) -> np.ndarray:
    """One of ``DERIVED_FIELDS`` as an ``(S, D)`` float64 plane, from the symbol
    and date labels alone (``LimitWidth``, ``BoardCode``) or those plus a
    lifecycle signal (``ListAge``; see ``_lifecycle_mask``). No stored data, no
    build step -- that is what makes these three fields free of a dependency on
    ``daily_root`` being attached at all (see ``resolve`` / ``available``).

    Raises ``ValueError`` for ``LimitWidth``/``BoardCode`` when the symbol
    spelling makes a board rule impossible (must_fix #2's format half, see
    :func:`_require_prefix_first`), and merely WARNS when the spelling is fine
    but no rule fired, which over a symbol subset is normal (see
    :func:`_warn_if_no_board_rule` / :func:`require_board_rule`)."""
    symbols = list(symbols)
    dates = list(dates)
    S, D = len(symbols), len(dates)
    if name == "LimitWidth":
        _require_prefix_first(symbols)
        _warn_if_no_board_rule(name, symbols, *_board_masks(symbols))
        # board_limit returns (D, S); transpose FIRST, then make contiguous -- the
        # other order would make ascontiguousarray's work free (the (D, S) array
        # already is contiguous) and ship a non-contiguous (S, D) view instead.
        return np.ascontiguousarray(board_limit(symbols, dates).T, dtype=np.float64)
    if name == "BoardCode":
        star, gem, bse = _board_masks(symbols)
        _require_prefix_first(symbols)
        _warn_if_no_board_rule(name, symbols, star, gem, bse)
        code = np.full((S, D), BOARD_MAIN, dtype=np.float64)
        code[star, :] = BOARD_STAR
        code[gem, :] = BOARD_GEM
        code[bse, :] = BOARD_BSE
        return code
    if name == "ListAge":
        return _list_age(symbols, dates, tradable=tradable, status=status)
    raise ValueError(f"dailyfields.derive: {name!r} is not a derived field; DERIVED_FIELDS is {DERIVED_FIELDS}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DailyFieldStore:
    """One directory of STORED_FIELDS planes: ``daily_<name>.npy`` (float64,
    shape ``(n_symbols, n_days)``) for each name in ``fields``, plus one
    ``meta.json`` naming the symbol/date labels those rows/columns are in.
    Read-only, mmap'd, cached per field -- the on-disk shape ``store.CubeStore``
    uses, without importing it."""

    def __init__(self, root) -> None:
        self.root = Path(root)
        meta_path = self.root / "meta.json"
        with meta_path.open("r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.fields: Tuple[str, ...] = tuple(self.meta["fields"])
        self.symbols: List[str] = [str(s) for s in self.meta["symbols"]]
        self.dates: List[str] = [str(d) for d in self.meta["dates"]]
        self.meta_sha256 = _sha256_file(meta_path)
        self._cache: Dict[str, np.ndarray] = {}
        self._sym_index: Dict[str, int] = {s: i for i, s in enumerate(self.symbols)}
        self._date_index: Dict[str, int] = {d: i for i, d in enumerate(self.dates)}

    def plane(self, name: str) -> np.ndarray:
        """``(n_symbols, n_days)`` float64, mmap'd and cached."""
        if name not in self.fields:
            raise KeyError(f"daily store has no field {name!r}; it has {self.fields}")
        if name not in self._cache:
            expected = (len(self.symbols), len(self.dates))
            arr = np.load(self.root / f"daily_{name}.npy", mmap_mode="r")
            if arr.shape != expected:
                raise ValueError(f"daily field {name}: shape {arr.shape} != {expected}")
            if arr.dtype != np.float64:
                raise ValueError(f"daily field {name}: dtype {arr.dtype} != float64")
            self._cache[name] = arr
        return self._cache[name]

    def align(self, name: str, symbols: Sequence[str], dates: Sequence[str]) -> np.ndarray:
        """``plane(name)`` re-indexed BY LABEL onto ``(symbols, dates)``: a
        superset, a subset, a permutation, or a label this store does not carry
        at all are all legal -- NaN fills exactly the cells this store has no
        row or column for."""
        src = self.plane(name)
        s_idx = np.array([self._sym_index.get(s, -1) for s in symbols], dtype=np.int64)
        d_idx = np.array([self._date_index.get(d, -1) for d in dates], dtype=np.int64)
        out = np.full((len(symbols), len(dates)), np.nan, dtype=np.float64)
        s_ok, d_ok = np.flatnonzero(s_idx >= 0), np.flatnonzero(d_idx >= 0)
        if s_ok.size and d_ok.size:
            out[np.ix_(s_ok, d_ok)] = np.asarray(src)[np.ix_(s_idx[s_ok], d_idx[d_ok])]
        return out


def _warn_once_if_late_start(name: str, plane: np.ndarray, dates: Sequence[str]) -> None:
    """The module's single "is this plane actually there?" guard, run on every
    :func:`resolve`. Three checks, most specific first:

    1. **Nothing anywhere.** An all-NaN plane is the loudest symptom of a LABEL
       mismatch -- the daily store spelling its symbols ``'SH.600000'`` while
       the view asks ``'600000.SH'``, or its dates ``'2020-01-01'`` against
       ``'20200101'`` -- which ``DailyFieldStore.align`` answers with a NaN fill
       by design, since a label the store does not carry is legal. This check
       exists because the late-start check below provably cannot fire here:
       with no finite cell there is no first finite date to compare.
    2. **Almost nothing.** Below ``LOW_COVERAGE_FLOOR`` of the requested cells
       finite is a PARTIAL label mismatch (a store overlapping the view on a
       handful of symbols), which check 1 misses -- one finite cell is enough to
       give the field a first finite date and a sane-looking warm-up.
    3. **Late start** -- the silent-empty-warm-up failure: if ``name``'s first
       finite date is after ``dates[0]``, every rolling window reaching back
       past it is NaN, which reads as a short warm-up rather than as missing
       data.

    Warns once per distinct message: Python's default warning filter already
    de-duplicates an identical (message, category, module, lineno), so no extra
    bookkeeping is needed here, and a test can still force it to re-fire with
    ``pytest.warns`` / ``simplefilter("always")``."""
    if not dates or plane.size == 0:
        return
    finite = np.isfinite(plane)
    finite_days = np.flatnonzero(finite.any(axis=0))
    if finite_days.size == 0:
        warnings.warn(
            f"daily field {name!r} has no finite value ANYWHERE in the requested range "
            f"[{dates[0]!r}, {dates[-1]!r}] over {plane.shape[0]} symbols: the store's "
            "symbol/date spelling almost certainly does not match the view's (align() "
            "NaN-fills every label it does not carry), so every factor built on this "
            "field will evaluate on an empty panel",
            RuntimeWarning, stacklevel=3,
        )
        return
    coverage = float(finite.mean())
    if coverage < LOW_COVERAGE_FLOOR:
        warnings.warn(
            f"daily field {name!r} is finite in only {coverage:.3%} of the "
            f"{plane.shape[0]}x{plane.shape[1]} requested cells (floor "
            f"{LOW_COVERAGE_FLOOR:.1%}): this is a near-empty panel, usually a PARTIAL "
            "symbol/date label mismatch between the daily store and the view",
            RuntimeWarning, stacklevel=3,
        )
    first = dates[int(finite_days[0])]
    if first <= dates[0]:
        return
    warnings.warn(
        f"daily field {name!r} has no finite value before {first!r}, after the requested "
        f"range's first date {dates[0]!r}: rolling windows will see this as an empty gap, "
        "not a short warm-up",
        RuntimeWarning, stacklevel=3,
    )


def resolve(name: str, symbols: Sequence[str], dates: Sequence[str], *,
            store: Optional[DailyFieldStore] = None, tradable: Optional[np.ndarray] = None,
            status: Optional[np.ndarray] = None) -> np.ndarray:
    """``(S, D)`` float32 for any of the 9 ``catalog.DAILY_FIELDS``.

    Stored and log fields need ``store`` (a ``DailyFieldStore`` carrying the
    relevant stored field); derived fields need neither -- only ``ListAge``
    reads ``tradable``/``status``, and only to tell an unlisted stock-day from a
    merely-suspended one (see ``_lifecycle_mask``).

    The cast to float32 happens LAST: a log field's ``log()`` runs on the
    float64 stored plane, per the design's structural contract (see the module
    docstring's CAUTION -- measured, this specific step turns out not to change
    ``log()``'s own float32 output here, since a value around log(1e9..1e13) is
    far from float64/float32's shared trouble spots; it is cheap and never
    worse, so it stays, but it is not the precision fix for ``ts_delta`` that
    motivated it)."""
    symbols = list(symbols)
    dates = list(dates)
    if name in LOG_OF:
        base = LOG_OF[name]
        if store is None or base not in store.fields:
            raise ValueError(f"dailyfields.resolve: {name!r} needs a daily store carrying {base!r} (daily_root)")
        with np.errstate(divide="ignore", invalid="ignore"):
            f64 = np.log(store.align(base, symbols, dates))
    elif name in STORED_FIELDS:
        if store is None or name not in store.fields:
            raise ValueError(f"dailyfields.resolve: {name!r} needs a daily store carrying it (daily_root)")
        f64 = store.align(name, symbols, dates)
    elif name in DERIVED_FIELDS:
        f64 = derive(name, symbols, dates, tradable=tradable, status=status)
    else:
        raise ValueError(f"dailyfields.resolve: unknown daily field {name!r}; available fields are "
                         f"{available(store)}")
    _warn_once_if_late_start(name, f64, dates)
    return f64.astype(np.float32)


def available(store: Optional[DailyFieldStore] = None) -> Tuple[str, ...]:
    """Daily field names actually resolvable right now, in ``catalog.
    DAILY_FIELDS``'s own order (stored, derived, log): derived fields always
    (symbols/dates alone are enough); a stored field and its ``LOG_OF``
    derivative only for the ones ``store`` actually carries. ``()`` when
    ``store`` is None except for the three derived fields."""
    stored = tuple(f for f in STORED_FIELDS if store is not None and f in store.fields)
    logs = tuple(n for n, base in LOG_OF.items() if store is not None and base in store.fields)
    return stored + DERIVED_FIELDS + logs


def fingerprint(store: Optional[DailyFieldStore]) -> Optional[str]:
    """``None`` with no daily source attached -- which is what keeps every
    existing ``contract_hash`` and ``ExemplarCache`` key byte-identical -- else
    ``f"{DERIVED_VERSION}:{store.meta_sha256}"``, so a later change to
    ``derive()``'s formulas (bump ``DERIVED_VERSION``) invalidates cached rank
    fingerprints without the caller doing anything."""
    if store is None:
        return None
    return f"{DERIVED_VERSION}:{store.meta_sha256}"


__all__ = [
    "STORED_FIELDS", "DERIVED_FIELDS", "LOG_OF", "LIST_AGE_SENTINEL", "DERIVED_VERSION",
    "LOW_COVERAGE_FLOOR", "BOARD_MAIN", "BOARD_STAR", "BOARD_GEM", "BOARD_BSE",
    "derive", "require_board_rule", "DailyFieldStore", "resolve", "available", "fingerprint",
]
