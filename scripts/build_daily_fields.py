"""Server build script for AlphaCube's stored daily fields (``dailyfields.
STORED_FIELDS``: FloatCap, TotalCap, FloatShares), meant to run ONCE on the
server next to a live cube store. See docs/alphacube_steering_design.md,
section "Building the data", and work package "### WP9".

    PYTHONPATH=src python scripts/build_daily_fields.py \\
        --store /root/autodl-tmp/qmt_daily_data/fast_factor_compute \\
        [--cap-dir /root/autodl-tmp/qmt_daily_data/daily_capital] \\
        [--out /root/autodl-tmp/AlphaCube/daily_fields] \\
        [--only FloatCap --only TotalCap] [--min-coverage 0.5] [--dry-run]

Reads one ``<cap_dir>/<date>.parquet`` per date -- never more than one day's
frame in memory -- and aligns everything onto the STORE's own symbols/dates
(opened read-only, purely for that canonical label list), so the planes this
writes are born aligned: no alignment code is needed downstream (see
dailyfields.py / CubeView.daily()). Writes a dailyfields.DailyFieldStore
directory -- one ``daily_<name>.npy`` (float64) per field plus one
``meta.json`` -- into a temporary directory beside ``--out`` and
``os.replace``s it into place at the very end, so a build that dies partway
through leaves nothing at ``--out`` for a DailyFieldStore to open.

Writes NO derived field (LimitWidth/BoardCode/ListAge -- dailyfields.derive
computes those straight from the store's own symbols/dates, no build step)
and NO log field (LogFloatCap/LogTotalCap/LogFloatShares -- dailyfields.
resolve derives those from the stored float64 plane); see the note this
script prints at the end, and the design's "Building the data" section.

Exits non-zero -- and never writes anything -- when any built field's finite
coverage falls below --min-coverage: the one way this silently produces a
useless factor is a symbol-format mismatch between --store and --cap-dir that
NaNs every cell, which is also why the store's and the parquet's first 10
symbols are always printed side by side (must_fix #2's third, visible
defence -- scripts/tearsheet.py's load_caps, the function this script's
column reads mirror, has zero test coverage of its own symbol mapping).

DEVIATION, recorded here rather than asserted as fact (see this package's
notes): the design pins FloatCap/TotalCap's exact source columns
(circulating_market_value / total_market_value -- scripts/tearsheet.py:52)
but, by its own wording, leaves FloatShares's share-count column unnamed ("a
share column if one exists") -- no production parquet schema was reachable
this session to confirm a real name. SHARE_COLUMN_CANDIDATES below is a
documented, best-effort guess among common vendor spellings, not a measured
fact; whichever one (if any) actually matches the probed schema is recorded
in meta.json's columns_used either way, so the choice stays auditable, and
FloatShares is simply never written when none match (never synthesised from
FloatCap / Close -- step 6).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SYMBOL_COLUMN = "symbol"
# Exact names scripts/tearsheet.py's load_caps already uses (tearsheet.py:52),
# pinned by the design (WP9 step 4) -- these two are mandatory, not optional.
CAP_COLUMN = "circulating_market_value"        # -> FloatCap
TOTAL_COLUMN = "total_market_value"            # -> TotalCap
# "a share column if one exists" (design step 6) -- genuinely unnamed by the
# design, unlike the two above; see the module docstring DEVIATION. The first
# name below present in the probed schema wins; none present means
# FloatShares is simply not built (never synthesised -- step 6).
SHARE_COLUMN_CANDIDATES: Tuple[str, ...] = (
    "circulating_shares", "circulating_share", "circ_shares", "circulating_a_shares",
    "float_shares", "float_a_shares", "free_shares", "share_count",
)
REQUIRED_COLUMNS: Tuple[str, ...] = (SYMBOL_COLUMN, CAP_COLUMN, TOTAL_COLUMN)
FIELD_COLUMN: Dict[str, str] = {"FloatCap": CAP_COLUMN, "TotalCap": TOTAL_COLUMN}


def refuse_if_out_inside_sources(out_dir: Path, cap_dir: Path, store_dir: Path) -> None:
    """Design step 2: the source data directories are never written to.
    Raises ``SystemExit`` -- never silently returns -- when ``out_dir``
    resolves to ``cap_dir``/``store_dir`` themselves or anywhere underneath
    either. Pure path arithmetic: deliberately runs before anything is opened
    or read, so a bad ``--out`` fails before any I/O happens."""
    out_r = Path(out_dir).resolve()
    for label, src in (("--cap-dir", Path(cap_dir)), ("the store directory (--store)", Path(store_dir))):
        src_r = src.resolve()
        if out_r == src_r or out_r.is_relative_to(src_r):
            raise SystemExit(
                f"refusing to run: --out ({out_r}) resolves inside {label} ({src_r}); "
                "the source data directories must never be written to"
            )


def probe_schema(cap_dir: Path, dates: Sequence[str]) -> Tuple[Path, List[str], List[str]]:
    """Find one parquet file to probe -- the store's first date that actually
    has one, else the earliest file name in ``cap_dir`` -- and return
    ``(path, column names, that file's own symbol column values)``.
    ``SystemExit`` when ``cap_dir`` carries no parquet file reachable either
    way (step 3)."""
    import pandas as pd

    cap_dir = Path(cap_dir)
    sample: Optional[Path] = None
    for d in dates:
        candidate = cap_dir / f"{d}.parquet"
        if candidate.is_file():
            sample = candidate
            break
    if sample is None:
        found = sorted(cap_dir.glob("*.parquet"))
        sample = found[0] if found else None
    if sample is None:
        tried = f"{dates[0]!r}..{dates[-1]!r}" if dates else "no dates at all"
        raise SystemExit(
            f"no .parquet files found in --cap-dir {cap_dir} -- tried every store date "
            f"({tried}) plus a plain directory listing"
        )
    df = pd.read_parquet(sample)
    symbols = list(df[SYMBOL_COLUMN]) if SYMBOL_COLUMN in df.columns else []
    return sample, list(df.columns), symbols


def choose_columns(columns: Sequence[str], only: Optional[Sequence[str]]) -> Dict[str, str]:
    """``{field_name: source_column_name}`` for the stored fields to actually
    build. FloatCap/TotalCap are mandatory (step 4); FloatShares is included
    only when one of SHARE_COLUMN_CANDIDATES is present in ``columns`` (step
    6 -- "only if a real share column exists", never synthesised). ``only``
    (the repeatable ``--only`` flag) restricts the result to a subset: an
    unknown field name in it is a hard error, a known-but-unbuildable one
    (e.g. FloatShares with no share column in this schema) is reported on
    stderr and dropped rather than raised."""
    from alphacube.dailyfields import STORED_FIELDS

    cols = set(columns)
    missing = [c for c in REQUIRED_COLUMNS if c not in cols]
    if missing:
        raise SystemExit(
            f"--cap-dir parquet schema is missing required column(s) {missing}; schema has {sorted(cols)}"
        )
    mapping = dict(FIELD_COLUMN)
    share_col = next((c for c in SHARE_COLUMN_CANDIDATES if c in cols), None)
    if share_col is not None:
        mapping["FloatShares"] = share_col
    if only:
        unknown = sorted(set(only) - set(STORED_FIELDS))
        if unknown:
            raise SystemExit(f"--only names unknown field(s) {unknown}; stored fields are {STORED_FIELDS}")
        wanted = set(only)
        dropped = sorted(f for f in wanted if f not in mapping)
        if dropped:
            print(f"--only asked for {dropped}, but this schema has no available source column "
                 "for it; skipping", file=sys.stderr, flush=True)
        mapping = {f: c for f, c in mapping.items() if f in wanted}
    return mapping


def build_planes(symbols: Sequence[str], dates: Sequence[str], cap_dir: Path,
                 columns_used: Dict[str, str]) -> Dict[str, np.ndarray]:
    """``{field_name: (S, D) float64}``, NaN where ``cap_dir``'s
    ``<date>.parquet`` is missing or ``symbol`` is absent that day (step 5).
    One parquet read per date -- never more than one day's frame in memory --
    following scripts/tearsheet.py:41-58's ``load_caps`` symbol -> store-row
    mapping exactly (one vectorised ``.map``, then boolean-masked writes)."""
    import pandas as pd

    S, D = len(symbols), len(dates)
    row_of = {s: i for i, s in enumerate(symbols)}
    planes = {name: np.full((S, D), np.nan, dtype=np.float64) for name in columns_used}
    read_cols = [SYMBOL_COLUMN] + sorted(set(columns_used.values()))
    cap_dir = Path(cap_dir)
    for d_idx, date in enumerate(dates):
        path = cap_dir / f"{date}.parquet"
        if not path.is_file():
            continue
        df = pd.read_parquet(path, columns=read_cols)
        idx = df[SYMBOL_COLUMN].map(row_of)
        ok = idx.notna().to_numpy()
        if not ok.any():
            continue
        rows = idx[ok].astype(int).to_numpy()
        for name, col in columns_used.items():
            planes[name][rows, d_idx] = df[col].to_numpy(dtype=np.float64)[ok]
    return planes


def summarize(planes: Dict[str, np.ndarray], dates: Sequence[str]
             ) -> Tuple[Dict[str, float], Dict[str, Optional[Tuple[str, str]]]]:
    """Per field: the finite fraction over the whole plane, and the
    ``(first, last)`` date with any finite value at all (``None`` if none) --
    the two pieces step 7/8 need (meta.json's coverage/date range, and the
    --min-coverage gate)."""
    coverage: Dict[str, float] = {}
    date_range: Dict[str, Optional[Tuple[str, str]]] = {}
    for name, plane in planes.items():
        finite = np.isfinite(plane)
        coverage[name] = float(finite.mean()) if plane.size else 0.0
        has_data = np.flatnonzero(finite.any(axis=0))
        date_range[name] = (dates[int(has_data[0])], dates[int(has_data[-1])]) if has_data.size else None
    return coverage, date_range


def write_daily_field_store(out_dir: Path, planes: Dict[str, np.ndarray], symbols: Sequence[str],
                            dates: Sequence[str], meta_extra: dict) -> Path:
    """Write one dailyfields.DailyFieldStore directory -- ``daily_<name>.npy``
    (float64) per field in ``planes`` plus one ``meta.json`` naming
    ``fields``/``n_symbols``/``n_days``/``symbols``/``dates`` (exactly what
    ``DailyFieldStore.__init__`` reads) and whatever audit keys
    ``meta_extra`` adds -- into a fresh temporary directory beside
    ``out_dir``, renamed into place only once every file is down (step 7).

    A build that raises partway through the writes leaves the temp directory
    cleaned up and ``out_dir`` completely untouched -- never half-written.
    ``out_dir`` already existing as a non-empty directory (a previous build)
    is also refused rather than silently deleted: ``os.replace`` cannot swap
    a directory onto a non-empty one (see Python's ``os.rename``/POSIX
    ``rename(2)``), and this script does not make that call for the caller --
    the existing directory is left exactly as it was, and the operator is
    told to remove it first."""
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", suffix=".tmp", dir=str(out_dir.parent)))
    try:
        names = sorted(planes)
        for name in names:
            np.save(tmp / f"daily_{name}.npy", np.ascontiguousarray(planes[name], dtype=np.float64))
        meta = {
            "fields": names,
            "n_symbols": len(symbols),
            "n_days": len(dates),
            "symbols": [str(s) for s in symbols],
            "dates": [str(d) for d in dates],
            **meta_extra,
        }
        (tmp / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    try:
        os.replace(tmp, out_dir)
    except OSError as e:
        shutil.rmtree(tmp, ignore_errors=True)
        raise SystemExit(
            f"could not move the newly built directory into place at {out_dir} ({e}); if {out_dir} "
            "already exists from a previous build, remove it first and re-run"
        ) from e
    return out_dir


def open_store_for_labels(store_dir: Path):
    """Open the cube store read-only, purely for its canonical symbols/dates
    (step 1) -- with NO daily source attached, whatever the operator's shell
    happens to say.

    ``store.open_cube_store`` defaults its ``daily_root`` to
    ``$ALPHACUBE_DAILY_FIELDS`` (store.py:444-445) and offers no way to spell
    "explicitly none" -- and that env var is the design's documented way to
    point AlphaCube at the daily store, so on the server it is most naturally
    set to exactly this script's ``--out``. Letting the fallback through would
    break the one script that CREATES that directory:

    * on the FIRST build ``<out>`` does not exist yet, so CubeStore.__init__
      (store.py:131) builds a ``DailyFieldStore`` over it and the script dies
      with an uncaught ``FileNotFoundError: <out>/meta.json`` -- before the
      schema probe, before the side-by-side symbol print, before any of the
      diagnostics that exist precisely for a first build;
    * that same attach runs ``require_board_rule`` from inside
      ``CubeStore.__init__``, i.e. ahead of this script's own call, so a store
      with the wrong symbol spelling escapes as a raw ``ValueError`` instead of
      the ``SystemExit`` main() converts it into;
    * and when the env var points at an EXISTING, stale store, the open
      silently attaches and fingerprints it into ``contract_hash`` -- not
      "read-only, purely for that canonical label list" either.

    So the fallback is neutralised for the duration of the open and then put
    back exactly as it was: the dispatch itself still comes from
    ``open_cube_store``, so this cannot drift from the real rule."""
    from alphacube.store import open_cube_store

    saved = os.environ.pop("ALPHACUBE_DAILY_FIELDS", None)
    try:
        return open_cube_store(str(store_dir))
    finally:
        if saved is not None:
            os.environ["ALPHACUBE_DAILY_FIELDS"] = saved


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", required=True,
                    help="cube store root (a compute root or a bare minute directory); opened "
                         "read-only, only for its canonical symbols/dates")
    ap.add_argument("--cap-dir", default="/root/autodl-tmp/qmt_daily_data/daily_capital",
                    help="directory of <YYYYMMDD>.parquet capital snapshots")
    ap.add_argument("--out", default="/root/autodl-tmp/AlphaCube/daily_fields",
                    help="DailyFieldStore directory to build")
    ap.add_argument("--only", action="append", default=None, metavar="FIELD",
                    help="restrict to this stored field (FloatCap/TotalCap/FloatShares); repeatable")
    ap.add_argument("--min-coverage", type=float, default=0.5,
                    help="exit non-zero, writing nothing, if any built field's finite fraction is below this")
    ap.add_argument("--dry-run", action="store_true", help="probe, build in memory and report; write nothing")
    args = ap.parse_args(argv)

    out_dir, cap_dir, store_dir = Path(args.out), Path(args.cap_dir), Path(args.store)
    refuse_if_out_inside_sources(out_dir, cap_dir, store_dir)

    from alphacube.dailyfields import DERIVED_FIELDS, LOG_OF, require_board_rule

    # NOT open_cube_store(...) directly: no daily source may be attached while
    # we are building one -- see open_store_for_labels.
    store = open_store_for_labels(store_dir)
    try:
        require_board_rule(store.symbols)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    symbols, dates = list(store.symbols), list(store.dates)

    sample_path, columns, parquet_symbols = probe_schema(cap_dir, dates)
    print(f"schema probe: {sample_path} ; columns: {columns}", flush=True)
    print(f"store symbols (first 10):   {symbols[:10]}", flush=True)
    print(f"parquet symbols (first 10): {parquet_symbols[:10]}", flush=True)

    columns_used = choose_columns(columns, args.only)
    if not columns_used:
        raise SystemExit("nothing to build: --only left no field with an available source column")
    print(f"building {sorted(columns_used)} from columns {columns_used}", flush=True)

    planes = build_planes(symbols, dates, cap_dir, columns_used)
    coverage, date_range = summarize(planes, dates)
    for name in sorted(coverage):
        span = date_range[name]
        print(f"  {name:12s} finite {coverage[name]:7.3%}  first/last date with data: "
             f"{span[0] if span else None} .. {span[1] if span else None}", flush=True)

    low = {n: round(c, 6) for n, c in coverage.items() if c < args.min_coverage}
    if low:
        raise SystemExit(
            f"finite coverage below --min-coverage={args.min_coverage:g} for {low}; this is almost "
            "always a symbol-format mismatch between --store and --cap-dir -- "
            f"store symbols (first 10): {symbols[:10]}; parquet symbols (first 10): {parquet_symbols[:10]}"
        )

    if args.dry_run:
        print(f"--dry-run: would write {sorted(columns_used)} to {out_dir}; nothing written", flush=True)
    else:
        meta_extra = {
            "source_paths": {"cap_dir": str(cap_dir.resolve()), "store": str(store_dir.resolve())},
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "columns_used": columns_used,
            "coverage": coverage,
            "date_range": {n: (list(v) if v is not None else None) for n, v in date_range.items()},
        }
        write_daily_field_store(out_dir, planes, symbols, dates, meta_extra)
        print(f"wrote {sorted(columns_used)} to {out_dir}", flush=True)

    print("derived fields (no build step -- dailyfields.derive, from the store's own symbols/dates): "
         f"{list(DERIVED_FIELDS)}", flush=True)
    print(f"log fields (no build step -- dailyfields.resolve, from the stored plane): {list(LOG_OF)}",
         flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
