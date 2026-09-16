"""Production factor calculation: expressions -> per-factor value files + a
status catalog, resumable, with the previous project's deferred-resource
policy and its Parquet export layout.

Differences from ``research.evaluate_factors``: no labels or metrics (factor
values only), one atomically written ``.npy`` per factor, a catalog with a
status per factor (SUCCESS / PARSE_FAILED / NOT_DAILY / EVAL_FAILED /
DEFERRED_RESOURCE_RISK), batches of ``factor_batch_size`` expressions, and
``resume=True`` skips factors the catalog already holds for the same
expression text (so a killed run continues where it stopped).
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Set, Union

import numpy as np
import pandas as pd

from .engine.engine import CubeEngine
from .expression import Expression, max_window
from .research import inline_references, parse_many, safe_names, warmed_view
from .store import CubeStore, SegmentedCubeStore, open_cube_store

# One definition, shared with the search vocabulary (catalog.Registry.searchable
# drops these unless asked), so the production path and the miner cannot drift.
from .catalog import RESOURCE_RISK_KERNELS as DEFERRED_RESOURCE_OPERATORS
STATUS_ORDER = ("SUCCESS", "EVAL_FAILED", "NOT_DAILY", "PARSE_FAILED", "DEFERRED_RESOURCE_RISK")


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npy")
    np.save(tmp, values.astype(np.float32, copy=False))
    os.replace(tmp, path)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def deferred_resource_factors(expressions: Dict[str, str]) -> Dict[str, str]:
    """Direct and transitive (by-name reference) users of deferred operators."""
    inlined = inline_references(expressions)
    out: Dict[str, str] = {}
    for name, src in inlined.items():
        try:
            names = {n.id for n in ast.walk(ast.parse(src, mode="eval")) if isinstance(n, ast.Name)}
        except SyntaxError:
            continue
        risky = sorted(names & DEFERRED_RESOURCE_OPERATORS)
        if risky:
            out[name] = "uses deferred resource operator: " + ", ".join(risky)
    return out


def _expr_digest(src: str) -> str:
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]


def run_factor_calculation(
    expressions: Dict[str, str],
    store: Union[CubeStore, SegmentedCubeStore, str, Path],
    *,
    save_root: Union[str, Path],
    run_id: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    symbols: Optional[Sequence[str]] = None,
    max_symbols: Optional[int] = None,
    factor_batch_size: int = 64,
    warmup_buffer: int = 10,
    device: str = "cpu",
    chunk_symbols: Optional[int] = None,
    chunk_days: Optional[int] = None,
    defer_resource_risk: bool = True,
    resume: bool = True,
    max_batches: Optional[int] = None,
    quiet: bool = False,
) -> pd.DataFrame:
    """Compute factor values into ``<save_root>/<run_id>/values/<factor>.npy``
    and return the status catalog (also written as ``factor_results.csv`` /
    ``factor_results.json``). ``max_batches`` stops early (for tests of
    resumption); the run is then left IN_PROGRESS and a later call with
    ``resume=True`` finishes it."""
    if not isinstance(store, (CubeStore, SegmentedCubeStore)):
        store = open_cube_store(store)
    run_dir = Path(save_root) / run_id
    values_dir = run_dir / "values"
    values_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "factor_results.json"
    t0 = time.time()

    rows: Dict[str, dict] = {}
    if resume and results_path.is_file():
        prior = json.loads(results_path.read_text(encoding="utf-8"))
        for name, row in prior.get("rows", {}).items():
            if name in expressions and row.get("digest") == _expr_digest(expressions[name]) \
                    and (row.get("status") != "SUCCESS" or (values_dir / row.get("file", "")).is_file()):
                rows[name] = row

    deferred = deferred_resource_factors(expressions) if defer_resource_risk else {}
    for name, reason in deferred.items():
        rows.setdefault(name, {"status": "DEFERRED_RESOURCE_RISK", "reason": reason, "digest": _expr_digest(expressions[name]),
                               "coverage": None, "finite_values": 0, "file": ""})
    todo = {n: s for n, s in expressions.items() if n not in rows}
    parsed, perr = parse_many(todo, n_minutes=store.n_minutes)
    for name, msg in perr.items():
        rows[name] = {"status": "PARSE_FAILED", "reason": msg, "digest": _expr_digest(expressions[name]),
                      "coverage": None, "finite_values": 0, "file": ""}
    for name, e in list(parsed.items()):
        if not e.shape.degenerate:
            rows[name] = {"status": "NOT_DAILY", "reason": "root is not one value per stock-day", "digest": _expr_digest(expressions[name]),
                          "coverage": None, "finite_values": 0, "file": ""}
            del parsed[name]

    warm = max((max_window(e, "D") for e in parsed.values()), default=0) + warmup_buffer
    view, trim = warmed_view(store, start, end, warm, symbols, max_symbols)
    tradable = view.tradable().T[trim:]                                   # (D, S)
    tradable_cells = int(tradable.sum())
    engine = CubeEngine(view, device=device, chunk_symbols=chunk_symbols, chunk_days=chunk_days)
    fnames = safe_names(list(expressions))
    names = list(parsed)
    n_batches = 0

    def flush(status: str) -> None:
        _atomic_json(results_path, {"run_status": status, "run_id": run_id, "rows": rows,
                                    "dates": view.dates[trim:], "symbols": view.symbols})

    for lo in range(0, len(names), factor_batch_size):
        if max_batches is not None and n_batches >= max_batches:
            break
        batch = names[lo: lo + factor_batch_size]
        res = engine.evaluate([parsed[n] for n in batch], tolerant=True)
        for i, name in enumerate(batch):
            digest = _expr_digest(expressions[name])
            if res.errors.get(i):
                rows[name] = {"status": "EVAL_FAILED", "reason": res.errors[i], "digest": digest,
                              "coverage": None, "finite_values": 0, "file": ""}
                continue
            panel = res.panels[i][trim:].astype(np.float32)
            panel[~tradable] = np.nan
            fn = fnames[name] + ".npy"
            _atomic_npy(values_dir / fn, panel)
            finite = int(np.isfinite(panel[tradable]).sum()) if tradable_cells else 0
            rows[name] = {"status": "SUCCESS", "reason": "", "digest": digest, "file": fn,
                          "coverage": finite / tradable_cells if tradable_cells else None, "finite_values": finite}
        n_batches += 1
        flush("IN_PROGRESS")
        if not quiet:
            done = sum(1 for n in names if n in rows)
            print(f"  batch {n_batches}: {done}/{len(names)} factors, {time.time() - t0:.0f}s", flush=True)

    complete = all(n in rows for n in expressions)
    flush("COMPLETE" if complete else "IN_PROGRESS")
    catalog = pd.DataFrame.from_dict(rows, orient="index").reindex(list(expressions))
    catalog.index.name = "factor"
    tmp = run_dir / f".factor_results.{os.getpid()}.tmp.csv"
    catalog.to_csv(tmp, encoding="utf-8-sig")
    os.replace(tmp, run_dir / "factor_results.csv")
    if complete:
        _atomic_json(run_dir / "meta.json", {
            "run_status": "COMPLETE", "factor_only": True, "label_used": False, "run_id": run_id,
            "start": start, "end": end, "device": device, "warmup_days": int(trim),
            "n_days": view.n_days - trim, "n_instruments": view.n_symbols,
            "store_contract_hash": getattr(store, "contract_hash", None),
            "dates": view.dates[trim:], "instruments": view.symbols,
            "expressions": dict(expressions),
            "value_files": {n: r["file"] for n, r in rows.items() if r.get("status") == "SUCCESS"},
            "status_counts": {k: int(v) for k, v in catalog["status"].value_counts().items()},
            "deferred_resource_factors": deferred,
            "value_shape": "(n_days, n_instruments) float32",
            "seconds": round(time.time() - t0, 1),
        })
    if not quiet:
        counts = catalog["status"].value_counts().to_dict()
        print(f"{'COMPLETE' if complete else 'IN_PROGRESS'} {run_dir}: " +
              ", ".join(f"{k} {counts[k]}" for k in STATUS_ORDER if k in counts))
    return catalog


def export_run_to_parquet(run_dir: Union[str, Path], output_root: Union[str, Path],
                          store: Union[CubeStore, SegmentedCubeStore, str, Path, None] = None,
                          progress_every: int = 50) -> dict:
    """``<output_root>/<YYYYMMDD>/<factor>.parquet`` with columns
    ``date, symbol, <factor>``: stock-days outside a symbol's lifecycle get no
    row, suspended stock-days a null. Resumable through a manifest."""
    run_dir, output_root = Path(run_dir), Path(output_root)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    if meta.get("factor_only") is not True or meta.get("label_used") is not False:
        raise ValueError("run meta is not a factor-only, label-free result")
    dates = [str(d) for d in meta["dates"]]
    symbols = [str(s) for s in meta["instruments"]]
    value_files: Dict[str, str] = dict(meta.get("value_files", {}))
    if not value_files:
        raise ValueError("run has no exportable factor values")
    if store is None:
        lifecycle = np.ones((len(dates), len(symbols)), dtype=bool)
    else:
        if not isinstance(store, (CubeStore, SegmentedCubeStore)):
            store = open_cube_store(store)
        v = store.view(symbols=symbols, start=dates[0], end=dates[-1])
        if v.dates != dates:
            raise ValueError("run dates are not the store's trading days in that range")
        st = store.read_status(v._rows(0, v.n_symbols), v.d0, v.d1).T          # (D, S)
        lifecycle = st != DAY_STATUS_OUTSIDE
    manifest_path = output_root / "_export_manifests" / f"{run_dir.name}.json"
    src_sha = hashlib.sha256((run_dir / "meta.json").read_bytes()).hexdigest()
    completed: list = []
    files_written = 0
    if manifest_path.is_file():
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        if m.get("source_meta_sha256") != src_sha:
            raise ValueError("existing export manifest belongs to a different run meta")
        completed, files_written = list(m.get("completed_factors", [])), int(m.get("files_written", 0))
    done: Set[str] = set(completed)
    sym_arr = np.asarray(symbols, dtype=object)
    for factor, fn in value_files.items():
        if factor in done:
            continue
        values = np.load(run_dir / "values" / fn, mmap_mode="r")
        if values.shape != (len(dates), len(symbols)):
            raise ValueError(f"{factor}: shape {values.shape} != {(len(dates), len(symbols))}")
        for di, day in enumerate(dates):
            keep = lifecycle[di]
            frame = pd.DataFrame({"date": f"{day[:4]}-{day[4:6]}-{day[6:]}", "symbol": sym_arr[keep],
                                  factor: np.asarray(values[di, keep], dtype=np.float32)})
            path = output_root / day / f"{factor}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            frame.to_parquet(tmp, engine="pyarrow", index=False, compression="snappy")
            os.replace(tmp, path)
            files_written += 1
        completed.append(factor)
        done.add(factor)
        _atomic_json(manifest_path, {"run_status": "COMPLETE" if len(completed) == len(value_files) else "IN_PROGRESS",
                                     "run_id": run_dir.name, "source_meta_sha256": src_sha,
                                     "completed_factors": completed, "files_written": files_written,
                                     "schema": "date,symbol,{factor_name}",
                                     "lifecycle_rule": "outside=no row; suspended=row with null"})
        if progress_every and len(completed) % progress_every == 0:
            print(f"parquet export {len(completed)}/{len(value_files)}: {factor}", flush=True)
    _atomic_json(output_root / "_meta.json", {"data_type": "daily_factors", "storage": "YYYYMMDD/{factor}.parquet",
                                              "schema": "date,symbol,{factor_name}", "last_export_run": run_dir.name,
                                              "last_run_factor_count": len(value_files), "label_used": False})
    return {"factors_total": len(value_files), "factors_completed": len(completed),
            "files_written": files_written, "manifest": str(manifest_path)}


DAY_STATUS_OUTSIDE = 0

__all__ = ["run_factor_calculation", "export_run_to_parquet", "deferred_resource_factors",
           "DEFERRED_RESOURCE_OPERATORS"]
