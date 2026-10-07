"""Research entry point: {name: expression string} -> metrics table (+ values on disk).

One engine, one parser, one namespace: DT, ND and NT operators may appear
anywhere in one expression as long as the tree type-checks (root = one value
per stock-day). Warm-up rows for day-axis windows are read before ``start``
and trimmed after evaluation.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .engine.engine import CubeEngine
from .expression import Expression, max_window
from .label import Label
from .metrics import ic_summary, quantile_longshort
from .parser import ParseError, parse
from .store import CubeStore, CubeView

_UNSAFE = re.compile(r'[/\\:*?"<>|\x00-\x1f\s]+')


def safe_names(names: Sequence[str]) -> Dict[str, str]:
    out, seen = {}, set()
    for n in names:
        base = _UNSAFE.sub("_", n).strip("_.") or "factor"
        cand, k = base, 1
        while cand in seen:
            k += 1
            cand = f"{base}__{k}"
        seen.add(cand)
        out[n] = cand
    return out


def inline_references(expressions: Dict[str, str]) -> Dict[str, str]:
    """Replace bare names that refer to EARLIER factors with that factor's
    (already inlined) source text, so a factor may build on another by name."""
    import ast

    out: Dict[str, str] = {}
    for name, src in expressions.items():
        try:
            tree = ast.parse(src, mode="eval")
        except SyntaxError:
            out[name] = src
            continue
        refs = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in out}
        if refs:
            class _Inline(ast.NodeTransformer):
                def visit_Name(self, node):
                    if node.id in refs:
                        return ast.copy_location(ast.parse(out[node.id], mode="eval").body, node)
                    return node
            tree = ast.fix_missing_locations(_Inline().visit(tree))
            src = ast.unparse(tree)
        out[name] = src
    return out


def parse_many(expressions: Dict[str, str], n_minutes: int = 241, inline: bool = True,
               strict: bool = False) -> Tuple[Dict[str, Expression], Dict[str, str]]:
    """Parse a batch with research settings (lenient rules, daily coercion,
    full catalog, factor references by name). Failures are collected, not raised."""
    sources = inline_references(expressions) if inline else dict(expressions)
    parsed, errors = {}, {}
    for name, src in sources.items():
        try:
            parsed[name] = parse(src, strict=strict, coerce_daily=True, n_minutes=n_minutes, allow_general=True)
        except (ParseError, ValueError, TypeError) as exc:
            errors[name] = str(exc)
    return parsed, errors


def warmed_view(store: CubeStore, start: Optional[str], end: Optional[str], warmup: int,
                symbols: Optional[Sequence[str]] = None, max_symbols: Optional[int] = None) -> Tuple[CubeView, int]:
    base = store.view(symbols=symbols, max_symbols=max_symbols, start=start, end=end)
    d0 = max(0, base.d0 - warmup)
    return CubeView(store, base.sym_idx, d0, base.d1), base.d0 - d0


def evaluate_factors(
    store: Union[CubeStore, str, Path],
    expressions: Dict[str, str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    symbols: Optional[Sequence[str]] = None,
    max_symbols: Optional[int] = None,
    daily_root: Optional[Union[str, Path]] = None,
    label: Label = Label("open_vwap15", 1),
    horizons: Sequence[int] = (1, 5, 10),
    quantiles: int = 5,
    warmup_buffer: int = 10,
    device: str = "cpu",
    chunk_symbols: Optional[int] = None,
    chunk_days: Optional[int] = None,
    out_dir: Optional[Union[str, Path]] = None,
    save_values: bool = True,
    quiet: bool = False,
) -> pd.DataFrame:
    if not isinstance(store, CubeStore):
        store = CubeStore(store, daily_root=daily_root)
    parsed, errors = parse_many(expressions, n_minutes=store.n_minutes)
    warm = max((max_window(e, "D") for e in parsed.values()), default=0) + warmup_buffer
    view, trim = warmed_view(store, start, end, warm, symbols, max_symbols)
    names = list(parsed)
    exprs = [parsed[n] for n in names]
    engine = CubeEngine(view, device=device, chunk_symbols=chunk_symbols, chunk_days=chunk_days)
    t0 = time.perf_counter()
    if not quiet:
        print(f"{view.dates[trim]}..{view.dates[-1]}: {view.n_days - trim} days (+{trim} warm-up) x "
              f"{view.n_symbols} symbols; {len(exprs)} expressions; {engine.plan(exprs).summary()}")
    res = engine.evaluate(exprs, tolerant=True) if exprs else None
    price = label.price(view)[trim:]
    rows: Dict[str, dict] = {}
    files: Dict[str, str] = {}
    fnames = safe_names(names)
    out = Path(out_dir) if out_dir else None
    if out is not None:
        (out / "values").mkdir(parents=True, exist_ok=True)
    targets = {h: Label(label.kind, h).compute(view, price=price) for h in sorted(set(horizons) | {label.horizon})}
    for i, name in enumerate(names):
        if res.errors.get(i):
            errors[name] = res.errors[i]
            continue
        panel = res.panels[i][trim:].astype(np.float64)
        row = {"expression": repr(parsed[name])}
        row.update(ic_summary(panel, targets[label.horizon]))
        row.update(quantile_longshort(panel, targets[1] if 1 in targets else targets[label.horizon], quantiles))
        for h, tgt in targets.items():
            row[f"ic_h{h}"] = ic_summary(panel, tgt)["ic_mean"]
        rows[name] = row
        if out is not None and save_values:
            fn = fnames[name] + ".npy"
            np.save(out / "values" / fn, panel.astype(np.float32))
            files[name] = fn
    table = pd.DataFrame(rows).T.reindex(list(expressions))
    if out is not None:
        table.to_csv(out / "metrics.csv", index_label="factor")
        (out / "meta.json").write_text(json.dumps({
            "start": start, "end": end, "dates": view.dates[trim:], "symbols": view.symbols,
            "label": label.describe(), "expressions": dict(expressions),
            "value_files": files, "errors": errors, "plan": res.plan.summary() if res else "",
            "seconds": round(time.perf_counter() - t0, 2),
            "value_shape": "(n_days, n_symbols) float32",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    if not quiet:
        with pd.option_context("display.width", 200, "display.max_columns", None,
                               "display.float_format", lambda v: f"{v:8.4f}"):
            cols = [c for c in ("ic_mean", "icir", "rank_ic", "coverage", "ls_sharpe") if c in table.columns]
            print(table[cols] if cols else table)
        if errors:
            print(f"{len(errors)} expression(s) failed:")
            for n, m in list(errors.items())[:10]:
                print(f"  - {n}: {m[:120]}")
        print(f"done in {time.perf_counter() - t0:.1f}s" + (f"; wrote {out}" if out else ""))
    return table


__all__ = ["evaluate_factors", "parse_many", "inline_references", "warmed_view", "safe_names"]
