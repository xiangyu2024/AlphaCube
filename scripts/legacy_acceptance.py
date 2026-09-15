"""Parse + evaluate the whole legacy corpus (1498 factors) on a synthetic
241-bar cube; print pass rates and the failure reasons grouped.

    PYTHONPATH=src python scripts/legacy_acceptance.py [--symbols 6 --days 60] [--store DIR]
"""
import argparse
import collections
import re
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=None, help="existing cube store; default: synthetic")
    ap.add_argument("--symbols", type=int, default=6)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    a = ap.parse_args()
    from alphacube.engine.engine import CubeEngine
    from alphacube.expression import is_valid_root
    from alphacube.legacy import load_legacy_factors
    from alphacube.research import parse_many
    from alphacube.store import CubeStore
    from alphacube.synthetic import write_synthetic_store

    factors = load_legacy_factors()
    if a.limit:
        factors = dict(list(factors.items())[: a.limit])
    t0 = time.time()
    parsed, perr = parse_many(factors, n_minutes=241)
    print(f"parsed {len(parsed)}/{len(factors)} in {time.time() - t0:.1f}s; {len(perr)} parse failures")
    groups = collections.Counter(re.sub(r"'[^']*'", "'…'", m)[:90] for m in perr.values())
    for msg, n in groups.most_common(12):
        print(f"  x{n:<4d} {msg}")
    bad_root = [n for n, e in parsed.items() if not e.shape.degenerate]
    print(f"parsed but not one-value-per-stock-day (minute axis live at the root): {len(bad_root)}")
    tmp = None
    if a.store:
        store = CubeStore(a.store)
    else:
        tmp = tempfile.TemporaryDirectory()
        store = write_synthetic_store(tmp.name, n_symbols=a.symbols, n_days=a.days, n_minutes=241, seed=1)
    view = store.view(start=a.start, end=a.end)
    names = [n for n in parsed if n not in set(bad_root)]
    exprs = [parsed[n] for n in names]
    eng = CubeEngine(view, chunk_symbols=8, chunk_days=16)
    t0 = time.time()
    plan = eng.plan(exprs)
    print(f"plan: {plan.summary()} ({time.time() - t0:.1f}s)")
    t0 = time.time()
    res = eng.evaluate(exprs, tolerant=True)
    ok = sum(1 for p in res.panels if p is not None)
    print(f"evaluated {ok}/{len(exprs)} in {time.time() - t0:.1f}s on {view.n_symbols}x{view.n_days}x{view.n_minutes}")
    egroups = collections.Counter(re.sub(r"\d+", "#", m.split("\n")[0])[:100] for m in res.errors.values())
    for msg, n in egroups.most_common(15):
        print(f"  x{n:<4d} {msg}")
    import numpy as np
    cov = [float(np.isfinite(p).mean()) for p in res.panels if p is not None]
    print(f"coverage: median {np.median(cov):.2f}, >0.5: {sum(c > 0.5 for c in cov)}, all-NaN: {sum(c == 0 for c in cov)}")
    print(f"TOTAL usable: {sum(c > 0 for c in cov)}/{len(factors)}")
    if tmp:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
