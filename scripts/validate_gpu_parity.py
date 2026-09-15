"""RUN ON THE GPU SERVER. (1) every registry operator on CUDA vs CPU through the
engine on a synthetic cube (device-mismatch bugs surface here); (2) max abs
delta per operator (trailing-ULP deltas are expected physics; anything large
is a real problem); (3) an end-to-end GP generation on cuda.

    PYTHONPATH=src python scripts/validate_gpu_parity.py [--symbols 40 --days 60]
"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=int, default=40)
    ap.add_argument("--days", type=int, default=60)
    a = ap.parse_args()
    import numpy as np
    import torch
    from alphacube.catalog import default_registry
    from alphacube.engine.engine import CubeEngine
    from alphacube.expression import Constant, Feature, make_node
    from alphacube.synthetic import write_synthetic_store

    if not torch.cuda.is_available():
        print("no CUDA device visible; nothing to validate here")
        return 2
    reg = default_registry()
    with tempfile.TemporaryDirectory() as d:
        st = write_synthetic_store(d, n_symbols=a.symbols, n_days=a.days, n_minutes=241, seed=1)
        view = st.view()
        base = {"full": Feature("Close"), "full2": Feature("Volume"), "daily": make_node(reg.get("day_mean"), (Feature("Close"),)),
                "daily2": make_node(reg.get("day_std"), (Feature("Ret"),))}
        exprs, names = [], []
        for spec in reg.searchable():
            n = 1 if spec.category in ("unary", "rolling") else 2
            if spec.needs_daily:
                ops = (base["daily"], base["daily2"])[:n]
            elif spec.restores_m:
                ops = (base["daily"],)
            else:
                ops = (base["full"], base["full2"])[:n]
            w = 241 if spec.restores_m else (spec.windows[1] if len(spec.windows) > 1 else None)
            try:
                e = make_node(spec, ops, w)
            except Exception:
                continue
            if not e.shape.degenerate:
                e = make_node(reg.get("day_mean"), (e,)) if not e.shape.degenerate else e
            exprs.append(e)
            names.append(spec.name)
        cpu = CubeEngine(view, device="cpu").evaluate(exprs, tolerant=True)
        gpu = CubeEngine(view, device="cuda").evaluate(exprs, tolerant=True)
        failed, deltas = [], []
        for name, i in zip(names, range(len(exprs))):
            if gpu.errors.get(i) and not cpu.errors.get(i):
                failed.append((name, gpu.errors[i][:100]))
            elif gpu.panels[i] is not None and cpu.panels[i] is not None:
                x, y = cpu.panels[i], gpu.panels[i]
                m = np.isfinite(x) & np.isfinite(y)
                deltas.append((name, float(np.abs(x[m] - y[m]).max()) if m.any() else 0.0,
                               int((np.isfinite(x) != np.isfinite(y)).sum())))
        print(f"{len(deltas)} operators ran on CUDA; {len(failed)} failed only on CUDA")
        for name, msg in failed:
            print(f"  FAIL {name}: {msg}")
        for name, dlt, nan_mismatch in sorted(deltas, key=lambda t: -t[1])[:20]:
            print(f"  {name:32s} max|cpu-gpu| {dlt:.3e}  nan-mismatch {nan_mismatch}")
        from alphacube.label import Label
        from alphacube.mining.gp import GPMiner
        from alphacube.mining.pool import MsePool
        target = Label("close", 1).compute(view)
        m = GPMiner(reg, view, target, MsePool(5, target), population_size=12, max_depth=3, device="cuda", seed=0)
        rep = m.mine(24)
        print(f"GP on cuda: {rep.evaluated} evaluations in {rep.engine_seconds:.1f}s, pool {m.pool.size}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
