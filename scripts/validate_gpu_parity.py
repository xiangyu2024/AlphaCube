"""Device parity: run every registry operator through the engine on a CPU and
on an accelerator, and report where they disagree.

    PYTHONPATH=src python scripts/validate_gpu_parity.py                  # --device cuda (the GPU server)
    PYTHONPATH=src python scripts/validate_gpu_parity.py --device mps     # Apple GPU (local shake-out)

Three classes of outcome, kept apart because only the first is our bug:

  BUG      the operator raised on the accelerator but not on the CPU with a
           message that says device mismatch / host call (``.numpy()``,
           a tensor built without ``device=``, an index on the wrong device).
  BACKEND  the accelerator simply does not implement the op ("not currently
           implemented for the MPS device", "no kernel"). Nothing to fix here;
           MPS is missing kernels CUDA has.
  DELTA    both ran: the difference measured in **float32 ULPs at the scale the
           value came from** (elementwise max(|cpu|, |dev|), floored at the
           panel's 99th percentile magnitude, so an error born of cancellation
           is charged to the scale it cancelled from, not to the tiny result).

A bare ULP count cannot be read as pass or fail on its own, because several
kernels reach their answer through catastrophic cancellation and are worth
thousands of ULPs in float32 *on one device*. So unless ``--no-ref64``, every
probe is also evaluated through a **float64 CPU engine**, and the table reports
how far each device lands from that reference. That turns the question into the
one actually being asked -- *is the GPU answer as good as the CPU answer?* --
and it is scale-free: both columns are errors against the same truth, so a
kernel that is simply ill-conditioned shows two large and equal numbers, while a
device that is genuinely worse shows one.

A device is flagged when its error exceeds ``--slack`` times the CPU's (two
float32 evaluations of the same expression differ by at most the sum of their
errors, and each column is a max over the panel, so a small multiple is the
honest comparison) and the raw gap clears the ``--max-ulp`` floor. Where the
float64 run comes back bit-identical to the float32 one the reference says
nothing -- the kernel is either exact or narrows internally with ``.float()``,
as ``day_mean`` does -- and the column reads ``n/a``; those fall back to the
floor alone.

Finally an end-to-end GP generation runs on the accelerator, which exercises
the miner/pool/label path rather than just the kernels.
"""
import argparse
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# a raise that names the device boundary is our bug; a missing kernel is not
_BUG_PAT = re.compile(
    r"same device|different devices|expected .*device|can't convert .*tensor to numpy|"
    r"numpy\(\) cannot|device type|cuda:0 and cpu|mps:0 and cpu|Placeholder storage",
    re.I,
)
_BACKEND_PAT = re.compile(
    r"not (currently )?implemented for|not supported (on|for) MPS|no kernel|unsupported|"
    r"MPS backend out of memory|Double is not supported|float64",
    re.I,
)


def classify(msg: str) -> str:
    if _BUG_PAT.search(msg):
        return "BUG"
    if _BACKEND_PAT.search(msg):
        return "BACKEND"
    return "BUG"          # unexplained accelerator-only failure: treat as ours until proven otherwise


def ulp_distance(x, y, floor_pct: float = 99.0):
    """Difference between two float32 panels in ULPs at the magnitude it came from.

    ``np.spacing`` of the elementwise larger magnitude is one float32 step there;
    flooring the reference at the panel's 99th-percentile magnitude is what keeps
    a cancelled difference (``in_demean``: two ~100 values subtracted to ~0) charged
    to the ~100 it cancelled from, where its 7.6e-06 is exactly one step, instead of
    reading as a 100%% error on the ~0 that came out."""
    import numpy as np
    m = np.isfinite(x) & np.isfinite(y)
    if not m.any():
        return 0.0, 0.0, 0.0
    ad = np.abs(x[m] - y[m])
    mag = np.maximum(np.abs(x[m]), np.abs(y[m]))
    floor = float(np.percentile(mag, floor_pct)) if mag.size else 0.0
    ref = np.maximum(mag, floor).astype(np.float32)
    step = np.spacing(ref).astype(np.float64)
    ulp = float((ad / np.where(step > 0, step, 1.0)).max())
    return float(ad.max()), ulp, floor


def flagged(t, a) -> bool:
    """A device is at fault when it drifts from the float64 truth further than the
    CPU does by more than ``--slack``, or when the two disagree on which entries
    are finite at all. Without a reference, only the flat ULP floor is left."""
    _name, _ad, ulp, nan_mismatch, _scale, cpu_err, dev_err = t
    if nan_mismatch:
        return True
    if ulp <= a.max_ulp:
        return False
    if dev_err is None:
        return True
    return dev_err > a.slack * max(cpu_err, 1.0)


def build_probe_expressions(reg):
    """One expression per searchable operator, with operands of the shape its
    face demands, wrapped so that every root is one value per stock-day.

    Both the ND-face operands and the collapse wrapper are ``day_last`` -- a
    selection, not a reduction -- and that choice is what makes the numbers mean
    anything. Two reasons:

    * ``day_mean`` as the *wrapper* would average an operator's per-minute output
      away (for ``in_demean`` it is analytically zero), leaving a root made of
      nothing but rounding noise whose two device copies then disagree by 100%%
      of a meaningless scale.
    * ``day_mean``/``day_std`` as the ND *operands* are reductions over 241
      minutes, so the two devices feed the ND kernel slightly different inputs
      and an ill-conditioned kernel (``ts_regbeta``, ``ts_reg_tvalue``) amplifies
      that inherited gap into what looks like -- but is not -- a device fault.
      Worse, the float64 reference cannot see it: ``day_mean`` narrows with
      ``.float()`` internally, so the reference inherits the *CPU's* rounding and
      the comparison is rigged against the GPU. ``day_last``/``day_first``/
      ``day_max`` select an element and come out bit-identical on both devices,
      which isolates the operator under test."""
    from alphacube.expression import Feature, make_node
    full, full2 = Feature("Close"), Feature("Volume")
    daily = make_node(reg.get("day_last"), (Feature("Close"),))
    daily2 = make_node(reg.get("day_last"), (Feature("Volume"),))
    exprs, names = [], []
    for spec in reg.searchable():
        n = 1 if spec.category in ("unary", "rolling") else 2
        if spec.needs_daily:
            ops = (daily, daily2)[:n]
        elif spec.restores_m:
            ops = (daily,)
        else:
            ops = (full, full2)[:n]
        w = 241 if spec.restores_m else (spec.windows[1] if len(spec.windows) > 1 else None)
        try:
            e = make_node(spec, ops, w)
            if not e.shape.degenerate:
                e = make_node(reg.get("day_last"), (e,))
        except Exception:
            continue
        exprs.append(e)
        names.append(spec.name)
    return exprs, names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", help="accelerator to compare against the CPU")
    ap.add_argument("--symbols", type=int, default=40)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--max-ulp", type=float, default=64.0,
                    help="ULP floor: a device gap under this is never flagged, whatever the reference says")
    ap.add_argument("--no-ref64", action="store_true", help="skip the float64 CPU reference run")
    ap.add_argument("--slack", type=float, default=4.0,
                    help="how many times the CPU's own error the device is allowed before it is flagged")
    ap.add_argument("--top", type=int, default=20, help="how many worst deltas to print")
    ap.add_argument("--skip-gp", action="store_true")
    a = ap.parse_args()
    import numpy as np
    import torch
    from alphacube.catalog import default_registry
    from alphacube.engine.engine import CubeEngine
    from alphacube.synthetic import write_synthetic_store

    dev = a.device
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("no CUDA device visible; nothing to validate here"
              + (" (torch sees MPS -- try --device mps)" if torch.backends.mps.is_available() else ""))
        return 2
    if dev.startswith("mps") and not torch.backends.mps.is_available():
        print("no MPS device visible")
        return 2

    reg = default_registry()
    with tempfile.TemporaryDirectory() as d:
        st = write_synthetic_store(d, n_symbols=a.symbols, n_days=a.days, n_minutes=241, seed=1)
        view = st.view()
        exprs, names = build_probe_expressions(reg)
        print(f"{len(exprs)} operators probed on a {a.symbols}x{a.days}x241 cube, cpu vs {dev}")
        cpu = CubeEngine(view, device="cpu").evaluate(exprs, tolerant=True)
        acc = CubeEngine(view, device=dev).evaluate(exprs, tolerant=True)
        ref = None if a.no_ref64 else CubeEngine(view, device="cpu", dtype="float64").evaluate(exprs, tolerant=True)

        bugs, backend, deltas, cpu_only = [], [], [], []
        for i, name in enumerate(names):
            e_cpu, e_acc = cpu.errors.get(i), acc.errors.get(i)
            if e_acc and not e_cpu:
                (bugs if classify(e_acc) == "BUG" else backend).append((name, " ".join(e_acc.split())[:160]))
                continue
            if e_cpu and not e_acc:
                cpu_only.append((name, " ".join(e_cpu.split())[:120]))
                continue
            x, y = cpu.panels[i], acc.panels[i]
            if x is None or y is None:
                continue
            x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
            nan_mismatch = int((np.isfinite(x) != np.isfinite(y)).sum())
            ad, ulp, scale = ulp_distance(x, y)
            cpu_err = dev_err = None               # each device's own distance from the float64 truth
            if ref is not None and ref.panels[i] is not None:
                r = np.asarray(ref.panels[i], dtype=np.float64)
                if not np.array_equal(x, r, equal_nan=True) or not np.array_equal(y, r, equal_nan=True):
                    cpu_err, dev_err = ulp_distance(x, r)[1], ulp_distance(y, r)[1]
            deltas.append((name, ad, ulp, nan_mismatch, scale, cpu_err, dev_err))

        ran = len(deltas)
        big = [t for t in deltas if flagged(t, a)]
        print(f"\n  ran on both: {ran}   accelerator-only failures: {len(bugs)} BUG + {len(backend)} BACKEND"
              f"   cpu-only failures: {len(cpu_only)}")
        for label, rows in (("BUG", bugs), ("BACKEND (accelerator lacks the kernel)", backend),
                            ("CPU-ONLY failure", cpu_only)):
            if rows:
                print(f"\n{label}:")
                for name, msg in sorted(rows):
                    print(f"  {name:34s} {msg}")
        order = sorted(deltas, key=lambda t: (-(t[6] / t[5]) if t[5] else 0.0, -t[2]))
        verdict = (f"{len(big)} operator(s) where {dev} is measurably worse than the CPU"
                   if big else f"no operator is measurably worse on {dev}")
        print(f"\n{verdict} (flagged at error > {a.slack:g}x the CPU's with a gap over"
              f" {a.max_ulp:g} ULP). Worst deltas either way:")
        print(f"  {'operator':34s} {'max|cpu-dev|':>12s} {'gap ULP':>10s} {'cpu err':>10s} {'%s err' % dev:>10s}  verdict")
        for t in order[:a.top]:
            name, ad, ulp, nm, _scale, cpu_err, dev_err = t
            c = "n/a" if cpu_err is None else f"{cpu_err:.1f}"
            g = "n/a" if dev_err is None else f"{dev_err:.1f}"
            note = "WORSE" if flagged(t, a) else ("order" if ulp > a.max_ulp else "ok")
            print(f"  {name:34s} {ad:12.3e} {ulp:10.1f} {c:>10s} {g:>10s}  {note}"
                  + (f"  nan-mismatch {nm}" if nm else ""))
        if ref is not None:
            worst = sorted((t for t in deltas if t[5]), key=lambda t: -t[5])[:8]
            if worst:
                print("\nseparately: kernels whose float32 answer is itself far from float64"
                      " -- a property of the kernel, not of the device:")
                for name, _, _, _, _, cpu_err, _dev in worst:
                    print(f"  {name:34s} {cpu_err:10.1f} ULP on cpu  ~ {cpu_err * 1.19e-7:.2e} relative")

        if not a.skip_gp:
            from alphacube.label import Label
            from alphacube.mining.gp import GPMiner
            from alphacube.mining.pool import MsePool
            target = Label("close", 1).compute(view)
            m = GPMiner(reg, view, target, MsePool(5, target), population_size=12, max_depth=3,
                        device=dev, seed=0)
            rep = m.mine(24)
            print(f"\nGP on {dev}: {rep.evaluated} evaluations in {rep.engine_seconds:.1f}s, pool {m.pool.size}")
    return 1 if (bugs or big) else 0


if __name__ == "__main__":
    sys.exit(main())
