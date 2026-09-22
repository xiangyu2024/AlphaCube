"""Re-evaluate every factor any run has archived, with today's engine (run on the server).

    cd /root/autodl-tmp/AlphaCube && PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
      /root/miniconda3/envs/quant/bin/python -u scripts/eval_all_factors.py mining_runs _bench/allfac

Collects the latest checkpoint of every run (``<run>/checkpoints/archive_genNNN.csv``,
or ``members.csv`` for runs without checkpoints), one entry per expression
(structural key), dropping expressions that use operators the search no longer
breeds -- the exploit-era ones (trivial booleans, binary pow, per-element loops).
Every panel is recomputed with the current engine (the numbers the runs printed
predate several fixes) and saved as ``panels/<i>.npy`` (float32, days x symbols
of the production view); ``manifest.csv`` maps i -> run, expression.

Resumable: panels already on disk are skipped, so a killed run just restarts.
Scoring and training read the panels afterwards (score_all_factors.py,
train_all_factors.py).
"""
import argparse
import csv
import glob
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np


def collect(runs_dir: Path):
    from alphacube.catalog import default_registry
    from alphacube.parser import parse

    reg = default_registry()
    searchable = {s.name for s in reg.searchable(include_boolean=False, include_resource_risk=False)}
    excluded = {s.name for s in reg.list_all()} - searchable
    gen = lambda p: int(re.search(r"gen(\d+)", os.path.basename(p)).group(1))   # noqa: E731
    seen, out, dropped = set(), [], 0
    for run in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        ck = sorted(glob.glob(str(run / "checkpoints" / "archive_gen*.csv")), key=gen)
        src = [Path(ck[-1])] if ck else ([run / "members.csv"] if (run / "members.csv").exists() else [])
        for f in src:
            for r in csv.DictReader(open(f, encoding="utf-8")):
                try:
                    e = parse(r["expression"])
                except Exception:  # noqa: BLE001
                    continue
                if e.key() in seen:
                    continue
                seen.add(e.key())
                if {n.spec.name for n in e.nodes() if n.spec is not None} & excluded:
                    dropped += 1
                    continue
                out.append((run.name, r["expression"], e))
    return out, dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--store", default="/root/autodl-tmp/qmt_daily_data/fast_factor_compute")
    ap.add_argument("--mirror-root", default="/root/autodl-tmp/AlphaCube/mirror")
    ap.add_argument("--start", default="20180102")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--batch", type=int, default=150)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="first N expressions only (testing)")
    a = ap.parse_args()

    from alphacube.engine.engine import CubeEngine
    from alphacube.research import warmed_view
    from alphacube.store import open_cube_store

    out = Path(a.out_dir)
    (out / "panels").mkdir(parents=True, exist_ok=True)
    items, dropped = collect(Path(a.runs_dir))
    if a.limit:
        items = items[: a.limit]
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["i", "run", "expression"])
        for i, (run, text, _) in enumerate(items):
            w.writerow([i, run, text])
    print(f"{len(items)} expressions ({dropped} dropped for retired operators)", flush=True)

    view, trim = warmed_view(open_cube_store(a.store, mirror_root=a.mirror_root or None), a.start or None,
                             a.end or None, warmup=a.warmup)
    np.save(out / "dates.npy", np.asarray(view.dates))
    print(f"view {view.n_days} days x {view.n_symbols} (warm-up {trim})", flush=True)
    eng = CubeEngine(view, device=a.device, workdir=Path(__file__).resolve().parents[1] / "tmp")
    todo = [i for i in range(len(items)) if not (out / "panels" / f"{i:05d}.npy").exists()]
    print(f"{len(items) - len(todo)} already on disk, {len(todo)} to evaluate", flush=True)
    failed = 0
    t_all = time.time()
    for b0 in range(0, len(todo), a.batch):
        idx = todo[b0:b0 + a.batch]
        t0 = time.time()
        res = eng.evaluate([items[i][2] for i in idx], tolerant=True)
        for i, p in zip(idx, res.panels):
            if p is None:
                failed += 1
                continue
            tmp = out / "panels" / f"{i:05d}.npy.tmp"
            with open(tmp, "wb") as fh:
                np.save(fh, np.asarray(p, dtype=np.float32))
            os.replace(tmp, out / "panels" / f"{i:05d}.npy")
        done = b0 + len(idx)
        el = time.time() - t_all
        print(f"batch {b0 // a.batch + 1}: {len(idx)} in {time.time() - t0:.0f}s, failed so far {failed}; "
              f"{done}/{len(todo)} done, eta {el / done * (len(todo) - done) / 60:.0f} min", flush=True)
    print(f"done: {len(items) - failed} panels, {failed} failed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
