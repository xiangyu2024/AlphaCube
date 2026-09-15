"""Run every miner on the same store / label / budget and print a leaderboard.

    PYTHONPATH=src python scripts/compare_miners.py --store <dir> [--budget 200 --miners gp,alphagen,...]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--miners", default="gp,alphagen,alphaqcm,alphaforge,alphasage,warmstart")
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--pool-capacity", type=int, default=5)
    ap.add_argument("--label", default="close")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--max-symbols", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    import pandas as pd
    from alphacube.pipeline import run_mining

    rows = []
    for miner in [m for m in a.miners.split(",") if m]:
        print(f"== {miner}", flush=True)
        r = run_mining(a.store, start=a.start, end=a.end, max_symbols=a.max_symbols, label=a.label,
                       horizon=a.horizon, miner=miner, budget=a.budget, seed=a.seed, device=a.device,
                       pool_capacity=a.pool_capacity, combine_methods=("equal", "ic", "lstsq"), log=None)
        best = None
        if r.combination is not None and "icir" in r.combination.columns:
            best = r.combination["icir"].astype(float).max()
        rows.append({"miner": miner, "pool": r.pool_size, "added": r.added,
                     "pool_ic": r.pool_metrics.get("ic_mean", float("nan")),
                     "pool_icir": r.pool_metrics.get("icir", float("nan")),
                     "best_oos_icir": best, "seconds": round(r.seconds, 1), "engine_s": round(r.engine_seconds, 1)})
    table = pd.DataFrame(rows).sort_values("pool_ic", ascending=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
