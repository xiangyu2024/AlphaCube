"""Score every panel eval_all_factors.py saved: health, single-factor backtests, rank rows.

    cd /root/autodl-tmp/AlphaCube && PYTHONPATH=src /root/miniconda3/envs/quant/bin/python -u \\
      scripts/score_all_factors.py _bench/allfac --workers 8

Per factor, on the production view and label (open_vwap15 h=1), train = the
first ``--train-frac`` of days, test = the rest:

- health: coverage (finite factor & label over finite label, the pool's gate),
  the same on the test days, and the share of usable days with a finite IC --
  healthy = finite IC, coverage >= 0.5, test coverage >= 0.5, scored days >= 0.5;
- IC and rank IC (train / test / full), daily turnover of the decile book;
- the decile book traded in the direction of the TRAINING-period IC, 15 bp a
  side: net Sharpe on the full, training and test days, gross Sharpe, annual net
  return, max drawdown, net return by calendar year;
- its compact rank rows (250 days x 512 symbols, float32) for de-duplication.

Writes ``metrics.csv`` and ``rows.npy`` (factor order = manifest order; rows of
factors without a panel are zero). Workers fork after the label is computed.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np

G = {}          # each worker's own copy (spawned, not forked)


def init(d, split, cost, threads):
    """Worker set-up. The pool is SPAWNED, not forked: the parent has already run
    torch (the label's ranks), and forking a process whose OpenMP pool is up
    deadlocks every child at its first torch call -- this scorer sat there for 36
    hours with eight workers asleep. Spawned workers re-import torch cleanly and
    mmap the label the parent saved, so the pages are shared, not copied."""
    import torch

    from alphacube.mining.rankban import CompactRankIndex

    torch.set_num_threads(threads)
    target = np.load(d / "_target.npy", mmap_mode="r")
    dates = [str(x) for x in np.load(d / "dates.npy")]
    G.update(dir=d, target=np.asarray(target), target_ranks=torch.from_numpy(np.asarray(
        np.load(d / "_target_ranks.npy", mmap_mode="r"))), split=split, cost=cost,
        years=np.asarray([x[:4] for x in dates]), years_list=sorted({x[:4] for x in dates}),
        index=CompactRankIndex(target.shape), threads=threads)


def score(i):
    from alphacube.metrics import (cs_pearson, long_short_weights, nanmean_quiet, portfolio_returns,
                                   rank_ic_turnover)

    path = G["dir"] / "panels" / f"{i:05d}.npy"
    if not path.exists():
        return i, None, None
    pan = np.load(path).astype(np.float64)
    tgt, split, cost = G["target"], G["split"], G["cost"]
    tf = np.isfinite(tgt)
    both = np.isfinite(pan) & tf
    cov = float(both.sum() / max(tf.sum(), 1))
    cov_test = float(both[split:].sum() / max(tf[split:].sum(), 1))
    ic_days = cs_pearson(pan, tgt)
    usable = max(int((tf.sum(axis=1) >= 5).sum()), 1)
    scored = float(np.isfinite(ic_days).sum() / usable)
    ric_days, turnover = rank_ic_turnover(pan, G["target_ranks"], 0.1)
    ic_tr = nanmean_quiet(ic_days[:split])
    sign = -1.0 if ic_tr < 0 else 1.0
    w = long_short_weights(sign * pan, 0.1)
    net, gross, _ = portfolio_returns(w, tgt, cost)

    def sh(v):
        f = v[np.isfinite(v)]
        return float(f.mean() / f.std() * np.sqrt(252)) if f.size > 1 and f.std() > 0 else float("nan")

    f = net[np.isfinite(net)]
    eq = np.cumsum(f) if f.size else np.zeros(1)
    by_year = {}
    for y in G["years_list"]:
        m = G["years"] == y
        v = net[m][np.isfinite(net[m])]
        by_year[y] = float(v.sum()) if v.size else None
    row = {
        "i": i, "coverage": cov, "coverage_test": cov_test, "scored": scored,
        "ic_train": ic_tr, "ic_test": nanmean_quiet(ic_days[split:]), "ic": nanmean_quiet(ic_days),
        "rank_ic_train": nanmean_quiet(ric_days[:split]), "rank_ic_test": nanmean_quiet(ric_days[split:]),
        "rank_ic": nanmean_quiet(ric_days), "turnover": float(turnover), "sign": sign,
        # ICIR = 逐日 IC 的均值/标准差。单因子层面看稳定性比看强度值钱:同样 |IC|,
        # ICIR 高的那个才是每天都在赚,而不是几天赚完剩下全在回吐。
        "icir": _icir(ic_days), "icir_train": _icir(ic_days[:split]), "icir_test": _icir(ic_days[split:]),
        "rank_icir": _icir(ric_days), "rank_icir_test": _icir(ric_days[split:]),
        "net_sharpe": sh(net), "net_sharpe_train": sh(net[:split]), "net_sharpe_test": sh(net[split:]),
        "gross_sharpe": sh(gross), "gross_sharpe_test": sh(gross[split:]),
        "ann_net": float(f.mean() * 252) if f.size else float("nan"),
        "max_dd": float((eq - np.maximum.accumulate(eq)).min()),
        "by_year_net": json.dumps(by_year),
    }
    row["healthy"] = bool(np.isfinite(row["ic"]) and cov >= 0.5 and cov_test >= 0.5 and scored >= 0.5)
    rows = G["index"].rows(pan)
    return i, row, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--store", default="/root/autodl-tmp/qmt_daily_data/fast_factor_compute")
    ap.add_argument("--start", default="20180102")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--label", default="open_vwap15")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--cost-per-side", type=float, default=0.0015)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    import multiprocessing as mp

    from alphacube.label import Label
    from alphacube.metrics import torch_rank_rows
    from alphacube.mining.gp import cpu_quota
    from alphacube.mining.rankban import CompactRankIndex
    from alphacube.research import warmed_view
    from alphacube.store import open_cube_store

    d = Path(a.dir)
    manifest = list(csv.DictReader(open(d / "manifest.csv", encoding="utf-8")))
    view, _ = warmed_view(open_cube_store(a.store), a.start or None, a.end or None, warmup=a.warmup)
    dates = [str(x) for x in view.dates]
    saved = [str(x) for x in np.load(d / "dates.npy")]
    if dates != saved:
        raise SystemExit("the view differs from the one the panels were evaluated on")
    target = Label(a.label, 1).compute(view).astype(np.float64)
    D = len(dates)
    split = max(1, min(D - 1, int(D * a.train_frac)))
    np.save(d / "_target.npy", target)
    np.save(d / "_target_ranks.npy", torch_rank_rows(target).numpy())
    threads = max(1, (cpu_quota() or os.cpu_count() or 8) // max(1, a.workers))
    width = CompactRankIndex(target.shape).width
    rows_all = np.lib.format.open_memmap(d / "rows.npy", mode="w+", dtype=np.float32,
                                         shape=(len(manifest), 3, width))
    out, t0 = [], time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=init, initargs=(d, split, a.cost_per_side, threads)) as pool:
        for n, (i, row, rows) in enumerate(pool.imap_unordered(score, range(len(manifest)), chunksize=4), 1):
            if row is not None:
                row.update(run=manifest[i]["run"], expression=manifest[i]["expression"])
                out.append(row)
                rows_all[i] = rows
            if n % 50 == 0:
                print(f"  {n}/{len(manifest)} scored, {time.time() - t0:.0f}s", flush=True)
    rows_all.flush()
    out.sort(key=lambda r: r["i"])
    cols = list(out[0].keys())
    with open(d / "metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    healthy = sum(r["healthy"] for r in out)
    print(f"scored {len(out)} of {len(manifest)} ({healthy} healthy) in {time.time() - t0:.0f}s -> "
          f"{d / 'metrics.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
