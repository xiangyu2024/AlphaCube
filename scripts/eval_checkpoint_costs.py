"""After-cost backtests for the leaders of a GP checkpoint (run on the server).

    PYTHONPATH=src python scripts/eval_checkpoint_costs.py \\
        mining_runs/gpu9/checkpoints/archive_gen010.csv out.json 20 0.045

    args: <archive.csv> <out.json> [top_n=10] [gpu_fraction=0.045]
          gpu_fraction 0 = evaluate on the CPU and leave the GPU alone (slower, but
          safe next to a running job); >0 caps this process's GPU memory so it can
          never crowd a live run (a CUDA OOM inside a run is caught per node and
          silently fails trees rather than crashing).

The checkpoint's own metrics are no-cost (backtest_metrics without costs). This
re-evaluates each expression on the production view the way pipeline.py scores
members -- 20180102-20251231, label open_vwap15 h=1 (realized = target),
top/bottom 10% equal-weight gross-1 book, daily rebalance -- with 15 bp per side
and a 30% daily turnover cap. Each factor is flipped to its tradable direction
(sign of IC) first, as the combination layer does. It first checks that the
no-cost numbers reproduce the checkpoint (they do to ~1e-7, except kernels with
thresholds on cancelled sums such as day_durbin_watson, which move with CUDA
chunk size -- see HANDOFF). Also writes per-year gross / net / capped P&L.
"""
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch

STORE = "/root/autodl-tmp/qmt_daily_data/fast_factor_compute"
arch, out_path = sys.argv[1], sys.argv[2]
top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
FRAC = float(sys.argv[4]) if len(sys.argv) > 4 else 0.045
DEV = "cpu" if FRAC == 0 else "cuda"
if DEV == "cuda":
    torch.cuda.set_per_process_memory_fraction(FRAC)

from alphacube.engine.engine import CubeEngine
from alphacube.label import Label
from alphacube.metrics import (apply_turnover_cap, backtest_metrics, cs_pearson, ic_summary, long_short_weights,
                               portfolio_returns)
from alphacube.parser import parse
from alphacube.research import warmed_view
from alphacube.store import open_cube_store

rows = list(csv.DictReader(open(arch, encoding="utf-8")))[:top_n]
store = open_cube_store(STORE)
view, trim = warmed_view(store, "20180102", "20251231", warmup=60)
t0 = time.time()
target = Label("open_vwap15", 1).compute(view).astype(np.float64)
print(f"view {view.n_days} days (trim {trim}) x {view.n_symbols}; label {time.time()-t0:.0f}s", flush=True)
exprs = [parse(r["expression"]) for r in rows]
t0 = time.time()
kw = {} if DEV == "cpu" else dict(chunk_symbols=5, chunk_days=2)
res = CubeEngine(view, device=DEV, **kw).evaluate(exprs, tolerant=True)
panels = list(res.panels)
print(f"{DEV}: evaluated {len(exprs)} in {time.time()-t0:.0f}s, failed {sorted(res.errors)}", flush=True)
if res.errors:                                      # what does not fit under the GPU cap goes to the CPU
    idx = sorted(res.errors)
    r2 = CubeEngine(view, device="cpu").evaluate([exprs[i] for i in idx], tolerant=True)
    for i, p in zip(idx, r2.panels):
        panels[i] = p

years = np.asarray([d[:4] for d in view.dates[trim:]])
tg = target[trim:]
out = []
for r, p in zip(rows, panels):
    if p is None:
        print(f"[{r['rank']:>2}] no panel")
        continue
    pan = np.asarray(p, dtype=np.float64)
    chk = {**ic_summary(pan, target), **backtest_metrics(cs_pearson(pan, target), pan, target)}
    d_ic = abs(chk["ic_mean"] - float(r["ic_mean"])) if r.get("ic_mean") else float("nan")
    pt = pan[trim:]
    sign = 1.0 if ic_summary(pt, tg)["ic_mean"] >= 0 else -1.0
    pf = sign * pt
    ic = ic_summary(pf, tg)
    bt = backtest_metrics(cs_pearson(pf, tg), pf, tg, cost_per_side=0.0015, turnover_cap=0.30)
    w = long_short_weights(pf, 0.1)
    net, gross, _ = portfolio_returns(w, tg, 0.0015)
    c_net, _, _ = portfolio_returns(apply_turnover_cap(w, 0.30), tg, 0.0015)
    by_year = {y: {"gross": float(np.nansum(gross[years == y])), "net": float(np.nansum(net[years == y])),
                   "capped_net": float(np.nansum(c_net[years == y]))} for y in sorted(set(years))}
    out.append({"rank": int(r["rank"]), "expression": r["expression"], "sign": sign, "check_d_ic": d_ic,
                **{k: ic[k] for k in ("ic_mean", "icir", "rank_ic", "coverage", "n_days", "ic_pos")},
                **{k: float(v) for k, v in bt.items() if k not in ("ic_mean", "ic_std", "icir")},
                "by_year": by_year})
    print(f"[{r['rank']:>2}] dIC {d_ic:.1e} | {'flip' if sign < 0 else 'keep'} IC {ic['ic_mean']:+.4f} "
          f"sharpe gross {bt['sharpe']:+.2f} net {bt['sharpe_net']:+.2f} capped {bt['sharpe_capped']:+.2f} "
          f"turnover {bt['turnover']:.0%}  {r['expression'][:80]}", flush=True)
json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=1)
print(f"wrote {out_path}", flush=True)
