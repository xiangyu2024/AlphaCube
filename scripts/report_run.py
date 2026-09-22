"""Single-factor and combination backtests of one run's archive, as JSON (run on the server).

    cd /root/autodl-tmp/AlphaCube && PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
      /root/miniconda3/envs/quant/bin/python scripts/report_run.py \\
      mining_runs/gpu12/checkpoints/archive_gen020.csv _bench/gpu12_report.json --gpu-fraction 0.95

Everything after costs (15 bp a side) on the top/bottom-decile, equal-weight,
gross-1 book the run reports use, label open_vwap15 h=1 (realized = target):

- factors: the archive's ``--top-factors`` leaders (one per family when the run
  used archive_dedup). Each is traded in the direction of its IC on the TRAINING
  days (the first ``--train-frac``), and reported on the full period, the
  training days and the test days, with daily turnover, IC / rank IC and the net
  return of every calendar year.
- combination: the archive's ``--pool-top`` leaders offered to the run's pool
  (MsePool, 30, dedup 0.95 -- the final commit took the top 60), then every
  combination method fitted on the training days only and traded on the test
  days: metrics, the daily net equity curve, per-year and per-month net returns,
  and the same for the EMA-smoothed signal with the half-life picked on the
  training days.

The factor ranking inside the archive is itself in-sample (the GP scored the
whole period), so "test" is out of sample for the combination weights only.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np

METHODS = ("equal", "ic", "icir", "lstsq", "ridge", "gbdt", "lightgbm", "xgboost")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("out")
    ap.add_argument("--store", default="/root/autodl-tmp/qmt_daily_data/fast_factor_compute")
    ap.add_argument("--start", default="20180102")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--label", default="open_vwap15")
    ap.add_argument("--top-factors", type=int, default=40)
    ap.add_argument("--pool-top", type=int, default=60)
    ap.add_argument("--capacity", type=int, default=30)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--cost-per-side", type=float, default=0.0015)
    ap.add_argument("--turnover-cap", type=float, default=0.30)
    ap.add_argument("--smooth-halflives", default="1,2,3,5,10,20")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--gpu-fraction", type=float, default=0.95, help="0 = CPU only")
    a = ap.parse_args()

    import torch
    dev = "cpu" if a.gpu_fraction == 0 or not torch.cuda.is_available() else "cuda"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.gpu_fraction)

    from alphacube.combine.post_combine import compare
    from alphacube.engine.engine import CubeEngine
    from alphacube.label import Label
    from alphacube.metrics import (apply_turnover_cap, cs_pearson, ema_rows, ic_summary, long_short_weights,
                                   portfolio_returns)
    from alphacube.parser import parse
    from alphacube.pipeline import build_pool
    from alphacube.research import warmed_view
    from alphacube.store import open_cube_store

    rows = list(csv.DictReader(open(a.archive, encoding="utf-8")))
    view, trim = warmed_view(open_cube_store(a.store), a.start or None, a.end or None, warmup=a.warmup)
    target = Label(a.label, 1).compute(view).astype(np.float64)
    dates = [str(d) for d in view.dates]
    years = np.asarray([d[:4] for d in dates])
    months = np.asarray([d[:6] for d in dates])
    D = view.n_days
    split = max(1, min(D - 1, int(D * a.train_frac)))            # compare()'s split
    train, test = slice(0, split), slice(split, D)
    print(f"view {D} days x {view.n_symbols}; train {dates[0]}..{dates[split - 1]}, "
          f"test {dates[split]}..{dates[-1]}", flush=True)

    # one evaluation for everything: factor leaders and pool candidates overlap
    wanted = rows[: max(a.top_factors, a.pool_top)]
    exprs, keep = [], []
    for r in wanted:
        try:
            exprs.append(parse(r["expression"]))
            keep.append(r)
        except Exception as exc:  # noqa: BLE001 - an archive may hold a retired operator
            print(f"  skip (parse): {r['expression'][:60]} -> {exc}", flush=True)
    t0 = time.time()
    panels = []
    eng = CubeEngine(view, device=dev)
    for lo in range(0, len(exprs), 100):
        res = eng.evaluate(exprs[lo:lo + 100], tolerant=True)
        panels.extend(res.panels)
    if dev == "cuda" and any(p is None for p in panels):
        idx = [i for i, p in enumerate(panels) if p is None]
        r2 = CubeEngine(view, device="cpu").evaluate([exprs[i] for i in idx], tolerant=True)
        for i, p in zip(idx, r2.panels):
            panels[i] = p
    print(f"{dev}: evaluated {len(exprs)} in {time.time() - t0:.0f}s, "
          f"failed {sum(p is None for p in panels)}", flush=True)

    cost, cap = a.cost_per_side, a.turnover_cap

    def curve_stats(v):
        f = v[np.isfinite(v)]
        if f.size < 2 or f.std() == 0:
            return {"sharpe": float("nan"), "ann_return": float("nan"), "max_dd": float("nan")}
        eq = np.cumsum(f)
        return {"sharpe": float(f.mean() / f.std() * np.sqrt(252)), "ann_return": float(f.mean() * 252),
                "max_dd": float((eq - np.maximum.accumulate(eq)).min())}

    def book_returns(alpha, days):
        """Daily (net, gross, turnover) of the decile book over ``days`` (a slice), plus the capped net."""
        w = long_short_weights(alpha[days], 0.1)
        net, gross, turn = portfolio_returns(w, target[days], cost)
        c_net, _, c_turn = portfolio_returns(apply_turnover_cap(w, cap), target[days], cost)
        return net, gross, turn, c_net, c_turn

    def by_period(values, keys):
        out = {}
        for k in sorted(set(keys)):
            m = keys == k
            f = values[m][np.isfinite(values[m])]
            out[k] = float(f.sum()) if f.size else None
        return out

    # ---- single factors --------------------------------------------------------------
    factors = []
    for r, e, p in list(zip(keep, exprs, panels))[: a.top_factors]:
        if p is None:
            continue
        pan = np.asarray(p, dtype=np.float64)
        ic_train = np.nanmean(cs_pearson(pan[train], target[train]))
        sign = -1.0 if ic_train < 0 else 1.0
        pf = sign * pan
        s_full = ic_summary(pf, target)
        net, gross, turn, c_net, c_turn = book_returns(pf, slice(0, D))
        entry = {
            "rank": int(r["rank"]), "expression": r["expression"], "fitness": float(r["fitness"]),
            "dups": int(r["dups"]) if r.get("dups") else None, "sign": sign,
            "ic": s_full["ic_mean"], "rank_ic": s_full["rank_ic"], "icir": s_full["icir"],
            "coverage": s_full["coverage"], "turnover": float(np.mean(turn)),
            "full": {"gross": curve_stats(gross), "net": curve_stats(net), "capped": curve_stats(c_net)},
            "train": {"net": curve_stats(net[train]), "gross": curve_stats(gross[train])},
            "test": {"net": curve_stats(net[test]), "gross": curve_stats(gross[test]),
                     "ic": float(np.nanmean(cs_pearson(pf[test], target[test])))},
            "by_year_net": by_period(net, years),
            "equity_net": [float(x) for x in np.nancumsum(net)[::5]],
        }
        factors.append(entry)
        print(f"  [{entry['rank']:>3}] net Sharpe full {entry['full']['net']['sharpe']:+.2f} "
              f"train {entry['train']['net']['sharpe']:+.2f} test {entry['test']['net']['sharpe']:+.2f} "
              f"turnover {entry['turnover']:.1%}  {r['expression'][:70]}", flush=True)

    # ---- the pool and every combination method ------------------------------------------
    cand = [(e, p) for e, p in list(zip(exprs, panels))[: a.pool_top] if p is not None]
    pool = build_pool("mse", a.capacity, target, target, 0.95, 0.5, min_scored_days=0.5)
    adm = pool.admit(cand)
    print(f"pool: {len(cand)} candidates -> {pool.size} members", flush=True)
    halflives = tuple(float(x) for x in a.smooth_halflives.split(",") if x)
    methods = tuple(x for x in a.methods.split(",") if x)
    results, signs = compare(pool.z_columns(), target, target, methods=methods, train_frac=a.train_frac,
                             align=True, seed=0, cost_per_side=cost, turnover_cap=cap,
                             smooth_halflives=halflives)
    combos = []
    for res in results:
        if "error" in res.metrics:
            combos.append({"method": res.method, "error": res.metrics["error"]})
            continue
        alpha = res.combined
        net, gross, turn, c_net, c_turn = book_returns(alpha, test)
        item = {"method": res.method, "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                                  for k, v in res.metrics.items()},
                "test": {"net": curve_stats(net), "gross": curve_stats(gross), "capped": curve_stats(c_net),
                         "turnover": float(np.mean(turn))},
                "train": {"net": curve_stats(book_returns(alpha, train)[0])},
                "by_year_net": by_period(net, years[test]), "by_month_net": by_period(net, months[test]),
                "equity_net": [float(x) for x in np.nancumsum(net)],
                "equity_gross": [float(x) for x in np.nancumsum(gross)]}
        hl = res.metrics.get("halflife")
        if hl is not None:
            sm = ema_rows(alpha, float(hl))
            s_net, _, s_turn, _, _ = book_returns(sm, test)
            item["smooth"] = {"halflife": float(hl), "net": curve_stats(s_net), "turnover": float(np.mean(s_turn)),
                              "by_year_net": by_period(s_net, years[test]),
                              "equity_net": [float(x) for x in np.nancumsum(s_net)]}
        combos.append(item)
        print(f"  {res.method:>8}: test net Sharpe {item['test']['net']['sharpe']:+.2f} "
              f"(gross {item['test']['gross']['sharpe']:+.2f}, turnover {item['test']['turnover']:.1%})"
              + (f"; smoothed hl {item['smooth']['halflife']:.0f}: {item['smooth']['net']['sharpe']:+.2f}"
                 if "smooth" in item else ""), flush=True)

    out = {
        "archive": a.archive, "label": f"{a.label} h=1", "cost_per_side": cost, "turnover_cap": cap,
        "view": {"start": dates[0], "end": dates[-1], "n_days": D, "n_symbols": view.n_symbols,
                 "split": dates[split], "train_frac": a.train_frac},
        "dates": dates, "factor_dates": dates[::5], "test_dates": dates[split:],
        "factors": factors,
        "pool": {"candidates": len(cand), "size": pool.size, "rejections": adm.reasons,
                 "members": [{"expression": repr(m.expr), "weight": float(w), "ic": float(m.ic),
                              "sign": float(s)} for m, w, s in zip(pool.members, pool.weights, signs)]},
        "combinations": combos,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
