"""Train combinations on every healthy factor from every run (after score_all_factors.py).

    cd /root/autodl-tmp/AlphaCube && PYTHONPATH=src /root/miniconda3/envs/quant/bin/python -u \\
      scripts/train_all_factors.py _bench/allfac _bench/allfac/train.json

Only TRAINING-period information chooses anything:
1. healthy factors (score_all_factors.py) ordered by training-period net Sharpe;
2. de-duplicated greedily: a factor whose daily ranks correlate >= ``--dedup``
   with an already kept one is its rewrite and dropped (compact rank rows);
3. two selections: the first ``--k`` distinct factors, and an MsePool of 30
   fitted to the training days only (the runs' pools saw the whole label) over
   the first ``--pool-top`` distinct ones;
4. every combination method fitted on the training days, traded on the test
   days after 15 bp a side, with the EMA-smoothed signal as well.

The factors themselves were found by GPs that scored the whole period, so the
test days are out of sample for the combination and the selection here only.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np

METHODS = ("equal", "ic", "icir", "lstsq", "ridge", "gbdt", "lightgbm")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("out")
    ap.add_argument("--store", default="/root/autodl-tmp/qmt_daily_data/fast_factor_compute")
    ap.add_argument("--start", default="20180102")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--label", default="open_vwap15")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--cost-per-side", type=float, default=0.0015)
    ap.add_argument("--turnover-cap", type=float, default=0.30)
    ap.add_argument("--dedup", type=float, default=0.9)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--pool-top", type=int, default=120)
    ap.add_argument("--smooth-halflives", default="1,2,3,5,10,20")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--roll-window", type=int, default=250)   # 滚动方法回看多少天
    ap.add_argument("--roll-step", type=int, default=20)      # 多少天重拟合一次
    a = ap.parse_args()

    from alphacube.combine.post_combine import compare
    from alphacube.label import Label
    from alphacube.metrics import apply_turnover_cap, ema_rows, long_short_weights, portfolio_returns, zscore_rows
    from alphacube.mining.pool import MsePool
    from alphacube.mining.rankban import CompactRankIndex
    from alphacube.parser import parse
    from alphacube.research import warmed_view
    from alphacube.store import open_cube_store

    d = Path(a.dir)
    t0 = time.time()
    metrics = list(csv.DictReader(open(d / "metrics.csv", encoding="utf-8")))
    num = lambda r, k: float(r[k]) if r[k] not in ("", "nan", "None") else float("nan")   # noqa: E731
    healthy = [r for r in metrics if r["healthy"] == "True" and np.isfinite(num(r, "net_sharpe_train"))]
    healthy.sort(key=lambda r: -num(r, "net_sharpe_train"))
    rows = np.load(d / "rows.npy", mmap_mode="r")
    buf = np.empty((3, 256, rows.shape[2]), dtype=np.float32)      # kept factors' rank rows, grown in place
    kept = []
    for r in healthy:
        x = np.asarray(rows[int(r["i"])])[:, None, :]
        if kept:
            c = CompactRankIndex._corr(x, buf[:, :len(kept)])[0]
            if np.isfinite(c).any() and np.nanmax(c) >= a.dedup:
                continue
        if len(kept) == buf.shape[1]:
            buf = np.concatenate([buf, np.empty_like(buf)], axis=1)
        buf[:, len(kept)] = x[:, 0, :]
        kept.append(r)
    print(f"{len(metrics)} scored, {len(healthy)} healthy, {len(kept)} distinct at |rank corr| < {a.dedup} "
          f"({time.time() - t0:.0f}s)", flush=True)

    view, _ = warmed_view(open_cube_store(a.store), a.start or None, a.end or None, warmup=a.warmup)
    dates = [str(x) for x in view.dates]
    target = Label(a.label, 1).compute(view).astype(np.float64)
    D = len(dates)
    split = max(1, min(D - 1, int(D * a.train_frac)))
    train, test = slice(0, split), slice(split, D)
    years = np.asarray([x[:4] for x in dates])
    halflives = tuple(float(x) for x in a.smooth_halflives.split(",") if x)
    methods = tuple(x for x in a.methods.split(",") if x)
    cost, cap = a.cost_per_side, a.turnover_cap
    load = lambda r: np.load(d / "panels" / f"{int(r['i']):05d}.npy").astype(np.float64)   # noqa: E731

    def curve(v):
        f = v[np.isfinite(v)]
        if f.size < 2 or f.std() == 0:
            return {"sharpe": None, "ann_return": None, "max_dd": None}
        eq = np.cumsum(f)
        return {"sharpe": float(f.mean() / f.std() * np.sqrt(252)), "ann_return": float(f.mean() * 252),
                "max_dd": float((eq - np.maximum.accumulate(eq)).min())}

    def book(alpha, days):
        w = long_short_weights(alpha[days], 0.1)
        net, gross, turn = portfolio_returns(w, target[days], cost)
        c_net, _, _ = portfolio_returns(apply_turnover_cap(w, cap), target[days], cost)
        return net, gross, turn, c_net

    def by_year(v, ys):
        out = {}
        for y in sorted(set(ys)):
            f = v[ys == y][np.isfinite(v[ys == y])]
            out[y] = float(f.sum()) if f.size else None
        return out

    def run_combos(label, members, columns):
        t1 = time.time()
        results, signs = compare(columns, target, target, methods=methods, train_frac=a.train_frac, align=True,
                                 seed=0, cost_per_side=cost, turnover_cap=cap, smooth_halflives=halflives,
                                 roll_window=a.roll_window, roll_step=a.roll_step)
        combos = []
        for res in results:
            if "error" in res.metrics:
                combos.append({"method": res.method, "error": res.metrics["error"]})
                continue
            net, gross, turn, c_net = book(res.combined, test)
            item = {"method": res.method,
                    "test": {"net": curve(net), "gross": curve(gross), "capped": curve(c_net),
                             "turnover": float(np.mean(turn))},
                    "train": {"net": curve(book(res.combined, train)[0])},
                    "rank_ic_test": res.metrics.get("rank_ic"),
                    "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                for k, v in res.metrics.items()},
                    "by_year_net": by_year(net, years[test]), "equity_net": [float(x) for x in np.nancumsum(net)]}
            hl = res.metrics.get("halflife")
            if hl is not None:
                sm_alpha = ema_rows(res.combined, float(hl))
                s_net, _, s_turn, _ = book(sm_alpha, test)
                from alphacube.metrics import cs_pearson, nanmean_quiet, rank_rows
                item["smooth"] = {"halflife": float(hl), "net": curve(s_net), "turnover": float(np.mean(s_turn)),
                                  "ic": nanmean_quiet(cs_pearson(sm_alpha[test], target[test])),
                                  "rank_ic": nanmean_quiet(cs_pearson(rank_rows(sm_alpha[test]),
                                                                      rank_rows(target[test]))),
                                  "by_year_net": by_year(s_net, years[test]),
                                  "equity_net": [float(x) for x in np.nancumsum(s_net)]}
            combos.append(item)
            sm = item.get("smooth", {})
            print(f"  [{label}] {res.method:>8}: test net Sharpe {item['test']['net']['sharpe'] or float('nan'):+.2f} "
                  f"turnover {item['test']['turnover']:.1%}"
                  + (f"; smoothed hl {sm['halflife']:.0f}: {sm['net']['sharpe'] or float('nan'):+.2f}" if sm else ""),
                  flush=True)
        print(f"  [{label}] {len(members)} members, {time.time() - t1:.0f}s", flush=True)
        return {"label": label, "n": len(members),
                "members": [{"i": int(r["i"]), "run": r["run"], "expression": r["expression"],
                             "net_sharpe_train": num(r, "net_sharpe_train"), "net_sharpe_test": num(r, "net_sharpe_test"),
                             "turnover": num(r, "turnover"), "sign": float(s)} for r, s in zip(members, signs)],
                "combinations": combos}

    selections = []
    top = kept[: a.k]
    selections.append(run_combos(f"top{len(top)}", top, [zscore_rows(load(r)) for r in top]))

    # the runs' pool, but fitted to the training days only
    tgt_train = target.copy()
    tgt_train[split:] = np.nan
    pool = MsePool(30, tgt_train, dedup_threshold=0.95, min_coverage=0.5, min_scored_days=0.5)
    cands = kept[: a.pool_top]
    adm = pool.admit([(parse(r["expression"]), load(r)) for r in cands])
    chosen = {repr(e) for e in adm.added}
    by_expr = {repr(parse(r["expression"])): r for r in cands}
    members = [by_expr[repr(m.expr)] for m in pool.members]
    print(f"pool (training days only): {len(cands)} candidates -> {pool.size}", flush=True)
    selections.append(run_combos("pool30", members, [zscore_rows(load(r)) for r in members]))

    summary = [{k: (num(r, k) if k not in ("run", "expression", "healthy", "by_year_net") else r[k])
                for k in ("i", "run", "expression", "healthy", "coverage", "coverage_test", "scored", "rank_ic",
                          "rank_ic_train", "rank_ic_test", "turnover", "net_sharpe", "net_sharpe_train",
                          "net_sharpe_test", "gross_sharpe", "ann_net", "max_dd", "by_year_net")}
               for r in metrics]
    distinct = {int(r["i"]) for r in kept}
    for s in summary:
        s["distinct"] = int(s["i"]) in distinct
        s["by_year_net"] = json.loads(s["by_year_net"])
    out = {"view": {"start": dates[0], "end": dates[-1], "split": dates[split], "n_days": D,
                    "n_symbols": view.n_symbols},
           "counts": {"scored": len(metrics), "healthy": len(healthy), "distinct": len(kept), "dedup": a.dedup},
           "test_dates": dates[split:], "factors": summary, "selections": selections}

    def clean(x):
        if isinstance(x, float):
            return x if np.isfinite(x) else None
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        return x

    Path(a.out).write_text(json.dumps(clean(out), ensure_ascii=False), encoding="utf-8")
    print(f"wrote {a.out} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
