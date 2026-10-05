#!/usr/bin/env python
"""纯多头奖励(mining/scoring.LongScorer)的校准:分量的尺度定多少,奖励看不看得准。只用训练段。

从现有因子库(挖掘 + 手写)里随机抽一批面板,把训练段切成前后两半:
  前一半  算奖励的四个分量(原始面板,和先做 EMA 平滑再算两种),以及老适应度 |rank IC| - 0.14 x 换手;
  后一半  按前一半定的方向,看这个因子单独做纯多头的扣费超额夏普 —— 原始的、固定半衰期平滑的、
          半衰期在前一半上选的三种。
然后报:
  1. 各分量在因子库上的分布,"90 分位 - 中位数"就是 LongScorer.LONG_SCALES 该填的数;
  2. 每个分量、合成的适应度、老适应度,对后一半三种结果的 Spearman —— 奖励在样本外认不认得出好因子。
测试段(切分点之后)一天都不碰。

  cd /root/autodl-tmp/AlphaCube && PYTHONPATH=src python -u scripts/calibrate_long_reward.py \\
      --lib 挖掘=_bench/trainto/allfac_buyable --lib 手写=_bench/trainto/legacy \\
      --n 1200 --workers 20 --out _bench/long_reward/calibration.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

COMPS = ("sharpe", "ann", "ic", "rank_ic")
SMOOTH = 5.0               # "先平滑再算"用的半衰期,也是组合层在训练段选出来的那个
OLD_LAMBDA = 0.14
_G: dict = {}


def _init(target_path: str, split: int, cost: float) -> None:
    import torch
    from alphacube.mining.scoring import LongScorer, PanelScorer

    torch.set_num_threads(1)
    tgt = np.asarray(np.load(target_path, mmap_mode="r")[:split], dtype=np.float64)
    mid = split // 2
    kw = dict(cost_per_side=cost, min_coverage=0.5, min_scored_days=0.5)
    _G.update(mid=mid, split=split, cost=cost, first=LongScorer(tgt[:mid], **kw), second=LongScorer(tgt[mid:], **kw),
              old=PanelScorer(tgt[:mid], kind="rank_ic", turnover_penalty=OLD_LAMBDA), tgt=tgt)


def components(s: Dict[str, float]) -> Dict[str, float]:
    return {k: s.get(f"long_{k}", np.nan) for k in COMPS}


def _one(job) -> dict:
    from alphacube.metrics import ema_rows
    from pick_single_factors import train_books

    lib, d, i = job
    mid, split = _G["mid"], _G["split"]
    p = np.asarray(np.load(Path(d) / "panels" / f"{i:05d}.npy", mmap_mode="r")[:split], dtype=np.float64)
    p[~np.isfinite(p)] = np.nan
    out = {"lib": lib, "i": i}
    raw = _G["first"].score(p[:mid])
    out["gated_out"] = not np.isfinite(raw["fit"])
    out["distinct"], out["turnover"] = raw["distinct"], raw["turnover"]
    out["old_fit"] = _G["old"].score(p[:mid])["fit"]
    if out["gated_out"]:
        return out
    sign = raw["long_sign"]
    sm = ema_rows(sign * p, SMOOTH)                              # 因果:前一半的平滑值不含后一半的数
    smooth = _G["first"].score(sm[:mid], gated=False, orientations=(1.0,))
    out.update(sign=sign, long_turnover=raw["long_turnover"],
               raw=components(raw), smooth=components(smooth))
    # 后一半的结果:方向用前一半定的
    out["y_raw"] = _G["second"].score(sign * p[mid:], gated=False, orientations=(1.0,))["long_sharpe"]
    out["y_smooth"] = _G["second"].score(sm[mid:], gated=False, orientations=(1.0,))["long_sharpe"]
    hl = train_books(sign * p[:mid], _G["tgt"][:mid], _G["cost"])["hl_long"]
    best = ema_rows(sign * p, hl) if hl else sign * p
    out["hl"] = hl
    out["y_best"] = _G["second"].score(best[mid:], gated=False, orientations=(1.0,))["long_sharpe"]
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    m = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[m], b[m])[0]) if m.sum() >= 10 else np.nan


def suggest_scales(rows: Sequence[dict], key: str) -> Dict[str, float]:
    """每个分量在因子库上的"90 分位 - 中位数"。"""
    out = {}
    for c in COMPS:
        v = np.array([r[key][c] for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        out[c] = float(np.percentile(v, 90) - np.percentile(v, 50))
    return out


def composite(rows: Sequence[dict], key: str, scales: Dict[str, float],
              weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    w = weights or {c: 1.0 for c in COMPS}
    return np.array([sum(w[c] * r[key][c] / scales[c] for c in COMPS) / sum(w.values()) for r in rows])


def main(argv: Optional[Sequence[str]] = None) -> int:
    from multiprocessing import get_context

    from pick_single_factors import dedupe, load_library

    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--cost", type=float, default=0.0015)
    a = ap.parse_args(argv)

    t0 = time.time()
    rows: List[dict] = []
    target_path = ""
    for spec in a.lib:
        tag, _, d = spec.partition("=")
        rows += load_library(Path(d), tag)
        target_path = target_path or str(Path(d) / "_target.npy")
    rows = dedupe(rows)
    D = np.load(target_path, mmap_mode="r").shape[0]
    split = max(1, min(D - 1, int(D * a.train_frac)))
    rng = np.random.default_rng(a.seed)
    pick = sorted(rng.choice(len(rows), size=min(a.n, len(rows)), replace=False).tolist())
    jobs = [(rows[j]["lib"], rows[j]["dir"], int(rows[j]["i"])) for j in pick]
    print(f"{len(rows)} 个候选里抽 {len(jobs)} 个;训练段 {split} 天,前一半 {split // 2} 天算奖励,后一半看结果", flush=True)
    res = []
    with get_context("fork").Pool(a.workers, initializer=_init, initargs=(target_path, split, a.cost)) as pool:
        for n, r in enumerate(pool.imap_unordered(_one, jobs, chunksize=2), 1):
            res.append(r)
            if n % 100 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    ok = [r for r in res if not r["gated_out"]]
    print(f"过门槛 {len(ok)} 个,被门槛拒掉 {len(res) - len(ok)} 个", flush=True)

    out = {"n": len(res), "n_ok": len(ok), "split": split, "smooth_halflife": SMOOTH, "rows": res}
    for key in ("raw", "smooth"):
        print(f"\n=== 分量的分布({'原始面板' if key == 'raw' else f'先做半衰期 {SMOOTH:g} 的 EMA'})===", flush=True)
        for c in COMPS:
            v = np.array([r[key][c] for r in ok], dtype=float)
            q = np.nanpercentile(v, [10, 25, 50, 75, 90, 95])
            print(f"  {c:8s} 10/25/50/75/90/95 分位: " + "  ".join(f"{x:+.4f}" for x in q), flush=True)
        out[f"scales_{key}"] = sc = suggest_scales(ok, key)
        print("  90 分位 - 中位数: " + ", ".join(f"{c}={sc[c]:.4g}" for c in COMPS), flush=True)
    ys = {"后一半 原始": "y_raw", f"后一半 EMA{SMOOTH:g}": "y_smooth", "后一半 半衰期在前一半选": "y_best"}
    cand: Dict[str, np.ndarray] = {"老适应度 |rank IC|-0.14x换手": np.array([r["old_fit"] for r in ok])}
    for key in ("raw", "smooth"):
        for c in COMPS:
            cand[f"{key}.{c}"] = np.array([r[key][c] for r in ok], dtype=float)
        cand[f"{key}.合成(等权)"] = composite(ok, key, out[f"scales_{key}"])
    print("\n=== 前一半的奖励 对 后一半纯多头扣费夏普 的 Spearman ===", flush=True)
    print("  " + " " * 34 + "  ".join(f"{k:>22s}" for k in ys), flush=True)
    table = {}
    for name, v in cand.items():
        table[name] = {k: spearman(v, np.array([r[y] for r in ok], dtype=float)) for k, y in ys.items()}
        print(f"  {name:34s}" + "  ".join(f"{table[name][k]:>+22.3f}" for k in ys), flush=True)
    out["spearman"] = table
    # 奖励挑出来的前 10% 因子,后一半的中位数表现(比相关系数更接近"搜索会往哪走")
    print("\n=== 按前一半奖励排在前 10% 的因子,后一半三种结果的中位数 ===", flush=True)
    top = {}
    for name, v in cand.items():
        k = max(1, len(ok) // 10)
        idx = np.argsort(-np.nan_to_num(v, nan=-np.inf))[:k]
        top[name] = {lab: float(np.nanmedian([ok[j][y] for j in idx])) for lab, y in ys.items()}
        print(f"  {name:34s}" + "  ".join(f"{top[name][lab]:>+22.2f}" for lab in ys), flush=True)
    out["top_decile_median"] = top
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写出 {a.out} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
