#!/usr/bin/env python
"""第二版纯多头奖励(mining/scoring.LongHitScorer)的校准:几种命中率、几种做多组合,哪个看得准。只用训练段。

和 calibrate_long_reward.py 同一个做法:从现有因子库里随机抽一批面板,训练段切成前后两半 ——
  前一半  LongHitScorer 的全部分量(两种命中率分界都算),外加老适应度和第一版纯多头适应度当参照;
  后一半  方向用前一半定的,看这个因子做多头的结果:头部 5% / 10% 等权和按分位加权全做多三种组合的扣费超额夏普,
          原样和先做半衰期 5 的 EMA 两种;再加头部 10% 的命中率本身。
输出每个因子一行的 json;汇总(分量的尺度、对后一半结果的 Spearman、按它排前 10% 的因子后一半中位数)
也在这里打,合成方案的比较在拿到 json 以后做。测试段一天都不碰。

  cd /root/autodl-tmp/AlphaCube && PYTHONPATH=src python -u scripts/calibrate_long_hit.py \\
      --lib 挖掘=_bench/trainto/allfac_buyable --lib 手写=_bench/trainto/legacy \\
      --n 900 --workers 16 --out _bench/long_reward/calibration_hit.json
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

SMOOTH = 5.0
OLD_LAMBDA = 0.14
OUTCOMES = ("xs_top10", "xs_top5", "xs_rankw", "sh_top10", "hit_top10_z")
_G: dict = {}


def _scorer(tgt: np.ndarray, cost: float):
    from alphacube.mining.scoring import LongHitScorer

    w = {k: 1.0 for k in LongHitScorer.COMPONENTS}
    return LongHitScorer(tgt, cost_per_side=cost, min_coverage=0.5, min_scored_days=0.5, weights=w,
                         scales=dict(w), boundaries=("z", "m"))


def _init(target_path: str, split: int, cost: float) -> None:
    import torch
    from alphacube.mining.scoring import LongScorer, PanelScorer

    torch.set_num_threads(1)
    tgt = np.asarray(np.load(target_path, mmap_mode="r")[:split], dtype=np.float64)
    mid = split // 2
    _G.update(mid=mid, split=split, first=_scorer(tgt[:mid], cost), second=_scorer(tgt[mid:], cost),
              v1=LongScorer(tgt[:mid], cost_per_side=cost, min_coverage=0.5),
              old=PanelScorer(tgt[:mid], kind="rank_ic", turnover_penalty=OLD_LAMBDA))


def evaluate(panel_train: np.ndarray, first, second, mid: int) -> dict:
    """前一半的分量、后一半(方向固定)的结果。panel_train 只含训练段。"""
    from alphacube.metrics import ema_rows
    from alphacube.mining.scoring import LongHitScorer

    p = np.asarray(panel_train, dtype=np.float64)
    p[~np.isfinite(p)] = np.nan
    raw = first.score(p[:mid])
    out = {"gated_out": "xs_top10" not in raw, "distinct": raw["distinct"]}
    if out["gated_out"]:
        return out
    sign = raw["long_sign"]
    out["sign"] = sign
    out["comps"] = {k: raw[k] for k in LongHitScorer.COMPONENTS}
    out["turnover"] = {k: raw[f"to_{k}"] for k in LongHitScorer.PORTS}
    one = dict(gated=False, orientations=(1.0,))
    y = second.score(sign * p[mid:], **one)
    sm = ema_rows(sign * p, SMOOTH)                             # 因果:后一半的平滑值只多用了前一半的历史
    ys = second.score(sm[mid:], **one)
    out["y"] = {k: y.get(k, float("nan")) for k in OUTCOMES}
    out["y_smooth"] = {k: ys.get(k, float("nan")) for k in OUTCOMES}
    return out


def _one(job) -> dict:
    lib, d, i = job
    p = np.load(Path(d) / "panels" / f"{i:05d}.npy", mmap_mode="r")[: _G["split"]]
    r = {"lib": lib, "i": i, **evaluate(p, _G["first"], _G["second"], _G["mid"])}
    q = np.asarray(p[: _G["mid"]], dtype=np.float64)
    q[~np.isfinite(q)] = np.nan
    r["old_fit"] = _G["old"].score(q)["fit"]
    r["v1_fit"] = _G["v1"].score(q)["fit"]
    return r


def spearman(a, b) -> float:
    from scipy.stats import spearmanr

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[m], b[m])[0]) if m.sum() >= 10 else float("nan")


def top_median(score, y, frac: float = 0.1) -> float:
    score, y = np.asarray(score, dtype=float), np.asarray(y, dtype=float)
    k = max(1, int(len(score) * frac))
    return float(np.nanmedian(y[np.argsort(-np.nan_to_num(score, nan=-np.inf))[:k]]))


def summarize(ok: Sequence[dict]) -> Dict[str, dict]:
    """每个分量:中位数、90 分位 - 中位数(尺度),对后一半几种结果的 Spearman 和前 10% 的中位数。"""
    ys = {"原样 top10": [r["y"]["xs_top10"] for r in ok], "EMA5 top10": [r["y_smooth"]["xs_top10"] for r in ok],
          "EMA5 top5": [r["y_smooth"]["xs_top5"] for r in ok], "EMA5 rankw": [r["y_smooth"]["xs_rankw"] for r in ok]}
    cand = {k: [r["comps"][k] for r in ok] for k in ok[0]["comps"]}
    cand["老适应度"] = [r["old_fit"] for r in ok]
    cand["第一版纯多头"] = [r["v1_fit"] for r in ok]
    out = {}
    for name, v in cand.items():
        a = np.asarray(v, dtype=float)
        a = a[np.isfinite(a)]
        out[name] = {"median": float(np.median(a)), "scale": float(np.percentile(a, 90) - np.percentile(a, 50)),
                     "spearman": {k: spearman(v, y) for k, y in ys.items()},
                     "top10": {k: top_median(v, y) for k, y in ys.items()}}
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    from multiprocessing import get_context

    from pick_single_factors import dedupe, load_library

    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=900)
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
    print(f"{len(rows)} 个候选里抽 {len(jobs)} 个;训练段 {split} 天,前一半 {split // 2} 天算分量,后一半看结果", flush=True)
    res = []
    with get_context("fork").Pool(a.workers, initializer=_init, initargs=(target_path, split, a.cost)) as pool:
        for n, r in enumerate(pool.imap_unordered(_one, jobs, chunksize=2), 1):
            res.append(r)
            if n % 100 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    ok = [r for r in res if not r["gated_out"]]
    print(f"过门槛 {len(ok)} 个,被门槛拒掉 {len(res) - len(ok)} 个", flush=True)
    table = summarize(ok)
    heads = list(next(iter(table.values()))["spearman"])
    print("\n分量" + " " * 14 + "中位数     尺度   | Spearman: " + "  ".join(f"{h:>10s}" for h in heads)
          + " | 前10%中位数: " + "  ".join(f"{h:>10s}" for h in heads), flush=True)
    for name, t in table.items():
        print(f"{name:16s} {t['median']:+8.4f} {t['scale']:8.4f} | " + "  ".join(f"{t['spearman'][h]:>+10.3f}" for h in heads)
              + " | " + "  ".join(f"{t['top10'][h]:>+10.2f}" for h in heads), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"n": len(res), "n_ok": len(ok), "split": split, "rows": res,
                                       "summary": table}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写出 {a.out} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
