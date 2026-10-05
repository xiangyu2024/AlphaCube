#!/usr/bin/env python
"""一批因子"单独拿来做纯多头"的训练段 / 测试段成绩单,用来回答:新奖励挖出来的因子,样本外对多头有没有用。

每个因子:
  训练段  第一版纯多头打分器(mining/scoring.LongScorer,带门槛)算适应度、方向和四个分量;
          顺带算老适应度 |rank IC| - 0.14 x 换手、和第二版(LongHitScorer:命中率 + 几种做多组合)的适应度,
          三种排法对着看。
  测试段  方向固定用训练段定的,不再挑:原始面板、和先做半衰期 5 的 EMA(组合层在训练段选的那个)各算一遍
          头部 10% 纯多头书的扣费超额夏普 / 年化 / 换手,以及多头端 IC / rank IC;平滑后的再加头部 20%、
          按分位加权全做多两种组合的超额夏普和头部 10% 的命中率。
然后按库汇总:训练段适应度排前 N 的因子,测试段成绩的中位数、为正的比例。
挖掘只到训练段结束(--end 20230807)的 run,测试段是干净的样本外;全样本挖出来的老库不是,只能当参照。

  cd /root/autodl-tmp/AlphaCube && PYTHONPATH=src python -u scripts/long_report.py \\
      --dir 多头奖励=_bench/allfac_long --dir 手写库=_bench/trainto/legacy --sample 手写库=400 \\
      --target _bench/trainto/allfac_buyable/_target.npy --workers 8 --out _bench/long_reward/report.json
"""
from __future__ import annotations

import argparse
import csv
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
TEST_KEYS = ("long_sharpe", "long_ann", "long_turnover", "long_ic", "long_rank_ic")
HIT_KEYS = ("xs_top20", "xs_rankw", "hit_top10_z", "hit_exp_z")
RANKINGS = (("fit_train", "按第一版纯多头适应度(训练段)排"), ("hit_fit_train", "按第二版(命中率+组合)适应度(训练段)排"),
            ("old_fit_train", "按老适应度(训练段)排"))
TOPS = (20, 50, 100)
_G: dict = {}


def _init(target_path: str, split: int, cost: float, min_coverage: float) -> None:
    import torch
    from alphacube.mining.scoring import LongHitScorer, LongScorer

    torch.set_num_threads(1)
    tgt = np.asarray(np.load(target_path, mmap_mode="r"), dtype=np.float64)
    hit_w = {k: 1.0 for k in HIT_KEYS}                    # 测试段只要这几个分量,合成值不用
    _G.update(split=split, train=LongScorer(tgt[:split], cost_per_side=cost, min_coverage=min_coverage),
              test=LongScorer(tgt[split:], cost_per_side=cost, min_coverage=0.0),
              hit_train=LongHitScorer(tgt[:split], cost_per_side=cost, min_coverage=min_coverage),
              hit_test=LongHitScorer(tgt[split:], cost_per_side=cost, min_coverage=0.0, weights=hit_w))


def evaluate(panel: np.ndarray, train, test, split: int, hit_train=None, hit_test=None) -> dict:
    """一个面板 -> 训练段(带门槛、挑方向)和测试段(方向固定)的纯多头成绩。给了第二版的打分器就再加:
    训练段第二版的适应度,和平滑后测试段的另外两种组合、命中率。"""
    from alphacube.metrics import ema_rows

    p = np.asarray(panel, dtype=np.float64)
    p[~np.isfinite(p)] = np.nan
    tr = train.score(p[:split])
    out = {"fit_train": tr["fit"], "hit_fit_train": hit_train.score(p[:split])["fit"] if hit_train is not None
           else float("nan"), "old_fit_train": abs(tr["rank_ic"]) - OLD_LAMBDA * tr["turnover"]
           if np.isfinite(tr.get("rank_ic", np.nan)) else float("nan"),
           "rank_ic_train": tr.get("rank_ic", float("nan")), "ls_turnover_train": tr.get("turnover", float("nan")),
           "distinct": tr.get("distinct", float("nan")), "gated_out": not np.isfinite(tr["fit"])}
    if out["gated_out"]:
        return out
    sign = tr["long_sign"]
    out["sign"] = sign
    out["train"] = {k: tr[k] for k in TEST_KEYS + ("long_tie", "long_code")}
    one = dict(gated=False, orientations=(1.0,))
    out["test"] = {k: v for k, v in test.score(sign * p[split:], **one).items() if k in TEST_KEYS + ("fit",)}
    sm = ema_rows(sign * p, SMOOTH)                            # 因果:训练段的平滑值不含测试段的数
    out["train_smooth"] = {k: v for k, v in train.score(sm[:split], **one).items() if k in TEST_KEYS + ("fit",)}
    out["test_smooth"] = {k: v for k, v in test.score(sm[split:], **one).items() if k in TEST_KEYS + ("fit",)}
    if hit_test is not None:
        ht = hit_test.score(sm[split:], **one)
        out["test_smooth"].update({k: ht.get(k, float("nan")) for k in HIT_KEYS})
    return out


def _one(job) -> dict:
    lib, d, i, run, expr = job
    panel = np.load(Path(d) / "panels" / f"{i:05d}.npy")
    return {"lib": lib, "i": i, "run": run, "expression": expr,
            **evaluate(panel, _G["train"], _G["test"], _G["split"], _G["hit_train"], _G["hit_test"])}


def summarize(rows: Sequence[dict], rank_key: str = "fit_train", tops: Sequence[int] = TOPS) -> Dict[str, dict]:
    """按 rank_key(训练段)从高到低,前 N 个因子的测试段成绩。"""
    ok = [r for r in rows if not r["gated_out"] and np.isfinite(r.get(rank_key, np.nan))]
    ok.sort(key=lambda r: -r[rank_key])
    out = {}
    for n in list(tops) + [len(ok)]:
        top = ok[:n]
        if not top:
            continue

        def med(part, key):
            return float(np.nanmedian([r[part][key] for r in top]))

        raw = np.array([r["test"]["long_sharpe"] for r in top], dtype=float)
        sm = np.array([r["test_smooth"]["long_sharpe"] for r in top], dtype=float)
        out["全部" if n >= len(ok) else f"前{n}"] = {
            "n": len(top), "train_sharpe": med("train", "long_sharpe"), "test_sharpe": float(np.nanmedian(raw)),
            "test_ann": med("test", "long_ann"), "test_pos": float(np.mean(raw > 0)),
            "train_sharpe_smooth": med("train_smooth", "long_sharpe"), "test_sharpe_smooth": float(np.nanmedian(sm)),
            "test_ann_smooth": med("test_smooth", "long_ann"), "test_pos_smooth": float(np.mean(sm > 0)),
            "test_rank_ic": med("test", "long_rank_ic"), "test_turnover": med("test", "long_turnover")}
        for k in HIT_KEYS:
            if all(k in r["test_smooth"] for r in top):
                out["全部" if n >= len(ok) else f"前{n}"][f"smooth_{k}"] = med("test_smooth", k)
    return out


def print_summary(name: str, table: Dict[str, dict]) -> None:
    print(f"  {name}", flush=True)
    for label, s in table.items():
        print(f"    {label:>5s}(n={s['n']:4d}) 原样:训练夏普 {s['train_sharpe']:+.2f} -> 测试 {s['test_sharpe']:+.2f} "
              f"(年化 {s['test_ann'] * 100:+.1f}%,为正 {s['test_pos']:.0%}) | EMA{SMOOTH:g}:训练 "
              f"{s['train_sharpe_smooth']:+.2f} -> 测试 {s['test_sharpe_smooth']:+.2f} (年化 "
              f"{s['test_ann_smooth'] * 100:+.1f}%,为正 {s['test_pos_smooth']:.0%}) | 测试多头端 rank IC "
              f"{s['test_rank_ic']:+.4f} 换手 {s['test_turnover']:.3f}"
              + (f" | EMA{SMOOTH:g} 测试:top20 夏普 {s['smooth_xs_top20']:+.2f} 分位加权 {s['smooth_xs_rankw']:+.2f} "
                 f"top10 命中 {s['smooth_hit_top10_z'] * 100:+.2f}%" if "smooth_xs_top20" in s else ""), flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from multiprocessing import get_context

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True, help="名字=目录(manifest.csv + panels/)")
    ap.add_argument("--sample", action="append", default=[], help="名字=N:这个库只随机抽 N 个(省时间)")
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--cost", type=float, default=0.0015)
    ap.add_argument("--min-coverage", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    t0 = time.time()
    D = np.load(a.target, mmap_mode="r").shape[0]
    split = max(1, min(D - 1, int(D * a.train_frac)))
    sample = {s.partition("=")[0]: int(s.partition("=")[2]) for s in a.sample}
    rng = np.random.default_rng(a.seed)
    jobs: List[tuple] = []
    for spec in a.dir:
        lib, _, d = spec.partition("=")
        man = [r for r in csv.DictReader(open(Path(d) / "manifest.csv", encoding="utf-8"))
               if (Path(d) / "panels" / f"{int(r['i']):05d}.npy").exists()]
        if lib in sample and sample[lib] < len(man):
            man = [man[j] for j in sorted(rng.choice(len(man), size=sample[lib], replace=False).tolist())]
        jobs += [(lib, d, int(r["i"]), r.get("run", ""), r["expression"]) for r in man]
        print(f"{lib}: {len(man)} 个面板", flush=True)
    print(f"训练段 {split} 天,测试段 {D - split} 天;共 {len(jobs)} 个因子", flush=True)
    rows = []
    with get_context("fork").Pool(a.workers, initializer=_init,
                                  initargs=(a.target, split, a.cost, a.min_coverage)) as pool:
        for n, r in enumerate(pool.imap_unordered(_one, jobs, chunksize=2), 1):
            rows.append(r)
            if n % 50 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    out = {"split": split, "smooth_halflife": SMOOTH, "rows": rows, "summary": {}}
    for spec in a.dir:
        lib = spec.partition("=")[0]
        part = [r for r in rows if r["lib"] == lib]
        gated = sum(r["gated_out"] for r in part)
        print(f"\n=== {lib}:{len(part)} 个,过门槛 {len(part) - gated} 个 ===", flush=True)
        out["summary"][lib] = {}
        for key, name in RANKINGS:
            out["summary"][lib][key] = summarize(part, key)
            print_summary(name, out["summary"][lib][key])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写出 {a.out} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
