"""汇总 ``tradable_backtest.py sweep --stage1-only`` 的 JSON:训练段选出那一格的训练 / 测试超额夏普、
测试超额年化、费用,以及第 1 轮全部格子测试段超额夏普的中位数和范围(看"选中那格"是不是运气)。

  python scripts/sweep_table.py _bench/tradable/sweep_gp_hs300_*.json
  python scripts/sweep_table.py --pick _bench/tradable/sweep_roll_legacy_*.json   # 按训练段挑一个,标出来
"""
from __future__ import annotations

import json
import os
import sys
from typing import List, Optional, Sequence

import numpy as np


def summarize(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    d = json.load(open(path, encoding="utf-8"))
    cells = d["stage1"]
    best = max(cells, key=lambda r: r["train"]["ex_sharpe"])
    test = np.array([r["test"]["ex_sharpe"] for r in cells])
    return {"name": os.path.basename(path)[len("sweep_"):-len(".json")], "train": best["train"]["ex_sharpe"],
            "test": best["test"]["ex_sharpe"], "test_ann": best["test"]["ex_ann"], "fees": best["test"]["fees_yr"],
            "median": float(np.median(test)), "lo": float(test.min()), "hi": float(test.max()), "n": len(cells),
            "cell": f"{best['name']} hl{best['halflife']:g}"}


def pick(rows: List[dict]) -> Optional[dict]:
    """在一组信号里按训练段挑:训练段选出那一格的训练段超额夏普最高的那个(测试段不参与)。"""
    ok = [r for r in rows if r is not None and np.isfinite(r["train"])]
    return max(ok, key=lambda r: r["train"]) if ok else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    paths: List[str] = list(sys.argv[1:] if argv is None else argv)
    do_pick = "--pick" in paths
    paths = [p for p in paths if p != "--pick"]
    rows = [summarize(p) for p in paths]
    chosen = pick(rows) if do_pick else None
    print(f"{'信号':<36} {'训练':>6} {'测试超额夏普':>8} {'测试超额年化':>8} {'费用/年':>6} {'全部格测试中位':>8} {'范围':>13}")
    for p, r in zip(paths, rows):
        if r is None:
            print(f"{os.path.basename(p):<36} (没有)")
            continue
        mark = "  <- 按训练段挑出" if chosen is r else ""
        print(f"{r['name']:<36} {r['train']:+6.2f} {r['test']:+12.2f} {r['test_ann']:+12.1%} {r['fees']:9.1%} "
              f"{r['median']:+14.2f}   {r['lo']:+.2f}~{r['hi']:+.2f}   [{r['cell']}]{mark}")
    if do_pick and chosen is not None:
        tests = np.array([r["test"] for r in rows if r is not None])
        print(f"\n按训练段挑出:{chosen['name']}(训练 {chosen['train']:+.2f} -> 测试 {chosen['test']:+.2f});"
              f"这一组 {len(tests)} 个信号的测试段中位 {np.median(tests):+.2f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
