"""用已经算好的面板离线预热 exemplar 秩指纹缓存(不碰 GPU)。

`_bench/allfac/panels/` 里有全部 4167 个因子的面板 —— ban-archive 里的表达式
都在其中。与其让下一轮开跑时在 GPU 上把它们重算一遍(gpu17 花了 2 小时 04 分),
不如直接从盘上读面板算指纹:一条约 0.3 秒 CPU,1700 条 8 分钟,而且只做一次。

    python scripts/warm_exemplar_cache.py scripts/gp_full_market.sh _bench/allfac tmp/exemplar_cache
"""
from __future__ import annotations

import csv
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("launcher", help="从这个启动脚本里取 --ban / --ban-archive")
    ap.add_argument("allfac", help="全因子目录(manifest.csv + panels/)")
    ap.add_argument("cache", help="缓存目录")
    ap.add_argument("--store", default="/root/autodl-tmp/qmt_daily_data/fast_factor_compute")
    ap.add_argument("--start", default="20180102")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--sample-days", type=int, default=250)
    ap.add_argument("--sample-symbols", type=int, default=1024)
    a = ap.parse_args()

    from alphacube.mining.rankban import CompactRankIndex, ExemplarCache, view_cache_key
    from alphacube.pipeline import ban_exemplars_from
    from alphacube.research import warmed_view
    from alphacube.store import open_cube_store

    text = Path(a.launcher).read_text(encoding="utf-8")
    bans = re.findall(r'--ban "([^"]+)"', text)
    archives = re.findall(r"--ban-archive (\S+)", text)
    exprs = ban_exemplars_from(archives, bans)
    print(f"{len(bans)} 个封禁键 x {len(archives)} 个存档 -> {len(exprs)} 条表达式", flush=True)

    view, _ = warmed_view(open_cube_store(a.store), a.start or None, a.end or None, warmup=a.warmup)
    key = view_cache_key(view, a.sample_days, a.sample_symbols)
    cache = ExemplarCache(a.cache, key)
    index = CompactRankIndex((view.n_days, view.n_symbols), a.sample_days, a.sample_symbols)
    print(f"视图键 {key};{view.n_days} 天 x {view.n_symbols} 只;指纹 {3 * index.width * 4 / 2**20:.2f} MB/条",
          flush=True)

    d = Path(a.allfac)
    where = {r["expression"]: int(r["i"]) for r in csv.DictReader(open(d / "manifest.csv", encoding="utf-8"))}
    hit = miss = wrote = 0
    t0 = time.time()
    for n, e in enumerate(exprs, 1):
        text_e = repr(e)
        if cache.get(text_e) is not None:
            hit += 1
            continue
        i = where.get(text_e)
        path = (d / "panels" / f"{i:05d}.npy") if i is not None else None
        if path is None or not path.exists():
            miss += 1       # 没算过或当初评估失败(4169 个里有 2 个),留给挖掘器在 GPU 上补
            continue
        panel = np.load(path).astype(np.float64)
        cache.put(text_e, index.rows(panel))
        wrote += 1
        if n % 200 == 0:
            print(f"  {n}/{len(exprs)},写入 {wrote},已有 {hit},没面板 {miss},{time.time() - t0:.0f}s", flush=True)
    print(f"完成:写入 {wrote},已有 {hit},没面板 {miss}({time.time() - t0:.0f}s)", flush=True)
    print(cache.summary(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
