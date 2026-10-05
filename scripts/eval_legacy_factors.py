"""原始手写因子库(legacy/factor_set_merged.py,1498 条)在生产视图上逐个求值(服务器上用)。

    cd /root/autodl-tmp/AlphaCube && PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
      /root/miniconda3/envs/quant/bin/python -u scripts/eval_legacy_factors.py _bench/legacy

输出和 eval_all_factors.py 同一个布局,score_all_factors.py / build_champion.py 不用改就能读:
``panels/<i>.npy``(float32,天 x 股票,生产视图 1942 x 5220)、``manifest.csv``(i, run, expression, name)、
``dates.npy``;另外 ``failures.csv`` 记下解析或求值失败的因子和原因。

eval_all_factors.py 不能直接吃这个库:它只读 mining_runs 的存档,用严格语法解析(手写因子要宽松语法 +
日频自动折叠 + 按名字内联引用),还会丢掉搜索不再繁殖的算子(eq / pow / in_sustain_stddev ...,手写因子里
有 400 多条用到)。这里:

1. ``load_legacy_factors()``(迁到 241 根分钟线)-> ``research.parse_many``(内联引用、宽松语法、日频折叠);
2. 只留根节点是"每只股票每天一个值"的,按结构去重;存 ``repr(e)``(已内联、折叠写明),不存原文;
3. 用到 ts_pearson / in_multireg 的 16 条排在最后一批(股票数平方的开销);
4. 分批求值(默认 50 条一批:手写因子的中间边界文件比挖出来的大,150 条一批最坏要 410 GB 临时盘);
   写完一个面板就从 page cache 里丢掉(见 eval_all_factors.drop_from_cache)。

可续跑:盘上已有的面板跳过。``--parity`` 先在一个小视图上 CPU / CUDA 各算一遍对数,不写面板。
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def legacy_items(n_minutes: int = 241) -> Tuple[List[Tuple[str, str, object]], List[Tuple[str, str]]]:
    """(name, repr, Expression) 列表(重算子排最后)和 (name, 原因) 的失败列表。"""
    from alphacube.catalog import RESOURCE_RISK_KERNELS
    from alphacube.legacy import load_legacy_factors
    from alphacube.research import parse_many

    parsed, errors = parse_many(load_legacy_factors(), n_minutes=n_minutes)
    failed = [(n, f"解析: {msg}") for n, msg in errors.items()]
    seen, light, heavy = set(), [], []
    for name, e in parsed.items():
        if not e.shape.degenerate:
            failed.append((name, "根节点不是每天一个值"))
            continue
        if e.key() in seen:
            failed.append((name, "和前面某条结构相同"))
            continue
        seen.add(e.key())
        risky = any(n.spec is not None and n.spec.name.removesuffix("_legacy") in RESOURCE_RISK_KERNELS
                    for n in e.nodes())          # 全库版本叫 in_multireg_legacy
        (heavy if risky else light).append((name, repr(e), e))
    return light + heavy, failed


def parity(items, view_args, n: int, batch: int) -> int:
    """小视图上 CPU / CUDA 各算一遍(分批:CPU 一次算一千多条,中间结果吃掉 38 GB 内存):
    报失败数差异和数值差异最大的几条。"""
    from alphacube.engine.engine import CubeEngine

    view = view_args()
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(items), size=min(n, len(items)), replace=False)) if n else range(len(items))
    exprs = [items[i][2] for i in pick]
    out = {}
    for dev in ("cpu", "cuda"):
        t0 = time.time()
        eng = CubeEngine(view, device=dev, workdir=Path(__file__).resolve().parents[1] / "tmp")
        out[dev] = [p for lo in range(0, len(exprs), batch)
                    for p in eng.evaluate(exprs[lo:lo + batch], tolerant=True).panels]
        print(f"{dev}: {sum(p is None for p in out[dev])} / {len(exprs)} 失败 ({time.time() - t0:.0f}s)", flush=True)
    bad = []
    for i, a, b in zip(pick, out["cpu"], out["cuda"]):
        if (a is None) != (b is None):
            bad.append((np.inf, items[i][0], "只在一边失败"))
            continue
        if a is None:
            continue
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        nan_diff = float((np.isnan(a) != np.isnan(b)).mean())
        both = np.isfinite(a) & np.isfinite(b)
        scale = np.nanstd(a[both]) if both.any() else 1.0
        err = float(np.max(np.abs(a[both] - b[both])) / (scale or 1.0)) if both.any() else 0.0
        if nan_diff > 1e-3 or err > 1e-3:
            bad.append((max(err, nan_diff), items[i][0], f"NaN 位置差 {nan_diff:.2%},最大差 / 标准差 {err:.2e}"))
    bad.sort(key=lambda r: -r[0])
    print(f"CPU / CUDA 不一致 {len(bad)} 条(阈值 1e-3)", flush=True)
    for _, name, why in bad[:30]:
        print(f"  {name}: {why}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--store", default="/root/autodl-tmp/qmt_daily_data/fast_factor_compute")
    ap.add_argument("--mirror-root", default="/root/autodl-tmp/AlphaCube/mirror")
    ap.add_argument("--start", default="20180102")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="只算前 N 条(测试用)")
    ap.add_argument("--require-all", action="store_true", help="有面板没算出来就以退出码 4 结束")
    ap.add_argument("--chunk-symbols", type=int, default=0,
                    help="引擎按股票分块(0 = 引擎自己定);全市场长窗口 ts_corr 一次要 17 GB,分块才装得下")
    ap.add_argument("--parity", type=int, default=-1,
                    help="不写面板:小视图(--parity-days 天 x --parity-symbols 只)上 CPU/CUDA 对数,抽 N 条(0 = 全部)")
    ap.add_argument("--parity-start", default="20220104")
    ap.add_argument("--parity-days", type=int, default=120)
    ap.add_argument("--parity-symbols", type=int, default=300)
    a = ap.parse_args(argv)

    frac = os.environ.get("ALPHACUBE_GPU_FRACTION")
    if frac and str(a.device).startswith("cuda"):
        # 和同卡的挖掘分着用显存(和 cli mine 一样);引擎在 tolerant 模式下 OOM 只会让节点失败,
        # 失败的面板不落盘,重跑一遍就补上
        import torch
        torch.cuda.set_per_process_memory_fraction(float(frac))
        print(f"显存封顶到整卡的 {float(frac):.0%}", flush=True)
    from eval_all_factors import drop_from_cache
    from alphacube.engine.engine import CubeEngine
    from alphacube.research import warmed_view
    from alphacube.store import open_cube_store

    store = open_cube_store(a.store, mirror_root=a.mirror_root or None)
    items, failed = legacy_items(241)
    print(f"{len(items)} 条可以求值;{len(failed)} 条解析失败 / 根节点不是日频 / 重复", flush=True)
    if a.parity >= 0:
        def small():
            v, _ = warmed_view(store, a.parity_start, None, warmup=0, max_symbols=a.parity_symbols)
            from alphacube.store import CubeView
            return CubeView(store, v.sym_idx, v.d0, min(v.d1, v.d0 + a.parity_days))
        return parity(items, small, a.parity, a.batch)
    if a.limit:
        items = items[: a.limit]

    out = Path(a.out_dir)
    (out / "panels").mkdir(parents=True, exist_ok=True)
    man = out / "manifest.csv"
    if man.exists():                                   # 续跑:序号必须和上次一致,不一致就停下来让人看
        old = [r["expression"] for r in csv.DictReader(open(man, encoding="utf-8"))]
        if old != [t for _, t, _ in items][: len(old)]:
            raise SystemExit(f"{man} 和这次的因子顺序对不上(因子库或解析器改过?),换个目录")
    with open(man, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["i", "run", "expression", "name"])
        for i, (name, text, _) in enumerate(items):
            w.writerow([i, "legacy", text, name])

    view, trim = warmed_view(store, a.start or None, a.end or None, warmup=a.warmup)
    np.save(out / "dates.npy", np.asarray(view.dates))
    print(f"视图 {view.n_days} 天 x {view.n_symbols} 只(预热 {trim})", flush=True)
    eng = CubeEngine(view, device=a.device, workdir=Path(__file__).resolve().parents[1] / "tmp",
                     chunk_symbols=a.chunk_symbols or None)
    todo = [i for i in range(len(items)) if not (out / "panels" / f"{i:05d}.npy").exists()]
    print(f"{len(items) - len(todo)} 个已在盘上,{len(todo)} 个要算", flush=True)
    t_all = time.time()
    for b0 in range(0, len(todo), a.batch):
        idx = todo[b0:b0 + a.batch]
        t0 = time.time()
        res = eng.evaluate([items[i][2] for i in idx], tolerant=True)
        errs = dict(res.errors) if isinstance(res.errors, dict) else {}
        for k, (i, p) in enumerate(zip(idx, res.panels)):
            if p is None:
                why = errs.get(k) or errs.get(items[i][2]) or "求值失败"
                failed.append((items[i][0], f"求值: {str(why)[:200]}"))
                continue
            tmp = out / "panels" / f"{i:05d}.npy.tmp"
            with open(tmp, "wb") as fh:
                np.save(fh, np.asarray(p, dtype=np.float32))
            final = out / "panels" / f"{i:05d}.npy"
            os.replace(tmp, final)
            drop_from_cache(final)
        done = b0 + len(idx)
        el = time.time() - t_all
        print(f"第 {b0 // a.batch + 1} 批:{len(idx)} 条 {time.time() - t0:.0f}s,"
              f"本批失败 {sum(p is None for p in res.panels)};{done}/{len(todo)},"
              f"还要约 {el / done * (len(todo) - done) / 60:.0f} 分钟", flush=True)
        with open(out / "failures.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["name", "reason"])
            w.writerows(failed)
    with open(out / "failures.csv", "w", newline="", encoding="utf-8") as fh:   # 续跑时没有要算的也要刷新
        w = csv.writer(fh)
        w.writerow(["name", "reason"])
        w.writerows(failed)
    n_ok = sum((out / "panels" / f"{i:05d}.npy").exists() for i in range(len(items)))
    print(f"完成:{n_ok} / {len(items)} 个面板;失败清单 {out / 'failures.csv'}", flush=True)
    return 0 if n_ok == len(items) or not a.require_all else 4

if __name__ == "__main__":
    sys.exit(main())
