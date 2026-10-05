"""股票池挖掘结果的干净样本外评估:训练段拟合,测试段只报数。

股票池挖掘(scripts/rl_universe.sh / gp_universe.sh)只看到切分点之前的数据。这里把一个 run 挖出来的表达式
(最终池子 members.csv,GP 还可以加上存档里适应度最高的若干条)放到全区间上重算:

1. 引擎只在全区间里进过股票池的票上算,`UniverseEngine` 把每个面板的池外格子置空;标签(剔除封板)也一样;
2. 单因子:训练段 / 测试段的池内 rank IC;
3. 合成:逐日 z 分数(池内),按训练段 IC 对齐符号,权重只在训练段拟合。equal / ridge / lstsq 三种都报、
   各存一份,不在测试段上挑:挖出来的池子常常是同一个想法的几十种写法(gp_hs300_1 的 30 条全是低波动),
   高度共线时 lstsq 的权重互相抵消,合成反而不如单个因子;
4. 报告用 build_champion.evaluate:十分位多空(原始、平滑后,半衰期训练段选)、rank IC、ICIR、纯多头前 20%·exp1;
5. 合成信号铺回 (天, 全市场股票) 的形状存成 .npy,交给 `tradable_backtest.py sweep --universe` 做真实回测。

  cd /root/autodl-tmp/AlphaCube && PYTHONPATH=src python -u scripts/eval_universe_run.py \\
      --run mining_runs/gp_hs300_1 --run mining_runs/rl_hs300_2 --universe _bench/universe/hs300_proxy_v1.npy \\
      --out-dir _bench/universe
"""
from __future__ import annotations

import argparse
import csv
import glob
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def run_expressions(run: Path, archive_top: int = 0) -> List[str]:
    """最终池子的成员;archive_top > 0 时再加上 GP 最后一个存档里适应度最高的若干条(去重,保持顺序)。"""
    exprs = [r["expression"] for r in csv.DictReader(open(run / "members.csv", encoding="utf-8"))]
    if archive_top:
        arcs = sorted(glob.glob(str(run / "checkpoints" / "archive_gen*.csv")))
        if arcs:
            rows = list(csv.DictReader(open(arcs[-1], encoding="utf-8")))
            key = "fitness" if rows and "fitness" in rows[0] else None
            if key:
                rows.sort(key=lambda r: -float(r[key]) if r[key] not in ("", "nan", "-inf") else np.inf)
            exprs += [r["expression"] for r in rows[:archive_top]]
    seen, out = set(), []
    for e in exprs:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def seed_keys(run: Path, extra: Sequence[str] = (), n_minutes: int = 241) -> set:
    """种子的结构键:GP 的 checkpoints/seeds_used.csv(带种子的 GP 自己写)加上 --mark-seeds 给的文件
    (RL 池子种子的 yaml,或带 expression 列的 csv)。用来区分池子里"挖出来的"和"原样留下的种子"。"""
    from alphacube.parser import parse

    texts = []
    f = run / "checkpoints" / "seeds_used.csv"
    if f.exists():
        texts += [r["expression"] for r in csv.DictReader(open(f, encoding="utf-8"))]
    for path in extra:
        if str(path).endswith((".yaml", ".yml")):
            import yaml
            texts += [e["expression"] for e in (yaml.safe_load(open(path, encoding="utf-8")) or {}).get("seed_factors", [])]
        else:
            texts += [r["expression"] for r in csv.DictReader(open(path, encoding="utf-8"))]
    keys = set()
    for t in texts:
        try:
            keys.add(parse(t, strict=False, coerce_daily=True, n_minutes=n_minutes).key())
        except Exception:                                     # noqa: BLE001
            continue
    return keys


def combine_train(panels: List[np.ndarray], target: np.ndarray, split: int, method: str = "lstsq") -> np.ndarray:
    """逐日 z 分数 -> 按训练段 IC 对齐符号 -> 训练段拟合的权重(equal / ridge / lstsq) -> 合成。"""
    from combine_two_level import blend, fit_weights
    from alphacube.metrics import cs_pearson, nanmean_quiet, zscore_rows

    train = slice(0, split)
    cols = []
    for p in panels:
        z = zscore_rows(p.astype(np.float64))
        if nanmean_quiet(cs_pearson(z[train], target[train])) < 0:
            z = -z
        cols.append(z)
    return blend(cols, fit_weights(cols, target, train, method))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="mining_runs/<run-id>,可以给多个")
    ap.add_argument("--universe", required=True)
    ap.add_argument("--store", default="/root/autodl-tmp/qmt_daily_data/fast_factor_compute")
    ap.add_argument("--mirror-root", default="/root/autodl-tmp/AlphaCube/mirror")
    ap.add_argument("--start", default="20180102")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--archive-top", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mark-seeds", action="append", default=[],
                    help="池子种子文件(yaml / csv),池子里原样留下的种子在表里标 *,另报一份去掉种子的合成")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)

    from build_champion import evaluate
    from alphacube.engine.engine import CubeEngine
    from alphacube.label import Label
    from alphacube.metrics import cs_pearson, nanmean_quiet, rank_rows
    from alphacube.parser import parse
    from alphacube.pipeline import UniverseEngine, align_universe, load_universe_file
    from alphacube.research import warmed_view
    from alphacube.store import open_cube_store

    t0 = time.time()
    store = open_cube_store(a.store, mirror_root=a.mirror_root)
    u_mask, u_dates, u_syms = load_universe_file(a.universe)
    rows = [i for i, d in enumerate(u_dates) if a.start <= d <= a.end]
    symbols = [s for s, keep in zip(u_syms, u_mask[rows].any(0)) if keep]
    view, trim = warmed_view(store, a.start, a.end, warmup=0, symbols=symbols)
    member = align_universe(u_mask, u_dates, u_syms, view.dates, view.symbols)
    target = np.where(member, Label("open_vwap15", 1, buyable=True).compute(view), np.nan)
    D = view.n_days
    split = max(1, min(D - 1, int(D * a.train_frac)))
    print(f"{view.dates[0]}..{view.dates[-1]} {D} 天 x {view.n_symbols} 只(全区间进过池子的);"
          f"切分点 {view.dates[split]};每天平均 {member.sum(1).mean():.0f} 只 ({time.time() - t0:.0f}s)", flush=True)
    engine = UniverseEngine(CubeEngine(view, device=a.device), member)
    full_idx = {s: i for i, s in enumerate(store.symbols)}
    cols_full = np.array([full_idx[s] for s in view.symbols])
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for run in map(Path, a.run):
        texts = run_expressions(run, a.archive_top)
        exprs, keep = [], []
        for t in texts:
            try:
                exprs.append(parse(t, n_minutes=view.n_minutes))
                keep.append(t)
            except Exception as exc:                          # noqa: BLE001 - 报出来,不中断
                print(f"  解析失败,跳过:{t[:80]} ({exc})")
        res = engine.evaluate(exprs, tolerant=True)
        seeds = seed_keys(run, a.mark_seeds, view.n_minutes)
        is_seed = [e.key() in seeds for e, p in zip(exprs, res.panels) if p is not None]
        panels = [(t, p) for t, p in zip(keep, res.panels) if p is not None]
        print(f"\n=== {run.name}:{len(texts)} 条表达式,算出 {len(panels)} 个面板"
              + (f",其中 {sum(is_seed)} 个是原样留下的种子(标 *)" if seeds else "")
              + f" ({time.time() - t0:.0f}s)", flush=True)
        tr, te = slice(0, split), slice(split, D)
        print(f"{'表达式':<70} {'训练 rankIC':>10} {'测试 rankIC':>10}")
        for (t, p), sd in list(zip(panels, is_seed))[:40]:
            ric = cs_pearson(rank_rows(p.astype(np.float64)), rank_rows(target))
            print(f"{('*' if sd else ' ') + t[:69]:<70} {nanmean_quiet(ric[tr]):+10.4f} {nanmean_quiet(ric[te]):+10.4f}")
        variants = [("", [p for _, p in panels])]
        if seeds and 0 < sum(is_seed) < len(panels):
            variants.append(("_noseed", [p for (_, p), sd in zip(panels, is_seed) if not sd]))
        for method, (suffix, cols) in [(m, v) for v in variants for m in ("equal", "ridge", "lstsq")]:
            method_label = method + suffix
            sig = combine_train(cols, target, split, method)
            sig[~member] = np.nan
            r = evaluate(sig, target, split, target)
            print(f"合成 {method_label:>12}(权重训练段定):测试段 rank IC {r['rank_ic']:+.4f} ICIR {r['icir']:+.3f} | "
                  f"十分位多空 净夏普 {r['ls_net']:+.2f} 平滑后 {r['ls_sm_net']:+.2f}(hl{r['ls_hl']:g},"
                  f"换手 {r['ls_turn']:.1%}) | 纯多头前20%·exp1 超额夏普 {r['lo_ex_sharpe']:+.2f} "
                  f"年化 {r['lo_ex_ann']:+.1%}(hl{r['lo_hl']:g})", flush=True)
            full = np.full((D, len(store.symbols)), np.nan, dtype=np.float32)
            full[:, cols_full] = sig.astype(np.float32)
            out = out_dir / f"signal_{run.name}_{method_label}.npy"
            np.save(out, full)
        print(f"信号写到 {out_dir}/signal_{run.name}_<method>[_noseed].npy(和 tradable_backtest 的行情面板对齐)")
    print(f"({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
