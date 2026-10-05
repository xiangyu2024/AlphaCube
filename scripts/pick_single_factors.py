#!/usr/bin/env python
"""挑一批互相低相关、单独拿出来最好用的因子,连表达式一起给出。只在训练段选,测试段只用来汇报。

候选 = 几个打分目录(挖掘库、手写库)里训练段 rank IC 有限的因子,按表达式去重后放在一起。
不用 metrics.csv 的 healthy 列:它要求测试段覆盖率 >= 0.5,不是纯训练段的门槛;这里自己在训练段设门槛。

"好用"的口径是**单独拿这一个因子做纯多头**(--rank long,默认):方向取训练段 rank IC 的符号,
信号做因果 EMA 平滑,每天等权持有前 10%,减去当天有信号有标签的票的等权收益,每单位换手付 15bp;
半衰期在训练段按这本书的扣费超额夏普从 {0,1,2,3,5,10,20} 里选,排序也按它。
不按十分位多空夏普排(--rank ls 还留着):A 股空头腿做不了,而且"次日开盘封涨停"的票标签是 NaN,
放在空头里算 0 收益 —— 空头腿被抬高了,而这批因子的多空差价八成来自空头腿。
也不用选族的适应度 |rank IC| - 0.14 x 换手:按它挑出来的单因子里有只在一半股票上有值的、只有 4 档取值的。

训练段的门槛:
  覆盖率    训练段有标签的格子里,因子有值的比例 >= --min-coverage
  取值离散  训练段每天"不同取值数 / 有值的票数"的中位数 >= --min-distinct(档位太少的因子十分位书靠代码顺序破平局)
  夏普      训练段那本书的扣费夏普 >= --min-sharpe

过了门槛的从高到低贪心:和已收的每一个 |相关| < 阈值 才收。相关在**实际拿来用的信号**上算 ——
方向 x 面板、按各自的半衰期平滑之后,训练段最后 250 天 x 等距 512 只,逐日 z 分数化秩,缺的记 0。
(原始面板上低相关不够:两个日频因子各自平滑以后,共同的慢变量会露出来。)
挑完以后在**整张面板**上重算:训练段 / 测试段各自的两两截面秩相关(平滑后的信号)、多头书日超额之间的相关,
和每个因子的纯多头书 / 多空书。--verify 再把表达式重新求值一遍,确认给出去的字符串算出来就是盘上打过分的面板。

  cd /root/autodl-tmp/AlphaCube && PYTHONPATH=src python -u scripts/pick_single_factors.py \\
      --lib 挖掘=_bench/trainto/allfac_buyable --lib 手写=_bench/trainto/legacy \\
      --scores _bench/single_factors/train_books.csv --workers 20 \\
      --out _bench/single_factors/long_thr0.3.json --verify
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

THRESHOLD = 0.3            # 平滑后信号的训练段指纹 |相关| 到这个数就算"和已收的重复"
FP_DAYS, FP_SYMBOLS = 250, 512
MIN_COMMON = 30            # 两个因子当天共同有值的票少于这个数,那天不算相关
MIN_COVERAGE = 0.95
MIN_DISTINCT = 0.5
MIN_SHARPE = 1.0           # 训练段扣费夏普不到这个数的不算"好用"
DISTINCT_STEP = 5          # 每隔这么多天抽一天数不同取值
TOP_FRAC = 0.1
COST = 0.0015
SCORE_COLS = ("lib", "i", "sign", "coverage_train", "distinct_train", "hl_long", "long_sharpe_train",
              "long_ann_train", "long_turnover_train", "hl_ls", "ls_sharpe_train")
RANKS = {"long": ("hl_long", "long_sharpe_train"), "ls": ("hl_ls", "ls_sharpe_train")}
REEVAL_MASK_TOL = 1e-3     # 重新求值:NaN 位置不一致的格子占比上限(0/0 边界上的浮点噪声)
REEVAL_CORR_TOL = 0.999    # 重新求值:共同格逐日秩相关的均值下限


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load_library(d: Path, tag: str) -> List[dict]:
    """打分目录 -> 训练段 rank IC 有限的因子行,补上 lib / dir / name。"""
    names = {r["i"]: r.get("name", "") for r in csv.DictReader(open(d / "manifest.csv", encoding="utf-8"))}
    rows = [r for r in csv.DictReader(open(d / "metrics.csv", encoding="utf-8"))
            if np.isfinite(_num(r.get("rank_ic_train"))) and _num(r["rank_ic_train"]) != 0.0]
    for r in rows:
        r.update(lib=tag, dir=str(d), name=names.get(r["i"], ""))
    return rows


def dedupe(rows: Sequence[dict]) -> List[dict]:
    """同一条表达式在两个库里都有时只留先出现的。"""
    seen, out = set(), []
    for r in rows:
        if r["expression"] in seen:
            continue
        seen.add(r["expression"])
        out.append(r)
    return out


def unit(v: np.ndarray) -> Optional[np.ndarray]:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else None


def greedy(order: Sequence[int], vec: Callable[[int], Optional[np.ndarray]], threshold: float,
           limit: int) -> List[int]:
    """按 order 的顺序收:和已收的每一个 |相关| 都 < threshold 才收,收满 limit 个停。"""
    kept: List[int] = []
    M = None
    for j in order:
        if len(kept) >= limit:
            break
        v = vec(j)
        if v is None:
            continue
        if M is None:
            M = np.zeros((limit, v.size))
        if kept and np.abs(M[:len(kept)] @ v).max() >= threshold:
            continue
        M[len(kept)] = v
        kept.append(j)
    return kept


def distinct_fraction(panel: np.ndarray, step: int = DISTINCT_STEP) -> float:
    """每隔 step 天抽一天:不同取值数 / 有值的票数,取中位数。"""
    out = []
    for t in range(0, panel.shape[0], step):
        row = panel[t]
        row = row[np.isfinite(row)]
        if row.size >= MIN_COMMON:
            out.append(np.unique(row).size / row.size)
    return float(np.median(out)) if out else 0.0


def long_book(w: np.ndarray, sig: np.ndarray, target: np.ndarray, cost: float = COST):
    """多空权重 w(多 +0.5 / 空 -0.5)里的多头腿当成一本纯多头的书:权重和 1,每天减去当天有信号、有标签的票的
    等权收益;标签缺的持仓(次日开盘封涨停,买不进)当天记 0。换手 = sum|w_t - w_{t-1}|(第一天从空仓建仓),
    每单位换手付 cost,和 portfolio_returns 同一个价。返回 (扣费超额, 毛超额, 单边换手)。"""
    wl = np.clip(w, 0.0, None) * 2.0
    ok = np.isfinite(sig) & np.isfinite(target)
    r = np.where(ok, target, 0.0)
    n = ok.sum(1)
    ew = np.where(n > 0, r.sum(1) / np.maximum(n, 1), np.nan)
    gross = np.where(wl.sum(1) > 0, (wl * r).sum(1) - ew, np.nan)
    traded = np.abs(np.diff(wl, axis=0, prepend=np.zeros((1, wl.shape[1])))).sum(1)
    return gross - cost * traded, gross, 0.5 * traded


def train_books(sig: np.ndarray, target: np.ndarray, cost: float = COST) -> dict:
    """sig / target 都只含训练段。每个半衰期各建一本纯多头书和一本多空书,各自留扣费夏普最高的那个半衰期
    (并列取先出现的,和 ls_report.report 一样;EMA 是因果的,截到训练段再平滑结果不变)。"""
    from alphacube.metrics import ema_rows, long_short_weights, portfolio_returns
    from long_only_grid import stats
    from ls_report import HALFLIVES

    best: Dict[str, tuple] = {}
    for hl in HALFLIVES:
        sm = ema_rows(sig, hl) if hl else sig
        w = long_short_weights(sm, TOP_FRAC)
        ls = stats(portfolio_returns(w, target, cost)[0])["sharpe"]
        net, _, turn = long_book(w, sm, target, cost)
        lo = stats(net)
        for name, s, extra in (("ls", ls, ()), ("long", lo["sharpe"], (lo["ann"], float(np.nanmean(turn))))):
            key = np.nan_to_num(s, nan=-9e9)
            if name not in best or key > best[name][0]:
                best[name] = (key, hl, s) + tuple(extra)
    return {"hl_long": best["long"][1], "long_sharpe_train": best["long"][2], "long_ann_train": best["long"][3],
            "long_turnover_train": best["long"][4], "hl_ls": best["ls"][1], "ls_sharpe_train": best["ls"][2]}


_G: dict = {}


def _init(target_path: str, split: int, cost: float) -> None:
    _G["target"] = np.asarray(np.load(target_path, mmap_mode="r")[:split], dtype=np.float64)
    _G["split"], _G["cost"] = split, cost


def _score(job: Tuple[str, str, int, float]) -> dict:
    lib, d, i, sign = job
    tgt = _G["target"]
    p = np.asarray(np.load(Path(d) / "panels" / f"{i:05d}.npy", mmap_mode="r")[: _G["split"]], dtype=np.float64)
    p[~np.isfinite(p)] = np.nan
    lab = np.isfinite(tgt)
    out = {"lib": lib, "i": i, "sign": sign,
           "coverage_train": float((np.isfinite(p) & lab).sum() / max(lab.sum(), 1)),
           "distinct_train": distinct_fraction(p)}
    out.update(train_books(sign * p, tgt, _G["cost"]))
    return out


def score_candidates(rows: Sequence[dict], target_path: str, split: int, cache: Path, workers: int,
                     cost: float = COST) -> Dict[Tuple[str, int], dict]:
    """每个候选的训练段书。结果追加写进 cache(csv),重跑只补缺的;cache 旁边的 .meta.json 记下切分点、
    费率、标签文件和各库的目录,对不上就停(缓存只按 (库, 序号) 认因子)。"""
    from multiprocessing import get_context

    meta = {"split": split, "cost": cost, "target": str(Path(target_path).resolve()),
            "dirs": {r["lib"]: str(Path(r["dir"]).resolve()) for r in rows}, "cols": list(SCORE_COLS)}
    meta_path = cache.with_suffix(cache.suffix + ".meta.json")
    done: Dict[Tuple[str, int], dict] = {}
    if cache.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
        if old is None or any(old.get(k) != meta[k] for k in ("split", "cost", "target", "cols")) \
                or any(old["dirs"].get(k, v) != v for k, v in meta["dirs"].items()):
            raise SystemExit(f"{cache} 是别的设置下算的(切分点 / 费率 / 标签 / 库目录对不上),换个文件名")
        meta["dirs"] = {**old["dirs"], **meta["dirs"]}
        for r in csv.DictReader(open(cache, encoding="utf-8")):
            done[(r["lib"], int(r["i"]))] = {k: (r[k] if k == "lib" else float(r[k])) for k in SCORE_COLS}
    jobs = [(r["lib"], r["dir"], int(r["i"]), float(np.sign(float(r["rank_ic_train"]))))
            for r in rows if (r["lib"], int(r["i"])) not in done]
    cache.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    if jobs:
        new = not cache.exists()
        t0 = time.time()
        with open(cache, "a", newline="", encoding="utf-8") as fh, \
                get_context("fork").Pool(workers, initializer=_init, initargs=(target_path, split, cost)) as pool:
            w = csv.DictWriter(fh, fieldnames=SCORE_COLS)
            if new:
                w.writeheader()
            for n, s in enumerate(pool.imap_unordered(_score, jobs, chunksize=4), 1):
                w.writerow(s)
                done[(s["lib"], int(s["i"]))] = s
                if n % 200 == 0 or n == len(jobs):
                    fh.flush()
                    print(f"  训练段书 {n}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    return done


def eligible_order(rows: Sequence[dict], scores: Dict[Tuple[str, int], dict], min_coverage: float,
                   min_distinct: float, min_sharpe: float, key: str = "long_sharpe_train") -> List[int]:
    """过了门槛的行号,按训练段扣费夏普(key)从高到低。只读训练段的数。"""
    ok = []
    for j, r in enumerate(rows):
        s = scores.get((r["lib"], int(r["i"])))
        if s is None or not np.isfinite(s[key]):
            continue
        if s["coverage_train"] >= min_coverage and s["distinct_train"] >= min_distinct and s[key] >= min_sharpe:
            ok.append(j)
    return sorted(ok, key=lambda j: -scores[(rows[j]["lib"], int(rows[j]["i"]))][key])


def smoothed_fingerprint(panel_train: np.ndarray, sign: float, halflife: float, cols: np.ndarray,
                         days: int = FP_DAYS) -> np.ndarray:
    """实际拿来用的信号的训练段指纹:方向 x 面板、平滑之后,最后 days 天 x cols 这些票,逐日 z 分数化秩,缺的记 0。
    panel_train 只含训练段;EMA 逐只股票独立,所以先取列再平滑是精确的。"""
    from alphacube.metrics import ema_rows, rank_rows, zscore_rows

    p = np.asarray(panel_train[:, cols], dtype=np.float64)
    p[~np.isfinite(p)] = np.nan
    sm = ema_rows(sign * p, halflife) if halflife else sign * p
    return np.nan_to_num(zscore_rows(rank_rows(sm[-days:])), nan=0.0).reshape(-1)


def pairwise_rank_corr(panels: Sequence[np.ndarray], days: slice) -> np.ndarray:
    """逐日两两截面秩相关在这些天上的平均。秩在每个因子自己有值的票上取,相关只在两边都有值的票上算
    (两个因子覆盖一样时就是 Spearman)。"""
    from alphacube.metrics import rank_rows

    R = [rank_rows(np.asarray(p[days], dtype=np.float64)) for p in panels]
    k = len(R)
    tot, cnt = np.zeros((k, k)), np.zeros((k, k))
    for t in range(R[0].shape[0]):
        X = np.stack([r[t] for r in R])
        M = np.isfinite(X).astype(np.float64)
        X0 = np.nan_to_num(X)
        n = M @ M.T
        with np.errstate(invalid="ignore", divide="ignore"):
            sx = X0 @ M.T                               # sx[i, j] = 因子 i 在 (i, j) 共同有值的票上的和
            sxx = (X0 * X0) @ M.T
            cov = X0 @ X0.T - sx * sx.T / n
            var = sxx - sx * sx / n
            c = cov / np.sqrt(var * var.T)
        ok = (n >= MIN_COMMON) & np.isfinite(c)
        tot[ok] += c[ok]
        cnt[ok] += 1
    with np.errstate(invalid="ignore"):
        return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def offdiag(c: np.ndarray) -> np.ndarray:
    return c[~np.eye(c.shape[0], dtype=bool)]


def return_corr(series: Sequence[np.ndarray]) -> np.ndarray:
    """日收益序列两两相关(只用两边都有值的天)。"""
    k = len(series)
    out = np.eye(k)
    for a in range(k):
        for b in range(a + 1, k):
            m = np.isfinite(series[a]) & np.isfinite(series[b])
            c = np.corrcoef(series[a][m], series[b][m])[0, 1] if m.sum() >= MIN_COMMON else np.nan
            out[a, b] = out[b, a] = c
    return out


def pick_report(sm: np.ndarray, target: np.ndarray, split: int, cost: float = COST) -> Tuple[dict, np.ndarray]:
    """平滑后的信号 -> 训练段 / 测试段各自建书(都从空仓起)的纯多头书和多空书;另外返回连着的多头书日扣费超额。"""
    from alphacube.metrics import long_short_weights, portfolio_returns
    from long_only_grid import stats

    out: dict = {}
    series = np.full(sm.shape[0], np.nan)
    for per, days in (("train", slice(0, split)), ("test", slice(split, sm.shape[0]))):
        w = long_short_weights(sm[days], TOP_FRAC)
        ls_net, _, ls_turn = portfolio_returns(w, target[days], cost)
        net, gross, turn = long_book(w, sm[days], target[days], cost)
        series[days] = net
        lo, ls = stats(net), stats(ls_net)
        out[per] = {"long_sharpe": lo["sharpe"], "long_ann": lo["ann"], "long_mdd": lo["mdd"],
                    "long_gross_ann": stats(gross)["ann"], "long_turnover": float(np.nanmean(turn)),
                    "ls_sharpe": ls["sharpe"], "ls_ann": ls["ann"], "ls_turnover": float(np.nanmean(ls_turn))}
    return out, series


def compare_panels(new: np.ndarray, old: np.ndarray) -> dict:
    """重新求值的面板和盘上的面板:NaN 位置不一致的占比,和共同格上逐日秩相关的均值 / 最小值。
    逐格相等要求不了 —— 分母贴着 0 的格子(一字板当天的 Durbin-Watson 是 0/0)算出 NaN 还是一个数,
    取决于同一批里一起算的是谁。"""
    from alphacube.metrics import cs_pearson, rank_rows

    if new.shape != old.shape:
        return {"ok": False, "why": f"形状 {new.shape} != {old.shape}"}
    fa, fb = np.isfinite(new), np.isfinite(old)
    both = fa & fb
    c = cs_pearson(rank_rows(np.where(both, new, np.nan)), rank_rows(np.where(both, old, np.nan)))
    mism = float((fa != fb).mean())
    mean_c = float(np.nanmean(c)) if np.isfinite(c).any() else np.nan
    return {"ok": bool(mism <= REEVAL_MASK_TOL and mean_c >= REEVAL_CORR_TOL), "mask_mismatch": mism,
            "rank_corr_mean": mean_c, "rank_corr_min": float(np.nanmin(c)) if np.isfinite(c).any() else np.nan}


def reevaluate(rows: Sequence[dict], store: str, mirror_root: str, device: str, start: str, end: str,
               warmup: int) -> List[dict]:
    """把表达式字符串重新解析、求值,和盘上打过分的面板比。"""
    from alphacube.engine.engine import CubeEngine
    from alphacube.parser import parse, parse_legacy
    from alphacube.research import warmed_view
    from alphacube.store import open_cube_store

    exprs = []
    for r in rows:
        try:
            exprs.append(parse(r["expression"]))
        except Exception:                               # 手写库的表达式用了搜索语法之外的算子
            exprs.append(parse_legacy(r["expression"]))
    view, _ = warmed_view(open_cube_store(store, mirror_root=mirror_root or None), start, end, warmup=warmup)
    eng = CubeEngine(view, device=device, workdir=Path(__file__).resolve().parents[1] / "tmp")
    out = []
    for r, e in zip(rows, exprs):                       # 一条一条算:重算子和别的挤在一批里会爆显存
        p = eng.evaluate([e], tolerant=True).panels[0]
        if p is None:
            out.append({"ok": False, "why": "重新求值失败"})
            continue
        out.append(compare_panels(np.asarray(p, dtype=np.float32),
                                  np.load(Path(r["dir"]) / "panels" / f"{int(r['i']):05d}.npy")))
    return out


def _spread(name: str, c: np.ndarray, t0: float, absolute: bool = True) -> None:
    od = offdiag(c)
    od = np.abs(od) if absolute else od
    print(f"{name}:中位 {np.nanmedian(od):.3f},90 分位 {np.nanpercentile(od, 90):.3f},最大 {np.nanmax(od):.3f} "
          f"({time.time() - t0:.0f}s)", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from alphacube.metrics import ema_rows

    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", action="append", required=True, help="名字=打分目录(metrics.csv / manifest.csv / "
                    "panels / _target.npy),可以给多个;同一条表达式留先给的那个库")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scores", required=True, help="训练段书的缓存 csv(没有就算,有就只补缺的)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rank", choices=sorted(RANKS), default="long", help="按哪本书的训练段扣费夏普排:"
                    "long = 前 10% 纯多头减等权,ls = 十分位多空")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--n", type=int, default=20, help="最多收几个")
    ap.add_argument("--min-coverage", type=float, default=MIN_COVERAGE)
    ap.add_argument("--min-distinct", type=float, default=MIN_DISTINCT)
    ap.add_argument("--min-sharpe", type=float, default=MIN_SHARPE)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--no-report", action="store_true", help="只挑,不在整张面板上重算(看数量用)")
    ap.add_argument("--verify", action="store_true", help="把表达式重新求值,和盘上的面板比")
    ap.add_argument("--store", default="/root/autodl-tmp/qmt_daily_data/fast_factor_compute")
    ap.add_argument("--mirror-root", default="/root/autodl-tmp/AlphaCube/mirror")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--start", default="20180102")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--warmup", type=int, default=60)
    a = ap.parse_args(argv)

    t0 = time.time()
    rows: List[dict] = []
    target, target_path = None, ""
    for spec in a.lib:
        tag, _, d = spec.partition("=")
        lib_rows = load_library(Path(d), tag)
        rows += lib_rows
        tgt = np.load(Path(d) / "_target.npy", mmap_mode="r")
        if target is None:
            target, target_path = np.asarray(tgt, dtype=np.float64), str(Path(d) / "_target.npy")
        elif tgt.shape != target.shape or not np.array_equal(np.asarray(tgt), target.astype(tgt.dtype),
                                                             equal_nan=True):
            raise SystemExit(f"{d} 的标签和前面的库不是同一份,不能放在一起比")
        print(f"{tag}: {len(lib_rows)} 个候选", flush=True)
    n_all = len(rows)
    rows = dedupe(rows)
    print(f"合计 {n_all} 个,按表达式去重后 {len(rows)} 个", flush=True)
    D, S = target.shape
    split = max(1, min(D - 1, int(D * a.train_frac)))
    cols = np.linspace(0, S - 1, num=min(FP_SYMBOLS, S)).astype(int)
    hl_key, sharpe_key = RANKS[a.rank]

    scores = score_candidates(rows, target_path, split, Path(a.scores), a.workers)
    order = eligible_order(rows, scores, a.min_coverage, a.min_distinct, a.min_sharpe, sharpe_key)

    def sc(j: int) -> dict:
        return scores[(rows[j]["lib"], int(rows[j]["i"]))]

    def panel(j: int) -> np.ndarray:
        return np.load(Path(rows[j]["dir"]) / "panels" / f"{int(rows[j]['i']):05d}.npy", mmap_mode="r")

    def vec(j: int) -> Optional[np.ndarray]:
        return unit(smoothed_fingerprint(panel(j)[:split], sc(j)["sign"], sc(j)[hl_key], cols))

    gates = {"覆盖率不够": sum(sc(j)["coverage_train"] < a.min_coverage for j in range(len(rows))),
             "取值档位太少": sum(sc(j)["distinct_train"] < a.min_distinct for j in range(len(rows))),
             "训练段扣费夏普不够": sum(not sc(j)[sharpe_key] >= a.min_sharpe for j in range(len(rows)))}
    idx = greedy(order, vec, a.threshold, a.n)
    print(f"按 {a.rank} 排;门槛(可重叠):{gates};过门槛 {len(order)} 个,平滑后信号 |相关| < {a.threshold} "
          f"收了 {len(idx)} 个(看到第 {order.index(idx[-1]) + 1 if idx else 0} 名) ({time.time() - t0:.0f}s)", flush=True)

    picks = []
    for j in idx:
        r, s = rows[j], sc(j)
        picks.append({"lib": r["lib"], "i": int(r["i"]), "run": r["run"], "name": r["name"],
                      "expression": r["expression"], "dir": r["dir"], "sign": float(s["sign"]),
                      "halflife": float(s[hl_key]), "train_rank": order.index(j) + 1,
                      "train_scores": {k: float(s[k]) for k in SCORE_COLS[3:]},
                      "rank_ic_train": float(r["rank_ic_train"]), "rank_ic_test": _num(r.get("rank_ic_test"))})
    out = {"rank": a.rank, "threshold": a.threshold, "min_coverage": a.min_coverage, "min_distinct": a.min_distinct,
           "min_sharpe": a.min_sharpe, "split": split, "n_candidates": len(rows), "n_eligible": len(order),
           "gates": gates, "picks": picks}
    if not a.no_report and picks:
        smoothed, series = [], []
        for p in picks:
            raw = np.asarray(np.load(Path(p["dir"]) / "panels" / f"{p['i']:05d}.npy"), dtype=np.float64)
            raw[~np.isfinite(raw)] = np.nan
            sm = ema_rows(p["sign"] * raw, p["halflife"]) if p["halflife"] else p["sign"] * raw
            p["book"], net = pick_report(sm, target, split)
            smoothed.append(sm.astype(np.float32))
            series.append(net)
            b = p["book"]
            print(f"  {p['lib']} #{p['i']} hl={p['halflife']:g} 纯多头扣费超额夏普 训练 {b['train']['long_sharpe']:+.2f} "
                  f"测试 {b['test']['long_sharpe']:+.2f};多空 {b['train']['ls_sharpe']:+.2f} / "
                  f"{b['test']['ls_sharpe']:+.2f} ({time.time() - t0:.0f}s)", flush=True)
        for per, days in (("train", slice(0, split)), ("test", slice(split, D))):
            c = pairwise_rank_corr(smoothed, days)
            out[f"corr_{per}"] = c.tolist()
            _spread(f"{per} 段平滑后信号的两两 |秩相关|", c, t0)
            rc = return_corr([s[days] for s in series])
            out[f"return_corr_{per}"] = rc.tolist()
            _spread(f"{per} 段多头书日超额的两两相关(带符号)", rc, t0, absolute=False)
        out["long_net"] = [[None if not np.isfinite(x) else float(x) for x in s] for s in series]
    if a.verify and picks:
        for p, v in zip(picks, reevaluate(picks, a.store, a.mirror_root, a.device, a.start, a.end, a.warmup)):
            p["reevaluated"] = v
        bad = [p for p in picks if not p["reevaluated"]["ok"]]
        print(f"表达式重新求值:{len(picks) - len(bad)}/{len(picks)} 和盘上的面板一致", flush=True)
        for p in bad:
            print(f"  对不上:{p['lib']} #{p['i']} {p['reevaluated']}  {p['expression']}", flush=True)
    for p in picks:
        del p["dir"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {a.out} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
