"""选因子:相关性聚类 -> 族内取候选 -> 前向选择 -> 族内替换 -> 全局交换。

为什么不用"按单因子分数排前 k"(现在 train_all_factors.py 的做法):那个规则
**不是单调的** —— 候选池从 2775 涨到 4157 之后,组合的样本外扣费夏普从 +3.25
掉到 +2.93,因为个体分数更高的新因子顶掉了更互补的老因子(有效因子数 4.7 -> 3.8)。
只要选择是在优化一个**集合目标**,单调性就是自动的:池子变大时旧解仍然可行。

四个阶段:
1. 聚类:按日内秩相关做层次聚类(1-|corr| 距离,平均连接),切在 `--cluster`。
   同一族里的因子是同一个想法的不同写法 —— gpu15 前 20 名就是一个母题的 20 种。
2. 每族留几个候选(按适应度/|rank IC|/最低换手各取一个),把 4000+ 压到几百,
   前向选择才跑得动。
3. 前向选择:每步加入"让**组合**的训练段目标涨得最多"的那个。判据是组合本身的
   `|rank IC| - lambda x 换手`,不是候选自己的分数,也不是残差的分数 —— 实测用
   残差 IC 选会让换手失控(+2.93 -> +1.10),加了阈值又只选出 4 个(+1.56)。
4. 族内替换 + 全局交换:修前向选择的路径依赖。族内替换要看**整个族**,不只是
   第 2 步留下的那几个:今天发现的抑制变量对里,那个因子单看任何指标都平庸
   (原始 rank IC -0.011),配上同族伙伴之后测试段 rank IC 到 +0.062。

一切只用训练段:指纹、方向对齐、聚类、目标,全部在切分点之前。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

LAMBDA = 0.14          # 换手在 rank IC 单位下的价格(15bp/边),见 mining/scoring.py


def cluster_by_correlation(corr: np.ndarray, threshold: float) -> np.ndarray:
    """1-|corr| 上的平均连接层次聚类,返回每行的族号。

    用 scipy 的实现;没有 scipy 时退回到一个等价的朴素合并(测试用的小矩阵够了)。
    """
    k = corr.shape[0]
    d = np.clip(1.0 - np.abs(corr), 0.0, 2.0)
    np.fill_diagonal(d, 0.0)
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        z = linkage(squareform((d + d.T) / 2, checks=False), method="average")
        return fcluster(z, t=1.0 - threshold, criterion="distance") - 1
    except ImportError:
        lab = np.arange(k)
        while True:
            best = None
            for a in range(k):
                for b in range(a + 1, k):
                    if lab[a] == lab[b]:
                        continue
                    m = np.mean([d[i, j] for i in np.flatnonzero(lab == lab[a])
                                 for j in np.flatnonzero(lab == lab[b])])
                    if m <= 1.0 - threshold and (best is None or m < best[0]):
                        best = (m, lab[a], lab[b])
            if best is None:
                break
            lab[lab == best[2]] = best[1]
        _, out = np.unique(lab, return_inverse=True)
        return out


class SetObjective:
    """一个成员集合 -> 组合在训练段的 |rank IC| - lambda x 日换手。

    成员按训练段 IC 的方向对齐后等权相加(等权而不是拟合权重:选择阶段每一步都要
    算几百上千次,拟合权重既慢又容易在选择过程中过拟合;最终报告仍然会把 lstsq /
    ridge / ic / icir 都跑一遍)。
    """

    def __init__(self, fp: np.ndarray, signs: np.ndarray, target_ranks: np.ndarray,
                 n_days: int, n_syms: int, lam: float = LAMBDA, top_frac: float = 0.1) -> None:
        self.fp = fp                      # (n_factors, n_days*n_syms) 逐日 z 分数化的秩
        self.signs = signs
        self.y = target_ranks             # 同形状的标签秩
        self.n_days, self.n_syms = n_days, n_syms
        self.lam, self.top_frac = lam, top_frac
        self._cache: Dict[Tuple[int, ...], float] = {}

    def combined(self, members: Sequence[int]) -> np.ndarray:
        acc = np.zeros(self.fp.shape[1], dtype=np.float64)
        for i in members:
            acc += self.signs[i] * self.fp[i]
        return acc / max(len(members), 1)

    def score(self, members: Sequence[int]) -> float:
        key = tuple(sorted(members))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        from alphacube.metrics import long_short_weights, rank_rows

        panel = self.combined(members).reshape(self.n_days, self.n_syms)
        rk = rank_rows(panel)
        ok = np.isfinite(rk) & np.isfinite(self.y)
        ics = []
        for t in range(self.n_days):
            m = ok[t]
            if m.sum() < 20:
                continue
            a, b = rk[t][m], self.y[t][m]
            sa, sb = a.std(), b.std()
            if sa > 1e-12 and sb > 1e-12:
                ics.append(float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb)))
        if not ics:
            return -np.inf
        w = long_short_weights(panel, self.top_frac)
        turn = float(np.nanmean(np.abs(np.diff(w, axis=0)).sum(axis=1) / 2.0))
        out = self._cache[key] = abs(float(np.mean(ics))) - self.lam * turn
        return out


def forward_select(obj: SetObjective, pool: Sequence[int], k: int, verbose=True) -> List[int]:
    chosen: List[int] = []
    best_score = -np.inf
    for step in range(k):
        cand, best = None, best_score
        for i in pool:
            if i in chosen:
                continue
            s = obj.score(chosen + [i])
            if s > best:
                cand, best = i, s
        if cand is None:
            if verbose:
                print(f"  前向选择:第 {step+1} 步没有候选能改善目标({best_score:+.4f}),停", flush=True)
            break
        chosen.append(cand)
        best_score = best
        if verbose and (step + 1) % 5 == 0:
            print(f"  前向选择 {step+1}/{k}:目标 {best_score:+.4f}", flush=True)
    return chosen


def replace_within_clusters(obj: SetObjective, chosen: List[int], labels: np.ndarray,
                            verbose=True) -> List[int]:
    """每个成员在**整个族**里找更好的替身。族内的都是同一个想法的不同写法,
    换一个写法可能换手更低、或者和其余成员配合得更好。"""
    cur = list(chosen)
    base = obj.score(cur)
    for pos, member in enumerate(list(cur)):
        family = np.flatnonzero(labels == labels[member])
        for alt in family:
            if alt == member or alt in cur:
                continue
            trial = list(cur)
            trial[pos] = int(alt)
            s = obj.score(trial)
            if s > base + 1e-9:
                cur, base = trial, s
                if verbose:
                    print(f"  族内替换:第 {pos} 位 {member} -> {alt},目标 {base:+.4f}", flush=True)
    return cur


def global_swap(obj: SetObjective, chosen: List[int], pool: Sequence[int],
                rounds: int = 3, verbose=True) -> List[int]:
    """删一个、加一个,只要目标涨就接受。修前向选择的路径依赖。"""
    cur = list(chosen)
    base = obj.score(cur)
    for r in range(rounds):
        improved = False
        for pos in range(len(cur)):
            rest = cur[:pos] + cur[pos + 1:]
            for i in pool:
                if i in cur:
                    continue
                s = obj.score(rest + [i])
                if s > base + 1e-9:
                    cur, base, improved = rest + [i], s, True
                    if verbose:
                        print(f"  全局交换:{cur[pos] if pos < len(cur) else '?'} -> {i},目标 {base:+.4f}",
                              flush=True)
                    break
        if not improved:
            break
    return cur


def champion_challenger(obj: SetObjective, challenger: List[int],
                        champion: Optional[List[int]]) -> Tuple[List[int], str]:
    """新解打不过旧解就留旧解 —— 这一步让"加因子只会更好"成为硬保证。"""
    if not champion:
        return challenger, "没有擂主"
    a, b = obj.score(challenger), obj.score(champion)
    return (challenger, f"挑战者胜 {a:+.4f} > {b:+.4f}") if a > b else (champion, f"擂主守擂 {b:+.4f} >= {a:+.4f}")


__all__ = ["cluster_by_correlation", "SetObjective", "forward_select",
           "replace_within_clusters", "global_swap", "champion_challenger", "LAMBDA"]


# --- 生产入口 -----------------------------------------------------------------

def build_train_fingerprints(panels_dir: Path, indices: Sequence[int], days: slice,
                             cols: np.ndarray, out: Optional[Path] = None) -> np.ndarray:
    """(n, n_days*n_syms) float32:只覆盖训练段的逐日 z 分数化秩。

    不能复用 _bench/allfac/rows.npy —— 那份是在**全样本**上等距抽 250 天建的,
    拿它来选因子会让测试段的信息漏进选择过程。
    """
    from alphacube.metrics import rank_rows, zscore_rows

    if out is not None and out.exists():
        return np.load(out, mmap_mode="r")
    n_days = len(range(*days.indices(10 ** 9))) if days.stop else 0
    fp = np.empty((len(indices), n_days * len(cols)), dtype=np.float32)
    t0 = time.time()
    for j, i in enumerate(indices):
        p = np.load(panels_dir / f"{int(i):05d}.npy", mmap_mode="r")
        sub = np.asarray(p[days][:, cols], dtype=np.float64)
        fp[j] = np.nan_to_num(zscore_rows(rank_rows(sub)), nan=0.0).reshape(-1)
        if (j + 1) % 500 == 0:
            print(f"  指纹 {j+1}/{len(indices)} ({time.time()-t0:.0f}s)", flush=True)
    if out is not None:
        np.save(out, fp)
    return fp


def pairwise_abs_corr(fp: np.ndarray, block: int = 512) -> np.ndarray:
    """|corr| 矩阵,分块算以控内存(4157 x 4157 的输出只有 69 MB,中间量才是大头)。"""
    n = fp.shape[0]
    norms = np.linalg.norm(fp, axis=1)
    norms[norms == 0] = 1.0
    out = np.empty((n, n), dtype=np.float32)
    for a in range(0, n, block):
        A = fp[a:a + block]
        out[a:a + block] = np.abs((A @ fp.T) / np.outer(norms[a:a + block], norms))
    np.fill_diagonal(out, 1.0)
    return out


def shortlist(labels: np.ndarray, rows: List[dict], per_family: int = 3) -> List[int]:
    """每族按三个不同口径各取一个:适应度最好、|rank IC| 最大、换手最低。

    三个口径而不是一个,是因为今天量到"单因子分数和边际贡献几乎无关",
    单一口径会系统性地漏掉某一类(按适应度筛会把高换手的全毙掉,包括 RL 挖的)。
    """
    num = lambda r, k: float(r[k]) if r[k] not in ("", "nan", "None") else np.nan   # noqa: E731
    keep: List[int] = []
    for fam in np.unique(labels):
        idx = np.flatnonzero(labels == fam)
        if idx.size == 0:
            continue
        order = {
            "fit": sorted(idx, key=lambda j: -(abs(num(rows[j], "rank_ic_train"))
                                               - LAMBDA * num(rows[j], "turnover"))),
            "ric": sorted(idx, key=lambda j: -abs(num(rows[j], "rank_ic_train"))),
            "turn": sorted(idx, key=lambda j: num(rows[j], "turnover")),
        }
        picked = []
        for key in ("fit", "ric", "turn"):
            for j in order[key]:
                if j not in picked:
                    picked.append(int(j))
                    break
        keep.extend(picked[:per_family])
    return sorted(set(keep))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="_bench/allfac", help="metrics.csv + panels/ 所在目录")
    ap.add_argument("--store", default="/root/autodl-tmp/qmt_daily_data/fast_factor_compute")
    ap.add_argument("--start", default="20180102")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--label", default="open_vwap15")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--days", type=int, default=250, help="选择窗口:训练段最后这么多个**连续**交易日")
    ap.add_argument("--symbols", type=int, default=512)
    ap.add_argument("--cluster", type=float, default=0.7, help="聚类阈值:|rank corr| 高于它算同一族")
    ap.add_argument("--per-family", type=int, default=3)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--penalty", type=float, default=LAMBDA)
    ap.add_argument("--champion", default="", help="上一轮的选择(json);新解打不过它就留它")
    ap.add_argument("--out", default="_bench/selection.json")
    a = ap.parse_args(argv)

    from alphacube.label import Label
    from alphacube.metrics import cs_pearson, nanmean_quiet, rank_rows
    from alphacube.research import warmed_view
    from alphacube.store import open_cube_store

    d = Path(a.dir)
    t0 = time.time()
    view, _ = warmed_view(open_cube_store(a.store), a.start or None, a.end or None, warmup=a.warmup)
    target = Label(a.label, 1).compute(view).astype(np.float64)
    D, S = target.shape
    split = max(1, min(D - 1, int(D * a.train_frac)))
    days = slice(max(0, split - a.days), split)
    cols = np.linspace(0, S - 1, num=min(a.symbols, S)).astype(int)
    n_days = len(range(*days.indices(D)))
    y_ranks = rank_rows(target[days][:, cols])
    print(f"选择窗口:训练段最后 {n_days} 个连续交易日 x {len(cols)} 只(切分点 {split}/{D})", flush=True)

    rows = [r for r in csv.DictReader(open(d / "metrics.csv", encoding="utf-8")) if r["healthy"] == "True"]
    num = lambda r, k: float(r[k]) if r[k] not in ("", "nan", "None") else np.nan   # noqa: E731
    rows = [r for r in rows if np.isfinite(num(r, "rank_ic_train")) and np.isfinite(num(r, "turnover"))]
    idx = [int(r["i"]) for r in rows]
    print(f"健康因子 {len(rows)} 个", flush=True)

    cache = d / f"fp_train_{n_days}x{len(cols)}.npy"
    fp = np.asarray(build_train_fingerprints(d / "panels", idx, days, cols, cache))
    print(f"指纹就绪 {fp.nbytes/2**30:.2f} GB ({time.time()-t0:.0f}s)", flush=True)

    corr = pairwise_abs_corr(fp)
    labels = cluster_by_correlation(corr, a.cluster)
    sizes = np.bincount(labels)
    print(f"聚类:{sizes.size} 族(最大 {sizes.max()},中位 {int(np.median(sizes))},"
          f"单点族 {int((sizes == 1).sum())}) ({time.time()-t0:.0f}s)", flush=True)

    pool = shortlist(labels, rows, a.per_family)
    print(f"候选 {len(pool)} 个(每族最多 {a.per_family} 个) ({time.time()-t0:.0f}s)", flush=True)

    signs = np.array([1.0 if nanmean_quiet(cs_pearson(
        np.load(d / "panels" / f"{int(rows[j]['i']):05d}.npy", mmap_mode="r")[days][:, cols].astype(np.float64),
        target[days][:, cols])) >= 0 else -1.0 for j in range(len(rows))])
    obj = SetObjective(fp, signs, y_ranks, n_days, len(cols), lam=a.penalty)

    chosen = forward_select(obj, pool, a.k)
    print(f"前向选择 {len(chosen)} 个,目标 {obj.score(chosen):+.4f} ({time.time()-t0:.0f}s)", flush=True)
    chosen = replace_within_clusters(obj, chosen, labels)
    print(f"族内替换后目标 {obj.score(chosen):+.4f} ({time.time()-t0:.0f}s)", flush=True)
    chosen = global_swap(obj, chosen, pool)
    print(f"全局交换后目标 {obj.score(chosen):+.4f} ({time.time()-t0:.0f}s)", flush=True)

    champion = None
    if a.champion and Path(a.champion).exists():
        prev = json.loads(Path(a.champion).read_text())["members"]
        where = {int(r["i"]): j for j, r in enumerate(rows)}
        champion = [where[int(m["i"])] for m in prev if int(m["i"]) in where]
    chosen, why = champion_challenger(obj, chosen, champion)
    print(f"擂主比较:{why}", flush=True)

    out = {"members": [{"i": int(rows[j]["i"]), "run": rows[j]["run"], "cluster": int(labels[j]),
                        "sign": float(signs[j]), "raw_rank_ic_train": num(rows[j], "rank_ic_train"),
                        "turnover": num(rows[j], "turnover"), "expression": rows[j]["expression"]}
                       for j in chosen],
           "objective": obj.score(chosen), "n_clusters": int(sizes.size), "pool": len(pool)}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    fams = len({labels[j] for j in chosen})
    print(f"写出 {a.out}:{len(chosen)} 个成员,来自 {fams} 个族 ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
