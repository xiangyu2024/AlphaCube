"""选因子流程:聚类 -> 前向选择 -> 族内替换 -> 全局交换 -> 擂主保护。

要钉住的是"按单因子分数排前 k"没有的性质:
- 目标是**集合**的函数,所以前向选择的每一步都不会让目标变差;
- 交换只接受改善,所以它单调;
- 擂主保护让"候选池变大 -> 结果不会变差"成为硬保证 —— 这正是现在缺的那条
  (候选从 2775 涨到 4157 时,组合的样本外扣费夏普从 +3.25 掉到 +2.93)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from select_factors import (SetObjective, champion_challenger, cluster_by_correlation,   # noqa: E402
                            forward_select, global_swap, replace_within_clusters)

D, S = 60, 40


def _planted(seed=0):
    """三个真信号 + 每个信号的两个近似复制 + 两个纯噪声。"""
    rng = np.random.default_rng(seed)
    y = rng.normal(size=(D, S))
    base = [y * 0.5 + rng.normal(size=(D, S)) * 0.5 for _ in range(3)]
    factors = []
    for b in base:
        factors.append(b)
        factors.append(b + 0.05 * rng.normal(size=(D, S)))     # 同族:几乎一样
        factors.append(b * -1.2 + 0.05 * rng.normal(size=(D, S)))   # 同族:反号也是同一个想法
    factors += [rng.normal(size=(D, S)), rng.normal(size=(D, S))]   # 噪声,各成一族
    return factors, y


def _objective(factors, y):
    from alphacube.metrics import rank_rows, zscore_rows

    fp = np.stack([np.nan_to_num(zscore_rows(rank_rows(f)), nan=0.0).reshape(-1) for f in factors])
    yr = rank_rows(y)
    signs = np.array([np.sign(np.corrcoef(fp[i], np.nan_to_num(zscore_rows(yr), nan=0.0).reshape(-1))[0, 1]) or 1.0
                      for i in range(len(factors))])
    return SetObjective(fp, signs, yr, D, S, lam=0.0), fp


def test_clustering_finds_the_planted_families():
    factors, y = _planted()
    _obj, fp = _objective(factors, y)
    corr = np.corrcoef(fp)
    labels = cluster_by_correlation(corr, threshold=0.7)
    for start in (0, 3, 6):                       # 每三个是一族
        assert labels[start] == labels[start + 1] == labels[start + 2], labels
    assert labels[0] != labels[3] != labels[6]
    assert len({labels[9], labels[10]} & {labels[0], labels[3], labels[6]}) == 0   # 噪声自成一族


def test_forward_selection_never_makes_the_objective_worse():
    factors, y = _planted()
    obj, _ = _objective(factors, y)
    pool = list(range(len(factors)))
    chosen, prev = [], -np.inf
    for k in range(1, 5):
        chosen = forward_select(obj, pool, k, verbose=False)
        s = obj.score(chosen)
        assert s >= prev - 1e-12, f"k={k} 目标反而下降:{prev} -> {s}"
        prev = s


def test_forward_selection_picks_one_per_family_before_repeating():
    """三个真信号各一个,应该先被选中,而不是同一族的三个复制。"""
    factors, y = _planted()
    obj, fp = _objective(factors, y)
    labels = cluster_by_correlation(np.corrcoef(fp), threshold=0.7)
    chosen = forward_select(obj, list(range(len(factors))), 3, verbose=False)
    assert len({labels[i] for i in chosen}) == 3, f"选了同族的:{[labels[i] for i in chosen]}"


def test_swap_and_replacement_only_accept_improvements():
    factors, y = _planted()
    obj, fp = _objective(factors, y)
    labels = cluster_by_correlation(np.corrcoef(fp), threshold=0.7)
    start = [9, 10, 0]                                  # 故意从两个噪声开始
    s0 = obj.score(start)
    after_rep = replace_within_clusters(obj, start, labels, verbose=False)
    assert obj.score(after_rep) >= s0 - 1e-12
    after_swap = global_swap(obj, after_rep, list(range(len(factors))), verbose=False)
    assert obj.score(after_swap) >= obj.score(after_rep) - 1e-12
    assert obj.score(after_swap) > s0, "从两个噪声出发,交换必须能救回来"


def test_champion_protects_against_a_worse_challenger():
    factors, y = _planted()
    obj, _ = _objective(factors, y)
    good = forward_select(obj, list(range(len(factors))), 3, verbose=False)
    bad = [9, 10]
    kept, why = champion_challenger(obj, bad, good)
    assert kept == good and "守擂" in why
    kept2, why2 = champion_challenger(obj, good, bad)
    assert kept2 == good and "挑战者胜" in why2
    assert champion_challenger(obj, bad, None)[0] == bad      # 没有擂主时照收


def test_shortlist_spans_the_cost_tradeoff_not_the_extremes():
    """每族三个候选来自同一个式子的三个成本假设,不是"|rank IC| 最大 + 换手最低"
    那种端点解 —— 端点上一个是靠高换手撑起来的强度,一个可能几乎不动、没信息。"""
    from select_factors import shortlist

    rows = [  # 同一族:慢而弱 / 中庸 / 快而强
        {"rank_ic_train": "0.020", "turnover": "0.01"},
        {"rank_ic_train": "0.035", "turnover": "0.10"},
        {"rank_ic_train": "0.070", "turnover": "0.30"},
        {"rank_ic_train": "0.005", "turnover": "0.001"},   # 几乎不动、没信息:不该被选
    ]
    labels = np.zeros(len(rows), dtype=int)
    picked = shortlist(labels, rows, per_family=3, lambdas=(0.07, 0.14, 0.20))
    assert picked == [0, 1, 2], picked        # 三个成本假设各挑一个,顺延不重复
    assert 3 not in picked, "换手最低但没信息的那个不该进来"
    # λ 越大越偏向慢的:极端成本假设下选的是最慢的那个
    assert shortlist(labels, rows, 1, (1.0,)) == [0]
    assert shortlist(labels, rows, 1, (0.0,)) == [2]       # 不计成本时选最强的
