"""选族只能用训练段:只改测试段的因子值,build_champion 选出的族和成员必须一字不差。

对抗审查查出(2026-10-03):pick_families 的适应度 |训练段 rank IC| - 0.14 x 换手 里,换手是
score_all_factors 在全样本上算的,平均进了测试段的因子值 —— 测试段越"抖"的因子越会被挤出族、挤出前 100 族。
这里走真实的打分代码(score_all_factors.score)和真实的选族代码(build_champion.pick_families):
- 现在的适应度(训练段换手):测试段随便改,选出来的不变;
- 换回全样本换手:同一份改动就会换掉成员 —— 证明这个测试真的能抓到那种依赖。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_champion as bc                                   # noqa: E402
import score_all_factors as saf                               # noqa: E402
import select_factors as sf                                   # noqa: E402

D, S, SPLIT = 100, 60, 70


def _write(d: Path, panels):
    (d / "panels").mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(panels):
        np.save(d / "panels" / f"{i:05d}.npy", p.astype(np.float32))


def _universe(seed=0):
    """族 1:A(0)、B(1) 训练段几乎一样、A 略强;族 2:三个成员;再加一个噪声单点族。"""
    rng = np.random.default_rng(seed)
    y = rng.normal(size=(D, S))
    slow1 = np.cumsum(rng.normal(size=(D, S)) * 0.15, axis=0) + rng.normal(size=(1, S)) * 2.0
    slow2 = np.cumsum(rng.normal(size=(D, S)) * 0.15, axis=0) + rng.normal(size=(1, S)) * 2.0
    base1 = slow1 + 0.6 * y
    base2 = slow2 - 0.5 * y
    a = base1 + 0.05 * rng.normal(size=(D, S))
    b = base1 + 0.4 * rng.normal(size=(D, S))                 # 噪声多一点:训练段适应度略低于 A
    fam2 = [base2 + 0.1 * rng.normal(size=(D, S)) for _ in range(3)]
    noise = rng.normal(size=(D, S))
    panels = [a, b] + fam2 + [noise]
    for p in panels:
        p[rng.random((D, S)) < 0.03] = np.nan                 # 零星缺失,形状在改动前后不变
    return panels, y


def _score_and_pick(d: Path, panels, y, max_families=2, max_members=1):
    from alphacube.metrics import torch_rank_rows

    _write(d, panels)
    np.save(d / "_target.npy", y)
    np.save(d / "_target_ranks.npy", torch_rank_rows(y).numpy())
    np.save(d / "dates.npy", np.array([f"2020{m:02d}{x:02d}" for m in range(1, 13) for x in range(1, 29)][:D]))
    saf.init(d, SPLIT, 0.0015, 1, want_rows=False)
    rows = []
    for i in range(len(panels)):
        _, row, _ = saf.score(i)
        rows.append({k: str(v) for k, v in row.items()})
    rows = [r for r in rows if r["healthy"] == "True"]
    fp = sf.build_train_fingerprints(d / "panels", [int(r["i"]) for r in rows], slice(0, SPLIT), np.arange(S))
    fams = bc.pick_families(rows, np.asarray(fp), 0.7, max_families, max_members)
    return [[int(rows[j]["i"]) for j in f] for f in fams], rows


def _churn_test_rows(panels, which, seed=1):
    """只动测试段:which 里的面板,有值的格子换成每天全新的随机数(缺失位置不变),换手拉满。"""
    rng = np.random.default_rng(seed)
    out = []
    for i, p in enumerate(panels):
        q = p.copy()
        if i in which:
            fin = np.isfinite(q[SPLIT:])
            q[SPLIT:][fin] = rng.normal(size=int(fin.sum()))
        out.append(q)
    return out


ONLY_A = (0,)                       # 只抖族 1 的 A:全样本换手口径下 A 会被 B 顶掉(见最后一个测试)
EVERYTHING = tuple(range(6))


def test_daily_turnover_split_is_the_turnover_of_the_training_rows():
    from alphacube.metrics import rank_ic_turnover, torch_rank_rows

    panels, y = _universe()
    p = panels[0]
    _, days = rank_ic_turnover(p, torch_rank_rows(y), 0.1, daily=True)
    _, full = rank_ic_turnover(p, torch_rank_rows(y), 0.1)
    _, train = rank_ic_turnover(p[:SPLIT], torch_rank_rows(y[:SPLIT]), 0.1)
    assert days.shape == (D,)
    assert abs(days.mean() - full) < 1e-12
    assert abs(days[:SPLIT].mean() - train) < 1e-12


@pytest.mark.parametrize("which", [ONLY_A, EVERYTHING], ids=["only-A", "everything"])
def test_changing_only_test_rows_does_not_change_the_picked_families(tmp_path, which):
    panels, y = _universe()
    before, rows_before = _score_and_pick(tmp_path / "before", panels, y)
    after, rows_after = _score_and_pick(tmp_path / "after", _churn_test_rows(panels, which), y)

    # 前提:测试段确实被改得面目全非(换手),训练段的数一个没动
    for r0, r1 in zip(rows_before, rows_after):
        assert r0["i"] == r1["i"]
        for k in ("rank_ic_train", "ic_train", "turnover_train", "sign"):
            assert r0[k] == r1[k], (r0["i"], k)
        if int(r0["i"]) in which and r0["i"] != "5":           # 噪声因子本来就每天全换
            assert float(r1["turnover_test"]) > float(r0["turnover_test"]) + 0.3, r0["i"]
    # 族 1 选 A,族 2 选一个,噪声单点族被小族门槛挡掉
    assert sorted(f[0] for f in before) in ([0, 2], [0, 3], [0, 4]), before
    assert after == before, (before, after)


def test_the_full_sample_turnover_would_have_changed_the_picks(tmp_path, monkeypatch):
    """同一份测试段改动,换回全样本换手(修之前的口径)就会把族 1 的成员从 A 换成 B:
    B 原样,全样本适应度不变;A 的被测试段换手拖到 B 下面。"""
    monkeypatch.setattr(sf, "TURNOVER", "turnover")
    panels, y = _universe()
    before, _ = _score_and_pick(tmp_path / "before", panels, y)
    after, _ = _score_and_pick(tmp_path / "after", _churn_test_rows(panels, ONLY_A), y)
    fam1 = lambda fams: next(f for f in fams if f[0] in (0, 1))   # noqa: E731
    assert fam1(before) == [0] and fam1(after) == [1], (before, after)


def test_old_metrics_without_train_turnover_are_refused(tmp_path):
    (tmp_path / "metrics.csv").write_text("i,healthy,rank_ic_train,turnover\n0,True,0.05,0.1\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="turnover_train"):
        bc.healthy_rows(tmp_path)
