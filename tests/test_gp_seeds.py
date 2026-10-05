"""GP 用种子(原始手写因子库)做起点:按训练段适应度挑、同骨架只要一个、每块换下一批,
以及种子带进来的窄分钟段(in_slice)在交叉 / 变异时不会和整天宽的子树拼在一起。"""
from __future__ import annotations

import csv
import random

import numpy as np
import pytest

from alphacube.catalog import default_registry
from alphacube.engine.engine import CubeEngine
from alphacube.label import Label
from alphacube.legacy import load_seed_population
from alphacube.mining.clustering import skeleton_key
from alphacube.mining.generator import TypedTreeGenerator
from alphacube.mining.gp import GPMiner
from alphacube.mining.pool import MsePool
from alphacube.synthetic import write_synthetic_store


@pytest.fixture(scope="module")
def cube(tmp_path_factory):
    store = write_synthetic_store(tmp_path_factory.mktemp("cube241"), n_symbols=8, n_days=50, n_minutes=241,
                                  seed=5)
    return store.view()


@pytest.fixture(scope="module")
def seeds():
    out = load_seed_population("legacy", n_minutes=241)
    assert 1350 <= len(out) <= 1430                               # 严格语法:1414 条合法根(移植时)
    return out


def test_seeded_population_takes_the_best_seeds_and_moves_on_each_block(cube, seeds, tmp_path):
    reg = default_registry()
    target = Label("close", 1).compute(cube)
    pool = [e for e in seeds if e.depth() <= 6][:80]
    m = GPMiner(reg, cube, target, MsePool(4, target), population_size=10, min_depth=2, max_depth=4, seed=1,
                initial_population=pool, seed_keep=4, seed_per_block=True, rolling_block=2, ban_top_n=1,
                fitness_kind="rank_ic", turnover_penalty=0.14, checkpoint_dir=tmp_path, checkpoint_every=1)
    every = GPMiner(reg, cube, target, MsePool(4, target), initial_population=seeds)
    assert 1000 <= every.seeds_eligible <= 1150 < every.seeds_offered   # eq / pow / 前视 / 重算子的不收(移植时 1080)
    m.mine(10 * 4)
    used = m.seeds_used
    b1 = [(e, f) for b, e, f in used if b == 1]
    b2 = [(e, f) for b, e, f in used if b == 2]
    assert len(b1) == 4 and b2
    assert [f for _, f in b1] == sorted((f for _, f in b1), reverse=True)
    best = max(f for _, f in m._seed_order + b1 + b2)
    assert b1[0][1] == best                                       # 第一块从最好的种子开始
    assert len({skeleton_key(e) for e, _ in b1}) == 4
    assert not {e.key() for e, _ in b1} & {e.key() for e, _ in b2}    # 下一块换下一批
    archived = set(m._archive)
    assert any(e.key() in archived for e, _ in b1)                 # 种子进了第一代
    rows = list(csv.DictReader(open(tmp_path / "seeds_used.csv", encoding="utf-8")))
    assert len(rows) == len(used)


def test_unseeded_runs_are_unchanged_by_the_seed_plumbing(registry, view):
    target = Label("close", 1).compute(view)
    a = GPMiner(registry, view, target, MsePool(4, target), population_size=8, max_depth=3, seed=4)
    b = GPMiner(registry, view, target, MsePool(4, target), population_size=8, max_depth=3, seed=4,
                initial_population=())
    a.mine(8 * 3)
    b.mine(8 * 3)
    assert list(a._archive) == list(b._archive)


def test_narrow_seed_segments_never_meet_full_width_subtrees(cube, seeds):
    reg = default_registry()
    gen = TypedTreeGenerator(reg, n_minutes=241, min_depth=2, max_depth=4, rng=random.Random(0))
    narrow = [e for e in seeds if any(n.shape.width not in (None, 1) for n in e.nodes())][:40]
    assert narrow
    kids = []
    rng = random.Random(1)
    for _ in range(300):
        s, r = rng.choice(narrow), gen.generate()
        for child in (gen.crossover(s, r), gen.crossover(r, s), gen.crossover(s, rng.choice(narrow)),
                      gen.mutate_subtree(s)):
            if child is not None:
                kids.append(child)
    assert len(kids) > 300
    res = CubeEngine(cube).evaluate(kids, tolerant=True)
    width_errors = [m for m in res.errors.values() if "minute-axis widths" in str(m)]
    assert not width_errors, width_errors[:3]


def test_operator_swap_on_an_off_grid_window(seeds):
    """手写因子的窗口不一定在搜索网格上(ts_mean(x, 21)):换算子时要从新算子的网格里抽一个窗口。
    这条路随机树从来走不到,第一次带种子跑就崩在这里(把窗口整数当成算子去抽)。"""
    from alphacube.parser import parse

    reg = default_registry()
    gen = TypedTreeGenerator(reg, n_minutes=241, min_depth=2, max_depth=4, rng=random.Random(0))
    e = parse("ts_mean(day_std(Ret), 21)", strict=True, coerce_daily=True, n_minutes=241)
    kids = [gen.mutate_operator(e) for _ in range(50)]
    root_swapped = [k for k in kids if k is not None and k.spec.name != "ts_mean"]
    assert root_swapped                                       # 根上换了算子的,窗口落回新算子的网格
    assert all(k.window in k.spec.windows for k in root_swapped if k.spec.requires_window and k.spec.windows)
