"""ban exemplar 的秩指纹缓存:每轮开跑不该为了拿指纹把面板重算一遍。

gpu15 装 1049 条 exemplar 花 1 小时,gpu17 装 1673 条花 **2 小时 04 分** —— 一轮
四个多小时里三分之一在重算前几轮已经算过的东西,而且 ban 列表每轮都在长。
"""
from __future__ import annotations

import numpy as np
import pytest

from alphacube.label import Label
from alphacube.mining import GPMiner, MsePool
from alphacube.mining.rankban import ExemplarCache, view_cache_key
from alphacube.parser import parse


def _rows(seed=0, width=12):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(3, width)).astype(np.float32)


def test_round_trip_and_miss(tmp_path):
    c = ExemplarCache(tmp_path, "viewA")
    assert c.get("day_mean(Ret)") is None and c.misses == 1
    rows = _rows()
    c.put("day_mean(Ret)", rows)
    back = c.get("day_mean(Ret)")
    np.testing.assert_array_equal(back, rows)
    assert c.hits == 1


def test_a_new_view_wipes_the_old_one(tmp_path):
    """指纹取决于哪些天、哪些股票 —— 视图一换,旧的永远用不到,不该继续占盘。"""
    old = ExemplarCache(tmp_path, "viewA")
    old.put("x", _rows())
    assert (tmp_path / "viewA").exists()
    new = ExemplarCache(tmp_path, "viewB")
    assert new.dropped_views == 1
    assert not (tmp_path / "viewA").exists()
    assert new.get("x") is None                      # 不会误用旧视图的指纹


def test_prune_drops_the_least_recently_used(tmp_path):
    import os, time

    c = ExemplarCache(tmp_path, "v", max_bytes=2 * (3 * 12 * 4 + 200))
    for i, name in enumerate(("a", "b", "c")):
        c.put(name, _rows(i))
        os.utime(c.dir / c._name(name), (1000 + i, 1000 + i))   # 明确的先后
    assert c.prune() >= 1
    assert c.get("a") is None and c.get("c") is not None          # 最旧的先走


def test_view_key_changes_with_the_view(store):
    v1 = store.view()
    v2 = store.view(start=store.dates[3])
    assert view_cache_key(v1, 250, 512) != view_cache_key(v2, 250, 512)
    assert view_cache_key(v1, 250, 512) != view_cache_key(v1, 250, 256)
    assert view_cache_key(v1, 250, 512) == view_cache_key(store.view(), 250, 512)


def test_second_run_loads_exemplars_without_touching_the_engine(registry, store, tmp_path):
    """第二次构造同一批 exemplar,引擎调用次数必须是 0。"""
    view = store.view()
    target = Label("close", 1).compute(view)
    exprs = [parse(e) for e in ("day_mean(Ret)", "day_max(High)", "day_std(Low)")]
    kw = dict(population_size=6, max_depth=3, seed=1, ban_rank_corr=0.95,
              banned_substructures=("O:log(F:Amt)",), ban_exemplars=exprs,
              exemplar_cache=str(tmp_path), exemplar_cache_gb=1.0)

    first = GPMiner(registry, view, target, MsePool(4, target), **kw)
    n1 = first._add_exemplars(first._pending_exemplars)
    assert n1 >= 1 and first._exemplar_cache.hits == 0

    second = GPMiner(registry, view, target, MsePool(4, target), **kw)
    calls = []
    second.engine.evaluate = lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(
        AssertionError("engine must not be called: the fingerprints are on disk"))
    n2 = second._add_exemplars(second._pending_exemplars)
    assert calls == []
    assert n2 == n1
    assert second.exemplars_equal(first) if hasattr(second, "exemplars_equal") else \
        second._rank_ban.exemplars == first._rank_ban.exemplars
    np.testing.assert_array_equal(second._rank_ban.index._y[:, :second._rank_ban.index.k],
                                  first._rank_ban.index._y[:, :first._rank_ban.index.k])
