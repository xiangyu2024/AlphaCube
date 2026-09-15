import numpy as np

from alphacube.mining.clustering import cluster_population, cull_clusters, skeleton_key
from alphacube.parser import parse


def test_skeleton_is_leaf_blind_but_window_and_class_aware():
    a, b = parse("ts_mean(add(day_mean(Close), day_mean(Open)), 5)"), parse("ts_mean(add(day_mean(High), day_mean(Low)), 5)")
    assert skeleton_key(a) == skeleton_key(b)
    assert skeleton_key(a) != skeleton_key(parse("ts_mean(add(day_mean(Close), day_mean(Open)), 10)"))
    assert skeleton_key(parse("mul(day_mean(Close), 0.5)")) != skeleton_key(parse("mul(day_mean(Close), 2)"))
    assert skeleton_key(parse("mul(day_mean(Close), 0.5)")) == skeleton_key(parse("mul(day_mean(Open), 1)"))


def test_cluster_population_groups_and_merges_singletons():
    exprs = [parse(s) for s in ("day_mean(Close)", "day_mean(Open)", "day_std(Ret)",
                                "cs_rank(day_mean(Close))", "cs_rank(day_mean(Open))")]
    clusters = cluster_population(exprs, merge_threshold=0.8)
    assert [0, 1, 2] in clusters and [3, 4] in clusters       # singleton day_std joins the day_* family
    assert cluster_population(exprs) == cluster_population(exprs)      # deterministic
    strict = cluster_population(exprs, merge_threshold=1.1)              # no merging
    assert [2] in strict and [0, 1] in strict


def test_cull_rule():
    clusters = [[0, 1], [2, 3], [4]]
    fit = np.array([0.5, 0.4, 0.1, 0.05, 0.6])
    out = cull_clusters(clusters, fit, rel_threshold=0.5)
    assert out[0] == [0, 1] and out[1] == [] and out[2] == [4]
    fit2 = np.array([0.5, 0.4, 0.7, -0.5, np.nan])          # cluster mean 0.1 < cutoff 0.225
    out2 = cull_clusters(clusters, fit2, rel_threshold=0.5)
    assert out2[1] == [2] and out2[2] == []                  # only the member beating 0.45 survives
    assert cull_clusters(clusters, np.full(5, -np.inf)) == [list(c) for c in clusters]
