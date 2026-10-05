"""mining/scoring.LongHitScorer:命中率和几种做多组合。逐日的 numpy 参照对一遍,方向、门槛、RL 的进池门槛。"""
import numpy as np
import pytest

from alphacube.mining.scoring import HIT_FLOOR, HIT_TAU, HIT_TOPS, LongHitScorer, LongScorer, make_scorer

ALL = {k: 1.0 for k in LongHitScorer.COMPONENTS}


def _data(D=90, S=160, seed=0, noise=1.0, nan=0.03):
    rng = np.random.default_rng(seed)
    x = np.zeros((D, S))
    for t in range(1, D):
        x[t] = 0.9 * x[t - 1] + rng.normal(size=S)
    target = 0.004 * x + 0.02 * rng.normal(size=(D, S)) + 0.01 * rng.normal(size=(D, 1))   # 带市场涨跌
    target[rng.random((D, S)) < 0.02] = np.nan
    sig = x + noise * rng.normal(size=(D, S))
    sig[rng.random((D, S)) < nan] = np.nan
    return sig, target


def _scorer(target, **kw):
    return LongHitScorer(target, weights=ALL, scales=dict(ALL), boundaries=("z", "m"), **kw)


def _sharpe(v):
    v = v[np.isfinite(v)]
    return v.mean() / v.std() * np.sqrt(252), v.mean() * 252


def _reference(sig, target, sign, cost=0.0015):
    """逐日的笨办法。分位 u = (排序位置 + 0.5) / 有值的票数,方向 -1 时取 1 - u。"""
    D, S = sig.shape
    lab = np.isfinite(target)
    r0 = np.where(lab, target, 0.0)
    ew = r0.sum(1) / np.maximum(lab.sum(1), 1)
    names = [f"top{int(round(f * 100))}" for f in HIT_TOPS] + ["rankw", "ramp"]
    W = {k: np.zeros((D, S)) for k in names}
    hits = {f"{h}_{b}": [] for b in "zm" for h in ("hit_top5", "hit_top10", "hit_top20", "hit_ramp", "hit_exp", "auc", "auc_long")}
    for t in range(D):
        fin = np.flatnonzero(np.isfinite(sig[t]))
        n = len(fin)
        if n < 2:
            continue
        order = fin[np.argsort(sig[t, fin], kind="stable")]
        u = (np.arange(n) + 0.5) / n
        if sign < 0:
            u = 1 - u
        books = {}
        for f in HIT_TOPS:
            k = max(1, int(np.floor(n * f)))
            books[f] = order[-k:] if sign > 0 else order[:k]
            W[f"top{int(round(f * 100))}"][t, books[f]] = 1.0 / k
        W["rankw"][t, order] = u / u.sum()
        ramp = np.clip(2 * u - 1, 0, None)
        W["ramp"][t, order] = ramp / ramp.sum()
        if n < 5 or not lab[t].any():
            for k in hits:
                hits[k].append(np.nan)
            continue
        l = lab[t, order]
        for b, cut in (("z", 0.0), ("m", ew[t])):
            win_all = lab[t] & (target[t] > cut)
            base = win_all.sum() / lab[t].sum()
            w = win_all[order]
            for f in HIT_TOPS:
                bl = lab[t, books[f]]
                hits[f"hit_top{int(round(f * 100))}_{b}"].append(win_all[books[f]].sum() / bl.sum() - base if bl.any() else np.nan)
            for name, wt in (("hit_ramp", ramp), ("hit_exp", np.exp(-(1 - u) / HIT_TAU))):
                den = (wt * l).sum()
                hits[f"{name}_{b}"].append((wt * w).sum() / den - base if den > 0 else np.nan)
            lose = l & ~w
            hits[f"auc_{b}"].append(0.5 * (u[w].mean() - u[lose].mean()) if w.any() and lose.any() else np.nan)
            up = u > 0.5
            hits[f"auc_long_{b}"].append(u[w & up].mean() - u[lose & up].mean() if (w & up).any() and (lose & up).any() else np.nan)
    out = {k: float(np.nanmean(v)) for k, v in hits.items()}
    for name, w in W.items():
        traded = np.abs(np.diff(w, axis=0, prepend=np.zeros((1, S)))).sum(1)
        net = (w * r0).sum(1) - cost * traded
        dead = (np.cumsum(w.sum(1) > 0) == 0) | ~lab.any(1)
        net[dead] = np.nan
        out[f"sh_{name}"] = _sharpe(net)[0] - _sharpe(np.where(np.isfinite(net), ew, np.nan))[0]   # 相对同一段日子的基准
        out[f"xs_{name}"], out[f"xa_{name}"] = _sharpe(net - ew)
        out[f"to_{name}"] = 0.5 * traded.mean()
    return out


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_every_component_matches_the_day_by_day_reference(sign):
    sig, target = _data()
    got = _scorer(target).score(sig, orientations=(sign,))
    want = _reference(sig, target, sign)
    assert got["long_sign"] == sign
    for k, v in want.items():
        assert got[k] == pytest.approx(v, rel=1e-9, abs=1e-12), k


def test_top10_book_and_long_half_ic_are_the_first_version_numbers():
    sig, target = _data(seed=1)
    new = _scorer(target).score(sig, orientations=(1.0,))
    old = LongScorer(target).score(sig, orientations=(1.0,))
    for a, b in (("xs_top10", "long_sharpe"), ("xa_top10", "long_ann"), ("to_top10", "long_turnover"),
                 ("lic", "long_ic"), ("lric", "long_rank_ic"), ("long_tie", "long_tie"), ("long_code", "long_code")):
        assert new[a] == pytest.approx(old[b], rel=1e-12), a
    for k in ("ic", "rank_ic", "turnover", "distinct"):
        assert new[k] == pytest.approx(old[k])


def test_orientation_goes_to_the_end_whose_top_book_earns_more_and_flipping_changes_nothing():
    sig, target = _data(seed=2)
    sc = _scorer(target)
    a, b = sc.score(sig), sc.score(-sig)
    assert a["long_sign"] == 1.0 and b["long_sign"] == -1.0
    for k in LongHitScorer.COMPONENTS + ("fit",):
        assert a[k] == pytest.approx(b[k], rel=1e-9, abs=1e-12), k
    assert sc.score_combined(-sig)["long_sign"] == 1.0 and sc.score_combined(-sig)["xs_top10"] < 0 < a["xs_top10"]


def test_a_good_factor_hits_and_noise_does_not():
    sig, target = _data(seed=3, noise=0.3, nan=0.0)
    sc = _scorer(target)
    good = sc.score(sig)
    rng = np.random.default_rng(9)
    junk = sc.score(rng.normal(size=sig.shape), orientations=(1.0,))
    for k in ("hit_top10_z", "hit_top10_m", "hit_ramp_z", "hit_exp_z", "auc_z", "auc_long_z"):
        assert good[k] > 5 * abs(junk[k]) and good[k] > 0.02, k        # 噪声在这么小的样本上有 0.02 左右的抖动
    assert good["hit_top5_z"] > good["hit_top10_z"] > good["hit_top20_z"] > 0      # 越靠头部命中越高
    assert good["to_rankw"] < good["to_ramp"] < good["to_top10"]                    # 摊得越开换手越低


def test_hit_rate_is_measured_against_the_market_base_rate():
    """全市场一起涨的日子头部当然大多是涨的:要减掉当天全市场收益为正的占比。"""
    rng = np.random.default_rng(4)
    D, S = 60, 200
    target = 0.02 * rng.normal(size=(D, S)) + 0.03                                # 天天普涨,约 93% 的票为正
    noise = rng.normal(size=(D, S))
    s = _scorer(target, min_scored_days=0.0).score(noise, orientations=(1.0,))
    assert (target > 0).mean() > 0.9
    for k in ("hit_top10_z", "hit_ramp_z", "hit_exp_z", "hit_top10_m"):
        assert abs(s[k]) < 0.03, k


def test_composite_uses_only_weighted_components_and_gates_still_reject():
    sig, target = _data(seed=5)
    w = {"xs_top10": 1.0, "hit_exp_z": 2.0}
    sc = LongHitScorer(target, weights=w, scales={"xs_top10": 2.0, "hit_exp_z": 0.01}, unit=0.03)
    assert sc.boundaries == ("z",)                                                # 只算合成用得到的分界
    s = sc.score(sig)
    assert "hit_exp_m" not in s
    assert s["fit"] == pytest.approx(0.03 * (s["xs_top10"] / 2.0 + 2.0 * s["hit_exp_z"] / 0.01) / 3.0)
    assert {"long_sign", "xs_top10", "hit_exp_z", "to_top10", "long_tie", "long_code", "distinct"} == set(sc.columns)
    assert sc.score(np.ones_like(sig))["fit"] == -np.inf                          # 常数
    capped = np.clip(sig, np.nanquantile(sig, 0.3, axis=1, keepdims=True), np.nanquantile(sig, 0.7, axis=1, keepdims=True))
    assert sc.score(capped)["fit"] == -np.inf                                     # 两头都并列
    one_side = np.fmin(sig, np.nanquantile(sig, 0.7, axis=1, keepdims=True))
    assert sc.score(one_side)["long_sign"] == -1.0                                # 头部并列:只剩另一头可用
    col = np.tile(np.arange(sig.shape[1], dtype=float), (sig.shape[0], 1))
    assert sc.score(col)["fit"] == -np.inf                                        # 列号
    with pytest.raises(ValueError):
        LongHitScorer(target, weights={"nope": 1.0}, scales={"nope": 1.0})
    with pytest.raises(ValueError):
        LongHitScorer(target, weights={"xs_top10": 1.0}, scales={"xs_top10": 0.0})
    default_scale = LongHitScorer(target, weights={"xs_top10": 1.0})              # 尺度不给就用校准出来的
    assert default_scale.score(sig)["fit"] == pytest.approx(0.03 * s["xs_top10"] / default_scale.scales["xs_top10"])


def test_make_scorer_builds_it_with_the_calibrated_defaults():
    sig, target = _data(seed=6)
    sc = make_scorer(target, "long_hit", cost_per_side=0.002, min_coverage=0.8)
    assert isinstance(sc, LongHitScorer) and sc.cost_per_side == 0.002 and sc.min_coverage == 0.8
    assert sc.weights and set(sc.weights) <= set(sc.scales) and all(w >= 0 for w in sc.weights.values())
    assert np.isfinite(sc.score(sig)["fit"])
    assert isinstance(make_scorer(target, "long"), LongScorer) and not isinstance(make_scorer(target, "long"), LongHitScorer)


def test_rl_pool_refuses_candidates_below_the_solo_bar(view):
    from alphacube.engine.engine import CubeEngine
    from alphacube.label import Label
    from alphacube.mining.env import BatchRewarder
    from alphacube.mining.pool import MsePool
    from alphacube.parser import parse

    target = Label("close", 1).compute(view)
    sc = _scorer(target, min_coverage=0.0, min_scored_days=0.0)
    exprs = [parse("day_std(Ret)"), parse("ts_mean(day_mean(Close), 5)")]
    free = BatchRewarder(CubeEngine(view), MsePool(capacity=4, target=target), scorer=sc)
    solo = [sc.score(np.asarray(p, dtype=np.float64))["fit"] for p in free.panels(exprs)]
    assert all(r.added for r in free.rewards(exprs))                               # 不设门槛:都进池
    bar = (min(solo) + max(solo)) / 2
    pool = MsePool(capacity=4, target=target)
    rew = BatchRewarder(CubeEngine(view), pool, scorer=sc, min_solo=bar)
    out = rew.rewards(exprs)
    lo, hi = int(np.argmin(solo)), int(np.argmax(solo))
    assert out[hi].added and not out[lo].added and "below" in out[lo].reason
    assert out[lo].reward == pytest.approx(max(solo[lo], 0.5 * rew.penalty)) and rew.refused_solo == 1
    assert pool.size == 1
    # 同一条不够格的再来:不能再领一次自己的分数(不然策略会塌到它上面),拿惩罚
    again = rew.rewards([exprs[lo]])[0]
    assert again.reward == rew.penalty and "repeat" in again.reason and not again.added


@pytest.mark.parametrize("miner", ["gp", "alphagen"])
def test_run_mining_end_to_end_with_the_hit_fitness(tmp_path, miner):
    """CLI 走的那条路:long_hit 下 GP 和 RL(带进池门槛)都能跑完、落盘,GP 的存档带这一版的分量。"""
    import csv

    from alphacube.pipeline import run_mining
    from alphacube.synthetic import write_synthetic_store

    st = write_synthetic_store(tmp_path / "cube", n_symbols=40, n_days=70, n_minutes=24, seed=3,
                               suspended_frac=0.03, unlisted_days=4)
    logs = []
    r = run_mining(st, start=st.dates[5], label="close", miner=miner, budget=24, pool_capacity=3,
                   cost_per_side=0.0015, min_coverage=0.8,
                   gp_params=dict(population_size=12, max_depth=3, fitness_kind="long_hit", archive_dedup=0.99,
                                  rl_min_solo=-0.5),
                   combine_methods=("equal", "ridge"), out_root=tmp_path / "runs", run_id="r",
                   checkpoint_every=1, log=logs.append)
    assert (r.run_dir / "meta.json").exists() and (r.run_dir / "members.csv").exists(), logs[-5:]
    if miner == "gp":
        arcs = sorted((r.run_dir / "checkpoints").glob("archive_gen*.csv"))
        rows = list(csv.DictReader(open(arcs[-1], encoding="utf-8")))
        assert rows and {"score_xs_top20", "score_sh_rankw", "score_hit_exp_z", "score_auc_long_z", "score_lric",
                         "score_long_sign"} <= set(rows[0])
        assert all(np.isfinite(float(x["fitness"])) for x in rows)


def test_own_sharpe_is_relative_to_the_benchmark_over_the_same_days():
    """预热期长的因子起算日晚:组合自己的夏普要减同一段日子里基准的夏普,不能白拿另一段行情。"""
    rng = np.random.default_rng(11)
    D, S = 300, 600
    drift = np.where(np.arange(D) < 100, -0.004, 0.003)                       # 先跌 100 天再涨
    market = (drift + 0.012 * rng.normal(size=D))[:, None]                    # 市场自己有 1.2% 的日波动
    target = market + 0.02 * rng.normal(size=(D, S))
    static = np.tile(rng.normal(size=S), (D, 1)) + 1e-6 * rng.normal(size=(D, S))   # 不含信息、几乎不换手
    late = static.copy()
    late[:100] = np.nan                                                        # 躲过了下跌那一段
    sc = _scorer(target, min_coverage=0.0, min_scored_days=0.0)
    a, b = sc.score(static, orientations=(1.0,)), sc.score(late, orientations=(1.0,))
    lab = np.isfinite(target)
    ew = np.where(lab, target, 0).sum(1) / lab.sum(1)
    bonus = _sharpe(ew[100:])[0] - _sharpe(ew)[0]                              # 不减基准的话,晚起算白拿这么多
    assert bonus > 2
    for k in ("sh_top20", "sh_rankw"):
        assert abs(a[k]) < 0.6 and abs(b[k]) < 0.6, k                          # 没有信息:相对基准的夏普在 0 附近
        assert abs(b[k] - a[k]) < 0.25 * bonus, k


def test_composite_floors_each_component():
    sig, target = _data(seed=7)
    sc = LongHitScorer(target, weights={"xs_top20": 1.0, "lric": 1.0})
    comps = {"xs_top20": -50.0, "lric": 0.014}
    assert sc.composite(comps) == pytest.approx(0.03 * (-HIT_FLOOR + 0.014 / sc.scales["lric"]) / 2)
    assert sc.composite({"xs_top20": -5.0, "lric": 0.0}) == pytest.approx(0.03 * (-5.0 / sc.scales["xs_top20"]) / 2)
    assert sc.composite({"xs_top20": float("nan"), "lric": 0.0}) == -np.inf
    with pytest.raises(ValueError):
        LongHitScorer(target, weights={"hit_exp_m": 1.0}, boundaries=("z",))


def test_a_candidate_pruned_on_entry_is_not_paid_the_pool_reward(view):
    from alphacube.engine.engine import CubeEngine
    from alphacube.label import Label
    from alphacube.mining.env import BatchRewarder
    from alphacube.mining.pool import MsePool
    from alphacube.parser import parse

    target = Label("close", 1).compute(view)
    sc = _scorer(target, min_coverage=0.0, min_scored_days=0.0)
    pool = MsePool(capacity=1, target=target)
    rew = BatchRewarder(CubeEngine(view), pool, scorer=sc, min_solo=-10.0)
    exprs = [parse(x) for x in ("day_std(Ret)", "ts_mean(day_mean(Close), 5)", "day_max(High)", "day_mean(Volume)")]
    out = rew.rewards(exprs)
    members = {m.expr.key() for m in pool.members}
    assert pool.size == 1
    for e, r in zip(exprs, out):
        if e.key() in members:
            assert r.added
        elif "pruned" in r.reason:                         # 进去又被自己挤出来:池子没变,不算进池
            assert not r.added
            assert rew.rewards([e])[0].reward == rew.penalty
