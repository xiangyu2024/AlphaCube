"""mining/scoring.LongScorer:纯多头口径的奖励。数必须和报告里量的那本书一样,方向取好用的那一头,
并列 / 常数因子拿不到分,GP 和 RL 两条线都接得上;老的 PanelScorer 不受影响。"""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from alphacube.metrics import cs_pearson, ema_rows, long_short_weights, nanmean_quiet, rank_rows  # noqa: E402
from alphacube.mining.scoring import (LONG_COLUMNS, LONG_SCALES, LONG_UNIT, LONG_WEIGHTS, LongScorer,  # noqa: E402
                                      PanelScorer, make_scorer)


def _data(D=120, S=150, seed=0, persistence=0.9, noise=1.0, nan=0.03):
    rng = np.random.default_rng(seed)
    x = np.zeros((D, S))
    for t in range(1, D):
        x[t] = persistence * x[t - 1] + rng.normal(size=S)
    target = 0.01 * x + 0.02 * rng.normal(size=(D, S))
    target[rng.random((D, S)) < 0.02] = np.nan              # 买不进的格子
    sig = x + noise * rng.normal(size=(D, S))
    sig[rng.random((D, S)) < nan] = np.nan
    return sig, target


def _stats(v):
    v = v[np.isfinite(v)]
    return v.mean() / v.std() * np.sqrt(252), v.mean() * 252


def _reference(sig, target, sign, cost, top=0.1, side=0.5):
    """numpy 参照:持仓是多空书的多头腿(方向 -1 是空头腿)那批票,权重和 1;基准是当天所有有标签的票等权;
    从第一次有持仓起每天都算。多头端 IC 逐日在排序位置上取。"""
    w = long_short_weights(sig, top)
    wl = np.clip(w if sign > 0 else -w, 0.0, None) * 2.0
    lab = np.isfinite(target)
    r0 = np.where(lab, target, 0.0)
    ew = np.where(lab.any(1), r0.sum(1) / np.maximum(lab.sum(1), 1), np.nan)
    traded = np.abs(np.diff(wl, axis=0, prepend=np.zeros((1, wl.shape[1])))).sum(1)
    net = (wl * r0).sum(1) - ew - cost * traded
    net[np.cumsum(wl.sum(1) > 0) == 0] = np.nan
    turn = 0.5 * traded
    sharpe, ann = _stats(net)
    ranks, tranks = rank_rows(sig), rank_rows(target)
    ics, rics = [], []
    for t in range(sig.shape[0]):
        fin = np.flatnonzero(np.isfinite(sig[t]))
        order = fin[np.argsort(sig[t, fin], kind="stable")]
        h = max(1, int(np.floor(len(order) * side)))
        sub = order[-h:] if sign > 0 else order[:h]
        mask = np.zeros(sig.shape[1], bool)
        mask[sub] = True
        ics.append(cs_pearson(np.where(mask, sig[t], np.nan)[None], target[t][None])[0])
        rics.append(cs_pearson(np.where(mask, ranks[t], np.nan)[None], tranks[t][None])[0])
    return {"long_sharpe": sharpe, "long_ann": ann, "long_turnover": float(np.nanmean(turn)),
            "long_ic": sign * nanmean_quiet(np.array(ics)), "long_rank_ic": sign * nanmean_quiet(np.array(rics))}


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_components_match_the_numpy_book(sign):
    sig, target = _data()
    sc = LongScorer(target, cost_per_side=0.0015)
    got = sc.score(sign * sig, orientations=(1.0,))          # 固定做多最高的那一头
    want = _reference(sign * sig, target, 1.0, 0.0015)
    for k, v in want.items():
        assert got[k] == pytest.approx(v, rel=1e-9, abs=1e-12), k
    low = sc.score(sig, orientations=(-1.0,))                # 做多最低的那一头
    want = _reference(sig, target, -1.0, 0.0015)
    for k, v in want.items():
        assert low[k] == pytest.approx(v, rel=1e-9, abs=1e-12), k
    assert low["long_sign"] == -1.0 and got["long_sign"] == 1.0


def test_fit_is_the_documented_composite_and_old_columns_are_still_there():
    sig, target = _data(seed=1)
    sc = LongScorer(target)
    s = sc.score(sig)
    want = LONG_UNIT * sum(LONG_WEIGHTS[k] * s[f"long_{k}"] / LONG_SCALES[k] for k in LONG_WEIGHTS) \
        / sum(LONG_WEIGHTS.values())
    assert s["fit"] == pytest.approx(want)
    assert set(LONG_COLUMNS) <= set(s) and {"ic", "rank_ic", "turnover", "fit"} <= set(s)
    old = PanelScorer(target, kind="rank_ic", turnover_penalty=0.14).score(sig)
    for k in ("ic", "rank_ic", "turnover"):                 # 老的三列和老打分器是同一个数
        assert s[k] == pytest.approx(old[k])
    w = LongScorer(target, weights={"sharpe": 1, "ann": 0, "ic": 0, "rank_ic": 0})
    assert w.score(sig)["fit"] == pytest.approx(LONG_UNIT * s["long_sharpe"] / LONG_SCALES["sharpe"])
    assert sc.composite({"sharpe": 1.0, "ann": np.nan, "ic": 0.0, "rank_ic": 0.0}) == -np.inf
    with pytest.raises(ValueError):
        LongScorer(target, weights={"sharp": 1.0})
    with pytest.raises(ValueError):
        LongScorer(target, scales={"ann": 0.0})
    with pytest.raises(ValueError):
        LongScorer(target, cost_per_side=-0.001)


def test_orientation_is_chosen_and_a_flipped_factor_scores_the_same():
    sig, target = _data(seed=2)
    sc = LongScorer(target)
    a, b = sc.score(sig), sc.score(-sig)
    assert a["long_sign"] == 1.0 and b["long_sign"] == -1.0
    for k in ("fit", "long_sharpe", "long_ann", "long_turnover", "long_ic", "long_rank_ic"):
        assert a[k] == pytest.approx(b[k], rel=1e-9), k      # 没有并列时两头是同一批票
    assert a["fit"] > sc.score(sig, orientations=(-1.0,))["fit"]
    # 合成信号的方向是定的:反着的信号在 score_combined 里拿不到"另一头"的分
    assert sc.score_combined(-sig)["fit"] < b["fit"] and sc.score_combined(-sig)["long_sign"] == 1.0
    assert sc.score_combined(sig)["fit"] == pytest.approx(a["fit"])


def test_turnover_is_charged_through_the_book():
    """同一个信号,平滑过的换手低、扣费后更好;不收费时原始的不吃亏。"""
    sig, target = _data(seed=3, noise=3.0, nan=0.0)
    calm = ema_rows(sig, 5.0)
    charged, free = LongScorer(target, cost_per_side=0.0015), LongScorer(target, cost_per_side=0.0)
    a, b = charged.score(sig), charged.score(calm)
    assert b["long_turnover"] < a["long_turnover"] - 0.2
    assert b["fit"] > a["fit"] and b["long_sharpe"] > a["long_sharpe"]
    assert free.score(sig)["long_ann"] == pytest.approx(a["long_ann"] + 0.0015 * 2 * a["long_turnover"] * 252, rel=1e-6)
    assert free.score(sig)["long_ic"] == pytest.approx(a["long_ic"])       # IC 分量不含费用


def test_degenerate_factors_are_rejected_not_scored():
    """常数 / 档位很少的因子:十分位书靠代码顺序破平局,是个零换手的"因子",必须拿 -inf。"""
    sig, target = _data(seed=4)
    sc = LongScorer(target)
    const = np.ones_like(sig)
    flags = (sig > 0).astype(float)
    coarse = np.round(sig)                                   # 十来档
    for bad in (const, flags, coarse):
        assert sc.score(bad)["fit"] == -np.inf
        assert sc.score(bad)["distinct"] < 0.5
    assert np.isfinite(sc.score(sig)["fit"]) and sc.score(sig)["distinct"] == pytest.approx(1.0)
    # 没有这道门槛时,两档的因子是能拿到一个有限的分数的(常数因子也算得出一本书)—— 所以门槛不是摆设
    loose = dict(min_distinct=0.0, max_boundary_ties=1.0, max_code_order=1.0)
    assert np.isfinite(LongScorer(target, **loose).score(flags)["fit"])
    assert np.isfinite(LongScorer(target, min_scored_days=0.0, **loose).score(const)["long_sharpe"])
    assert LongScorer(target, min_distinct=0.0).score(flags)["fit"] == -np.inf       # 并列那道门槛也拦得住
    half = sig.copy()
    half[:, ::2] = np.nan
    assert LongScorer(target, min_coverage=0.9).score(half)["fit"] == -np.inf
    assert np.isfinite(LongScorer(target, min_coverage=0.4).score(half)["fit"])
    # 合成信号不套覆盖率 / 取值档位这些单因子门槛,但并列那道照套(小池子可能把成员并列的那一头拿来做多)
    assert np.isfinite(LongScorer(target, min_coverage=0.9).score_combined(half)["fit"])
    assert sc.score_combined(coarse)["fit"] == -np.inf
    assert sc.score(np.full_like(sig, np.nan))["fit"] == -np.inf


def test_unbuyable_names_earn_nothing_and_leave_the_benchmark():
    D, S = 4, 20
    sig = np.tile(np.arange(S, dtype=float), (D, 1)) + np.linspace(0, 0.5, D)[:, None]   # 每天持 18、19
    target = np.zeros((D, S))
    target[:, 19] = 0.02
    target[1, 19] = np.nan                                   # 这天 19 买不进:记 0,基准分母少一只
    target[:, 0] = 0.01
    sc = LongScorer(target, cost_per_side=0.0, min_scored_days=0.0)
    s = sc.score(sig, orientations=(1.0,))
    days = [0.01 - 0.03 / 20, 0.0 - 0.01 / 19, 0.01 - 0.03 / 20, 0.01 - 0.03 / 20]
    assert s["long_ann"] == pytest.approx(np.mean(days) * 252)
    assert s["long_turnover"] == pytest.approx(0.5 / D)      # 只有第一天建仓


def test_make_scorer_and_the_old_scorer_is_untouched():
    sig, target = _data(seed=5)
    assert isinstance(make_scorer(target, "long", cost_per_side=0.002), LongScorer)
    assert make_scorer(target, "long", cost_per_side=0.002).cost_per_side == 0.002
    old = make_scorer(target, "rank_ic", turnover_penalty=0.14)
    assert isinstance(old, PanelScorer)
    s = old.score(sig)
    assert s["fit"] == pytest.approx(abs(s["rank_ic"]) - 0.14 * s["turnover"]) and set(s) == {"ic", "rank_ic", "turnover", "fit"}
    assert old.score_combined(sig) == old.score(sig, gated=False)
    with pytest.raises(ValueError):
        PanelScorer(target, kind="long")


def test_gp_runs_on_the_long_fitness_and_checkpoints_its_parts(registry, view, tmp_path):
    from alphacube.label import Label
    from alphacube.mining.gp import GPMiner
    from alphacube.mining.pool import MsePool
    from alphacube.parser import parse

    target = Label("close", 1).compute(view)
    m = GPMiner(registry, view, target, MsePool(4, target), seed=0, fitness_kind="long", cost_per_side=0.0015,
                population_size=12, checkpoint_dir=tmp_path, checkpoint_every=1)
    exprs = [parse(s) for s in ("day_std(Ret)", "ts_mean(day_mean(Close), 5)", "sub(0.5, day_max(Amt))")]
    fit = m.fitness(exprs)
    ref = LongScorer(target, cost_per_side=0.0015, min_coverage=m.min_coverage, min_scored_days=m.min_scored_days)
    for e, p, f in zip(exprs, m._panels(exprs), fit):
        assert f == pytest.approx(ref.score(np.asarray(p))["fit"]), e
    rep = m.mine(24)
    assert rep.generations >= 1 and np.isfinite(rep.best_fitness)
    rows = list(csv.DictReader(open(sorted(tmp_path.glob("archive_gen*.csv"))[-1], encoding="utf-8")))
    assert rows and {f"score_{c}" for c in LONG_COLUMNS} <= set(rows[0]) and "score_rank_ic" in rows[0]
    assert rows[0]["score_long_sign"] in ("1.000000", "-1.000000")
    old = GPMiner(registry, view, target, MsePool(4, target), seed=0, fitness_kind="rank_ic", turnover_penalty=0.14,
                  checkpoint_dir=tmp_path / "old", checkpoint_every=1)
    old.mine(24)
    head = open(sorted((tmp_path / "old").glob("archive_gen*.csv"))[-1], encoding="utf-8").readline()
    assert "score_long" not in head and head.startswith("rank,fitness,score_ic,score_rank_ic,score_turnover,dups,")


def test_rl_rewarder_uses_the_long_fitness_for_candidate_and_pool(view):
    from alphacube.engine.engine import CubeEngine
    from alphacube.label import Label
    from alphacube.mining.env import BatchRewarder
    from alphacube.mining.pool import MsePool
    from alphacube.parser import parse

    target = Label("close", 1).compute(view)
    scorer = LongScorer(target, cost_per_side=0.0015, min_coverage=0.0, min_scored_days=0.0)
    pool = MsePool(capacity=4, target=target)
    rew = BatchRewarder(CubeEngine(view), pool, scorer=scorer)
    out = rew.rewards([parse("day_std(Ret)"), parse("ts_mean(day_mean(Close), 5)")])
    added = [r for r in out if r.added]
    assert added, [r.reason for r in out]
    assert added[-1].reward == pytest.approx(scorer.score_combined(pool.combined())["fit"])
    assert all(np.isfinite(r.reward) for r in out)

    class Reject(LongScorer):
        def score(self, panel, gated=True, orientations=(1.0, -1.0)):
            return {**super().score(panel, gated, orientations), "fit": -np.inf}

    rew2 = BatchRewarder(CubeEngine(view), MsePool(capacity=4, target=target), scorer=Reject(target))
    bad = rew2.rewards([parse("day_std(Ret)")])[0]
    assert bad.reward == rew2.penalty and "rejected" in bad.reason


def test_a_book_boundary_inside_a_tie_block_is_rejected_for_that_side_only():
    """头部被截平的因子取值不少(过得了 min_distinct),但前 10% 是从一段并列里按代码顺序挑的 —— 那一头不算分。"""
    sig, target = _data(seed=6, nan=0.0)
    sc = LongScorer(target)
    capped = np.minimum(sig, np.quantile(sig, 0.7, axis=1, keepdims=True))     # 最高的 30% 并列
    s = sc.score(capped)
    assert s["distinct"] > 0.5
    top = sc.score(capped, orientations=(1.0,))
    assert top["fit"] == -np.inf and top["long_tie"] == pytest.approx(1.0)
    assert s["long_sign"] == -1.0 and np.isfinite(s["fit"]) and s["long_tie"] == 0.0   # 另一头没有并列,照常打分
    both = np.clip(sig, np.quantile(sig, 0.3, axis=1, keepdims=True), np.quantile(sig, 0.7, axis=1, keepdims=True))
    assert LongScorer(target, min_distinct=0.0).score(both)["fit"] == -np.inf          # 两头都截平:没有可用的方向
    assert sc.score_combined(capped)["fit"] == -np.inf                                 # 合成信号也不能做多并列的那一头
    loose = LongScorer(target, max_boundary_ties=1.0, max_code_order=1.0)          # 并列那段是按代码排的,两道都得放开
    assert np.isfinite(loose.score(capped, orientations=(1.0,))["fit"])
    # 书里有并列、但并列没有跨出书的边界:哪些票进书是确定的,不算
    D, S = sig.shape
    exact = sig.copy()
    k = int(np.floor(S * 0.1))
    order = np.argsort(sig, axis=1)
    np.put_along_axis(exact, order[:, -k:], 99.0, axis=1)                              # 恰好最高的 k 只并列
    assert sc.score(exact, orientations=(1.0,))["long_tie"] == 0.0
    assert sc.score(sig)["long_tie"] == 0.0


@pytest.mark.parametrize("miner", ["gp", "alphagen"])
def test_run_mining_end_to_end_with_long_fitness(tmp_path, miner):
    """CLI 走的那条路(run_mining):--fitness long 下 GP 和 RL 都能跑完、落盘,GP 的存档带纯多头的分量。"""
    from alphacube.pipeline import run_mining
    from alphacube.synthetic import write_synthetic_store

    st = write_synthetic_store(tmp_path / "cube", n_symbols=40, n_days=70, n_minutes=24, seed=3,
                               suspended_frac=0.03, unlisted_days=4)
    logs = []
    r = run_mining(st, start=st.dates[5], label="close", miner=miner, budget=24, pool_capacity=3,
                   cost_per_side=0.0015, min_coverage=0.8,
                   gp_params=dict(population_size=12, max_depth=3, fitness_kind="long", archive_dedup=0.99),
                   combine_methods=("equal", "ridge"), out_root=tmp_path / "runs", run_id="r",
                   checkpoint_every=1, log=logs.append)
    assert (r.run_dir / "meta.json").exists() and (r.run_dir / "members.csv").exists(), logs[-5:]
    if miner == "gp":
        arcs = sorted((r.run_dir / "checkpoints").glob("archive_gen*.csv"))
        rows = list(csv.DictReader(open(arcs[-1], encoding="utf-8")))
        assert rows and "score_long_sharpe" in rows[0] and "score_long_sign" in rows[0]
        assert all(np.isfinite(float(x["fitness"])) for x in rows)


def test_benchmark_is_the_same_for_every_candidate():
    """基准是当天所有有标签的票,不是"这个因子有值的票":在自己不持有的票上给 NaN 不该改变这本书的成绩。"""
    sig, target = _data(seed=7, nan=0.0)
    sc = LongScorer(target, min_coverage=0.5)
    full = sc.score(sig, orientations=(1.0,))
    hollow = sig.copy()
    lo, hi = np.quantile(sig, 0.55, axis=1, keepdims=True), np.quantile(sig, 0.8, axis=1, keepdims=True)
    hollow[(sig > lo) & (sig < hi)] = np.nan                 # 挖掉 55%~80% 分位:前 10% 的持仓数变了,先对齐
    k = np.floor(np.isfinite(hollow).sum(1) * 0.1).astype(int)
    assert (k < np.floor(sig.shape[1] * 0.1)).all()
    # 同一批持仓才好比:把被挖掉的票换成一个很小的数(排到最后),持仓和原来一样
    sunk = np.where(np.isfinite(hollow), sig, -1e9)
    a, b = sc.score(sunk, orientations=(1.0,)), full
    for key in ("long_sharpe", "long_ann", "long_turnover"):
        assert a[key] == pytest.approx(b[key]), key
    # 而基准不随覆盖变:同样的持仓权重下,NaN 掉的票不再进基准的话 long_ann 会变 —— 这里直接验基准
    lab = np.isfinite(target)
    ew = np.where(lab, target, 0).sum(1) / lab.sum(1)
    assert np.allclose(sc._ew_all.numpy(), ew)


def test_days_without_factor_values_are_cash_and_the_exit_is_paid():
    sig, target = _data(seed=8, nan=0.0)
    gap = sig.copy()
    gap[40:45] = np.nan                                      # 5 天整天没值:清仓、空仓跑输基准、再建仓
    sc = LongScorer(target, cost_per_side=0.0015)
    b = sc.score(gap, orientations=(1.0,))
    want = _reference(gap, target, 1.0, 0.0015)
    dropped = _reference(np.delete(gap, np.s_[40:45], axis=0), np.delete(target, np.s_[40:45], axis=0), 1.0, 0.0015)
    assert abs(want["long_ann"] - dropped["long_ann"]) > 1e-4          # 空仓那几天算不算,数是不一样的
    for key, v in want.items():
        if key in ("long_sharpe", "long_ann", "long_turnover"):
            assert b[key] == pytest.approx(v, rel=1e-9), key
    late = sig.copy()
    late[:30] = np.nan                                       # 开头的预热期不算空仓:从第一次有持仓起算
    c = sc.score(late, orientations=(1.0,))
    assert c["long_ann"] == pytest.approx(_reference(late, target, 1.0, 0.0015)["long_ann"], rel=1e-9)
    assert c["long_ann"] == pytest.approx(_reference(sig[30:], target[30:], 1.0, 0.0015)["long_ann"], rel=1e-9)


def test_code_order_books_are_rejected():
    """cs_rank 按代码顺序破平局:吐出来的值互不相同,取值档位和并列两道门槛都看不见,得另外拦。"""
    sig, target = _data(seed=9, nan=0.0)
    D, S = sig.shape
    sc = LongScorer(target)
    col = np.tile(np.arange(S, dtype=float), (D, 1))         # cs_rank(截面常数) = 列号
    s = sc.score(col)
    assert s["distinct"] == pytest.approx(1.0) and s["fit"] == -np.inf
    assert abs(sc.score(col, orientations=(1.0,))["long_code"]) > 0.99
    # 头部 45% 并列、再被 cs_rank 按代码顺序拉开:做多最高那头的书是按代码挑的
    capped = np.minimum(sig, np.quantile(sig, 0.55, axis=1, keepdims=True))
    ranked = np.argsort(np.argsort(capped, axis=1, kind="stable"), axis=1, kind="stable").astype(float)
    top = sc.score(ranked, orientations=(1.0,))
    assert top["long_tie"] == 0.0 and top["long_code"] > 0.9 and top["fit"] == -np.inf
    low = sc.score(ranked, orientations=(-1.0,))             # 另一头是真因子,照常打分
    assert abs(low["long_code"]) < 0.2 and np.isfinite(low["fit"])
    honest = sc.score(sig)
    assert abs(honest["long_code"]) < 0.2 and np.isfinite(honest["fit"])
    assert np.isfinite(LongScorer(target, max_code_order=1.0).score(col, orientations=(1.0,))["fit"])


def test_cli_refuses_long_fitness_without_a_cost(tmp_path):
    from alphacube.cli import build_parser, cmd_mine

    a = build_parser().parse_args(["mine", "--store", str(tmp_path), "--fitness", "long"])
    with pytest.raises(SystemExit):
        cmd_mine(a)


def test_rl_rewarder_scores_each_expression_once(view):
    from alphacube.engine.engine import CubeEngine
    from alphacube.label import Label
    from alphacube.mining.env import BatchRewarder
    from alphacube.mining.pool import MsePool
    from alphacube.parser import parse

    target = Label("close", 1).compute(view)
    calls = []

    class Counting(LongScorer):
        def score(self, panel, gated=True, orientations=(1.0, -1.0)):
            calls.append(gated)
            return super().score(panel, gated, orientations)

    rew = BatchRewarder(CubeEngine(view), MsePool(capacity=4, target=target),
                        scorer=Counting(target, min_coverage=0.0, min_scored_days=0.0))
    e = parse("day_std(Ret)")
    rew.rewards([e])
    solo = sum(1 for g in calls if g)
    rew.rewards([e, e])                                      # 同一条再来两次:不再重新给候选打分
    assert sum(1 for g in calls if g) == solo == 1
