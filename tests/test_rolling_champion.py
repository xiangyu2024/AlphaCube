"""两层滚动合成:不看未实现的标签、窗口内和静态 fit_weights 同口径、能跟上 regime shift。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rolling_champion as rc                                 # noqa: E402


def _members(T=160, N=40, seed=0, shift=None):
    """3 族,每族 2~3 个成员;shift 给了就让第 0 族在 shift 之前有效、之后第 1 族有效(regime shift)。"""
    rng = np.random.default_rng(seed)
    base = [rng.normal(size=(T, N)) for _ in range(3)]
    if shift is None:
        target = 0.05 * base[0] + 0.03 * base[1] + rng.normal(size=(T, N))
    else:
        target = rng.normal(size=(T, N))
        target[:shift] += 0.3 * base[0][:shift]
        target[shift:] += 0.3 * base[1][shift:]
    target[rng.random((T, N)) < 0.05] = np.nan
    members = []
    for f, b in enumerate(base):
        mem = [b + rng.normal(0, 0.5, (T, N)) for _ in range(2 + (f == 0))]
        for m in mem:
            m[rng.random((T, N)) < 0.03] = np.nan
        members.append(mem)
    return members, target


def test_window_uses_only_labels_realised_by_the_close():
    assert rc.window(100, 50, 50) == (49, 99)          # 最后一个能用的标签是 t − 2 = 98
    assert rc.window(100, 0, 50) == (0, 99)
    assert rc.window(40, 50, 50) is None
    assert rc.window(100, 50, 250) == (49, 99)         # 固定窗口攒满窗口长度就开始,不受扩张窗口的门槛管
    assert rc.window(100, 0, 250) is None


def test_a_window_that_never_fits_raises_instead_of_returning_nan():
    members, target = _members(T=60)
    with pytest.raises(ValueError):
        rc.rolling_two_level(members, target, 0, min_days=500)


@pytest.mark.parametrize("size,cf", [(40, 0), (0, 0), (40, 3), (0, 4)])
def test_no_signal_moves_when_unrealised_labels_change(size, cf):
    members, target = _members()
    sig, _ = rc.rolling_two_level(members, target, size, step=7, betweens=rc.BETWEEN, min_days=40, crossfit=cf)
    rng = np.random.default_rng(9)
    for d in (60, 83, 120, 150):
        bumped = target.copy()
        bumped[d - 1:] = rng.normal(size=bumped[d - 1:].shape) * 5        # 第 d 天收盘时还没实现的标签
        sig2, _ = rc.rolling_two_level(members, bumped, size, step=7, betweens=rc.BETWEEN, min_days=40, crossfit=cf)
        for b in rc.BETWEEN:
            assert np.array_equal(np.isnan(sig[b][:d + 1]), np.isnan(sig2[b][:d + 1]))
            assert np.allclose(sig[b][:d + 1], sig2[b][:d + 1], equal_nan=True), (b, d)
        # 反过来:改 d − 2 那天(已实现)的标签,后面的信号应该会变 —— 说明测试有鉴别力
        poked = target.copy()
        poked[d - 2] = rng.normal(size=poked.shape[1]) * 50
        sig3, _ = rc.rolling_two_level(members, poked, size, step=1, betweens=("lstsq",), min_days=40)
        sig1, _ = rc.rolling_two_level(members, target, size, step=1, betweens=("lstsq",), min_days=40)
        assert not np.allclose(sig1["lstsq"][d:d + 1], sig3["lstsq"][d:d + 1], equal_nan=True)


def test_window_fit_matches_static_fit_weights():
    from combine_two_level import blend, fit_weights

    members, target = _members()
    t, size = 101, 0
    lo, hi = rc.window(t, size, 40)
    train = slice(lo, hi)
    mom = [rc.moments(m, target) for m in members]
    for (cg, cb), mem in zip(mom, members):
        w_roll = rc.solve(cg[hi] - cg[lo], cb[hi] - cb[lo], "ridge")
        w_static = fit_weights(mem, target, train, "ridge")
        assert np.allclose(w_roll, w_static, rtol=1e-4, atol=1e-8)
    # 两层合在一起:第 t 天的信号 = 在 [lo, hi) 上照静态做法拟合、再用到第 t 天
    sig, hist = rc.rolling_two_level(members, target, size, step=200, betweens=("lstsq",), min_days=t - 1)
    assert hist[0]["t"] == t
    fam = [blend(mem, fit_weights(mem, target, train, "ridge")) for mem in members]
    static = blend(fam, fit_weights(fam, target, train, "lstsq"))
    # fit_weights 把设计矩阵摊成 float32,这里是 float64:差在 1e-7 量级的舍入
    assert np.allclose(sig["lstsq"][t:], static[t:], rtol=1e-4, atol=1e-5, equal_nan=True)
    assert np.isnan(sig["lstsq"][:t]).all()


def test_rolling_follows_a_regime_shift_that_a_static_fit_misses():
    from combine_two_level import blend, fit_weights
    from alphacube.metrics import cs_pearson

    members, target = _members(T=400, N=60, seed=3, shift=200)
    sig, _ = rc.rolling_two_level(members, target, 60, step=10, betweens=("lstsq",), min_days=60)
    train = slice(0, 200)
    fam = [blend(mem, fit_weights(mem, target, train, "ridge")) for mem in members]
    static = blend(fam, fit_weights(fam, target, train, "lstsq"))
    late = slice(300, 400)
    ic_roll = np.nanmean(cs_pearson(sig["lstsq"][late], target[late]))
    ic_static = np.nanmean(cs_pearson(static[late], target[late]))
    assert ic_roll > 0.15 and ic_static < 0.05


def test_rank_ic_weights_carry_sign_and_normalise():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(50, 30))
    w = rc.rank_ic_weights([y + rng.normal(0, 0.3, y.shape), -y + rng.normal(0, 0.3, y.shape)], y)
    assert w[0] > 0 > w[1] and np.isclose(np.abs(w).sum(), 1.0)


def test_unknown_between_method_is_rejected():
    members, target = _members()
    with pytest.raises(ValueError):
        rc.rolling_two_level(members, target, 40, betweens=("gbdt",))


def test_singleton_family_on_z_scale_does_not_take_over_ridge_or_rank_ic():
    """单成员族进族间层时是标准分(标准差约 1),多成员族是收益尺度的预测(标签是日收益,约 1e-3):不统一尺度,
    ridge 的惩罚和 rank IC 加权会被这个弱的单成员族主导(实测 IC 0.275 -> ridge 0.149 / ric 0.059)。"""
    from alphacube.metrics import cs_pearson

    rng = np.random.default_rng(5)
    T, N = 300, 50
    good, weak = rng.normal(size=(T, N)), rng.normal(size=(T, N))
    target = 0.02 * (0.3 * good + 0.05 * weak + rng.normal(size=(T, N)))     # 收益尺度
    members = [[good + rng.normal(0, 0.3, (T, N)) for _ in range(3)], [weak]]
    sig, _ = rc.rolling_two_level(members, target, 100, step=20, betweens=rc.BETWEEN, min_days=100)
    late = slice(150, T)
    for b in rc.BETWEEN:
        ic = np.nanmean(cs_pearson(sig[b][late], target[late]))
        assert ic > 0.25, (b, ic)


def test_crossfit_family_factor_is_out_of_block():
    from combine_two_level import fit_weights

    members, target = _members(T=120)
    mem = members[0]
    cg, cb = rc.moments(mem, target)
    lo, hi, k = 10, 110, 4
    out = rc.crossfit_family(mem, cg, cb, lo, hi, k)
    edges = np.linspace(lo, hi, k + 1).round().astype(int)
    a, b = edges[1], edges[2]
    keep = np.ones(target.shape[0], bool)
    keep[:lo] = keep[hi:] = False
    keep[max(lo, a - rc.LABEL_LAG):min(hi, b + rc.LABEL_LAG)] = False   # 留出块 + 两侧隔离
    days = np.flatnonzero(keep)
    w = fit_weights([m[days] for m in mem], target[days], slice(None), "ridge")
    expect = rc.combine([m[a:b] for m in mem], w)
    assert np.allclose(out[a - lo:b - lo], expect, rtol=1e-4, atol=1e-6, equal_nan=True)


# --------------------------------------------------------------------- oos 模式(样本外堆叠,全市场用)

def _oos(members, target, size, step=7, min_days=40, betweens=rc.BETWEEN, min_between=30):
    Fs, t1 = [], None
    for mem in members:
        f, t1 = rc.oos_family_factor(mem, rc.moments_chunked(mem, target, chunk=17), size, step, min_days)
        Fs.append(f)
    return rc.oos_stack(np.stack(Fs), target, size, t1, step, betweens, min_days, min_between)


def test_chunked_moments_match_the_plain_ones():
    members, target = _members()
    for mem in members:
        a, b = rc.moments(mem, target), rc.moments_chunked(mem, target, chunk=13)
        assert np.allclose(a[0], b[0], rtol=1e-5, atol=1e-3) and np.allclose(a[1], b[1], rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize("size", [40, 0])
def test_oos_mode_never_reads_unrealised_labels(size):
    members, target = _members()
    sig, _ = _oos(members, target, size)
    rng = np.random.default_rng(4)
    for d in (90, 117, 150):
        bumped = target.copy()
        bumped[d - 1:] = rng.normal(size=bumped[d - 1:].shape) * 5
        sig2, _ = _oos(members, bumped, size)
        for b in rc.BETWEEN:
            assert np.array_equal(sig[b][:d + 1], sig2[b][:d + 1], equal_nan=True), (b, d)   # 前缀和:逐位相等
        poked = target.copy()                                  # 正向对照:改 d − 2(已实现)的标签,第 d 天的信号会变
        poked[d - 2] = rng.normal(size=poked.shape[1]) * 50
        sig1, _ = _oos(members, target, size, step=1)
        sig3, _ = _oos(members, poked, size, step=1)
        for b in rc.BETWEEN:
            assert not np.allclose(sig1[b][d], sig3[b][d], equal_nan=True), (b, d)


def test_oos_family_factor_uses_weights_fitted_before_each_day():
    from combine_two_level import blend, fit_weights

    members, target = _members()
    mem = members[0]
    F, t1 = rc.oos_family_factor(mem, rc.moments(mem, target), 40, 7, 40)
    assert t1 == 41 and np.isnan(F[:t1]).all()
    t = t1 + 14                                                # 第 3 次重拟合
    lo, hi = rc.window(t, 40, 40)
    w = fit_weights(mem, target, slice(lo, hi), "ridge")
    assert np.allclose(F[t:t + 7], blend([m[t:t + 7] for m in mem], w), rtol=1e-4, atol=1e-5, equal_nan=True)


def test_oos_stack_singleton_does_not_take_over_and_follows_a_regime_shift():
    from alphacube.metrics import cs_pearson

    rng = np.random.default_rng(6)
    T, N = 420, 50
    a, b, weak = (rng.normal(size=(T, N)) for _ in range(3))
    target = rng.normal(size=(T, N)) + 0.05 * weak
    target[:210] += 0.3 * a[:210]
    target[210:] += 0.3 * b[210:]
    target *= 0.02                                              # 收益尺度
    members = [[a + rng.normal(0, 0.3, (T, N)) for _ in range(3)], [b + rng.normal(0, 0.3, (T, N)) for _ in range(2)],
               [weak]]                                          # 第 3 族是单成员、只有一点信息
    sig, hist = _oos(members, target, 60, step=10, min_days=60, min_between=60)
    late = slice(330, T)
    for m in rc.BETWEEN:
        ic = np.nanmean(cs_pearson(sig[m][late], target[late]))
        assert ic > 0.2, (m, ic)
    w = np.array(hist[-1]["lstsq"])
    assert abs(w[1]) > abs(w[0])                                # 后半段换成第 2 族在起作用


def test_main_runs_end_to_end_on_the_full_market_in_oos_mode(tmp_path):
    import json
    members, target = _members(T=140, N=30)
    d = tmp_path / "scored"
    (d / "panels").mkdir(parents=True)
    fams, i = [], 0
    for mem in members:
        ids = []
        for m in mem:
            np.save(d / "panels" / f"{i:05d}.npy", m.astype(np.float32))
            ids.append(i)
            i += 1
        fams.append({"members": ids, "example": f"f{len(fams)}"})
    np.save(d / "_target.npy", target)
    np.save(d / "dates.npy", np.array([f"2020{k:04d}" for k in range(140)]))
    (tmp_path / "fam.json").write_text(json.dumps(fams), encoding="utf-8")
    out = tmp_path / "roll"
    args = ["--dir", str(d), "--families", str(tmp_path / "fam.json"), "--mode", "oos", "--windows", "20,0",
            "--min-days", "20", "--min-between", "15", "--step", "5", "--out-prefix", str(out)]
    assert rc.main(args) == 0
    res = json.loads((tmp_path / "roll.json").read_text(encoding="utf-8"))
    assert set(res["history"]) == {"oos_w20", "oos_wexp"} and res["chosen_ls"].startswith("oos_")
    sig = np.load(f"{out}_oos_w20_lstsq.npy")
    assert sig.shape == target.shape and np.isfinite(sig[-20:]).any()
    assert not list(tmp_path.glob("roll_F_*.npy"))             # 中间的族因子用完就删了
    win = ["--dir", str(d), "--families", str(tmp_path / "fam.json"), "--windows", "20", "--min-days", "20",
           "--step", "5", "--crossfit", "0", "--out-prefix", str(tmp_path / "win")]
    assert rc.main(win) == 0                                    # 小数据的全市场也能用 window 模式
    import pytest as _pytest
    mp = _pytest.MonkeyPatch()
    mp.setattr(rc, "WINDOW_MODE_BUDGET", 1)
    try:
        with _pytest.raises(SystemExit):                        # 超过内存预算就要求改用 --mode oos
            rc.main(win)
    finally:
        mp.undo()


def test_main_refuses_runs_that_cannot_work(tmp_path):
    import json
    members, target = _members(T=140, N=30)
    d = tmp_path / "scored"
    (d / "panels").mkdir(parents=True)
    np.save(d / "panels" / "00000.npy", members[0][0].astype(np.float32))
    np.save(d / "dates.npy", np.array([f"2020{k:04d}" for k in range(140)]))
    (tmp_path / "fam.json").write_text(json.dumps([{"members": [0]}]), encoding="utf-8")
    base = ["--dir", str(d), "--families", str(tmp_path / "fam.json"), "--out-prefix", str(tmp_path / "o")]
    np.save(d / "_target.npy", np.full(target.shape, np.nan))                  # 标签全空(按别的池子打的分)
    with pytest.raises(SystemExit):
        rc.main(base + ["--mode", "oos"])
    np.save(d / "_target.npy", target)
    with pytest.raises(SystemExit):                                           # 固定窗口短于族间最少天数
        rc.main(base + ["--mode", "oos", "--windows", "20", "--min-between", "50"])
    assert not list(tmp_path.glob("o_F_*.npy"))                               # 先检查,一个族因子都没算
