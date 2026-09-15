"""Engine correctness: references are computed by calling the raw kernels the
way a single stock / a single day / a single panel would, never through the
engine itself."""

import numpy as np
import pytest
import torch

from alphacube import operators as ops
from alphacube.axes import MODE_D, MODE_P, MODE_S
from alphacube.engine.engine import CubeEngine
from alphacube.parser import parse


def _t(a):
    return torch.from_numpy(np.ascontiguousarray(a))


def _mask_sd(panel_sd, trad):
    out = panel_sd.copy()
    out[~trad] = np.nan
    return out


def test_dt_families_bit_exact_vs_per_symbol(view, raw):
    cubes, trad = raw
    S, D, M = view.n_symbols, view.n_days, view.n_minutes
    exprs = [parse(s) for s in (
        "day_std(in_mean_rolling(Close, 5))",
        "cross_mean(day_max(High), 3)",
        "day_last(cross_corr(Close, Volume, 4))",
        "day_sum(sub(Close, in_expand(day_mean(Close), 24)))",
        "day_mean(in_rank(mul(Ret, Volume)))",
    )]
    res = CubeEngine(view, chunk_symbols=3).evaluate(exprs)
    assert not res.errors and res.plan.summary().startswith("rounds: S[")
    refs = []
    for s in range(S):
        c, h, v, r = (_t(cubes[k][s]) for k in ("Close", "High", "Volume", "Ret"))
        m = _t(trad[s])
        r0 = ops.dt_day.day_std(ops.dt_in.in_mean_rolling(c, 5))
        r1 = ops.dt_cross.cross_mean(_masked(ops.dt_day.day_max(h), m), 3)
        r2 = ops.dt_day.day_last(_masked(ops.dt_cross.cross_corr(c, v, 4), m))
        dm = _masked(ops.dt_day.day_mean(c), m)
        r3 = ops.dt_day.day_sum(_masked(ops.common.sub(c, _masked(ops.dt_in.in_expand(dm, 24), m)), m))
        r4 = ops.dt_day.day_mean(_masked(ops.dt_in.in_rank(_masked(ops.common.mul(r, v), m)), m))
        refs.append([_masked(x, m)[:, 0].numpy() for x in (r0, r1, r2, r3, r4)])
    for i in range(len(exprs)):
        ref = np.stack([refs[s][i] for s in range(S)], axis=1)          # (D, S)
        assert np.array_equal(ref, res.panels[i], equal_nan=True), exprs[i]


def _masked(x, m):
    return torch.where(m.unsqueeze(-1) if x.dim() == 2 else m, x, torch.full((), float("nan")))


def test_nt_cs_exact_vs_per_day(view, raw):
    cubes, trad = raw
    S, D, M = view.n_symbols, view.n_days, view.n_minutes
    e = parse("day_mean(nt_cs_rank(Close))")
    res = CubeEngine(view, chunk_days=7).evaluate([e])
    x = _t(cubes["Close"])
    ranked = torch.empty_like(x)
    for d in range(D):
        ranked[:, d, :] = ops.nd_cs.cs_rank(x[:, d, :].T.contiguous()).T     # rows = minutes, cols = stocks
    ranked = torch.where(_t(trad).unsqueeze(-1), ranked, torch.full((), float("nan")))
    dm = ops.dt_day.day_mean(ranked.reshape(S * D, M)).reshape(S, D)
    ref = np.where(trad, dm.numpy(), np.nan).T
    assert np.array_equal(ref, res.panels[0], equal_nan=True)
    assert [r.mode for r in res.plan.rounds] == [MODE_D]


def test_nt_ts_exact_vs_per_day_face(view, raw):
    """NT = the ND kernel applied to each day's (M, S) face: bit-exact, incl. the
    column-mixing ts_spearman."""
    cubes, trad = raw
    S, D, M = view.n_symbols, view.n_days, view.n_minutes
    exprs = [parse("day_last(nt_ts_mean(Ret, 5))"), parse("day_var(nt_ts_spearman(Ret, 5))")]
    res = CubeEngine(view, chunk_days=7).evaluate(exprs)
    assert [r.mode for r in res.plan.rounds] == [MODE_D]
    x = _t(cubes["Ret"])
    for e, fn, reducer in zip(exprs, (ops.nd_ts.ts_mean, ops.nd_ts.ts_spearman), (ops.dt_day.day_last, ops.dt_day.day_var)):
        cube = torch.empty_like(x)
        for d in range(D):
            cube[:, d, :] = fn(x[:, d, :].T.contiguous(), 5).T           # (M, S) face -> (S, M)
        cube = torch.where(_t(trad).unsqueeze(-1), cube, torch.full((), float("nan")))
        red = reducer(cube.reshape(S * D, M)).reshape(S, D)
        ref = np.where(trad, red.numpy(), np.nan).T
        assert np.array_equal(ref, res.panels[exprs.index(e)], equal_nan=True), e


def test_nd_panel_ops_bit_exact(view, raw):
    cubes, trad = raw
    e = parse("cs_rank(ts_mean(day_std(Ret), 5))")
    res = CubeEngine(view).evaluate([e])
    assert [r.mode for r in res.plan.rounds] == [MODE_S, MODE_P]
    S, D, M = view.n_symbols, view.n_days, view.n_minutes
    ds = ops.dt_day.day_std(_t(cubes["Ret"]).reshape(S * D, M)).reshape(S, D)
    ds = torch.where(_t(trad), ds, torch.full((), float("nan"))).T.contiguous()     # (D, S)
    tm = torch.where(_t(trad).T, ops.nd_ts.ts_mean(ds, 5), torch.full((), float("nan")))
    ref = torch.where(_t(trad).T, ops.nd_cs.cs_rank(tm), torch.full((), float("nan"))).numpy()
    assert np.array_equal(ref, res.panels[0], equal_nan=True)


def test_panel_to_cube_handoff_and_round_structure(view):
    e = parse("day_mean(mul(Close, cs_rank(day_mean(Volume))))")
    eng = CubeEngine(view, chunk_symbols=4)
    plan = eng.plan([e])
    assert [r.mode for r in plan.rounds] == [MODE_S, MODE_P, MODE_S] and not plan.boundaries
    res = eng.evaluate([e])
    ref = CubeEngine(view).evaluate([parse("mul(day_mean(Close), cs_rank(day_mean(Volume)))")]).panels[0]
    assert np.allclose(res.panels[0], ref, equal_nan=True, rtol=1e-5)


def test_boundary_materialisation(view):
    e = parse("day_last(nt_cs_rank(cross_mean(Close, 3)))")
    eng = CubeEngine(view, chunk_symbols=3, chunk_days=5)
    plan = eng.plan([e])
    assert [r.mode for r in plan.rounds] == [MODE_S, MODE_D] and len(plan.boundaries) == 1
    a = eng.evaluate([e])
    b = CubeEngine(view, chunk_symbols=10, chunk_days=60).evaluate([e])
    assert np.array_equal(a.panels[0], b.panels[0], equal_nan=True)
    assert a.stats["boundary_bytes"] == view.n_symbols * view.n_days * view.n_minutes * 4


@pytest.mark.parametrize("cs, cd", [(1, 1), (3, 7), (10, 60)])
def test_chunk_invariance(view, cs, cd):
    exprs = [parse(s) for s in (
        "day_std(in_mean_rolling(Close, 5))", "cs_rank(ts_mean(day_std(Ret), 5))",
        "day_mean(nt_cs_rank(Close))", "day_last(nt_cs_rank(cross_mean(Close, 3)))",
        "ret_1d", "day_mean(nt_ts_mean(Ret, 5))", "sub(day_mean(Close), cs_mean(day_mean(Close)))",
    )]
    a = CubeEngine(view, chunk_symbols=cs, chunk_days=cd).evaluate(exprs)
    b = CubeEngine(view).evaluate(exprs)
    for x, y, e in zip(a.panels, b.panels, exprs):
        assert np.array_equal(x, y, equal_nan=True), e


def test_mask_every_node(view):
    """A suspended stock-day must stay NaN even through is_gt0 (which maps NaN -> 0.0)."""
    res = CubeEngine(view).evaluate([parse("day_sum(is_gt0(Ret))"), parse("day_mean(Close)")])
    dead = ~view.tradable().T
    for p in res.panels:
        assert np.isnan(p[dead]).all()
        assert np.isfinite(p[~dead]).mean() > 0.95
    unmasked = CubeEngine(view, apply_mask=False).evaluate([parse("day_sum(is_gt0(Ret))")]).panels[0]
    assert np.isfinite(unmasked[dead]).all()          # the same op without masking produces fake zeros


def test_tolerant_and_root_checks(view):
    good = parse("day_mean(Close)")
    bad_root = parse("in_mean_rolling(Close, 3)")
    wrong_width = parse("day_sum(sub(Close, in_expand(day_mean(Close), 7)))")   # 7 != 24 minutes
    with pytest.raises(ValueError):
        CubeEngine(view).evaluate([bad_root])
    res = CubeEngine(view).evaluate([good, bad_root, wrong_width], tolerant=True)
    assert res.panels[0] is not None and res.panels[1] is None and res.panels[2] is None
    assert "minute axis" in res.errors[1] and 2 in res.errors


def test_constant_and_degenerate_roots(view):
    res = CubeEngine(view).evaluate([parse("mul(day_mean(Close), 2)"), parse("add(1, 2)")])
    two = CubeEngine(view).evaluate([parse("day_mean(Close)")]).panels[0] * 2
    assert np.allclose(res.panels[0], two, equal_nan=True)
    dead = ~view.tradable().T
    assert np.all(res.panels[1][~dead] == 3.0) and np.isnan(res.panels[1][dead]).all()


def test_chunk_invariance_over_random_population(registry, view):
    """Every grammar-valid tree, incl. multi-round hand-offs (S -> D -> S), is
    bit-identical under any chunking."""
    from alphacube.mining.generator import TypedTreeGenerator

    pop = TypedTreeGenerator(registry, n_minutes=view.n_minutes, seed=9, max_depth=4).generate_population(150)
    a = CubeEngine(view, chunk_symbols=3, chunk_days=7).evaluate(pop, tolerant=True)
    b = CubeEngine(view).evaluate(pop, tolerant=True)
    assert set(a.errors) == set(b.errors)
    for e, x, y in zip(pop, a.panels, b.panels):
        if x is not None:
            assert np.array_equal(x, y, equal_nan=True), e
