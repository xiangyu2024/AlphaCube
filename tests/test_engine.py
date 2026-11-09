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

    # sliding_*_masked_p5 rolls along dim 0, which the engine fills with stock-days,
    # so it is chunk-dependent by construction -- pinned separately as a strict xfail
    # in test_sliding_masked_family_is_chunk_invariant. Excluded here so that this
    # test measures the ENGINE and does not flip to red whenever a vocabulary change
    # reshuffles the draw into one of them.
    broken = {"sliding_top_sum_masked_p5", "sliding_bottom_sum_masked_p5",
              "sliding_top_mean_masked_p5", "sliding_bottom_mean_masked_p5"}
    gen = TypedTreeGenerator(registry, n_minutes=view.n_minutes, seed=9, max_depth=4)
    pop = [e for e in gen.generate_population(200)
           if not any(n.spec is not None and n.spec.name in broken for n in e.nodes())][:150]
    a = CubeEngine(view, chunk_symbols=3, chunk_days=7).evaluate(pop, tolerant=True)
    b = CubeEngine(view).evaluate(pop, tolerant=True)
    assert set(a.errors) == set(b.errors)
    for e, x, y in zip(pop, a.panels, b.panels):
        if x is not None:
            assert np.array_equal(x, y, equal_nan=True), e


@pytest.mark.xfail(strict=True, reason="_sliding_windows rolls along dim 0, which the engine fills with "
                                       "stock-days, so sliding_*_masked_p5 mixes one stock's window into "
                                       "another's and the answer moves with the chunking. Registered dt/in "
                                       "(minute axis) but not computed on it -- see docs/design.md section 5.")
def test_sliding_masked_family_is_chunk_invariant(view):
    """Pinned as a known break rather than left to chance: whether the random
    population in test_chunk_invariance_over_random_population happens to draw one
    of these four operators depends on the vocabulary size, so it silently stopped
    covering them when the trivially-boolean operators left the action set."""
    exprs = [parse(f"day_last({op}(Low, Ret, 10))") for op in
             ("sliding_top_sum_masked_p5", "sliding_bottom_sum_masked_p5",
              "sliding_top_mean_masked_p5", "sliding_bottom_mean_masked_p5")]
    a = CubeEngine(view, chunk_symbols=3, chunk_days=7).evaluate(exprs, tolerant=True)
    b = CubeEngine(view).evaluate(exprs, tolerant=True)
    for e, x, y in zip(exprs, a.panels, b.panels):
        assert x is None or np.array_equal(x, y, equal_nan=True), e


def test_each_operand_is_read_once_per_chunk(view, monkeypatch):
    """A chunk used to re-read a field from the store at EVERY use: one chunk of
    the gpu4 population made 89-104 store reads of 8 distinct fields, and the
    leaf read was the engine's largest cost on the GPU. Now each distinct operand
    is read once per chunk -- and every value equals evaluating that expression
    on its own, where nothing is shared across trees."""
    from collections import Counter
    exprs = [parse(s) for s in ("day_mean(Close)", "day_std(Close)", "day_max(add(Close, Volume))",
                                "day_min(mul(Close, Close))", "day_last(in_rank(Volume))")]
    reads = Counter()
    read_s = view.read_s

    def counted(name, lo, hi):
        reads[(name, lo)] += 1
        return read_s(name, lo, hi)

    monkeypatch.setattr(view, "read_s", counted)
    res = CubeEngine(view, chunk_symbols=3).evaluate(exprs)
    assert [r.mode for r in res.plan.rounds] == [MODE_S]
    n_chunks = -(-view.n_symbols // 3)
    assert sorted(reads) == sorted((f, lo) for f in ("Close", "Volume") for lo in range(0, view.n_symbols, 3))
    assert set(reads.values()) == {1}, "the old engine read Close 5x and Volume 2x per chunk"
    assert sum(reads.values()) == 2 * n_chunks
    for e, p in zip(exprs, res.panels):
        alone = CubeEngine(view, chunk_symbols=3).evaluate([e]).panels[0]
        assert np.array_equal(p, alone, equal_nan=True), e


def test_kernels_never_write_into_operands(registry, tmp_path_factory, monkeypatch):
    """The engine hands ONE tensor to every consumer of an operand in a chunk --
    computed nodes through CSE, store reads since leaves are read once per chunk
    -- so a kernel that wrote into an input would corrupt every later consumer.
    Probe every registered operator (not only the searchable ones): snapshot each
    tensor operand bit-for-bit before the call and compare after it."""
    from alphacube.engine import engine as E
    from alphacube.expression import Feature, make_node
    from alphacube.synthetic import write_synthetic_store

    def bits(t):
        if t.is_floating_point():
            return t.contiguous().view(torch.int32 if t.dtype == torch.float32 else torch.int64).clone()
        return t.clone()

    dirty = []

    def guard(fn):
        def wrap(expr, vals, *rest):
            flat = [t for v in vals for t in (v if isinstance(v, list) else [v]) if isinstance(t, torch.Tensor)]
            before = [bits(t) for t in flat]
            out = fn(expr, vals, *rest)
            if any(not torch.equal(bits(t), b) for t, b in zip(flat, before)):
                dirty.append(expr.spec.name)
            return out
        return wrap

    monkeypatch.setattr(E, "apply_cube", guard(E.apply_cube))
    monkeypatch.setattr(E, "apply_panel", guard(E.apply_panel))
    st = write_synthetic_store(tmp_path_factory.mktemp("purity"), n_symbols=5, n_days=40, n_minutes=241,
                               seed=2, suspended_frac=0.05)
    full, full2 = Feature("Close"), Feature("Volume")
    daily, daily2 = (make_node(registry.get("day_last"), (f,)) for f in (full, full2))
    exprs = []
    for spec in registry.list_all():
        n = 1 if spec.category in ("unary", "rolling") else 2
        args = (daily, daily2)[:n] if spec.needs_daily else (daily,) if spec.restores_m else (full, full2)[:n]
        w = 241 if spec.restores_m else (spec.windows[1] if len(spec.windows) > 1 else
                                         spec.windows[0] if spec.windows else None)
        e = make_node(spec, args, w)
        exprs.append(e if e.shape.degenerate else make_node(registry.get("day_last"), (e,)))
    res = E.CubeEngine(st.view(), chunk_symbols=2, chunk_days=11).evaluate(exprs, tolerant=True)
    assert len(exprs) == len(registry) and not res.errors
    assert not dirty, f"kernels that wrote into an operand: {sorted(set(dirty))}"


def test_boundary_reads_and_writes_follow_the_access_pattern(view, monkeypatch):
    """A boundary file is read by later rounds; in (S, D, M) a D round's read of it
    is the strided crumb-per-stride pattern (~1 KB per 1.9 MB), which is what held
    gpu14's heaviest start-up batch -- four such files. A boundary read only by D
    rounds is therefore stored (D, S, M): its D-round reads are contiguous and the
    strided side becomes the S round's WRITE into a fresh file, which reads nothing.
    The read-ahead advice follows whichever access is strided, is given afresh
    every call (boundaries are new files), and the values never change."""
    from alphacube.engine import engine as engine_mod

    calls = []
    real = engine_mod.advise_access
    monkeypatch.setattr(engine_mod, "advise_access",
                        lambda arr, random: calls.append(bool(random)) or real(arr, random))

    e = parse("day_last(nt_cs_rank(cross_mean(Close, 3)))")   # S round -> boundary -> D round
    eng = CubeEngine(view, chunk_symbols=3, chunk_days=5)
    plan = eng.plan([e])
    assert [r.mode for r in plan.rounds] == [MODE_S, MODE_D] and [b.bound_layout for b in plan.boundaries] == ["D"]
    res = eng.evaluate([e])
    assert list(eng._bound_layout.values()) == ["D"]
    assert calls == [True, False], "strided write from the S round, contiguous read in the D round"
    old = CubeEngine(view, chunk_symbols=3, chunk_days=5, boundary_layout="S")
    calls.clear()
    plain = old.evaluate([e])
    assert calls == [False, True], "the (S, D, M) layout: contiguous write, strided read"
    assert np.array_equal(res.panels[0], plain.panels[0], equal_nan=True)

    # D -> S -> D: a boundary read only by the second D round (day-major), one read by
    # the S round (symbol-major), one written by the S round for the second D round
    # (day-major). Every read is contiguous; each call advises its new files afresh.
    dsd = parse("day_last(nt_cs_rank(add(nt_cs_rank(Close), cross_mean(nt_cs_rank(Open), 3))))")
    eng = CubeEngine(view, chunk_symbols=3, chunk_days=5)
    plan = eng.plan([dsd])
    assert [r.mode for r in plan.rounds] == [MODE_D, MODE_S, MODE_D]
    assert sorted(b.bound_layout for b in plan.boundaries) == ["D", "D", "S"]
    calls.clear()
    first = eng.evaluate([dsd])
    c1 = list(calls)
    calls.clear()
    second = eng.evaluate([dsd])
    assert calls == c1 and c1.count(True) == 2, (c1, calls)
    assert np.array_equal(first.panels[0], second.panels[0], equal_nan=True)


@pytest.mark.parametrize("src", ["day_last(nt_cs_rank(cross_mean(Close, 3)))",
                                 "day_last(nt_cs_rank(add(nt_cs_rank(Close), cross_mean(nt_cs_rank(Open), 3))))",
                                 "day_mean(nt_cs_rank(mul(cross_mean(Vwap, 5), in_rank(Volume))))",
                                 "day_std(nt_ts_mean(cross_max(High, 3), 5))"])
@pytest.mark.parametrize("cs, cd", [(1, 1), (3, 7), (10, 60)])
def test_day_major_boundaries_give_the_same_values(view, src, cs, cd):
    """Storing a boundary (D, S, M) transposes every write and read of it; the
    values must be exactly those of the (S, D, M) layout, at every chunking."""
    e = parse(src)
    a = CubeEngine(view, chunk_symbols=cs, chunk_days=cd).evaluate([e]).panels[0]
    b = CubeEngine(view, chunk_symbols=cs, chunk_days=cd, boundary_layout="S").evaluate([e]).panels[0]
    assert np.array_equal(a, b, equal_nan=True), src


@pytest.mark.parametrize("src", ["day_last(nt_cs_rank(cross_mean(Close, 3)))",
                                 "day_mean(nt_cs_rank(mul(cross_mean(Vwap, 5), in_rank(Volume))))",
                                 "day_std(nt_ts_mean(cross_max(High, 3), 5))"])
@pytest.mark.parametrize("layout", ["auto", "S"])
def test_in_memory_boundaries_give_the_same_values(view, src, layout, monkeypatch, tmp_path):
    """$ALPHACUBE_BOUNDARY_MEMORY=1 keeps boundaries in process memory: the same
    values bit for bit, and no file in the work directory."""
    e = parse(src)
    a = CubeEngine(view, chunk_symbols=3, chunk_days=7, boundary_layout=layout).evaluate([e])
    monkeypatch.setenv("ALPHACUBE_BOUNDARY_MEMORY", "1")
    eng = CubeEngine(view, chunk_symbols=3, chunk_days=7, boundary_layout=layout, workdir=str(tmp_path))
    assert eng.plan([e]).boundaries
    b = eng.evaluate([e])
    assert np.array_equal(a.panels[0], b.panels[0], equal_nan=True), src
    assert b.stats["boundary_bytes"] == a.stats["boundary_bytes"] > 0
    assert list(tmp_path.iterdir()) == []
