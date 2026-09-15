"""The non-CPU path of the engine, and the float64 reference mode that makes
its numbers readable.

The device tests run against whatever accelerator torch can see (CUDA on the
server, MPS on a Mac) and skip when there is none, so the same file is the
local shake-out and the server's acceptance test. What they lock down is the
part that is device-INdependent by construction: no kernel may fail only on the
accelerator, the finite/NaN pattern may not move, and selection operators
(``day_last`` and friends, which pick an element rather than reduce) must be bit
identical. Reduction order is explicitly not promised across devices -- see
docs/design.md section 5 -- so values are compared through the float64
reference instead of against each other.
"""

import numpy as np
import pytest
import torch

from alphacube.engine.engine import CubeEngine
from alphacube.expression import Feature, make_node
from alphacube.parser import parse

ACCEL = ("cuda" if torch.cuda.is_available()
         else "mps" if torch.backends.mps.is_available() else None)
needs_accel = pytest.mark.skipif(ACCEL is None, reason="no CUDA or MPS device visible")

EXPRS = ("day_std(in_mean_rolling(Close, 5))", "cs_rank(day_mean(Close))",
         "day_last(nt_cs_mean(Ret))", "cross_mean(day_max(High), 3)",
         "day_sum(sub(Close, in_expand(day_mean(Close), 24)))",
         "ts_mean(day_last(Close), 5)")
# Everything in EXPRS except the deviations-from-the-mean one, whose exact value
# is zero: what survives in float32 is rounding noise, so the two devices'
# copies of it are uncorrelated and any relative comparison of them is
# meaningless. Its finite/NaN pattern is still checked.
VALUE_EXPRS = tuple(e for e in EXPRS if not e.startswith("day_sum(sub("))


# ---- float64 reference mode ---------------------------------------------------

def test_float64_mode_widens_the_whole_path(view):
    """Leaves, boundaries and panels all widen, and the answer moves in the
    place it should: a cancellation-prone kernel, not a selection."""
    exprs = [parse(s) for s in ("day_last(in_zscore_rolling(Close, 5))", "day_last(Close)")]
    r32 = CubeEngine(view, dtype="float32").evaluate(exprs)
    r64 = CubeEngine(view, dtype="float64").evaluate(exprs)
    assert r32.panels[0].dtype == np.float32 and r64.panels[0].dtype == np.float64
    assert not np.array_equal(r32.panels[0], r64.panels[0], equal_nan=True)   # z-score cancels
    assert np.array_equal(r32.panels[1], r64.panels[1].astype(np.float32), equal_nan=True)


def test_float64_mode_keeps_the_nan_pattern_and_chunk_invariance(view):
    exprs = [parse(s) for s in EXPRS]
    a = CubeEngine(view, dtype="float64", chunk_symbols=3, chunk_days=7).evaluate(exprs)
    b = CubeEngine(view, dtype="float64", chunk_symbols=10, chunk_days=60).evaluate(exprs)
    ref32 = CubeEngine(view, dtype="float32").evaluate(exprs)
    for x, y, z in zip(a.panels, b.panels, ref32.panels):
        assert np.array_equal(x, y, equal_nan=True)                  # chunk-invariant in float64 too
        assert np.array_equal(np.isfinite(x), np.isfinite(z))        # same entries are finite as float32


def test_float64_rejects_nonsense_dtype(view):
    with pytest.raises(ValueError):
        CubeEngine(view, dtype="float16")


# ---- accelerator parity -------------------------------------------------------

@needs_accel
def test_no_operator_fails_only_on_the_accelerator(registry, view):
    """A device-mismatch bug (a tensor built without ``device=``, a ``.numpy()``
    on a device tensor) shows up as an operator that raises only on the GPU."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from validate_gpu_parity import build_probe_expressions

    exprs, names = build_probe_expressions(registry)
    cpu = CubeEngine(view, device="cpu").evaluate(exprs, tolerant=True)
    acc = CubeEngine(view, device=ACCEL).evaluate(exprs, tolerant=True)
    only_accel = {names[i]: acc.errors[i] for i in acc.errors if i not in cpu.errors}
    assert not only_accel, f"failed only on {ACCEL}: {only_accel}"


@needs_accel
def test_accelerator_keeps_the_finite_pattern(view):
    """Which stock-days come out finite is a property of the mask and the data,
    not of the arithmetic, so it must not move at all."""
    exprs = [parse(s) for s in EXPRS]
    cpu = CubeEngine(view, device="cpu").evaluate(exprs)
    acc = CubeEngine(view, device=ACCEL).evaluate(exprs)
    for e, x, y in zip(EXPRS, cpu.panels, acc.panels):
        assert np.array_equal(np.isfinite(x), np.isfinite(y)), e


@needs_accel
def test_accelerator_values_agree_to_the_panel_scale(view):
    """Values are compared against the panel's own scale rather than elementwise:
    a near-zero entry sitting next to values of order 1 carries no information
    about whether the device computed it correctly."""
    exprs = [parse(s) for s in VALUE_EXPRS]
    cpu = CubeEngine(view, device="cpu").evaluate(exprs)
    acc = CubeEngine(view, device=ACCEL).evaluate(exprs)
    for e, x, y in zip(VALUE_EXPRS, cpu.panels, acc.panels):
        m = np.isfinite(x) & np.isfinite(y)
        if not m.any():
            continue
        scale = float(np.percentile(np.abs(y[m]), 99)) or 1.0
        assert float(np.abs(x[m] - y[m]).max()) <= 1e-4 * scale, e


@needs_accel
@pytest.mark.parametrize("expr", ["day_last(Close)", "day_first(Volume)", "day_max(High)",
                                  "day_min(Low)", "in_expand(day_last(Close), 24)"])
def test_selection_operators_are_bit_identical_across_devices(view, expr):
    """Selections pick an element instead of summing one, so nothing about
    reduction order can move them; the parity script relies on exactly this to
    feed both devices identical operands."""
    e = [parse(expr)]
    cpu = CubeEngine(view, device="cpu").evaluate(e)
    acc = CubeEngine(view, device=ACCEL).evaluate(e)
    assert np.array_equal(cpu.panels[0], acc.panels[0], equal_nan=True)


@needs_accel
def test_accelerator_runs_a_gp_generation(registry, view):
    from alphacube.label import Label
    from alphacube.mining.gp import GPMiner
    from alphacube.mining.pool import MsePool

    target = Label("close", 1).compute(view)
    miner = GPMiner(registry, view, target, MsePool(4, target), population_size=8,
                    max_depth=3, device=ACCEL, seed=0)
    report = miner.mine(16)
    assert report.evaluated > 0


def test_rank_breaks_ties_by_index():
    """``common.rank`` sorts with ``stable=True``, so tied values are ranked in
    index order -- the one tie-break every torch backend agrees on, and the same
    convention as pandas' ``method="first"``.

    Without it ``argsort`` is free to permute ties however the backend likes, and
    tied ranks came back swapped on the GPU: a 1/n difference, not an ULP one.
    Minute data ties constantly (prices sit on a tick grid, volumes repeat,
    suspended minutes are zero), and ranks run through ``in_rank`` / ``cs_rank`` /
    ``nt_cs_rank`` and the Spearman kernels, so this has to hold. See
    docs/design.md section 5."""
    from alphacube.operators.common import rank

    x = torch.tensor([[1.0, 2.0, 2.0, 2.0, 3.0, 1.0, 5.0, 5.0]])
    got = rank(x, pct=False)[0].tolist()
    assert got == [1.0, 3.0, 4.0, 5.0, 6.0, 2.0, 7.0, 8.0]      # ties ascend by column
    assert rank(x.flip(-1), pct=False)[0].flip(-1).tolist() != got   # and it is position, not value


@needs_accel
def test_rank_agrees_across_devices_on_tied_data():
    """The reason the stable sort is there. Ranks a tick-grid panel -- ties in
    every row -- and requires both backends to produce the same ranks."""
    from alphacube.operators.common import rank

    g = torch.Generator().manual_seed(17)
    x = (torch.randint(0, 40, (256, 241), generator=g).float() / 4.0)
    assert (x.shape[-1] - torch.tensor([len(np.unique(r)) for r in x.numpy()])).max() > 0
    assert torch.equal(rank(x), rank(x.to(ACCEL)).cpu())


@needs_accel
def test_rank_expressions_agree_across_devices_through_the_engine(view):
    """Collapsed with ``day_last`` rather than ``day_mean``: a mean over the minute
    axis is a reduction, and reduction order is not promised across devices, so it
    would put back exactly the noise this test is trying to exclude."""
    exprs = [parse(s) for s in ("day_last(in_rank(Volume))", "cs_rank(day_last(Close))",
                                "day_last(nt_cs_rank(Ret))", "day_max(in_rank(Close))")]
    cpu = CubeEngine(view, device="cpu").evaluate(exprs)
    acc = CubeEngine(view, device=ACCEL).evaluate(exprs)
    for e, x, y in zip(exprs, cpu.panels, acc.panels):
        assert np.array_equal(x, y, equal_nan=True), e
