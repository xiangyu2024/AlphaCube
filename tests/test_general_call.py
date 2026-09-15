"""Research-only expression features: GeneralCall (literals, lists, three
tensors, width changes), legacy aliases and sugar, daily coercion, lenient
rules -- and their evaluation on the cube."""

import numpy as np
import pytest
import torch

from alphacube import operators as ops
from alphacube.engine.engine import CubeEngine
from alphacube.expression import COLLAPSED, GeneralCall, ShapeError, is_valid_root
from alphacube.fullcatalog import default_full_catalog
from alphacube.parser import ParseError, parse, parse_legacy


def test_full_catalog_classification():
    cat = default_full_catalog()
    g = cat.get
    assert g("select_where").face == "shared" and g("select_where").category == "general"
    assert g("in_slice").face == "dt" and g("in_slice").family == "in" and g("in_slice").compute_axis == "M"
    assert g("day_pdf_skew").reduces_m and g("day_pdf_skew").face == "dt"
    assert g("energy_ratio_topm").reduces_m                      # decorated day-collapsing spectral kernel
    assert g("column_correlation").face == "nd" and g("column_correlation").family == "ts"
    assert g("scale").face == "nd" and g("scale").family == "cs"
    assert g("in_multireg").name == "in_multireg_legacy"         # legacy override before the new api
    assert g("abs").name == "abs_" and g("_max").name == "pairwise_max"
    assert g("rank").kernel_id == "common.rank"                  # 3-arg rank(x, valid_mask, pct) is the cs_rank kernel
    assert g("nt_cs_scale") is not None and g("nt_cs_scale").face == "nt"
    assert g("nope") is None and "hellinger_distance" in cat.ambiguous


def test_general_call_typing_and_keys():
    e = parse_legacy("day_autocorr(in_slice(ret, 10, 229), 1)")
    assert isinstance(e.operands[0], GeneralCall) and e.operands[0].shape.width == 220
    assert e.shape == COLLAPSED and is_valid_root(e)
    one = parse_legacy("day_sum(in_slice(vol, 0, 0))")           # width-1 slice: a real minute, allowed under day_*
    assert one.operands[0].shape.collapsed
    a, b = parse_legacy("in_slice(ret, 0, 19)"), parse_legacy("in_slice(ret, 0, 49)")
    assert a.key() != b.key()                                     # literals are part of the identity
    # no minute operator anywhere -> an ND-style expression: every bare field is its daily aggregate
    assert repr(parse_legacy("select_where(is_gt(ret, 0), close, create_like(close, 0))")) == \
        "select_where(is_gt(sub(div(day_last(Close), ts_delay(day_last(Close), 1)), 1), 0), day_last(Close), create_like(day_last(Close), 0))"
    assert repr(parse_legacy("day_mean(select_where(is_gt(ret, 0), close, create_like(close, 0)))")) == \
        "day_mean(select_where(is_gt(Ret, 0), Close, create_like(Close, 0)))"
    assert repr(parse_legacy("day_pdf_tail_sum_m(ret, vol, 50, False, 'rank')")) == \
        "day_pdf_tail_sum_m(Ret, Volume, 50, False, 'rank')"
    with pytest.raises(ParseError):
        parse("in_mean_rolling(day_mean(Close), 5)", strict=True)
    parse_legacy("day_sum(abs(sub(day_mean(close), day_mean(open))))")   # lenient: day_* over width 1


def test_legacy_sugar_and_aliases():
    assert repr(parse_legacy("(close - open) / open")) == "div(sub(day_last(Close), day_first(Open)), day_first(Open))"
    assert repr(parse_legacy("day_mean((close - open) / open)")) == "day_mean(div(sub(Close, Open), Open))"
    assert repr(parse_legacy("day_mean((ret > 0) | (ret < -0.01))")) == \
        "day_mean(logical_or(greater(Ret, 0), less(Ret, -0.01)))"
    assert repr(parse_legacy("day_sum(is_gt(ret, 0.0).float())")) == "day_sum(is_gt(Ret, 0.0))"
    assert repr(parse_legacy("in_stddev(ret)")) == "in_stddev_full(Ret)"
    assert repr(parse_legacy("in_stddev(ret, 5)")) == "in_stddev_rolling(Ret, 5)"
    assert repr(parse_legacy("sum(close, open)")) == "add(day_last(Close), day_first(Open))"   # ND-style: daily
    assert repr(parse_legacy("day_mean(sum(close, open))")) == "day_mean(add(Close, Open))"
    assert repr(parse_legacy("cross_zscore_rolling(amt, 21)")) == "in_zscore_rolling(Amt, 21)"
    assert repr(parse_legacy("in_expand(day_mean(close))", n_minutes=24)) == "in_expand(day_mean(Close), 24)"
    assert repr(parse_legacy("day_mean(_max(ret, 0.0))")) == "day_mean(pairwise_max(Ret, 0.0))"
    assert repr(parse_legacy("in_mean_rolling_multi([open, close], 5)")).startswith("in_mean_rolling_multi([Open, Close], 5)")
    assert repr(parse_legacy("day_mean(ret >= day_quantile(ret, 0.9))")) == "day_mean(greater_equal(Ret, day_quantile(Ret, 0.9)))"


def test_daily_coercion_for_nd_operators():
    e = parse_legacy("ts_corr(open, vol, 10)")
    assert repr(e) == "ts_corr(day_first(Open), day_sum(Volume), 10)"
    assert repr(parse_legacy("rank(div(close, open))")) == "cs_rank(div(day_last(Close), day_first(Open)))"
    assert repr(parse_legacy("cs_rank(ts_mean(day_std(ret), 5))")) == "cs_rank(ts_mean(day_std(Ret), 5))"
    with pytest.raises(ParseError):
        parse_legacy("cs_rank(in_mean_rolling(close, 5))")        # a genuine minute subtree cannot be coerced
    with pytest.raises(ParseError):
        parse("cs_rank(Close)")                                   # strict parse: no coercion


def test_general_calls_evaluate_exactly(view, raw):
    cubes, trad = raw
    S, D, M = view.n_symbols, view.n_days, view.n_minutes
    exprs = [parse_legacy(s, n_minutes=M) for s in (
        "day_sum(in_slice(ret, 2, 9))",
        "day_mean(select_where(is_gt(ret, 0), close, create_like(close, 0)))",
        "day_last(delete_open_close(close))",
        "cross_mean(day_pdf_skew(ret, vol), 3)",
    )]
    res = CubeEngine(view, chunk_symbols=3).evaluate(exprs, tolerant=False)
    ret, close = _t(cubes["Ret"]), _t(cubes["Close"])
    m = _t(trad)
    r0 = ops.dt_day.day_sum(_m(ops.dt_in.in_slice(ret.reshape(S * D, M), 2, 9).reshape(S, D, -1), m).reshape(S * D, -1)).reshape(S, D)
    assert np.array_equal(np.where(trad, r0.numpy(), np.nan).T, res.panels[0], equal_nan=True)
    cond = _m(ops.common.is_gt(ret.reshape(S * D, M), 0).float().reshape(S, D, M), m)
    sel = _m(ops.common.select_where(cond.reshape(S * D, M), close.reshape(S * D, M),
                                     ops.common.create_like(close.reshape(S * D, M), 0)).reshape(S, D, M), m)
    r1 = ops.dt_day.day_mean(sel.reshape(S * D, M)).reshape(S, D)
    assert np.array_equal(np.where(trad, r1.numpy(), np.nan).T, res.panels[1], equal_nan=True)
    assert all(np.isfinite(p).mean() > 0.5 for p in res.panels[2:])


def test_list_valued_intermediate(view):
    e = parse_legacy("day_last(tensor_list_std(in_multireg([open, high, low], close, 10, 'tvalue')))", n_minutes=view.n_minutes)
    res = CubeEngine(view).evaluate([e], tolerant=False)
    assert np.isfinite(res.panels[0]).mean() > 0.5
    bad = parse_legacy("in_multireg([open, high], close, 10, 'beta')", n_minutes=view.n_minutes)
    assert not bad.shape.degenerate                                # a list root is rejected by the root check
    with pytest.raises(ValueError):
        CubeEngine(view).evaluate([bad])


def _t(a):
    return torch.from_numpy(np.ascontiguousarray(a))


def _m(x, m):
    return torch.where(m.unsqueeze(-1), x, torch.full((), float("nan")))
