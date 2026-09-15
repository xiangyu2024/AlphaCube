import pytest

from alphacube.expression import (COLLAPSED, FULL, WIDE_DEGENERATE, Constant, Feature, ShapeError,
                                  contains_lookahead, is_valid_root, make_node, max_window, replace)
from alphacube.parser import ParseError, parse


@pytest.mark.parametrize("src, shape, root", [
    ("day_std(in_mean_rolling(Close, 30))", COLLAPSED, True),
    ("cs_rank(ts_mean(day_std(Ret), 5))", COLLAPSED, True),
    ("div(day_max(Close), close_1d)", COLLAPSED, True),
    ("day_mean(nt_cs_rank(Close))", COLLAPSED, True),
    ("nt_cs_rank(cross_mean(Close, 3))", FULL, False),
    ("sub(Close, in_expand(day_mean(Close), 241))", FULL, False),
    ("in_expand(day_mean(Close), 241)", WIDE_DEGENERATE, True),     # column 0 is day_mean(Close)
    ("mul(day_mean(Close), 2)", WIDE_DEGENERATE, True),
    ("cross_mean(day_mean(Close), 5)", COLLAPSED, True),
    ("is_gt0(day_mean(Ret))", COLLAPSED, False),        # boolean root
    ("cs_rank(mul(day_mean(Close), 2))", COLLAPSED, True),
    ("Close", FULL, False),
])
def test_shape_typing(src, shape, root):
    e = parse(src)
    assert e.shape == shape
    assert is_valid_root(e) is root


@pytest.mark.parametrize("src", [
    "cs_rank(Close)",                          # ND op on a live minute axis
    "in_mean_rolling(day_mean(Close), 5)",     # in_* on collapsed
    "day_std(day_mean(Close))",                # day on collapsed
    "nt_cs_rank(day_mean(Close))",             # NT on collapsed
    "in_expand(Close, 241)",                   # expand needs width 1
    "in_expand(in_expand(day_mean(Close), 241), 241)",
    "in_rank(in_expand(day_mean(Close), 241))",   # width restored but no information
    "nt_ts_mean(mul(day_mean(Close), 2), 5)",  # wide-degenerate is still degenerate
])
def test_shape_errors(src):
    with pytest.raises(ParseError):
        parse(src)


def test_constant_only_tree_is_not_a_root():
    assert not is_valid_root(parse("sigmoid(5)"))
    assert not is_valid_root(parse("add(1, 2)"))


def test_structural_key_distinguishes_constants_and_windows():
    reg_add = parse("add(Close, 1.0)").spec
    a = make_node(reg_add, (Feature("Close"), Constant(1.0)))
    b = make_node(reg_add, (Feature("Close"), Constant(1.0000001)))
    assert a.key() != b.key()
    assert parse("ts_mean(day_mean(Close), 5)").key() != parse("ts_mean(day_mean(Close), 10)").key()
    assert parse("day_mean(Close)").key() == parse("day_mean(close)").key()
    assert Feature("1").key() != Constant(1.0).key()


def test_repr_round_trip_and_replace():
    e = parse("cs_rank(ts_mean(div(day_std(Ret), day_mean(Volume)), 5))")
    assert parse(repr(e)).key() == e.key()
    target = e.operands[0].operands[0].operands[1]          # day_mean(Volume)
    e2 = replace(e, target, parse("day_sum(Amt)"))
    assert repr(e2) == "cs_rank(ts_mean(div(day_std(Ret), day_sum(Amt)), 5))"
    with pytest.raises(ShapeError):
        replace(e, target, Feature("Close"))                 # div(collapsed, full) -> ts_mean rejects
    assert not contains_lookahead(e)
    assert contains_lookahead(parse("cs_rank(ts_shift(day_mean(Close), 2))"))
    assert max_window(e, "D") == 5 and max_window(parse("day_mean(in_mean_rolling(Close, 30))"), "D") == 0
