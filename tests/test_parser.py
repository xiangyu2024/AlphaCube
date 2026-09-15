import pytest

from alphacube.parser import ParseError, parse


def test_aliases_and_macros():
    assert repr(parse("day_mean(close)")) == "day_mean(Close)"
    assert repr(parse("close_1d")) == "day_last(Close)"
    assert repr(parse("vwap_1d")) == "div(day_sum(Amt), day_sum(Volume))"
    assert repr(parse("ret_1d")) == "sub(div(day_last(Close), ts_delay(day_last(Close), 1)), 1)"
    assert repr(parse("abs(day_mean(Ret))")) == "abs_(day_mean(Ret))"


def test_python_operator_syntax():
    assert repr(parse("day_mean(Close) + 1")) == "add(day_mean(Close), 1)"
    assert repr(parse("day_mean(Close) - day_mean(Open)")) == "sub(day_mean(Close), day_mean(Open))"
    assert repr(parse("-day_mean(Close) * 2 / 3")) == "div(mul(neg(day_mean(Close)), 2), 3)"
    assert repr(parse("day_mean(Close) ** 2")) == "pow(day_mean(Close), 2)"
    assert repr(parse("day_mean(Ret) > 0")) == "greater(day_mean(Ret), 0)"
    assert repr(parse("-1.5")) == "-1.5"


@pytest.mark.parametrize("src, msg", [
    ("no_such_op(Close)", "unknown operator"),
    ("day_mean(Nope)", "unknown field"),
    ("ts_mean(day_mean(Close), Close)", "integer window"),
    ("ts_mean(day_mean(Close))", "expects 2 argument"),
    ("day_mean(Close, 5)", "expects 1 argument"),
    ("day_mean(x=Close)", "keyword"),
    ("day_mean(Close", "not a valid expression"),
])
def test_errors(src, msg):
    with pytest.raises(ParseError, match=msg):
        parse(src)
