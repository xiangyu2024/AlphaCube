import numpy as np
import pytest

from alphacube.models import targets as T


def test_forward_return_from_price_hand_computed():
    # 6 days x 2 instruments
    price = np.array(
        [
            [100.0, 50.0],
            [102.0, 51.0],
            [104.0, 49.0],
            [103.0, 52.0],
            [105.0, 53.0],
            [107.0, 54.0],
        ]
    )
    out = T.forward_return(price, horizon=2, from_returns=False)
    assert out.shape == price.shape

    expected_t0 = [104.0 / 100.0 - 1.0, 49.0 / 50.0 - 1.0]
    expected_t1 = [103.0 / 102.0 - 1.0, 52.0 / 51.0 - 1.0]
    expected_t2 = [105.0 / 104.0 - 1.0, 53.0 / 49.0 - 1.0]
    expected_t3 = [107.0 / 103.0 - 1.0, 54.0 / 52.0 - 1.0]
    np.testing.assert_allclose(out[0], expected_t0)
    np.testing.assert_allclose(out[1], expected_t1)
    np.testing.assert_allclose(out[2], expected_t2)
    np.testing.assert_allclose(out[3], expected_t3)
    # tail: last `horizon` rows NaN, not wrapped
    assert np.isnan(out[4]).all()
    assert np.isnan(out[5]).all()


def test_forward_return_from_returns_matches_price_version():
    price = np.array(
        [
            [100.0, 50.0],
            [102.0, 51.0],
            [104.0, 49.0],
            [103.0, 52.0],
            [105.0, 53.0],
            [107.0, 54.0],
        ]
    )
    daily_returns = np.full_like(price, np.nan)
    daily_returns[1:] = price[1:] / price[:-1] - 1.0

    from_price = T.forward_return(price, horizon=2, from_returns=False)
    from_returns = T.forward_return(daily_returns, horizon=2, from_returns=True)

    # Valid range for horizon=2 on 6 rows is t=0..3
    np.testing.assert_allclose(from_returns[:4], from_price[:4])
    assert np.isnan(from_returns[4:]).all()


def test_forward_return_rejects_bad_horizon():
    price = np.ones((5, 2))
    with pytest.raises(ValueError):
        T.forward_return(price, horizon=0)


def test_forward_volatility_hand_computed():
    returns = np.array([[np.nan], [0.1], [0.2], [-0.1], [0.05]])
    out = T.forward_volatility(returns, horizon=2)
    assert out.shape == returns.shape

    # t=0: window [r1, r2] = [0.1, 0.2] -> mean .15, std = 0.05
    assert out[0, 0] == pytest.approx(0.05, abs=1e-9)
    # t=1: window [r2, r3] = [0.2, -0.1] -> mean .05, std = 0.15
    assert out[1, 0] == pytest.approx(0.15, abs=1e-9)
    # t=2: window [r3, r4] = [-0.1, 0.05] -> mean -.025, std = 0.075
    assert out[2, 0] == pytest.approx(0.075, abs=1e-9)
    # tail (last horizon=2 rows) NaN
    assert np.isnan(out[3:]).all()


def test_forward_downside_volatility_hand_computed():
    returns = np.array([[np.nan], [0.1], [0.2], [-0.1], [0.05]])
    out = T.forward_downside_volatility(returns, horizon=2, mar=0.0)
    assert out.shape == returns.shape

    # t=0: window [0.1, 0.2], nothing below mar=0 -> NaN
    assert np.isnan(out[0, 0])
    # t=1: window [0.2, -0.1], only -0.1 below mar -> std of single value = 0
    assert out[1, 0] == pytest.approx(0.0, abs=1e-9)
    # t=2: window [-0.1, 0.05], only -0.1 below mar -> std of single value = 0
    assert out[2, 0] == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(out[3:]).all()


def test_forward_downside_volatility_multiple_below_mar():
    returns = np.array([[-0.1], [-0.3], [0.4], [np.nan]])
    out = T.forward_downside_volatility(returns, horizon=3, mar=0.0)
    # t=0: window [r1,r2,r3] = [-0.3, 0.4, nan] -> only -0.3 below mar (nan is not < mar)
    # single value below mar -> std 0
    assert out[0, 0] == pytest.approx(0.0, abs=1e-9)


def test_forward_return_rank_hand_computed():
    # horizon=1 trick: forward_return(t) = returns[t+1] exactly (single-step compounding)
    returns = np.array(
        [
            [np.nan, np.nan, np.nan],
            [0.01, 0.05, -0.02],
            [0.03, -0.01, 0.02],
            [-0.05, 0.04, 0.01],
        ]
    )
    out = T.forward_return_rank(returns, horizon=1, pct=True)
    assert out.shape == returns.shape

    np.testing.assert_allclose(out[0], [2 / 3, 3 / 3, 1 / 3])
    np.testing.assert_allclose(out[1], [3 / 3, 1 / 3, 2 / 3])
    np.testing.assert_allclose(out[2], [1 / 3, 3 / 3, 2 / 3])
    assert np.isnan(out[3]).all()

    out_raw = T.forward_return_rank(returns, horizon=1, pct=False)
    np.testing.assert_allclose(out_raw[0], [2.0, 3.0, 1.0])


def test_forward_return_class_hand_computed():
    returns = np.zeros((2, 5))
    returns[1] = [0.01, 0.02, 0.03, 0.04, 0.05]
    out = T.forward_return_class(returns, horizon=1, top_frac=0.4, bottom_frac=0.4)
    # forward_return(t=0) = returns[1] = [0.01, 0.02, 0.03, 0.04, 0.05]
    # bottom_frac=0.4 quantile -> 2.6-scale threshold -> {0.01,0.02} = class 0
    # top_frac=0.4 -> (1-0.4)=0.6 quantile -> {0.04,0.05} = class 2, middle 0.03 = class 1
    np.testing.assert_array_equal(out[0], [0.0, 0.0, 1.0, 2.0, 2.0])
    assert np.isnan(out[1]).all()  # tail row (horizon=1, last row)


def test_forward_return_class_rejects_bad_fracs():
    returns = np.zeros((3, 5))
    with pytest.raises(ValueError):
        T.forward_return_class(returns, horizon=1, top_frac=0.6, bottom_frac=0.6)
    with pytest.raises(ValueError):
        T.forward_return_class(returns, horizon=1, top_frac=1.5, bottom_frac=0.1)


def test_forward_event_label_threshold_hand_computed():
    returns = np.zeros((2, 4))
    returns[1] = [0.01, 0.05, -0.05, 0.04]
    spike = T.forward_event_label(returns, horizon=1, kind="spike", threshold=0.03)
    crash = T.forward_event_label(returns, horizon=1, kind="crash", threshold=0.03)

    np.testing.assert_array_equal(spike[0], [0.0, 1.0, 0.0, 1.0])
    np.testing.assert_array_equal(crash[0], [0.0, 0.0, 1.0, 0.0])
    assert np.isnan(spike[1]).all()
    assert np.isnan(crash[1]).all()


def test_forward_event_label_quantile_hand_computed():
    returns = np.zeros((2, 5))
    returns[1] = [0.01, 0.02, 0.03, 0.04, 0.05]
    spike = T.forward_event_label(returns, horizon=1, kind="spike", quantile=0.8)
    crash = T.forward_event_label(returns, horizon=1, kind="crash", quantile=0.2)

    # quantile(0.8) of [.01..05] linear-interp = 0.042 -> only 0.05 > 0.042
    np.testing.assert_array_equal(spike[0], [0.0, 0.0, 0.0, 0.0, 1.0])
    # quantile(0.2) = 0.018 -> only 0.01 < 0.018
    np.testing.assert_array_equal(crash[0], [1.0, 0.0, 0.0, 0.0, 0.0])


def test_forward_event_label_requires_exactly_one_of_threshold_quantile():
    returns = np.zeros((3, 4))
    with pytest.raises(ValueError):
        T.forward_event_label(returns, horizon=1, kind="spike")
    with pytest.raises(ValueError):
        T.forward_event_label(returns, horizon=1, kind="spike", threshold=0.02, quantile=0.9)


def test_forward_event_label_rejects_bad_kind():
    returns = np.zeros((3, 4))
    with pytest.raises(ValueError):
        T.forward_event_label(returns, horizon=1, kind="bogus", threshold=0.01)


@pytest.mark.parametrize("horizon", [1, 3, 5])
def test_no_lookahead_tail_is_nan_not_wrapped(horizon):
    rng = np.random.default_rng(0)
    n_days, n_instruments = 20, 6
    returns = rng.normal(0.0, 0.02, size=(n_days, n_instruments))
    price = 100.0 * np.cumprod(1.0 + returns, axis=0)

    panels_expect_dense = [
        T.forward_return(price, horizon, from_returns=False),
        T.forward_return(returns, horizon, from_returns=True),
        T.forward_volatility(returns, horizon),
        T.forward_return_rank(returns, horizon),
        T.forward_return_class(returns, horizon),
        T.forward_event_label(returns, horizon, kind="spike", threshold=0.5),
        T.forward_event_label(returns, horizon, kind="crash", threshold=0.5),
    ]
    for panel in panels_expect_dense:
        assert panel.shape == (n_days, n_instruments)
        # Exactly the trailing `horizon` rows are fully NaN...
        assert np.isnan(panel[n_days - horizon :]).all()
        # ...and every row strictly before the tail has at least one finite
        # value (proves we didn't wrap around and NaN-out everything, nor
        # leak a wrapped value into the tail rows).
        for t in range(n_days - horizon):
            assert np.isfinite(panel[t]).any(), f"row {t} unexpectedly all-NaN for horizon={horizon}"

    # forward_downside_volatility can legitimately be NaN on a non-tail row
    # (no returns below `mar` in that window) -- only check the tail here.
    downside = T.forward_downside_volatility(returns, horizon)
    assert downside.shape == (n_days, n_instruments)
    assert np.isnan(downside[n_days - horizon :]).all()
