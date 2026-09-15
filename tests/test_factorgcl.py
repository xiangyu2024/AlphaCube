import numpy as np
import pandas as pd
import pytest

from alphacube.combine import FactorGCLCombiner
from alphacube.daily import synthetic_daily

HORIZON = 5


@pytest.fixture(scope="module")
def data():
    return synthetic_daily(n_days=150, n_instruments=8, seed=11)


@pytest.fixture(scope="module")
def target(data):
    return data.forward_returns(HORIZON)


@pytest.fixture
def panels(data):
    """Five hand-built daily factor panels standing in for a mined pool's
    members. The previous project's version of this test evaluated the
    equivalent expressions -- close, |close - open|, ts_std(close, 10),
    cs_rank(volume), ts_delta(close, 5) -- through its AST into a pool; the
    cube engine now does that upstream, and the combiner only ever sees
    (D, S) panels (pairwise |corr| well under the pool's dedup threshold,
    as before)."""
    close, open_, volume = (data.get_field(f) for f in ("Close", "Open", "Volume"))
    close_df = pd.DataFrame(close)
    return [
        close,
        np.abs(close - open_),
        close_df.rolling(10).std().to_numpy(),
        pd.DataFrame(volume).rank(axis=1, pct=True).to_numpy(),
        close_df.diff(5).to_numpy(),
    ]


def test_factorgcl_fit_predict_shapes_and_loss_decreases(panels, target, data):
    combiner = FactorGCLCombiner(
        embed_dim=8, n_hidden_factors=4, max_days_per_epoch=30, lr=5e-3, seed=0
    )
    losses = combiner.fit(panels, target, epochs=6)

    assert len(losses) == 6
    assert all(np.isfinite(loss) for loss in losses)
    assert losses[-1] < losses[0]

    preds = combiner.predict(panels)
    assert preds.shape == (data.n_days, data.n_instruments)
    assert np.all(np.isfinite(preds))


def test_factorgcl_handles_no_panels(target, data):
    combiner = FactorGCLCombiner(embed_dim=8, n_hidden_factors=4, max_days_per_epoch=20)
    losses = combiner.fit([], target, epochs=3)
    assert len(losses) == 3

    preds = combiner.predict([])
    assert preds.shape == (data.n_days, data.n_instruments)
    assert np.all(np.isfinite(preds))


def test_factorgcl_rejects_mismatched_shapes_and_predict_before_fit(panels, target):
    combiner = FactorGCLCombiner(embed_dim=8, n_hidden_factors=4)
    with pytest.raises(RuntimeError):
        combiner.predict(panels)
    with pytest.raises(ValueError):
        combiner.fit(panels, target[:-1])
    combiner.fit(panels, target, epochs=1)
    with pytest.raises(ValueError):
        combiner.predict(panels[:2])
