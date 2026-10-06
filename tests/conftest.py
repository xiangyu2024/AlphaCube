import numpy as np
import pytest

from alphacube.catalog import default_registry
from alphacube.store import CubeStore
from alphacube.synthetic import make_synthetic_cube, write_daily_fields, write_synthetic_store


@pytest.fixture(scope="session")
def registry():
    return default_registry()


@pytest.fixture(scope="session")
def store(tmp_path_factory):
    """Small synthetic cube with suspended and unlisted stock-days."""
    root = tmp_path_factory.mktemp("cube")
    return write_synthetic_store(root, n_symbols=10, n_days=60, n_minutes=24, seed=11,
                                 suspended_frac=0.05, unlisted_days=8)


@pytest.fixture(scope="session")
def view(store):
    return store.view()


@pytest.fixture(scope="session")
def raw(store):
    """Masked float32 cubes {name: (S, D, M)} and the tradable plane, for references."""
    trad = store.tradable()
    out = {}
    for f in store.fields:
        arr = np.array(store.field(f), dtype=np.float32)
        arr[~trad] = np.nan
        out[f] = arr
    return out, trad


@pytest.fixture(scope="session")
def daily_root(tmp_path_factory, store):
    """A dailyfields.DailyFieldStore directory over the SAME symbols/dates as
    `store` (passed straight from the live store, not re-derived), with
    identifiable -- not random -- planes: see synthetic.make_synthetic_cube's
    daily_fields kwarg and must_fix #2's "identifiable values" graft."""
    _, _, _, _, planes = make_synthetic_cube(n_symbols=store.n_symbols, n_days=store.n_days,
                                              daily_fields=("FloatCap", "TotalCap", "FloatShares"))
    root = tmp_path_factory.mktemp("daily")
    return write_daily_fields(root, planes, store.symbols, store.dates)


@pytest.fixture(scope="session")
def daily_view(store, daily_root):
    """A CubeView of the SAME on-disk cube as `store`/`view`, reopened with
    daily_root attached -- `store` itself is left untouched (and keeps
    `.daily is None`), matching how a read-only production store is reopened."""
    return CubeStore(store.directory, daily_root=daily_root).view()
