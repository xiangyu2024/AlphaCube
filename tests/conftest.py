import numpy as np
import pytest

from alphacube.catalog import default_registry
from alphacube.synthetic import write_synthetic_store


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
