"""The user's 1498-factor legacy corpus on the three-axis engine: grid
migration, reference inlining, parse rate, evaluation on a synthetic
241-bar cube, and pool seeding from the extracted seed libraries."""

import numpy as np
import pytest

from alphacube.engine.engine import CubeEngine
from alphacube.label import Label
from alphacube.legacy import (CONFIG_DIR, LEGACY_240_FACTORS, legacy_migration_audit, load_legacy_factors,
                              load_seed_library, migrate_factor_expression, seed_pool)
from alphacube.mining.pool import MsePool
from alphacube.research import inline_references, parse_many
from alphacube.synthetic import write_synthetic_store


def test_grid_migration():
    out, audit = migrate_factor_expression("a", "day_autocorr(in_slice(ret,10,229),1)")
    assert out == "day_autocorr(in_slice(ret, 11, 230), 1)" and audit.slice_calls_shifted == 1
    out, audit = migrate_factor_expression("b", "in_roll_labels(close, 240)")
    assert out == "in_roll_labels(close, 241)" and audit.full_day_windows_changed == 1
    out, audit = migrate_factor_expression("c", "day_mean(in_expand(day_mean(close)))")
    assert out == "day_mean(in_expand(day_mean(close)))" and audit.policy == "use_241_aware_operator_without_expression_rewrite"
    audits = legacy_migration_audit()
    assert len(audits) == len(LEGACY_240_FACTORS) == 1498
    assert sum(a.slice_calls_shifted for a in audits) > 1000


def test_reference_inlining():
    src = {"a": "day_mean(ret)", "b": "ts_mean(a, 5)", "c": "div(b, a)"}
    out = inline_references(src)
    assert out["b"] == "ts_mean(day_mean(ret), 5)" and out["c"] == "div(ts_mean(day_mean(ret), 5), day_mean(ret))"


@pytest.fixture(scope="module")
def corpus_parsed():
    factors = load_legacy_factors()
    parsed, errors = parse_many(factors, n_minutes=241)
    return factors, parsed, errors


def test_corpus_parse_rate(corpus_parsed):
    factors, parsed, errors = corpus_parsed
    assert len(parsed) >= 1440                                  # 1442 at port time
    missing_data = sum("unknown field" in m for m in errors.values())
    assert missing_data >= 30                                   # industry / fundamentals / *_1d minute columns
    assert sum("merge_and_forward_fill" in m for m in errors.values()) == 8   # not in the kernel library


def test_corpus_evaluates_on_synthetic_cube(corpus_parsed, tmp_path):
    factors, parsed, _ = corpus_parsed
    store = write_synthetic_store(tmp_path / "cube", n_symbols=5, n_days=45, n_minutes=241, seed=1)
    names = [n for n, e in parsed.items() if e.shape.degenerate]
    res = CubeEngine(store.view(), chunk_symbols=3, chunk_days=20).evaluate([parsed[n] for n in names], tolerant=True)
    ok = [p for p in res.panels if p is not None]
    assert len(ok) >= 1400
    assert sum(np.isfinite(p).mean() > 0.5 for p in ok) >= 1300


def test_seed_libraries_load_and_seed_a_pool(tmp_path):
    dt, dt_err = load_seed_library(CONFIG_DIR / "seed_alpha_library_dt.yaml", n_minutes=24)
    nd, nd_err = load_seed_library(CONFIG_DIR / "seed_alpha_library_nd.yaml", n_minutes=24)
    assert len(dt) >= 40 and len(nd) >= 30
    store = write_synthetic_store(tmp_path / "cube", n_symbols=12, n_days=80, n_minutes=24, seed=4)
    view = store.view()
    target = Label("close", 1).compute(view)
    pool = MsePool(capacity=6, target=target, dedup_threshold=0.999)
    exprs = [e for _, e in nd + dt if e.shape.degenerate][:60]
    rewards = seed_pool(pool, exprs, CubeEngine(view))
    assert pool.size >= 3 and all(np.isfinite(r) for r in rewards)
