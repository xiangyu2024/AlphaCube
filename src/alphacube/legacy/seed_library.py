"""Seed alpha libraries (``configs/seed_alpha_library_{nd,dt}.yaml``, extracted
from the user's historical archive by the previous project) and pool seeding."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from ..expression import Expression
from ..parser import ParseError, parse

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def load_seed_library(path: Union[str, Path], n_minutes: int = 241,
                      strict: bool = False) -> Tuple[List[Tuple[str, Expression]], Dict[str, str]]:
    """Parse every entry's ``expression``; returns ([(name, expr)], {name: error})."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    out, errors = [], {}
    for entry in data.get("seed_factors", []):
        name, src = str(entry.get("name", "")), entry["expression"]
        try:
            out.append((name, parse(src, strict=strict, coerce_daily=True, n_minutes=n_minutes)))
        except (ParseError, ValueError) as exc:
            errors[name or src] = str(exc)
    return out, errors


def seed_pool(pool, exprs: Sequence[Expression], engine) -> List[float]:
    """Evaluate ``exprs`` in one engine call and offer them to ``pool`` in order.
    Returns the pool IC after each accepted insertion (rejected ones are skipped)."""
    res = engine.evaluate(list(exprs), tolerant=True)
    rewards = []
    for e, p in zip(exprs, res.panels):
        if p is None:
            continue
        try:
            rewards.append(pool.try_add(e, p.astype(float)))
        except Exception:  # noqa: BLE001 - best-effort seeding
            continue
    return rewards


__all__ = ["load_seed_library", "seed_pool", "CONFIG_DIR"]
