"""Legacy factor corpus and seed libraries on the three-axis engine."""

from __future__ import annotations

from typing import Dict, List, Tuple

from .factor_set_merged import LEGACY_240_FACTORS
from .minute_grid import FactorGridMigration, migrate_factor_dict, migrate_factor_expression
from .seed_library import CONFIG_DIR, load_seed_library, seed_pool


def load_legacy_factors(migrate: bool = True) -> Dict[str, str]:
    """The 1498-factor merged corpus, migrated to the 241-bar grid by default."""
    if not migrate:
        return dict(LEGACY_240_FACTORS)
    migrated, _ = migrate_factor_dict(LEGACY_240_FACTORS)
    return migrated


def legacy_migration_audit() -> List[FactorGridMigration]:
    return migrate_factor_dict(LEGACY_240_FACTORS)[1]


def load_seed_population(source: str, n_minutes: int = 241) -> list:
    """GP 初始种群的种子:``"legacy"`` = 原始手写因子库(迁到 241 根、按名字内联引用),否则是一个
    带 ``expression`` 列的 csv。用严格语法解析:宽松语法解析出来的树,交叉 / 变异时祖先节点按严格
    规则重建会失败,子代静悄悄地变成 None。只留合法根。"""
    import csv

    from ..expression import is_valid_root
    from ..parser import parse
    from ..research import inline_references

    if source == "legacy":
        texts = list(inline_references(load_legacy_factors()).values())
    else:
        with open(source, encoding="utf-8") as fh:
            texts = [r["expression"] for r in csv.DictReader(fh)]
    out = []
    for t in texts:
        try:
            e = parse(t, strict=True, coerce_daily=True, n_minutes=n_minutes, allow_general=True)
        except Exception:                                  # noqa: BLE001 - 解析不了的不当种子
            continue
        if is_valid_root(e):
            out.append(e)
    return out


__all__ = ["LEGACY_240_FACTORS", "load_legacy_factors", "load_seed_population", "legacy_migration_audit", "migrate_factor_dict",
           "migrate_factor_expression", "FactorGridMigration", "load_seed_library", "seed_pool", "CONFIG_DIR"]
