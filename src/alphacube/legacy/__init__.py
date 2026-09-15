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


__all__ = ["LEGACY_240_FACTORS", "load_legacy_factors", "legacy_migration_audit", "migrate_factor_dict",
           "migrate_factor_expression", "FactorGridMigration", "load_seed_library", "seed_pool", "CONFIG_DIR"]
