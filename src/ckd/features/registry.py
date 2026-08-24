"""Feature configurations derived from the machine-readable registry.

``data/feature_registry.csv`` (built by ``scripts/22_build_feature_registry.py``)
is the declarative source of truth: for every column it records the leakage
category, availability at index time, whether the column participates in
constructing the outcome, and which configurations may contain it.

This module derives the prohibited sets and the configuration memberships
from that file, so a classification cannot drift out of step with the
configuration that depends on it. The derivation is checked against the
hand-maintained definitions in :mod:`ckd.features.configs` by
``tests/test_registry.py``; the two must agree exactly, which is what makes
it safe to treat the registry as authoritative without re-running the
reference analysis.

Categories that may never enter a clinically valid configuration:

* ``direct_outcome_copy`` --- the target or a renamed copy of it
* ``deterministic_outcome_derivative`` --- a deterministic function of the
  target or of the quantity defining it

Categories that are permitted but carry interpretation caveats, recorded so
that a reader can see which are load-bearing:

* ``incorporation_risk`` --- an input to the diagnostic criterion
* ``consequence_of_advanced_disease`` --- downstream of established disease
* ``uncertain_provenance`` --- meaning or timing unverified
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from ..config import project_root

#: Categories barred from every clinically valid configuration.
PROHIBITED_CATEGORIES = frozenset({
    "direct_outcome_copy",
    "deterministic_outcome_derivative",
})

#: Categories that are admissible but whose use must be disclosed.
CAVEATED_CATEGORIES = frozenset({
    "incorporation_risk",
    "consequence_of_advanced_disease",
    "uncertain_provenance",
})

REGISTRY_PATH = "data/feature_registry.csv"


@lru_cache(maxsize=1)
def load_registry() -> pd.DataFrame:
    """Load the feature registry, validating its schema."""
    path = project_root() / REGISTRY_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"{REGISTRY_PATH} missing; run scripts/22_build_feature_registry.py"
        )
    registry = pd.read_csv(path)
    required = {
        "variable", "leakage_category", "available_at_index",
        "constructs_or_stages_outcome", "decision", "config_membership",
        "uncertainty_flag", "evidence", "cost_tier",
    }
    missing = required - set(registry.columns)
    if missing:
        raise ValueError(f"feature registry missing columns: {sorted(missing)}")
    unknown = set(registry["leakage_category"]) - (
        PROHIBITED_CATEGORIES | CAVEATED_CATEGORIES | {"legitimate_pre_index"}
    )
    if unknown:
        raise ValueError(f"unknown leakage categories: {sorted(unknown)}")
    return registry


def prohibited_variables() -> frozenset[str]:
    """Columns barred from every clinically valid configuration."""
    registry = load_registry()
    return frozenset(
        registry.loc[
            registry["leakage_category"].isin(PROHIBITED_CATEGORIES), "variable"
        ]
    )


def caveated_variables() -> dict[str, str]:
    """Admissible columns whose use requires disclosure, and why."""
    registry = load_registry()
    subset = registry[registry["leakage_category"].isin(CAVEATED_CATEGORIES)]
    return dict(zip(subset["variable"], subset["leakage_category"]))


def configuration_features(config_name: str) -> tuple[str, ...]:
    """Columns the registry assigns to one configuration, in registry order."""
    registry = load_registry()
    members = registry[
        registry["config_membership"].fillna("").str.split(";").apply(
            lambda names: config_name in names
        )
    ]
    return tuple(members["variable"])


def configuration_names() -> tuple[str, ...]:
    """Every configuration mentioned anywhere in the registry."""
    registry = load_registry()
    names: set[str] = set()
    for entry in registry["config_membership"].fillna(""):
        names.update(n for n in entry.split(";") if n and n != "none")
    return tuple(sorted(names))


def variables_by_category(category: str) -> tuple[str, ...]:
    """All columns in one leakage category."""
    registry = load_registry()
    return tuple(
        registry.loc[registry["leakage_category"] == category, "variable"]
    )
