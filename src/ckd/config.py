"""Configuration loading and global path resolution.

The single source of truth for every tunable knob is ``config/experiment.yaml``.
This module resolves the project root robustly (so scripts, tests and notebooks
all agree) and exposes the parsed configuration as a plain dictionary.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the repository root.

    Resolution order:
    1. ``CKD_PROJECT_ROOT`` environment variable (used by CI / notebooks).
    2. The first ancestor of this file that contains ``config/experiment.yaml``.
    """
    env = os.environ.get("CKD_PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "experiment.yaml").is_file():
            return parent
    raise RuntimeError(
        "Could not locate the project root. Set CKD_PROJECT_ROOT to the "
        "directory containing config/experiment.yaml."
    )


@lru_cache(maxsize=None)
def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load ``config/experiment.yaml`` (cached)."""
    cfg_path = Path(path) if path is not None else project_root() / "config" / "experiment.yaml"
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def resolve(relative: str) -> Path:
    """Resolve a config-relative path against the project root."""
    return project_root() / relative


def ensure_dirs(*relatives: str) -> list[Path]:
    """Create (if needed) and return project-relative directories."""
    out: list[Path] = []
    for rel in relatives:
        p = resolve(rel)
        p.mkdir(parents=True, exist_ok=True)
        out.append(p)
    return out
