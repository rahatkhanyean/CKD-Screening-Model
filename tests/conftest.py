"""Shared fixtures. Data is loaded once per session; the raw file is read-only."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ckd.data.clean import CleanResult, clean_dataset, feature_matrix  # noqa: E402
from ckd.data.load import load_raw  # noqa: E402


@pytest.fixture(scope="session")
def raw_result():
    return load_raw()


@pytest.fixture(scope="session")
def clean_result(raw_result) -> CleanResult:
    return clean_dataset(raw_result)


@pytest.fixture(scope="session")
def X(clean_result):
    return feature_matrix(clean_result)


@pytest.fixture(scope="session")
def y(clean_result):
    return clean_result.target.to_numpy()
