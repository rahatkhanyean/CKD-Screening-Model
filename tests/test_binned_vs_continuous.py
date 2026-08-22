"""Model-level binned-versus-continuous comparison (Phase 3d).

The comparison is only meaningful if the two arms differ in exactly one
respect. These tests pin that: same patients, same cells, same design,
with only the representation of the recoverable variables changed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ckd.config import project_root

PROC = project_root() / "data" / "processed"
TABLES = project_root() / "reports" / "tables"


def _arm(name: str) -> pd.DataFrame:
    path = PROC / f"cv_predictions_{name}.csv.gz"
    if not path.is_file():
        pytest.skip("run scripts/12_binned_vs_continuous.py")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def binned() -> pd.DataFrame:
    return _arm("binned")


@pytest.fixture(scope="module")
def continuous() -> pd.DataFrame:
    return _arm("continuous")


@pytest.fixture(scope="module")
def comparison() -> pd.DataFrame:
    path = TABLES / "table_35_binned_vs_continuous.csv"
    if not path.is_file():
        pytest.skip("run scripts/12_binned_vs_continuous.py")
    return pd.read_csv(path)


class TestArmsAreComparable:
    def test_same_patients_and_labels(self, binned, continuous):
        b = binned.groupby("sample_index")["y_true"].first()
        c = continuous.groupby("sample_index")["y_true"].first()
        assert list(b.index) == list(c.index)
        assert np.array_equal(b.to_numpy(), c.to_numpy())

    def test_same_cells_evaluated(self, binned, continuous):
        keys = ["config", "model", "calibration"]
        assert (set(map(tuple, binned[keys].drop_duplicates().to_numpy()))
                == set(map(tuple, continuous[keys].drop_duplicates().to_numpy())))

    def test_same_number_of_predictions(self, binned, continuous):
        assert len(binned) == len(continuous)

    def test_cohort_is_the_uniquely_pinned_set(self, binned):
        recovered = PROC / "ckd_recovered_continuous.csv"
        if not recovered.is_file():
            pytest.skip("run scripts/11_continuous_recovery.py")
        n_patients = binned["sample_index"].nunique()
        assert n_patients == len(pd.read_csv(recovered))


class TestFindings:
    def test_deltas_are_within_bootstrap_noise(self, comparison):
        """The reference-run finding: restoring the measurements changes no
        conclusion. Bootstrap intervals in section 5.8 span ~0.02, so every
        difference must sit well inside that."""
        assert comparison["delta_roc_auc"].abs().max() < 0.02

    def test_delta_columns_are_consistent(self, comparison):
        recomputed = comparison["roc_auc_continuous"] - comparison["roc_auc_binned"]
        assert np.allclose(recomputed, comparison["delta_roc_auc"], atol=1e-12)

    def test_univariate_gain_did_not_transfer(self, comparison):
        """Serum creatinine gains >0.2 AUC univariately (table 31), yet no
        laboratory cell gains meaningfully - the dissociation the report
        builds its argument on."""
        costs_path = TABLES / "table_31_binning_cost.csv"
        if not costs_path.is_file():
            pytest.skip("run scripts/11_continuous_recovery.py")
        costs = pd.read_csv(costs_path).set_index("variable")
        assert costs.loc["sc", "binning_cost"] > 0.2
        lab = comparison[comparison["config"] == "laboratory_model"]
        assert lab["delta_roc_auc"].max() < 0.02
