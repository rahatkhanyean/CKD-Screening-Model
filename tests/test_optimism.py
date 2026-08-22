"""Optimism-accounting invariants (Phase 1e).

Apparent (resubstitution) performance can never be honestly *below* the
nested out-of-fold estimate by more than numerical noise — if it were, the
apparent fit or the aggregation would be broken. The tests also pin the
aggregation contract: nested numbers in table_29 must equal metrics
recomputed from the pooled predictions with the same helper the published
tables use.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from ckd.config import project_root
from ckd.evaluation.bootstrap import pooled_predictions

TABLE = project_root() / "reports" / "tables" / "table_29_optimism.csv"


@pytest.fixture(scope="module")
def optimism() -> pd.DataFrame:
    if not TABLE.is_file():
        pytest.skip("table_29_optimism.csv not yet generated (run stage 9)")
    return pd.read_csv(TABLE)


class TestOptimismInvariants:
    def test_expected_cells_present(self, optimism):
        cells = set(zip(optimism["config"], optimism["model"]))
        assert cells == {
            ("low_cost_model", "svm"),
            ("low_cost_model", "random_forest"),
            ("full_valid_model", "svm"),
            ("full_valid_model", "random_forest"),
            ("laboratory_model", "svm"),
            ("laboratory_model", "random_forest"),
        }

    def test_apparent_never_below_nested(self, optimism):
        assert (optimism["optimism_roc_auc"] >= -1e-9).all(), (
            optimism[optimism["optimism_roc_auc"] < -1e-9]
        )

    def test_apparent_brier_never_worse_than_nested(self, optimism):
        # Resubstitution can only flatter the Brier score too.
        assert (optimism["apparent_brier"] <= optimism["nested_brier"] + 1e-9).all()

    def test_nested_numbers_match_published_aggregation(self, optimism):
        preds = pd.read_csv(
            project_root() / "data" / "processed" / "cv_predictions.csv.gz"
        )
        pooled = pooled_predictions(preds)
        for _, r in optimism.iterrows():
            cell = pooled[
                (pooled["config"] == r["config"])
                & (pooled["model"] == r["model"])
                & (pooled["calibration"] == "none")
            ]
            auc = roc_auc_score(cell["y_true"], cell["y_prob"])
            assert abs(auc - r["nested_roc_auc"]) < 1e-9
