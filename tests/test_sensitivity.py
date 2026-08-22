"""Label-robustness variant integrity (Phase 1c / N4).

The four internally inconsistent records (notckd label, stage s3-s5; CSV
lines 12, 18, 52, 123) are evidence about label quality and must be
handled as sensitivity analyses, never as silent deletions. These tests
pin that behaviour:

* the variant runs touch exactly those four records — the exclusion run
  drops precisely them, the flip run flips precisely them;
* the primary artefacts are untouched by variants;
* the comparison table recomputes from the variant predictions.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ckd.config import project_root

PROC = project_root() / "data" / "processed"
INCONSISTENT_LINES = [12, 18, 52, 123]


def _load(stem: str):
    path = PROC / f"{stem}.csv.gz"
    if not path.is_file():
        pytest.skip(f"{path.name} not generated (run the label-variant stage)")
    manifest_path = PROC / f"{stem}_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text("utf-8")) if manifest_path.is_file() else {}
    )
    return pd.read_csv(path), manifest


@pytest.fixture(scope="module")
def inconsistent_positions(clean_result) -> list[int]:
    mask = (
        (clean_result.clean["class"] == "notckd")
        & clean_result.clean["stage"].isin(["s3", "s4", "s5"])
    ).to_numpy()
    lines = sorted(int(v) for v in clean_result.clean.loc[mask, "source_csv_line"])
    assert lines == INCONSISTENT_LINES
    return [int(i) for i in np.flatnonzero(mask)]


class TestPrimaryUntouched:
    def test_primary_manifest_is_not_a_variant(self):
        manifest = json.loads((PROC / "cv_predictions_manifest.json").read_text("utf-8"))
        assert manifest.get("label_variant", "none") == "none"
        assert manifest["n_patients"] == 200


class TestExclusionVariant:
    def test_manifest_records_variant_and_lines(self):
        _, manifest = _load("cv_predictions_labels_excl")
        assert manifest["label_variant"] == "exclude_inconsistent"
        assert manifest["label_variant_csv_lines"] == INCONSISTENT_LINES
        assert manifest["n_patients"] == 196

    def test_every_patient_predicted_once_per_repeat(self):
        preds, _ = _load("cv_predictions_labels_excl")
        counts = preds.groupby(["config", "model", "calibration", "repeat"]).size()
        assert (counts == 196).all()


class TestFlipVariant:
    def test_manifest_records_variant_and_lines(self):
        _, manifest = _load("cv_predictions_labels_flip")
        assert manifest["label_variant"] == "flip_inconsistent"
        assert manifest["label_variant_csv_lines"] == INCONSISTENT_LINES
        assert manifest["n_patients"] == 200
        # 4 notckd records became ckd: positives rise from 128 to 132.
        assert manifest["n_positive"] == 132

    def test_labels_flipped_exactly_at_the_four_records(self, inconsistent_positions):
        preds, _ = _load("cv_predictions_labels_flip")
        primary = pd.read_csv(PROC / "cv_predictions.csv.gz")
        flip_truth = preds.groupby("sample_index")["y_true"].first()
        primary_truth = primary.groupby("sample_index")["y_true"].first()
        shared = flip_truth.index.intersection(primary_truth.index)
        differs = flip_truth.loc[shared] != primary_truth.loc[shared]
        assert sorted(differs[differs].index) == inconsistent_positions


class TestComparisonTable:
    def test_table_recomputes_from_variant_predictions(self):
        path = project_root() / "reports" / "tables" / "table_27_label_robustness.csv"
        if not path.is_file():
            pytest.skip("table_27 not yet generated (run stage 8)")
        table = pd.read_csv(path)
        assert {"exclude_inconsistent", "flip_inconsistent"} == set(table["variant"])
        assert (table.loc[table["variant"] == "exclude_inconsistent", "n_variant"] == 196).all()
        assert (table.loc[table["variant"] == "flip_inconsistent", "n_variant"] == 200).all()
        # Deltas must be consistent with their own columns.
        recomputed = table["variant_roc_auc"] - table["primary_roc_auc"]
        assert np.allclose(recomputed, table["delta_roc_auc"], atol=1e-12)
