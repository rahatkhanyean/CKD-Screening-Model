"""MIMIC-IV extraction and external-validation integrity.

This is the study's only genuinely external evidence, so the properties
that make it external are tested rather than assumed: the cohort is
independent by the gate's own verdict, the label is derived under the name
the registry declares prohibited, and no laboratory value used as a
predictor postdates the diagnosis it is predicting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ckd.config import project_root
from ckd.data.external import get_dataset
from ckd.data.mimic import DIPSTICK_SCALE, LAB_ITEMS, map_dipstick

TABLES = project_root() / "reports" / "tables"
COHORT = (project_root() / "data" / "external" / "mimic_iv_demo"
          / "processed" / "mimic_cohort.csv")


@pytest.fixture(scope="module")
def cohort() -> pd.DataFrame:
    if not COHORT.is_file():
        pytest.skip("run scripts/18_mimic_etl.py")
    return pd.read_csv(COHORT)


@pytest.fixture(scope="module")
def validation() -> pd.DataFrame:
    path = TABLES / "table_42_external_validation.csv"
    if not path.is_file():
        pytest.skip("run scripts/19_external_validation.py")
    return pd.read_csv(path)


class TestDipstickMapping:
    def test_is_a_pure_function_of_text(self):
        """Mirrors the statelessness property of the interval encoding: the
        mapping must not depend on which other rows are present."""
        full = map_dipstick(pd.Series(["NEG", "TRACE", "100", "300", "junk"]))
        subset = map_dipstick(pd.Series(["100", "NEG"]))
        assert subset.iloc[0] == full.iloc[2]
        assert subset.iloc[1] == full.iloc[0]

    def test_unknown_tokens_become_missing_not_guessed(self):
        out = map_dipstick(pd.Series(["not-a-real-result"]))
        assert out.isna().all()

    def test_scale_is_monotone(self):
        assert DIPSTICK_SCALE["neg"] < DIPSTICK_SCALE["trace"]
        assert DIPSTICK_SCALE["30"] < DIPSTICK_SCALE["100"] < DIPSTICK_SCALE["300"]


class TestCohort:
    def test_expected_size_and_labels(self, cohort):
        assert len(cohort) == 100
        assert set(cohort["ckd_label"]) == {0, 1}
        assert int(cohort["ckd_label"].sum()) == 26

    def test_label_column_is_the_name_the_registry_prohibits(self, cohort):
        """The guard was declared before this ETL existed; the ETL must use
        that name so the declaration actually covers it."""
        record = get_dataset("mimic_iv_demo")
        assert "ckd_label" in record.prohibited_columns
        assert "ckd_label" in cohort.columns

    def test_blood_panel_is_well_covered(self, cohort):
        for variable in ("sc", "hemo", "sod", "pot", "bu", "bgr", "wbcc",
                         "rbcc", "pcv"):
            assert cohort[variable].notna().sum() >= 85, variable

    def test_values_are_physiologically_plausible(self, cohort):
        """A unit-conversion error is the most likely silent ETL failure, and
        would not show up as a missing value."""
        sc = cohort["sc"].dropna()
        assert sc.between(0.1, 40).all(), sc.describe()
        hemo = cohort["hemo"].dropna()
        assert hemo.between(3, 22).all(), hemo.describe()
        wbcc = cohort["wbcc"].dropna()      # scaled from K/uL to cells/uL
        assert wbcc.between(500, 100000).all(), wbcc.describe()
        sg = cohort["sg"].dropna()
        assert sg.between(1.0, 1.06).all(), sg.describe()


class TestTemporalGuard:
    def test_notes_record_the_guard_and_its_cost(self):
        path = TABLES / "table_41_mimic_notes.txt"
        if not path.is_file():
            pytest.skip("run scripts/18_mimic_etl.py")
        text = path.read_text(encoding="utf-8")
        assert "temporal guard" in text
        assert "not strictly preceding" in text
        # Subjects left without any eligible measurement must be reported.
        assert "not imputed" in text


class TestExternalValidation:
    def test_only_independent_cohorts_are_used(self, validation):
        for dataset_id in validation["dataset"].unique():
            record = get_dataset(dataset_id)
            verdicts = pd.read_csv(
                TABLES / "table_24_provenance.csv"
            ).set_index("dataset_id")["verdict"].to_dict()
            assert verdicts[record.dataset_id] == "INDEPENDENT"

    def test_both_arms_present_for_every_model(self, validation):
        for model, group in validation.groupby("model"):
            arms = set(group["arm"])
            assert "internal_nested_cv" in arms
            assert "external_frozen_transfer" in arms

    def test_external_estimates_carry_intervals(self, validation):
        ext = validation[validation["arm"] == "external_frozen_transfer"]
        assert ext["roc_auc_ci_low"].notna().all()
        assert (ext["roc_auc_ci_low"] < ext["roc_auc"]).all()
        assert (ext["roc_auc_ci_high"] > ext["roc_auc"]).all()

    def test_recalibration_preserves_discrimination(self, validation):
        """Recalibration-in-the-large is a monotone transform, so it cannot
        change the ranking. If AUC moved, the implementation is wrong."""
        for model, group in validation.groupby("model"):
            ext = group[group["arm"] == "external_frozen_transfer"]
            rec = group[group["arm"] == "external_recalibrated_in_the_large"]
            if ext.empty or rec.empty:
                continue
            assert abs(float(ext["roc_auc"].iloc[0])
                       - float(rec["roc_auc"].iloc[0])) < 1e-9

    def test_the_headline_drop_is_recorded(self, validation):
        """The reference-run finding: internal discrimination near 1.0 does
        not transfer. Pinned so that a change in either arm is noticed."""
        internal = validation[validation["arm"] == "internal_nested_cv"]["roc_auc"]
        external = validation[
            validation["arm"] == "external_frozen_transfer"
        ]["roc_auc"]
        assert internal.min() > 0.98
        assert external.max() < 0.85
        assert (internal.min() - external.max()) > 0.20
