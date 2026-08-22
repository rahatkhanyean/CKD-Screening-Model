"""Tests for the encoders, feature specifications and data-quality audit helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ckd.data.quality import audit_bin_monotonicity, cramers_v
from ckd.features.configs import (
    FEATURE_CONFIGS,
    SPEC_BY_NAME,
    VARIABLES,
    get_config,
    uncertain_variables,
    valid_config_names,
)
from ckd.features.encoders import ColumnSelector, LeakageGuard


class TestVariableSpecifications:
    def test_every_dataset_column_has_a_specification(self, clean_result):
        cols = set(clean_result.clean.columns) - {"source_csv_line"}
        specified = {v.name for v in VARIABLES}
        assert cols == specified, f"unspecified columns: {cols - specified}"

    def test_every_variable_has_a_rationale(self):
        for v in VARIABLES:
            assert v.rationale.strip(), f"{v.name} has no cost rationale"
            assert v.description.strip(), f"{v.name} has no description"

    def test_uncertain_variables_explain_their_uncertainty(self):
        for v in uncertain_variables():
            assert v.uncertainty_note.strip(), (
                f"{v.name} is flagged uncertain but gives no explanation"
            )

    def test_certain_variables_do_not_carry_an_uncertainty_note(self):
        for v in VARIABLES:
            if not v.uncertain:
                assert not v.uncertainty_note

    def test_tiers_are_from_the_declared_vocabulary(self):
        allowed = {"history", "exam", "urine_dip", "urine_micro",
                   "blood_lab", "derived_dx", "target"}
        for v in VARIABLES:
            assert v.tier in allowed

    def test_ane_is_conservatively_tiered_as_laboratory(self):
        """Documented decision: provenance is ambiguous, so it is kept out of low-cost."""
        assert SPEC_BY_NAME["ane"].tier == "blood_lab"
        assert SPEC_BY_NAME["ane"].uncertain

    def test_ane_is_not_a_deterministic_function_of_haemoglobin(self, clean_result):
        """The empirical claim used to justify the ane tiering decision."""
        clean = clean_result.clean
        purity = clean.groupby("hemo")["ane"].apply(
            lambda s: max((s == "1").mean(), (s == "0").mean())
        )
        assert (purity < 1.0).any(), (
            "ane appears to be a deterministic recoding of hemo; the "
            "documented justification in configs.py needs revising"
        )

    def test_dm_is_not_a_deterministic_function_of_blood_glucose(self, clean_result):
        clean = clean_result.clean
        purity = clean.groupby("bgr")["dm"].apply(
            lambda s: max((s == "1").mean(), (s == "0").mean())
        )
        assert (purity < 1.0).any()


class TestFeatureConfigurationIntegrity:
    def test_the_four_required_configurations_exist(self):
        for required in ("leaky_model", "full_valid_model",
                         "low_cost_model", "laboratory_model"):
            assert required in FEATURE_CONFIGS

    def test_every_configuration_has_a_stated_purpose(self):
        for name, cfg in FEATURE_CONFIGS.items():
            assert len(cfg.purpose) > 40, f"{name} has no meaningful purpose statement"

    def test_no_duplicate_features_within_a_configuration(self):
        for name, cfg in FEATURE_CONFIGS.items():
            assert len(cfg.features) == len(set(cfg.features)), f"{name} repeats a feature"

    def test_low_cost_is_a_strict_subset_of_full_valid(self):
        assert set(FEATURE_CONFIGS["low_cost_model"].features) < set(
            FEATURE_CONFIGS["full_valid_model"].features
        )

    def test_clinical_only_is_a_strict_subset_of_low_cost(self):
        assert set(FEATURE_CONFIGS["clinical_only_model"].features) < set(
            FEATURE_CONFIGS["low_cost_model"].features
        )

    def test_low_cost_and_laboratory_partition_the_full_valid_set(self):
        low = set(FEATURE_CONFIGS["low_cost_model"].features)
        lab = set(FEATURE_CONFIGS["laboratory_model"].features)
        full = set(FEATURE_CONFIGS["full_valid_model"].features)
        assert low | lab == full
        assert not (low & lab), "a variable is in both the low-cost and laboratory sets"

    def test_leaky_is_full_valid_plus_exactly_the_three_prohibited_columns(self):
        leaky = set(FEATURE_CONFIGS["leaky_model"].features)
        full = set(FEATURE_CONFIGS["full_valid_model"].features)
        assert leaky - full == {"grf", "stage", "affected"}

    def test_get_config_rejects_unknown_names(self):
        with pytest.raises(KeyError):
            get_config("nope")

    def test_valid_config_names_excludes_the_leaky_one(self):
        assert "leaky_model" not in valid_config_names()
        assert len(valid_config_names()) == len(FEATURE_CONFIGS) - 1


class TestColumnSelector:
    def test_selects_in_the_declared_order(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        out = ColumnSelector(["c", "a"]).fit_transform(df)
        assert list(out.columns) == ["c", "a"]

    def test_raises_on_missing_column(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(KeyError, match="absent"):
            ColumnSelector(["a", "zzz"]).fit(df)

    def test_rejects_numpy_input(self):
        with pytest.raises(TypeError):
            ColumnSelector(["a"]).fit(np.zeros((2, 1)))

    def test_feature_names_out(self):
        sel = ColumnSelector(["b", "a"]).fit(pd.DataFrame({"a": [1], "b": [2]}))
        assert list(sel.get_feature_names_out()) == ["b", "a"]


class TestLeakageGuardBehaviour:
    def test_passes_clean_frames_through_unchanged(self):
        df = pd.DataFrame({"age": [1, 2], "sg": [3, 4]})
        out = LeakageGuard(label="t").fit_transform(df)
        pd.testing.assert_frame_equal(out, df)

    def test_error_message_names_the_offending_columns(self):
        df = pd.DataFrame({"age": [1], "grf": [2], "stage": [3]})
        with pytest.raises(Exception) as exc:
            LeakageGuard(label="cfg").fit(df)
        msg = str(exc.value)
        assert "grf" in msg and "stage" in msg and "cfg" in msg

    def test_custom_forbidden_set(self):
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(Exception):
            LeakageGuard(forbidden={"x"}, label="t").fit(df)


class TestQualityHelpers:
    def test_cramers_v_is_one_for_a_perfect_relationship(self):
        a = pd.Series(["a", "a", "b", "b"] * 25)
        assert cramers_v(a, a) == pytest.approx(1.0, abs=1e-6)

    def test_cramers_v_is_near_zero_for_independence(self):
        rng = np.random.default_rng(0)
        a = pd.Series(rng.integers(0, 2, 4000).astype(str))
        b = pd.Series(rng.integers(0, 2, 4000).astype(str))
        assert cramers_v(a, b) < 0.1

    def test_cramers_v_handles_a_constant_column(self):
        a = pd.Series(["x"] * 50)
        b = pd.Series(["y", "z"] * 25)
        assert np.isnan(cramers_v(a, b))

    def test_monotonicity_audit_detects_a_synthetic_overlap(self):
        bin_map = {
            "fake": [
                {"label": "1 - 2", "lower": 1.0, "upper": 2.0, "representative": 1.5},
                {"label": "1.5 - 3", "lower": 1.5, "upper": 3.0, "representative": 2.25},
            ]
        }
        findings = audit_bin_monotonicity(bin_map)
        assert findings and findings[0]["overlapping_edges"]

    def test_monotonicity_audit_passes_a_clean_set(self):
        bin_map = {
            "fake": [
                {"label": "< 1", "lower": -np.inf, "upper": 1.0, "representative": 1.0},
                {"label": "1 - 3", "lower": 1.0, "upper": 3.0, "representative": 2.0},
                {"label": ">= 3", "lower": 3.0, "upper": np.inf, "representative": 3.0},
            ]
        }
        assert audit_bin_monotonicity(bin_map) == []


class TestSpectrumAnalysis:
    """The case-mix analysis must partition, never predict, with `stage`."""

    def test_subgroups_cover_all_stages(self):
        from ckd.evaluation.spectrum import SUBGROUPS

        covered = set()
        for stages in SUBGROUPS.values():
            covered |= set(stages)
        assert covered == {"s1", "s2", "s3", "s4", "s5"}

    def test_non_ckd_patients_are_retained_in_every_subgroup(self, clean_result):
        from ckd.evaluation.spectrum import spectrum_analysis

        y = clean_result.target.to_numpy()
        rng = np.random.default_rng(0)
        pooled = pd.DataFrame({
            "config": "low_cost_model", "model": "logreg", "calibration": "none",
            "sample_index": np.arange(len(y)),
            "y_true": y,
            "y_prob": np.clip(0.3 + 0.4 * y + rng.normal(0, 0.15, len(y)), 0.01, 0.99),
            "screening_threshold": 0.4,
        })
        out = spectrum_analysis(pooled, clean_result.clean["stage"])
        assert (out["n_non_ckd"] == int((1 - y).sum())).all(), (
            "non-CKD comparison group must be identical across subgroups"
        )

    def test_subgroup_ckd_counts_sum_to_total(self, clean_result):
        from ckd.evaluation.spectrum import spectrum_analysis

        y = clean_result.target.to_numpy()
        pooled = pd.DataFrame({
            "config": "c", "model": "m", "calibration": "none",
            "sample_index": np.arange(len(y)), "y_true": y,
            "y_prob": np.full(len(y), 0.5), "screening_threshold": 0.5,
        })
        out = spectrum_analysis(pooled, clean_result.clean["stage"])
        parts = out[out["subgroup"].isin(
            ["early_ckd_only", "moderate_ckd_only", "advanced_ckd_only"]
        )]["n_ckd"].sum()
        total = out[out["subgroup"] == "all_patients"]["n_ckd"].iloc[0]
        assert parts == total

    def test_separability_report_flags_the_most_separable_predictor(self, X, y):
        from ckd.evaluation.spectrum import separability_report
        from ckd.features.configs import FEATURE_CONFIGS

        rep = separability_report(
            X, y, list(FEATURE_CONFIGS["full_valid_model"].features)
        )
        assert rep.iloc[0]["univariate_auc"] >= rep.iloc[-1]["univariate_auc"]
        assert (rep["overlap_fraction_of_range"].between(0, 1)).all()
        assert (rep["fraction_patients_in_overlap"].between(0, 1)).all()
        # Documented finding: haemoglobin is the most separable single predictor.
        assert rep.iloc[0]["feature"] == "hemo"
