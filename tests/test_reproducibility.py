"""Reproducibility and end-to-end consistency tests.

These check that the study is deterministic given its seed, that the raw file
is never mutated, and that whatever artefacts exist on disk are internally
consistent with the code that claims to have produced them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ckd.config import load_config, project_root
from ckd.data.clean import clean_dataset, feature_matrix
from ckd.data.load import load_raw, sha256_of
from ckd.features.configs import FEATURE_CONFIGS
from ckd.models.nested_cv import NestedCVSettings, _run_one_repeat

ROOT = project_root()


class TestDeterminism:
    def test_cleaning_is_deterministic(self):
        a = clean_dataset()
        b = clean_dataset(load_raw())
        pd.testing.assert_frame_equal(
            feature_matrix(a).reset_index(drop=True),
            feature_matrix(b).reset_index(drop=True),
        )
        np.testing.assert_array_equal(a.target.to_numpy(), b.target.to_numpy())

    def test_same_seed_gives_identical_predictions(self, X, y):
        s = NestedCVSettings(seed=42, outer_folds=5, inner_folds=4, repeats=1, n_jobs=1)
        a, _ = _run_one_repeat(X, y, "random_forest", "low_cost_model", "none", 0, s)
        b, _ = _run_one_repeat(X, y, "random_forest", "low_cost_model", "none", 0, s)
        pd.testing.assert_frame_equal(pd.DataFrame(a), pd.DataFrame(b))

    def test_different_seeds_give_different_partitions(self, X, y):
        s1 = NestedCVSettings(seed=42, repeats=1, n_jobs=1)
        s2 = NestedCVSettings(seed=99, repeats=1, n_jobs=1)
        a = pd.DataFrame(_run_one_repeat(X, y, "dummy", "low_cost_model", "none", 0, s1)[0])
        b = pd.DataFrame(_run_one_repeat(X, y, "dummy", "low_cost_model", "none", 0, s2)[0])
        merged = a.merge(b, on="sample_index", suffixes=("_a", "_b"))
        assert not (merged["fold_a"] == merged["fold_b"]).all()

    def test_config_declares_a_seed(self):
        cfg = load_config()
        assert isinstance(cfg["seed"], int)


class TestRawFileIntegrity:
    def test_raw_and_top_level_copies_agree(self):
        top = ROOT / "ckd-dataset-v2.csv"
        raw = ROOT / "data" / "raw" / "ckd-dataset-v2.csv"
        assert raw.is_file(), "the preserved raw copy is missing"
        if top.is_file():
            assert sha256_of(top) == sha256_of(raw), (
                "data/raw/ has drifted from the original supplied file"
            )

    def test_processed_files_are_separate_from_raw(self):
        raw = ROOT / "data" / "raw" / "ckd-dataset-v2.csv"
        processed = ROOT / "data" / "processed"
        if processed.is_dir():
            for p in processed.glob("*.csv"):
                assert p.resolve() != raw.resolve()


class TestConfigIntegrity:
    def test_all_configured_feature_sets_exist(self):
        cfg = load_config()
        for name in cfg["feature_configs"]:
            assert name in FEATURE_CONFIGS

    def test_validation_design_matches_the_specification(self):
        cv = load_config()["cross_validation"]
        assert cv["outer_folds"] == 5
        assert cv["inner_folds"] == 4
        assert cv["repeats"] >= 1

    def test_calibration_methods_are_the_three_studied(self):
        assert set(load_config()["calibration_methods"]) == {"none", "sigmoid", "isotonic"}

    def test_primary_threshold_is_prespecified(self):
        assert load_config()["evaluation"]["primary_threshold"] == 0.5


def _artifact(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


@pytest.mark.skipif(
    not _artifact("data", "processed", "cv_predictions_manifest.json").is_file(),
    reason="nested CV has not been run yet",
)
class TestProducedArtifacts:
    @pytest.fixture(scope="class")
    def manifest(self):
        return json.loads(
            _artifact("data", "processed", "cv_predictions_manifest.json").read_text("utf-8")
        )

    @pytest.fixture(scope="class")
    def preds(self):
        return pd.read_csv(_artifact("data", "processed", "cv_predictions.csv.gz"))

    def test_manifest_matches_the_configuration(self, manifest):
        cfg = load_config()
        assert manifest["seed"] == cfg["seed"]
        assert manifest["outer_folds"] == cfg["cross_validation"]["outer_folds"]
        assert manifest["inner_folds"] == cfg["cross_validation"]["inner_folds"]

    def test_no_prohibited_column_names_appear_in_predictions(self, preds):
        assert "affected" not in preds.columns
        assert "stage" not in preds.columns
        assert "grf" not in preds.columns

    def test_every_patient_predicted_once_per_repeat_per_cell(self, preds):
        counts = preds.groupby(["config", "model", "calibration", "repeat"]).size()
        assert set(counts.unique()) == {200}

    def test_probabilities_are_in_range(self, preds):
        assert preds["y_prob"].between(0, 1).all()

    def test_labels_are_consistent_across_all_cells(self, preds):
        assert (preds.groupby("sample_index")["y_true"].nunique() == 1).all()

    def test_class_balance_preserved(self, preds):
        one = preds[
            (preds["config"] == preds["config"].iloc[0])
            & (preds["model"] == preds["model"].iloc[0])
            & (preds["calibration"] == preds["calibration"].iloc[0])
            & (preds["repeat"] == preds["repeat"].iloc[0])
        ]
        assert int(one["y_true"].sum()) == 128


@pytest.mark.skipif(
    not _artifact("reports", "tables", "table_12_headline_results.csv").is_file(),
    reason="evaluation stage has not been run yet",
)
class TestReportedResultsArePlausible:
    @pytest.fixture(scope="class")
    def headline(self):
        return pd.read_csv(_artifact("reports", "tables", "table_12_headline_results.csv"))

    def test_dummy_baseline_has_chance_discrimination(self, headline):
        d = headline[headline["model"] == "dummy"]
        assert d["roc_auc"].max() < 0.60, "dummy classifier should not discriminate"

    def test_near_perfect_valid_performance_is_documented_not_unexplained(self, headline):
        """Perfection in a valid configuration must be explained, not ignored.

        A valid configuration reaching ROC-AUC ~1.0 is exactly the pattern that
        undetected leakage produces, so it cannot simply be accepted. On this
        dataset it is instead genuine near-separability: the guard tests prove
        no prohibited column reaches these pipelines, and the case-mix analysis
        shows why the sample is separable.

        This test therefore does not forbid perfection - it requires that,
        whenever perfection occurs, the supporting evidence exists on disk. If
        someone later removes the case-mix analysis, or the separability
        explanation stops holding, this fails and forces a re-investigation.
        """
        valid = headline[headline["valid"] & (headline["model"] != "dummy")]
        best = float(valid["roc_auc"].max())
        if best < 0.999:
            return  # nothing extraordinary to explain

        sep_path = _artifact("reports", "tables", "table_22_univariate_separability.csv")
        assert sep_path.is_file(), (
            f"a valid configuration reached ROC-AUC {best:.4f} but the "
            "univariate separability analysis that explains it is missing"
        )
        sep = pd.read_csv(sep_path)
        assert sep["univariate_auc"].max() >= 0.90, (
            f"a valid configuration reached ROC-AUC {best:.4f} yet no single "
            "predictor is strongly discriminative; this is not explained by "
            "dataset separability and should be investigated as possible leakage"
        )

        spec_path = _artifact("reports", "tables", "table_21_spectrum_analysis.csv")
        assert spec_path.is_file(), (
            "near-perfect valid performance requires the case-mix analysis"
        )
        spec = pd.read_csv(spec_path)
        assert "early_ckd_only" in set(spec["subgroup"]), (
            "the case-mix analysis must report the early-CKD subgroup, which is "
            "where a screening-relevant model would be tested"
        )

    def test_perfect_valid_cells_use_no_prohibited_column(self, headline):
        """Independent confirmation that near-perfect cells are not leaking."""
        from ckd.features.configs import FORBIDDEN_IN_VALID
        from ckd.models.pipeline import build_pipeline, pipeline_feature_names

        valid = headline[headline["valid"] & (headline["model"] != "dummy")]
        top = valid[valid["roc_auc"] >= 0.999]
        for _, r in top.iterrows():
            pipe = build_pipeline(r["model"], r["config"], seed=0)
            used = set(pipeline_feature_names(pipe))
            assert not (used & FORBIDDEN_IN_VALID), (
                f"{r['config']}/{r['model']} reaches ROC-AUC {r['roc_auc']:.4f} "
                f"using prohibited columns {sorted(used & FORBIDDEN_IN_VALID)}"
            )

    def test_leaky_configuration_outperforms_valid_ones(self, headline):
        leaky = headline[~headline["valid"] & (headline["model"] != "dummy")]["roc_auc"].max()
        valid = headline[headline["valid"] & (headline["model"] != "dummy")]["roc_auc"].max()
        assert leaky >= valid, (
            "the leaky configuration did not outperform valid ones; "
            "the leakage demonstration is not working as intended"
        )

    def test_metrics_are_within_valid_ranges(self, headline):
        for col in ("roc_auc", "pr_auc", "sensitivity", "specificity",
                    "precision", "npv", "f1", "balanced_accuracy", "brier"):
            s = headline[col].dropna()
            assert s.between(0, 1).all(), f"{col} outside [0, 1]"

    def test_confidence_intervals_bracket_their_estimates(self, headline):
        for col in ("roc_auc", "sensitivity", "specificity", "npv"):
            sub = headline[[col, f"{col}_ci_low", f"{col}_ci_high"]].dropna()
            assert (sub[f"{col}_ci_low"] <= sub[col] + 1e-6).all()
            assert (sub[f"{col}_ci_high"] >= sub[col] - 1e-6).all()
