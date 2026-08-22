"""Leakage tests.

These are the tests that must never be relaxed. They assert that prohibited
columns cannot enter a clinically valid model pipeline by any route: not
through the feature configuration, not by passing the wrong DataFrame, not at
prediction time, and not through a preprocessing step fitted on data outside
the training fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from ckd.features.configs import (
    ALWAYS_FORBIDDEN,
    FEATURE_CONFIGS,
    FORBIDDEN_IN_VALID,
    POST_DIAGNOSIS,
    valid_config_names,
)
from ckd.features.encoders import LeakageError, LeakageGuard
from ckd.models.pipeline import (
    assert_pipeline_is_clean,
    build_pipeline,
    pipeline_feature_names,
)
from ckd.models.zoo import available_models

VALID_CONFIGS = valid_config_names()
MODELS = available_models()


class TestFeatureConfigurationDeclarations:
    def test_valid_configs_declare_no_prohibited_columns(self):
        for name in VALID_CONFIGS:
            cfg = FEATURE_CONFIGS[name]
            illegal = set(cfg.features) & FORBIDDEN_IN_VALID
            assert not illegal, f"{name} declares prohibited columns {illegal}"

    def test_no_config_ever_includes_the_outcome_column(self):
        for name, cfg in FEATURE_CONFIGS.items():
            assert "class" not in cfg.features, f"{name} includes the outcome column"

    def test_leaky_config_is_flagged_invalid(self):
        cfg = FEATURE_CONFIGS["leaky_model"]
        assert cfg.valid_for_clinical_interpretation is False
        assert set(cfg.contains_forbidden) == {"grf", "stage", "affected"}

    def test_leaky_config_actually_contains_the_leaks(self):
        """The leakage demonstration is only meaningful if the leaks are present."""
        cfg = FEATURE_CONFIGS["leaky_model"]
        assert {"affected", "stage", "grf"} <= set(cfg.features)

    def test_affected_appears_in_no_valid_config(self):
        for name in VALID_CONFIGS:
            assert "affected" not in FEATURE_CONFIGS[name].features

    def test_post_diagnosis_columns_appear_in_no_valid_config(self):
        for name in VALID_CONFIGS:
            assert not (POST_DIAGNOSIS & set(FEATURE_CONFIGS[name].features))

    def test_low_cost_contains_no_blood_or_microscopy_variables(self):
        from ckd.features.configs import SPEC_BY_NAME

        for f in FEATURE_CONFIGS["low_cost_model"].features:
            assert SPEC_BY_NAME[f].tier in {"history", "exam", "urine_dip"}, (
                f"{f} is tier {SPEC_BY_NAME[f].tier}, which is not low-cost"
            )

    def test_laboratory_config_excludes_target_proxies(self):
        feats = set(FEATURE_CONFIGS["laboratory_model"].features)
        assert not (feats & FORBIDDEN_IN_VALID)

    def test_every_declared_feature_exists_in_the_data(self, X):
        for name, cfg in FEATURE_CONFIGS.items():
            missing = [f for f in cfg.features if f not in X.columns]
            assert not missing, f"{name} references non-existent columns {missing}"


class TestLeakageGuard:
    def test_guard_raises_on_forbidden_column_at_fit(self):
        guard = LeakageGuard(label="test")
        df = pd.DataFrame({"age": [1, 2], "affected": [0, 1]})
        with pytest.raises(LeakageError, match="affected"):
            guard.fit(df)

    def test_guard_raises_on_forbidden_column_at_transform(self):
        guard = LeakageGuard(label="test")
        clean = pd.DataFrame({"age": [1, 2]})
        guard.fit(clean)
        dirty = pd.DataFrame({"age": [1, 2], "stage": [1, 2]})
        with pytest.raises(LeakageError, match="stage"):
            guard.transform(dirty)

    def test_guard_rejects_arrays_without_column_names(self):
        guard = LeakageGuard(label="test")
        with pytest.raises(LeakageError, match="DataFrame"):
            guard.fit(np.zeros((3, 2)))

    def test_guard_allows_explicit_exemptions(self):
        guard = LeakageGuard(allow=("stage",), label="leaky")
        df = pd.DataFrame({"age": [1, 2], "stage": [1, 2]})
        guard.fit(df)  # must not raise

    def test_exemption_cannot_unlock_the_outcome_column(self):
        """Even the leaky configuration must not be able to admit `class`."""
        pipe = build_pipeline("logreg", "leaky_model", seed=0)
        df = pd.DataFrame({"age": [1.0, 2.0], "class": ["ckd", "notckd"]})
        with pytest.raises(LeakageError):
            pipe.named_steps["outcome_guard"].fit(df)


class TestPipelineConstruction:
    @pytest.mark.parametrize("config", VALID_CONFIGS)
    @pytest.mark.parametrize("model", MODELS)
    def test_valid_pipelines_pass_the_cleanliness_assertion(self, model, config):
        pipe = build_pipeline(model, config, seed=0)
        assert_pipeline_is_clean(pipe, config)

    @pytest.mark.parametrize("config", VALID_CONFIGS)
    def test_valid_pipeline_selects_only_declared_features(self, config, X):
        pipe = build_pipeline("logreg", config, seed=0)
        selected = set(pipeline_feature_names(pipe))
        assert selected == set(FEATURE_CONFIGS[config].features)
        assert not (selected & FORBIDDEN_IN_VALID)

    @pytest.mark.parametrize("config", VALID_CONFIGS)
    def test_valid_pipeline_refuses_a_frame_containing_prohibited_columns(
        self, config, X, y
    ):
        """The single most important test in the suite.

        Even if a caller hands the pipeline the *whole* dataframe including
        `affected`, `stage` and `grf`, it must refuse rather than quietly
        select its own subset.
        """
        pipe = build_pipeline("logreg", config, seed=0)
        with pytest.raises(LeakageError):
            pipe.fit(X, y)

    def test_leaky_pipeline_does_fit_on_the_full_frame(self, X, y):
        """Control condition: the invalid configuration is allowed to leak."""
        pipe = build_pipeline("logreg", "leaky_model", seed=0)
        pipe.fit(X, y)
        assert {"affected", "stage", "grf"} <= set(pipeline_feature_names(pipe))

    @pytest.mark.parametrize("config", VALID_CONFIGS)
    def test_valid_pipeline_fits_on_a_correctly_restricted_frame(self, config, X, y):
        pipe = build_pipeline("logreg", config, seed=0)
        cols = list(FEATURE_CONFIGS[config].features)
        pipe.fit(X[cols], y)
        proba = pipe.predict_proba(X[cols])[:, 1]
        assert proba.shape == (len(y),)
        assert np.all((proba >= 0) & (proba <= 1))

    @pytest.mark.parametrize("config", VALID_CONFIGS)
    def test_prohibited_column_rejected_at_prediction_time(self, config, X, y):
        pipe = build_pipeline("logreg", config, seed=0)
        cols = list(FEATURE_CONFIGS[config].features)
        pipe.fit(X[cols], y)
        contaminated = X[cols].copy()
        contaminated["affected"] = y
        with pytest.raises(LeakageError):
            pipe.predict_proba(contaminated)


class TestNoPreprocessingOnFullData:
    """Confirm that preprocessing statistics are fold-local, not global."""

    def test_imputer_and_scaler_statistics_differ_between_folds(self, X, y):
        cols = list(FEATURE_CONFIGS["full_valid_model"].features)
        Xc = X[cols]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        means = []
        for train_idx, _ in cv.split(Xc, y):
            pipe = build_pipeline("logreg", "full_valid_model", seed=0)
            pipe.fit(Xc.iloc[train_idx], y[train_idx])
            means.append(pipe.named_steps["scale"].mean_.copy())
        stacked = np.vstack(means)
        assert not np.allclose(stacked[0], stacked[1]), (
            "scaler means identical across folds: preprocessing may have been "
            "fitted on the full dataset"
        )

    def test_scaler_mean_matches_training_fold_not_full_data(self, X, y):
        cols = list(FEATURE_CONFIGS["full_valid_model"].features)
        Xc = X[cols]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        train_idx, _ = next(iter(cv.split(Xc, y)))

        pipe = build_pipeline("logreg", "full_valid_model", seed=0)
        pipe.fit(Xc.iloc[train_idx], y[train_idx])

        imputed_train = pipe.named_steps["impute"].transform(
            pipe.named_steps["select"].transform(Xc.iloc[train_idx])
        )
        expected = imputed_train.mean(axis=0)
        np.testing.assert_allclose(pipe.named_steps["scale"].mean_, expected, rtol=1e-9)

        full_imputed = pipe.named_steps["impute"].transform(
            pipe.named_steps["select"].transform(Xc)
        )
        assert not np.allclose(
            pipe.named_steps["scale"].mean_, full_imputed.mean(axis=0), rtol=1e-9
        ), "scaler appears to have been fitted on the complete dataset"

    def test_imputer_uses_training_fold_median_only(self, X, y):
        """grf holds the only missing cell; its imputed value must be fold-local."""
        pipe_a = build_pipeline("logreg", "leaky_model", seed=0)
        pipe_b = build_pipeline("logreg", "leaky_model", seed=0)
        pipe_a.fit(X.iloc[:100], y[:100])
        pipe_b.fit(X.iloc[100:], y[100:])
        stats_a = pipe_a.named_steps["impute"].statistics_
        stats_b = pipe_b.named_steps["impute"].statistics_
        assert not np.allclose(stats_a, stats_b), (
            "imputer statistics identical on disjoint halves: likely fitted globally"
        )


class TestOutcomeProxyDetection:
    def test_affected_is_detected_as_a_deterministic_proxy(self, clean_result):
        from ckd.features.encoders import assert_no_target_correlation_leak

        clean = clean_result.clean.drop(columns=["source_csv_line", "class"])
        proxies = assert_no_target_correlation_leak(clean, clean_result.target.to_numpy())
        assert "affected" in proxies

    def test_no_valid_feature_is_a_deterministic_proxy(self, clean_result):
        from ckd.features.encoders import assert_no_target_correlation_leak

        clean = clean_result.clean.drop(columns=["source_csv_line", "class"])
        proxies = set(assert_no_target_correlation_leak(clean, clean_result.target.to_numpy()))
        for name in VALID_CONFIGS:
            overlap = proxies & set(FEATURE_CONFIGS[name].features)
            assert not overlap, f"{name} contains deterministic outcome proxies {overlap}"

    def test_always_forbidden_set_is_what_we_think_it_is(self):
        assert ALWAYS_FORBIDDEN == {"class", "affected"}
        assert POST_DIAGNOSIS == {"stage", "grf"}
