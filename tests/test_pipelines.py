"""Tests for pipeline construction, the model zoo and the nested-CV engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from ckd.features.configs import FEATURE_CONFIGS, valid_config_names
from ckd.models.nested_cv import NestedCVSettings, _fold_seed, _run_one_repeat, run_nested_cv
from ckd.models.pipeline import build_pipeline, pipeline_feature_names
from ckd.models.zoo import MODEL_SPECS, available_models, get_model, unavailable_models

MODELS = available_models()


class TestModelZoo:
    def test_no_neural_network_models_are_registered(self):
        """The sample is far too small; deep models are excluded by design."""
        banned = {"mlp", "neural", "nn", "deep", "keras", "torch", "perceptron"}
        for name, spec in MODEL_SPECS.items():
            assert not (banned & set(name.lower().split("_")))
            cls = spec.factory(0).__class__.__name__.lower() if spec.available else ""
            assert "mlp" not in cls and "perceptron" not in cls

    def test_required_model_families_are_present(self):
        for required in ("dummy", "logreg", "random_forest", "svm"):
            assert required in MODEL_SPECS

    def test_gradient_boosting_is_available(self):
        assert "xgboost" in MODEL_SPECS or "catboost" in MODEL_SPECS

    def test_unavailable_models_report_a_reason(self):
        for name, reason in unavailable_models().items():
            assert reason, f"{name} is unavailable but gives no reason"

    def test_get_model_raises_for_unknown(self):
        with pytest.raises(KeyError):
            get_model("does_not_exist")

    def test_search_spaces_are_small_enough_for_the_sample_size(self):
        """A large grid searched on ~32-patient inner folds would select on noise."""
        for name, spec in MODEL_SPECS.items():
            n = 1
            for values in spec.param_grid.values():
                n *= len(values)
            assert n <= 12, f"{name} has {n} candidates, too many for n=200"


class TestPipelineStructure:
    @pytest.mark.parametrize("model", MODELS)
    def test_pipeline_has_guards_before_everything_else(self, model):
        pipe = build_pipeline(model, "full_valid_model", seed=0)
        names = [n for n, _ in pipe.steps]
        assert names[0] == "outcome_guard"
        assert names[1] == "leakage_guard"
        assert names[2] == "select"
        assert names[-1] == "clf"

    @pytest.mark.parametrize("model", MODELS)
    def test_imputer_is_always_present(self, model):
        pipe = build_pipeline(model, "full_valid_model", seed=0)
        assert "impute" in dict(pipe.steps)

    def test_scaling_is_applied_only_where_needed(self):
        assert "scale" in dict(build_pipeline("logreg", "low_cost_model", 0).steps)
        assert "scale" in dict(build_pipeline("svm", "low_cost_model", 0).steps)
        assert "scale" not in dict(build_pipeline("random_forest", "low_cost_model", 0).steps)

    @pytest.mark.parametrize("config", valid_config_names())
    def test_selected_feature_order_is_deterministic(self, config):
        a = pipeline_feature_names(build_pipeline("logreg", config, 0))
        b = pipeline_feature_names(build_pipeline("logreg", config, 999))
        assert a == b == list(FEATURE_CONFIGS[config].features)

    def test_missing_column_raises_a_clear_error(self, X, y):
        pipe = build_pipeline("logreg", "low_cost_model", seed=0)
        cols = list(FEATURE_CONFIGS["low_cost_model"].features)
        with pytest.raises(KeyError, match="absent"):
            pipe.fit(X[cols].drop(columns=["age"]), y)


class TestNestedCVEngine:
    @pytest.fixture(scope="class")
    def settings(self):
        return NestedCVSettings(seed=123, outer_folds=5, inner_folds=4, repeats=1, n_jobs=1)

    def test_every_patient_is_predicted_exactly_once_per_repeat(self, X, y, settings):
        preds, _ = _run_one_repeat(X, y, "logreg", "low_cost_model", "none", 0, settings)
        df = pd.DataFrame(preds)
        assert len(df) == len(y)
        assert df["sample_index"].nunique() == len(y)

    def test_stored_labels_match_the_source(self, X, y, settings):
        preds, _ = _run_one_repeat(X, y, "logreg", "low_cost_model", "none", 0, settings)
        df = pd.DataFrame(preds).sort_values("sample_index")
        np.testing.assert_array_equal(df["y_true"].to_numpy(), y)

    def test_probabilities_are_valid(self, X, y, settings):
        preds, _ = _run_one_repeat(X, y, "logreg", "low_cost_model", "none", 0, settings)
        p = pd.DataFrame(preds)["y_prob"].to_numpy()
        assert np.all((p >= 0) & (p <= 1))

    def test_outer_folds_are_stratified(self, X, y, settings):
        _, info = _run_one_repeat(X, y, "dummy", "low_cost_model", "none", 0, settings)
        info = pd.DataFrame(info)
        assert set(info["n_test"]) == {40}
        assert set(info["n_train"]) == {160}

    def test_screening_threshold_is_recorded_per_fold(self, X, y, settings):
        _, info = _run_one_repeat(X, y, "logreg", "low_cost_model", "none", 0, settings)
        info = pd.DataFrame(info)
        assert info["screening_threshold"].notna().all()
        assert ((info["screening_threshold"] >= 0) & (info["screening_threshold"] <= 1)).all()

    def test_runner_restricts_columns_so_the_guard_is_satisfied(self, X, y, settings):
        """The runner must hand each pipeline only its declared columns."""
        preds, _ = _run_one_repeat(X, y, "logreg", "full_valid_model", "none", 0, settings)
        assert len(preds) == len(y)

    def test_leaky_configuration_runs_and_uses_the_leaks(self, X, y, settings):
        preds, info = _run_one_repeat(X, y, "logreg", "leaky_model", "none", 0, settings)
        assert pd.DataFrame(info)["n_features"].iloc[0] == 28

    @pytest.mark.parametrize("calibration", ["none", "sigmoid", "isotonic"])
    def test_all_calibration_methods_produce_valid_probabilities(self, X, y, settings, calibration):
        preds, _ = _run_one_repeat(X, y, "logreg", "low_cost_model", calibration, 0, settings)
        p = pd.DataFrame(preds)["y_prob"].to_numpy()
        assert np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))

    def test_grid_runner_produces_the_expected_shape(self, X, y):
        s = NestedCVSettings(seed=5, outer_folds=5, inner_folds=4, repeats=2, n_jobs=1)
        preds, info = run_nested_cv(
            X, y, ["dummy"], ["low_cost_model", "clinical_only_model"], ["none"], s, verbose=0
        )
        assert len(preds) == 2 * 2 * len(y)
        assert len(info) == 2 * 2 * 5


class TestSeeding:
    def test_fold_seeds_are_distinct(self):
        seeds = {_fold_seed(1000, r, f) for r in range(5) for f in range(5)}
        assert len(seeds) == 25

    def test_fold_seeds_are_deterministic(self):
        assert _fold_seed(7, 1, 2) == _fold_seed(7, 1, 2)

    def test_fold_seeds_stay_in_valid_range(self):
        for r in range(10):
            for f in range(10):
                s = _fold_seed(20240517, r, f)
                assert 0 <= s < 2**31 - 1
