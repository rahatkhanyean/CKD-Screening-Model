"""Alternative-encoding safety and the reference-run regression gate.

Two guarantees:

1. **Every encoding is leakage-safe by the same argument as the reference
   one.** Each is a pure function of one cell's label and the published bin
   list, so re-encoding an arbitrary subset of rows must reproduce exactly
   the values that subset had in the full encoding. If that failed, the
   encoding would be using cross-row information and could not legitimately
   be applied before splitting.

2. **The ``feature_override`` mechanism cannot be used to widen access.**
   It changes only which columns the selector passes on; both guards still
   run first, and every overridden name must trace back to a declared
   feature.

Plus a regression gate on the reference predictions, which documents an
honest limit: agreement across an environment rebuild is to ~1e-16, not
bit-for-bit. See ``test_reference_predictions_reproduce_to_tolerance``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ckd.config import project_root
from ckd.data.encodings import ENCODINGS, encode, encoded_feature_names
from ckd.features.configs import FORBIDDEN_IN_VALID, get_config
from ckd.features.encoders import LeakageError
from ckd.models.pipeline import (
    assert_pipeline_is_clean,
    build_pipeline,
    pipeline_feature_names,
)

LOW_COST = list(get_config("low_cost_model").features)


@pytest.fixture(scope="module")
def bin_map(clean_result):
    return clean_result.bin_map


class TestStatelessness:
    @pytest.mark.parametrize("method", ENCODINGS)
    def test_subset_encoding_matches_full_encoding(self, clean_result, bin_map, method):
        """The property that makes pre-split encoding legitimate."""
        full = encode(clean_result.clean, bin_map, LOW_COST, method)
        rng = np.random.default_rng(0)
        for _ in range(5):
            rows = rng.choice(len(clean_result.clean), size=37, replace=False)
            subset = clean_result.clean.iloc[rows]
            partial = encode(subset, bin_map, LOW_COST, method)
            expected = full.iloc[rows]
            assert list(partial.columns) == list(expected.columns)
            assert np.allclose(
                partial.to_numpy(dtype=float),
                expected.to_numpy(dtype=float),
                equal_nan=True,
            )

    @pytest.mark.parametrize("method", ENCODINGS)
    def test_encoding_ignores_row_order(self, clean_result, bin_map, method):
        full = encode(clean_result.clean, bin_map, LOW_COST, method)
        shuffled = clean_result.clean.iloc[::-1]
        reversed_encoding = encode(shuffled, bin_map, LOW_COST, method)
        assert np.allclose(
            reversed_encoding.to_numpy(dtype=float),
            full.iloc[::-1].to_numpy(dtype=float),
            equal_nan=True,
        )

    @pytest.mark.parametrize("method", ENCODINGS)
    def test_no_outcome_column_is_touched(self, clean_result, bin_map, method):
        out = encode(clean_result.clean, bin_map, LOW_COST, method)
        for name in out.columns:
            assert name.split("__", 1)[0] not in FORBIDDEN_IN_VALID


class TestEncodingShapes:
    def test_scalar_encodings_keep_one_column_per_feature(self, clean_result, bin_map):
        for method in ("midpoint", "bin_index", "rank_normal"):
            out = encode(clean_result.clean, bin_map, LOW_COST, method)
            assert list(out.columns) == LOW_COST

    def test_onehot_expands_and_names_match_helper(self, clean_result, bin_map):
        out = encode(clean_result.clean, bin_map, LOW_COST, "onehot")
        names = encoded_feature_names(bin_map, LOW_COST, "onehot")
        assert list(out.columns) == names
        assert len(names) > len(LOW_COST)

    def test_onehot_rows_sum_to_one_where_observed(self, clean_result, bin_map):
        out = encode(clean_result.clean, bin_map, LOW_COST, "onehot")
        for feature in LOW_COST:
            block = out[[c for c in out.columns if c.split("__", 1)[0] == feature]]
            sums = block.sum(axis=1, skipna=False)
            observed = clean_result.clean[feature].notna().to_numpy()
            assert np.allclose(sums.to_numpy()[observed], 1.0)

    def test_bin_index_is_monotone_in_midpoint(self, clean_result, bin_map):
        """Order is preserved: the two encodings must rank rows identically."""
        idx = encode(clean_result.clean, bin_map, LOW_COST, "bin_index")
        mid = encode(clean_result.clean, bin_map, LOW_COST, "midpoint")
        for feature in LOW_COST:
            a = idx[feature].to_numpy(float)
            b = mid[feature].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 3:
                continue
            corr = np.corrcoef(pd.Series(a[ok]).rank(), pd.Series(b[ok]).rank())[0, 1]
            assert corr > 0.999, (feature, corr)

    def test_unknown_encoding_raises(self, clean_result, bin_map):
        with pytest.raises(ValueError, match="unknown encoding"):
            encode(clean_result.clean, bin_map, LOW_COST, "not_an_encoding")


class TestFeatureOverrideCannotWidenAccess:
    def test_override_still_traces_to_declared_features(self, bin_map):
        names = encoded_feature_names(bin_map, LOW_COST, "onehot")
        pipe = build_pipeline("svm", "low_cost_model", 1, feature_override=names)
        assert_pipeline_is_clean(pipe, "low_cost_model")
        assert pipeline_feature_names(pipe) == names

    def test_override_with_undeclared_column_is_rejected(self):
        pipe = build_pipeline("svm", "low_cost_model", 1,
                              feature_override=LOW_COST + ["hemo"])
        with pytest.raises(AssertionError, match="not declared"):
            assert_pipeline_is_clean(pipe, "low_cost_model")

    def test_override_cannot_hide_a_prohibited_column_behind_a_suffix(self):
        """A suffixed prohibited column must still be caught by base name."""
        pipe = build_pipeline("svm", "low_cost_model", 1,
                              feature_override=LOW_COST + ["stage__s3"])
        with pytest.raises(AssertionError):
            assert_pipeline_is_clean(pipe, "low_cost_model")

    def test_guard_still_raises_on_a_prohibited_column_at_fit(self, clean_result, bin_map):
        names = encoded_feature_names(bin_map, LOW_COST, "onehot")
        pipe = build_pipeline("svm", "low_cost_model", 1, feature_override=names)
        frame = encode(clean_result.clean, bin_map, LOW_COST, "onehot")
        frame["stage"] = clean_result.clean["stage"].to_numpy()
        with pytest.raises(LeakageError):
            pipe.fit(frame, clean_result.target.to_numpy())


class TestReferenceRunRegression:
    def test_reference_predictions_reproduce_to_tolerance(self):
        """The refactor must not perturb the reference run.

        Bit-identity is NOT asserted, and the reason is documented rather
        than hidden: rebuilding the environment from the pinned
        requirements reproduces the stored predictions only to ~1e-16,
        because BLAS summation order depends on the installed build. This
        was verified to be pre-existing - the unmodified code shows the
        same 1.1e-16 discrepancy - so it reflects the environment, not the
        analysis. Determinism *within* an environment is asserted
        separately by tests/test_reproducibility.py.
        """
        import warnings

        from ckd.config import load_config
        from ckd.data.clean import clean_dataset, feature_matrix
        from ckd.models.nested_cv import NestedCVSettings, run_nested_cv

        path = project_root() / "data" / "processed" / "cv_predictions.csv.gz"
        if not path.is_file():
            pytest.skip("reference predictions not present")

        cfg = load_config()
        result = clean_dataset()
        X = feature_matrix(result)
        y = result.target.to_numpy()
        settings = NestedCVSettings(
            seed=int(cfg["seed"]), outer_folds=5, inner_folds=4, repeats=1,
            inner_scoring="roc_auc", target_sensitivity=0.9, n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fresh, _ = run_nested_cv(
                X, y, model_names=["svm"], config_names=["low_cost_model"],
                calibration_methods=["none"], settings=settings, verbose=0,
            )
        reference = pd.read_csv(path)
        reference = reference[
            (reference["config"] == "low_cost_model")
            & (reference["model"] == "svm")
            & (reference["calibration"] == "none")
            & (reference["repeat"] == 0)
        ]
        merged = fresh.merge(
            reference,
            on=["config", "model", "calibration", "repeat", "sample_index"],
            suffixes=("_new", "_ref"),
        )
        assert len(merged) == len(y)
        delta = np.abs(merged["y_prob_new"] - merged["y_prob_ref"]).max()
        assert delta < 1e-12, f"reference predictions drifted by {delta:.3e}"
