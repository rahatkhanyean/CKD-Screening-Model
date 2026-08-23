"""Benchmark Informativeness Diagnostics: correctness on known answers.

A newly proposed metric earns nothing from being reported on real data,
where the right answer is unknown. These tests construct cases where the
answer is determined by the construction, and check the implementation
returns it. Only then are the values it produces on this benchmark worth
reading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from ckd.evaluation.informativeness import (
    FLAG_THRESHOLDS,
    best_single_predictor,
    flag,
    folded_auc,
    fraction_to_reach,
    multivariable_lift,
    standardised_auc,
    standardised_sensitivity,
    weighted_auc,
)


@pytest.fixture
def separable():
    """40 cases and 40 controls with one perfectly separating column and
    one pure-noise column."""
    rng = np.random.default_rng(0)
    y = np.array([1] * 40 + [0] * 40)
    X = pd.DataFrame({
        "perfect": np.concatenate([rng.uniform(2, 3, 40), rng.uniform(0, 1, 40)]),
        "noise": rng.normal(size=80),
    })
    return y, X


class TestFoldedAUC:
    def test_orientation_is_chosen_to_favour_the_score(self):
        y = np.array([1, 1, 0, 0])
        ascending = np.array([0.9, 0.8, 0.2, 0.1])
        assert folded_auc(y, ascending) == 1.0
        assert folded_auc(y, -ascending) == 1.0   # reversed, same value

    def test_chance_score_is_half(self):
        y = np.array([1, 0, 1, 0])
        assert folded_auc(y, np.array([1.0, 1.0, 1.0, 1.0])) == 0.5


class TestBestSinglePredictor:
    def test_finds_the_separating_column(self, separable):
        y, X = separable
        name, auc = best_single_predictor(X, y)
        assert name == "perfect"
        assert auc == pytest.approx(1.0)

    def test_ignores_columns_with_too_little_data(self):
        y = np.array([1] * 20 + [0] * 20)
        X = pd.DataFrame({
            "mostly_missing": [np.nan] * 35 + [1.0, 2.0, 3.0, 4.0, 5.0],
            "usable": np.concatenate([np.ones(20), np.zeros(20)]),
        })
        name, _auc = best_single_predictor(X, y)
        assert name == "usable"


class TestMultivariableLift:
    def test_zero_when_the_model_only_matches_its_best_feature(self, separable):
        """The property the metric exists for: a model that reproduces one
        raw column has added nothing, and must score zero."""
        y, X = separable
        values = X["perfect"].to_numpy()
        prob = (values - values.min()) / (values.max() - values.min())
        result = multivariable_lift(y, prob, X, n_boot=200, seed=1)
        assert result.best_feature == "perfect"
        assert abs(result.lift) < 1e-9

    def test_large_when_the_model_beats_every_single_column(self):
        rng = np.random.default_rng(2)
        y = np.array([1] * 40 + [0] * 40)
        # Each column alone is uninformative; their XOR-like sum is not.
        a = np.concatenate([rng.uniform(0, 1, 40), rng.uniform(0, 1, 40)])
        X = pd.DataFrame({"a": a, "b": rng.uniform(size=80)})
        perfect = np.concatenate([np.ones(40), np.zeros(40)])
        result = multivariable_lift(y, perfect, X, n_boot=200, seed=1)
        assert result.lift > 0.3

    def test_interval_brackets_the_estimate(self, separable):
        y, X = separable
        prob = np.concatenate([np.full(40, 0.9), np.full(40, 0.1)])
        result = multivariable_lift(y, prob, X, n_boot=400, seed=3)
        assert result.lift_ci_low <= result.lift <= result.lift_ci_high


class TestWeightedAUC:
    def test_uniform_weights_reproduce_sklearn(self):
        rng = np.random.default_rng(4)
        y = rng.integers(0, 2, 200)
        score = rng.uniform(size=200)
        assert weighted_auc(y, score, np.ones(200)) == pytest.approx(
            roc_auc_score(y, score), abs=1e-9
        )

    def test_ties_count_as_half(self):
        y = np.array([1, 0])
        assert weighted_auc(y, np.array([0.5, 0.5]), np.ones(2)) == 0.5

    def test_zero_weight_cases_are_excluded(self):
        y = np.array([1, 1, 0])
        score = np.array([0.9, 0.1, 0.5])       # one case above, one below
        weights = np.array([1.0, 0.0, 1.0])     # drop the low-scoring case
        assert weighted_auc(y, score, weights) == 1.0


class TestStandardisation:
    def test_no_shift_when_target_equals_observed_mix(self):
        rng = np.random.default_rng(5)
        y = np.array([1] * 60 + [0] * 40)
        score = np.concatenate([rng.uniform(0.4, 1.0, 60), rng.uniform(0, 0.6, 40)])
        stage = pd.Series(["s1"] * 30 + ["s3"] * 30 + [None] * 40)
        result = standardised_auc(y, score, stage, {"s1": 0.5, "s3": 0.5})
        assert abs(result.shift) < 1e-9

    def test_reweighting_toward_hard_cases_lowers_the_estimate(self):
        """Cases in the 'easy' stratum outrank every control; those in the
        'hard' stratum do not. Standardising toward the hard stratum must
        reduce the estimate."""
        y = np.array([1] * 20 + [0] * 20)
        score = np.concatenate([
            np.full(10, 0.99),   # easy cases
            np.full(10, 0.10),   # hard cases, below every control
            np.full(20, 0.50),   # controls
        ])
        stage = pd.Series(["easy"] * 10 + ["hard"] * 10 + [None] * 20)
        balanced = standardised_auc(y, score, stage, {"easy": 0.5, "hard": 0.5})
        hard_heavy = standardised_auc(y, score, stage, {"easy": 0.1, "hard": 0.9})
        assert hard_heavy.standardised_auc < balanced.standardised_auc

    def test_effective_cases_falls_when_weights_concentrate(self):
        y = np.array([1] * 20 + [0] * 10)
        score = np.linspace(0, 1, 30)
        stage = pd.Series(["a"] * 10 + ["b"] * 10 + [None] * 10)
        even = standardised_auc(y, score, stage, {"a": 0.5, "b": 0.5})
        skewed = standardised_auc(y, score, stage, {"a": 0.98, "b": 0.02})
        assert skewed.effective_cases < even.effective_cases


class TestStandardisedSensitivity:
    def test_detects_case_mix_dependence_that_auc_hides(self):
        """The finding that motivated adding this function: a model can keep
        a perfect AUC while missing an entire severity stratum, because AUC
        only asks whether cases outrank controls."""
        y = np.array([1] * 20 + [0] * 20)
        score = np.concatenate([
            np.full(10, 0.90),   # advanced cases: detected at 0.5
            np.full(10, 0.30),   # early cases: MISSED at 0.5, still outrank
            np.full(20, 0.05),   # controls
        ])
        stage = pd.Series(["advanced"] * 10 + ["early"] * 10 + [None] * 20)
        auc = standardised_auc(y, score, stage, {"early": 0.9, "advanced": 0.1})
        sens = standardised_sensitivity(
            y, score, stage, {"early": 0.9, "advanced": 0.1}, threshold=0.5
        )
        assert auc.standardised_auc == pytest.approx(1.0)   # AUC sees nothing
        assert sens.observed_sensitivity == pytest.approx(0.5)
        assert sens.standardised_sensitivity == pytest.approx(0.1)
        assert sens.shift > 0.35                            # sensitivity does

    def test_no_shift_when_target_equals_observed(self):
        y = np.array([1] * 20 + [0] * 10)
        score = np.concatenate([np.full(20, 0.8), np.full(10, 0.2)])
        stage = pd.Series(["a"] * 10 + ["b"] * 10 + [None] * 10)
        result = standardised_sensitivity(y, score, stage, {"a": 0.5, "b": 0.5})
        assert abs(result.shift) < 1e-9


class TestFractionToReach:
    def test_finds_the_first_fraction_over_the_threshold(self):
        fractions = np.array([0.1, 0.2, 0.5, 1.0])
        scores = np.array([0.60, 0.90, 0.92, 0.93])
        # ceiling 0.93 -> gain 0.43 -> 90% threshold = 0.5 + 0.387 = 0.887
        assert fraction_to_reach(fractions, scores, 0.90) == 0.2

    def test_nan_when_never_above_chance(self):
        fractions = np.array([0.5, 1.0])
        assert np.isnan(fraction_to_reach(fractions, np.array([0.5, 0.49])))

    def test_a_saturated_curve_reports_a_small_fraction(self):
        fractions = np.array([0.1, 0.5, 1.0])
        scores = np.array([0.97, 0.98, 0.99])
        assert fraction_to_reach(fractions, scores) == 0.1


class TestFlags:
    def test_thresholds_are_applied_as_documented(self):
        raised = flag({
            "multivariable_lift": FLAG_THRESHOLDS["multivariable_lift"] - 0.01,
            "fraction_to_reach_90pct":
                FLAG_THRESHOLDS["fraction_to_reach_90pct"] - 0.01,
            "standardisation_shift":
                FLAG_THRESHOLDS["standardisation_shift"] + 0.01,
        })
        assert raised["low_multivariable_lift"]
        assert raised["saturated_by_subsample"]
        assert raised["case_mix_dependent_auc"]

    def test_healthy_benchmark_raises_nothing(self):
        raised = flag({
            "multivariable_lift": 0.30,
            "fraction_to_reach_90pct": 0.85,
            "standardisation_shift": 0.001,
            "sensitivity_standardisation_shift": 0.002,
        })
        assert not any(raised.values())

    def test_missing_diagnostics_are_simply_absent(self):
        assert flag({}) == {}
        assert "saturated_by_subsample" not in flag({"multivariable_lift": 0.1})


class TestOnThisBenchmark:
    """The values the protocol produces here, pinned so a change is noticed."""

    def test_summary_exists_and_flags_the_benchmark(self):
        from ckd.config import project_root

        path = project_root() / "reports" / "tables" / "table_43_bid_summary.csv"
        if not path.is_file():
            pytest.skip("run scripts/20_benchmark_diagnostics.py")
        summary = pd.read_csv(path)
        assert (summary["n_flags_raised"] > 0).all(), (
            "every configuration of this benchmark should raise at least one "
            "flag; if none does, either the data or the thresholds changed"
        )

    def test_saturation_is_extreme_here(self):
        from ckd.config import project_root

        path = project_root() / "reports" / "tables" / "table_43_bid_saturation.csv"
        if not path.is_file():
            pytest.skip("run scripts/20_benchmark_diagnostics.py")
        curve = pd.read_csv(path)
        smallest = curve.sort_values("train_fraction").iloc[0]
        largest = curve.sort_values("train_fraction").iloc[-1]
        # A tenth of the training data already reaches within 0.03 of the end.
        assert largest["mean_roc_auc"] - smallest["mean_roc_auc"] < 0.05
