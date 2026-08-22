"""Tests for the metric, threshold and calibration implementations."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ckd.evaluation.bootstrap import calibration_curve_points
from ckd.evaluation.metrics import (
    all_metrics,
    calibration_slope_intercept,
    expected_calibration_error,
    logit,
    threshold_metrics,
)
from ckd.evaluation.thresholds import (
    net_benefit,
    threshold_for_target_sensitivity,
    youden_threshold,
)


@pytest.fixture
def toy():
    rng = np.random.default_rng(7)
    n = 500
    y = rng.binomial(1, 0.4, n)
    p = np.clip(0.2 + 0.55 * y + rng.normal(0, 0.18, n), 0.001, 0.999)
    return y, p


class TestAgainstSklearn:
    def test_roc_auc_matches_sklearn(self, toy):
        y, p = toy
        assert all_metrics(y, p)["roc_auc"] == pytest.approx(roc_auc_score(y, p))

    def test_pr_auc_matches_sklearn(self, toy):
        y, p = toy
        assert all_metrics(y, p)["pr_auc"] == pytest.approx(average_precision_score(y, p))

    def test_brier_matches_sklearn(self, toy):
        y, p = toy
        assert all_metrics(y, p)["brier"] == pytest.approx(brier_score_loss(y, p))

    @pytest.mark.parametrize("thr", [0.2, 0.5, 0.8])
    def test_threshold_metrics_match_sklearn(self, toy, thr):
        y, p = toy
        m = threshold_metrics(y, p, thr)
        pred = (p >= thr).astype(int)
        assert m.sensitivity == pytest.approx(recall_score(y, pred))
        assert m.precision == pytest.approx(precision_score(y, pred, zero_division=0))
        assert m.f1 == pytest.approx(f1_score(y, pred, zero_division=0))
        assert m.balanced_accuracy == pytest.approx(balanced_accuracy_score(y, pred))

    def test_specificity_equals_recall_of_the_negative_class(self, toy):
        y, p = toy
        m = threshold_metrics(y, p, 0.5)
        pred = (p >= 0.5).astype(int)
        assert m.specificity == pytest.approx(recall_score(1 - y, 1 - pred))


class TestConfusionMatrixIdentities:
    def test_counts_sum_to_n(self, toy):
        y, p = toy
        m = threshold_metrics(y, p, 0.5)
        assert m.tp + m.fp + m.tn + m.fn == len(y)

    def test_npv_definition(self, toy):
        y, p = toy
        m = threshold_metrics(y, p, 0.5)
        assert m.npv == pytest.approx(m.tn / (m.tn + m.fn))

    def test_threshold_zero_flags_everyone(self, toy):
        y, p = toy
        m = threshold_metrics(y, p, 0.0)
        assert m.sensitivity == 1.0
        assert m.specificity == 0.0
        assert m.fn == 0


class TestCalibration:
    def test_perfectly_calibrated_predictions_give_slope_one_intercept_zero(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.05, 0.95, 20000)
        y = rng.binomial(1, p)
        slope, intercept = calibration_slope_intercept(y, p)
        assert slope == pytest.approx(1.0, abs=0.08)
        assert intercept == pytest.approx(0.0, abs=0.08)

    def test_overconfident_predictions_give_slope_below_one(self):
        """Sharpening the linear predictor must reduce the estimated slope."""
        rng = np.random.default_rng(1)
        p_true = rng.uniform(0.05, 0.95, 20000)
        y = rng.binomial(1, p_true)
        sharp = 1.0 / (1.0 + np.exp(-2.0 * logit(p_true)))
        slope, _ = calibration_slope_intercept(y, sharp)
        assert slope < 0.75

    def test_underconfident_predictions_give_slope_above_one(self):
        rng = np.random.default_rng(2)
        p_true = rng.uniform(0.05, 0.95, 20000)
        y = rng.binomial(1, p_true)
        flat = 1.0 / (1.0 + np.exp(-0.5 * logit(p_true)))
        slope, _ = calibration_slope_intercept(y, flat)
        assert slope > 1.5

    def test_systematic_underestimation_gives_positive_intercept(self):
        rng = np.random.default_rng(3)
        p_true = rng.uniform(0.1, 0.9, 20000)
        y = rng.binomial(1, p_true)
        shifted = 1.0 / (1.0 + np.exp(-(logit(p_true) - 1.0)))
        _, intercept = calibration_slope_intercept(y, shifted)
        assert intercept == pytest.approx(1.0, abs=0.15)

    def test_single_class_returns_nan(self):
        slope, intercept = calibration_slope_intercept(np.ones(50), np.full(50, 0.7))
        assert np.isnan(slope) and np.isnan(intercept)

    def test_constant_predictions_return_nan_slope(self):
        y = np.array([0, 1] * 50)
        slope, _ = calibration_slope_intercept(y, np.full(100, 0.5))
        assert np.isnan(slope)

    def test_ece_is_zero_for_perfect_calibration(self):
        rng = np.random.default_rng(4)
        p = rng.uniform(0.05, 0.95, 50000)
        y = rng.binomial(1, p)
        assert expected_calibration_error(y, p, n_bins=10) < 0.02

    def test_calibration_curve_bins_cover_all_patients(self, toy):
        y, p = toy
        curve = calibration_curve_points(y, p, n_bins=10)
        assert curve["n"].sum() == len(y)
        assert (curve["ci_low"] <= curve["observed_rate"] + 1e-9).all()
        assert (curve["ci_high"] >= curve["observed_rate"] - 1e-9).all()


class TestThresholdPolicies:
    def test_target_sensitivity_is_achieved(self, toy):
        y, p = toy
        thr = threshold_for_target_sensitivity(y, p, 0.90)
        m = threshold_metrics(y, p, thr)
        assert m.sensitivity >= 0.90

    def test_target_sensitivity_picks_the_highest_qualifying_threshold(self, toy):
        """Any higher threshold must fail the sensitivity constraint."""
        y, p = toy
        thr = threshold_for_target_sensitivity(y, p, 0.90)
        higher = p[p > thr]
        if higher.size:
            worse = threshold_metrics(y, p, float(higher.min()))
            assert worse.sensitivity < 0.90

    def test_perfect_separation_threshold(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        thr = threshold_for_target_sensitivity(y, p, 1.0)
        m = threshold_metrics(y, p, thr)
        assert m.sensitivity == 1.0
        assert m.specificity == 1.0

    def test_youden_threshold_maximises_j(self, toy):
        y, p = toy
        thr = youden_threshold(y, p)
        m = threshold_metrics(y, p, thr)
        best = m.sensitivity + m.specificity - 1
        for other in np.linspace(0.01, 0.99, 99):
            mo = threshold_metrics(y, p, other)
            assert mo.sensitivity + mo.specificity - 1 <= best + 1e-9

    def test_net_benefit_of_a_useless_model_is_not_positive_vs_treat_all(self):
        rng = np.random.default_rng(5)
        y = rng.binomial(1, 0.3, 2000)
        p = rng.uniform(0, 1, 2000)
        nb = net_benefit(y, p, 0.3)
        assert nb == pytest.approx(0.0, abs=0.05)


class TestAllMetricsContract:
    def test_extra_threshold_metrics_are_prefixed(self, toy):
        y, p = toy
        m = all_metrics(y, p, threshold=0.5, extra_thresholds={"screening": 0.2})
        assert "screening_sensitivity" in m
        assert m["screening_threshold"] == 0.2
        assert m["screening_sensitivity"] >= m["sensitivity"]

    def test_single_class_input_does_not_crash(self):
        m = all_metrics(np.ones(10, dtype=int), np.full(10, 0.6))
        assert np.isnan(m["roc_auc"])
        assert np.isfinite(m["brier"])


class TestSeparationHandling:
    """Under complete separation the calibration slope does not exist.

    This matters on the real data: some valid configurations separate the
    classes perfectly out of fold, and an optimiser will happily return a
    slope of several hundred. Reporting that as a calibration measurement
    would be meaningless, so it must come back as undefined.
    """

    def test_perfect_separation_is_detected(self):
        from ckd.evaluation.metrics import separates_perfectly

        y = np.array([0, 0, 0, 1, 1, 1])
        assert separates_perfectly(y, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]))
        assert separates_perfectly(y, np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1]))
        assert not separates_perfectly(y, np.array([0.1, 0.2, 0.8, 0.3, 0.7, 0.9]))

    def test_separation_detector_needs_both_classes(self):
        from ckd.evaluation.metrics import separates_perfectly

        assert not separates_perfectly(np.ones(5, dtype=int), np.arange(5.0))

    def test_slope_is_undefined_under_perfect_separation(self):
        y = np.array([0] * 20 + [1] * 20)
        p = np.concatenate([np.full(20, 0.05), np.full(20, 0.95)])
        slope, intercept = calibration_slope_intercept(y, p)
        assert np.isnan(slope), "a separated fit must not report a finite slope"
        assert np.isfinite(intercept), "the intercept remains identified"

    def test_slope_is_undefined_under_quasi_separation(self):
        rng = np.random.default_rng(0)
        y = np.array([0] * 40 + [1] * 40)
        p = np.concatenate([
            rng.uniform(0.001, 0.02, 40),
            rng.uniform(0.98, 0.999, 40),
        ])
        slope, _ = calibration_slope_intercept(y, p)
        assert np.isnan(slope)

    def test_ordinary_overlapping_data_still_yields_a_slope(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(0.05, 0.95, 4000)
        y = rng.binomial(1, p)
        slope, _ = calibration_slope_intercept(y, p)
        assert np.isfinite(slope)
        assert 0.5 < slope < 1.6

    def test_all_metrics_survives_a_separated_input(self):
        y = np.array([0] * 10 + [1] * 10)
        p = np.concatenate([np.full(10, 0.1), np.full(10, 0.9)])
        m = all_metrics(y, p, threshold=0.5)
        assert m["roc_auc"] == 1.0
        assert np.isnan(m["calibration_slope"])
        assert np.isfinite(m["brier"])
