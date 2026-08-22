"""Whole-procedure bootstrap and calibration-experiment checks (Phase 4c/4d).

Both stages exist to test claims the main analysis makes about its own
uncertainty, so the tests here concentrate on the properties that would
make those claims wrong:

* a procedure bootstrap must be *wider* than the prediction-level
  bootstrap it is meant to correct - if it were narrower, it would not be
  measuring what it claims to;
* the calibration experiment must actually use more resampling than the
  headline analysis, otherwise it cannot settle anything about it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ckd.config import project_root

TABLES = project_root() / "reports" / "tables"


@pytest.fixture(scope="module")
def procedure() -> pd.DataFrame:
    path = TABLES / "table_38_procedure_bootstrap.csv"
    if not path.is_file():
        pytest.skip("run scripts/15_procedure_bootstrap.py")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def prediction_intervals() -> pd.DataFrame:
    return pd.read_csv(TABLES / "table_11_bootstrap_confidence_intervals.csv")


@pytest.fixture(scope="module")
def calibration() -> pd.DataFrame:
    path = TABLES / "table_39_calibration_experiment.csv"
    if not path.is_file():
        pytest.skip("run scripts/16_calibration_experiment.py")
    return pd.read_csv(path)


class TestProcedureBootstrap:
    def test_intervals_bracket_their_point_estimate(self, procedure):
        for _, r in procedure.iterrows():
            if not np.isfinite(r["point_estimate_roc_auc"]):
                continue
            assert r["procedure_ci_low"] <= r["point_estimate_roc_auc"] + 1e-9
            assert r["procedure_ci_high"] >= r["point_estimate_roc_auc"] - 1e-9

    def test_enough_resamples_survived(self, procedure):
        assert (procedure["n_resamples_used"] >= 100).all()
        # Failures are recorded, not silently dropped.
        assert "n_failed" in procedure.columns

    def test_wider_than_the_prediction_level_bootstrap(
        self, procedure, prediction_intervals
    ):
        """The reason this stage exists: resampling predictions while
        holding the fitted models fixed understates uncertainty. If the
        honest interval were not wider, section 5.8's caveat would be
        wrong."""
        pred = prediction_intervals[
            (prediction_intervals["calibration"] == "none")
            & (prediction_intervals["metric"] == "roc_auc")
        ]
        compared = 0
        for _, r in procedure.iterrows():
            match = pred[
                (pred["config"] == r["config"]) & (pred["model"] == r["model"])
            ]
            if match.empty:
                continue
            width = float(match.iloc[0]["ci_high"] - match.iloc[0]["ci_low"])
            # A saturated cell has a degenerate interval in both schemes;
            # only compare where the prediction-level interval is non-zero.
            if width <= 1e-9:
                continue
            compared += 1
            assert r["procedure_ci_width"] >= width - 1e-6, (
                f"{r['config']}/{r['model']}: procedure width "
                f"{r['procedure_ci_width']:.4f} < prediction width {width:.4f}"
            )
        assert compared > 0, "no comparable cells found"


class TestCalibrationExperiment:
    def test_uses_more_resampling_than_the_headline_analysis(self, calibration):
        assert (calibration["n_repeats"] >= 20).all()
        assert (calibration["n_outer_test_sets"] >= 100).all()

    def test_all_three_methods_present(self, calibration):
        assert set(calibration["calibration"]) == {"none", "sigmoid", "isotonic"}

    def test_unidentified_slopes_are_counted_not_dropped(self, calibration):
        assert "n_slope_unidentified" in calibration.columns
        assert (calibration["n_slope_unidentified"] >= 0).all()

    def test_paired_comparison_is_internally_consistent(self):
        path = TABLES / "table_39_calibration_paired.csv"
        if not path.is_file():
            pytest.skip("run scripts/16_calibration_experiment.py")
        paired = pd.read_csv(path).iloc[0]
        assert 0 <= paired["isotonic_better_fraction"] <= 1
        assert paired["isotonic_better_in"] <= paired["n_pairs"]
        assert abs(
            paired["isotonic_better_in"] / paired["n_pairs"]
            - paired["isotonic_better_fraction"]
        ) < 1e-6
