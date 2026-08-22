"""Readable low-cost rule: EBM shape-function checks (Phase 1d / N8).

The shape functions are the study's answer to "explainable" in the title,
so they are checked rather than merely plotted:

* every low-cost feature has a curve, averaged over the 25 outer folds;
* the direction of each curve agrees with that feature's univariate
  association with the outcome — a GAM whose fitted effect ran opposite to
  the marginal association would be reporting a suppression artefact, and
  the report should not present it as readable without saying so;
* the comparison table places the EBM against the SVM honestly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from ckd.config import project_root
from ckd.features.configs import get_config

TABLES = project_root() / "reports" / "tables"


@pytest.fixture(scope="module")
def shapes() -> pd.DataFrame:
    path = TABLES / "table_28_ebm_shape_functions.csv"
    if not path.is_file():
        pytest.skip("run scripts/10_ebm_shapes.py")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def comparison() -> pd.DataFrame:
    path = TABLES / "table_28_ebm_vs_svm.csv"
    if not path.is_file():
        pytest.skip("run scripts/10_ebm_shapes.py")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def signed_auc(X, y) -> dict[str, float]:
    """Signed univariate AUC per low-cost feature, computed here on purpose.

    The published tables report unsigned strength only (`table_03` uses
    Cramer's V; `table_22` folds the AUC to max(auc, 1-auc)), so neither
    can say which direction a feature points. Direction is what this test
    needs, so it is derived from the encoded data directly.
    """
    from sklearn.metrics import roc_auc_score

    out = {}
    for feature in get_config("low_cost_model").features:
        values = X[feature]
        ok = values.notna().to_numpy()
        if ok.sum() < 2 or len(set(y[ok])) < 2:
            continue
        out[feature] = float(roc_auc_score(y[ok], values.to_numpy()[ok]))
    return out


class TestCoverage:
    def test_every_low_cost_feature_has_a_shape(self, shapes):
        expected = set(get_config("low_cost_model").features)
        assert set(shapes["feature"]) == expected

    def test_curves_average_all_outer_folds(self, shapes):
        assert (shapes["n_folds"] == 25).all()

    def test_contributions_are_finite(self, shapes):
        assert np.isfinite(shapes["mean_contribution"]).all()
        assert np.isfinite(shapes["sd_contribution"]).all()


class TestDirectionConsistency:
    def test_every_direction_discrepancy_is_documented(self, shapes, signed_auc):
        """A GAM term may legitimately run against its marginal association
        (suppression), but then it is NOT readable on its own. The rule is
        therefore: agree, or be named in the notes file — mirroring the
        table_20 importance-notes pattern. Silent discrepancies fail."""
        notes_path = TABLES / "table_28_ebm_shape_notes.txt"
        assert notes_path.is_file(), "shape notes file missing (run stage 10)"
        notes = notes_path.read_text(encoding="utf-8")
        undocumented = []
        for feature, grp in shapes.groupby("feature"):
            uni_auc = signed_auc.get(feature)
            if uni_auc is None or abs(uni_auc - 0.5) < 0.03:
                continue
            rho = spearmanr(grp["value"], grp["mean_contribution"]).statistic
            if (rho > 0) != (uni_auc > 0.5) and feature not in notes:
                undocumented.append((feature, round(float(rho), 3), round(uni_auc, 3)))
        assert not undocumented, (
            f"undocumented shape-direction discrepancies: {undocumented}"
        )

    def test_summary_table_agrees_with_recomputation(self, shapes, signed_auc):
        summary = pd.read_csv(TABLES / "table_28_ebm_shape_summary.csv")
        for _, r in summary.iterrows():
            grp = shapes[shapes["feature"] == r["feature"]]
            rho = spearmanr(grp["value"], grp["mean_contribution"]).statistic
            assert abs(rho - r["shape_spearman_rho"]) < 5e-3
            if r["feature"] in signed_auc:
                assert abs(signed_auc[r["feature"]] - r["signed_univariate_auc"]) < 5e-3

    def test_bp_diastolic_is_the_known_suppression_term(self, shapes):
        """Pin the reference-run finding: the binary diastolic flag runs
        opposite to its marginal association once 'bp limit' and 'htn' are
        present. If this ever changes, the report's discussion of it must
        change too."""
        summary = pd.read_csv(TABLES / "table_28_ebm_shape_summary.csv")
        disagreeing = set(summary.loc[~summary["direction_agrees"], "feature"])
        assert disagreeing == {"bp (Diastolic)"}, disagreeing

    def test_specific_gravity_is_protective(self, shapes):
        """Domain anchor: impaired urinary concentration (LOW specific
        gravity) is the CKD-associated direction, so the fitted curve must
        decrease in sg."""
        sg = shapes[shapes["feature"] == "sg"].sort_values("value")
        rho = spearmanr(sg["value"], sg["mean_contribution"]).statistic
        assert rho < 0, f"sg shape should decrease with value, rho={rho:.3f}"

    def test_albuminuria_increases_risk(self, shapes):
        al = shapes[shapes["feature"] == "al"].sort_values("value")
        rho = spearmanr(al["value"], al["mean_contribution"]).statistic
        assert rho > 0, f"al shape should increase with value, rho={rho:.3f}"


class TestComparisonTable:
    def test_contains_both_models(self, comparison):
        assert {"ebm", "svm"} <= set(comparison["model"])

    def test_only_ebm_is_marked_readable(self, comparison):
        readable = comparison[comparison["readable_shape_functions"]]
        assert set(readable["model"]) == {"ebm"}

    def test_ebm_calibration_slope_is_identified(self, comparison):
        ebm = comparison[(comparison["model"] == "ebm")
                         & (comparison["calibration"] == "isotonic")].iloc[0]
        assert np.isfinite(ebm["calibration_slope"])
