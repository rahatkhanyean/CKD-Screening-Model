"""Group-aware outer folds (added for the whole-procedure bootstrap).

A bootstrap sample contains the same patient several times. Splitting
those copies across folds would let a model be evaluated on a patient it
was trained on - the identity leakage this project exists to prevent - and
would make each replicate's estimate optimistic, narrowing the very
interval stage 15 is meant to widen.

These tests assert that (a) passing ``groups`` actually keeps copies
together, (b) omitting it leaves the reference behaviour untouched, and
(c) the leakage it prevents is real and detectable.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from ckd.config import load_config
from ckd.models.nested_cv import NestedCVSettings, run_nested_cv


def _settings(seed: int) -> NestedCVSettings:
    return NestedCVSettings(
        seed=seed, outer_folds=5, inner_folds=4, repeats=1,
        inner_scoring="roc_auc", target_sensitivity=0.9, n_jobs=1,
    )


@pytest.fixture(scope="module")
def resample(X, y):
    """A stratified bootstrap resample with its original-index groups."""
    rng = np.random.default_rng(20240517)
    parts = [rng.choice(np.flatnonzero(y == label),
                        size=int((y == label).sum()), replace=True)
             for label in (0, 1)]
    rows = np.concatenate(parts)
    return X.iloc[rows].reset_index(drop=True), y[rows], rows


class TestGroupsKeepCopiesTogether:
    def test_bootstrap_sample_really_contains_duplicates(self, resample):
        """Guard the premise: if there were no duplicates the grouping
        would be pointless and this test file would be misleading."""
        _X_b, _y_b, rows = resample
        assert len(rows) > len(np.unique(rows))

    def test_no_patient_appears_in_both_train_and_test(self, X, y, resample):
        X_b, y_b, rows = resample
        cfg = load_config()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds, folds = run_nested_cv(
                X_b, y_b, model_names=["logreg"], config_names=["low_cost_model"],
                calibration_methods=["none"], settings=_settings(int(cfg["seed"])),
                verbose=0, groups=rows,
            )
        # Each row is predicted exactly once, so the test-fold membership of
        # every row is recoverable; map it back to original patient identity.
        by_fold = preds.groupby("fold")["sample_index"].apply(
            lambda s: set(rows[s.to_numpy()])
        )
        seen: set[int] = set()
        for fold, patients in by_fold.items():
            assert not (patients & seen), (
                f"fold {fold} shares original patients with an earlier fold"
            )
            seen |= patients

    def test_every_row_predicted_exactly_once(self, resample):
        X_b, y_b, rows = resample
        cfg = load_config()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds, _ = run_nested_cv(
                X_b, y_b, model_names=["logreg"], config_names=["low_cost_model"],
                calibration_methods=["none"], settings=_settings(int(cfg["seed"])),
                verbose=0, groups=rows,
            )
        assert len(preds) == len(y_b)
        assert preds["sample_index"].nunique() == len(y_b)


class TestUngroupedBehaviourUnchanged:
    def test_groups_none_matches_the_reference_path(self, X, y):
        """Passing no groups must reproduce the analysis as it was: this is
        what keeps every published number valid after the change."""
        cfg = load_config()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds, _ = run_nested_cv(
                X, y, model_names=["logreg"], config_names=["low_cost_model"],
                calibration_methods=["none"], settings=_settings(int(cfg["seed"])),
                verbose=0,
            )
        from ckd.config import project_root

        reference = pd.read_csv(
            project_root() / "data" / "processed" / "cv_predictions.csv.gz"
        )
        reference = reference[
            (reference["config"] == "low_cost_model")
            & (reference["model"] == "logreg")
            & (reference["calibration"] == "none")
            & (reference["repeat"] == 0)
        ]
        merged = preds.merge(
            reference, on=["config", "model", "calibration", "repeat", "sample_index"],
            suffixes=("_new", "_ref"),
        )
        assert len(merged) == len(y)
        assert np.abs(merged["y_prob_new"] - merged["y_prob_ref"]).max() < 1e-12


class TestTheLeakageIsStructural:
    """Why grouping is required, tested as a structural fact.

    An earlier version of this file asserted that an ungrouped bootstrap
    replicate must score *higher* than a grouped one. That claim did not
    survive contact with the data: on the reference resample the grouped
    run scored 0.9923 against the ungrouped 0.9876. Two reasons it was a
    bad test - StratifiedGroupKFold does not merely remove leakage, it
    also changes fold composition, so the comparison is confounded; and
    at a discrimination ceiling of ~0.99 a single draw cannot resolve a
    difference of this size in either direction.

    The justification for grouping does not rest on that empirical claim.
    It rests on a definitional one, which is what is tested here: without
    grouping, copies of the same patient demonstrably land on both sides
    of a split, so a model is evaluated on a patient it was trained on.
    That is disqualifying in this project regardless of which way it moves
    a score.
    """

    def test_without_grouping_duplicated_patients_span_the_split(self, resample):
        from sklearn.model_selection import StratifiedKFold

        X_b, y_b, rows = resample
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        contaminated = 0
        for train_idx, test_idx in splitter.split(X_b, y_b):
            train_patients = set(rows[train_idx])
            test_patients = set(rows[test_idx])
            contaminated += len(train_patients & test_patients)
        assert contaminated > 0, (
            "no duplicated patient spanned a split - if this is ever true, "
            "the grouping requirement should be revisited"
        )

    def test_with_grouping_no_duplicated_patient_spans_the_split(self, resample):
        from sklearn.model_selection import StratifiedGroupKFold

        X_b, y_b, rows = resample
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
        for train_idx, test_idx in splitter.split(X_b, y_b, rows):
            assert not (set(rows[train_idx]) & set(rows[test_idx]))
