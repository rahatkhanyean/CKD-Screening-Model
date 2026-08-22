"""Repeated nested stratified cross-validation.

Structure of one repeat
-----------------------
::

    outer StratifiedKFold(5)                      <- gives the evaluation folds
      |
      +-- inner StratifiedKFold(4) on the outer TRAINING data only
      |     GridSearchCV selects hyper-parameters (scoring: ROC-AUC)
      |
      +-- refit the selected pipeline on the outer training data
      +-- optional probability calibration, fitted on the outer training data
      |     via CalibratedClassifierCV(cv=4) - the base estimator is refitted
      |     inside that wrapper, so no calibration data overlaps the outer test
      +-- choose the screening threshold from a separate cross-validated
      |     prediction on the outer training data (never on the test fold)
      +-- predict_proba on the outer TEST fold  <- the only use of test data

Nothing - imputation, scaling, hyper-parameter choice, calibration, threshold
selection - is computed on the outer test fold or on the complete dataset.

Output
------
A single long-format table of per-patient predictions. Every metric, confidence
interval, calibration curve and figure in the study is derived from that one
table, so a reported number can always be traced back to the predictions that
produced it.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from ..features.configs import get_config
from .pipeline import assert_pipeline_is_clean, build_pipeline
from .zoo import get_model

CALIBRATION_METHODS = ("none", "sigmoid", "isotonic")


@dataclass(frozen=True)
class NestedCVSettings:
    """All knobs of the validation design, resolved from config/experiment.yaml."""

    seed: int
    outer_folds: int = 5
    inner_folds: int = 4
    repeats: int = 5
    inner_scoring: str = "roc_auc"
    target_sensitivity: float = 0.90
    n_jobs: int = -1


def _fold_seed(base: int, repeat: int, fold: int = 0, salt: int = 0) -> int:
    """Deterministic, well-separated seed for one (repeat, fold) position."""
    return int((base * 1_000_003 + repeat * 10_007 + fold * 101 + salt) % (2**31 - 1))


def _screening_threshold_from_training(
    estimator, X_train: pd.DataFrame, y_train: np.ndarray, seed: int,
    inner_folds: int, target_sensitivity: float,
) -> float:
    """Pick the screening threshold using training-fold data only.

    A fresh stratified k-fold split of the outer *training* data produces
    out-of-fold probabilities; the threshold is the highest one still meeting
    the sensitivity target on those. The outer test fold is not involved.
    """
    from ..evaluation.thresholds import threshold_for_target_sensitivity

    cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            oof = cross_val_predict(
                clone(estimator), X_train, y_train, cv=cv, method="predict_proba", n_jobs=1
            )[:, 1]
        except Exception:
            # A model that cannot be cross-validated on the training fold falls
            # back to the prespecified threshold rather than failing the run.
            return 0.5
    return threshold_for_target_sensitivity(y_train, oof, target_sensitivity)


def _run_one_repeat(
    X: pd.DataFrame,
    y: np.ndarray,
    model_name: str,
    config_name: str,
    calibration: str,
    repeat: int,
    settings: NestedCVSettings,
    feature_override: list[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Execute one full outer cross-validation pass. Returns (predictions, fold_info)."""
    spec = get_model(model_name)

    # Restrict the frame to exactly the columns this configuration declares.
    # This is the first of two independent defences: the caller narrows the
    # data, and the LeakageGuard inside the pipeline then verifies that nothing
    # prohibited survived. Passing the full frame would (correctly) raise.
    # ``feature_override`` carries encoded column names (see
    # ckd.data.encodings); assert_pipeline_is_clean checks each one traces
    # back to a declared feature, so the narrowing stays authorised.
    columns = (list(feature_override) if feature_override is not None
               else list(get_config(config_name).features))
    X = X.loc[:, columns]

    outer = StratifiedKFold(
        n_splits=settings.outer_folds,
        shuffle=True,
        random_state=_fold_seed(settings.seed, repeat, salt=1),
    )

    predictions: list[dict] = []
    fold_info: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        seed = _fold_seed(settings.seed, repeat, fold, salt=2)
        pipe = build_pipeline(model_name, config_name, seed,
                              feature_override=feature_override)
        assert_pipeline_is_clean(pipe, config_name)

        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if spec.param_grid:
                inner = StratifiedKFold(
                    n_splits=settings.inner_folds, shuffle=True, random_state=seed
                )
                search = GridSearchCV(
                    pipe,
                    param_grid=spec.param_grid,
                    scoring=settings.inner_scoring,
                    cv=inner,
                    n_jobs=1,
                    refit=True,
                    error_score=np.nan,
                )
                search.fit(X_tr, y_tr)
                best = search.best_estimator_
                best_params = {k: str(v) for k, v in search.best_params_.items()}
                inner_score = float(search.best_score_)
            else:
                best = pipe.fit(X_tr, y_tr)
                best_params = {}
                inner_score = float("nan")

            # --- calibration, fitted on the outer training fold only ---
            if calibration == "none":
                final = best
            else:
                cal_cv = StratifiedKFold(
                    n_splits=settings.inner_folds, shuffle=True, random_state=seed + 7
                )
                final = CalibratedClassifierCV(
                    clone(best), method=calibration, cv=cal_cv, ensemble=True
                )
                final.fit(X_tr, y_tr)

            thr = _screening_threshold_from_training(
                final, X_tr, y_tr, seed + 13, settings.inner_folds,
                settings.target_sensitivity,
            )

            proba = final.predict_proba(X_te)[:, 1]

        elapsed = time.perf_counter() - t0

        for pos, idx in enumerate(test_idx):
            predictions.append(
                {
                    "config": config_name,
                    "model": model_name,
                    "calibration": calibration,
                    "repeat": repeat,
                    "fold": fold,
                    "sample_index": int(idx),
                    "y_true": int(y_te[pos]),
                    "y_prob": float(proba[pos]),
                    "screening_threshold": float(thr),
                }
            )

        fold_info.append(
            {
                "config": config_name,
                "model": model_name,
                "calibration": calibration,
                "repeat": repeat,
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_features": len(get_config(config_name).features),
                "inner_best_score": inner_score,
                "screening_threshold": float(thr),
                "fit_seconds": round(elapsed, 3),
                **{f"param_{k}": v for k, v in best_params.items()},
            }
        )

    return predictions, fold_info


def run_nested_cv(
    X: pd.DataFrame,
    y: Sequence[int],
    model_names: Iterable[str],
    config_names: Iterable[str],
    calibration_methods: Iterable[str],
    settings: NestedCVSettings,
    verbose: int = 1,
    feature_override: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full grid of (configuration x model x calibration x repeat).

    Returns
    -------
    predictions:
        Long table with one row per (cell, repeat, patient).
    fold_info:
        One row per outer fold, recording the selected hyper-parameters, the
        inner-loop score and the fold-local screening threshold.
    """
    y_arr = np.asarray(y, dtype=int)

    tasks = [
        (m, c, cal, r)
        for c in config_names
        for m in model_names
        for cal in calibration_methods
        for r in range(settings.repeats)
    ]
    if verbose:
        print(
            f"[nested-cv] {len(tasks)} tasks "
            f"({settings.outer_folds}-fold outer x {settings.inner_folds}-fold inner, "
            f"{settings.repeats} repeats), n_jobs={settings.n_jobs}"
        )

    results = Parallel(n_jobs=settings.n_jobs, verbose=5 if verbose else 0)(
        delayed(_run_one_repeat)(X, y_arr, m, c, cal, r, settings, feature_override)
        for (m, c, cal, r) in tasks
    )

    preds = [row for res in results for row in res[0]]
    infos = [row for res in results for row in res[1]]
    return pd.DataFrame(preds), pd.DataFrame(infos)
