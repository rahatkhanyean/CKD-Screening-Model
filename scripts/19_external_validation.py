"""Stage 19: external validation by frozen transfer (Phase 3a).

Models are trained on the analysed benchmark and evaluated, without any
refitting, on an independent cohort. This is the first genuinely external
evidence in the study: the 2015 UCI release was disqualified by the
provenance gate, so MIMIC-IV is the only registered source carrying the same
measurements on different patients.

Design
------
The transfer feature set is the **intersection** of the two schemas: nine
blood analytes plus urine specific gravity. Rather than declaring a new
feature configuration, the set is passed as a ``feature_override`` on the
existing ``full_valid_model`` configuration, so both leakage guards still run
and ``assert_pipeline_is_clean`` verifies every name traces back to a
declared feature.

Both an internal reference and the external result are reported for the same
feature set, so the drop attributable to transfer is separable from the drop
attributable to using fewer variables.

Gated: refuses to run on any dataset the provenance report has not classified
INDEPENDENT.

Output: reports/tables/table_42_external_validation.csv

Usage
-----
    python scripts/19_external_validation.py
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402

from ckd.config import ensure_dirs, load_config, project_root  # noqa: E402
from ckd.data.clean import clean_dataset, feature_matrix  # noqa: E402
from ckd.evaluation.bootstrap import pooled_predictions  # noqa: E402
from ckd.evaluation.metrics import calibration_slope_intercept  # noqa: E402
from ckd.models.nested_cv import NestedCVSettings, run_nested_cv  # noqa: E402
from ckd.models.pipeline import assert_pipeline_is_clean, build_pipeline  # noqa: E402
from ckd.models.zoo import get_model  # noqa: E402

#: Intersection of the analysed schema and the MIMIC extraction.
TRANSFER_FEATURES = [
    "bgr", "bu", "sod", "sc", "pot", "hemo", "pcv", "rbcc", "wbcc", "sg",
]
HOST_CONFIG = "full_valid_model"
MODELS = ["logreg", "svm", "random_forest"]
DATASET_ID = "mimic_iv_demo"


def check_gate(tables_dir: Path, dataset_id: str) -> None:
    path = tables_dir / "table_24_provenance.csv"
    if not path.is_file():
        raise SystemExit("run scripts/00_provenance.py first")
    verdicts = pd.read_csv(path).set_index("dataset_id")["verdict"].to_dict()
    verdict = verdicts.get(dataset_id)
    if verdict != "INDEPENDENT":
        raise SystemExit(
            f"{dataset_id} is classified {verdict!r}; only INDEPENDENT datasets "
            "may support external claims."
        )
    print(f"gate ok: {dataset_id} is {verdict}")


def auc_ci(y, prob, n_boot: int = 2000, seed: int = 0,
           score=roc_auc_score) -> tuple[float, float]:
    """Stratified patient-level bootstrap interval for a ranking metric.

    Reported because the external cohort is small: an estimate on 20 events
    without an interval invites over-reading. ``score`` selects the metric,
    so the ROC-AUC and PR-AUC intervals come from the same resamples.
    """
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True),
        ])
        if len(np.unique(y[idx])) < 2:
            continue
        draws.append(score(y[idx], prob[idx]))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def recalibrate_intercept(prob: np.ndarray, target_prevalence: float) -> np.ndarray:
    """Shift the log-odds so mean predicted risk matches a stated prevalence.

    This is recalibration-in-the-large, the minimal correction a deployer
    would apply when moving a model to a population with a different base
    rate. Applying it separates the part of the calibration failure that is
    merely a prevalence mismatch from the part that is not. It cannot change
    the ranking, so discrimination is untouched by construction.
    """
    eps = 1e-9
    clipped = np.clip(prob, eps, 1 - eps)
    logit = np.log(clipped / (1 - clipped))
    lo, hi = -30.0, 30.0
    for _ in range(200):  # bisection on the offset
        mid = (lo + hi) / 2
        mean_risk = float(np.mean(1.0 / (1.0 + np.exp(-(logit + mid)))))
        if mean_risk < target_prevalence:
            lo = mid
        else:
            hi = mid
    offset = (lo + hi) / 2
    return 1.0 / (1.0 + np.exp(-(logit + offset)))


def metrics(y, prob, label: str, with_ci: bool = False, seed: int = 0) -> dict:
    pred = prob >= 0.5
    slope, intercept = calibration_slope_intercept(y, prob)
    pos, neg = (y == 1), (y == 0)
    row = {
        "arm": label,
        "n": int(len(y)),
        "n_positive": int(pos.sum()),
        "prevalence": round(float(pos.mean()), 4),
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
        "pr_auc_baseline": round(float(pos.mean()), 4),
        "brier": float(brier_score_loss(y, prob)),
        "sensitivity": float((pred & pos).sum() / max(pos.sum(), 1)),
        "specificity": float((~pred & neg).sum() / max(neg.sum(), 1)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }
    if with_ci:
        low, high = auc_ci(y, prob, seed=seed)
        row["roc_auc_ci_low"], row["roc_auc_ci_high"] = low, high
        plow, phigh = auc_ci(y, prob, seed=seed,
                             score=average_precision_score)
        row["pr_auc_ci_low"], row["pr_auc_ci_high"] = plow, phigh
    return row


def fit_frozen(X, y, model_name: str, seed: int, inner: int, scoring: str):
    """Tune and fit on the complete internal data, exactly as the nested
    procedure tunes within a training fold."""
    pipe = build_pipeline(model_name, HOST_CONFIG, seed,
                          feature_override=TRANSFER_FEATURES)
    assert_pipeline_is_clean(pipe, HOST_CONFIG)
    spec = get_model(model_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if spec.param_grid:
            search = GridSearchCV(
                pipe, param_grid=spec.param_grid, scoring=scoring,
                cv=StratifiedKFold(n_splits=inner, shuffle=True, random_state=seed),
                n_jobs=1, refit=True,
            )
            search.fit(X, y)
            best = search.best_estimator_
        else:
            best = pipe.fit(X, y)
        calibrated = CalibratedClassifierCV(
            clone(best), method="isotonic",
            cv=StratifiedKFold(n_splits=inner, shuffle=True, random_state=seed + 7),
            ensemble=True,
        )
        calibrated.fit(X, y)
    return best, calibrated


def main() -> int:
    cfg = load_config()
    (tables_dir,) = ensure_dirs(cfg["paths"]["tables_dir"])
    check_gate(tables_dir, DATASET_ID)

    internal = clean_dataset()
    X_int = feature_matrix(internal)[TRANSFER_FEATURES]
    y_int = internal.target.to_numpy()

    external_path = (project_root() / "data" / "external" / DATASET_ID
                     / "processed" / "mimic_cohort.csv")
    if not external_path.is_file():
        raise SystemExit("run scripts/18_mimic_etl.py first")
    ext = pd.read_csv(external_path)
    # A subject with no eligible measurement carries a label and nothing else;
    # it cannot be predicted and is excluded here rather than imputed.
    usable = ext[TRANSFER_FEATURES].notna().any(axis=1)
    dropped = int((~usable).sum())
    ext = ext[usable].reset_index(drop=True)
    X_ext = ext[TRANSFER_FEATURES]
    y_ext = ext["ckd_label"].to_numpy()
    print(f"external cohort: {len(y_ext)} subjects ({int(y_ext.sum())} CKD); "
          f"{dropped} excluded for having no eligible measurement")

    seed = int(cfg["seed"])
    inner = int(cfg["cross_validation"]["inner_folds"])
    scoring = str(cfg["cross_validation"]["inner_scoring"])

    # ---- internal reference on the same feature set -----------------------
    settings = NestedCVSettings(
        seed=seed, outer_folds=int(cfg["cross_validation"]["outer_folds"]),
        inner_folds=inner, repeats=int(cfg["cross_validation"]["repeats"]),
        inner_scoring=scoring,
        target_sensitivity=float(cfg["evaluation"]["target_sensitivity"]),
        n_jobs=-1,
    )
    print("\ninternal reference (nested CV on the transfer feature set) ...")
    preds, _ = run_nested_cv(
        X_int, y_int, model_names=MODELS, config_names=[HOST_CONFIG],
        calibration_methods=["isotonic"], settings=settings, verbose=0,
        feature_override=TRANSFER_FEATURES,
    )
    pooled = pooled_predictions(preds)

    rows = []
    for model_name in MODELS:
        cell = pooled[pooled["model"] == model_name]
        row = metrics(cell["y_true"].to_numpy(), cell["y_prob"].to_numpy(),
                      "internal_nested_cv")
        row["model"] = model_name
        rows.append(row)

        _best, calibrated = fit_frozen(X_int, y_int, model_name, seed, inner, scoring)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob_ext = calibrated.predict_proba(X_ext)[:, 1]
        row_ext = metrics(y_ext, prob_ext, "external_frozen_transfer",
                          with_ci=True, seed=seed)
        row_ext["model"] = model_name
        rows.append(row_ext)

        # Minimal deployer's correction: match the external base rate.
        prob_recal = recalibrate_intercept(prob_ext, float(y_ext.mean()))
        row_recal = metrics(y_ext, prob_recal,
                            "external_recalibrated_in_the_large")
        row_recal["model"] = model_name
        rows.append(row_recal)

        print(f"  {model_name:14s} internal {row['roc_auc']:.3f} -> external "
              f"{row_ext['roc_auc']:.3f} "
              f"({row_ext['roc_auc_ci_low']:.3f}-{row_ext['roc_auc_ci_high']:.3f}) "
              f"| Brier {row_ext['brier']:.3f} -> {row_recal['brier']:.3f} "
              f"after recalibration-in-the-large")

    table = pd.DataFrame(rows)
    ordered = ["model", "arm", "n", "n_positive", "prevalence", "roc_auc",
               "roc_auc_ci_low", "roc_auc_ci_high",
               # PR-AUC matters here precisely because prevalence differs
               # between the cohorts (0.64 internally, 0.21 externally); the
               # baseline column is the prevalence a no-skill model achieves.
               "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high", "pr_auc_baseline",
               "brier", "sensitivity",
               "specificity", "calibration_slope", "calibration_intercept"]
    table = table[[c for c in ordered if c in table.columns]]
    table["dataset"] = DATASET_ID
    table["n_features"] = len(TRANSFER_FEATURES)
    out = tables_dir / "table_42_external_validation.csv"
    table.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
