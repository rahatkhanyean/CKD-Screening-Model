"""Stage 9: optimism accounting — apparent minus nested-CV performance (N10).

For each headline cell (configuration x model, uncalibrated), fit the SAME
selection procedure the nested CV uses (4-fold inner grid search on
ROC-AUC), but on the COMPLETE dataset, then evaluate on that same data.
The difference between this "apparent" performance and the pooled nested
out-of-fold performance is the optimism that internal validation is
already correcting for — a number reviewers ask for and readers rarely get.

Cells: {low_cost, full_valid, laboratory} x {SVM, random forest},
uncalibrated. Restricted deliberately: calibrated apparent performance
conflates two optimisms, and the dummy has nothing to overfit with.

Output: ``reports/tables/table_29_optimism.csv``; injected into the report
by stage 6.

Usage
-----
    python scripts/09_optimism.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402

from ckd.config import ensure_dirs, load_config  # noqa: E402
from ckd.data.clean import clean_dataset, feature_matrix  # noqa: E402
from ckd.evaluation.bootstrap import pooled_predictions  # noqa: E402
from ckd.features.configs import get_config  # noqa: E402
from ckd.models.pipeline import assert_pipeline_is_clean, build_pipeline  # noqa: E402
from ckd.models.zoo import get_model  # noqa: E402

CELLS = [
    ("low_cost_model", "svm"),
    ("low_cost_model", "random_forest"),
    ("full_valid_model", "svm"),
    ("full_valid_model", "random_forest"),
    ("laboratory_model", "svm"),
    ("laboratory_model", "random_forest"),
]


def apparent_cell(X, y, config_name, model_name, inner_folds, seed, scoring):
    """Tune on the full data, refit on the full data, score the full data."""
    X_config = X.loc[:, list(get_config(config_name).features)]
    pipe = build_pipeline(model_name, config_name, seed)
    assert_pipeline_is_clean(pipe, config_name)
    spec = get_model(model_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if spec.param_grid:
            search = GridSearchCV(
                pipe,
                param_grid=spec.param_grid,
                scoring=scoring,
                cv=StratifiedKFold(n_splits=inner_folds, shuffle=True,
                                   random_state=seed),
                n_jobs=1,
                refit=True,
            )
            search.fit(X_config, y)
            fitted = search.best_estimator_
        else:
            fitted = pipe.fit(X_config, y)
        prob = fitted.predict_proba(X_config)[:, 1]
    return {
        "apparent_roc_auc": float(roc_auc_score(y, prob)),
        "apparent_brier": float(brier_score_loss(y, prob)),
        "apparent_sensitivity": float(((prob >= 0.5) & (y == 1)).sum() / (y == 1).sum()),
    }


def nested_cell(pooled, config_name, model_name):
    """Out-of-fold metrics for the uncalibrated cell, aggregated exactly as
    the published tables aggregate (per-patient probabilities averaged
    across repeats by ``ckd.evaluation.bootstrap.pooled_predictions``, so
    the nested numbers here are the same numbers the audit tables carry)."""
    cell = pooled[
        (pooled["config"] == config_name)
        & (pooled["model"] == model_name)
        & (pooled["calibration"] == "none")
    ]
    if cell.empty:
        return None
    y = cell["y_true"].to_numpy()
    prob = cell["y_prob"].to_numpy()
    return {
        "nested_roc_auc": float(roc_auc_score(y, prob)),
        "nested_brier": float(brier_score_loss(y, prob)),
        "nested_sensitivity": float(((prob >= 0.5) & (y == 1)).sum() / (y == 1).sum()),
    }


def main() -> int:
    cfg = load_config()
    proc_dir, tables_dir = ensure_dirs(cfg["paths"]["processed_dir"],
                                       cfg["paths"]["tables_dir"])
    result = clean_dataset()
    X = feature_matrix(result)
    y = result.target.to_numpy()
    preds = pd.read_csv(proc_dir / "cv_predictions.csv.gz")
    pooled = pooled_predictions(preds)

    seed = int(cfg["seed"])
    inner = int(cfg["cross_validation"]["inner_folds"])
    scoring = str(cfg["cross_validation"]["inner_scoring"])

    rows = []
    for config_name, model_name in CELLS:
        print(f"[optimism] {config_name} x {model_name}")
        nested = nested_cell(pooled, config_name, model_name)
        if nested is None:
            print("  no nested predictions for this cell; skipped")
            continue
        apparent = apparent_cell(X, y, config_name, model_name, inner, seed, scoring)
        row = {"config": config_name, "model": model_name,
               "calibration": "none", **apparent, **nested}
        for metric in ("roc_auc", "brier", "sensitivity"):
            row[f"optimism_{metric}"] = row[f"apparent_{metric}"] - row[f"nested_{metric}"]
        rows.append(row)
        print(f"  apparent AUC {row['apparent_roc_auc']:.4f} "
              f"vs nested {row['nested_roc_auc']:.4f} "
              f"(optimism {row['optimism_roc_auc']:+.4f})")

    table = pd.DataFrame(rows)
    out = tables_dir / "table_29_optimism.csv"
    table.to_csv(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
