"""Stage 12: binned versus continuous at the model level (Phase 3d).

Stage 11 measured the cost of the published interval encoding one variable
at a time. This stage asks the question that matters for the benchmark:
**would a study have reached different conclusions had the continuous
values been released?**

Design. The same uniquely pinned patients are modelled twice under the
identical nested design, seeds and pipelines. The only difference is the
representation of the variables that are recoverable as continuous
quantities; every other column keeps its released coding. Two arms:

``binned``      the file as released - interval midpoints, and the
                constant values the release supplied where the source
                observed nothing.
``continuous``  the source's measurements, with unobserved cells left
                missing and imputed inside training folds like any other
                missing value.

The contrast is therefore *the released file versus the source data for
the same patients*, which is the practically meaningful comparison: it
combines the encoding with the release's handling of missingness, and both
are properties of the artefact under study. Table 31 separates the two
components univariately for readers who want them apart.

Gated on the same SAME-SOURCE verdict as stage 11.

Usage
-----
    python scripts/12_binned_vs_continuous.py [--repeats N]
"""

from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402

from ckd.config import ensure_dirs, load_config  # noqa: E402
from ckd.data.clean import clean_dataset, feature_matrix  # noqa: E402
from ckd.data.external import load_uci2015  # noqa: E402
from ckd.data.provenance import (  # noqa: E402
    CONTAINMENT_VARIABLES,
    compatibility_matrix,
    unique_pins,
)
from ckd.data.recovery import RECOVERABLE, recover  # noqa: E402
from ckd.evaluation.bootstrap import pooled_predictions  # noqa: E402
from ckd.evaluation.metrics import calibration_slope_intercept  # noqa: E402
from ckd.models.nested_cv import NestedCVSettings, run_nested_cv  # noqa: E402

CONFIGS = ["low_cost_model", "laboratory_model", "full_valid_model"]
MODELS = ["svm", "random_forest", "logreg"]
CALIBRATIONS = ["none", "isotonic"]
SOURCE_ID = "uci2015"


def check_gate(tables_dir: Path) -> None:
    path = tables_dir / "table_24_provenance.csv"
    if not path.is_file():
        raise SystemExit("run scripts/00_provenance.py first")
    verdicts = pd.read_csv(path).set_index("dataset_id")["verdict"].to_dict()
    if verdicts.get(SOURCE_ID) != "SAME-SOURCE":
        raise SystemExit(
            f"binned-vs-continuous requires {SOURCE_ID} SAME-SOURCE "
            f"(found {verdicts.get(SOURCE_ID)!r})"
        )


def metrics_for(pooled: pd.DataFrame) -> list[dict]:
    rows = []
    for (config, model, calibration), cell in pooled.groupby(
        ["config", "model", "calibration"]
    ):
        y = cell["y_true"].to_numpy()
        prob = cell["y_prob"].to_numpy()
        pred = prob >= 0.5
        slope, intercept = calibration_slope_intercept(y, prob)
        rows.append({
            "config": config, "model": model, "calibration": calibration,
            "roc_auc": float(roc_auc_score(y, prob)),
            "brier": float(brier_score_loss(y, prob)),
            "sensitivity": float((pred & (y == 1)).sum() / max((y == 1).sum(), 1)),
            "specificity": float((~pred & (y == 0)).sum() / max((y == 0).sum(), 1)),
            "calibration_slope": slope,
            "calibration_intercept": intercept,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    cfg = load_config()
    parser.add_argument("--repeats", type=int,
                        default=int(cfg["cross_validation"]["repeats"]))
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    proc_dir, tables_dir = ensure_dirs(cfg["paths"]["processed_dir"],
                                       cfg["paths"]["tables_dir"])
    check_gate(tables_dir)

    internal = clean_dataset()
    X = feature_matrix(internal)
    y = internal.target.to_numpy()
    source = load_uci2015()
    mappings = {
        k: {b["label"]: (float(b["lower"]), float(b["upper"])) for b in v}
        for k, v in internal.bin_map.items()
    }
    compat = compatibility_matrix(
        internal.clean, y, source.frame, source.target.to_numpy(),
        mappings, CONTAINMENT_VARIABLES,
    )
    pins = unique_pins(compat)
    result = recover(internal.clean, y, source.frame, pins)

    # Both arms: identical patients, identical columns, identical order.
    X_binned = X.iloc[result.internal_positions].reset_index(drop=True)
    y_sub = result.target
    X_continuous = X_binned.copy()
    substituted = []
    for variable in RECOVERABLE:
        if variable in X_continuous.columns:
            X_continuous[variable] = result.frame[variable].to_numpy()
            substituted.append(variable)

    print(f"patients: {len(y_sub)} ({int(y_sub.sum())} CKD)")
    print(f"variables substituted with source measurements: {substituted}")
    n_missing = int(X_continuous[substituted].isna().sum().sum())
    print(f"cells left missing in the continuous arm (imputed within folds): "
          f"{n_missing}")

    settings = NestedCVSettings(
        seed=int(cfg["seed"]),
        outer_folds=int(cfg["cross_validation"]["outer_folds"]),
        inner_folds=int(cfg["cross_validation"]["inner_folds"]),
        repeats=int(args.repeats),
        inner_scoring=str(cfg["cross_validation"]["inner_scoring"]),
        target_sensitivity=float(cfg["evaluation"]["target_sensitivity"]),
        n_jobs=int(args.n_jobs),
    )

    rows = []
    for arm, matrix in (("binned", X_binned), ("continuous", X_continuous)):
        print(f"\n=== arm: {arm} ===")
        preds, _folds = run_nested_cv(
            matrix, y_sub,
            model_names=MODELS, config_names=CONFIGS,
            calibration_methods=CALIBRATIONS, settings=settings,
        )
        preds.to_csv(proc_dir / f"cv_predictions_{arm}.csv.gz",
                     index=False, compression="gzip")
        for row in metrics_for(pooled_predictions(preds)):
            row["arm"] = arm
            rows.append(row)

    table = pd.DataFrame(rows)
    wide = table.pivot_table(
        index=["config", "model", "calibration"], columns="arm",
        values=["roc_auc", "brier", "sensitivity", "calibration_slope"],
    ).reset_index()
    wide.columns = [
        c[0] if not c[1] else f"{c[0]}_{c[1]}" for c in wide.columns.to_flat_index()
    ]
    wide["delta_roc_auc"] = wide["roc_auc_continuous"] - wide["roc_auc_binned"]
    wide["delta_brier"] = wide["brier_continuous"] - wide["brier_binned"]
    wide = wide.sort_values("delta_roc_auc", ascending=False)
    wide.to_csv(tables_dir / "table_35_binned_vs_continuous.csv", index=False)
    print(f"\nwrote {tables_dir / 'table_35_binned_vs_continuous.csv'}")
    print(wide[["config", "model", "calibration", "roc_auc_binned",
                "roc_auc_continuous", "delta_roc_auc"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
