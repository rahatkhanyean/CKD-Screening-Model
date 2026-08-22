"""Stage 16: is the isotonic-beats-Platt result real? (N9 / Phase 4d)

Section 5.6 reports that isotonic calibration achieved a better median
Brier score than Platt scaling, which is contrary to the usual
small-sample expectation and rests on 25 outer folds. This stage
re-examines it with four times the resampling: 20 repeats of the same
nested design on a restricted grid, giving 100 outer test sets.

It answers a narrow question - whether the observed ordering survives more
resampling - and not the broader one of which calibrator is preferable in
general, which 200 patients cannot settle.

Input:  ``data/processed/cv_predictions_calib20.csv.gz``
        (produced by ``03_nested_cv.py --repeats 20 --out cv_predictions_calib20``)
Output: ``reports/tables/table_39_calibration_experiment.csv``

Usage
-----
    python scripts/16_calibration_experiment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import brier_score_loss  # noqa: E402

from ckd.config import ensure_dirs, load_config  # noqa: E402
from ckd.evaluation.metrics import (  # noqa: E402
    calibration_slope_intercept,
    expected_calibration_error,
)

STEM = "cv_predictions_calib20"


def per_repeat_metrics(preds: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """Calibration statistics computed WITHIN each repeat.

    Averaging a patient's probability across repeats shrinks it toward the
    centre and biases the slope upward, so the statistics are computed per
    repeat and summarised afterwards - the same rule the main analysis
    follows (design decision (e)).
    """
    rows = []
    for (config, model, calibration, repeat), cell in preds.groupby(
        ["config", "model", "calibration", "repeat"]
    ):
        y = cell["y_true"].to_numpy()
        prob = cell["y_prob"].to_numpy()
        slope, intercept = calibration_slope_intercept(y, prob)
        rows.append({
            "config": config, "model": model, "calibration": calibration,
            "repeat": int(repeat),
            "brier": float(brier_score_loss(y, prob)),
            "ece": float(expected_calibration_error(y, prob, n_bins=n_bins)),
            "calibration_slope": slope,
            "calibration_intercept": intercept,
        })
    return pd.DataFrame(rows)


def main() -> int:
    cfg = load_config()
    proc_dir, tables_dir = ensure_dirs(cfg["paths"]["processed_dir"],
                                       cfg["paths"]["tables_dir"])
    path = proc_dir / f"{STEM}.csv.gz"
    if not path.is_file():
        raise SystemExit(
            f"{path.name} missing - run:\n"
            f"  python scripts/03_nested_cv.py --repeats 20 --out {STEM} "
            f"--configs low_cost_model laboratory_model "
            f"--models svm random_forest logreg "
            f"--calibrations none sigmoid isotonic"
        )
    preds = pd.read_csv(path)
    n_bins = int(cfg["evaluation"]["calibration_bins"])
    per_repeat = per_repeat_metrics(preds, n_bins)
    n_repeats = per_repeat["repeat"].nunique()
    n_outer = n_repeats * int(cfg["cross_validation"]["outer_folds"])

    rows = []
    for calibration, group in per_repeat.groupby("calibration"):
        finite_slope = group["calibration_slope"].dropna()
        rows.append({
            "calibration": calibration,
            "n_repeats": int(n_repeats),
            "n_outer_test_sets": int(n_outer),
            "n_cells": int(group.groupby(["config", "model"]).ngroups),
            "median_brier": float(group["brier"].median()),
            "iqr_brier": float(group["brier"].quantile(0.75)
                               - group["brier"].quantile(0.25)),
            "median_ece": float(group["ece"].median()),
            "median_slope": (float(finite_slope.median())
                             if len(finite_slope) else float("nan")),
            "n_slope_unidentified": int(group["calibration_slope"].isna().sum()),
        })
    summary = pd.DataFrame(rows).sort_values("median_brier")

    # Paired comparison: same (config, model, repeat), isotonic vs sigmoid.
    wide = per_repeat.pivot_table(
        index=["config", "model", "repeat"], columns="calibration", values="brier"
    )
    paired = wide.dropna(subset=["isotonic", "sigmoid"])
    diff = paired["isotonic"] - paired["sigmoid"]
    wins = int((diff < 0).sum())
    summary.attrs["paired"] = True
    pair_row = pd.DataFrame([{
        "comparison": "isotonic minus sigmoid (Brier, paired by config x model x repeat)",
        "n_pairs": int(len(diff)),
        "isotonic_better_in": wins,
        "isotonic_better_fraction": round(float(wins / max(len(diff), 1)), 4),
        "median_difference": float(diff.median()),
        "mean_difference": float(diff.mean()),
    }])

    out = tables_dir / "table_39_calibration_experiment.csv"
    summary.to_csv(out, index=False)
    pair_row.to_csv(tables_dir / "table_39_calibration_paired.csv", index=False)
    per_repeat.to_csv(tables_dir / "table_39_calibration_per_repeat.csv", index=False)
    print(f"wrote {out}")
    print(summary.to_string(index=False))
    print()
    print(pair_row.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
