"""Stage 8: label-robustness sensitivity analysis (N4 / Phase 1c).

Compares the headline cells of the primary run against the two variant
runs produced by ``03_nested_cv.py --label-variant ...``:

* ``exclude_inconsistent`` — the 4 internally inconsistent records
  (notckd label, stage s3-s5; CSV lines 12/18/52/123) removed;
* ``flip_inconsistent``   — their outcome labels flipped to ckd.

All metrics use the published aggregation (per-patient probabilities
averaged across repeats). Output: ``reports/tables/table_27_label_robustness.csv``,
injected into the report by stage 6.

Usage
-----
    python scripts/08_sensitivity_labels.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402

from ckd.config import ensure_dirs, load_config  # noqa: E402
from ckd.evaluation.bootstrap import pooled_predictions  # noqa: E402

VARIANTS = {
    "primary": "cv_predictions",
    "exclude_inconsistent": "cv_predictions_labels_excl",
    "flip_inconsistent": "cv_predictions_labels_flip",
}

CELLS = [
    ("low_cost_model", "svm", "none"),
    ("low_cost_model", "svm", "isotonic"),
    ("low_cost_model", "random_forest", "none"),
    ("full_valid_model", "svm", "none"),
    ("full_valid_model", "svm", "isotonic"),
    ("laboratory_model", "svm", "none"),
    ("laboratory_model", "svm", "isotonic"),
    ("laboratory_model", "random_forest", "none"),
]


def cell_metrics(pooled: pd.DataFrame, config: str, model: str, calibration: str):
    cell = pooled[
        (pooled["config"] == config)
        & (pooled["model"] == model)
        & (pooled["calibration"] == calibration)
    ]
    if cell.empty:
        return None
    y = cell["y_true"].to_numpy()
    prob = cell["y_prob"].to_numpy()
    pred = prob >= 0.5
    return {
        "n": int(len(y)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "sensitivity": float((pred & (y == 1)).sum() / max((y == 1).sum(), 1)),
        "specificity": float((~pred & (y == 0)).sum() / max((y == 0).sum(), 1)),
        "brier": float(brier_score_loss(y, prob)),
    }


def main() -> int:
    cfg = load_config()
    proc_dir, tables_dir = ensure_dirs(cfg["paths"]["processed_dir"],
                                       cfg["paths"]["tables_dir"])

    pooled_by_variant: dict[str, pd.DataFrame] = {}
    manifests: dict[str, dict] = {}
    for variant, stem in VARIANTS.items():
        pred_path = proc_dir / f"{stem}.csv.gz"
        if not pred_path.is_file():
            raise SystemExit(
                f"{pred_path.name} missing - run scripts/03_nested_cv.py "
                f"--label-variant {variant} first" if variant != "primary"
                else f"{pred_path.name} missing - run stage 3 first"
            )
        pooled_by_variant[variant] = pooled_predictions(pd.read_csv(pred_path))
        mpath = proc_dir / f"{stem}_manifest.json"
        manifests[variant] = json.loads(mpath.read_text("utf-8")) if mpath.is_file() else {}

    rows = []
    for config, model, calibration in CELLS:
        base = cell_metrics(pooled_by_variant["primary"], config, model, calibration)
        if base is None:
            continue
        for variant in ("exclude_inconsistent", "flip_inconsistent"):
            metrics = cell_metrics(pooled_by_variant[variant], config, model, calibration)
            if metrics is None:
                continue
            row = {
                "config": config, "model": model, "calibration": calibration,
                "variant": variant,
                "n_primary": base["n"], "n_variant": metrics["n"],
            }
            for m in ("roc_auc", "sensitivity", "specificity", "brier"):
                row[f"primary_{m}"] = base[m]
                row[f"variant_{m}"] = metrics[m]
                row[f"delta_{m}"] = metrics[m] - base[m]
            rows.append(row)

    table = pd.DataFrame(rows)
    out = tables_dir / "table_27_label_robustness.csv"
    table.to_csv(out, index=False)
    print(f"wrote {out} ({len(table)} rows)")
    if len(table):
        worst = table.loc[table["delta_roc_auc"].abs().idxmax()]
        print(f"largest |delta AUC|: {worst['delta_roc_auc']:+.4f} "
              f"({worst['config']} x {worst['model']} x {worst['calibration']}, "
              f"{worst['variant']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
