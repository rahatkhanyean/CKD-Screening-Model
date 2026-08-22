"""Stage 3: repeated nested stratified cross-validation over the full grid.

Grid = feature configurations x models x calibration methods x repeats.

Output is a single long table of per-patient out-of-fold predictions
(``data/processed/cv_predictions.csv.gz``). Every downstream number in the
study is derived from that table, so results cannot drift apart from the
predictions that produced them.

Usage
-----
    python scripts/03_nested_cv.py                 # full grid from config
    python scripts/03_nested_cv.py --repeats 1     # quick smoke run
    python scripts/03_nested_cv.py --n-jobs 4
"""

from __future__ import annotations

import os

# Pin BLAS/OpenMP to a single thread BEFORE numpy or scikit-learn are imported.
# Without this, 12 joblib worker processes each spawn 12 BLAS threads on a
# 12-core machine, and the resulting oversubscription dominates the runtime.
for _var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ckd.config import ensure_dirs, load_config  # noqa: E402
from ckd.data.clean import clean_dataset, feature_matrix  # noqa: E402
from ckd.features.configs import FEATURE_CONFIGS  # noqa: E402
from ckd.models.nested_cv import NestedCVSettings, run_nested_cv  # noqa: E402
from ckd.models.pipeline import assert_pipeline_is_clean, build_pipeline  # noqa: E402
from ckd.models.zoo import available_models, unavailable_models  # noqa: E402


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repeats", type=int, default=cfg["cross_validation"]["repeats"])
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--models", nargs="*", default=cfg["models"])
    p.add_argument("--configs", nargs="*", default=cfg["feature_configs"])
    p.add_argument("--calibrations", nargs="*", default=cfg["calibration_methods"])
    p.add_argument("--out", type=str, default="cv_predictions")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()
    proc_dir, tab_dir = ensure_dirs(cfg["paths"]["processed_dir"], cfg["paths"]["tables_dir"])

    result = clean_dataset()
    X = feature_matrix(result)
    y = result.target.to_numpy()

    models = available_models(list(args.models))
    missing = unavailable_models(list(args.models))
    if missing:
        print("WARNING: unavailable models will be skipped and reported:")
        for name, reason in missing.items():
            print(f"  - {name}: {reason}")

    settings = NestedCVSettings(
        seed=int(cfg["seed"]),
        outer_folds=int(cfg["cross_validation"]["outer_folds"]),
        inner_folds=int(cfg["cross_validation"]["inner_folds"]),
        repeats=int(args.repeats),
        inner_scoring=str(cfg["cross_validation"]["inner_scoring"]),
        target_sensitivity=float(cfg["evaluation"]["target_sensitivity"]),
        n_jobs=int(args.n_jobs),
    )

    print("=" * 78)
    print("Repeated nested stratified cross-validation")
    print("=" * 78)
    print(f"  patients             : {len(y)} ({int(y.sum())} CKD / {int((1-y).sum())} non-CKD)")
    print(f"  outer folds          : {settings.outer_folds}")
    print(f"  inner folds          : {settings.inner_folds}")
    print(f"  repeats              : {settings.repeats}")
    print(f"  outer test sets      : {settings.outer_folds * settings.repeats}")
    print(f"  inner selection score: {settings.inner_scoring}")
    print(f"  master seed          : {settings.seed}")
    print(f"  configurations       : {list(args.configs)}")
    print(f"  models               : {models}")
    print(f"  calibration methods  : {list(args.calibrations)}")

    # Pre-flight leakage check on every pipeline the run will build.
    print("\nPre-flight leakage check ...")
    n_checked = 0
    for config_name in args.configs:
        for model_name in models:
            pipe = build_pipeline(model_name, config_name, seed=settings.seed)
            assert_pipeline_is_clean(pipe, config_name)
            n_checked += 1
    print(f"  {n_checked} pipelines verified clean")

    t0 = time.perf_counter()
    preds, folds = run_nested_cv(
        X, y,
        model_names=models,
        config_names=list(args.configs),
        calibration_methods=list(args.calibrations),
        settings=settings,
    )
    elapsed = time.perf_counter() - t0

    pred_path = proc_dir / f"{args.out}.csv.gz"
    preds.to_csv(pred_path, index=False, compression="gzip")
    fold_path = proc_dir / f"{args.out}_folds.csv"
    folds.to_csv(fold_path, index=False, encoding="utf-8")

    print(f"\nCompleted in {elapsed/60:.1f} min")
    print(f"  predictions : {pred_path.name}  {preds.shape}")
    print(f"  fold detail : {fold_path.name}  {folds.shape}")

    # ---- integrity checks on the produced predictions ----
    print("\nIntegrity checks")
    expected_per_repeat = len(y)
    counts = preds.groupby(["config", "model", "calibration", "repeat"]).size()
    ok_counts = bool((counts == expected_per_repeat).all())
    print(f"  every patient predicted exactly once per repeat : {ok_counts}")

    dupes = preds.duplicated(
        subset=["config", "model", "calibration", "repeat", "sample_index"]
    ).sum()
    print(f"  duplicate (cell, repeat, patient) rows          : {dupes}")

    bad_prob = int(((preds["y_prob"] < 0) | (preds["y_prob"] > 1)).sum())
    print(f"  probabilities outside [0, 1]                    : {bad_prob}")

    truth = preds.groupby("sample_index")["y_true"].nunique()
    print(f"  patients with inconsistent labels               : {int((truth > 1).sum())}")

    if not (ok_counts and dupes == 0 and bad_prob == 0 and (truth <= 1).all()):
        print("\nFAILED integrity checks - not writing the run manifest.")
        return 1

    manifest = {
        "seed": settings.seed,
        "outer_folds": settings.outer_folds,
        "inner_folds": settings.inner_folds,
        "repeats": settings.repeats,
        "inner_scoring": settings.inner_scoring,
        "target_sensitivity": settings.target_sensitivity,
        "primary_threshold": cfg["evaluation"]["primary_threshold"],
        "models_run": models,
        "models_unavailable": missing,
        "configs_run": list(args.configs),
        "calibrations_run": list(args.calibrations),
        "n_patients": int(len(y)),
        "n_positive": int(y.sum()),
        "runtime_minutes": round(elapsed / 60, 2),
        "n_prediction_rows": int(len(preds)),
        "n_outer_folds_total": int(len(folds)),
        "config_feature_counts": {
            k: len(v.features) for k, v in FEATURE_CONFIGS.items() if k in args.configs
        },
    }
    with open(proc_dir / f"{args.out}_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    # Which hyper-parameters were selected, and how often?
    param_cols = [c for c in folds.columns if c.startswith("param_")]
    if param_cols:
        rows = []
        for (config, model), grp in folds.groupby(["config", "model"]):
            for col in param_cols:
                vals = grp[col].dropna()
                if vals.empty:
                    continue
                vc = vals.value_counts(normalize=True)
                rows.append({
                    "config": config,
                    "model": model,
                    "hyperparameter": col.replace("param_clf__", ""),
                    "modal_value": str(vc.index[0]),
                    "modal_selection_frequency": round(float(vc.iloc[0]), 3),
                    "n_distinct_selected": int(vals.nunique()),
                    "n_folds": int(len(grp)),
                })
        pd.DataFrame(rows).to_csv(
            tab_dir / "table_07_hyperparameter_selection.csv", index=False, encoding="utf-8"
        )
        print(f"  wrote table_07_hyperparameter_selection.csv")

    print("\nAll integrity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
