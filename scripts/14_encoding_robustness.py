"""Stage 14: does the interval encoding drive any result? (N5 / Phase 4a)

Section 5.16 showed that replacing the bins with the underlying
measurements changes nothing. This stage asks the complementary question:
does the *choice of number* used to represent each bin matter? Four
encodings are compared on the identical nested design, seeds and folds:

``midpoint``     the reference encoding (bin representative values)
``bin_index``    ordinal position only, discarding bin spacing
``rank_normal``  ordinal position on a Gaussian-like spacing
``onehot``       one indicator per bin, discarding order entirely

All four are pure functions of one cell's label and the published bin
list, so all are leakage-safe by the same argument as the reference
encoding (see :mod:`ckd.data.encodings`).

Output: ``reports/tables/table_37_encoding_robustness.csv``.

Usage
-----
    python scripts/14_encoding_robustness.py [--repeats N]
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
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402

from ckd.config import ensure_dirs, load_config  # noqa: E402
from ckd.data.clean import clean_dataset  # noqa: E402
from ckd.data.encodings import ENCODINGS, encode, encoded_feature_names  # noqa: E402
from ckd.evaluation.bootstrap import pooled_predictions  # noqa: E402
from ckd.evaluation.metrics import calibration_slope_intercept  # noqa: E402
from ckd.features.configs import FORBIDDEN_IN_VALID, get_config  # noqa: E402
from ckd.models.nested_cv import NestedCVSettings, run_nested_cv  # noqa: E402
from ckd.models.pipeline import assert_pipeline_is_clean, build_pipeline  # noqa: E402

CONFIGS = ["low_cost_model", "laboratory_model"]
MODELS = ["svm", "random_forest", "logreg"]
CALIBRATIONS = ["none", "isotonic"]


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int,
                        default=int(cfg["cross_validation"]["repeats"]))
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    proc_dir, tables_dir = ensure_dirs(cfg["paths"]["processed_dir"],
                                       cfg["paths"]["tables_dir"])
    result = clean_dataset()
    y = result.target.to_numpy()

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
    for method in ENCODINGS:
        print(f"\n=== encoding: {method} ===")
        t0 = time.perf_counter()
        for config_name in CONFIGS:
            features = list(get_config(config_name).features)
            illegal = set(features) & FORBIDDEN_IN_VALID
            if illegal:
                raise SystemExit(f"{config_name} declares prohibited {illegal}")
            matrix = encode(result.clean, result.bin_map, features, method)
            names = encoded_feature_names(result.bin_map, features, method)
            assert list(matrix.columns) == names, "encoding column order mismatch"

            # Pre-flight: the guards still run, now against encoded names.
            for model_name in MODELS:
                pipe = build_pipeline(model_name, config_name, settings.seed,
                                      feature_override=names)
                assert_pipeline_is_clean(pipe, config_name)

            preds, _folds = run_nested_cv(
                matrix, y,
                model_names=MODELS, config_names=[config_name],
                calibration_methods=CALIBRATIONS, settings=settings,
                feature_override=names,
                verbose=0,
            )
            for (config, model, calibration), cell in pooled_predictions(
                preds
            ).groupby(["config", "model", "calibration"]):
                yy = cell["y_true"].to_numpy()
                prob = cell["y_prob"].to_numpy()
                slope, intercept = calibration_slope_intercept(yy, prob)
                pred = prob >= 0.5
                rows.append({
                    "encoding": method, "config": config, "model": model,
                    "calibration": calibration,
                    "n_encoded_columns": len(names),
                    "roc_auc": float(roc_auc_score(yy, prob)),
                    "brier": float(brier_score_loss(yy, prob)),
                    "sensitivity": float((pred & (yy == 1)).sum() / (yy == 1).sum()),
                    "calibration_slope": slope,
                })
        print(f"  {time.perf_counter() - t0:.0f}s")

    table = pd.DataFrame(rows)
    reference = table[table["encoding"] == "midpoint"].set_index(
        ["config", "model", "calibration"]
    )["roc_auc"]
    table["delta_vs_midpoint"] = [
        r["roc_auc"] - reference.loc[(r["config"], r["model"], r["calibration"])]
        for _, r in table.iterrows()
    ]
    out = tables_dir / "table_37_encoding_robustness.csv"
    table.to_csv(out, index=False)
    print(f"\nwrote {out}")
    summary = (
        table.groupby("encoding")["delta_vs_midpoint"]
        .agg(["min", "max", lambda s: s.abs().max()])
        .rename(columns={"<lambda_0>": "max_abs"})
    )
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
