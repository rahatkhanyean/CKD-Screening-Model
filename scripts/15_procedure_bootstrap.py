"""Stage 15: bootstrap the whole procedure, not just the predictions (N6).

The confidence intervals reported elsewhere in this study resample
patients over the pooled out-of-fold predictions. That captures patient
sampling but holds the *fitted models* fixed, so it understates total
uncertainty: it cannot see the variability that comes from the model
selection, tuning and calibration decisions themselves being re-made on a
different sample. Section 5.8 says so explicitly.

This stage does the honest thing. For each resample of the patients, the
**entire nested procedure is re-run from scratch** - fresh outer folds,
fresh inner grid search, fresh calibration - and the metric is recomputed.
The spread across resamples is then an interval for the procedure, not
just for the predictions.

It is expensive by construction, so it is restricted to a few headline
cells and run with a resample count set on the command line.

Output: ``reports/tables/table_38_procedure_bootstrap.csv``.

Usage
-----
    python scripts/15_procedure_bootstrap.py --resamples 200
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

from joblib import Parallel, delayed  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from ckd.config import ensure_dirs, load_config  # noqa: E402
from ckd.data.clean import clean_dataset, feature_matrix  # noqa: E402
from ckd.evaluation.bootstrap import pooled_predictions  # noqa: E402
from ckd.models.nested_cv import NestedCVSettings, run_nested_cv  # noqa: E402

CELLS = [
    ("low_cost_model", "svm"),
    ("low_cost_model", "random_forest"),
    ("laboratory_model", "svm"),
    ("full_valid_model", "svm"),
]


def stratified_resample(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Patient-level resample preserving the class balance."""
    out = []
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        out.append(rng.choice(idx, size=len(idx), replace=True))
    return np.concatenate(out)


def one_resample(
    X: pd.DataFrame, y: np.ndarray, config: str, model: str,
    settings: NestedCVSettings, seed: int,
) -> dict | None:
    """Re-run the entire nested procedure on one resample of the patients."""
    rng = np.random.default_rng(seed)
    rows = stratified_resample(y, rng)
    X_b = X.iloc[rows].reset_index(drop=True)
    y_b = y[rows]
    if len(np.unique(y_b)) < 2:
        return None
    try:
        preds, _ = run_nested_cv(
            X_b, y_b, model_names=[model], config_names=[config],
            calibration_methods=["none"], settings=settings, verbose=0,
            # A bootstrap sample contains the same patient several times.
            # Grouping the outer folds by the ORIGINAL patient index keeps
            # every copy of a patient on the same side of every split;
            # without it a model would be tested on patients it trained on,
            # and each replicate's estimate would be optimistic - which
            # would narrow exactly the interval this stage exists to widen.
            groups=rows,
        )
    except Exception as exc:  # a degenerate resample must not kill the run
        return {"config": config, "model": model, "seed": seed,
                "roc_auc": float("nan"), "error": type(exc).__name__}
    pooled = pooled_predictions(preds)
    return {
        "config": config, "model": model, "seed": seed,
        "roc_auc": float(roc_auc_score(pooled["y_true"], pooled["y_prob"])),
        "error": "",
    }


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resamples", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=1,
                        help="repeats inside each resample (1 keeps it affordable)")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    proc_dir, tables_dir = ensure_dirs(cfg["paths"]["processed_dir"],
                                       cfg["paths"]["tables_dir"])
    result = clean_dataset()
    X = feature_matrix(result)
    y = result.target.to_numpy()

    settings = NestedCVSettings(
        seed=int(cfg["seed"]),
        outer_folds=int(cfg["cross_validation"]["outer_folds"]),
        inner_folds=int(cfg["cross_validation"]["inner_folds"]),
        repeats=int(args.repeats),
        inner_scoring=str(cfg["cross_validation"]["inner_scoring"]),
        target_sensitivity=float(cfg["evaluation"]["target_sensitivity"]),
        n_jobs=1,  # parallelism is across resamples, not inside them
    )

    master = int(cfg["seed"])
    print(f"resamples: {args.resamples} x {len(CELLS)} cells "
          f"(each re-runs the full nested procedure)")

    jobs = [
        (config, model, master + 100_000 + k)
        for config, model in CELLS
        for k in range(args.resamples)
    ]
    t0 = time.perf_counter()
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(one_resample)(X, y, config, model, settings, seed)
        for config, model, seed in jobs
    )
    elapsed = time.perf_counter() - t0
    print(f"completed in {elapsed/60:.1f} min")

    draws = pd.DataFrame([r for r in results if r is not None])
    rows = []
    reference = pooled_predictions(
        pd.read_csv(proc_dir / "cv_predictions.csv.gz")
    )
    for (config, model), group in draws.groupby(["config", "model"]):
        values = group["roc_auc"].dropna().to_numpy()
        cell = reference[
            (reference["config"] == config) & (reference["model"] == model)
            & (reference["calibration"] == "none")
        ]
        point = (float(roc_auc_score(cell["y_true"], cell["y_prob"]))
                 if len(cell) else float("nan"))
        rows.append({
            "config": config, "model": model,
            "point_estimate_roc_auc": point,
            "n_resamples_used": int(len(values)),
            "n_failed": int(group["roc_auc"].isna().sum()),
            "procedure_ci_low": float(np.percentile(values, 2.5)),
            "procedure_ci_high": float(np.percentile(values, 97.5)),
            "procedure_ci_width": float(
                np.percentile(values, 97.5) - np.percentile(values, 2.5)
            ),
            "procedure_sd": float(values.std(ddof=1)),
        })

    table = pd.DataFrame(rows)
    out = tables_dir / "table_38_procedure_bootstrap.csv"
    table.to_csv(out, index=False)
    draws.to_csv(tables_dir / "table_38_procedure_bootstrap_draws.csv", index=False)
    print(f"wrote {out}")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
