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

#: (configuration, model, restriction). ``restriction`` names a
#: registry-derived feature subset applied on top of the configuration, or
#: None for the configuration as declared. The restricted cells are here
#: because they are the ones whose intervals the manuscript now leans on:
#: the conservative pre-index set is the only configuration that is not
#: saturated, so it is the only one whose width carries information.
CELLS = [
    ("low_cost_model", "svm", None),
    ("low_cost_model", "random_forest", None),
    ("laboratory_model", "svm", None),
    ("full_valid_model", "svm", None),
    ("full_valid_model", "random_forest", "conservative_pre_index"),
    ("full_valid_model", "svm", "no_incorporation_or_consequences"),
]


def restriction_features(name: str) -> list[str]:
    """Feature subsets named in CELLS, derived from the registry."""
    from ckd.features.configs import get_config
    from ckd.features.registry import (
        incorporation_variables,
        load_registry,
        variables_by_category,
    )

    full = list(get_config("full_valid_model").features)
    if name == "conservative_pre_index":
        registry = load_registry()
        uncertain = set(registry.loc[registry["uncertainty_flag"], "variable"])
        pre_index = set(variables_by_category("legitimate_pre_index"))
        return [f for f in full if f in pre_index and f not in uncertain]
    if name == "no_incorporation_or_consequences":
        drop = (set(incorporation_variables())
                | set(variables_by_category("consequence_of_advanced_disease")))
        return [f for f in full if f not in drop]
    raise ValueError(f"unknown restriction {name!r}")


def stratified_resample(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Patient-level resample preserving the class balance."""
    out = []
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        out.append(rng.choice(idx, size=len(idx), replace=True))
    return np.concatenate(out)


def one_resample(
    X: pd.DataFrame, y: np.ndarray, config: str, model: str,
    settings: NestedCVSettings, seed: int, restriction: str | None = None,
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
            feature_override=(restriction_features(restriction)
                              if restriction else None),
            # A bootstrap sample contains the same patient several times.
            # Grouping the outer folds by the ORIGINAL patient index keeps
            # every copy of a patient on the same side of every split;
            # without it a model would be tested on patients it trained on,
            # and each replicate's estimate would be optimistic - which
            # would narrow exactly the interval this stage exists to widen.
            groups=rows,
        )
    except Exception as exc:  # a degenerate resample must not kill the run
        return {"config": config, "model": model,
                "restriction": restriction or "none", "seed": seed,
                "roc_auc": float("nan"), "error": type(exc).__name__}
    pooled = pooled_predictions(preds)
    return {
        "config": config, "model": model,
        "restriction": restriction or "none", "seed": seed,
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
        (config, model, restriction, master + 100_000 + k)
        for config, model, restriction in CELLS
        for k in range(args.resamples)
    ]
    t0 = time.perf_counter()
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(one_resample)(X, y, config, model, settings, seed, restriction)
        for config, model, restriction, seed in jobs
    )
    elapsed = time.perf_counter() - t0
    print(f"completed in {elapsed/60:.1f} min")

    draws = pd.DataFrame([r for r in results if r is not None])
    rows = []
    reference = pooled_predictions(
        pd.read_csv(proc_dir / "cv_predictions.csv.gz")
    )
    checkpoints: list[dict] = []
    for (config, model, restriction), group in draws.groupby(
            ["config", "model", "restriction"]):
        values = group["roc_auc"].dropna().to_numpy()
        cell = reference[
            (reference["config"] == config) & (reference["model"] == model)
            & (reference["calibration"] == "none")
        ]
        point = (float(roc_auc_score(cell["y_true"], cell["y_prob"]))
                 if len(cell) and restriction == "none" else float("nan"))
        # Checkpoint estimates, so that the choice of resample count is
        # justified by evidence rather than asserted. Each checkpoint uses
        # the first n draws, which are a prefix of the same deterministic
        # seed sequence, so the series is nested rather than independent.
        for n in (100, 200, 500, 1000):
            if n > len(values):
                continue
            prefix = values[:n]
            checkpoints.append({
                "config": config, "model": model,
                "restriction": restriction, "n_resamples": n,
                "ci_low": float(np.percentile(prefix, 2.5)),
                "ci_high": float(np.percentile(prefix, 97.5)),
                "ci_width": float(np.percentile(prefix, 97.5)
                                  - np.percentile(prefix, 2.5)),
                "sd": float(prefix.std(ddof=1)),
            })
        rows.append({
            "config": config, "model": model, "restriction": restriction,
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
    check = pd.DataFrame(checkpoints)
    check.to_csv(tables_dir / "table_56_bootstrap_checkpoints.csv", index=False)
    print(f"wrote table_56_bootstrap_checkpoints.csv ({len(check)} rows)")
    draws.to_csv(tables_dir / "table_38_procedure_bootstrap_draws.csv", index=False)
    print(f"wrote {out}")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
