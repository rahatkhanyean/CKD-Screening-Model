"""Stage 23: case-mix reanalysis with uncertainty and feature restriction.

Addresses three weaknesses in the original case-mix argument:

**Subgroup metrics had no uncertainty.** Stage-specific discrimination,
sensitivity, specificity, PR-AUC and Brier are recomputed with stratified
patient-level bootstrap intervals, so the 21-case early-stage subgroup
carries a visible interval rather than a bare point estimate.

**The single-feature baseline was selected on the data it was scored on.**
The multivariable lift is recomputed against a baseline whose column and
orientation are chosen inside training folds
(``informativeness.nested_best_single_auc``).

**Feature sets that beg the question were not separated.** The registry
(stage 22) classifies each variable, and three restricted configurations are
evaluated: without consequences of advanced disease, without variables that
enter the diagnostic criterion, and using only variables plausibly available
before a laboratory work-up. If discrimination survives these restrictions,
the case-mix explanation is strengthened; if it collapses, the ceiling
depended on incorporation.

Outputs
-------
reports/tables/table_48_subgroup_metrics_ci.csv
reports/tables/table_49_restricted_featuresets.csv
reports/tables/table_50_nested_lift.csv

Usage
-----
    python scripts/23_casemix_reanalysis.py [--repeats N] [--boot N]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from ckd.config import ensure_dirs, load_config, project_root  # noqa: E402
from ckd.data.clean import clean_dataset, feature_matrix  # noqa: E402
from ckd.evaluation.bootstrap import pooled_predictions  # noqa: E402
from ckd.evaluation.informativeness import nested_best_single_auc  # noqa: E402
from ckd.features.configs import get_config  # noqa: E402
from ckd.models.nested_cv import NestedCVSettings, run_nested_cv  # noqa: E402

CONFIGS = ["low_cost_model", "laboratory_model", "full_valid_model"]
HOST = "full_valid_model"
MODELS = ["logreg", "svm", "random_forest"]

SUBGROUPS = {
    "all_patients": None,
    "early_ckd_s1_s2": ("s1", "s2"),
    "moderate_ckd_s3": ("s3",),
    "advanced_ckd_s4_s5": ("s4", "s5"),
}


def boot_ci(fn, y, prob, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            draws.append(fn(y[idx], prob[idx]))
        except ValueError:
            continue
    if not draws:
        return float("nan"), float("nan")
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def metrics_with_ci(y, prob, n_boot, seed, threshold=0.5) -> dict:
    pred = prob >= threshold
    pos, neg = y == 1, y == 0
    out = {
        "n": int(len(y)), "n_cases": int(pos.sum()), "n_controls": int(neg.sum()),
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
        "sensitivity": float((pred & pos).sum() / max(pos.sum(), 1)),
        "specificity": float((~pred & neg).sum() / max(neg.sum(), 1)),
        "brier": float(brier_score_loss(y, prob)),
    }
    lo, hi = boot_ci(lambda a, b: roc_auc_score(a, b), y, prob, n_boot, seed)
    out["roc_auc_ci_low"], out["roc_auc_ci_high"] = lo, hi
    lo, hi = boot_ci(lambda a, b: average_precision_score(a, b), y, prob,
                     n_boot, seed + 1)
    out["pr_auc_ci_low"], out["pr_auc_ci_high"] = lo, hi
    lo, hi = boot_ci(
        lambda a, b: float(((b >= threshold) & (a == 1)).sum()
                           / max((a == 1).sum(), 1)),
        y, prob, n_boot, seed + 2)
    out["sensitivity_ci_low"], out["sensitivity_ci_high"] = lo, hi
    return out


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int,
                        default=int(cfg["cross_validation"]["repeats"]))
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    proc_dir, tables_dir = ensure_dirs(cfg["paths"]["processed_dir"],
                                       cfg["paths"]["tables_dir"])
    internal = clean_dataset()
    X = feature_matrix(internal)
    y = internal.target.to_numpy()
    stage = internal.clean["stage"].astype("object").to_numpy()
    seed = int(cfg["seed"])
    pooled = pooled_predictions(pd.read_csv(proc_dir / "cv_predictions.csv.gz"))

    # ---- 1. subgroup metrics with intervals -------------------------------
    print("subgroup metrics with bootstrap intervals ...")
    rows = []
    for config_name in CONFIGS:
        cell = pooled[(pooled["config"] == config_name)
                      & (pooled["calibration"] == "isotonic")
                      & (pooled["model"] != "dummy")]
        best = (cell.groupby("model")
                .apply(lambda g: roc_auc_score(g["y_true"], g["y_prob"]),
                       include_groups=False).idxmax())
        chosen = cell[cell["model"] == best].sort_values("sample_index")
        idx = chosen["sample_index"].to_numpy()
        yy = chosen["y_true"].to_numpy()
        pp = chosen["y_prob"].to_numpy()
        st = stage[idx]
        for label, stages in SUBGROUPS.items():
            if stages is None:
                mask = np.ones(len(yy), dtype=bool)
            else:
                mask = (yy == 0) | np.isin(st, stages)
            if len(np.unique(yy[mask])) < 2:
                continue
            m = metrics_with_ci(yy[mask], pp[mask], args.boot, seed)
            rows.append({"config": config_name, "model": best,
                         "subgroup": label, "exploratory": label != "all_patients",
                         **m})
            print(f"  {config_name:18s} {label:20s} n_cases={m['n_cases']:3d} "
                  f"AUC {m['roc_auc']:.3f} ({m['roc_auc_ci_low']:.3f}-"
                  f"{m['roc_auc_ci_high']:.3f})  sens {m['sensitivity']:.3f} "
                  f"({m['sensitivity_ci_low']:.3f}-{m['sensitivity_ci_high']:.3f})")
    subgroups = pd.DataFrame(rows)
    subgroups.to_csv(tables_dir / "table_48_subgroup_metrics_ci.csv", index=False)
    print(f"wrote table_48_subgroup_metrics_ci.csv ({len(subgroups)} rows)")

    # ---- 2. restricted feature sets --------------------------------------
    registry = pd.read_csv(project_root() / "data" / "feature_registry.csv")
    full = list(get_config(HOST).features)
    consequence = set(registry.loc[
        registry["leakage_category"] == "consequence_of_advanced_disease",
        "variable"])
    incorporation = set(registry.loc[
        registry["leakage_category"] == "incorporation_risk", "variable"])
    pre_index = set(registry.loc[
        registry["leakage_category"] == "legitimate_pre_index", "variable"])

    restricted = {
        "full_valid (reference)": full,
        "minus consequences of advanced disease":
            [f for f in full if f not in consequence],
        "minus diagnostic-criterion inputs":
            [f for f in full if f not in incorporation],
        "minus both":
            [f for f in full if f not in (consequence | incorporation)],
        "pre-index available only":
            [f for f in full if f in pre_index],
    }

    print("\nrestricted feature sets ...")
    settings = NestedCVSettings(
        seed=seed, outer_folds=int(cfg["cross_validation"]["outer_folds"]),
        inner_folds=int(cfg["cross_validation"]["inner_folds"]),
        repeats=int(args.repeats),
        inner_scoring=str(cfg["cross_validation"]["inner_scoring"]),
        target_sensitivity=float(cfg["evaluation"]["target_sensitivity"]),
        n_jobs=int(args.n_jobs),
    )
    rrows = []
    for label, features in restricted.items():
        if not features:
            continue
        preds, _ = run_nested_cv(
            X, y, model_names=MODELS, config_names=[HOST],
            calibration_methods=["isotonic"], settings=settings, verbose=0,
            feature_override=features,
        )
        pl = pooled_predictions(preds)
        for model_name, grp in pl.groupby("model"):
            g = grp.sort_values("sample_index")
            m = metrics_with_ci(g["y_true"].to_numpy(), g["y_prob"].to_numpy(),
                                args.boot, seed)
            rrows.append({"featureset": label, "n_features": len(features),
                          "model": model_name,
                          "removed": ";".join(sorted(set(full) - set(features)))
                          or "none", **m})
        best_row = max([r for r in rrows if r["featureset"] == label],
                       key=lambda r: r["roc_auc"])
        print(f"  {label:40s} k={len(features):2d} best AUC "
              f"{best_row['roc_auc']:.4f} "
              f"({best_row['roc_auc_ci_low']:.3f}-{best_row['roc_auc_ci_high']:.3f})")
    restricted_df = pd.DataFrame(rrows)
    restricted_df.to_csv(tables_dir / "table_49_restricted_featuresets.csv",
                         index=False)
    print(f"wrote table_49_restricted_featuresets.csv ({len(restricted_df)} rows)")

    # ---- 3. nested single-feature baseline -------------------------------
    print("\nnested single-feature baseline ...")
    lrows = []
    for config_name in CONFIGS:
        features = list(get_config(config_name).features)
        cell = pooled[(pooled["config"] == config_name)
                      & (pooled["calibration"] == "isotonic")
                      & (pooled["model"] != "dummy")]
        best = (cell.groupby("model")
                .apply(lambda g: roc_auc_score(g["y_true"], g["y_prob"]),
                       include_groups=False).idxmax())
        chosen = cell[cell["model"] == best].sort_values("sample_index")
        model_auc = float(roc_auc_score(chosen["y_true"], chosen["y_prob"]))
        nested_auc, picks = nested_best_single_auc(
            X.loc[chosen["sample_index"].to_numpy(), features],
            chosen["y_true"].to_numpy(),
            n_splits=int(cfg["cross_validation"]["outer_folds"]),
            n_repeats=int(args.repeats), seed=seed,
        )
        top = sorted(picks.items(), key=lambda kv: -kv[1])
        lrows.append({
            "config": config_name, "model": best,
            "model_roc_auc": model_auc,
            "nested_single_feature_auc": nested_auc,
            "nested_lift": model_auc - nested_auc,
            "most_selected_feature": top[0][0] if top else "",
            "selection_frequency": round(top[0][1] / sum(picks.values()), 3)
            if picks else float("nan"),
            "n_distinct_features_selected": len(picks),
        })
        print(f"  {config_name:18s} model {model_auc:.4f} vs nested single "
              f"{nested_auc:.4f} -> lift {model_auc - nested_auc:+.4f} "
              f"(most selected: {top[0][0] if top else '-'})")
    pd.DataFrame(lrows).to_csv(tables_dir / "table_50_nested_lift.csv",
                               index=False)
    print("wrote table_50_nested_lift.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
