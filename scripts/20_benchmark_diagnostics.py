"""Stage 20: Benchmark Informativeness Diagnostics on this dataset.

Runs the three BID measurements (see ``ckd.evaluation.informativeness``) on
the analysed benchmark and reports which proposed thresholds they breach.

The standardisation target is not invented: it is the case mix of a
published community-based CKD screening series (Kabir et al., 2026), in
which 22.3% of cases are stage 1, 45.5% stage 2 and 32.1% stage 3, with
stages 4-5 excluded as too rare to screen for. Standardising to it answers
"what would this discrimination be in a population that actually looked
like a screening series?".

Output:
  reports/tables/table_43_bid_lift.csv
  reports/tables/table_43_bid_saturation.csv
  reports/tables/table_43_bid_standardisation.csv
  reports/tables/table_43_bid_summary.csv
  reports/figures/fig_r16_bid_saturation

Usage
-----
    python scripts/20_benchmark_diagnostics.py [--repeats N]
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit  # noqa: E402

from ckd.config import ensure_dirs, load_config  # noqa: E402
from ckd.data.clean import clean_dataset, feature_matrix  # noqa: E402
from ckd.evaluation.bootstrap import pooled_predictions  # noqa: E402
from ckd.evaluation.informativeness import (  # noqa: E402
    FLAG_THRESHOLDS,
    flag,
    fraction_to_reach,
    multivariable_lift,
    standardised_auc,
    standardised_sensitivity,
)
from ckd.evaluation.plots import apply_style, save  # noqa: E402
from ckd.features.configs import get_config  # noqa: E402
from ckd.models.pipeline import assert_pipeline_is_clean, build_pipeline  # noqa: E402

CONFIGS = ["low_cost_model", "laboratory_model", "full_valid_model"]
SATURATION_CONFIG = "low_cost_model"
SATURATION_MODEL = "random_forest"
FRACTIONS = np.array([0.10, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 1.00])

#: Case mix of a published community screening series (Kabir et al. 2026):
#: stage 1 n=25, stage 2 n=51, stage 3 n=36 of 112 cases.
SCREENING_TARGET = {"s1": 25 / 112, "s2": 51 / 112, "s3": 36 / 112}


def saturation_curve(X, y, config_name, model_name, seed, fractions, n_repeats):
    """Held-out AUC as a function of how much training data is used."""
    features = list(get_config(config_name).features)
    Xc = X.loc[:, features]
    rows = []
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fraction in fractions:
        scores = []
        for repeat in range(n_repeats):
            for fold, (train_idx, test_idx) in enumerate(outer.split(Xc, y)):
                if fraction < 1.0:
                    splitter = StratifiedShuffleSplit(
                        n_splits=1, train_size=float(fraction),
                        random_state=seed + 1000 * repeat + fold,
                    )
                    sub, _ = next(splitter.split(Xc.iloc[train_idx], y[train_idx]))
                    use = train_idx[sub]
                else:
                    use = train_idx
                if len(np.unique(y[use])) < 2:
                    continue
                pipe = build_pipeline(model_name, config_name,
                                      seed + 1000 * repeat + fold)
                assert_pipeline_is_clean(pipe, config_name)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe.fit(Xc.iloc[use], y[use])
                    prob = pipe.predict_proba(Xc.iloc[test_idx])[:, 1]
                scores.append(roc_auc_score(y[test_idx], prob))
        rows.append({
            "config": config_name, "model": model_name,
            "train_fraction": float(fraction),
            "n_train_median": int(np.median([len(y)])) if False else
            int(round(float(fraction) * 0.8 * len(y))),
            "mean_roc_auc": float(np.mean(scores)),
            "sd_roc_auc": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "n_fits": len(scores),
        })
        print(f"  fraction {fraction:.2f}: AUC {rows[-1]['mean_roc_auc']:.4f}")
    return pd.DataFrame(rows)


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    proc_dir, tables_dir = ensure_dirs(cfg["paths"]["processed_dir"],
                                       cfg["paths"]["tables_dir"])
    (fig_dir,) = ensure_dirs(cfg["paths"]["figures_dir"])
    apply_style()

    internal = clean_dataset()
    X = feature_matrix(internal)
    y = internal.target.to_numpy()
    stage = internal.clean["stage"].astype("object")
    seed = int(cfg["seed"])
    pooled = pooled_predictions(pd.read_csv(proc_dir / "cv_predictions.csv.gz"))

    # ---- 1. multivariable lift -------------------------------------------
    print("multivariable lift ...")
    lift_rows = []
    for config_name in CONFIGS:
        features = list(get_config(config_name).features)
        cell = pooled[(pooled["config"] == config_name)
                      & (pooled["calibration"] == "isotonic")]
        best = (cell.groupby("model")
                .apply(lambda g: roc_auc_score(g["y_true"], g["y_prob"]),
                       include_groups=False)
                .drop(labels=["dummy"], errors="ignore").idxmax())
        chosen = cell[cell["model"] == best].sort_values("sample_index")
        result = multivariable_lift(
            chosen["y_true"].to_numpy(), chosen["y_prob"].to_numpy(),
            X.loc[chosen["sample_index"].to_numpy(), features], seed=seed,
        )
        lift_rows.append({
            "config": config_name, "model": best,
            "model_roc_auc": result.model_auc,
            "best_single_predictor": result.best_feature,
            "best_single_roc_auc": result.best_feature_auc,
            "multivariable_lift": result.lift,
            "lift_ci_low": result.lift_ci_low,
            "lift_ci_high": result.lift_ci_high,
        })
        print(f"  {config_name:20s} {result.model_auc:.4f} vs "
              f"{result.best_feature} {result.best_feature_auc:.4f} -> "
              f"lift {result.lift:+.4f} "
              f"({result.lift_ci_low:+.4f}, {result.lift_ci_high:+.4f})")
    lift = pd.DataFrame(lift_rows)
    lift.to_csv(tables_dir / "table_43_bid_lift.csv", index=False)

    # ---- 2. saturation ---------------------------------------------------
    print("\nsaturation curve ...")
    curve = saturation_curve(X, y, SATURATION_CONFIG, SATURATION_MODEL,
                             seed, FRACTIONS, args.repeats)
    curve.to_csv(tables_dir / "table_43_bid_saturation.csv", index=False)
    n90 = fraction_to_reach(curve["train_fraction"].to_numpy(),
                            curve["mean_roc_auc"].to_numpy(), 0.90)
    print(f"  fraction reaching 90% of achieved gain: {n90:.2f}")

    # ---- 3. case-mix standardisation -------------------------------------
    print("\ncase-mix standardisation ...")
    std_rows = []
    for config_name in CONFIGS:
        cell = pooled[(pooled["config"] == config_name)
                      & (pooled["calibration"] == "isotonic")]
        best = lift[lift["config"] == config_name]["model"].iloc[0]
        chosen = cell[cell["model"] == best].sort_values("sample_index")
        idx = chosen["sample_index"].to_numpy()
        result = standardised_auc(
            chosen["y_true"].to_numpy(), chosen["y_prob"].to_numpy(),
            stage.iloc[idx].reset_index(drop=True), SCREENING_TARGET,
        )
        sens = standardised_sensitivity(
            chosen["y_true"].to_numpy(), chosen["y_prob"].to_numpy(),
            stage.iloc[idx].reset_index(drop=True), SCREENING_TARGET,
            threshold=float(cfg["evaluation"]["primary_threshold"]),
        )
        std_rows.append({
            "config": config_name, "model": best,
            "observed_roc_auc": result.observed_auc,
            "standardised_roc_auc": result.standardised_auc,
            "standardisation_shift": result.shift,
            "observed_sensitivity": sens.observed_sensitivity,
            "standardised_sensitivity": sens.standardised_sensitivity,
            "sensitivity_standardisation_shift": sens.shift,
            "effective_cases": result.effective_cases,
        })
        print(f"  {config_name:20s} AUC {result.observed_auc:.4f} -> "
              f"{result.standardised_auc:.4f} (shift {result.shift:+.4f}) | "
              f"sens {sens.observed_sensitivity:.4f} -> "
              f"{sens.standardised_sensitivity:.4f} "
              f"(shift {sens.shift:+.4f})")
    standardisation = pd.DataFrame(std_rows)
    standardisation.to_csv(tables_dir / "table_43_bid_standardisation.csv",
                           index=False)

    # ---- summary and flags -----------------------------------------------
    summary_rows = []
    for config_name in CONFIGS:
        lift_row = lift[lift["config"] == config_name].iloc[0]
        std_row = standardisation[standardisation["config"] == config_name].iloc[0]
        diagnostics = {
            "multivariable_lift": float(lift_row["multivariable_lift"]),
            "fraction_to_reach_90pct": float(n90)
            if config_name == SATURATION_CONFIG else float("nan"),
            "standardisation_shift": float(std_row["standardisation_shift"]),
            "sensitivity_standardisation_shift":
                float(std_row["sensitivity_standardisation_shift"]),
        }
        flags = flag(diagnostics)
        summary_rows.append({
            "config": config_name,
            **{k: round(v, 4) for k, v in diagnostics.items()},
            **{k: bool(v) for k, v in flags.items()},
            "n_flags_raised": int(sum(flags.values())),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(tables_dir / "table_43_bid_summary.csv", index=False)
    print("\n--- BID summary ---")
    print(summary.to_string(index=False))
    print(f"\nthresholds (proposed, not established): {FLAG_THRESHOLDS}")

    # ---- figure ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.errorbar(curve["train_fraction"], curve["mean_roc_auc"],
                yerr=curve["sd_roc_auc"], marker="o", lw=1.8, capsize=3,
                color="#0072B2")
    ceiling = float(curve["mean_roc_auc"].max())
    ax.axhline(0.5 + 0.9 * (ceiling - 0.5), ls="--", lw=1.0, color="#a0453a",
               label="90% of achieved gain above chance")
    if np.isfinite(n90):
        ax.axvline(n90, ls=":", lw=1.2, color="#a0453a")
        ax.annotate(f"reached at {n90:.0%}\nof the training data",
                    xy=(n90, 0.5 + 0.9 * (ceiling - 0.5)),
                    xytext=(n90 + 0.10, 0.5 + 0.45 * (ceiling - 0.5)),
                    fontsize=9, color="#a0453a",
                    arrowprops=dict(arrowstyle="->", color="#a0453a", lw=1.1))
    ax.axhline(0.5, lw=0.9, color="0.5")
    ax.set_xlabel("fraction of the training fold used")
    ax.set_ylabel("held-out ROC-AUC")
    ax.set_title("Saturation: how much data the benchmark actually needs")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    save(fig, fig_dir, "fig_r16_bid_saturation")
    plt.close(fig)
    print("wrote fig_r16_bid_saturation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
