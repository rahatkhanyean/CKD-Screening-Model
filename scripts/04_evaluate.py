"""Stage 4: metrics, confidence intervals, calibration analysis and leakage audit.

Consumes only ``data/processed/cv_predictions.csv.gz``. Produces every headline
table and every results figure. Nothing is refitted here, so the tables cannot
drift away from the cross-validation that produced the predictions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ckd.config import ensure_dirs, load_config  # noqa: E402
from ckd.evaluation.bootstrap import (  # noqa: E402
    GROUP_KEYS,
    bootstrap_ci,
    calibration_curve_points,
    metrics_per_fold,
    metrics_per_repeat,
    pooled_predictions,
    summarise_across_repeats,
)
from ckd.evaluation.metrics import HEADLINE_METRICS, METRIC_LABELS  # noqa: E402
from ckd.evaluation.plots import (  # noqa: E402
    annotate_invalid,  # noqa: F401  (kept for reuse by ad-hoc figures)
    apply_style,
    calibration_label,
    config_colour,
    config_label,
    forest_plot,
    model_label,
    save,
)
from ckd.evaluation.thresholds import net_benefit, treat_all_net_benefit  # noqa: E402
from ckd.features.configs import FEATURE_CONFIGS  # noqa: E402

CONFIG_ORDER = [
    "leaky_model",
    "full_valid_model",
    "laboratory_model",
    "low_cost_plus_urine_micro_model",
    "low_cost_model",
    "clinical_only_model",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", default="cv_predictions")
    return p.parse_args()


def _order(df: pd.DataFrame, col: str = "config") -> pd.DataFrame:
    present = [c for c in CONFIG_ORDER if c in set(df[col])]
    return df.set_index(col).loc[present].reset_index()


def main() -> int:
    args = parse_args()
    cfg = load_config()
    proc_dir, fig_dir, tab_dir = ensure_dirs(
        cfg["paths"]["processed_dir"], cfg["paths"]["figures_dir"], cfg["paths"]["tables_dir"]
    )
    apply_style()

    thr_primary = float(cfg["evaluation"]["primary_threshold"])
    n_boot = int(cfg["evaluation"]["bootstrap_iterations"])
    ci_level = float(cfg["evaluation"]["bootstrap_ci"])
    n_bins = int(cfg["evaluation"]["calibration_bins"])
    seed = int(cfg["seed"])

    preds = pd.read_csv(proc_dir / f"{args.predictions}.csv.gz")
    print(f"Loaded {len(preds):,} prediction rows")
    print(f"  configs      : {sorted(preds['config'].unique())}")
    print(f"  models       : {sorted(preds['model'].unique())}")
    print(f"  calibrations : {sorted(preds['calibration'].unique())}")
    print(f"  repeats      : {preds['repeat'].nunique()}")

    # ------------------------------------------------------------------
    # Core metric tables
    # ------------------------------------------------------------------
    print("\nComputing metrics ...")
    per_repeat = metrics_per_repeat(preds, thr_primary, n_bins)
    per_fold = metrics_per_fold(preds, thr_primary, n_bins)
    pooled = pooled_predictions(preds)

    per_repeat.to_csv(tab_dir / "table_08_metrics_per_repeat.csv", index=False, encoding="utf-8")
    per_fold.to_csv(tab_dir / "table_09_metrics_per_outer_fold.csv", index=False, encoding="utf-8")

    summary = summarise_across_repeats(per_repeat)
    summary.to_csv(tab_dir / "table_10_metrics_summary_across_repeats.csv",
                   index=False, encoding="utf-8")

    # ------------------------------------------------------------------
    # Bootstrap CIs on the pooled out-of-fold predictions
    # ------------------------------------------------------------------
    cells = list(pooled.groupby(GROUP_KEYS))
    print(f"Bootstrapping {len(cells)} cells x {n_boot} resamples (parallel) ...")

    def _one_cell(i, keys, grp):
        ci = bootstrap_ci(
            grp["y_true"].to_numpy(),
            grp["y_prob"].to_numpy(),
            threshold=thr_primary,
            screening_threshold=float(grp["screening_threshold"].mean()),
            n_boot=n_boot,
            ci=ci_level,
            seed=seed + i,
            n_bins=n_bins,
        )
        for k, v in zip(GROUP_KEYS, keys):
            ci.insert(0, k, v)
        return ci

    ci_rows = Parallel(n_jobs=-1, verbose=0)(
        delayed(_one_cell)(i, keys, grp) for i, (keys, grp) in enumerate(cells, start=1)
    )
    ci_all = pd.concat(ci_rows, ignore_index=True)
    ci_all.to_csv(tab_dir / "table_11_bootstrap_confidence_intervals.csv",
                  index=False, encoding="utf-8")

    def ci_lookup(config: str, model: str, calib: str, metric: str) -> tuple[float, float, float]:
        r = ci_all[
            (ci_all["config"] == config) & (ci_all["model"] == model)
            & (ci_all["calibration"] == calib) & (ci_all["metric"] == metric)
        ]
        if r.empty:
            return float("nan"), float("nan"), float("nan")
        r = r.iloc[0]
        return float(r["estimate"]), float(r["ci_low"]), float(r["ci_high"])

    # ------------------------------------------------------------------
    # Headline table: uncalibrated cells, every config x model
    # ------------------------------------------------------------------
    rows = []
    for (config, model, calib), grp in pooled.groupby(GROUP_KEYS):
        row = {
            "config": config,
            "model": model,
            "calibration": calib,
            "valid": FEATURE_CONFIGS[config].valid_for_clinical_interpretation,
            "n_features": len(FEATURE_CONFIGS[config].features),
        }
        for metric in HEADLINE_METRICS:
            est, lo, hi = ci_lookup(config, model, calib, metric)
            row[metric] = est
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
        for metric in ("screening_sensitivity", "screening_specificity",
                       "screening_npv", "screening_precision", "screening_threshold"):
            est, lo, hi = ci_lookup(config, model, calib, metric)
            row[metric] = est
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
        sd = per_repeat[
            (per_repeat["config"] == config) & (per_repeat["model"] == model)
            & (per_repeat["calibration"] == calib)
        ]["roc_auc"].std()
        row["roc_auc_sd_across_repeats"] = float(sd) if sd == sd else float("nan")
        rows.append(row)
    headline = pd.DataFrame(rows)
    headline.to_csv(tab_dir / "table_12_headline_results.csv", index=False, encoding="utf-8")

    # ------------------------------------------------------------------
    # Leakage audit
    # ------------------------------------------------------------------
    print("\nLeakage audit ...")
    audit = headline[headline["calibration"] == "none"].copy()
    audit = audit[audit["model"] != "dummy"]
    best_per_config = (
        audit.sort_values("roc_auc", ascending=False).groupby("config", as_index=False).first()
    )
    best_per_config = _order(best_per_config)
    leaky_auc = float(
        best_per_config.loc[best_per_config["config"] == "leaky_model", "roc_auc"].iloc[0]
    )
    full_auc = float(
        best_per_config.loc[best_per_config["config"] == "full_valid_model", "roc_auc"].iloc[0]
    )
    best_per_config["roc_auc_inflation_vs_full_valid"] = best_per_config["roc_auc"] - full_auc
    best_per_config["error_reduction_vs_full_valid"] = (
        (best_per_config["roc_auc"] - full_auc) / (1.0 - full_auc)
    )
    best_per_config.to_csv(tab_dir / "table_13_leakage_audit.csv", index=False, encoding="utf-8")
    print(f"  best leaky ROC-AUC      : {leaky_auc:.4f}")
    print(f"  best full-valid ROC-AUC : {full_auc:.4f}")
    print(f"  absolute inflation      : {leaky_auc - full_auc:+.4f}")

    # ------------------------------------------------------------------
    # Figure R1. Model x configuration ROC-AUC heatmap
    # ------------------------------------------------------------------
    hm = (
        headline[headline["calibration"] == "none"]
        .pivot(index="model", columns="config", values="roc_auc")
    )
    cols = [c for c in CONFIG_ORDER if c in hm.columns]
    model_order = [m for m in ["dummy", "logreg", "svm", "random_forest", "xgboost", "ebm"]
                   if m in hm.index]
    hm = hm.loc[model_order, cols]

    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    im = ax.imshow(hm.to_numpy(), cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto")
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            v = hm.iloc[i, j]
            if v == v:
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                        color="white" if v < 0.85 else "black")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([config_label(c) for c in cols], rotation=18, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(model_order)))
    ax.set_yticklabels([model_label(m) for m in model_order], fontsize=9)
    ax.set_title("Figure R1. Nested-CV ROC-AUC by model and feature configuration\n"
                 "(uncalibrated; pooled out-of-fold predictions)")
    ax.grid(False)
    for j, c in enumerate(cols):
        if not FEATURE_CONFIGS[c].valid_for_clinical_interpretation:
            ax.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, hm.shape[0],
                                       fill=False, ec="#B22222", lw=2.5, zorder=5))
    fig.colorbar(im, ax=ax, shrink=0.85, label="ROC-AUC")
    ax.text(1.0, -0.30, "Red outline = deliberately invalid (leakage demonstration)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color="#B22222", style="italic")
    save(fig, fig_dir, "fig_r1_auc_heatmap")
    print("  fig_r1_auc_heatmap")

    # ------------------------------------------------------------------
    # Figure R2. Leakage audit
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), gridspec_kw={"wspace": 0.32})

    ax = axes[0]
    labels = [f"{config_label(c)}\n({m})" for c, m in
              zip(best_per_config["config"], best_per_config["model"].map(model_label))]
    colours = [config_colour(c) for c in best_per_config["config"]]
    forest_plot(
        ax, labels,
        best_per_config["roc_auc"].to_numpy(),
        best_per_config["roc_auc_ci_low"].to_numpy(),
        best_per_config["roc_auc_ci_high"].to_numpy(),
        colours=colours, xlabel="ROC-AUC (95% bootstrap CI)", reference=0.5,
    )
    lo_x = float(np.nanmin(best_per_config["roc_auc_ci_low"]))
    ax.set_xlim(min(0.48, lo_x - 0.03), 1.02)
    ax.set_title("A. Best model per configuration")

    ax = axes[1]
    sub = headline[(headline["calibration"] == "none") & (headline["model"] != "dummy")]
    for c in [c for c in CONFIG_ORDER if c in set(sub["config"])]:
        ss = sub[sub["config"] == c].sort_values("roc_auc")
        ax.scatter(ss["roc_auc"], [config_label(c)] * len(ss),
                   color=config_colour(c), s=42, alpha=0.85,
                   edgecolor="white", linewidth=0.6)
    ax.set_xlabel("ROC-AUC (each point = one model)")
    ax.set_xlim(min(0.48, float(sub["roc_auc"].min()) - 0.03), 1.02)
    ax.axvline(0.5, color="#555555", ls="--", lw=0.9)
    ax.set_title("B. Spread across models within each configuration")
    ax.grid(axis="y", visible=False)

    fig.suptitle("Figure R2. Leakage audit: how much do prohibited variables inflate performance?",
                 y=1.03, fontsize=12, fontweight="bold")
    fig.text(0.5, -0.03,
             "The leaky configuration is INVALID and is shown only to quantify leakage-driven "
             "inflation. Note that valid configurations already sit near the ceiling "
             "(see Figure R12).",
             ha="center", fontsize=8.5, style="italic", color="#B22222")
    save(fig, fig_dir, "fig_r2_leakage_audit")
    print("  fig_r2_leakage_audit")

    # ------------------------------------------------------------------
    # Figure R3. Screening metrics for valid configurations
    # ------------------------------------------------------------------
    valid_best = best_per_config[best_per_config["config"] != "leaky_model"].copy()
    metrics_to_show = ["sensitivity", "specificity", "npv", "precision",
                       "balanced_accuracy", "brier"]
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.2),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.42})
    for ax, metric in zip(axes.ravel(), metrics_to_show):
        labels = [config_label(c) for c in valid_best["config"]]
        forest_plot(
            ax, labels,
            valid_best[metric].to_numpy(),
            valid_best[f"{metric}_ci_low"].to_numpy(),
            valid_best[f"{metric}_ci_high"].to_numpy(),
            colours=[config_colour(c) for c in valid_best["config"]],
            xlabel=METRIC_LABELS[metric],
        )
        ax.set_title(METRIC_LABELS[metric], fontsize=10)
        if metric != "brier":
            ax.set_xlim(0, 1.02)
    fig.suptitle(
        f"Figure R3. Screening performance of valid configurations at the prespecified "
        f"threshold p = {thr_primary}\n(best model per configuration; 95% bootstrap CI)",
        y=1.02, fontsize=12, fontweight="bold",
    )
    save(fig, fig_dir, "fig_r3_screening_metrics")
    print("  fig_r3_screening_metrics")

    # ------------------------------------------------------------------
    # Figure R4. ROC and PR curves
    # ------------------------------------------------------------------
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), gridspec_kw={"wspace": 0.26})
    for _, r in best_per_config.iterrows():
        g = pooled[(pooled["config"] == r["config"]) & (pooled["model"] == r["model"])
                   & (pooled["calibration"] == "none")]
        fpr, tpr, _ = roc_curve(g["y_true"], g["y_prob"])
        style = "--" if r["config"] == "leaky_model" else "-"
        axes[0].plot(fpr, tpr, style, color=config_colour(r["config"]), lw=2.0,
                     label=f"{config_label(r['config'])} ({r['roc_auc']:.3f})")
        prec, rec, _ = precision_recall_curve(g["y_true"], g["y_prob"])
        axes[1].plot(rec, prec, style, color=config_colour(r["config"]), lw=2.0,
                     label=f"{config_label(r['config'])} ({r['pr_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], color="#888888", ls=":", lw=1.0)
    axes[0].set_xlabel("1 - specificity"); axes[0].set_ylabel("Sensitivity")
    axes[0].set_title("A. ROC curves"); axes[0].legend(loc="lower right", fontsize=8)
    prevalence = float(pooled["y_true"].mean())
    axes[1].axhline(prevalence, color="#888888", ls=":", lw=1.0)
    axes[1].set_xlabel("Recall (sensitivity)"); axes[1].set_ylabel("Precision (PPV)")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("B. Precision-recall curves"); axes[1].legend(loc="lower left", fontsize=8)
    fig.suptitle("Figure R4. Discrimination, pooled out-of-fold predictions",
                 y=1.02, fontsize=12, fontweight="bold")
    fig.text(0.5, -0.03,
             "Dashed red curve = the INVALID leaky configuration, shown for comparison only.",
             ha="center", fontsize=8.5, style="italic", color="#B22222")
    save(fig, fig_dir, "fig_r4_roc_pr_curves")
    print("  fig_r4_roc_pr_curves")

    # ------------------------------------------------------------------
    # Calibration analysis
    # ------------------------------------------------------------------
    # NOTE ON METHOD. Calibration statistics are computed PER REPEAT and then
    # averaged, never on the repeat-averaged probabilities used elsewhere.
    # Averaging a patient's probability over repeats shrinks it towards the
    # centre of the distribution; that shrinkage flattens the predictions and
    # biases the calibration slope upwards, making any model look
    # systematically under-confident. Within a single repeat each patient is
    # predicted exactly once, so no shrinkage occurs. Discrimination metrics
    # are rank-based and largely unaffected, which is why the pooled
    # predictions are still used for ROC/PR curves and their bootstrap CIs.
    print("\nCalibration analysis (per-repeat, to avoid averaging shrinkage) ...")
    cal_metrics = ["brier", "calibration_slope", "calibration_intercept", "ece", "roc_auc"]
    agg = per_repeat.groupby(GROUP_KEYS)[cal_metrics].agg(["mean", "std", "min", "max"])
    agg.columns = [f"{a}__{b}" for a, b in agg.columns]
    calib_df = agg.reset_index()
    for m in cal_metrics:
        calib_df[m] = calib_df[f"{m}__mean"]
    calib_df["valid"] = calib_df["config"].map(
        lambda c: FEATURE_CONFIGS[c].valid_for_clinical_interpretation
    )
    # Range across repeats, reported instead of a bootstrap interval because
    # these statistics are averages of per-repeat quantities.
    for m in cal_metrics:
        calib_df[f"{m}_repeat_min"] = calib_df[f"{m}__min"]
        calib_df[f"{m}_repeat_max"] = calib_df[f"{m}__max"]
        calib_df[f"{m}_repeat_sd"] = calib_df[f"{m}__std"]
    calib_df.to_csv(tab_dir / "table_14_calibration_results.csv", index=False, encoding="utf-8")

    # Calibration curve points. Predictions from all repeats are STACKED rather
    # than averaged, for the same reason: stacking preserves each repeat's
    # un-shrunk probability scale. Each patient therefore contributes one point
    # per repeat, which makes the bins denser without altering the probability
    # distribution being displayed.
    curve_rows = []
    for (config, model, calib), grp in preds.groupby(GROUP_KEYS):
        pts = calibration_curve_points(grp["y_true"].to_numpy(), grp["y_prob"].to_numpy(), n_bins)
        pts.insert(0, "calibration", calib); pts.insert(0, "model", model); pts.insert(0, "config", config)
        curve_rows.append(pts)
    pd.concat(curve_rows, ignore_index=True).to_csv(
        tab_dir / "table_15_calibration_curve_points.csv", index=False, encoding="utf-8"
    )

    # Which model is best among valid configurations? (documented rule)
    valid_cells = headline[
        headline["config"].map(lambda c: FEATURE_CONFIGS[c].valid_for_clinical_interpretation)
        & (headline["model"] != "dummy")
    ]
    best_overall = valid_cells.sort_values(
        ["roc_auc", "brier"], ascending=[False, True]
    ).iloc[0]
    best_low_cost = (
        valid_cells[valid_cells["config"] == "low_cost_model"]
        .sort_values(["roc_auc", "brier"], ascending=[False, True]).iloc[0]
    )
    print(f"  best valid cell    : {best_overall['config']} / {best_overall['model']} "
          f"/ {best_overall['calibration']} (AUC {best_overall['roc_auc']:.4f})")
    print(f"  best low-cost cell : {best_low_cost['model']} / {best_low_cost['calibration']} "
          f"(AUC {best_low_cost['roc_auc']:.4f})")

    pd.DataFrame([
        {"selection": "best_valid_overall", **{k: best_overall[k] for k in
         ["config", "model", "calibration", "roc_auc", "pr_auc", "sensitivity",
          "specificity", "npv", "brier", "calibration_slope", "calibration_intercept"]}},
        {"selection": "best_low_cost", **{k: best_low_cost[k] for k in
         ["config", "model", "calibration", "roc_auc", "pr_auc", "sensitivity",
          "specificity", "npv", "brier", "calibration_slope", "calibration_intercept"]}},
    ]).to_csv(tab_dir / "table_16_selected_models.csv", index=False, encoding="utf-8")

    # ------------------------------------------------------------------
    # Figure R5. Calibration curves for the low-cost and full-valid models
    # ------------------------------------------------------------------
    focus_configs = [c for c in ("low_cost_model", "full_valid_model", "laboratory_model")
                     if c in set(pooled["config"])]
    fig, axes = plt.subplots(1, len(focus_configs), figsize=(4.7 * len(focus_configs), 4.9),
                             gridspec_kw={"wspace": 0.28})
    axes = np.atleast_1d(axes)
    for ax, config in zip(axes, focus_configs):
        model = (
            valid_cells[valid_cells["config"] == config]
            .sort_values("roc_auc", ascending=False).iloc[0]["model"]
        )
        ax.plot([0, 1], [0, 1], color="#888888", ls=":", lw=1.2, label="Perfect")
        for k, calib in enumerate(["none", "sigmoid", "isotonic"]):
            g = preds[(preds["config"] == config) & (preds["model"] == model)
                      & (preds["calibration"] == calib)]
            if g.empty:
                continue
            pts = calibration_curve_points(g["y_true"].to_numpy(), g["y_prob"].to_numpy(), n_bins)
            row = calib_df[(calib_df["config"] == config) & (calib_df["model"] == model)
                           & (calib_df["calibration"] == calib)].iloc[0]
            ax.plot(pts["mean_predicted"], pts["observed_rate"], "o-",
                    color=["#0072B2", "#D55E00", "#009E73"][k], lw=1.8, ms=5,
                    label=(f"{calibration_label(calib)}: "
                           f"slope {row['calibration_slope']:.2f}, "
                           f"int {row['calibration_intercept']:+.2f}"))
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed CKD proportion")
        ax.set_title(f"{config_label(config)}\n({model_label(model)})", fontsize=10)
        ax.legend(loc="upper left", fontsize=7.5)
    fig.suptitle(
        "Figure R5. Calibration curves (quantile bins; predictions stacked across "
        "repeats,\nnot averaged, to avoid shrinkage)",
        y=1.05, fontsize=12, fontweight="bold",
    )
    save(fig, fig_dir, "fig_r5_calibration_curves")
    print("  fig_r5_calibration_curves")

    # ------------------------------------------------------------------
    # Figure R6. Calibration method comparison
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4), gridspec_kw={"wspace": 0.34})
    sub = calib_df[calib_df["valid"] & (calib_df["model"] != "dummy")]
    for ax, (metric, ref, label) in zip(
        axes,
        [("calibration_slope", 1.0, "Calibration slope (ideal = 1)"),
         ("calibration_intercept", 0.0, "Calibration intercept (ideal = 0)"),
         ("brier", None, "Brier score (lower is better)")],
    ):
        for k, calib in enumerate(["none", "sigmoid", "isotonic"]):
            s = sub[sub["calibration"] == calib]
            ax.scatter(s[metric], np.arange(len(s)) * 0 + k + np.random.default_rng(k).normal(0, 0.07, len(s)),
                       color=["#0072B2", "#D55E00", "#009E73"][k], s=34, alpha=0.75,
                       edgecolor="white", linewidth=0.5)
            med = s[metric].median()
            ax.plot([med, med], [k - 0.28, k + 0.28], color="black", lw=2.0, zorder=5)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels([calibration_label(c) for c in ["none", "sigmoid", "isotonic"]])
        ax.set_xlabel(label)
        if ref is not None:
            ax.axvline(ref, color="#555555", ls="--", lw=1.0)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Figure R6. Effect of calibration method across all valid configurations and models\n"
                 "(black bar = median; each point = one configuration x model cell)",
                 y=1.06, fontsize=12, fontweight="bold")
    save(fig, fig_dir, "fig_r6_calibration_methods")
    print("  fig_r6_calibration_methods")

    # ------------------------------------------------------------------
    # Figure R7. Threshold trade-off and decision curve for the low-cost model
    # ------------------------------------------------------------------
    lc_model = best_low_cost["model"]
    lc_calib = best_low_cost["calibration"]
    g = pooled[(pooled["config"] == "low_cost_model") & (pooled["model"] == lc_model)
               & (pooled["calibration"] == lc_calib)]
    yt = g["y_true"].to_numpy(); yp = g["y_prob"].to_numpy()

    from ckd.evaluation.metrics import threshold_metrics

    grid = np.linspace(0.01, 0.99, 99)
    tm = [threshold_metrics(yt, yp, t) for t in grid]
    trade = pd.DataFrame({
        "threshold": grid,
        "sensitivity": [m.sensitivity for m in tm],
        "specificity": [m.specificity for m in tm],
        "npv": [m.npv for m in tm],
        "precision": [m.precision for m in tm],
        "missed_cases": [m.fn for m in tm],
        "unnecessary_referrals": [m.fp for m in tm],
        "net_benefit": [net_benefit(yt, yp, t) for t in grid],
        "net_benefit_refer_all": [treat_all_net_benefit(yt, t) for t in grid],
    })
    trade.to_csv(tab_dir / "table_17_threshold_tradeoff_low_cost.csv", index=False, encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), gridspec_kw={"wspace": 0.34})
    ax = axes[0]
    for col, colour in [("sensitivity", "#0072B2"), ("specificity", "#D55E00"),
                        ("npv", "#009E73"), ("precision", "#CC79A7")]:
        ax.plot(trade["threshold"], trade[col], color=colour, lw=1.9, label=METRIC_LABELS[col])
    ax.axvline(thr_primary, color="#555555", ls="--", lw=1.0)
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Metric value"); ax.set_ylim(0, 1.02)
    ax.set_title("A. Operating characteristics"); ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    ax.plot(trade["threshold"], trade["missed_cases"], color="#B22222", lw=2.0,
            label="Missed CKD cases (FN)")
    ax.plot(trade["threshold"], trade["unnecessary_referrals"], color="#0072B2", lw=2.0,
            label="Unnecessary referrals (FP)")
    ax.axvline(thr_primary, color="#555555", ls="--", lw=1.0)
    ax.set_xlabel("Decision threshold"); ax.set_ylabel(f"Patients (of {len(yt)})")
    ax.set_title("B. Missed cases vs unnecessary referrals"); ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(trade["threshold"], trade["net_benefit"], color="#0072B2", lw=2.0, label="Model")
    ax.plot(trade["threshold"], trade["net_benefit_refer_all"], color="#888888", lw=1.4,
            ls="--", label="Refer everyone")
    ax.axhline(0, color="#333333", lw=1.0, label="Refer no one")
    ax.set_ylim(-0.15, max(0.4, float(trade["net_benefit"].max()) * 1.15))
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
    ax.set_title("C. Decision curve (EXPLORATORY)"); ax.legend(fontsize=8)

    fig.suptitle(
        f"Figure R7. Screening trade-offs, low-cost model ({model_label(lc_model)}, "
        f"{calibration_label(lc_calib)})", y=1.03, fontsize=12, fontweight="bold",
    )
    save(fig, fig_dir, "fig_r7_threshold_tradeoff")
    print("  fig_r7_threshold_tradeoff")

    # ------------------------------------------------------------------
    # Confusion matrices at both operating points
    # ------------------------------------------------------------------
    cm_rows = []
    for config in [c for c in CONFIG_ORDER if c in set(pooled["config"])]:
        sel = valid_cells if config != "leaky_model" else headline[headline["model"] != "dummy"]
        s = sel[sel["config"] == config].sort_values("roc_auc", ascending=False)
        if s.empty:
            continue
        r = s.iloc[0]
        g = pooled[(pooled["config"] == config) & (pooled["model"] == r["model"])
                   & (pooled["calibration"] == r["calibration"])]
        yt2, yp2 = g["y_true"].to_numpy(), g["y_prob"].to_numpy()
        for point, thr in [("prespecified_0.50", thr_primary),
                           ("screening_target_sens", float(g["screening_threshold"].mean()))]:
            m = threshold_metrics(yt2, yp2, thr)
            cm_rows.append({
                "config": config, "model": r["model"], "calibration": r["calibration"],
                "operating_point": point, **m.as_dict(),
            })
    pd.DataFrame(cm_rows).to_csv(
        tab_dir / "table_18_confusion_matrices.csv", index=False, encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # Figure R8. Confusion matrices for the low-cost model
    # ------------------------------------------------------------------
    lc_thr_screen = float(
        pooled[(pooled["config"] == "low_cost_model") & (pooled["model"] == lc_model)
               & (pooled["calibration"] == lc_calib)]["screening_threshold"].mean()
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.3), gridspec_kw={"wspace": 0.42})
    for ax, (title, thr) in zip(axes, [
        (f"A. Prespecified threshold p = {thr_primary:.2f}", thr_primary),
        (f"B. Screening threshold p = {lc_thr_screen:.3f}\n(chosen in training folds)",
         lc_thr_screen),
    ]):
        m = threshold_metrics(yt, yp, thr)
        mat = np.array([[m.tn, m.fp], [m.fn, m.tp]])
        im = ax.imshow(mat, cmap="Blues", aspect="auto")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]}", ha="center", va="center", fontsize=17,
                        fontweight="bold",
                        color="white" if mat[i, j] > mat.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted\nnon-CKD", "Predicted\nCKD"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual\nnon-CKD", "Actual\nCKD"])
        ax.set_title(f"{title}\nSens {m.sensitivity:.2f} | Spec {m.specificity:.2f} | "
                     f"NPV {m.npv:.2f} | PPV {m.precision:.2f}", fontsize=9.5)
        ax.grid(False)
    fig.suptitle(f"Figure R8. Low-cost model confusion matrices "
                 f"({model_label(lc_model)}, {calibration_label(lc_calib)}, "
                 f"pooled out-of-fold, n = {len(yt)})",
                 y=1.04, fontsize=11.5, fontweight="bold")
    save(fig, fig_dir, "fig_r8_confusion_matrices")
    print("  fig_r8_confusion_matrices")

    # ------------------------------------------------------------------
    # Figure R9. Uncertainty: spread across repeats and folds
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.6), gridspec_kw={"wspace": 0.42})
    valid_cfgs = [c for c in CONFIG_ORDER if c != "leaky_model" and c in set(per_fold["config"])]

    def _wrap(name: str) -> str:
        # The longest configuration name otherwise runs into the neighbouring
        # panel, because tick labels are drawn outside their own axes.
        lab = config_label(name)
        return lab.replace(" + ", "\n+ ") if len(lab) > 16 else lab

    ax = axes[0]
    data, labels, colours = [], [], []
    for c in valid_cfgs:
        r = valid_cells[valid_cells["config"] == c].sort_values("roc_auc", ascending=False).iloc[0]
        vals = per_fold[(per_fold["config"] == c) & (per_fold["model"] == r["model"])
                        & (per_fold["calibration"] == r["calibration"])]["roc_auc"].dropna()
        data.append(vals.to_numpy()); labels.append(_wrap(c)); colours.append(config_colour(c))
    bp = ax.boxplot(data, orientation="horizontal", patch_artist=True, widths=0.6,
                    medianprops=dict(color="black", lw=1.6))
    for patch, col in zip(bp["boxes"], colours):
        patch.set_facecolor(col); patch.set_alpha(0.55)
    ax.set_yticklabels(labels)
    ax.set_xlabel("ROC-AUC within a single outer test fold (n approx. 40)")
    ax.set_title("A. Fold-level spread")
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    for c in valid_cfgs:
        r = valid_cells[valid_cells["config"] == c].sort_values("roc_auc", ascending=False).iloc[0]
        vals = per_repeat[(per_repeat["config"] == c) & (per_repeat["model"] == r["model"])
                          & (per_repeat["calibration"] == r["calibration"])]["roc_auc"]
        ax.scatter(vals, [_wrap(c)] * len(vals), color=config_colour(c),
                   s=52, alpha=0.85, edgecolor="white", linewidth=0.6)
    ax.set_xlabel("ROC-AUC of one complete repeat (all 200 patients)")
    ax.set_title("B. Repeat-level spread")
    ax.grid(axis="y", visible=False)

    fig.suptitle("Figure R9. Uncertainty from cross-validation partitioning\n"
                 "(this is NOT sampling uncertainty about the population)",
                 y=1.05, fontsize=12, fontweight="bold")
    save(fig, fig_dir, "fig_r9_uncertainty")
    print("  fig_r9_uncertainty")

    # ------------------------------------------------------------------
    # Case-mix / spectrum analysis  (EXPLORATORY, post hoc)
    #
    # The valid configurations reach discrimination close to the ceiling. That
    # has to be explained before it is celebrated, so the same out-of-fold
    # predictions are re-evaluated within clinically defined subgroups.
    # `stage` is used ONLY to partition patients for evaluation; it never
    # entered any model as a predictor.
    # ------------------------------------------------------------------
    print("\nCase-mix (spectrum) analysis ...")
    from ckd.data.clean import clean_dataset, feature_matrix
    from ckd.evaluation.spectrum import (
        SUBGROUP_LABELS,
        separability_report,
        spectrum_analysis,
    )

    result = clean_dataset()
    stage_by_index = result.clean["stage"]
    Xfull = feature_matrix(result)
    yfull = result.target.to_numpy()

    spec = spectrum_analysis(pooled, stage_by_index, threshold=thr_primary)
    spec.to_csv(tab_dir / "table_21_spectrum_analysis.csv", index=False, encoding="utf-8")

    sep = separability_report(
        Xfull, yfull, list(FEATURE_CONFIGS["full_valid_model"].features)
    )
    sep.to_csv(tab_dir / "table_22_univariate_separability.csv", index=False, encoding="utf-8")
    print(f"  most separable single predictor: {sep.iloc[0]['feature']} "
          f"(univariate AUC {sep.iloc[0]['univariate_auc']:.3f}; "
          f"{sep.iloc[0]['fraction_patients_in_overlap']:.0%} of patients lie in the "
          f"overlapping value range)")

    stage_counts = (
        result.clean.loc[yfull == 1, "stage"].value_counts().sort_index()
    )
    advanced = int(stage_counts.reindex(["s3", "s4", "s5"]).fillna(0).sum())
    print(f"  CKD patients staged s3-s5: {advanced} of {int(yfull.sum())} "
          f"({advanced/int(yfull.sum()):.0%})")

    # Figure R12: spectrum effect
    focus = []
    for cname in ("full_valid_model", "laboratory_model", "low_cost_model",
                  "clinical_only_model"):
        s = valid_cells[valid_cells["config"] == cname]
        if s.empty:
            continue
        r = s.sort_values(["roc_auc", "brier"], ascending=[False, True]).iloc[0]
        focus.append((cname, r["model"], r["calibration"]))

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9),
                             gridspec_kw={"wspace": 0.34, "width_ratios": [1.1, 1.1, 1]})
    sub_order = ["all_patients", "advanced_ckd_only", "moderate_ckd_only", "early_ckd_only"]
    short_labels = ["All patients\n(as sampled)", "Advanced\ns4-s5", "Moderate\ns3", "Early\ns1-s2"]
    width = 0.2

    for ax, metric, title, ylab, ylim in [
        (axes[0], "roc_auc", "A. Discrimination by disease severity", "ROC-AUC", (0.4, 1.05)),
        (axes[1], "sensitivity",
         f"B. Sensitivity at the prespecified threshold p = {thr_primary:.2f}",
         "Sensitivity (CKD detected)", (0.0, 1.30)),
    ]:
        for k, (cname, mname, calname) in enumerate(focus):
            vals = []
            for sg in sub_order:
                row = spec[(spec["config"] == cname) & (spec["model"] == mname)
                           & (spec["calibration"] == calname) & (spec["subgroup"] == sg)]
                vals.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
            pos = np.arange(len(sub_order)) + (k - len(focus) / 2 + 0.5) * width
            ax.bar(pos, vals, width=width, color=config_colour(cname),
                   label=config_label(cname))
        ax.set_xticks(np.arange(len(sub_order)))
        ax.set_xticklabels(short_labels, fontsize=8.5)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10.5)
        if metric == "roc_auc":
            ax.axhline(0.5, color="#555555", ls="--", lw=1.0)
            ax.legend(fontsize=8, loc="lower left", ncols=2)

    # Annotate the early-CKD sensitivity drop, which is the clinically decisive one.
    early = spec[(spec["subgroup"] == "early_ckd_only")]
    if not early.empty:
        n_early = int(early["n_ckd"].iloc[0])
        axes[1].text(
            0.5, 0.985,
            f"Early CKD: only {n_early} of the {int(spec[spec['subgroup']=='all_patients']['n_ckd'].iloc[0])} "
            f"CKD patients.\nThese are the cases screening exists to find.",
            transform=axes[1].transAxes, ha="center", va="top", fontsize=8.2,
            style="italic", color="#B22222",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FDECEC", ec="#B22222", lw=0.6),
        )

    ax = axes[2]
    top = sep.head(10).iloc[::-1]
    ax.barh(range(len(top)), top["univariate_auc"], color="#D55E00", height=0.7)
    for i, (_, r) in enumerate(top.iterrows()):
        ax.text(r["univariate_auc"] + 0.008, i, f"{r['fraction_patients_in_overlap']:.0%}",
                va="center", fontsize=7.5, color="#333333")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=9)
    ax.set_xlim(0.5, 1.10)
    ax.set_xlabel("Univariate ROC-AUC (whole dataset)")
    ax.set_title("C. Single predictors already\nseparate the groups", fontsize=10.5)
    ax.text(0.985, 0.02, "% = patients in the overlapping range", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5, color="#444444", style="italic")
    ax.grid(axis="y", visible=False)

    fig.suptitle(
        "Case-mix analysis: discrimination and sensitivity by disease severity "
        "(EXPLORATORY)",
        y=1.03, fontsize=12, fontweight="bold",
    )
    fig.text(0.5, -0.04,
             "The stage variable is used only to define evaluation subgroups; it never "
             "entered any model "
             "as a predictor.",
             ha="center", fontsize=8.5, style="italic", color="#444444")
    save(fig, fig_dir, "fig_r12_spectrum_effect")
    print("  fig_r12_spectrum_effect")

    # Ceiling-aware leakage framing: measure inflation against a baseline that
    # is not already saturated.
    ceiling_rows = []
    for base_cfg in ("full_valid_model", "low_cost_model", "clinical_only_model"):
        b = best_per_config[best_per_config["config"] == base_cfg]
        if b.empty:
            continue
        b = b.iloc[0]
        ceiling_rows.append({
            "baseline_config": base_cfg,
            "baseline_roc_auc": b["roc_auc"],
            "leaky_roc_auc": leaky_auc,
            "absolute_inflation": leaky_auc - b["roc_auc"],
            "headroom_below_ceiling": 1.0 - b["roc_auc"],
            "fraction_of_headroom_closed_by_leakage": (
                (leaky_auc - b["roc_auc"]) / (1.0 - b["roc_auc"])
                if (1.0 - b["roc_auc"]) > 1e-9 else float("nan")
            ),
        })
    pd.DataFrame(ceiling_rows).to_csv(
        tab_dir / "table_23_leakage_ceiling_analysis.csv", index=False, encoding="utf-8"
    )

    print("\nStage 4 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
