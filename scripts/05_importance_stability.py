"""Stage 5: explainability and feature-importance stability for the best valid model.

Model selection rule (fixed in advance, applied to nested-CV results, not to
any quantity computed here):

    Among cells whose feature configuration is valid for clinical
    interpretation and whose model is not the dummy baseline, take the highest
    mean ROC-AUC; break ties on the lower Brier score.

The selected model is then refitted inside each outer training fold exactly as
in stage 3, and importance is computed on the corresponding outer *test* fold.
Using test outcomes for permutation importance is legitimate because it is
reporting, not selection: no hyper-parameter, feature or threshold is chosen
from it.

The low-cost configuration is always analysed as well, because it is the
subject of the primary research question regardless of whether it wins.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.base import clone  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402

from ckd.config import ensure_dirs, load_config  # noqa: E402
from ckd.data.clean import clean_dataset, feature_matrix  # noqa: E402
from ckd.evaluation.importance import (  # noqa: E402
    linear_coefficients,
    permutation_importance_fold,
    shap_importance_fold,
    summarise_stability,
)
from ckd.evaluation.plots import apply_style, config_label, model_label, save  # noqa: E402
from ckd.features.configs import FEATURE_CONFIGS, SPEC_BY_NAME, get_config  # noqa: E402
from ckd.models.nested_cv import _fold_seed  # noqa: E402
from ckd.models.pipeline import assert_pipeline_is_clean, build_pipeline  # noqa: E402
from ckd.models.zoo import get_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", default="cv_predictions")
    p.add_argument("--repeats", type=int, default=None)
    return p.parse_args()


def analyse(X, y, config_name, model_name, calibration, settings, top_k, n_perm):
    """Refit across outer folds and collect per-fold importances."""
    spec = get_model(model_name)
    Xc = X.loc[:, list(get_config(config_name).features)]
    rows: list[dict] = []
    notes: list[str] = []
    fold_id = 0

    for repeat in range(settings["repeats"]):
        outer = StratifiedKFold(
            n_splits=settings["outer_folds"], shuffle=True,
            random_state=_fold_seed(settings["seed"], repeat, salt=1),
        )
        for fold, (tr, te) in enumerate(outer.split(Xc, y)):
            seed = _fold_seed(settings["seed"], repeat, fold, salt=2)
            pipe = build_pipeline(model_name, config_name, seed)
            assert_pipeline_is_clean(pipe, config_name)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if spec.param_grid:
                    inner = StratifiedKFold(
                        n_splits=settings["inner_folds"], shuffle=True, random_state=seed
                    )
                    search = GridSearchCV(pipe, spec.param_grid, scoring="roc_auc",
                                          cv=inner, n_jobs=1, refit=True, error_score=np.nan)
                    search.fit(Xc.iloc[tr], y[tr])
                    best = search.best_estimator_
                else:
                    best = pipe.fit(Xc.iloc[tr], y[tr])

                if calibration == "none":
                    final = best
                else:
                    cal_cv = StratifiedKFold(n_splits=settings["inner_folds"],
                                             shuffle=True, random_state=seed + 7)
                    final = CalibratedClassifierCV(clone(best), method=calibration,
                                                   cv=cal_cv, ensemble=True)
                    final.fit(Xc.iloc[tr], y[tr])

                # --- permutation importance (model-agnostic, on the test fold) ---
                perm = permutation_importance_fold(
                    final, Xc.iloc[te], y[te], seed=seed, n_repeats=n_perm
                )
                for feat, val in perm.items():
                    rows.append({"fold_id": fold_id, "repeat": repeat, "fold": fold,
                                 "method": "permutation", "feature": feat,
                                 "importance": float(val)})

                # --- SHAP and coefficients on the uncalibrated tuned pipeline ---
                shap_vals, reason = shap_importance_fold(best, Xc.iloc[te], model_name)
                if shap_vals is not None:
                    for feat, val in shap_vals.items():
                        rows.append({"fold_id": fold_id, "repeat": repeat, "fold": fold,
                                     "method": "shap", "feature": feat,
                                     "importance": float(val)})
                elif reason and reason not in notes:
                    notes.append(reason)

                coefs = linear_coefficients(best, list(Xc.columns))
                if coefs is not None:
                    for feat, val in coefs.items():
                        rows.append({"fold_id": fold_id, "repeat": repeat, "fold": fold,
                                     "method": "coefficient", "feature": feat,
                                     "importance": float(val)})
                    for feat, val in coefs.items():
                        rows.append({"fold_id": fold_id, "repeat": repeat, "fold": fold,
                                     "method": "coefficient_abs", "feature": feat,
                                     "importance": float(abs(val))})
            fold_id += 1
    return pd.DataFrame(rows), notes


def importance_figure(stab, per_fold, config_name, model_name, top_k, fig_dir, stem, title):
    """Two-panel figure: mean importance with spread, and top-k selection frequency."""
    apply_style()
    method = "permutation"
    s = stab[stab["method"] == method].sort_values("mean_importance", ascending=True)
    if s.empty:
        return
    pf = per_fold[per_fold["method"] == method]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, max(4.2, 0.34 * len(s) + 2.0)),
                             gridspec_kw={"wspace": 0.42, "width_ratios": [1.25, 1]})

    ax = axes[0]
    order = list(s["feature"])
    data = [pf[pf["feature"] == f]["importance"].to_numpy() for f in order]
    bp = ax.boxplot(data, orientation="horizontal", patch_artist=True, widths=0.62,
                    medianprops=dict(color="black", lw=1.4),
                    flierprops=dict(marker=".", markersize=3, alpha=0.5))
    tiers = [SPEC_BY_NAME[f].tier for f in order]
    tier_colours = {"history": "#0072B2", "exam": "#56B4E9", "urine_dip": "#009E73",
                    "urine_micro": "#E69F00", "blood_lab": "#D55E00",
                    "derived_dx": "#B22222", "target": "#000000"}
    for patch, t in zip(bp["boxes"], tiers):
        patch.set_facecolor(tier_colours[t]); patch.set_alpha(0.6)
    ax.set_yticklabels(order, fontsize=8.5)
    ax.axvline(0, color="#555555", ls="--", lw=1.0)
    ax.set_xlabel("Permutation importance (drop in ROC-AUC)")
    ax.set_title(f"A. Importance across {pf['fold_id'].nunique()} outer folds")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=tier_colours[t], alpha=0.6)
               for t in dict.fromkeys(tiers)]
    ax.legend(handles, list(dict.fromkeys(tiers)), fontsize=7.5, loc="lower right",
              title="Tier", title_fontsize=8)

    ax = axes[1]
    freq_col = f"top{top_k}_frequency"
    s2 = s.sort_values(freq_col, ascending=True)
    colours = ["#0072B2" if v >= 0.8 else "#E69F00" if v >= 0.4 else "#CCCCCC"
               for v in s2[freq_col]]
    ax.barh(range(len(s2)), s2[freq_col], color=colours, height=0.7)
    ax.set_yticks(range(len(s2))); ax.set_yticklabels(s2["feature"], fontsize=8.5)
    ax.set_xlim(0, 1.05); ax.set_xlabel(f"Fraction of folds in the top {top_k}")
    w = float(s["kendalls_w_all_features"].iloc[0])
    j = float(s[f"mean_pairwise_jaccard_top{top_k}"].iloc[0])
    ax.set_title(f"B. Top-{top_k} selection frequency\n"
                 f"Kendall's W = {w:.3f} | mean pairwise Jaccard = {j:.3f}", fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.axvline(0.8, color="#555555", ls=":", lw=0.9)

    fig.suptitle(title, y=1.02, fontsize=12, fontweight="bold")
    fig.text(0.5, -0.02,
             "Importance describes how this fitted model uses a column given the others present. "
             "It is not a causal effect.",
             ha="center", fontsize=8.5, style="italic", color="#444444")
    save(fig, fig_dir, stem)


def main() -> int:
    args = parse_args()
    cfg = load_config()
    proc_dir, fig_dir, tab_dir = ensure_dirs(
        cfg["paths"]["processed_dir"], cfg["paths"]["figures_dir"], cfg["paths"]["tables_dir"]
    )
    apply_style()

    top_k = int(cfg["stability"]["top_k"])
    n_perm = int(cfg["stability"]["permutation_repeats"])

    headline = pd.read_csv(tab_dir / "table_12_headline_results.csv")
    valid = headline[
        headline["config"].map(lambda c: FEATURE_CONFIGS[c].valid_for_clinical_interpretation)
        & (headline["model"] != "dummy")
    ]
    best = valid.sort_values(["roc_auc", "brier"], ascending=[False, True]).iloc[0]
    best_low_cost = (
        valid[valid["config"] == "low_cost_model"]
        .sort_values(["roc_auc", "brier"], ascending=[False, True]).iloc[0]
    )

    manifest_path = proc_dir / f"{args.predictions}_manifest.json"
    import json
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    settings = {
        "seed": int(cfg["seed"]),
        "outer_folds": int(cfg["cross_validation"]["outer_folds"]),
        "inner_folds": int(cfg["cross_validation"]["inner_folds"]),
        "repeats": int(args.repeats or manifest["repeats"]),
    }

    result = clean_dataset()
    X = feature_matrix(result)
    y = result.target.to_numpy()

    targets = [
        ("best_valid", best["config"], best["model"], best["calibration"]),
    ]
    if (best["config"], best["model"]) != (best_low_cost["config"], best_low_cost["model"]):
        targets.append(
            ("low_cost", best_low_cost["config"], best_low_cost["model"],
             best_low_cost["calibration"])
        )

    # Not every model family admits an exact SHAP explainer - the RBF SVM in
    # particular does not. When the selection rule picks such a model, the best
    # SHAP-capable valid model is analysed as well, so that the SHAP versus
    # permutation comparison the study promises still exists rather than being
    # quietly dropped.
    from ckd.models.zoo import MODEL_SPECS

    shap_capable = {n for n, s in MODEL_SPECS.items()
                    if s.supports_shap_tree or s.supports_shap_linear}
    if best["model"] not in shap_capable:
        cand = valid[valid["model"].isin(shap_capable)]
        if not cand.empty:
            bs = cand.sort_values(["roc_auc", "brier"], ascending=[False, True]).iloc[0]
            targets.append(
                ("shap_capable", bs["config"], bs["model"], bs["calibration"])
            )
            print(f"\nSelected model ({best['model']}) has no exact SHAP explainer; "
                  f"additionally analysing {bs['config']} / {bs['model']} for SHAP.")

    all_stab, all_notes = [], []
    for tag, config_name, model_name, calibration in targets:
        print(f"\n{tag}: {config_name} / {model_name} / calibration={calibration}")
        print(f"  refitting across {settings['outer_folds'] * settings['repeats']} outer folds ...")
        per_fold, notes = analyse(X, y, config_name, model_name, calibration,
                                  settings, top_k, n_perm)
        stab = summarise_stability(per_fold, top_k=top_k)
        per_fold.insert(0, "target", tag)
        stab.insert(0, "target", tag)
        stab.insert(1, "config", config_name)
        stab.insert(2, "model", model_name)

        per_fold.to_csv(tab_dir / f"table_19_{tag}_importance_per_fold.csv",
                        index=False, encoding="utf-8")
        all_stab.append(stab)
        all_notes.extend(notes)

        methods = sorted(per_fold["method"].unique())
        print(f"  methods computed: {methods}")
        for note in notes:
            print(f"  note: {note}")

        top = stab[stab["method"] == "permutation"].head(top_k)
        print(f"  top {top_k} by permutation importance:")
        for _, r in top.iterrows():
            print(f"    {r['feature']:<16} mean={r['mean_importance']:+.4f} "
                  f"sd={r['sd_importance']:.4f} top{top_k}_freq={r[f'top{top_k}_frequency']:.2f}")

        suffix = {"best_valid": "", "low_cost": "b", "shap_capable": "c"}.get(tag, "")
        importance_figure(
            stab, per_fold, config_name, model_name, top_k, fig_dir,
            f"fig_r10_{tag}_importance_stability",
            f"Figure R10{suffix}. Feature importance and stability - "
            f"{config_label(config_name)} ({model_label(model_name)})",
        )
        print(f"  wrote fig_r10_{tag}_importance_stability")

        # SHAP-vs-permutation agreement, where SHAP was available
        if "shap" in methods:
            piv = stab.pivot_table(index="feature", columns="method", values="mean_importance")
            if {"shap", "permutation"} <= set(piv.columns):
                rho = piv["shap"].corr(piv["permutation"], method="spearman")
                print(f"  Spearman agreement SHAP vs permutation: {rho:.3f}")
                fig, ax = plt.subplots(figsize=(5.6, 5.2))
                ax.scatter(piv["permutation"], piv["shap"], s=45, color="#0072B2",
                           alpha=0.8, edgecolor="white")
                # Label only the features that carry the message. Annotating all
                # 25 makes the crowded low-importance corner unreadable and adds
                # nothing: those points are collectively "unimportant".
                labelled = piv.nlargest(10, "shap")
                for feat, row in labelled.iterrows():
                    ax.annotate(feat, (row["permutation"], row["shap"]), fontsize=8,
                                xytext=(5, 3), textcoords="offset points")
                n_hidden = len(piv) - len(labelled)
                if n_hidden > 0:
                    ax.text(0.02, 0.98,
                            f"{n_hidden} lower-importance features unlabelled",
                            transform=ax.transAxes, ha="left", va="top",
                            fontsize=7.5, style="italic", color="#666666")
                ax.set_xlabel("Mean permutation importance")
                ax.set_ylabel("Mean |SHAP|")
                ax.set_title(f"Figure R11. SHAP vs permutation importance\n"
                             f"{config_label(config_name)} ({model_label(model_name)}), "
                             f"Spearman rho = {rho:.3f}")
                save(fig, fig_dir, f"fig_r11_{tag}_shap_vs_permutation")

    stab_all = pd.concat(all_stab, ignore_index=True)
    stab_all.to_csv(tab_dir / "table_20_importance_stability.csv", index=False, encoding="utf-8")

    with open(tab_dir / "table_20_importance_notes.txt", "w", encoding="utf-8") as fh:
        fh.write("Notes recorded while computing importance\n")
        fh.write("=" * 50 + "\n")
        if all_notes:
            for n in dict.fromkeys(all_notes):
                fh.write(f"- {n}\n")
        else:
            fh.write("- none; all requested explainers ran successfully\n")

    print("\nStage 5 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
