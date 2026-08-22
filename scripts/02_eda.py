"""Stage 2: exploratory data analysis.

Exploratory by construction. Nothing computed here feeds model selection or
hyper-parameter choice; the whole script looks at the complete dataset, which
is legitimate for description but would be leakage if used for modelling
decisions. Every figure is labelled EXPLORATORY.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ckd.config import ensure_dirs, load_config  # noqa: E402
from ckd.data.clean import clean_dataset, feature_matrix  # noqa: E402
from ckd.data.quality import cramers_v  # noqa: E402
from ckd.evaluation.plots import (  # noqa: E402
    OKABE_ITO,
    apply_style,
    config_colour,
    config_label,
    save,
)
from ckd.features.configs import FEATURE_CONFIGS, SPEC_BY_NAME, VARIABLES  # noqa: E402

TIER_COLOURS = {
    "history": "#0072B2",
    "exam": "#56B4E9",
    "urine_dip": "#009E73",
    "urine_micro": "#E69F00",
    "blood_lab": "#D55E00",
    "derived_dx": "#B22222",
    "target": "#000000",
}


def main() -> int:
    cfg = load_config()
    fig_dir, tab_dir = ensure_dirs(cfg["paths"]["figures_dir"], cfg["paths"]["tables_dir"])
    apply_style()

    result = clean_dataset()
    clean = result.clean
    X = feature_matrix(result)
    y = result.target.to_numpy()

    print("Exploratory data analysis (all figures labelled EXPLORATORY)")

    # ---------------------------------------------------------------
    # Figure 1: cohort composition
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

    ax = axes[0]
    counts = clean["class"].value_counts()
    bars = ax.bar(
        [str(i) for i in counts.index],
        counts.to_numpy(),
        color=[OKABE_ITO[1], OKABE_ITO[0]],
        width=0.6,
    )
    for b, v in zip(bars, counts.to_numpy()):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v}\n({v/len(clean):.1%})",
                ha="center", va="bottom", fontsize=9)
    ax.set_title("A. Outcome distribution")
    ax.set_ylabel("Patients")
    ax.set_ylim(0, max(counts) * 1.25)

    ax = axes[1]
    order = [b["label"] for b in result.bin_map["age"]]
    ct = pd.crosstab(clean["age"], clean["class"]).reindex(order).fillna(0)
    bottom = np.zeros(len(ct))
    for i, cls in enumerate(["notckd", "ckd"]):
        if cls in ct.columns:
            ax.barh(range(len(ct)), ct[cls].to_numpy(), left=bottom,
                    color=OKABE_ITO[1] if cls == "notckd" else OKABE_ITO[0], label=cls)
            bottom += ct[cls].to_numpy()
    ax.set_yticks(range(len(ct)))
    ax.set_yticklabels(ct.index, fontsize=8)
    ax.set_title("B. Age band by outcome")
    ax.set_xlabel("Patients")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)

    ax = axes[2]
    tier_counts: dict[str, int] = {}
    for v in VARIABLES:
        if v.name == "class":
            continue
        tier_counts[v.tier] = tier_counts.get(v.tier, 0) + 1
    tiers = list(tier_counts)
    ax.bar(range(len(tiers)), [tier_counts[t] for t in tiers],
           color=[TIER_COLOURS[t] for t in tiers], width=0.65)
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels([t.replace("_", "\n") for t in tiers], fontsize=8)
    ax.set_title("C. Variables by measurement tier")
    ax.set_ylabel("Variables")

    fig.suptitle("Figure E1. Cohort composition (EXPLORATORY)", y=1.04, fontsize=12, fontweight="bold")
    save(fig, fig_dir, "fig_e1_cohort_composition")
    print("  fig_e1_cohort_composition")

    # ---------------------------------------------------------------
    # Figure 2: association with the outcome, coloured by tier
    # ---------------------------------------------------------------
    assoc = []
    for v in VARIABLES:
        if v.name == "class":
            continue
        val = cramers_v(clean[v.name].astype(str), clean["class"].astype(str))
        assoc.append({"variable": v.name, "tier": v.tier, "cramers_v": val})
    adf = pd.DataFrame(assoc).sort_values("cramers_v", ascending=True)
    adf.to_csv(tab_dir / "table_03_univariate_association.csv", index=False, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.2, 8.2))
    colours = [TIER_COLOURS[t] for t in adf["tier"]]
    ax.barh(range(len(adf)), adf["cramers_v"], color=colours, height=0.72)
    ax.set_yticks(range(len(adf)))
    ax.set_yticklabels(adf["variable"], fontsize=9)
    ax.set_xlabel("Cramer's V with outcome (bias-corrected)")
    ax.set_title("Figure E2. Univariate association with CKD status (EXPLORATORY)")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in TIER_COLOURS.values()]
    ax.legend(handles, list(TIER_COLOURS), title="Measurement tier",
              loc="lower right", fontsize=8, borderaxespad=0.8)
    fig.text(0.5, -0.015,
             "Association is not predictive importance and is not causal. "
             "Variables shown in red/black are prohibited from all valid models.",
             ha="center", fontsize=8.5, style="italic", color="#444444")
    save(fig, fig_dir, "fig_e2_univariate_association")
    print("  fig_e2_univariate_association")

    # ---------------------------------------------------------------
    # Figure 3: leakage structure
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), gridspec_kw={"wspace": 0.45})

    ax = axes[0]
    ct = pd.crosstab(clean["class"], clean["affected"])
    im = ax.imshow(ct.to_numpy(), cmap="Reds", aspect="auto")
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            ax.text(j, i, int(ct.iloc[i, j]), ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if ct.iloc[i, j] > ct.to_numpy().max() / 2 else "black")
    ax.set_xticks(range(ct.shape[1])); ax.set_xticklabels(ct.columns)
    ax.set_yticks(range(ct.shape[0])); ax.set_yticklabels(ct.index)
    ax.set_xlabel("affected"); ax.set_ylabel("class")
    ax.set_title("A. `affected` is an exact copy\nof the outcome")
    ax.grid(False)

    ax = axes[1]
    ct = pd.crosstab(clean["stage"], clean["class"])
    prop = (ct.get("ckd", 0) / ct.sum(axis=1)).fillna(0)
    ax.bar(range(len(prop)), prop.to_numpy(), color="#B22222", width=0.65)
    for i, (n, p) in enumerate(zip(ct.sum(axis=1), prop)):
        ax.text(i, p + 0.02, f"{p:.0%}\nn={n}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(prop))); ax.set_xticklabels(prop.index)
    ax.set_ylim(0, 1.30)
    ax.set_ylabel("Proportion CKD"); ax.set_xlabel("stage")
    ax.set_title("B. `stage` nearly determines\nthe outcome")

    ax = axes[2]
    grf_order = [b["label"] for b in result.bin_map["grf"] if str(b["label"]) != "nan"]
    ct = pd.crosstab(clean["grf"], clean["class"]).reindex(grf_order).fillna(0)
    prop = (ct.get("ckd", 0) / ct.sum(axis=1).replace(0, np.nan))
    ax.barh(range(len(prop)), prop.to_numpy(), color="#B22222", height=0.72)
    ax.set_yticks(range(len(prop)))
    ax.set_yticklabels([str(s) for s in prop.index], fontsize=7.5)
    ax.set_xlabel("Proportion CKD"); ax.set_xlim(0, 1.05)
    ax.set_title("C. `grf` (eGFR) band vs outcome")
    ax.grid(axis="y", visible=False)

    fig.suptitle(
        "Figure E3. Why `affected`, `stage` and `grf` are prohibited (EXPLORATORY)",
        y=1.04, fontsize=12, fontweight="bold", color="#B22222",
    )
    save(fig, fig_dir, "fig_e3_leakage_structure")
    print("  fig_e3_leakage_structure")

    # ---------------------------------------------------------------
    # Figure 4: correlation structure among valid predictors
    # ---------------------------------------------------------------
    valid_feats = list(FEATURE_CONFIGS["full_valid_model"].features)
    corr = X[valid_feats].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9.2, 8.0))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(valid_feats)))
    ax.set_xticklabels(valid_feats, rotation=90, fontsize=8)
    ax.set_yticks(range(len(valid_feats)))
    ax.set_yticklabels(valid_feats, fontsize=8)
    ax.set_title("Figure E4. Spearman correlation among valid predictors (EXPLORATORY)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Spearman rho")
    save(fig, fig_dir, "fig_e4_predictor_correlation")
    print("  fig_e4_predictor_correlation")

    hi = []
    for i, a in enumerate(valid_feats):
        for b in valid_feats[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) >= 0.6:
                hi.append({"variable_1": a, "variable_2": b, "spearman_rho": round(float(r), 3)})
    pd.DataFrame(hi).sort_values("spearman_rho", key=abs, ascending=False).to_csv(
        tab_dir / "table_04_collinear_pairs.csv", index=False, encoding="utf-8"
    )

    # ---------------------------------------------------------------
    # Figure 5: feature-configuration composition
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    names = list(FEATURE_CONFIGS)
    tiers = ["history", "exam", "urine_dip", "urine_micro", "blood_lab", "derived_dx", "target"]
    bottom = np.zeros(len(names))
    for tier in tiers:
        vals = np.array([
            sum(1 for f in FEATURE_CONFIGS[n].features if SPEC_BY_NAME[f].tier == tier)
            for n in names
        ], dtype=float)
        if vals.sum() == 0:
            continue
        ax.bar(range(len(names)), vals, bottom=bottom, label=tier, color=TIER_COLOURS[tier], width=0.62)
        bottom += vals
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([config_label(n) for n in names], rotation=18, ha="right", fontsize=8.5)
    ax.set_ylabel("Number of predictors")
    ax.set_title("Figure E5. Composition of each feature configuration")
    ax.legend(title="Tier", ncols=4, fontsize=8, loc="upper right")
    for i, n in enumerate(names):
        if not FEATURE_CONFIGS[n].valid_for_clinical_interpretation:
            ax.text(i, bottom[i] + 0.4, "INVALID", ha="center", color="#B22222",
                    fontsize=8, fontweight="bold")
    save(fig, fig_dir, "fig_e5_configuration_composition")
    print("  fig_e5_configuration_composition")

    # ---------------------------------------------------------------
    # Table: per-variable distribution by outcome
    # ---------------------------------------------------------------
    rows = []
    for v in VARIABLES:
        if v.name == "class":
            continue
        for level, grp in clean.groupby(v.name, dropna=True):
            n_ckd = int((grp["class"] == "ckd").sum())
            n_tot = int(len(grp))
            rows.append({
                "variable": v.name,
                "tier": v.tier,
                "level": str(level),
                "n": n_tot,
                "n_ckd": n_ckd,
                "n_notckd": n_tot - n_ckd,
                "proportion_ckd": round(n_ckd / n_tot, 4),
            })
    pd.DataFrame(rows).to_csv(
        tab_dir / "table_05_level_distribution_by_outcome.csv", index=False, encoding="utf-8"
    )

    # Missingness / near-constant summary
    summary = []
    for v in VARIABLES:
        if v.name == "class":
            continue
        col = clean[v.name]
        vc = col.value_counts(normalize=True, dropna=True)
        summary.append({
            "variable": v.name,
            "tier": v.tier,
            "n_missing": int(col.isna().sum()),
            "n_distinct": int(col.nunique(dropna=True)),
            "modal_proportion": round(float(vc.iloc[0]), 4) if len(vc) else None,
            "near_constant_90pct": bool(len(vc) and vc.iloc[0] >= 0.90),
            "cramers_v_outcome": round(
                cramers_v(col.astype(str), clean["class"].astype(str)), 4
            ),
        })
    pd.DataFrame(summary).to_csv(
        tab_dir / "table_06_variable_summary.csv", index=False, encoding="utf-8"
    )

    print(f"\nWrote figures to {fig_dir} and tables to {tab_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
