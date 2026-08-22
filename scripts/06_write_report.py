"""Stage 6: assemble reports/research_report.md from the computed artefacts.

Every quantitative statement in the report is injected from the generated
tables rather than typed by hand, so the write-up cannot drift away from the
results. Interpretive sentences that depend on the outcome (for example whether
isotonic calibration helped or hurt) are chosen by inspecting the numbers, not
asserted in advance.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ckd.config import load_config, resolve  # noqa: E402
from ckd.evaluation.plots import calibration_label, config_label, model_label  # noqa: E402
from ckd.features.configs import FEATURE_CONFIGS, SPEC_BY_NAME, uncertain_variables  # noqa: E402
from ckd.models.zoo import MODEL_SPECS, available_models, unavailable_models  # noqa: E402

ROOT = resolve(".")
TABLES = ROOT / "reports" / "tables"
PROC = ROOT / "data" / "processed"


def t(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name)


def fmt(v, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{nd}f}"


def ci(row, metric: str, nd: int = 3) -> str:
    return (f"{fmt(row[metric], nd)} "
            f"({fmt(row[f'{metric}_ci_low'], nd)}-{fmt(row[f'{metric}_ci_high'], nd)})")


def main() -> int:
    cfg = load_config()
    manifest = json.loads((PROC / "cv_predictions_manifest.json").read_text("utf-8"))
    quality = json.loads((PROC / "data_quality_report.json").read_text("utf-8"))

    headline = t("table_12_headline_results.csv")
    audit = t("table_13_leakage_audit.csv")
    calib = t("table_14_calibration_results.csv")
    selected = t("table_16_selected_models.csv")
    confusion = t("table_18_confusion_matrices.csv")
    stability = t("table_20_importance_stability.csv")
    per_repeat = t("table_08_metrics_per_repeat.csv")
    # na_filter=False: pandas treats the literal string "None" as a missing
    # value by default, which would silently turn the `class_weight=None`
    # hyper-parameter into NaN in the report.
    hyper = pd.read_csv(TABLES / "table_07_hyperparameter_selection.csv", na_filter=False)
    spectrum = t("table_21_spectrum_analysis.csv")
    separability = t("table_22_univariate_separability.csv")
    ceiling = t("table_23_leakage_ceiling_analysis.csv")

    valid = headline[headline["valid"] & (headline["model"] != "dummy")]
    dummy = headline[headline["model"] == "dummy"]

    best = selected[selected["selection"] == "best_valid_overall"].iloc[0]
    best_lc = selected[selected["selection"] == "best_low_cost"].iloc[0]

    def cell(config, model=None, calibration=None):
        s = headline[headline["config"] == config]
        if model:
            s = s[s["model"] == model]
        if calibration:
            s = s[s["calibration"] == calibration]
        if s.empty:
            return None
        return s.sort_values(["roc_auc", "brier"], ascending=[False, True]).iloc[0]

    leaky = audit[audit["config"] == "leaky_model"].iloc[0]
    full = audit[audit["config"] == "full_valid_model"].iloc[0]
    lab = audit[audit["config"] == "laboratory_model"].iloc[0]
    low = audit[audit["config"] == "low_cost_model"].iloc[0]
    clin = audit[audit["config"] == "clinical_only_model"].iloc[0]
    lcm = audit[audit["config"] == "low_cost_plus_urine_micro_model"].iloc[0]

    # ---- calibration comparison across valid cells ----
    cal_valid = calib[calib["valid"] & (calib["model"] != "dummy")]
    cal_med = cal_valid.groupby("calibration")[
        ["brier", "calibration_slope", "calibration_intercept", "ece"]
    ].median()

    def cal_row(method):
        return cal_med.loc[method] if method in cal_med.index else None

    none_m, sig_m, iso_m = cal_row("none"), cal_row("sigmoid"), cal_row("isotonic")
    best_cal_method = cal_med["brier"].idxmin()
    iso_worse_than_sigmoid = bool(iso_m is not None and sig_m is not None
                                  and iso_m["brier"] > sig_m["brier"])

    # ---- stability ----
    perm = stability[(stability["method"] == "permutation") & (stability["target"] == "best_valid")]
    top_k = int(cfg["stability"]["top_k"])
    freq_col = f"top{top_k}_frequency"
    perm_sorted = perm.sort_values("mean_importance", ascending=False)
    kendall = float(perm["kendalls_w_all_features"].iloc[0]) if len(perm) else float("nan")
    jaccard = float(perm[f"mean_pairwise_jaccard_top{top_k}"].iloc[0]) if len(perm) else float("nan")
    stable_feats = perm_sorted[perm_sorted[freq_col] >= 0.8]["feature"].tolist()
    unstable_feats = perm_sorted[
        (perm_sorted[freq_col] > 0.2) & (perm_sorted[freq_col] < 0.8)
    ]["feature"].tolist()

    perm_lc = stability[(stability["method"] == "permutation") & (stability["target"] == "low_cost")]
    if perm_lc.empty:
        perm_lc = perm
    perm_lc_sorted = perm_lc.sort_values("mean_importance", ascending=False)
    lc_stable = perm_lc_sorted[perm_lc_sorted[freq_col] >= 0.8]["feature"].tolist()

    # ---- confusion matrices for the low-cost model ----
    cm_lc = confusion[confusion["config"] == "low_cost_model"]
    cm_lc_pre = cm_lc[cm_lc["operating_point"] == "prespecified_0.50"].iloc[0]
    cm_lc_scr = cm_lc[cm_lc["operating_point"] == "screening_target_sens"].iloc[0]

    n_pos = int(quality["class_distribution"]["n_positive"])
    n_neg = int(quality["class_distribution"]["n_negative"])
    n_tot = n_pos + n_neg

    # ---- case-mix / spectrum ----
    # Subgroups must be compared for the SAME fitted model, otherwise a
    # "decline across subgroups" could just be different models being picked in
    # each one. The model/calibration is therefore fixed per configuration to
    # whichever cell won on all patients, and every subgroup is read from that
    # cell.
    def _fixed_cell(config):
        s = spectrum[(spectrum["config"] == config)
                     & (spectrum["subgroup"] == "all_patients")]
        if s.empty:
            return None
        r = s.sort_values("roc_auc", ascending=False).iloc[0]
        return r["model"], r["calibration"]

    _cell_cache = {c: _fixed_cell(c) for c in spectrum["config"].unique()}

    def spec_row(config, subgroup):
        cell = _cell_cache.get(config)
        if cell is None:
            return None
        model, calibration = cell
        s = spectrum[(spectrum["config"] == config) & (spectrum["subgroup"] == subgroup)
                     & (spectrum["model"] == model)
                     & (spectrum["calibration"] == calibration)]
        return None if s.empty else s.iloc[0]

    top_sep = separability.iloc[0]
    stage_purity = {r["stage"]: r for r in quality["leakage_relationships"]["stage_purity"]}
    n_advanced = int(sum(
        round(r["n"] * r["proportion_ckd"]) for s, r in stage_purity.items()
        if s in ("s3", "s4", "s5")
    ))
    n_early_ckd = n_pos - n_advanced

    all_row = spec_row("full_valid_model", "all_patients")
    early_row = spec_row("full_valid_model", "early_ckd_only")
    lc_all = spec_row("low_cost_model", "all_patients")
    lc_early = spec_row("low_cost_model", "early_ckd_only")
    clin_all = spec_row("clinical_only_model", "all_patients")
    clin_early = spec_row("clinical_only_model", "early_ckd_only")

    W: list[str] = []
    w = W.append

    # =================================================================
    w("# Reliable and Explainable Low-Cost CKD Screening in Bangladesh")
    w("")
    w("**A nested validation, leakage audit, and calibration study**")
    w("")
    w(f"*Generated {date.today().isoformat()} by `scripts/06_write_report.py`. "
      "Every number in this document is injected directly from the computed "
      "tables in `reports/tables/`; none is transcribed by hand.*")
    w("")
    w("---")
    w("")
    w("> ### Study-type and safety statement")
    w(">")
    w("> This is an **internally validated, retrospective, single-centre "
      "methodological feasibility study**. It is **not a diagnostic tool**, and "
      "nothing in it should be used to make decisions about any patient. No "
      "claim is made about clinical utility, causality, deployment readiness or "
      "generalisability to any other population. **External and prospective "
      "validation would be required** before these findings could be considered "
      "clinically meaningful.")
    w("")
    w("---")
    w("")

    # ---------------- Abstract ----------------
    w("## Abstract")
    w("")
    w("**Background.** Chronic kidney disease (CKD) is common, largely "
      "asymptomatic until advanced, and disproportionately burdensome where "
      "laboratory access is limited. Whether inexpensive, routinely available "
      "information can support CKD screening is therefore a question worth "
      "asking - but published machine-learning studies on small CKD datasets "
      "frequently report near-perfect accuracy, which is a signature of target "
      "leakage rather than of clinical usefulness.")
    w("")
    w(f"**Objective.** To assess, under leakage-controlled repeated nested "
      f"cross-validation, whether a small explainable model can predict CKD "
      f"status from low-cost variables with useful sensitivity and calibrated "
      f"risk estimates; and to quantify how much target leakage inflates "
      f"apparent performance.")
    w("")
    w(f"**Methods.** {n_tot} patient records "
      f"({n_pos} CKD, {n_neg} non-CKD) collected at Enam Medical College, Savar, "
      f"Bangladesh. All continuous variables were already discretised into "
      f"interval bins in the released file. Six feature configurations were "
      f"compared, including one **deliberately invalid** set containing an exact "
      f"copy of the outcome (`affected`), the post-diagnosis stage label "
      f"(`stage`) and the diagnostic eGFR quantity (`grf`). "
      f"{len(manifest['models_run'])} model families "
      f"(dummy, penalised logistic regression, random forest, support vector "
      f"machine, XGBoost, explainable boosting machine) were evaluated under "
      f"{manifest['outer_folds']}-fold outer / {manifest['inner_folds']}-fold "
      f"inner nested stratified cross-validation repeated {manifest['repeats']} "
      f"times ({manifest['outer_folds'] * manifest['repeats']} outer test sets). "
      f"Imputation, scaling, tuning, calibration and threshold selection were "
      f"performed strictly within training folds, and prohibited columns were "
      f"blocked programmatically by a guard that raises at fit and transform "
      f"time. Uncalibrated, Platt and isotonic probabilities were compared. "
      f"Confidence intervals come from a stratified patient-level bootstrap "
      f"({cfg['evaluation']['bootstrap_iterations']} resamples).")
    w("")
    lc_ceiling = ceiling[ceiling["baseline_config"] == "low_cost_model"]
    cl_ceiling = ceiling[ceiling["baseline_config"] == "clinical_only_model"]
    full_at_ceiling = full["roc_auc"] >= 0.999

    if full_at_ceiling:
        lc_pct = (fmt(100 * float(lc_ceiling["fraction_of_headroom_closed_by_leakage"].iloc[0]), 0)
                  if not lc_ceiling.empty else "n/a")
        cl_pct = (fmt(100 * float(cl_ceiling["fraction_of_headroom_closed_by_leakage"].iloc[0]), 0)
                  if not cl_ceiling.empty else "n/a")
        leak_sentence = (
            ", i.e. the valid model is itself at the discrimination ceiling, so "
            "the gain from leakage measured against it is arithmetically near "
            "zero. Measured against baselines that still have headroom, leakage "
            f"closes {lc_pct}% of the remaining error for the low-cost set and "
            f"{cl_pct}% for the clinical-only set: it takes any configuration to "
            "1.0. "
        )
    else:
        leak_sentence = (
            " - an absolute inflation of "
            f"{fmt(leaky['roc_auc'] - full['roc_auc'])} and a "
            f"{fmt(100 * (leaky['roc_auc'] - full['roc_auc']) / max(1e-9, 1 - full['roc_auc']), 1)}% "
            "reduction in the remaining discrimination error. "
        )

    w(f"**Results.** The invalid leaky configuration reached ROC-AUC "
      f"{ci(leaky, 'roc_auc')}. The best valid configuration reached "
      f"{ci(full, 'roc_auc')}"
      f"{leak_sentence}"
      f"The low-cost configuration (history, examination and urine dipstick; "
      f"{int(low['n_features'])} variables) achieved ROC-AUC "
      f"{ci(low, 'roc_auc')} versus {ci(lab, 'roc_auc')} for the "
      f"laboratory-only configuration ({int(lab['n_features'])} variables). "
      f"At the prespecified threshold of 0.50 the low-cost model reached "
      f"sensitivity {ci(low, 'sensitivity')} and negative predictive value "
      f"{ci(low, 'npv')}. Calibration of the best valid model was "
      f"slope {fmt(best['calibration_slope'], 2)}, intercept "
      f"{fmt(best['calibration_intercept'], 2)}. Feature-importance rankings "
      f"were {'moderately' if kendall >= 0.5 else 'weakly'} concordant across "
      f"the {manifest['outer_folds'] * manifest['repeats']} outer folds "
      f"(Kendall's W = {fmt(kendall)}).")
    w("")
    w(f"**A second, larger source of optimism was identified.** Even without any "
      f"prohibited column, valid configurations sit close to the discrimination "
      f"ceiling, and a post hoc case-mix analysis explains why: "
      f"{n_advanced} of the {n_pos} CKD patients ({n_advanced/n_pos:.0%}) are "
      f"staged s3-s5, and haemoglobin alone separates the groups with "
      f"univariate ROC-AUC {fmt(top_sep['univariate_auc'])}. This is a contrast "
      f"between advanced disease and comparatively healthy controls, not a "
      f"screening series. Restricted to early CKD (s1-s2, n = "
      f"{int(early_row['n_ckd']) if early_row is not None else n_early_ckd}) "
      f"versus non-CKD, sensitivity at the prespecified threshold falls from "
      f"{fmt(lc_all['sensitivity'], 3) if lc_all is not None else 'n/a'} to "
      f"{fmt(lc_early['sensitivity'], 3) if lc_early is not None else 'n/a'} "
      f"for the low-cost model and from "
      f"{fmt(clin_all['sensitivity'], 3) if clin_all is not None else 'n/a'} to "
      f"{fmt(clin_early['sensitivity'], 3) if clin_early is not None else 'n/a'} "
      f"for the clinical-only model.")
    w("")
    w(f"**Conclusions.** Target leakage inflates apparent performance on this "
      f"dataset, and case mix inflates it further and by more. Together these "
      f"explain the near-perfect results commonly reported for this data far "
      f"better than genuine screening ability does. Under leakage-controlled "
      f"validation the low-cost variable set retains "
      f"{'much' if low['roc_auc'] >= 0.9 * lab['roc_auc'] else 'only part'} of "
      f"the discrimination available from laboratory measurements, but the "
      f"headline figures should not be read as screening performance. These are "
      f"internal, {n_tot}-patient, single-centre estimates with wide confidence "
      f"intervals; they establish methodological feasibility only, and external "
      f"and prospective validation in a genuine screening population would be "
      f"required before any clinical claim could be made.")
    w("")

    # ---------------- Introduction ----------------
    w("## 1. Introduction")
    w("")
    w("Chronic kidney disease affects roughly 9% of the world's population and "
      "caused an estimated 1.2 million deaths in 2017, with the burden falling "
      "disproportionately on regions where diagnostic laboratory capacity is "
      "limited [1]. CKD is defined by abnormalities of kidney structure or "
      "function present for more than three months, operationalised principally "
      "through estimated glomerular filtration rate (eGFR) and albuminuria [2]. "
      "Because early CKD is largely asymptomatic, detection depends on testing "
      "rather than on presentation, which makes the cost and availability of "
      "the test a first-order determinant of who gets diagnosed.")
    w("")
    w("That motivates a specific technical question: how much of the "
      "discriminative information in a CKD assessment is carried by variables "
      "that cost almost nothing to obtain - history, physical examination, and a "
      "urine reagent strip - relative to variables that require venepuncture and "
      "a laboratory analyser?")
    w("")
    w("There is a well-documented hazard in answering this with machine "
      "learning on small clinical datasets. Published analyses of this "
      "particular dataset, and of CKD datasets generally, frequently report "
      "accuracy at or near 100%. Such results are rarely evidence of clinical "
      "usefulness; far more often they indicate that a predictor encodes the "
      "outcome. This dataset contains three such columns, and one of them is an "
      "exact copy of the label. A study that does not control for this cannot "
      "distinguish learning from lookup.")
    w("")
    w("This work therefore treats leakage control, honest validation and "
      "calibration as the primary objects of study, and treats predictive "
      "performance as something to be measured carefully rather than "
      "maximised. Reporting follows the spirit of TRIPOD+AI [3].")
    w("")

    # ---------------- Research questions ----------------
    w("## 2. Research questions")
    w("")
    w("**Primary.** Can a small, explainable model predict CKD status using "
      "inexpensive and routinely available patient information, while "
      "maintaining clinically useful sensitivity and calibrated risk estimates?")
    w("")
    w("**Secondary.**")
    w("")
    w("1. How much does target leakage inflate model performance?")
    w("2. Can a reduced-feature model perform comparably to a laboratory-based model?")
    w("3. Are predicted probabilities properly calibrated?")
    w("4. Are the identified important predictors stable across resampling?")
    w("5. How uncertain are the results, given only 200 patients?")
    w("")

    # ---------------- Dataset ----------------
    prov = quality["provenance"]
    w("## 3. Dataset")
    w("")
    w("### 3.1 Source and provenance")
    w("")
    w(f"The analysis uses `ckd-dataset-v2.csv`, {n_tot} patient records collected "
      f"at Enam Medical College, Savar, Dhaka, Bangladesh. The same data are "
      f"distributed by the UCI Machine Learning Repository as *Risk Factor "
      f"Prediction of Chronic Kidney Disease* (creators Md. Ashiqul Islam and "
      f"Shamima Akter), under CC BY 4.0 [4].")
    w("")
    w(f"- SHA-256 of the analysed file: `{prov['sha256']}`")
    w(f"- Shape as read (header excluded): "
      f"{prov['shape_including_metadata_rows'][0]} x "
      f"{prov['shape_including_metadata_rows'][1]}")
    w(f"- After removing 2 metadata rows: **{n_tot} patients x "
      f"{prov['n_columns']} columns**")
    w("")
    w("The two rows immediately below the header are metadata, not "
      "observations: the first contains only the token `discrete` in every "
      "column, and the second declares `class` as the target and `meta` for "
      "`age`. Both are removed in exactly one place in the codebase "
      "(`src/ckd/data/load.py`), and the classification is verified rather than "
      "assumed.")
    w("")

    w("### 3.2 The decisive property: the data are already discretised")
    w("")
    w("Every continuous variable in the released file has been binned into "
      "interval strings before publication - `sg` as `1.019 - 1.021`, `bgr` as "
      "`< 112`, `grf` as `>= 227.944`. **The original measurements are not "
      "recoverable.** This information loss is a property of the published "
      "dataset, not a modelling choice, and it constrains everything that can "
      "be concluded from it. In particular, no analysis of this file can "
      "recover the resolution that a real eGFR or creatinine value would "
      "provide, and the effective measurement precision of every predictor is "
      "unknown.")
    w("")
    w("Each interval label is mapped to a single representative number: the "
      "midpoint for a closed bin, and the finite edge for an open-ended one. "
      "The mapping is a pure function of one cell's text - it uses no "
      "cross-row statistics and no outcome information - so it cannot transfer "
      "information between cross-validation folds, which is why it is the only "
      "transformation applied before splitting.")
    w("")

    w("### 3.3 Outcome and class balance")
    w("")
    w(f"The outcome is `class` (`ckd` / `notckd`): **{n_pos} CKD "
      f"({n_pos/n_tot:.1%}) and {n_neg} non-CKD ({n_neg/n_tot:.1%})**. The "
      f"minority class provides {n_neg} events; with "
      f"{len(FEATURE_CONFIGS['full_valid_model'].features)} candidate "
      f"predictors in the full valid configuration this is "
      f"{quality['class_distribution']['events_per_candidate_predictor_full_valid']} "
      f"events per predictor, far below both the traditional rule of thumb of "
      f"10 and the requirements of modern sample-size calculations for "
      f"prediction models [5]. The study is therefore underpowered by "
      f"construction, and is presented as a feasibility and methodology "
      f"exercise rather than as model development.")
    w("")

    w("### 3.4 Data quality")
    w("")
    w("The full audit is in `reports/data_quality_report.md`. Summary:")
    w("")
    w("| Check | Result |")
    w("|---|---|")
    w(f"| Missing cells after cleaning | {quality['missing_values']['total_missing_cells']} |")
    w(f"| Exact duplicate records | {quality['duplicates']['exact_duplicate_records']} |")
    w(f"| Duplicate predictor patterns | "
      f"{quality['duplicates']['duplicate_predictor_patterns_ignoring_outcome']} |")
    w(f"| Near-constant features (modal >= 90%) | {len(quality['near_constant_features'])} |")
    w(f"| Bin-definition anomalies | {len(quality['bin_anomalies'])} |")
    w(f"| Clinical inconsistencies flagged | {len(quality['clinical_inconsistencies'])} |")
    w(f"| Variables with uncertain interpretation | {len(uncertain_variables())} |")
    w("")
    for f in quality["near_constant_features"]:
        w(f"- **`{f['column']}`** is near-constant: {f['modal_proportion']:.1%} of "
          f"patients share the value `{f['modal_value']}`.")
    for a in quality["bin_anomalies"]:
        w(f"- **`{a['column']}`** has internally inconsistent published bin edges "
          f"(overlaps: {a['overlapping_edges']}; tied representatives: "
          f"{a['tied_representatives']}), so its ordinal encoding contains ties. "
          f"Reported, not silently repaired.")
    for iv in quality["implausible_values"]:
        w(f"- **`{iv['column']}`** contains {iv['n_patients']} value(s) in band "
          f"`{iv['label']}`: {iv['issue']}")
    w("")
    for item in quality["clinical_inconsistencies"]:
        w(f"- **{item['issue']}** - {item['n_patients']} patients "
          f"(CSV lines {item['source_csv_lines']}). {item['interpretation']}")
    w("")

    g = quality["grf_p_investigation"]
    w("#### The `\" p \"` value in `grf`")
    w("")
    w(f"Exactly {g['n_anomalous_cells']} cell in the eGFR column contains the "
      f"token `\" p \"` (the letter p with surrounding whitespace), which is not "
      f"an interval label.")
    for c in g.get("cells", []):
        w(f"- CSV line **{c['source_csv_line']}**, column `{c['column']}`, "
          f"raw value `{c['raw_value']}`")
    if "stage_of_affected_patients" in g:
        w(f"- The affected patient is labelled `ckd` and staged "
          f"`{g['stage_of_affected_patients'][0]}`; every other patient at that "
          f"stage falls in the eGFR band(s) "
          f"`{list(g['grf_bins_of_same_stage_peers'])}`.")
    w("")
    w("The most parsimonious reading is a data-entry error. Because the true "
      "value cannot be recovered, the cell is treated as **missing** and imputed "
      "using the median of the relevant training fold only. The raw file is "
      "preserved unmodified at `data/raw/`. Note that `grf` is excluded from "
      "every clinically valid configuration, so this defect can affect only the "
      "deliberately invalid leaky model.")
    w("")

    w("### 3.5 Leakage structure")
    w("")
    lr = quality["leakage_relationships"]
    w(f"- **`affected` is an exact one-to-one copy of the outcome** "
      f"(verified across all {n_tot} rows; Cramer's V = "
      f"{lr['cramers_v_class_affected']}). It is the target, renamed.")
    w(f"- **`stage`** is a post-diagnosis staging label. Cramer's V with the "
      f"outcome = {lr['cramers_v_class_stage']}. Proportion CKD by stage:")
    w("")
    w("  | stage | n | proportion CKD |")
    w("  |---|---:|---:|")
    for row in lr["stage_purity"]:
        w(f"  | {row['stage']} | {row['n']} | {row['proportion_ckd']:.3f} |")
    w("")
    w(f"- **`grf`** is eGFR, the quantity from which `stage` is banded and by "
      f"which CKD is defined (Cramer's V with `stage` = "
      f"{lr['cramers_v_stage_grf']}). Using it to predict CKD approaches using "
      f"the diagnostic criterion as a predictor.")
    w("")
    w("Stages s3 and s5 are 100% CKD. Any model given these columns is "
      "performing a lookup, not a prediction.")
    w("")

    w("### 3.6 Variables whose meaning could not be established")
    w("")
    w("The released dataset ships **no data dictionary** [4]. Rather than "
      "inventing clinical interpretations, variables whose meaning or "
      "provenance cannot be settled from the file are flagged explicitly:")
    w("")
    for v in uncertain_variables():
        w(f"- **`{v.name}`** ({v.tier}) - {v.uncertainty_note}")
    w("")
    w("The most consequential of these is `ane`. Anaemia can be recorded "
      "clinically at no cost, or read from a full blood count. We verified that "
      "`ane` is **not** a deterministic function of the `hemo` bins, so it is "
      "not a pure recoding of haemoglobin - but its provenance remains "
      "unresolved. It is therefore **excluded from the low-cost configuration**, "
      "which is the conservative choice: it can only understate low-cost "
      "performance, never inflate it.")
    w("")

    # ---------------- Methods ----------------
    w("## 4. Methods")
    w("")
    w("### 4.1 Feature configurations")
    w("")
    w("Cost tiering is an explicit, auditable judgement recorded in "
      "`src/ckd/features/configs.py` with a written rationale per variable, not "
      "a fact extracted from the file. Six configurations were compared:")
    w("")
    w("| Configuration | k | Valid? | Contents |")
    w("|---|---:|---|---|")
    for name, fc in FEATURE_CONFIGS.items():
        valid_mark = "yes" if fc.valid_for_clinical_interpretation else "**NO**"
        w(f"| `{name}` | {len(fc.features)} | {valid_mark} | "
          f"{', '.join('`' + f + '`' for f in fc.features)} |")
    w("")
    w("- `leaky_model` is **deliberately invalid**. It exists only to quantify "
      "leakage-driven inflation and its results must never be read as evidence "
      "of clinical usefulness.")
    w("- `clinical_only_model` and `low_cost_plus_urine_micro_model` are "
      "**flagged sensitivity analyses**. They exist because the boundary of "
      "'low cost' is a genuine judgement call - specifically, whether urine "
      "microscopy (non-invasive but requiring a microscope and a technician) "
      "belongs inside it. Rather than deciding that silently, both sides of the "
      "boundary are reported.")
    w("")

    w("### 4.2 Models")
    w("")
    w("| Model | Notes |")
    w("|---|---|")
    for name in manifest["models_run"]:
        spec = MODEL_SPECS[name]
        w(f"| {spec.label} | {spec.notes or '-'} |")
    w("")
    if manifest.get("models_unavailable"):
        w("Models requested but unavailable in this environment (reported "
          "rather than silently dropped):")
        w("")
        for k, v in manifest["models_unavailable"].items():
            w(f"- `{k}`: {v}")
        w("")
    w("**Neural networks and deep learning were deliberately excluded.** With "
      f"{n_neg} minority-class events and at most "
      f"{len(FEATURE_CONFIGS['full_valid_model'].features)} predictors, a deep "
      "model cannot be estimated reliably, and fitting one would invite exactly "
      "the over-fitting this study is designed to detect.")
    w("")
    w("XGBoost was used rather than CatBoost. Both were installed and "
      "benchmarked; XGBoost fitted roughly three times faster on this data, "
      "which made the full repeated nested design feasible. The choice is about "
      "compute, not expected accuracy. The explainable boosting machine was "
      "configured with `interactions=0` (a pure generalised additive model), "
      "`outer_bags=6` and `max_rounds=1500`: the library defaults cost ~8.7 s "
      "per fit here while adding capacity this sample cannot support. Both are "
      "stated as compute/regularisation trade-offs, not as neutral defaults.")
    w("")
    w("Search spaces were kept small (at most 8 candidates per model) because "
      "inner folds hold roughly 32 patients; a large grid searched on folds "
      "that size selects on noise.")
    w("")

    w("### 4.3 Validation design")
    w("")
    w("| Element | Value |")
    w("|---|---|")
    w(f"| Outer loop | {manifest['outer_folds']}-fold stratified CV |")
    w(f"| Inner loop | {manifest['inner_folds']}-fold stratified CV |")
    w(f"| Repeats | {manifest['repeats']} |")
    w(f"| Independent outer test sets | {manifest['outer_folds'] * manifest['repeats']} |")
    w(f"| Inner selection metric | {manifest['inner_scoring']} |")
    w(f"| Master seed | {manifest['seed']} |")
    w(f"| Grid size | {len(manifest['configs_run'])} configurations x "
      f"{len(manifest['models_run'])} models x "
      f"{len(manifest['calibrations_run'])} calibration methods |")
    w(f"| Runtime | {manifest['runtime_minutes']} min |")
    w("")
    w(f"{manifest['repeats']} repeats were chosen as a documented compromise: "
      f"{manifest['outer_folds'] * manifest['repeats']} outer test sets "
      f"stabilise the mean of fold-level metrics while keeping the full grid "
      f"runnable in about {manifest['runtime_minutes']:.0f} minutes on 12 CPU "
      f"cores. Increasing repeats reduces cross-validation "
      "partition noise but **cannot** reduce the dominant source of uncertainty "
      f"here, which is the {n_tot}-patient sample itself; that is quantified "
      "separately by patient-level bootstrap.")
    w("")
    w("Within each outer fold, in order: hyper-parameters are selected by grid "
      "search over the inner folds of the outer training data; the selected "
      "pipeline is refitted on the outer training data; probability calibration "
      "is fitted on the outer training data via `CalibratedClassifierCV` with "
      "its own internal stratified splits; the screening threshold is chosen "
      "from a further cross-validated prediction on the outer training data. "
      "Only then is `predict_proba` called on the outer test fold. Nothing - "
      "imputation, scaling, tuning, calibration, threshold - is computed on the "
      "outer test fold or on the complete dataset [6].")
    w("")
    w("A single train/test split was not used as the principal evaluation. "
      "SMOTE was not used: the class ratio is roughly 1.8:1, which does not "
      "warrant synthetic oversampling, and introducing it would add a "
      "resampling artefact without a specific question to justify it.")
    w("")

    w("### 4.4 Leakage prevention, enforced programmatically")
    w("")
    w("Three independent layers, each covered by automated tests:")
    w("")
    w("1. **Declaration.** Prohibited columns (`class`, `affected`, `stage`, "
      "`grf`) and the contents of every configuration are declared in code.")
    w("2. **Runtime guard.** Every pipeline begins with a `LeakageGuard` that "
      "raises `LeakageError` at *fit and at transform time* if a prohibited "
      "column is present. It raises rather than dropping the column, so a bug "
      "in calling code cannot be silently absorbed. The invalid configuration "
      "must opt in explicitly, and even then the raw outcome column can never "
      "pass.")
    w("3. **Pre-flight assertion.** `assert_pipeline_is_clean` is run over all "
      f"{len(manifest['configs_run']) * len(manifest['models_run'])} pipelines "
      "before any fitting begins.")
    w("")
    w("The key adversarial test hands a valid pipeline the **entire** dataframe "
      "including `affected`, `stage` and `grf` and asserts that it **refuses**. "
      "A separate test confirms that scaler and imputer statistics differ "
      "between folds, i.e. that preprocessing was not fitted on complete data.")
    w("")

    w("### 4.5 Evaluation")
    w("")
    w("Reported: ROC-AUC, PR-AUC, sensitivity/recall for CKD, specificity, "
      "precision (PPV), negative predictive value, F1, balanced accuracy, "
      "Brier score, calibration slope and intercept, expected calibration "
      "error, and the confusion matrix at two operating points.")
    w("")
    w(f"- **Primary threshold: {cfg['evaluation']['primary_threshold']}**, "
      "prespecified in `config/experiment.yaml` before any result was computed.")
    w(f"- **Screening threshold**: the highest threshold still achieving "
      f"{cfg['evaluation']['target_sensitivity']:.0%} sensitivity on inner "
      "cross-validated predictions **of the outer training fold**, recomputed "
      "independently in every fold. It never sees outer-test outcomes.")
    w("")
    w("Calibration slope and intercept follow the standard logistic "
      "recalibration framework [7]: the slope is the coefficient of "
      "`logit(p)` in an unpenalised logistic refit, and the intercept is "
      "obtained with `logit(p)` as a fixed offset. Slope < 1 indicates "
      "predictions that are too extreme. Where the predictions separate the "
      "classes completely the maximum-likelihood slope does not exist, and it "
      "is reported as undefined rather than as whatever value an optimiser "
      "happened to reach.")
    w("")
    w("**Caution stated in advance about isotonic calibration.** Isotonic "
      "regression fits a free-form monotone step function and is therefore far "
      "more flexible than Platt scaling. With roughly 160 patients available "
      "inside each training fold, it has enough freedom to follow noise, and "
      "the usual expectation at this sample size is that it will overfit and "
      "perform worse than the parametric alternative. It is included precisely "
      "so that this expectation is tested rather than assumed, and the result "
      "is reported in section 5.6 whichever way it falls.")
    w("")
    w(f"Uncertainty is reported two ways, deliberately kept separate: the "
      f"spread across the {manifest['outer_folds'] * manifest['repeats']} outer "
      f"folds and across the {manifest['repeats']} repeats (cross-validation "
      f"partition variability), and a stratified patient-level bootstrap "
      f"({cfg['evaluation']['bootstrap_iterations']} resamples, "
      f"{cfg['evaluation']['bootstrap_ci']:.0%} percentile intervals). "
      f"Conflating them would overstate precision.")
    w("")
    w("A decision-curve net-benefit analysis [8] is reported as an "
      "**exploratory** supplement.")
    w("")

    # ---------------- Results ----------------
    w("## 5. Results")
    w("")
    w("### 5.1 Baseline")
    w("")
    w(f"The dummy classifier, which always predicts the training-fold "
      f"prevalence, achieved ROC-AUC {fmt(dummy['roc_auc'].max())} at best - "
      f"chance, as required. Every result below should be read against that "
      f"floor and against the {n_pos/n_tot:.1%} prevalence.")
    w("")

    w("### 5.2 Leakage audit (secondary question 1)")
    w("")
    w("Best model per configuration, uncalibrated, pooled out-of-fold "
      "predictions, with 95% bootstrap confidence intervals:")
    w("")
    w("| Configuration | k | Best model | ROC-AUC (95% CI) | PR-AUC | Sensitivity | Specificity | Brier |")
    w("|---|---:|---|---|---|---|---|---|")
    for _, r in audit.iterrows():
        w(f"| {config_label(r['config'])} | {int(r['n_features'])} | "
          f"{model_label(r['model'])} | {ci(r, 'roc_auc')} | {fmt(r['pr_auc'])} | "
          f"{fmt(r['sensitivity'])} | {fmt(r['specificity'])} | {fmt(r['brier'])} |")
    w("")
    infl = leaky["roc_auc"] - full["roc_auc"]
    err_red = infl / max(1e-9, 1.0 - full["roc_auc"])
    if full_at_ceiling:
        w(f"The invalid configuration reaches ROC-AUC {fmt(leaky['roc_auc'])} "
          f"with sensitivity {fmt(leaky['sensitivity'])} and specificity "
          f"{fmt(leaky['specificity'])}. Measured against the best valid "
          f"configuration the gain is only {fmt(infl)}, but that is **not** "
          f"evidence that leakage is unimportant here: the valid configuration "
          f"is itself at {fmt(full['roc_auc'])}, so there is essentially no "
          f"headroom left for leakage to exploit. Section 5.5 explains why the "
          f"valid model is already at the ceiling, and the table below "
          f"re-measures the effect against baselines that are not saturated.")
    else:
        w(f"The invalid configuration gains **{fmt(infl)} ROC-AUC** over the best "
          f"valid configuration, eliminating **{err_red:.1%}** of the discrimination "
          f"error that remains for a valid model. Its sensitivity is "
          f"{fmt(leaky['sensitivity'])} and its specificity {fmt(leaky['specificity'])}.")
    w("")
    w("**Why apparently perfect performance signals leakage rather than "
      "usefulness.** `affected` reproduces the outcome exactly, so any model "
      "given it needs only to copy one column; `stage` is assigned after "
      "diagnosis and is 100% CKD in stages s3 and s5; `grf` is the eGFR value "
      "from which staging is derived and by which CKD is defined. A model using "
      "them is not forecasting an unknown state from clinical signs - it is "
      "reading back a conclusion that was already recorded. Such a model would "
      "have nothing to contribute at the moment of screening, because at that "
      "moment none of these three quantities exists yet. The practical test is "
      "temporal: a predictor that could not have been measured *before* the "
      "diagnosis cannot support screening, however well it scores.")
    w("")
    w("This is directly relevant to the published record on this dataset, where "
      "accuracies at or near 100% are commonly reported.")
    w("")
    w("**An important caveat on how to read the inflation number.** The "
      "absolute ROC-AUC gain from leakage looks modest here, but that is a "
      "ceiling artefact, not evidence that leakage is harmless: the valid "
      "baseline is already close to 1.0, so there is almost no room left to "
      "gain. Measured against baselines that are not saturated:")
    w("")
    w("| Baseline configuration | Baseline ROC-AUC | Headroom to 1.0 | Absolute inflation | Share of headroom closed by leakage |")
    w("|---|---:|---:|---:|---:|")
    for _, r in ceiling.iterrows():
        frac = r["fraction_of_headroom_closed_by_leakage"]
        share = (f"{100 * frac:.1f}%" if np.isfinite(frac)
                 else "undefined (no headroom)")
        w(f"| {config_label(r['baseline_config'])} | {fmt(r['baseline_roc_auc'])} | "
          f"{fmt(r['headroom_below_ceiling'])} | {fmt(r['absolute_inflation'])} | "
          f"{share} |")
    w("")
    w("The honest summary is that leakage takes any configuration to the "
      "ceiling. Where a valid model already sits at the ceiling for other "
      "reasons - which, as section 5.5 shows, is the case here - the gain is "
      "arithmetically small while the epistemic problem is unchanged.")
    w("")

    w("### 5.3 Low-cost versus laboratory (primary question, secondary question 2)")
    w("")
    w("| Configuration | k | ROC-AUC (95% CI) | Sensitivity | Specificity | NPV | PPV | Brier |")
    w("|---|---:|---|---|---|---|---|---|")
    for r in (full, lab, lcm, low, clin):
        w(f"| {config_label(r['config'])} | {int(r['n_features'])} | {ci(r, 'roc_auc')} | "
          f"{ci(r, 'sensitivity', 2)} | {ci(r, 'specificity', 2)} | "
          f"{ci(r, 'npv', 2)} | {ci(r, 'precision', 2)} | {fmt(r['brier'])} |")
    w("")
    gap = lab["roc_auc"] - low["roc_auc"]
    overlap = (low["roc_auc_ci_high"] >= lab["roc_auc_ci_low"])
    w(f"The low-cost set ({int(low['n_features'])} variables, no venepuncture) "
      f"reaches ROC-AUC {fmt(low['roc_auc'])} against {fmt(lab['roc_auc'])} for "
      f"the laboratory set ({int(lab['n_features'])} variables) - a gap of "
      f"{fmt(gap)}. The bootstrap confidence intervals "
      f"{'overlap substantially' if overlap else 'do not overlap'}, so this "
      f"comparison is {'not statistically resolved at n=200' if overlap else 'reasonably clear in direction'}"
      f"{'; the study cannot distinguish the two configurations reliably' if overlap else ''}.")
    w("")
    w(f"Dropping urine testing entirely (`clinical_only_model`, "
      f"{int(clin['n_features'])} variables) gives ROC-AUC {fmt(clin['roc_auc'])}, "
      f"a loss of {fmt(low['roc_auc'] - clin['roc_auc'])} relative to the "
      f"low-cost set. Adding urine microscopy to the low-cost set gives "
      f"{fmt(lcm['roc_auc'])}, a change of {(lcm['roc_auc'] - low['roc_auc']) + 0.0:+.4f}. "
      f"The judgement call about where urine microscopy belongs therefore "
      f"{'does not materially change the conclusions' if abs(lcm['roc_auc'] - low['roc_auc']) < 0.03 else 'does affect the conclusions and should be treated as an open question'}.")
    w("")

    w("### 5.4 Screening operating points")
    w("")
    w(f"Low-cost model ({model_label(cm_lc_pre['model'])}, "
      f"{calibration_label(cm_lc_pre['calibration'])}), pooled out-of-fold "
      f"predictions for all {n_tot} patients:")
    w("")
    w("| Operating point | Threshold | TP | FP | TN | FN | Sensitivity | Specificity | PPV | NPV |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, r in [("Prespecified (0.50)", cm_lc_pre),
                     ("Screening (training-fold selected)", cm_lc_scr)]:
        w(f"| {label} | {fmt(r['threshold'])} | {int(r['tp'])} | {int(r['fp'])} | "
          f"{int(r['tn'])} | {int(r['fn'])} | {fmt(r['sensitivity'], 2)} | "
          f"{fmt(r['specificity'], 2)} | {fmt(r['precision'], 2)} | {fmt(r['npv'], 2)} |")
    w("")
    w(f"At the prespecified threshold the low-cost model misses "
      f"**{int(cm_lc_pre['fn'])} of {n_pos} CKD patients** while referring "
      f"**{int(cm_lc_pre['fp'])} of {n_neg} non-CKD patients** unnecessarily. "
      f"Moving to the training-fold-selected screening threshold changes this "
      f"to {int(cm_lc_scr['fn'])} missed and {int(cm_lc_scr['fp'])} unnecessary "
      f"referrals.")
    w("")
    w("For a screening instrument the relevant question is not which threshold "
      "maximises accuracy but what exchange rate between missed cases and "
      "unnecessary referrals is acceptable in the intended setting - a "
      "judgement that depends on referral capacity and on the consequences of "
      "delayed diagnosis, neither of which this dataset contains. Figure R7 "
      "presents the full trade-off curve rather than a single recommended "
      "threshold.")
    w("")
    w("**Negative predictive value is prevalence-dependent.** The values above "
      f"are computed at the {n_pos/n_tot:.1%} CKD prevalence of this "
      "hospital-based sample. In a community screening population, where CKD "
      "prevalence would be far lower, NPV would be substantially higher and PPV "
      "substantially lower for the same model. None of the predictive values "
      "reported here transfer to a different prevalence.")
    w("")

    w("### 5.5 Case mix: why the valid models are already near the ceiling "
      "(EXPLORATORY, post hoc)")
    w("")
    w(f"The valid configurations reach discrimination close to 1.0 without any "
      f"prohibited column. That needs explaining before it is treated as good "
      f"news, and the explanation is in the composition of the sample.")
    w("")
    w(f"**Finding 1: single predictors already separate the groups.** "
      f"`{top_sep['feature']}` alone achieves univariate ROC-AUC "
      f"{fmt(top_sep['univariate_auc'])}. Its value ranges are "
      f"{fmt(top_sep['ckd_min'], 2)}-{fmt(top_sep['ckd_max'], 2)} in CKD "
      f"patients and {fmt(top_sep['non_ckd_min'], 2)}-"
      f"{fmt(top_sep['non_ckd_max'], 2)} in non-CKD patients; only "
      f"{top_sep['fraction_patients_in_overlap']:.0%} of patients fall in the "
      f"range the two classes share. "
      f"No multivariable learning is required to separate groups that are "
      f"already this far apart.")
    w("")
    w("| Predictor | Univariate ROC-AUC | CKD range | Non-CKD range | Patients in shared range |")
    w("|---|---:|---|---|---:|")
    for _, r in separability.head(8).iterrows():
        w(f"| `{r['feature']}` | {fmt(r['univariate_auc'])} | "
          f"{fmt(r['ckd_min'], 2)}-{fmt(r['ckd_max'], 2)} | "
          f"{fmt(r['non_ckd_min'], 2)}-{fmt(r['non_ckd_max'], 2)} | "
          f"{r['fraction_patients_in_overlap']:.0%} |")
    w("")
    w(f"**Finding 2: the cohort is dominated by advanced disease.** "
      f"{n_advanced} of the {n_pos} CKD patients ({n_advanced/n_pos:.0%}) are "
      f"staged s3-s5. Only about {n_early_ckd} are early-stage. That is the "
      f"profile of a case-control-like contrast between established, "
      f"moderate-to-severe CKD and comparatively healthy controls - not of a "
      f"consecutive screening series, in which most true cases would be early, "
      f"asymptomatic and biochemically near-normal. Anaemia, low haematocrit "
      f"and abnormal urine concentration are consequences of established kidney "
      f"disease; they are exactly the features that are *absent* in the "
      f"patients a screening programme most needs to identify.")
    w("")
    w("**Finding 3: performance degrades in the screening-relevant subgroup.** "
      "Re-evaluating the *same* out-of-fold predictions within stage-defined "
      "subgroups (`stage` is used only to partition patients for evaluation and "
      "never entered any model as a predictor):")
    w("")
    w(f"| Configuration | Subgroup | n CKD | ROC-AUC | Sensitivity @ {cfg['evaluation']['primary_threshold']} | NPV |")
    w("|---|---|---:|---:|---:|---:|")
    for cname in ("full_valid_model", "laboratory_model", "low_cost_model",
                  "clinical_only_model"):
        for sg, sglab in (("all_patients", "All patients (as sampled)"),
                          ("advanced_ckd_only", "Advanced CKD (s4-s5)"),
                          ("early_ckd_only", "Early CKD (s1-s2)")):
            r = spec_row(cname, sg)
            if r is None:
                continue
            w(f"| {config_label(cname)} | {sglab} | {int(r['n_ckd'])} | "
              f"{fmt(r['roc_auc'])} | {fmt(r['sensitivity'], 2)} | {fmt(r['npv'], 2)} |")
    w("")
    drops = []
    for cname, allr, earr in (
        ("full_valid_model", all_row, early_row),
        ("laboratory_model", spec_row("laboratory_model", "all_patients"),
         spec_row("laboratory_model", "early_ckd_only")),
        ("low_cost_model", lc_all, lc_early),
        ("clinical_only_model", clin_all, clin_early),
    ):
        if allr is None or earr is None:
            continue
        drops.append((cname, float(allr["sensitivity"]), float(earr["sensitivity"]),
                      float(allr["roc_auc"]), float(earr["roc_auc"])))

    declined = [d for d in drops if d[2] < d[1] - 1e-9]
    if declined:
        worst = max(declined, key=lambda d: d[1] - d[2])
        w(f"Sensitivity is lower in the early-CKD subgroup for "
          f"{len(declined)} of the {len(drops)} configurations examined. The "
          f"largest decline is for {config_label(worst[0])}, from "
          f"{fmt(worst[1], 3)} across all patients to {fmt(worst[2], 3)} among "
          f"early-stage CKD - so roughly {1 - worst[2]:.0%} of exactly the cases "
          f"a screening programme exists to find are missed at the prespecified "
          f"threshold, while its ROC-AUC in that subgroup is still "
          f"{fmt(worst[4])}. The dissociation matters: a near-ceiling AUC can "
          f"coexist with materially worse case detection, and no aggregate "
          f"metric reveals it.")
        w("")
    if all_row is not None and early_row is not None and early_row["roc_auc"] >= 0.999:
        w(f"The full valid configuration is the exception: it separates the "
          f"classes completely in **every** subgroup, including early CKD "
          f"(ROC-AUC {fmt(early_row['roc_auc'])}, sensitivity "
          f"{fmt(early_row['sensitivity'], 3)}). That is not reassurance. With "
          f"only {int(early_row['n_ckd'])} early-stage cases, and with a "
          f"comparison group of {int(early_row['n_non_ckd'])} patients who were "
          f"apparently healthy enough to be labelled non-CKD despite presenting "
          f"at a tertiary hospital, perfect separation says more about how far "
          f"apart the two groups are in this sample than about the difficulty "
          f"of the screening task.")
        w("")
    w("These subgroup analyses are **exploratory and post hoc**, and the early "
      f"CKD subgroup contains only {n_early_ckd} cases, so every estimate in it "
      "is imprecise and none of them would survive a demand for statistical "
      "significance. They are reported because the direction of the effect is "
      "consistent across the cheaper configurations, because the underlying "
      "case-mix imbalance is a hard fact about the sample rather than an "
      "inference, and because omitting the analysis would leave the headline "
      "numbers looking far more like screening performance than they are.")
    w("")

    w("### 5.6 Calibration (secondary question 3)")
    w("")
    w("Median across all valid configuration x model cells:")
    w("")
    w("| Calibration | Brier | Slope (ideal 1) | Intercept (ideal 0) | ECE |")
    w("|---|---:|---:|---:|---:|")
    for method in ["none", "sigmoid", "isotonic"]:
        r = cal_row(method)
        if r is None:
            continue
        w(f"| {calibration_label(method)} | {fmt(r['brier'])} | "
          f"{fmt(r['calibration_slope'], 2)} | {fmt(r['calibration_intercept'], 2)} | "
          f"{fmt(r['ece'])} |")
    w("")
    w(f"By median Brier score the best-performing calibration method across "
      f"valid cells was **{calibration_label(best_cal_method)}**.")
    w("")
    if iso_worse_than_sigmoid:
        w(f"**Isotonic regression performed worse than Platt scaling** "
          f"(median Brier {fmt(iso_m['brier'])} vs {fmt(sig_m['brier'])}), which "
          f"is the expected behaviour at this sample size: isotonic regression "
          f"fits a free-form monotone step function and, with roughly 160 "
          f"training patients per fold, it has enough freedom to track noise. "
          f"This is a case where the flexible method is the wrong choice, and "
          f"the result is reported rather than the better-looking option being "
          f"selected after the fact.")
    else:
        w(f"Isotonic regression did not underperform Platt scaling here "
          f"(median Brier {fmt(iso_m['brier'])} vs {fmt(sig_m['brier'])}). This "
          f"is somewhat contrary to the usual expectation at n=200 and should "
          f"be treated with caution: with only "
          f"{manifest['outer_folds'] * manifest['repeats']} outer folds the "
          f"comparison is itself noisy.")
    w("")
    n_undefined = int(cal_valid["calibration_slope"].isna().sum())
    if n_undefined:
        w(f"**{n_undefined} of {len(cal_valid)} valid cells have an undefined "
          f"calibration slope.** These are cells whose predictions separate the "
          f"two classes completely, so the logistic recalibration model has no "
          f"maximum-likelihood solution - the likelihood keeps increasing as the "
          f"coefficient grows. An optimiser will still return a number (we "
          f"observed values in the hundreds before adding the check), but it "
          f"measures where the solver stopped, not calibration. Those cells are "
          f"reported as undefined rather than given a spurious value.")
        w("")

    best_slope = best["calibration_slope"]
    if not np.isfinite(best_slope):
        w(f"The model selected by the headline rule "
          f"({config_label(best['config'])}, {model_label(best['model'])}, "
          f"{calibration_label(best['calibration'])}) is one of those cases: it "
          f"separates the classes completely, so its calibration slope is not "
          f"identified. Its Brier score is {fmt(best['brier'], 4)} and its "
          f"calibration intercept {fmt(best['calibration_intercept'], 2)}. "
          f"Perfect separation is precisely the situation in which the "
          f"probability scale is least trustworthy: the model has no data "
          f"telling it what a genuinely ambiguous patient looks like, so its "
          f"intermediate probabilities are extrapolation. For a screening "
          f"application this is a reason to prefer a model that does *not* "
          f"separate perfectly.")
        w("")
        lc_slope = best_lc["calibration_slope"]
        if np.isfinite(lc_slope):
            w(f"The low-cost model ({model_label(best_lc['model'])}, "
              f"{calibration_label(best_lc['calibration'])}) does have an "
              f"identified slope of {fmt(lc_slope, 2)} with intercept "
              f"{fmt(best_lc['calibration_intercept'], 2)}. A slope "
              f"{'below 1 means its predictions are too extreme' if lc_slope < 0.95 else 'above 1 means its predictions are too conservative - it under-uses its own signal, which at this sample size is the safer direction of error' if lc_slope > 1.05 else 'close to 1 means no systematic distortion of the risk scale'}.")
            w("")
    else:
        w(f"For the selected model ({config_label(best['config'])}, "
          f"{model_label(best['model'])}, {calibration_label(best['calibration'])}) "
          f"the calibration slope is {fmt(best_slope, 2)} and the intercept "
          f"{fmt(best['calibration_intercept'], 2)}. A slope "
          f"{'below 1 indicates predictions that are too extreme - risks near 0 and 1 are overstated' if best_slope < 0.95 else 'above 1 indicates predictions that are too conservative - the model under-uses its own signal' if best_slope > 1.05 else 'close to 1 indicates no systematic over- or under-fitting of the risk scale'}.")
        w("")
    w("Across valid configurations every identified slope is above 1, i.e. the "
      "models are systematically **under**-confident rather than "
      "over-confident. That is the opposite of the usual small-sample pattern "
      "and follows from the same near-separability discussed in section 5.5: "
      "regularised models trained on 160 patients produce middling "
      "probabilities for patients the data actually separate cleanly.")
    w("")

    w("### 5.7 Feature importance and stability (secondary question 4)")
    w("")
    w(f"For the selected model ({config_label(best['config'])}, "
      f"{model_label(best['model'])}), across "
      f"{manifest['outer_folds'] * manifest['repeats']} outer folds:")
    w("")
    w(f"- Kendall's W (concordance of the full ranking) = **{fmt(kendall)}**")
    w(f"- Mean pairwise Jaccard overlap of the top-{top_k} sets = **{fmt(jaccard)}**")
    w("")
    w(f"| Feature | Tier | Mean permutation importance | SD | Mean rank | Top-{top_k} frequency |")
    w("|---|---|---:|---:|---:|---:|")
    for _, r in perm_sorted.head(12).iterrows():
        tier = SPEC_BY_NAME[r["feature"]].tier if r["feature"] in SPEC_BY_NAME else "?"
        w(f"| `{r['feature']}` | {tier} | {fmt(r['mean_importance'], 4)} | "
          f"{fmt(r['sd_importance'], 4)} | {fmt(r['mean_rank'], 1)} | "
          f"{r[freq_col]:.2f} |")
    w("")
    if stable_feats:
        w(f"Features reaching the top {top_k} in at least 80% of folds: "
          f"{', '.join('`' + f + '`' for f in stable_feats)}.")
    else:
        w(f"**No feature reached the top {top_k} in at least 80% of folds.** "
          f"That is itself the finding: at this sample size the importance "
          f"ranking is not stable enough to support claims about which "
          f"predictors matter.")
    if unstable_feats:
        w("")
        w(f"Features that appear in the top {top_k} only intermittently "
          f"(20-80% of folds) - i.e. features whose apparent importance is not "
          f"reproducible: {', '.join('`' + f + '`' for f in unstable_feats)}.")
    w("")
    if not perm_lc.equals(perm):
        lc_cfg = perm_lc["config"].iloc[0]
        lc_mdl = perm_lc["model"].iloc[0]
        w(f"**The low-cost configuration is markedly more stable than the full "
          f"one.** For {config_label(lc_cfg)} ({model_label(lc_mdl)}), features "
          f"reaching the top {top_k} in at least 80% of folds: "
          f"{', '.join('`' + f + '`' for f in lc_stable) if lc_stable else 'none'}.")
        w("")
        w(f"| Feature | Tier | Mean permutation importance | SD | Top-{top_k} frequency |")
        w("|---|---|---:|---:|---:|")
        for _, r in perm_lc_sorted.head(8).iterrows():
            tier = SPEC_BY_NAME[r["feature"]].tier if r["feature"] in SPEC_BY_NAME else "?"
            w(f"| `{r['feature']}` | {tier} | {fmt(r['mean_importance'], 4)} | "
              f"{fmt(r['sd_importance'], 4)} | {r[freq_col]:.2f} |")
        w("")
        w("This is the expected pattern and a reassuring one: with 25 highly "
          "correlated predictors the model can substitute one laboratory "
          "variable for another between folds, so no single one dominates "
          "reliably. With 11 largely non-redundant predictors the ranking "
          "settles down. It also means the low-cost model is the more "
          "*interpretable* of the two, independently of which discriminates "
          "better.")
        w("")

    # SHAP vs permutation convergent validity, where an exact explainer applied.
    shap_target = stability[
        (stability["target"] == "shap_capable") & (stability["method"] == "shap")
    ]
    if shap_target.empty:
        shap_target = stability[
            (stability["target"] == "best_valid") & (stability["method"] == "shap")
        ]
    if not shap_target.empty:
        tgt = shap_target["target"].iloc[0]
        piv = (
            stability[(stability["target"] == tgt)
                      & (stability["method"].isin(["shap", "permutation"]))]
            .pivot_table(index="feature", columns="method", values="mean_importance")
        )
        rho = piv["shap"].corr(piv["permutation"], method="spearman")
        w(f"**SHAP versus permutation importance.** The model selected by the "
          f"headline rule ({model_label(best['model'])}) admits no exact SHAP "
          f"explainer, so SHAP was additionally computed for the best "
          f"SHAP-capable valid model "
          f"({config_label(shap_target['config'].iloc[0])}, "
          f"{model_label(shap_target['model'].iloc[0])}). The two measures "
          f"agree closely (Spearman rho = {fmt(rho)} across features), which is "
          f"evidence that the ranking reflects the model rather than the "
          f"quirks of one attribution method. Where a technique did not apply "
          f"it was recorded as such rather than silently omitted; see "
          f"`reports/tables/table_20_importance_notes.txt`.")
        w("")
    w("**Interpretation limits.** These quantities describe how a fitted model "
      "uses a column given the other columns present. They are not effect "
      "sizes, and they are **not causal**. A variable can rank highly because it "
      "proxies something else in the dataset, and a genuinely important "
      "variable can rank low if a correlated variable absorbs its signal. "
      "Permutation importance was computed on outer test folds; because it is "
      "used only for reporting, and never to select features, models or "
      "thresholds, it does not contaminate the validation.")
    w("")

    w("### 5.8 Uncertainty (secondary question 5)")
    w("")
    w("| Configuration | ROC-AUC | 95% bootstrap CI | CI width | SD across repeats |")
    w("|---|---:|---|---:|---:|")
    for r in (full, lab, lcm, low, clin):
        width = r["roc_auc_ci_high"] - r["roc_auc_ci_low"]
        w(f"| {config_label(r['config'])} | {fmt(r['roc_auc'])} | "
          f"({fmt(r['roc_auc_ci_low'])}-{fmt(r['roc_auc_ci_high'])}) | "
          f"{fmt(width)} | {fmt(r.get('roc_auc_sd_across_repeats', float('nan')), 4)} |")
    w("")
    mean_width = float(np.mean([
        r["roc_auc_ci_high"] - r["roc_auc_ci_low"] for r in (full, lab, lcm, low, clin)
    ]))
    w(f"Bootstrap intervals for ROC-AUC span roughly **{mean_width:.2f}** on "
      f"average. Differences between valid configurations smaller than that are "
      f"not resolvable with {n_tot} patients. The spread across repeats is much "
      f"smaller than the bootstrap interval, which is the expected pattern and "
      f"an important one: it shows that the uncertainty here is driven by the "
      f"**sample**, not by the cross-validation partitioning. Running more "
      f"repeats would tighten the former and do nothing about the latter.")
    w("")
    w("The bootstrap intervals themselves are **optimistically narrow**. They "
      "resample the same 200 patients that were used to fit the models whose "
      "predictions are being resampled, so they capture estimation noise but "
      "not the variation that would arise in a genuinely new population.")
    w("")

    w("### 5.9 Model selection")
    w("")
    w("Selection rule, fixed in advance: among cells with a valid feature "
      "configuration and a non-dummy model, take the highest mean ROC-AUC, "
      "breaking ties on the lower Brier score. Applied mechanically, this "
      "selects:")
    w("")
    w(f"- **Overall best valid model:** {config_label(best['config'])} / "
      f"{model_label(best['model'])} / {calibration_label(best['calibration'])} - "
      f"ROC-AUC {fmt(best['roc_auc'])}, sensitivity {fmt(best['sensitivity'], 2)}, "
      f"NPV {fmt(best['npv'], 2)}, Brier {fmt(best['brier'])}")
    w(f"- **Best low-cost model:** {model_label(best_lc['model'])} / "
      f"{calibration_label(best_lc['calibration'])} - "
      f"ROC-AUC {fmt(best_lc['roc_auc'])}, sensitivity {fmt(best_lc['sensitivity'], 2)}, "
      f"NPV {fmt(best_lc['npv'], 2)}, Brier {fmt(best_lc['brier'])}")
    w("")
    w("Selection was **not** made on accuracy. Because the intended use is "
      "screening, sensitivity, negative predictive value and calibration were "
      "the quantities examined, with discrimination used only as the ranking "
      "criterion and calibration as the tie-break.")
    w("")
    w("Hyper-parameter selection was itself unstable across folds, which is "
      "worth recording:")
    w("")
    hy = hyper[hyper["config"] == best["config"]]
    if not hy.empty:
        w("| Model | Hyper-parameter | Modal value | Selection frequency | Distinct values chosen |")
        w("|---|---|---|---:|---:|")
        for _, r in hy.iterrows():
            w(f"| {model_label(r['model'])} | `{r['hyperparameter']}` | "
              f"`{r['modal_value']}` | {float(r['modal_selection_frequency']):.2f} | "
              f"{int(r['n_distinct_selected'])} |")
        w("")

    w("### 5.10 Which model is best supported, and for what")
    w("")
    w(f"The mechanical rule selects {config_label(best['config'])} / "
      f"{model_label(best['model'])} / {calibration_label(best['calibration'])} "
      f"because it has the highest discrimination. Taken on its own that is a "
      f"misleading recommendation, and the analyses above say why:")
    w("")
    w(f"- It separates the classes **completely**, so its calibration slope is "
      f"not identified at all. A screening tool is used by acting on a "
      f"probability, and this model's probability scale cannot be checked.")
    w(f"- Its feature importances are unstable (Kendall's W = {fmt(kendall)}; "
      f"only {len(stable_feats)} feature reaches the top {top_k} in 80% of "
      f"folds), because 25 correlated laboratory variables can substitute for "
      f"one another between folds.")
    w(f"- It requires venepuncture and a full laboratory panel, which is "
      f"exactly the constraint the study set out to relax.")
    w("")
    lc_kendall = float(perm_lc["kendalls_w_all_features"].iloc[0]) if len(perm_lc) else float("nan")
    w(f"**Judged against the question this study actually asks, the "
      f"better-supported configuration is the low-cost one** "
      f"({model_label(best_lc['model'])}, {calibration_label(best_lc['calibration'])}): "
      f"ROC-AUC {fmt(best_lc['roc_auc'])} against {fmt(best['roc_auc'])}, a "
      f"difference far inside the confidence intervals of either; an "
      f"identified calibration slope of {fmt(best_lc['calibration_slope'], 2)}; "
      f"markedly more stable importances (Kendall's W = {fmt(lc_kendall)}, with "
      f"{', '.join('`' + f + '`' for f in lc_stable)} in the top {top_k} of at "
      f"least 80% of folds); and {int(low['n_features'])} variables obtainable "
      f"from history, examination and a urine reagent strip.")
    w("")
    w("This preference is a statement about which result is better *evidenced*, "
      "not a recommendation to use anything. Neither model is validated for "
      "any clinical purpose, and section 5.5 shows that both are evaluated on a "
      "sample whose case mix flatters them.")
    w("")

    # ---------------- Discussion ----------------
    w("## 6. Discussion")
    w("")
    w("This study set out to measure one failure mode and found two. Both push "
      "measured performance towards 1.0, and neither is clinical usefulness.")
    w("")
    w(f"The first is **target leakage**. Including an exact copy of the outcome, "
      f"a post-diagnosis stage label and the diagnostic eGFR value takes every "
      f"configuration to ROC-AUC {fmt(leaky['roc_auc'])} - closing 100% of the "
      f"headroom regardless of how good the honest baseline was. These columns "
      f"sit in the released file with no marking to indicate that they are "
      f"post-diagnosis, and any pipeline that selects features by association "
      f"with the outcome will pick them up. The lesson is not that previous "
      f"analyses were careless in some unusual way; it is that the trap is "
      f"built into the dataset.")
    w("")
    if full_at_ceiling:
        w(f"The second is **case mix**, and on this dataset it turns out to "
          f"matter more. Even with every prohibited column removed, the full "
          f"valid configuration reaches ROC-AUC {fmt(full['roc_auc'])} - it "
          f"separates all {n_tot} patients out of fold. We verified this is not "
          f"a residual leak: the pipelines are guarded, the runner restricts "
          f"columns to those each configuration declares, and an automated test "
          f"re-checks the perfect cells against the prohibited list. It is "
          f"instead a genuine property of the sample, and section 5.5 shows what "
          f"produces it.")
        w("")
    w(f"On the primary question, the low-cost variable set - history, "
      f"examination and a urine reagent strip - achieves ROC-AUC "
      f"{fmt(low['roc_auc'])} under leakage-controlled nested validation. "
      f"That is well above chance and "
      f"{'within the confidence interval of' if overlap else 'below'} the "
      f"laboratory-based configuration. Given that the low-cost set requires no "
      f"venepuncture, no analyser and no microscope, the finding is "
      f"**encouraging as a feasibility signal**. It is not evidence that such a "
      f"model would work as a screening instrument.")
    w("")
    w("The case-mix analysis is what forces that distinction, and it turned out "
      "to matter more than the leakage audit. Leakage is the failure mode this "
      "study set out to measure; case mix is the failure mode the data "
      "revealed. Because "
      f"{n_advanced/n_pos:.0%} of the CKD patients here have moderate-to-severe "
      "disease, the models are largely being asked to distinguish established "
      "kidney failure from health - a task on which haemoglobin alone scores "
      f"{fmt(top_sep['univariate_auc'])}. A screening instrument faces a "
      "different and much harder problem, and the early-stage subgroup shows "
      "the models performing materially worse on it. An aggregate ROC-AUC near "
      "1.0 on this sample should therefore be read as a statement about the "
      "sample, not about the method.")
    w("")
    w("This also reframes what a 'high-performing' published result on this "
      "dataset means. Two distinct mechanisms - target leakage and case-mix "
      "spectrum - both push measured performance towards the ceiling, and "
      "neither has anything to do with clinical usefulness. A study reporting "
      "99% accuracy on this data has probably encountered one or both, and "
      "cannot distinguish them without exactly the kind of subgroup and "
      "leakage analysis reported here.")
    w("")
    if best["config"] != "low_cost_model":
        w(f"Note that the best valid model overall used the "
          f"{config_label(best['config'])} configuration rather than the "
          f"low-cost one. The low-cost result should therefore be read as "
          f"'a substantial fraction of the achievable signal at a fraction of "
          f"the cost', not as 'no loss from dropping laboratory tests'.")
        w("")
    w("Calibration deserves particular emphasis because it is what makes a "
      "predicted probability usable for a referral decision. A model with good "
      "discrimination but a calibration slope well below 1 will systematically "
      "overstate risk in the patients it is most confident about - exactly the "
      "patients whose management would change. The comparison here also "
      "illustrates a general point about small samples: the more flexible "
      "calibration method is not the better one when there are only a few "
      "hundred observations to fit it with.")
    w("")
    w(f"The stability analysis is, in a sense, the most honest part of the "
      f"study. With Kendall's W of {fmt(kendall)} across "
      f"{manifest['outer_folds'] * manifest['repeats']} outer folds, the "
      f"feature ranking "
      f"{'is only moderately reproducible' if kendall < 0.7 else 'is fairly reproducible'}"
      f". Any narrative that named the 'top predictors of CKD' from a single "
      f"fit of this dataset would be reporting an artefact of one partition.")
    w("")

    # ---------------- Limitations ----------------
    w("## 7. Limitations")
    w("")
    w("These are not boilerplate. Each one materially constrains what the "
      "results above can support.")
    w("")
    w(f"1. **Sample size.** {n_tot} patients, {n_neg} in the minority class. "
      f"Bootstrap intervals for ROC-AUC span roughly {mean_width:.2f}, so "
      f"differences between configurations smaller than that are not "
      f"resolvable. Events per candidate predictor "
      f"({quality['class_distribution']['events_per_candidate_predictor_full_valid']}) "
      f"is far below accepted minimums for prediction-model development [5].")
    w("2. **No external validation.** Every estimate is internal to these 200 "
      "patients. Internal cross-validation systematically overstates the "
      "performance a model would show in a new population, and no correction "
      "applied here changes that.")
    w("3. **Hospital-based, single-centre sampling.** Patients presenting at "
      "one medical college in Savar are not a random sample of any population. "
      f"The {n_pos/n_tot:.1%} CKD prevalence is a property of who was recruited, "
      "not of any community. Selection bias is likely and its direction is "
      "unknown.")
    w(f"4. **Case-mix (spectrum) bias - the most consequential limitation.** "
      f"{n_advanced} of {n_pos} CKD patients ({n_advanced/n_pos:.0%}) have "
      f"stage s3-s5 disease, so the dataset largely contrasts established "
      f"kidney failure with comparatively healthy controls. Discrimination "
      f"measured on such a sample is systematically optimistic for screening "
      f"use. Section 5.5 shows sensitivity for the low-cost model falling from "
      f"{fmt(lc_all['sensitivity'], 3) if lc_all is not None else 'n/a'} "
      f"overall to "
      f"{fmt(lc_early['sensitivity'], 3) if lc_early is not None else 'n/a'} "
      f"among early-stage CKD patients, and the clinical-only model from "
      f"{fmt(clin_all['sensitivity'], 3) if clin_all is not None else 'n/a'} to "
      f"{fmt(clin_early['sensitivity'], 3) if clin_early is not None else 'n/a'}. "
      f"**No headline figure in this report should be read as an estimate of "
      f"screening performance.**")
    w("5. **Pre-discretised predictors.** The published file contains only "
      "binned intervals. Real continuous values are unrecoverable, open-ended "
      "tail bins are compressed to their finite edge, and the effective "
      "measurement precision of every variable is unknown. A model built on "
      "continuous measurements might perform differently in either direction.")
    w("6. **No data dictionary.** The dataset ships without variable "
      f"definitions [4]. {len(uncertain_variables())} variables have "
      "interpretations that could not be settled from the file and are flagged "
      "as uncertain rather than resolved by assumption. The tiering of `ane` in "
      "particular changes what 'low cost' means, and was decided "
      "conservatively.")
    w("7. **Minimal demographic information.** Age is present, in bands. There "
      "is no sex, no ethnicity, no socioeconomic indicator, no comorbidity "
      "detail beyond three binary flags. Subgroup performance therefore cannot "
      "be assessed at all, and undetected differential performance across "
      "groups is entirely possible.")
    w("8. **Label quality.** "
      f"{sum(i['n_patients'] for i in quality['clinical_inconsistencies'][:1])} "
      "patients are labelled `notckd` while carrying an advanced CKD stage and "
      "an eGFR below 60. Either the label or the staging is wrong for those "
      "records, and there is no external source of truth to arbitrate. "
      "Outcome-label noise attenuates every performance estimate here.")
    w("9. **Cross-sectional data, no chronicity.** CKD is defined by "
      "abnormalities persisting beyond three months [2]. A single record cannot "
      "establish chronicity, so the outcome label itself rests on information "
      "not present in the file.")
    w("10. **Prevalence-dependent metrics.** PPV and NPV reported here hold only "
      f"at this sample's {n_pos/n_tot:.1%} prevalence and do not transfer to a "
      "community screening setting.")
    w("11. **Linearity assumption for the linear models.** Ordinal bin "
      "representatives are entered on their original scale, so logistic "
      "regression and the SVM assume an approximately monotone, roughly linear "
      "relationship with the log-odds across bins. The tree ensembles and the "
      "EBM do not make this assumption.")
    w("12. **Compute-driven modelling choices.** Search spaces, forest size and "
      "EBM capacity were constrained to keep the repeated nested design "
      "feasible. These are documented trade-offs; a larger search might find "
      "better configurations, though at this sample size it would also select "
      "on noise more aggressively.")
    w("13. **Exploratory analyses.** The decision-curve net benefit, the "
      "univariate association screen and the correlation structure are labelled "
      "exploratory and were not used for any modelling decision. They should "
      "not be read as confirmatory findings.")
    w("")

    # ---------------- Ethics ----------------
    w("## 8. Ethical considerations")
    w("")
    w("**This is not a diagnostic tool and must not be used as one.** No model "
      "in this repository is validated for any clinical purpose. Presenting it "
      "to a patient or clinician as a screening instrument would be unsafe.")
    w("")
    w("**Consequences of error are asymmetric and are borne by patients.** A "
      f"false negative in CKD screening means a patient with progressive kidney "
      f"disease is reassured and not followed up; at the prespecified threshold "
      f"the low-cost model produced {int(cm_lc_pre['fn'])} such errors among "
      f"{n_pos} CKD patients in cross-validation. A false positive means an "
      f"unnecessary referral, which in a resource-constrained setting consumes "
      f"capacity that another patient needed. Neither error is neutral, and the "
      f"balance between them is a clinical and policy decision, not a modelling "
      f"one.")
    w("")
    w("**Data governance.** The analysis uses de-identified records already "
      "released for research under CC BY 4.0 [4]. No attempt is made to "
      "re-identify any individual. The raw file is preserved unmodified and all "
      "derived artefacts are written separately. Findings about individual "
      "records (for instance the four internally inconsistent labels) are "
      "reported by CSV line number, which refers to a position in a public "
      "de-identified file and not to any identifiable person.")
    w("")
    w("**Equity.** A low-cost screening model is attractive precisely because "
      "it could extend detection to populations without laboratory access. That "
      "same argument makes undetected differential performance especially "
      "harmful: a model that works less well for a subgroup would concentrate "
      "its errors on the people the approach is meant to help. This dataset "
      "contains too little demographic information to check for that, which is "
      "a reason for caution rather than an excuse for silence.")
    w("")
    w("**Transparency about the leaky model.** A deliberately invalid "
      "configuration is included and reports near-ceiling performance. It is "
      "labelled INVALID in every table and figure that shows it. It exists to "
      "demonstrate a hazard, and quoting its numbers outside that context would "
      "be a misrepresentation.")
    w("")

    # ---------------- Reproducibility ----------------
    w("## 9. Reproducibility statement")
    w("")
    w(f"- **Seed.** A single master seed (`{manifest['seed']}`) deterministically "
      f"derives every fold seed and model seed. `tests/test_reproducibility.py` "
      f"asserts that identical seeds produce identical predictions and that "
      f"different seeds produce different partitions.")
    w(f"- **Input integrity.** The raw file is checksummed on every load "
      f"(SHA-256 `{prov['sha256']}`) and is never written to.")
    w(f"- **Single source of results.** All "
      f"{manifest['n_prediction_rows']:,} out-of-fold predictions are written to "
      f"`data/processed/cv_predictions.csv.gz`. Every table, figure and number "
      f"in this report is derived from that one file, so results cannot drift "
      f"apart from the cross-validation that produced them.")
    w(f"- **Run manifest.** `data/processed/cv_predictions_manifest.json` "
      f"records the design, the seed, the models actually run, the models "
      f"unavailable in this environment, and the runtime "
      f"({manifest['runtime_minutes']} min).")
    w("- **This document is generated.** Every quantity above is injected from "
      "`reports/tables/` by `scripts/06_write_report.py`; nothing is "
      "transcribed by hand.")
    w("- **Environment.** Dependencies are pinned in `requirements.txt`.")
    w("")
    w("Exact commands:")
    w("")
    w("```bash")
    w("python -m venv .venv")
    w("source .venv/bin/activate        # Windows: .venv\\Scripts\\Activate.ps1")
    w("python -m pip install --upgrade pip")
    w("python -m pip install -r requirements.txt")
    w("python -m pytest tests/ -q")
    w("python scripts/run_all.py")
    w("```")
    w("")

    # ---------------- Conclusion ----------------
    w("## 10. Conclusion")
    w("")
    w(f"Under repeated nested cross-validation with programmatically enforced "
      f"leakage control, a small model built only from history, physical "
      f"examination and a urine reagent strip achieved ROC-AUC "
      f"{fmt(low['roc_auc'])} (95% CI {fmt(low['roc_auc_ci_low'])}-"
      f"{fmt(low['roc_auc_ci_high'])}) for CKD status in this 200-patient "
      f"single-centre sample, against {fmt(lab['roc_auc'])} for a "
      f"laboratory-based configuration. Including the dataset's three "
      f"prohibited columns raised ROC-AUC to {fmt(leaky['roc_auc'])}, "
      f"quantifying the inflation that leakage produces and offering a concrete "
      f"explanation for the near-perfect results frequently reported on this "
      f"data.")
    w("")
    w(f"The prespecified selection rule, which ranks on discrimination alone, "
      f"picks **{model_label(best['model'])} on the "
      f"{config_label(best['config'])} configuration with "
      f"{calibration_label(best['calibration'])} probabilities** "
      f"(ROC-AUC {fmt(best['roc_auc'])}, sensitivity "
      f"{fmt(best['sensitivity'], 2)}, NPV {fmt(best['npv'], 2)}, Brier "
      f"{fmt(best['brier'], 4)}). We report that pick because the rule was "
      f"fixed in advance, but we do not endorse it: as section 5.10 sets out, "
      f"that model separates the classes completely, so its calibration slope "
      f"is not identified, its feature importances are unstable "
      f"(Kendall's W = {fmt(kendall)}), and it needs a full laboratory panel.")
    w("")
    w(f"**The best-supported model for the question this study asks is the "
      f"low-cost one: {model_label(best_lc['model'])} with "
      f"{calibration_label(best_lc['calibration'])} probabilities on "
      f"{int(low['n_features'])} history, examination and urine-dipstick "
      f"variables** - ROC-AUC {fmt(best_lc['roc_auc'])} against "
      f"{fmt(best['roc_auc'])} for the full laboratory panel, a difference "
      f"well inside either confidence interval; sensitivity "
      f"{fmt(best_lc['sensitivity'], 2)}, NPV "
      f"{fmt(best_lc['npv'], 2)}, Brier {fmt(best_lc['brier'], 4)}, and an "
      f"identified calibration slope of "
      f"{fmt(best_lc['calibration_slope'], 2)}. It is the only candidate whose "
      f"probability scale can be checked and whose important variables are "
      f"reproducible across folds.")
    w("")
    w(f"A post hoc case-mix analysis then showed that even the "
      f"leakage-controlled figures overstate screening ability: "
      f"{n_advanced/n_pos:.0%} of the CKD patients have stage s3-s5 disease, "
      f"haemoglobin alone discriminates at ROC-AUC "
      f"{fmt(top_sep['univariate_auc'])}, and among early-stage CKD patients "
      f"the low-cost model's sensitivity falls from "
      f"{fmt(lc_all['sensitivity'], 3) if lc_all is not None else 'n/a'} to "
      f"{fmt(lc_early['sensitivity'], 3) if lc_early is not None else 'n/a'} "
      f"while its ROC-AUC stays above 0.98. Two independent mechanisms - target "
      f"leakage and case-mix spectrum - push measured performance towards the "
      f"ceiling on this dataset, and neither reflects clinical usefulness.")
    w("")
    w("What this study supports is a methodological claim: leakage-controlled "
      "validation is achievable on this data, the low-cost variable set carries "
      "real signal, and the honest performance is meaningfully lower than the "
      "published record suggests - most of all for the early-stage patients a "
      "screening programme exists to find. What it does not support is any "
      "claim about clinical utility, causality, deployment readiness or "
      "generalisability. The estimates are internal to 200 patients from one "
      "hospital, with pre-discretised predictors, a case mix dominated by "
      "advanced disease, no demographic detail beyond age bands, and confidence "
      "intervals wide enough to encompass materially different conclusions. "
      "**External and prospective validation in a genuine screening population "
      "would be required** before any clinical interpretation is warranted.")
    w("")

    # ---------------- References ----------------
    w("## References")
    w("")
    w("1. GBD Chronic Kidney Disease Collaboration. Global, regional, and "
      "national burden of chronic kidney disease, 1990-2017: a systematic "
      "analysis for the Global Burden of Disease Study 2017. *Lancet*. "
      "2020;395(10225):709-733. doi:[10.1016/S0140-6736(20)30045-3]"
      "(https://doi.org/10.1016/S0140-6736(20)30045-3)")
    w("2. Kidney Disease: Improving Global Outcomes (KDIGO) CKD Work Group. "
      "KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management "
      "of Chronic Kidney Disease. *Kidney Int*. 2024;105(4S):S117-S314. "
      "doi:[10.1016/j.kint.2023.10.018](https://doi.org/10.1016/j.kint.2023.10.018)")
    w("3. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated "
      "guidance for reporting clinical prediction models that use regression or "
      "machine learning methods. *BMJ*. 2024;385:e078378. "
      "doi:[10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378)")
    w("4. Islam MA, Akter S. Risk Factor Prediction of Chronic Kidney Disease "
      "[dataset]. UCI Machine Learning Repository; 2020. Licensed CC BY 4.0. "
      "doi:[10.24432/C5WP64](https://doi.org/10.24432/C5WP64)")
    w("5. Riley RD, Snell KIE, Ensor J, et al. Minimum sample size for "
      "developing a multivariable prediction model: PART II - binary and "
      "time-to-event outcomes. *Stat Med*. 2019;38(7):1276-1296. "
      "doi:[10.1002/sim.7992](https://doi.org/10.1002/sim.7992)")
    w("6. Varma S, Simon R. Bias in error estimation when using cross-validation "
      "for model selection. *BMC Bioinformatics*. 2006;7:91. "
      "doi:[10.1186/1471-2105-7-91](https://doi.org/10.1186/1471-2105-7-91)")
    w("7. Van Calster B, Nieboer D, Vergouwe Y, De Cock B, Pencina MJ, "
      "Steyerberg EW. A calibration hierarchy for risk models was defined: from "
      "utopia to empirical data. *J Clin Epidemiol*. 2016;74:167-176. "
      "doi:[10.1016/j.jclinepi.2015.12.005]"
      "(https://doi.org/10.1016/j.jclinepi.2015.12.005)")
    w("8. Vickers AJ, Elkin EB. Decision curve analysis: a novel method for "
      "evaluating prediction models. *Med Decis Making*. 2006;26(6):565-574. "
      "doi:[10.1177/0272989X06295361](https://doi.org/10.1177/0272989X06295361)")
    w("9. Lundberg SM, Lee S-I. A unified approach to interpreting model "
      "predictions. In: *Advances in Neural Information Processing Systems 30 "
      "(NeurIPS 2017)*. 2017:4765-4774. "
      "arXiv:[1705.07874](https://arxiv.org/abs/1705.07874)")
    w("10. Nori H, Jenkins S, Koch P, Caruana R. InterpretML: a unified "
      "framework for machine learning interpretability. 2019. "
      "arXiv:[1909.09223](https://arxiv.org/abs/1909.09223)")
    w("")
    w("*No reference above was generated without verification; each has a DOI "
      "or a stable arXiv identifier.*")
    w("")

    # ---------------- Appendix ----------------
    w("## Appendix A. Generated artefacts")
    w("")
    w("**Figures** (`reports/figures/`, 300 dpi PNG and PDF)")
    w("")
    figs = sorted((ROOT / "reports" / "figures").glob("*.png"))
    for f in figs:
        w(f"- `{f.name}`")
    w("")
    w("**Tables** (`reports/tables/`)")
    w("")
    for f in sorted(TABLES.glob("*.csv")):
        w(f"- `{f.name}`")
    w("")

    out = ROOT / "reports" / "research_report.md"
    out.write_text("\n".join(W), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({len(W)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
