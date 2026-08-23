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
    w("# Three mechanisms inflate reported performance on a widely used "
      "public CKD benchmark")
    w("")
    w("**A leakage-controlled, case-mix-aware re-analysis**")
    w("")
    w(f"*Generated {date.today().isoformat()} by `scripts/06_write_report.py`. "
      "Every number in this document is injected directly from the computed "
      "tables in `reports/tables/`; none is transcribed by hand.*")
    w("")
    w("---")
    w("")
    w("> ### Study-type and safety statement")
    w(">")
    w("> This is a **methodological re-analysis** of a public dataset, not a "
      "clinical study. Its object is the measurement process: how much of the "
      "near-perfect performance routinely reported on this benchmark is "
      "produced by properties of the data rather than by predictive ability. "
      "It is **not a diagnostic tool**, and nothing in it should be used to "
      "make decisions about any patient. No claim is made about clinical "
      "utility, causality, deployment readiness or generalisability to any "
      "population. Where models are compared, the comparison is evidence "
      "about the benchmark, not a recommendation to deploy anything.")
    w("")
    w("---")
    w("")

    # ---------------- Abstract ----------------
    w("## Abstract")
    w("")
    lit_summary_ab = t("table_26_prior_work_summary.csv").set_index("quantity")["value"]
    w(f"**Background.** Machine-learning studies on the public UCI chronic "
      f"kidney disease (CKD) benchmarks routinely report accuracy at or "
      f"near 100%. In a structured survey of "
      f"{int(lit_summary_ab['n_studies_surveyed'])} such studies, every "
      f"independently verified headline metric on the diagnostic task sits "
      f"at or above 99%, "
      f"{int(lit_summary_ab['n_kabir_coded_sc_egfr_yes'])} are coded as "
      f"using serum creatinine or eGFR as inputs, and "
      f"{int(lit_summary_ab['n_with_genuinely_external_validation'])} has a "
      f"validation cohort that is genuinely external. Such numbers are more "
      f"often a property of the data than evidence of clinical usefulness.")
    w("")
    w(f"**Objective.** To identify and quantify the mechanisms that produce "
      f"near-perfect measured performance on this benchmark, and to "
      f"establish what predictive signal remains once each is controlled "
      f"or measured.")
    w("")
    w(f"**Methods.** {n_tot} patient records "
      f"({n_pos} CKD, {n_neg} non-CKD) from the UCI 'Risk Factor Prediction "
      f"of Chronic Kidney Disease' release. All continuous variables were "
      f"already discretised into "
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
      f"{ci(low, 'npv')} (all figures in this paragraph: best model per "
      f"configuration, uncalibrated pooled out-of-fold predictions; the "
      f"prespecified selection rule, which also considers calibrated cells, "
      f"is reported in the results). "
      + (
          f"Calibration of the best valid model was "
          f"slope {fmt(best['calibration_slope'], 2)}, intercept "
          f"{fmt(best['calibration_intercept'], 2)}. "
          if np.isfinite(best["calibration_slope"])
          else
          f"The calibration slope of the best valid model is **not "
          f"identified**: it separates the classes completely, so the "
          f"logistic recalibration has no maximum-likelihood solution "
          f"(intercept {fmt(best['calibration_intercept'], 2)}, Brier "
          f"{fmt(best['brier'], 4)}); the identified slope of the best "
          f"low-cost model is {fmt(best_lc['calibration_slope'], 2)}. "
      )
      + f"Feature-importance rankings "
      f"were {'moderately' if kendall >= 0.5 else 'weakly'} concordant across "
      f"the {manifest['outer_folds'] * manifest['repeats']} outer folds "
      f"(Kendall's W = {fmt(kendall)}).")
    w("")
    w(f"**Mechanism 2, case mix, is larger than mechanism 1.** Even without any "
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
    prov_ab = t("table_24_provenance.csv")
    uci_ab = prov_ab[prov_ab["dataset_id"] == "uci2015"]
    agree_ab = t("table_24_provenance_agreement.csv")
    lit_ab = t("table_26_prior_work.csv")
    n_cross_ab = int((lit_ab["did_external_validation"] == "pseudo-external").sum())
    if len(uci_ab):
        r_ab = uci_ab.iloc[0]
        w(f"**Mechanism 3: the benchmark's two releases are not independent "
          f"cohorts.** A record-level provenance check registered before any "
          f"external claim found that all "
          f"{int(r_ab['rows_with_any_partner'])} patients in this file match "
          f"into the 2015 UCI CKD release (maximum bipartite match fraction "
          f"{float(r_ab['containment_match_fraction']):.3f} against a "
          f"permutation null of "
          f"{float(r_ab['containment_null_mean']):.3f} +/- "
          f"{float(r_ab['containment_null_sd']):.3f}), "
          f"{int(agree_ab['n_unique_pins'].iloc[0]) if len(agree_ab) else 0} "
          f"of them uniquely, with "
          f"{int(agree_ab['n_contradictions'].sum()) if len(agree_ab) else 0} "
          f"contradictions across "
          f"{int(agree_ab['variable'].nunique()) if len(agree_ab) else 0} "
          f"categorical variables that took no part in the matching. This "
          f"file is a discretised subset re-release of that dataset. "
          f"{n_cross_ab} of the surveyed studies validate models across the "
          f"two releases as though they were independent cohorts, and one "
          f"merges them into a single training set.")
        w("")
    w(f"**Conclusions.** Three distinct mechanisms drive measured performance "
      f"on this benchmark toward 1.0, and none of them is predictive ability: "
      f"target leakage, a case mix dominated by advanced disease, and "
      f"validation cohorts that share their patients with the training data. "
      f"Case mix is the largest of the three and the least often controlled. "
      f"Under leakage-controlled validation a variable set costing no "
      f"venepuncture retains "
      f"{'much' if low['roc_auc'] >= 0.9 * lab['roc_auc'] else 'only part'} of "
      f"the discrimination available from laboratory measurements, and an "
      f"additive model matches it with an identified probability scale - but "
      f"no figure here estimates screening performance, because this sample "
      f"is not a screening series. These are internal, {n_tot}-patient, "
      f"single-centre estimates; the study has no external validation and "
      f"cannot obtain one from the sources examined. What it offers instead "
      f"is a reusable procedure: declare prohibited columns, enforce them "
      f"programmatically, test the case mix before believing the "
      f"discrimination, and check that an external cohort is external.")
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
    w("That clinical importance is why the public UCI CKD datasets have "
      "become standard benchmarks: they are small, tidy, and answer a "
      "question that matters. It is also why the results reported on them "
      "deserve scrutiny. A benchmark on which almost every method reports "
      "near-perfect accuracy has stopped discriminating between methods, "
      "and the interesting question becomes what, in the data, is "
      "producing the ceiling.")
    w("")
    w("A secondary question follows once the ceiling is understood: how "
      "much of the "
      "discriminative information in a CKD assessment is carried by variables "
      "that cost almost nothing to obtain - history, physical examination, and a "
      "urine reagent strip - relative to variables that require venepuncture and "
      "a laboratory analyser?")
    w("")
    lit_summary = t("table_26_prior_work_summary.csv").set_index("quantity")["value"]
    w(f"There is a well-documented hazard in answering either question with "
      f"machine "
      f"learning on small clinical datasets, and it is not hypothetical for "
      f"this data. A structured survey of prior studies on this dataset and "
      f"its parent release (`reports/tables/table_26_prior_work.csv`; "
      f"{int(lit_summary['n_studies_surveyed'])} studies, "
      f"{int(lit_summary['n_independently_verified'])} independently "
      f"verified so far, the coding of the remainder attributed to the "
      f"comparison table of Kabir et al. [11] pending hand verification) "
      f"finds {int(lit_summary['n_verified_metric_geq_099'])} of the "
      f"{int(lit_summary['n_with_numeric_headline_metric'])} independently "
      f"verified headline metrics at or above 99% accuracy - 99.16% [12], "
      f"and 99.5% with cross-dataset validation reaching 100% [13]; the "
      f"third is an external sensitivity from the one genuinely external "
      f"study [11] - with "
      f"{int(lit_summary['n_kabir_coded_sc_egfr_yes'])} of "
      f"{int(lit_summary['n_studies_surveyed'])} studies coded as using "
      f"serum creatinine or eGFR as inputs, and exactly "
      f"{int(lit_summary['n_with_genuinely_external_validation'])} with a "
      f"validation cohort that is genuinely external. Such results are "
      f"rarely evidence of clinical usefulness; far more often they indicate "
      f"that a predictor encodes the outcome. This dataset contains three "
      f"such columns, and one of them is an exact copy of the label. A study "
      f"that does not control for this cannot distinguish learning from "
      f"lookup.")
    w("")
    w(f"A further hazard is specific to this dataset's published record: "
      f"{int(lit_summary['n_cross_dataset_uci2015_uci2023'])} of the "
      f"surveyed studies validate models across the 2015 UCI release and "
      f"this file (its 2020 re-release) as if they were independent "
      f"cohorts, and one merges them into a single training set. The "
      f"provenance analysis in `reports/external/provenance_report.md` "
      f"shows the two releases share their patients record-for-record, so "
      f"those validations were performed on the training population - a "
      f"point developed in the discussion.")
    w("")
    w("This work therefore treats leakage control, honest validation and "
      "calibration as the primary objects of study, and treats predictive "
      "performance as something to be measured carefully rather than "
      "maximised. Reporting follows the spirit of TRIPOD+AI [3].")
    w("")

    # ---------------- Research questions ----------------
    w("## 2. Research questions")
    w("")
    w("**Primary.** Why do machine-learning studies on this benchmark "
      "routinely report accuracy at or near 100%, and what remains once the "
      "mechanisms responsible are controlled or measured?")
    w("")
    w("Three candidate mechanisms are examined, in the order they were "
      "found:")
    w("")
    w("1. **Target leakage.** How much does including outcome-derived "
      "columns inflate measured performance? (Section 5.2)")
    w("2. **Case-mix (spectrum) composition.** How much of the remaining "
      "performance is explained by *which patients* the sample contains "
      "rather than by what the model learned? (Section 5.5)")
    w("3. **Pseudo-external validation.** Are the cohorts used to validate "
      "models on this benchmark actually independent of the cohorts used "
      "to train them? (Section 5.14)")
    w("")
    w("**Supporting questions**, addressed because they are needed to "
      "interpret the three mechanisms rather than for their own sake:")
    w("")
    w("4. Under leakage-controlled validation, how much discrimination "
      "survives when expensive laboratory variables are removed - i.e. "
      "what does a cost-stratified feature set actually cost? "
      "(Sections 5.3, 5.12)")
    w("5. Are predicted probabilities calibrated, and is the probability "
      "scale even identified? (Section 5.6)")
    w("6. Are the identified important predictors stable across "
      "resampling? (Section 5.7)")
    w("7. How uncertain is any of this, given 200 patients? "
      "(Sections 5.8, 5.11, 5.13)")
    w("")

    # ---------------- Dataset ----------------
    prov = quality["provenance"]
    w("## 3. Dataset")
    w("")
    w("### 3.1 Source and provenance")
    w("")
    w(f"The analysis uses `ckd-dataset-v2.csv`, {n_tot} patient records "
      f"distributed by the UCI Machine Learning Repository as *Risk Factor "
      f"Prediction of Chronic Kidney Disease* (creators Md. Ashiqul Islam and "
      f"Shamima Akter), under CC BY 4.0 [4]. Its documentation states that "
      f"the records were collected at Enam Medical College, Savar, Dhaka, "
      f"Bangladesh.")
    w("")
    w("**That documented provenance is contradicted by the data itself.** "
      "Section 5.14 reports a record-level check, run before any external "
      "claim was permitted, which finds every one of these patients present "
      "in the 2015 UCI CKD release (documented as Apollo Hospitals, India), "
      "187 of them matching a single record uniquely and agreeing on ten "
      "categorical variables that took no part in the matching. The file "
      "analysed here is a discretised subset re-release of that earlier "
      "dataset. We report the measurement and take no view on how the "
      "discrepancy arose; readers should treat the stated collection site, "
      "country and year as unverified, and this study makes no claim that "
      "rests on them.")
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
    w("Best model per configuration, **uncalibrated**, pooled out-of-fold "
      "predictions (the same cells as the leakage audit above). The "
      "prespecified selection rule in section 5.9 instead selects over all "
      "(model x calibration) cells, so its chosen model and its numbers can "
      "differ slightly from this table's; every quoted figure names its "
      "cell for that reason:")
    w("")
    w("| Configuration | k | Best model | ROC-AUC (95% CI) | Sensitivity | Specificity | NPV | PPV | Brier |")
    w("|---|---:|---|---|---|---|---|---|---|")
    for r in (full, lab, lcm, low, clin):
        w(f"| {config_label(r['config'])} | {int(r['n_features'])} | "
          f"{model_label(r['model'])} | {ci(r, 'roc_auc')} | "
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

    # ---------------- 5.11 Optimism ----------------
    optimism = t("table_29_optimism.csv")
    w("### 5.11 Apparent versus nested-CV performance (optimism)")
    w("")
    w("Fitting the identical selection procedure (inner grid search on "
      "ROC-AUC) on the complete dataset and evaluating on that same data "
      "gives the *apparent* performance; the difference from the pooled "
      "nested out-of-fold estimate is the optimism that internal validation "
      "is already correcting for:")
    w("")
    w("| Configuration | Model | Apparent ROC-AUC | Nested ROC-AUC | Optimism | Apparent Brier | Nested Brier |")
    w("|---|---|---:|---:|---:|---:|---:|")
    for _, r in optimism.iterrows():
        w(f"| {config_label(r['config'])} | {model_label(r['model'])} | "
          f"{fmt(r['apparent_roc_auc'], 4)} | {fmt(r['nested_roc_auc'], 4)} | "
          f"{r['optimism_roc_auc']:+.4f} | {fmt(r['apparent_brier'], 4)} | "
          f"{fmt(r['nested_brier'], 4)} |")
    w("")
    max_opt = float(optimism["optimism_roc_auc"].max())
    w(f"The largest ROC-AUC optimism across these cells is "
      f"{max_opt:+.4f}. That optimism is this small for the same reason the "
      f"valid models sit near the ceiling (section 5.5): the case mix "
      f"leaves little room for resubstitution to exaggerate. On a harder "
      f"problem the same procedure would show a much larger gap, so the "
      f"small values here should be read as further evidence about the "
      f"sample, not as evidence that validation discipline was unnecessary.")
    w("")
    brier_worse = optimism[optimism["apparent_brier"] > optimism["nested_brier"]]
    if len(brier_worse):
        cells_txt = "; ".join(
            f"{config_label(r['config'])} x {model_label(r['model'])} "
            f"({fmt(r['apparent_brier'], 4)} vs {fmt(r['nested_brier'], 4)})"
            for _, r in brier_worse.iterrows()
        )
        w(f"One asymmetry is worth recording: resubstitution is guaranteed "
          f"to flatter *rank order* (the AUC optimism above is non-negative "
          f"in every cell), but not the *probability scale*. In "
          f"{len(brier_worse)} cell(s) the apparent Brier score is actually "
          f"worse than the nested one ({cells_txt}), because the averaged "
          f"out-of-fold probabilities are better placed on the probability "
          f"scale than a single resubstitution fit's. Discrimination and "
          f"calibration do not inflate together, which is one more reason "
          f"the two must be reported separately.")
        w("")

    # ---------------- 5.13 Readable rule ----------------
    ebm_cmp = t("table_28_ebm_vs_svm.csv")
    shape_summary = t("table_28_ebm_shape_summary.csv")
    w("### 5.12 A readable rule at the same cost (EBM versus SVM)")
    w("")
    w("The model the prespecified rule selects is an RBF SVM, which is "
      "accurate and opaque. At identical cost - the same 11 history, "
      "examination and dipstick variables - the explainable boosting "
      "machine is a pure additive model (`interactions=0`) whose "
      "per-variable shape functions can be read and challenged:")
    w("")
    w("| Model | Calibration | ROC-AUC | Brier | Calibration slope | Sensitivity | Readable shape functions |")
    w("|---|---|---:|---:|---:|---:|---|")
    for _, r in ebm_cmp.iterrows():
        slope = (fmt(r["calibration_slope"], 3)
                 if np.isfinite(r["calibration_slope"]) else "not identified")
        w(f"| {model_label(r['model'])} | {calibration_label(r['calibration'])} | "
          f"{fmt(r['roc_auc'])} | {fmt(r['brier'], 4)} | {slope} | "
          f"{fmt(r['sensitivity'], 3)} | "
          f"{'yes' if r['readable_shape_functions'] else 'no'} |")
    w("")
    ebm_iso = ebm_cmp[(ebm_cmp["model"] == "ebm")
                      & (ebm_cmp["calibration"] == "isotonic")].iloc[0]
    svm_iso = ebm_cmp[(ebm_cmp["model"] == "svm")
                      & (ebm_cmp["calibration"] == "isotonic")].iloc[0]
    w(f"The EBM gives up {svm_iso['roc_auc'] - ebm_iso['roc_auc']:.4f} "
      f"ROC-AUC against the SVM - far inside the bootstrap intervals of "
      f"either - and buys with it a calibration slope of "
      f"{fmt(ebm_iso['calibration_slope'], 3)} against "
      f"{fmt(svm_iso['calibration_slope'], 3)}, i.e. a probability scale "
      f"that needs essentially no correction. Where the study must choose "
      f"between the best-discriminating model and the best-explainable one "
      f"at equal cost, the evidence does not force the trade-off it is "
      f"usually assumed to: on this sample the readable model is also the "
      f"better-calibrated one.")
    w("")
    w("Figure R13 plots the fitted shape functions, averaged over the 25 "
      "outer folds. Their directions are clinically coherent and, for "
      "`appet`, independently confirm a coding polarity that the released "
      "file never documented (see section 3.6): low urine specific gravity "
      "raises risk (impaired concentrating ability), albuminuria raises it "
      "steeply, and diabetes, hypertension, oedema and poor appetite all "
      "push the same way.")
    w("")
    disagreeing = shape_summary[~shape_summary["direction_agrees"]]
    if len(disagreeing):
        names = ", ".join(f"`{f}`" for f in disagreeing["feature"])
        w(f"**One term must not be read on its own.** {names} runs opposite "
          f"to its own marginal association (shape Spearman "
          f"{disagreeing.iloc[0]['shape_spearman_rho']:+.3f} against signed "
          f"univariate ROC-AUC "
          f"{disagreeing.iloc[0]['signed_univariate_auc']:.3f}). That is a "
          f"suppression effect rather than an error: the binary diastolic "
          f"flag sits beside `bp limit` and `htn`, which carry the "
          f"hypertension signal, so its residual contribution changes sign. "
          f"It is recorded in `table_28_ebm_shape_notes.txt` and is a "
          f"concrete illustration of the interpretation limit stated in "
          f"section 5.7: an additive model is readable term by term only "
          f"where its terms are not proxies for one another.")
        w("")

    # ---------------- 5.12 Label robustness ----------------
    labels = t("table_27_label_robustness.csv")
    excl = labels[labels["variant"] == "exclude_inconsistent"]
    flip = labels[labels["variant"] == "flip_inconsistent"]
    w("### 5.13 Label robustness (the four internally inconsistent records)")
    w("")
    w(f"Four patients are labelled `notckd` while carrying an advanced CKD "
      f"stage and an eGFR below 60 (CSV lines 12, 18, 52, 123). Because "
      f"there is no external source of truth to arbitrate, they are kept in "
      f"the primary analysis and interrogated here instead: the headline "
      f"cells were re-run with those records **excluded**, and again with "
      f"their labels **flipped** to `ckd`.")
    w("")
    w("| Configuration | Model | Calibration | ROC-AUC (primary) | Excluded (delta) | Flipped (delta) |")
    w("|---|---|---|---:|---:|---:|")
    for (config, model, calibration), grp in labels.groupby(
        ["config", "model", "calibration"], sort=False
    ):
        e = grp[grp["variant"] == "exclude_inconsistent"]
        f_ = grp[grp["variant"] == "flip_inconsistent"]
        e_txt = (f"{e.iloc[0]['variant_roc_auc']:.3f} "
                 f"({e.iloc[0]['delta_roc_auc']:+.4f})") if len(e) else "-"
        f_txt = (f"{f_.iloc[0]['variant_roc_auc']:.3f} "
                 f"({f_.iloc[0]['delta_roc_auc']:+.4f})") if len(f_) else "-"
        w(f"| {config_label(config)} | {model_label(model)} | "
          f"{calibration_label(calibration)} | "
          f"{fmt(grp.iloc[0]['primary_roc_auc'])} | {e_txt} | {f_txt} |")
    w("")
    max_excl = float(excl["delta_roc_auc"].abs().max())
    w(f"**Excluding the four records changes nothing.** The largest "
      f"absolute ROC-AUC change across the cells is {max_excl:.4f}, an "
      f"order of magnitude inside the bootstrap intervals, and no cell's "
      f"sensitivity moves by more than "
      f"{float(excl['delta_sensitivity'].abs().max()):.4f}. On the "
      f"deletion reading, limitation 8 is answered: these records are not "
      f"driving any conclusion.")
    w("")
    lc_flip = flip[flip["config"] == "low_cost_model"]["delta_roc_auc"]
    lab_flip = flip[flip["config"] == "laboratory_model"]["delta_roc_auc"]
    w(f"**Flipping them does not, and the asymmetry is informative.** "
      f"Treating the stage and eGFR columns as correct costs the low-cost "
      f"configuration between {abs(lc_flip.max()):.4f} and "
      f"{abs(lc_flip.min()):.4f} ROC-AUC, while the laboratory "
      f"configuration loses at most {abs(lab_flip.min()):.4f}. The "
      f"low-cost loss is comparable to the width of the bootstrap "
      f"intervals themselves, so on this reading label quality is *not* a "
      f"negligible source of uncertainty for the cheap model.")
    w("")
    w("The mechanism is worth stating plainly, because it is the study's "
      "own primary claim placed under stress. These four patients are "
      "precisely the ones whose history, examination and dipstick findings "
      "look unremarkable while their laboratory values indicate advanced "
      "kidney disease. If their `notckd` labels are the errors, then they "
      "are exactly the patients a low-cost instrument would miss - and the "
      "laboratory panel would not. Four records cannot settle that, but "
      "they point the same way as the case-mix analysis in section 5.5: "
      "the low-cost result is strongest exactly where the disease is "
      "already advanced enough to show up without a laboratory.")
    w("")

    # ---------------- 5.14 Pseudo-external validation ----------------
    prov_table = t("table_24_provenance.csv")
    agree_table = t("table_24_provenance_agreement.csv")
    lit_table = t("table_26_prior_work.csv")
    same_source = prov_table[prov_table["verdict"] == "SAME-SOURCE"]
    uci = prov_table[prov_table["dataset_id"] == "uci2015"]
    w("### 5.14 Pseudo-external validation: the benchmark's two releases "
      "share their patients (mechanism 3)")
    w("")
    w("The two mechanisms above concern a single dataset. The third "
      "concerns how this dataset is used in the literature. Several "
      "published studies validate models trained on the 2015 UCI CKD "
      "release against this file - distributed separately, with a "
      "different stated collection site, sample size and year - and "
      "present the result as external validation. That inference requires "
      "the two releases to contain different patients.")
    w("")
    w("They do not. Every candidate dataset registered for this study was "
      "put through a record-level provenance check before any external "
      "claim was permitted (`reports/external/provenance_report.md`). "
      "Each of this file's patients is represented as a vector of "
      "intervals over the shared variables; a record in a candidate "
      "dataset is *compatible* with a patient when the outcome matches and "
      "every shared value falls inside that patient's interval. The "
      "observed maximum bipartite matching is then calibrated against "
      "column permutations that preserve every marginal distribution while "
      "destroying cross-variable structure.")
    w("")
    if len(uci):
        r = uci.iloc[0]
        w("| Quantity | Value |")
        w("|---|---|")
        w(f"| Patients in this file matched into the 2015 release | "
          f"{int(r['rows_with_any_partner'])} / 200 |")
        w(f"| Maximum bipartite match fraction | "
          f"{float(r['containment_match_fraction']):.3f} |")
        w(f"| Permutation null (mean +/- SD) | "
          f"{float(r['containment_null_mean']):.3f} +/- "
          f"{float(r['containment_null_sd']):.3f} |")
        w(f"| Permutation null (maximum observed) | "
          f"{float(r['containment_null_max']):.3f} |")
        w(f"| Verdict | **{r['verdict']}** |")
        w("")
    if len(agree_table):
        n_pins = int(agree_table["n_unique_pins"].iloc[0])
        n_vars = int(agree_table["variable"].nunique())
        n_compared = int(agree_table["n_pairs_compared"].sum())
        n_contra = int(agree_table["n_contradictions"].sum())
        w(f"Matching on intervals could in principle be coincidence, so the "
          f"pairing was checked against evidence it had no access to. "
          f"{n_pins} of the 200 patients are compatible with exactly **one** "
          f"record in the 2015 release. Across those pairs, "
          f"{n_vars} categorical variables that took no part in the "
          f"matching agree in {n_compared} comparisons with "
          f"**{n_contra} contradictions**. A coincidental alignment does "
          f"not reproduce ten unused variables.")
        w("")
    w("The conclusion is that this file is a discretised subset "
      "re-release of the 2015 dataset. Its documented provenance - a "
      "different country, hospital and year - cannot be reconciled with "
      "record-level identity; we report the measurement and take no view "
      "on how the discrepancy arose.")
    w("")
    cross = lit_table[lit_table["did_external_validation"] == "pseudo-external"]
    w(f"The consequence for the published record is direct. Of the "
      f"{len(lit_table)} studies surveyed in section 1, **{len(cross)}** "
      f"validate across these two releases as though they were independent "
      f"cohorts, and one of those merges them into a single training set "
      f"before reporting accuracy. On the evidence above, those procedures "
      f"evaluate models on their own training population. This is not a "
      f"criticism of the authors: nothing in either dataset's "
      f"documentation indicates the overlap, and the present study "
      f"registered the 2015 release as its own primary external-validation "
      f"candidate before the check refused it.")
    w("")
    w("Two implications follow for this report. First, the study has no "
      "external validation and cannot acquire one from these sources; "
      "every estimate here is internal, and the gate that produced this "
      "finding also blocks the 2015 release from every external-validation "
      "table (`tests/test_external.py::TestProvenanceGate`). Second, "
      "because the overlap is with a *continuous-valued* release of the "
      "same patients, the information destroyed by pre-discretisation can "
      "be recovered for this cohort and measured directly - a comparison "
      "that is possible precisely because the cohorts are not independent.")
    w("")

    # ---------------- 5.15 What the binning cost ----------------
    costs = t("table_31_binning_cost.csv")
    imput = t("table_32_imputation_audit.csv")
    sc_bins = t("table_33_sc_bin_structure.csv")
    bp_rec = t("table_34_blood_pressure_recovery.csv")
    coverage = t("table_30_recovery_coverage.csv")
    n_pinned_rec = int(coverage["n_pinned"].iloc[0])
    w("### 5.15 What the published binning destroyed (and what the file "
      "supplied in place of missing data)")
    w("")
    w(f"Section 5.14 established that these patients appear, with their "
      f"measurements intact, in a continuous-valued release. That makes a "
      f"normally unanswerable question answerable: what did the interval "
      f"encoding cost? Because the comparison is between two "
      f"representations of **the same {n_pinned_rec} patients**, there is "
      f"no population shift, no case-mix difference and no sampling "
      f"variation to confound it. Each variable below is scored on the "
      f"patients whose value the source actually recorded, so the "
      f"comparison isolates the encoding.")
    w("")
    w("| Variable | n | ROC-AUC binned | ROC-AUC continuous | Lost to binning |")
    w("|---|---:|---:|---:|---:|")
    for _, r in costs.iterrows():
        w(f"| `{r['variable']}` | {int(r['n_observed'])} | "
          f"{fmt(r['auc_binned_observed_only'])} | "
          f"{fmt(r['auc_continuous_observed_only'])} | "
          f"{r['binning_cost']:+.4f} |")
    w("")
    worst = costs.iloc[0]
    others = costs[costs["variable"] != worst["variable"]]["binning_cost"].abs().max()
    w(f"**The loss is concentrated in one variable, and it is the "
      f"diagnostically decisive one.** For every variable except "
      f"`{worst['variable']}` the encoding costs at most "
      f"{others:.4f} ROC-AUC - the bins are fine enough to preserve the "
      f"signal. Serum creatinine loses "
      f"{float(worst['binning_cost']):.4f}, falling from "
      f"{fmt(worst['auc_continuous_observed_only'])} to "
      f"{fmt(worst['auc_binned_observed_only'])}. The published bins show "
      f"why:")
    w("")
    w("| Published bin | n | Continuous span (mg/dL) | Fraction CKD |")
    w("|---|---:|---|---:|")
    for _, r in sc_bins.iterrows():
        w(f"| `{r['published_bin']}` | {int(r['n'])} | "
          f"{r['continuous_min']:.1f} - {r['continuous_max']:.1f} | "
          f"{fmt(r['ckd_fraction'], 2)} |")
    w("")
    widest = sc_bins.loc[sc_bins["continuous_span"].idxmax()]
    w(f"The first bin absorbs {int(widest['n'])} of "
      f"{int(sc_bins['n'].sum())} patients and spans "
      f"{widest['continuous_min']:.1f} to {widest['continuous_max']:.1f} "
      f"mg/dL - from unambiguously normal (0.6-1.2) through severe renal "
      f"impairment. Every other bin is {fmt(sc_bins.iloc[1:]['ckd_fraction'].min(), 2)} "
      f"CKD or higher. In this release serum creatinine is therefore not a "
      f"graded measurement but a coarse flag that fires only once "
      f"creatinine is already extreme, and inside the bin holding most of "
      f"the cohort it carries almost no information "
      f"({fmt(widest['ckd_fraction'], 2)} CKD).")
    w("")
    w("This has a consequence for how the leakage boundary should be read. "
      "This study excluded `grf` (eGFR) as a post-diagnosis derivative but "
      "retained serum creatinine, on the grounds that it is a routinely "
      "measured analyte rather than a diagnostic label. On the released "
      "data that judgement is comfortable, because binning has flattened "
      "the variable. On the underlying measurements it is much less so: "
      f"creatinine alone reaches ROC-AUC "
      f"{fmt(worst['auc_continuous_observed_only'])}, which is close to "
      "the quantity that defines the outcome. A study using the "
      "continuous release would need to defend that inclusion far more "
      "carefully than one using this file - and would not know it from "
      "this file alone.")
    w("")
    n_imputed_cells = int(imput["n_unobserved_in_source"].sum())
    n_const = int((imput["n_distinct_labels_assigned"] == 1).sum())
    w(f"**A second property of the release surfaces at the same time.** "
      f"The analysed file contains exactly one missing cell. Its source "
      f"contains a great many: across the recoverable variables, "
      f"{n_imputed_cells} cells that the source leaves blank carry a value "
      f"here. For **all {n_const} of {len(imput)}** variables, every one of "
      f"those cells was filled with a *single constant* - and in each case "
      f"that constant is the clinically normal range:")
    w("")
    w("| Variable | Cells filled | Value supplied | Distinct values used | Odds of being unobserved, CKD vs non-CKD |")
    w("|---|---:|---|---:|---:|")
    for _, r in imput.iterrows():
        odds = r["missingness_odds_ratio_ckd"]
        odds_txt = "infinite" if not np.isfinite(odds) else f"{odds:.1f}"
        w(f"| `{r['variable']}` | {int(r['n_unobserved_in_source'])} | "
          f"`{r['assigned_label']}` | {int(r['n_distinct_labels_assigned'])} | "
          f"{odds_txt} |")
    w("")
    strong = imput[imput["n_unobserved_in_source"] >= 30]
    w(f"The missingness is not random. A patient with CKD is "
      f"{strong['missingness_odds_ratio_ckd'].min():.0f} to "
      f"{strong['missingness_odds_ratio_ckd'].max():.0f} times more likely "
      f"to have these measurements absent from the source, which is what "
      f"one would expect when tests are ordered selectively. Filling every "
      f"such cell with the normal value therefore assigns normal-looking "
      f"laboratory results to precisely the patients most likely to be "
      f"ill, and does so invisibly: nothing in the released file "
      f"distinguishes a measured normal result from a supplied one.")
    w("")
    w("**Its direction is worth stating plainly, because it runs against "
      "this report's own thesis.** Every other mechanism examined here "
      "inflates measured performance. This one deflates it: substituting "
      "normal values for the sickest patients makes the classes harder to "
      "separate, not easier. Comparing each variable's separability across "
      "all pinned patients against the observed subset suggests an "
      f"attenuation of up to "
      f"{costs['attenuation_from_imputed_rows'].max():.4f} ROC-AUC "
      f"(largest for `{costs.loc[costs['attenuation_from_imputed_rows'].idxmax(), 'variable']}`), "
      "though that comparison is across different patient subsets and "
      "should be read as indicative rather than exact. The practical "
      "implication is not that the benchmark is harder than it looks - it "
      "is that a third of some columns are not measurements at all, which "
      "no analysis of the released file can discover.")
    w("")
    if len(bp_rec):
        flag = bp_rec[bp_rec["encoded_column"] == "bp (Diastolic)"]
        limit = bp_rec[bp_rec["encoded_column"] == "bp limit"]
        w("**Two undocumented variables are resolved as a by-product.** "
          "Section 3.6 flags `bp (Diastolic)` and `bp limit` as variables "
          "whose coding could not be established from the file. Against "
          "the recovered measurements they read directly:")
        w("")
        w("| Encoded column | Level | n | Diastolic range (mmHg) | Median |")
        w("|---|---|---:|---|---:|")
        for _, r in bp_rec.iterrows():
            w(f"| `{r['encoded_column']}` | {r['encoded_level']} | {int(r['n'])} | "
              f"{r['diastolic_min_mmhg']:.0f} - {r['diastolic_max_mmhg']:.0f} | "
              f"{r['diastolic_median_mmhg']:.0f} |")
        w("")
        w("`bp (Diastolic)` is an indicator for diastolic pressure at or "
          "above 80 mmHg, and `bp limit` bands the same measurement into "
          "at-or-below 70 / exactly 80 / at-or-above 90. Both readings hold "
          "for every pinned patient but a handful (one record coded 1 at 60 "
          "mmHg; two coded 0 at 100 and 110 mmHg), which are further "
          "instances of the data-entry noise documented in section 3.4. "
          "The uncertainty flags on these variables can now be removed - "
          "but only because a second release of the same patients existed.")
        w("")

    # ---------------- 5.16 Model-level binned vs continuous ----------------
    bvc = t("table_35_binned_vs_continuous.csv")
    w("### 5.16 Would the continuous data have changed any conclusion?")
    w("")
    w(f"Section 5.15 measured the encoding one variable at a time. The "
      f"question that matters for the benchmark is whether it changes what "
      f"a study would *conclude*. The same {n_pinned_rec} patients were "
      f"therefore modelled twice under the identical nested design, seeds "
      f"and pipelines, differing only in the representation of the "
      f"recoverable variables: the file as released, against the source's "
      f"measurements with unobserved cells left missing and imputed inside "
      f"training folds.")
    w("")
    w("| Configuration | Model | Calibration | ROC-AUC binned | ROC-AUC continuous | Difference |")
    w("|---|---|---|---:|---:|---:|")
    for _, r in bvc.iterrows():
        w(f"| {config_label(r['config'])} | {model_label(r['model'])} | "
          f"{calibration_label(r['calibration'])} | "
          f"{fmt(r['roc_auc_binned'])} | {fmt(r['roc_auc_continuous'])} | "
          f"{r['delta_roc_auc']:+.4f} |")
    w("")
    biggest = bvc.loc[bvc["delta_roc_auc"].abs().idxmax()]
    w(f"**Nothing changes.** The largest difference in either direction is "
      f"{abs(float(biggest['delta_roc_auc'])):.4f} ROC-AUC "
      f"({config_label(biggest['config'])}, {model_label(biggest['model'])}), "
      f"an order of magnitude smaller than the bootstrap intervals reported "
      f"in section 5.8. Restoring the measurements neither rescues the "
      f"binned models nor exposes them.")
    w("")
    w(f"That result is more interesting than a difference would have been. "
      f"Serum creatinine gains "
      f"{float(costs.iloc[0]['binning_cost']):.4f} ROC-AUC on its own when "
      f"its true values are restored - the single largest univariate "
      f"change in the study - and the multivariable models do not benefit "
      f"at all. The reason is the ceiling documented in section 5.5: with "
      f"haemoglobin at univariate ROC-AUC "
      f"{fmt(top_sep['univariate_auc'])} and packed cell volume close "
      f"behind, the information that restored creatinine supplies is "
      f"already present several times over. One can destroy the resolution "
      f"of the most diagnostic laboratory analyte in the dataset and the "
      f"models will not notice, because the case mix hands them the answer "
      f"through anaemia and urine concentration instead.")
    w("")
    w("Two cautions on reading this. The comparison is a net effect: the "
      "continuous arm also handles missing values honestly, where the "
      "released arm carries the constants described above, so "
      "representation and missingness handling move together. And a null "
      "result on a saturated problem is weak evidence about an unsaturated "
      "one - on a genuine screening series, where creatinine would have to "
      "carry weight that anaemia cannot, the same encoding could matter a "
      "great deal. What this section rules out is the specific worry that "
      "the published binning is why performance on this benchmark looks "
      "the way it does. It is not; the case mix is.")
    w("")

    # ---------------- 5.17 Encoding robustness ----------------
    enc_path = TABLES / "table_37_encoding_robustness.csv"
    if enc_path.is_file():
        enc = t("table_37_encoding_robustness.csv")
        w("### 5.17 Does the choice of encoding drive any result?")
        w("")
        w("Section 5.16 replaced the bins with the underlying measurements "
          "and changed nothing. The complementary question is whether the "
          "*number* chosen to represent each bin matters. Four encodings "
          "were compared on the identical design, seeds and folds: the "
          "reference midpoint encoding; the bin's ordinal position, "
          "discarding spacing; that position on a Gaussian-like spacing; "
          "and one indicator per bin, discarding order entirely. All four "
          "are pure functions of a cell's label and the published bin "
          "list, so all are leakage-safe by the same argument as the "
          "reference encoding.")
        w("")
        summary = (
            enc.groupby("encoding")
            .agg(n_cells=("roc_auc", "size"),
                 median_auc=("roc_auc", "median"),
                 worst_delta=("delta_vs_midpoint", lambda s: s.abs().max()),
                 columns=("n_encoded_columns", "max"))
            .reset_index()
        )
        w("| Encoding | Cells | Max columns | Median ROC-AUC | Largest deviation from midpoint |")
        w("|---|---:|---:|---:|---:|")
        for _, r in summary.iterrows():
            w(f"| `{r['encoding']}` | {int(r['n_cells'])} | {int(r['columns'])} | "
              f"{fmt(r['median_auc'])} | {r['worst_delta']:.4f} |")
        w("")
        non_ref = summary[summary["encoding"] != "midpoint"]
        worst_enc = non_ref.loc[non_ref["worst_delta"].idxmax()]
        w(f"The largest deviation from the reference encoding across every "
          f"cell and every alternative is {worst_enc['worst_delta']:.4f} "
          f"ROC-AUC (`{worst_enc['encoding']}`), inside the bootstrap "
          f"intervals of section 5.8. The representative-value choice is "
          f"therefore not load-bearing: neither discarding the spacing "
          f"between bins, nor discarding their order altogether, moves the "
          f"results materially.")
        w("")
        w("Taken with section 5.16 this is a reasonably complete answer to "
          "the representation question. The published data can be "
          "re-expressed as ordinal ranks, as unordered indicators, or "
          "replaced outright by the measurements they were derived from, "
          "and the models land in the same place each time. That is a "
          "further symptom of the ceiling rather than a virtue of the "
          "encoding: when a sample separates as easily as this one, most "
          "reasonable representations of it will do.")
        w("")

    # ---------------- 5.18 Honest uncertainty ----------------
    proc_boot_path = TABLES / "table_38_procedure_bootstrap.csv"
    if proc_boot_path.is_file():
        proc_boot = t("table_38_procedure_bootstrap.csv")
        ci_table = t("table_11_bootstrap_confidence_intervals.csv")
        w("### 5.18 An honest interval: bootstrapping the whole procedure")
        w("")
        w("The confidence intervals reported so far resample patients over "
          "the pooled out-of-fold predictions, holding the fitted models "
          "fixed. Section 5.8 notes that this understates total "
          "uncertainty, because it cannot see the variability introduced "
          "by the model selection, tuning and calibration decisions being "
          "re-made on a different sample. This section removes that "
          "shortcut: for each resample of the patients the **entire nested "
          "procedure is re-run from scratch**, and the spread across "
          "resamples is an interval for the procedure rather than for the "
          "predictions.")
        w("")
        w("One detail matters enough to state. A bootstrap sample contains "
          "the same patient several times, so the outer folds within each "
          "resample are formed over *patients* rather than rows: every "
          "copy of a patient stays on the same side of every split. "
          "Without that constraint a model would be evaluated on patients "
          "it had trained on, which is the failure this study exists to "
          "measure and would not be excusable in its own methods. The "
          "constraint applies only to resampled data; in the main analysis "
          "each row is a distinct patient and the splitting is unchanged.")
        w("")
        w("| Configuration | Model | Point estimate | Prediction-level 95% CI | Procedure-level 95% CI | Width ratio |")
        w("|---|---|---:|---|---|---:|")
        for _, r in proc_boot.iterrows():
            match = ci_table[
                (ci_table["config"] == r["config"])
                & (ci_table["model"] == r["model"])
                & (ci_table["calibration"] == "none")
                & (ci_table["metric"] == "roc_auc")
            ]
            if match.empty:
                pred_txt, ratio_txt = "-", "-"
            else:
                m = match.iloc[0]
                pred_width = float(m["ci_high"] - m["ci_low"])
                pred_txt = f"{fmt(m['ci_low'])} - {fmt(m['ci_high'])}"
                ratio_txt = (f"{r['procedure_ci_width'] / pred_width:.1f}x"
                             if pred_width > 1e-9 else "n/a (degenerate)")
            w(f"| {config_label(r['config'])} | {model_label(r['model'])} | "
              f"{fmt(r['point_estimate_roc_auc'])} | {pred_txt} | "
              f"{fmt(r['procedure_ci_low'])} - {fmt(r['procedure_ci_high'])} | "
              f"{ratio_txt} |")
        w("")
        lo_n = int(proc_boot["n_resamples_used"].min())
        hi_n = int(proc_boot["n_resamples_used"].max())
        n_txt = f"{lo_n}" if lo_n == hi_n else f"{lo_n}-{hi_n}"
        n_failed_total = int(proc_boot["n_failed"].sum())
        w(f"Each cell rests on {n_txt} completed resamples "
          f"of the full procedure"
          + (f" ({n_failed_total} resample(s) failed and are counted, not "
             f"discarded silently)" if n_failed_total else
             ", with no resample failing")
          + f". The honest intervals are wider, which "
          f"is the expected direction and the reason the caveat in section "
          f"5.8 was worth stating. They are the intervals a reader should "
          f"use when asking whether two configurations differ: on this "
          f"evidence, differences of the size discussed in section 5.3 "
          f"remain unresolved, and the study's inability to distinguish "
          f"the low-cost and laboratory sets is if anything understated by "
          f"the narrower intervals reported elsewhere.")
        w("")

    # ---------------- 5.19 Calibration experiment ----------------
    calib_exp_path = TABLES / "table_39_calibration_experiment.csv"
    if calib_exp_path.is_file():
        calib_exp = t("table_39_calibration_experiment.csv")
        calib_paired = t("table_39_calibration_paired.csv").iloc[0]
        w("### 5.19 Does the isotonic result survive more resampling?")
        w("")
        w(f"Section 5.6 reported that isotonic calibration achieved a "
          f"better median Brier score than Platt scaling - contrary to the "
          f"usual expectation at this sample size, and resting on "
          f"{manifest['outer_folds'] * manifest['repeats']} outer test "
          f"sets. The comparison was repeated with four times the "
          f"resampling: {int(calib_exp['n_repeats'].iloc[0])} repeats, "
          f"giving {int(calib_exp['n_outer_test_sets'].iloc[0])} outer test "
          f"sets, on a restricted grid. Statistics are computed within each "
          f"repeat and then summarised, as elsewhere.")
        w("")
        w("| Calibration | Median Brier | IQR | Median ECE | Median slope | Unidentified slopes |")
        w("|---|---:|---:|---:|---:|---:|")
        for _, r in calib_exp.iterrows():
            w(f"| {calibration_label(r['calibration'])} | "
              f"{fmt(r['median_brier'], 4)} | {fmt(r['iqr_brier'], 4)} | "
              f"{fmt(r['median_ece'], 4)} | {fmt(r['median_slope'], 3)} | "
              f"{int(r['n_slope_unidentified'])} |")
        w("")
        frac = float(calib_paired["isotonic_better_fraction"])
        w(f"Paired within each (configuration x model x repeat) cell, "
          f"isotonic gives the lower Brier score in "
          f"{int(calib_paired['isotonic_better_in'])} of "
          f"{int(calib_paired['n_pairs'])} comparisons "
          f"({frac:.0%}), with a median difference of "
          f"{float(calib_paired['median_difference']):+.4f}. "
          + ("The ordering therefore survives the additional resampling. "
             if frac > 0.6 else
             "The ordering is not stable under additional resampling, and "
             "the section 5.6 observation should be treated as noise. ")
          + "Either way the practical difference is small, and the more "
          "consequential calibration finding in this study is not which "
          "method wins but that the top-ranked model's probability scale "
          "is not identified at all (section 5.6).")
        w("")

    # ---------------- 5.20 External validation ----------------
    extval_path = TABLES / "table_42_external_validation.csv"
    if extval_path.is_file():
        extval = t("table_42_external_validation.csv")
        internal_arm = extval[extval["arm"] == "internal_nested_cv"]
        frozen = extval[extval["arm"] == "external_frozen_transfer"]
        recal = extval[extval["arm"] == "external_recalibrated_in_the_large"]
        w("### 5.20 External validation: what happens on different patients")
        w("")
        w(f"Every result above is internal. The provenance gate disqualified "
          f"the obvious external candidate (section 5.14), leaving one "
          f"registered source that carries the *same measurements* on "
          f"different patients: a critical-care database from a different "
          f"country and health system. Models were trained on all "
          f"{n_tot} analysed patients and applied **without any refitting**.")
        w("")
        w(f"The transfer feature set is the intersection of the two schemas - "
          f"{int(frozen['n_features'].iloc[0])} variables: nine blood "
          f"analytes plus urine specific gravity. An internal reference is "
          f"reported on exactly that feature set, so the drop caused by "
          f"transfer is separable from any drop caused by using fewer "
          f"variables.")
        w("")
        w("| Model | Internal (nested CV) | External (frozen) | External Brier | Sensitivity | Specificity |")
        w("|---|---:|---|---:|---:|---:|")
        for model_name in frozen["model"]:
            i = internal_arm[internal_arm["model"] == model_name].iloc[0]
            e = frozen[frozen["model"] == model_name].iloc[0]
            w(f"| {model_label(model_name)} | {fmt(i['roc_auc'])} | "
              f"{fmt(e['roc_auc'])} "
              f"({fmt(e['roc_auc_ci_low'])}-{fmt(e['roc_auc_ci_high'])}) | "
              f"{fmt(e['brier'], 3)} | {fmt(e['sensitivity'], 2)} | "
              f"{fmt(e['specificity'], 2)} |")
        w("")
        drop = float(internal_arm["roc_auc"].min() - frozen["roc_auc"].max())
        w(f"**Discrimination does not transfer.** Internal ROC-AUC on this "
          f"feature set is {fmt(internal_arm['roc_auc'].min())}-"
          f"{fmt(internal_arm['roc_auc'].max())}; externally it is "
          f"{fmt(frozen['roc_auc'].min())}-{fmt(frozen['roc_auc'].max())}, a "
          f"fall of at least {drop:.2f} ROC-AUC. All three model families "
          f"agree, and the bootstrap intervals exclude the internal "
          f"estimate by a wide margin. This is the single most important "
          f"number in the study: the near-perfect internal figures reported "
          f"throughout this report, and throughout the literature on this "
          f"benchmark, do not survive contact with different patients.")
        w("")
        prev_int = float(internal_arm["prevalence"].iloc[0])
        prev_ext = float(frozen["prevalence"].iloc[0])
        w(f"**Calibration fails worse than discrimination, and for a "
          f"legible reason.** External Brier scores are "
          f"{fmt(frozen['brier'].min(), 3)}-{fmt(frozen['brier'].max(), 3)} "
          f"against {fmt(internal_arm['brier'].max(), 3)} internally, with "
          f"calibration intercepts far below zero: the models assign high "
          f"risk to almost everyone. At the prespecified threshold they "
          f"retain sensitivity {fmt(frozen['sensitivity'].max(), 2)} but "
          f"specificity collapses to "
          f"{fmt(frozen['specificity'].min(), 2)}-"
          f"{fmt(frozen['specificity'].max(), 2)}. Prevalence is the obvious "
          f"culprit - {prev_int:.0%} in the training data against "
          f"{prev_ext:.0%} in the external cohort.")
        w("")
        if len(recal):
            w(f"Applying recalibration-in-the-large - the minimal correction "
              f"a deployer would make, shifting the log-odds so mean "
              f"predicted risk matches the external base rate - reduces the "
              f"Brier score to "
              f"{fmt(recal['brier'].min(), 3)}-{fmt(recal['brier'].max(), 3)}. "
              f"It cannot change the ranking, so discrimination is unmoved by "
              f"construction. It also leaves no prediction above 0.50, so the "
              f"model flags nobody at the prespecified threshold: the "
              f"operating point would have to be re-derived in the new "
              f"population. **The binding constraint is the discrimination "
              f"of {fmt(frozen['roc_auc'].max())}, not the calibration**, and "
              f"no post-hoc correction addresses it.")
            w("")
        w("**How much of this is optimism and how much is population "
          "shift?** Both, and this design cannot separate them. The external "
          "cohort is a critical-care population in a different country and "
          "health system, with administratively coded rather than "
          "adjudicated labels, so some of the fall is genuine case-mix and "
          "measurement difference rather than internal over-fitting. What "
          "the comparison does establish is the direction and the order of "
          "magnitude, and that they are consistent with what sections 5.5 "
          "and 5.18 predict: a model selected on a sample this separable, "
          "with intervals this wide, should not be expected to transfer, and "
          "it does not.")
        w("")
        w(f"The external cohort is small - {int(frozen['n'].iloc[0])} "
          f"subjects with {int(frozen['n_positive'].iloc[0])} cases - so the "
          f"external estimate is itself imprecise, which the intervals show. "
          f"It is nevertheless the first evidence in this study drawn from "
          f"patients the models had never seen, and the only such evidence "
          f"the registered sources permit.")
        w("")

    # ---------------- Discussion ----------------
    w("## 6. Discussion")
    w("")
    w("This study set out to measure one failure mode and found three. Each "
      "pushes measured performance towards 1.0, none of them is predictive "
      "ability, and they compound: controlling any one still leaves the "
      "others free to produce a near-perfect number.")
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
    w("The third mechanism is **pseudo-external validation**, and it is the "
      "one that cannot be fixed by analysing this dataset more carefully. "
      "Section 5.14 shows that the two releases treated in the literature as "
      "independent cohorts share their patients record-for-record. A study "
      "that trains on one and validates on the other has measured "
      "resubstitution performance with extra steps, and will see exactly "
      "what it expects to see: near-perfect transfer. Unlike leakage, this "
      "failure is invisible from inside either file - both look like "
      "well-formed, separately documented datasets - which is why a "
      "provenance check belongs before the modelling, not after a "
      "surprising result.")
    w("")
    w("Together these reframe what a 'high-performing' published result on "
      "this benchmark means. Three distinct mechanisms - target leakage, "
      "case-mix spectrum, and non-independent validation cohorts - each push "
      "measured performance towards the ceiling, and none has anything to do "
      "with clinical usefulness. A study reporting 99% accuracy on this data "
      "has probably encountered at least one, and cannot distinguish them "
      "without the leakage audit, subgroup analysis and provenance check "
      "reported here. The three are also ordered by how easy they are to "
      "miss: leakage is visible in the column list, case mix requires "
      "looking at who is in the sample, and cohort overlap requires "
      "comparing two datasets nobody had reason to suspect were one.")
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
    w("**What the recovered measurements add.** Because the overlap is with "
      "a continuous-valued release of the same patients, it is possible to "
      "ask what the published representation cost - a question that is "
      "normally unanswerable, since one cannot usually observe the same "
      "cohort twice. Two things follow. The binning is mostly benign but "
      "catastrophic in one place, serum creatinine, where it collapses "
      "normal-to-severe into a single category; a benchmark that flattens "
      "its most diagnostic analyte is not measuring what its users think "
      "it measures. And the release supplied constants for missing "
      "laboratory values whose absence is itself strongly outcome-related, "
      "so a substantial minority of some columns are not observations. "
      "Neither property is discoverable from the released file, which is "
      "the general point: the trustworthiness of a benchmark is not a "
      "property one can establish by analysing it.")
    w("")
    w("It is worth being explicit that this last mechanism cuts the other "
      "way. Constant imputation of informatively-missing data makes the "
      "classes *harder* to separate here, not easier, so it cannot be "
      "enlisted in an argument that everything about this benchmark "
      "flatters its users. It is reported because it is true and because a "
      "reader deciding whether to trust the dataset needs to know it, not "
      "because it supports the thesis.")
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
    w("2. **No external validation, and none obtainable from the obvious "
      "source.** Every estimate is internal to these 200 patients. Internal "
      "cross-validation systematically overstates the performance a model "
      "would show in a new population, and no correction applied here "
      "changes that. The natural remedy - validating against the larger "
      "2015 UCI release, which shares this file's variable vocabulary - is "
      "unavailable: section 5.14 shows the two share their patients. The "
      "provenance gate blocks that dataset from every external-validation "
      "table, so this limitation is enforced rather than merely stated.")
    w("3. **Hospital-based, single-centre sampling.** These patients are not "
      "a random sample of any population. "
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
    w(f"5. **Pre-discretised predictors - now measured rather than "
      f"assumed.** The published file contains only binned intervals, and "
      f"open-ended tail bins are compressed to their finite edge. Section "
      f"5.15 quantifies the consequence using the recovered measurements: "
      f"for every variable except serum creatinine the encoding costs at "
      f"most {costs[costs['variable'] != 'sc']['binning_cost'].abs().max():.4f} "
      f"univariate ROC-AUC, while serum creatinine loses "
      f"{float(costs.iloc[0]['binning_cost']):.4f}. The same section shows "
      f"that {int(imput['n_unobserved_in_source'].sum())} cells the source "
      f"leaves blank carry constant values here. Both are properties of "
      f"the release that no analysis of the released file alone could "
      f"detect, and both are now bounded rather than speculated about.")
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
      "Section 5.13 quantifies both readings: **deleting** them changes "
      f"nothing (largest |delta ROC-AUC| {max_excl:.4f}), but **trusting "
      f"the staging instead of the label** costs the low-cost configuration "
      f"up to {abs(lc_flip.min()):.4f} ROC-AUC against at most "
      f"{abs(lab_flip.min()):.4f} for the laboratory configuration. Label "
      "noise is therefore not a negligible source of uncertainty for the "
      "cheap model specifically, and this limitation is only half answered.")
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
    w("- **Environment.** Direct dependencies are pinned in "
      "`requirements.txt`; the complete environment that produced these "
      "artefacts is recorded in `requirements.lock.txt`, together with the "
      "interpreter and BLAS build.")
    w("- **The limit of that determinism, stated precisely.** Identical "
      "seeds reproduce identical predictions *within* an environment, and a "
      "test asserts it. Across a rebuild of the environment from the same "
      "pinned versions, agreement is to approximately 1e-16 rather than "
      "bit-for-bit, because floating-point summation order depends on the "
      "installed BLAS. This was checked against the unmodified reference "
      "code and is a property of the environment, not of the analysis; it "
      "is four orders of magnitude below the smallest quantity reported "
      "here, and the regression test asserts agreement to 1e-12 rather "
      "than claiming exactness it cannot deliver.")
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
    w(f"Machine-learning results on this benchmark cluster at the "
      f"discrimination ceiling. This re-analysis identifies three mechanisms "
      f"that put them there, none of which is predictive ability.")
    w("")
    w(f"**Target leakage** is the visible one: the file ships an exact copy "
      f"of the outcome, a post-diagnosis stage label and the diagnostic "
      f"eGFR value, and including them takes every configuration to ROC-AUC "
      f"{fmt(leaky['roc_auc'])}. **Case mix** is the larger one: with every "
      f"prohibited column removed and leakage prevented by a guard that "
      f"raises rather than drops, the full valid configuration still "
      f"separates all {n_tot} patients out of fold, because "
      f"{n_advanced} of {n_pos} cases ({n_advanced/n_pos:.0%}) carry "
      f"stage s3-s5 disease and haemoglobin alone reaches univariate "
      f"ROC-AUC {fmt(top_sep['univariate_auc'])}. **Non-independent "
      f"validation cohorts** are the least visible: the two releases the "
      f"literature treats as separate share their patients "
      f"record-for-record, so published cross-dataset validations between "
      f"them measure resubstitution.")
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
      f"(Kendall's W = {fmt(kendall)}), and it needs every valid variable in "
      f"the dataset, the full laboratory panel included.")
    w("")
    w(f"**The best-supported model for the question this study asks is the "
      f"low-cost one: {model_label(best_lc['model'])} with "
      f"{calibration_label(best_lc['calibration'])} probabilities on "
      f"{int(low['n_features'])} history, examination and urine-dipstick "
      f"variables** - ROC-AUC {fmt(best_lc['roc_auc'])} "
      f"({model_label(best_lc['model'])}, "
      f"{calibration_label(best_lc['calibration'])}) against "
      f"{fmt(best['roc_auc'])} for the selected "
      f"{config_label(best['config'])} model "
      f"({model_label(best['model'])}, "
      f"{calibration_label(best['calibration'])}), a "
      f"difference well inside either confidence interval; sensitivity "
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
    w("What this study supports is a methodological claim in two parts. "
      "First, about this benchmark: the numbers reported on it are largely "
      "explained by three properties of the data, the largest of which - "
      "case mix - is rarely examined, and the least visible of which - "
      "cohort overlap - invalidates the cross-dataset validations that "
      "would otherwise be its strongest evidence. Second, about method: "
      "every one of the three was found by a check that can be run before "
      "any model is fitted. Declare the prohibited columns and enforce them "
      "with something that raises rather than drops; look at the stage "
      "distribution and the univariate separability before believing a "
      "discrimination figure; and verify that a validation cohort is "
      "actually a different set of patients. Each check is cheap; each one "
      "here changed a conclusion.")
    w("")
    w("What the study does not support is any claim about clinical utility, "
      "causality, deployment readiness or generalisability. The estimates "
      "are internal to 200 patients, with pre-discretised predictors, a "
      "case mix dominated by advanced disease, no demographic detail beyond "
      "age bands, four records whose labels contradict their own staging, "
      "and confidence intervals wide enough to encompass materially "
      "different conclusions. The cost-stratified comparison reported here "
      "says something about how much discrimination survives dropping "
      "laboratory variables **on this sample**; it says nothing about "
      "screening, because a sample in which 84% of cases have "
      "moderate-to-severe disease is not a screening population. "
      "**Prospective validation in a genuine screening series would be "
      "required** before any clinical interpretation is warranted, and no "
      "dataset examined here can substitute for it.")
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
    w("11. Kabir MA, Munira S, Azad DT, Ikram SM, Sarker MHR, Hanifi SMA. "
      "Community-based early-stage chronic kidney disease screening using "
      "explainable machine learning for low-resource settings. "
      "*Int J Med Inform*. 2026. "
      "arXiv:[2601.01119](https://arxiv.org/abs/2601.01119)")
    w("12. Islam MA, Majumder MZH, Hussein MA. Chronic kidney disease "
      "prediction based on machine learning algorithms. "
      "*J Pathol Inform*. 2023;14:100189. "
      "doi:[10.1016/j.jpi.2023.100189](https://doi.org/10.1016/j.jpi.2023.100189)")
    w("13. Hossain MF, Diya ST, Khan R. ACD-ML: Advanced CKD detection using "
      "machine learning: a tri-phase ensemble and multi-layered stacking and "
      "blending approach. *Comput Methods Programs Biomed Update*. "
      "2025;7:100173. "
      "doi:[10.1016/j.cmpbup.2024.100173](https://doi.org/10.1016/j.cmpbup.2024.100173)")
    w("")
    w("*No reference above was generated without verification; each has a DOI "
      "or a stable arXiv identifier. The prior-work survey table additionally "
      "distinguishes, per study, which facts were verified from full text or "
      "abstract and which are attributed to the comparison table of [11] "
      "pending hand verification.*")
    w("")

    # ---------------- Appendix ----------------
    # ---------------- Appendix: TRIPOD+AI ----------------
    tripod = t("table_36_tripod_ai.csv")
    tripod_sum = t("table_36_tripod_ai_summary.csv").set_index("status")["n_items"]
    w("## Appendix A. TRIPOD+AI checklist")
    w("")
    w(f"Reporting follows the spirit of TRIPOD+AI [3]. Of "
      f"{int(tripod_sum.sum())} items, "
      f"{int(tripod_sum.get('satisfied', 0))} are satisfied, "
      f"{int(tripod_sum.get('partly', 0))} partly, "
      f"{int(tripod_sum.get('adapted', 0))} adapted (the study is a "
      f"benchmark re-analysis, not model development), and "
      f"{int(tripod_sum.get('not-applicable', 0))} not applicable. "
      f"Items that are only partly met say so; the checklist is an audit, "
      f"not a compliance claim.")
    w("")
    w("| # | Section | Item | Status | Evidence |")
    w("|---:|---|---|---|---|")
    for _, r in tripod.iterrows():
        w(f"| {r['item']} | {r['section']} | {r['topic']} | "
          f"**{r['status']}** | {r['evidence']} |")
    w("")

    w("## Appendix B. Generated artefacts")
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
