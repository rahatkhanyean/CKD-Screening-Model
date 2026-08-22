"""Generate notebooks/01_analysis_walkthrough.ipynb.

The notebook is generated rather than hand-edited so that it stays consistent
with the modules it documents. It contains explanation and presentation only:
every computation it shows is a call into ``src/ckd``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().split("\n")}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").split("\n"),
    }


CELLS = [
    md("""
# Reliable and Explainable Low-Cost CKD Screening in Bangladesh

**A nested validation, leakage audit, and calibration study**

---

> **This is not a diagnostic tool.** It is an internally validated,
> retrospective methodological feasibility study on 200 patient records from a
> single hospital. No claim is made about clinical utility, causality,
> deployment readiness or generalisability. External and prospective validation
> would be required before any of this could be considered clinically
> meaningful.

---

This notebook is a **guided tour of the results**. It contains no analysis
logic: every computation lives in `src/ckd/` and every number shown here is
read back from the artefacts produced by the pipeline stages. Run
`python scripts/run_all.py` first.
"""),
    code("""
import json
import sys
from pathlib import Path

import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"

def table(name):
    return pd.read_csv(TABLES / name)

def figure(stem):
    display(Image(filename=str(FIGURES / f"{stem}.png")))

print("project root:", ROOT)
"""),
    md("""
## 1. The dataset, and the fact that shapes the whole study

The released file contains 200 patient records in 29 columns, preceded by two
metadata rows that are not patients.

The decisive property is that **every continuous variable has already been
discretised into interval strings** by the data publishers - `1.019 - 1.021`,
`< 112`, `>= 227.944`. The original measurements are unrecoverable. That
information loss belongs to the published data, not to any choice made here,
and it bounds what any analysis of this file can conclude.
"""),
    code("""
raw = pd.read_csv(ROOT / "data" / "raw" / "ckd-dataset-v2.csv", dtype=str, keep_default_na=False)
display(raw.head(4))
print("The first two rows below the header are metadata, not patients.")
"""),
    code("""
display(table("table_00_dataset_summary.csv"))
"""),
    md("""
## 2. Encoding: why it is safe to do before splitting

Each interval label maps to one representative number - the midpoint for a
closed bin, the finite edge for an open-ended one. That mapping is a **pure
function of a single cell's text**. It uses no information from other rows and
none from the outcome, so it cannot transfer information between
cross-validation folds.

Everything else - imputation, scaling, hyper-parameter tuning, calibration,
threshold selection - happens strictly inside training folds.
"""),
    code("""
from ckd.data.bins import parse_bin

for label in ["< 1.007", "1.019 - 1.021", ">= 1.023", "< 112", "112 - 154", ">= 448"]:
    lo, hi, rep = parse_bin(label)
    print(f"{label:>16}  ->  lower={lo:>10.4g}  upper={hi:>10.4g}  representative={rep:.4g}")
"""),
    md("""
## 3. Data quality

The full audit is in `reports/data_quality_report.md`. The headline findings:

- No missing values other than a single invalid cell, and no duplicate records.
- `affected` is an **exact one-to-one copy of the outcome**.
- `stage` is a deterministic banding of `grf` (eGFR); stages s3 and s5 are 100 %
  CKD.
- The `su` bin edges published with the dataset overlap, so its ordinal encoding
  contains ties.
- Four patients labelled `notckd` carry an advanced CKD stage and an eGFR below
  60 - an internal inconsistency, reported rather than silently corrected.
- One `grf` cell contains the token `" p "`.
"""),
    code("""
display(table("table_06_variable_summary.csv"))
"""),
    code("""
figure("fig_e3_leakage_structure")
"""),
    md("""
### The `" p "` value in `grf`

One cell in the eGFR column contains the token `" p "`, which is not an
interval label. The affected patient is labelled `ckd` and staged `s5`; every
other `s5` patient falls in the lowest eGFR band.

The most parsimonious reading is a data-entry error. Because the true value
cannot be recovered, the cell is treated as **missing** and imputed inside
training folds only. The raw file is preserved unmodified. Note that `grf` is
excluded from every valid configuration, so this defect can only affect the
deliberately invalid leaky model.
"""),
    md("""
## 4. Feature configurations

Six configurations. Four are required by the study design; two are flagged
sensitivity analyses that exist because the boundary of "low cost" is a genuine
judgement call that should not be made silently.

The dataset ships **no data dictionary**, so the cost tiering is an explicit,
auditable judgement recorded in `src/ckd/features/configs.py`, with an
`uncertain` flag wherever the file cannot settle the question.
"""),
    code("""
cfg = table("table_02_feature_configurations.csv")
display(cfg[["configuration", "n_features", "valid_for_clinical_interpretation",
             "contains_prohibited_columns"]])
"""),
    code("""
dd = table("table_01_data_dictionary.csv")
display(dd[dd["interpretation_uncertain"]][["variable", "tier", "uncertainty_note"]])
"""),
    code("""
figure("fig_e5_configuration_composition")
"""),
    md("""
## 5. Validation design

| Element | Choice |
|---|---|
| Outer loop | 5-fold stratified cross-validation |
| Inner loop | 4-fold stratified cross-validation |
| Repeats | 5, giving 25 independent outer test sets |
| Inner selection metric | ROC-AUC |
| Primary threshold | 0.50, prespecified before any result was seen |
| Screening threshold | lowest threshold reaching 90 % sensitivity on inner CV predictions of the **training** fold |
| Calibration | none / Platt / isotonic, each fitted inside training folds |

Leakage is prevented by three independent, tested layers: declaration, a
runtime `LeakageGuard` that raises rather than dropping, and an assertion run
over all 36 pipelines before any fitting.
"""),
    code("""
manifest = json.loads((ROOT / "data" / "processed" / "cv_predictions_manifest.json").read_text())
for k, v in manifest.items():
    print(f"{k:<28} {v}")
"""),
    md("""
## 6. Leakage audit

The central methodological result: how much do the prohibited columns inflate
apparent performance?
"""),
    code("""
audit = table("table_13_leakage_audit.csv")
display(audit[["config", "model", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high",
               "sensitivity", "specificity", "roc_auc_inflation_vs_full_valid"]])
"""),
    code("""
figure("fig_r2_leakage_audit")
"""),
    code("""
figure("fig_r1_auc_heatmap")
"""),
    md("""
## 7. Can a low-cost model compete with a laboratory model?

This is the primary research question. Compare the `low_cost_model`
(history, examination and urine dipstick) with the `laboratory_model`
(blood chemistry, full blood count and urine microscopy).
"""),
    code("""
figure("fig_r4_roc_pr_curves")
"""),
    code("""
figure("fig_r3_screening_metrics")
"""),
    md("""
## 8. Calibration

Predicted probabilities are compared uncalibrated, with Platt (sigmoid)
scaling, and with isotonic regression. Isotonic regression is flexible and, on
200 patients, is expected to overfit; that expectation is tested rather than
assumed.

A calibration slope below 1 means predictions are too extreme; a positive
intercept means risk is systematically under-estimated.
"""),
    code("""
figure("fig_r5_calibration_curves")
"""),
    code("""
figure("fig_r6_calibration_methods")
"""),
    code("""
calib = table("table_14_calibration_results.csv")
display(
    calib[calib["valid"] & (calib["model"] != "dummy")]
    .groupby("calibration")[["brier", "calibration_slope", "calibration_intercept", "ece"]]
    .median()
    .round(4)
)
"""),
    md("""
## 9. Screening trade-offs

For a screening application, the decisive quantities are sensitivity, negative
predictive value, and the exchange rate between missed cases and unnecessary
referrals. Accuracy alone is not an acceptable basis for choosing a model.
"""),
    code("""
figure("fig_r7_threshold_tradeoff")
"""),
    code("""
figure("fig_r8_confusion_matrices")
"""),
    md("""
## 10. Explainability and stability

Feature importance is computed three ways - permutation (model-agnostic), SHAP,
and, for linear models, standardised coefficients - and then examined for
**stability across the 25 outer folds**.

A feature that is important in one fold and absent in the next is not a
finding; it is noise. That is reported explicitly through top-5 selection
frequency, rank spread, Kendall's W and pairwise Jaccard overlap.

Importance describes how a fitted model uses a column given the other columns
present. It is **not** a causal effect and **not** an effect size.
"""),
    code("""
figure("fig_r10_best_valid_importance_stability")
"""),
    code("""
stab = table("table_20_importance_stability.csv")
perm = stab[stab["method"] == "permutation"]
cols = [c for c in perm.columns if c.startswith("top") and c.endswith("frequency")]
display(perm[["target", "config", "model", "feature", "mean_importance",
              "sd_importance", "mean_rank", "sd_rank"] + cols].head(15))
"""),
    md("""
## 11. Uncertainty

Two different quantities, kept separate because conflating them overstates
precision:

1. **Cross-validation partition variability** - how much the result depends on
   which patients landed in which fold.
2. **Patient-level bootstrap** - the closer analogue of a sampling interval.

Neither substitutes for external validation. Both are computed on the same 200
patients that trained the models, so both are optimistic.
"""),
    code("""
figure("fig_r9_uncertainty")
"""),
    md("""
## 12. Conclusions

See `reports/research_report.md` for the full write-up, including limitations,
ethical considerations and the reproducibility statement.

The essential caution: this is a 200-patient, single-centre, retrospective
feasibility study with pre-discretised predictors and no external validation.
It is evidence about **method**, not about patients.
"""),
]


def main() -> int:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = ROOT / "notebooks" / "01_analysis_walkthrough.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
