# Reliable and Explainable Low-Cost CKD Screening in Bangladesh

**A nested validation, leakage audit, and calibration study.**

---

> ### Scope and safety statement — read first
>
> This repository contains an **internally validated, retrospective methodological
> feasibility study** on 200 patient records from a single hospital.
>
> * It is **not a diagnostic tool** and must not be used to make decisions about
>   any patient.
> * No claim is made about clinical utility, causality, deployment readiness, or
>   generalisability to any other population.
> * The models are validated **only** by cross-validation on the same 200
>   patients used to build them. **External and prospective validation would be
>   required** before any of this could be considered clinically meaningful.
> * One feature configuration (`leaky_model`) is **deliberately invalid**. It
>   exists solely to quantify how much target leakage inflates apparent
>   performance, and is labelled as such in every table and figure.

---

## What this study asks

**Primary question.** Can a small, explainable model predict CKD status using
inexpensive and routinely available patient information, while keeping
clinically useful sensitivity and calibrated risk estimates?

**Secondary questions.**

1. How much does target leakage inflate model performance?
2. Can a reduced-feature (low-cost) model perform comparably to a
   laboratory-based model?
3. Are the predicted probabilities properly calibrated?
4. Are the identified important predictors stable across resampling?
5. How uncertain are the results, given only 200 patients?

## The dataset, and one fact that shapes everything

`ckd-dataset-v2.csv` holds 200 records collected at Enam Medical College,
Savar, Bangladesh, in 29 columns, preceded by two metadata rows that are not
patients.

**Every continuous variable in the released file has already been discretised
into interval strings** — `1.019 - 1.021`, `< 112`, `>= 227.944`. The original
measurements are not recoverable. This information loss is a property of the
published data, not a modelling choice, and it constrains what any analysis of
this file can conclude.

The encoding used here maps each interval label to a single representative
number (midpoint for closed bins, the finite edge for open-ended bins). That
mapping is a **pure function of one cell's text**: it uses no information from
any other row and no information from the outcome, so it cannot transfer
information between cross-validation folds. Everything else — imputation,
scaling, hyper-parameter tuning, calibration, threshold selection — happens
strictly inside training folds.

## What the study found

Full detail is in [`reports/research_report.md`](reports/research_report.md). In short:

1. **Target leakage takes any configuration to ROC-AUC 1.000.** `affected` is an
   exact copy of the outcome; `stage` is a post-diagnosis label that is 100 % CKD
   in stages s3 and s5; `grf` is the eGFR value that defines staging. Whatever
   the honest baseline was, adding these closes 100 % of the remaining error.
2. **A second, larger source of optimism: case mix.** Even with every prohibited
   column removed, the full valid configuration separates all 200 patients out
   of fold. That is not a residual leak — the guards and an automated test
   confirm it — it is the sample. 107 of 128 CKD patients (84 %) are stage s3–s5,
   and haemoglobin alone reaches univariate ROC-AUC 0.968. This dataset contrasts
   advanced kidney disease with comparatively healthy controls, which is not the
   problem a screening instrument faces. In the early-CKD subgroup, sensitivity
   falls (low-cost model 0.969 → 0.905; clinical-only 0.891 → 0.762).
3. **The low-cost model is competitive.** History, examination and a urine
   dipstick (11 variables, no venepuncture) reach ROC-AUC 0.992 (95 % CI
   0.981–0.999) against 0.995 for the 14-variable laboratory panel — a difference
   well inside either interval. Adding urine microscopy changes nothing
   (0.9919 vs 0.9921), which supports the cost-tiering judgement.
4. **The low-cost model is also the more trustworthy one.** It has an identified
   calibration slope (1.26) where the top-ranked full-panel model does not — that
   model separates perfectly, so its recalibration slope has no maximum-likelihood
   solution. Its feature importances are far more stable too (Kendall's W 0.694
   vs 0.471), with `sg`, `al`, `dm` and `htn` in the top 5 of ≥ 80 % of folds.
5. **Isotonic calibration did *not* overfit here** (median Brier 0.029 vs 0.036
   for Platt), contrary to the usual small-sample expectation. Reported as
   observed, with the caveat that the comparison rests on 25 outer folds.

None of this is evidence of clinical utility. See the safety statement above and
the limitations section of the report.

## Quick start

```bash
# 1. Clone / open the project directory, then create an isolated environment
python -m venv .venv

#    Linux / macOS
source .venv/bin/activate
#    Windows PowerShell
.venv\Scripts\Activate.ps1
#    Windows Git Bash
source .venv/Scripts/activate

# 2. Install dependencies (pinned to the reference run)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. Confirm the environment is sound before spending compute
python -m pytest tests/ -q

# 4. Reproduce every result, figure and table
python scripts/run_all.py
```

`scripts/run_all.py` runs all stages in order and stops at the first
failure. To run them individually:

```bash
python scripts/01_prepare_data.py          # clean, data dictionary, quality audit   (~5 s)
python scripts/02_eda.py                   # exploratory figures and tables          (~15 s)
python scripts/03_nested_cv.py             # repeated nested cross-validation        (~72 min)
python scripts/04_evaluate.py              # metrics, CIs, calibration, leakage audit (~3 min)
python scripts/05_importance_stability.py  # SHAP, permutation importance, stability (~30 min)
python scripts/06_write_report.py          # assemble reports/research_report.md      (~2 s)
python scripts/make_notebook.py            # regenerate the walkthrough notebook      (~1 s)
```

`06_write_report.py` injects every number in the research report directly from
the generated tables, so the write-up cannot drift away from the computed
results.

Useful flags:

```bash
python scripts/03_nested_cv.py --repeats 1            # fast smoke run
python scripts/03_nested_cv.py --n-jobs 4             # limit parallelism
python scripts/03_nested_cv.py --models logreg ebm    # subset of models
python scripts/03_nested_cv.py --configs low_cost_model laboratory_model
```

Runtimes are for 12 CPU cores. Stage 3 is the only expensive step; it fits
roughly 100,000 models.

## Reproducibility

* One master seed (`config/experiment.yaml: seed`) deterministically derives
  every fold seed and every model seed.
* Stage 3 writes a manifest (`data/processed/cv_predictions_manifest.json`)
  recording the seed, design, models actually run, models unavailable in this
  environment, and the runtime.
* The raw file is checksummed on every load and the SHA-256 is printed into the
  data-quality report. It is never modified.
* Every reported number is derived from one predictions table
  (`data/processed/cv_predictions.csv.gz`), so tables and figures cannot drift
  apart from the cross-validation that produced them.
* `tests/test_reproducibility.py` asserts that identical seeds give identical
  predictions and that different seeds give different fold partitions.

## Repository layout

```
config/experiment.yaml          Every tunable knob: seed, CV design, thresholds, bootstrap
data/raw/                       The original file, preserved and checksummed. Never written to.
data/processed/                 Cleaned data, encoded matrix, data dictionary, CV predictions
notebooks/                      Narrative walkthrough; all logic lives in src/
src/ckd/
  config.py                     Config loading and project-root resolution
  data/bins.py                  Interval-string parsing; the stateless encoding
  data/load.py                  Raw loading and metadata-row removal
  data/clean.py                 Missing-token handling, encoding, target coding
  data/quality.py               The structured data-quality audit
  features/configs.py           Variable specs, cost tiers, the six feature configurations
  features/encoders.py          LeakageGuard, ColumnSelector, outcome-proxy detection
  models/zoo.py                 Estimators and their (small) search spaces
  models/pipeline.py            Leakage-safe pipeline construction
  models/nested_cv.py           The repeated nested CV engine
  evaluation/metrics.py         Discrimination, classification and calibration metrics
  evaluation/thresholds.py      Threshold policies and decision-curve net benefit
  evaluation/bootstrap.py       Aggregation and patient-level bootstrap CIs
  evaluation/importance.py      Permutation / SHAP importance and stability statistics
  evaluation/plots.py           House figure style
scripts/                        The five numbered pipeline stages plus run_all.py
tests/                          Automated leakage, preprocessing, metric and reproducibility tests
reports/
  data_quality_report.md        Generated data-quality audit
  research_report.md            The study write-up
  figures/                      All figures, 300 dpi PNG + PDF
  tables/                       All tables, CSV
```

## How leakage is prevented, programmatically

Three independent layers, each tested:

1. **Declaration.** `src/ckd/features/configs.py` names every prohibited column
   (`class`, `affected`, `stage`, `grf`) and each configuration's contents.
2. **Runtime guard.** Every pipeline starts with a `LeakageGuard` that raises
   `LeakageError` at *fit and at transform time* if a prohibited column is
   present. It raises rather than dropping the column, so a bug cannot be
   silently absorbed. The invalid configuration must opt in explicitly, and even
   then the raw outcome column can never pass.
3. **Assertion before every run.** `assert_pipeline_is_clean` is called for each
   of the 36 pipelines before stage 3 does any work.

`tests/test_leakage.py` includes the key adversarial test: hand a valid
pipeline the **whole** dataframe including `affected`, `stage` and `grf`, and it
must **refuse**, not quietly select its own subset.

`tests/test_leakage.py::TestNoPreprocessingOnFullData` separately confirms that
scaler and imputer statistics differ between folds, i.e. that preprocessing was
not fitted on the complete dataset.

## Validation design

| Element | Choice |
|---|---|
| Outer loop | 5-fold stratified cross-validation |
| Inner loop | 4-fold stratified cross-validation |
| Repeats | 5 (giving 25 independent outer test sets) |
| Inner selection metric | ROC-AUC |
| Primary threshold | 0.50, prespecified in config before any result was seen |
| Screening threshold | Lowest threshold reaching 90 % sensitivity **on inner CV predictions of the outer training fold** |
| Calibration | none / Platt (sigmoid) / isotonic, each fitted inside training folds |
| Uncertainty | Repeat-level spread **and** patient-level stratified bootstrap (2000 resamples) |

Not used: a single train/test split as the principal evaluation; SMOTE;
neural networks; any tuning on outer-fold results.

## Interpreting the outputs

* `reports/research_report.md` — the full write-up.
* `reports/data_quality_report.md` — every data defect found, with the CSV line
  number of each affected record.
* `reports/tables/table_13_leakage_audit.csv` — the leakage comparison.
* `reports/tables/table_16_selected_models.csv` — the model selected by the
  documented rule, not by hand.
* `reports/tables/table_20_importance_stability.csv` — how often each feature
  reaches the top 5 across the 25 outer folds.

Feature importance describes how a fitted model uses a column **given the other
columns present**. It is not an effect size and it is not causal.

## License and data

Code in this repository is MIT licensed. The dataset is not redistributed by
this license; it is included as supplied for the purpose of reproducing this
analysis. Any reuse of the patient data is governed by the terms under which it
was originally released.
