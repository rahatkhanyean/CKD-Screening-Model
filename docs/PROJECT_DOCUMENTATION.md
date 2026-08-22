# Project documentation

**Reliable and Explainable Low-Cost CKD Screening in Bangladesh — a nested validation, leakage audit, and calibration study.**

This document has three parts:

| Part | Question it answers |
|---|---|
| [Part 1](#part-1--what-was-built) | What was built, and why each piece exists |
| [Part 2](#part-2--what-the-study-establishes) | What the study establishes, what it does not, and what makes it defensible |
| [Part 3](#part-3--next-course-of-action) | What to do next, in priority order |

It is a companion to, not a replacement for, the two generated documents:
`reports/research_report.md` (the study write-up, every number injected from
`reports/tables/`) and `reports/data_quality_report.md` (the data audit).
Where those two are generated, this one is written by hand and describes the
*engineering and the reasoning*.

**Scope reminder.** Everything below concerns an internally validated,
retrospective methodological feasibility study on 200 patient records from a
single hospital. Nothing here is a diagnostic tool, and no claim of clinical
utility, causality, deployment readiness or generalisability is made or
supported.

---

## Part 1 — What was built

### 1.1 The problem, stated precisely

Given `ckd-dataset-v2.csv` — 200 records, 29 columns, two non-patient metadata
rows, outcome `class` in {ckd, notckd} — can a small, explainable model predict
CKD status from **inexpensive** information (history, examination, urine
reagent strip) with clinically useful sensitivity and calibrated risk?

Two properties of the file dominate every design decision:

1. **Three columns are not legitimate predictors.** `affected` is an exact copy
   of the outcome (bias-corrected Cramer's V = 1.0). `stage` is a
   post-diagnosis CKD stage label — 100% CKD at s3 and s5. `grf` is the eGFR
   value from which `stage` is banded. Any pipeline that selects features by
   association with the outcome will grab all three.
2. **Every continuous variable is already discretised** into interval strings
   (`1.019 - 1.021`, `< 112`, `>= 227.944`). The original measurements are not
   recoverable. This is a property of the published data, not a modelling
   choice.

### 1.2 The pipeline

Each stage is a standalone script; `scripts/run_all.py` runs them in order and
stops at the first failure. All reusable logic lives in `src/ckd/` — the
scripts orchestrate, they do not implement.

Stages 0 and 7–12 were added when the study was reframed from a screening
feasibility study to a re-analysis of the benchmark itself (see Part 4).
Stage numbering reflects the order in which stages were written, not the
order they run in; `run_all.py` is the authority on order.

| # | Script | Produces | Runtime |
|---|---|---|---|
| 0 | `00_provenance.py` | Provenance gate over registered external datasets: tables 24/24-evidence/24-agreement, `reports/external/provenance_report.md` | ~30 s |
| 1 | `01_prepare_data.py` | `ckd_clean.csv`, `ckd_encoded.csv`, `bin_mappings.json`, `data_dictionary.csv`, `feature_configurations.csv`, `reports/data_quality_report.md` | 2.3 s |
| 2 | `02_eda.py` | Figures E1–E5, tables 00–06 (all labelled EXPLORATORY) | 6.5 s |
| — | `pytest tests/` | 338 automated checks | ~170 s |
| 3 | `03_nested_cv.py` | `cv_predictions.csv.gz` (108,000 rows), `cv_predictions_folds.csv`, run manifest. `--label-variant` produces the two label-robustness runs | 71.9 min |
| 4 | `04_evaluate.py` | Tables 08–18 and 21–23, figures R1–R9 and R12 | 165 s |
| 5 | `05_importance_stability.py` | Tables 19–20, figures R10 (x3) and R11 | ~30 min |
| 7 | `07_literature.py` | Prior-work survey: tables 26 and 26-summary | 1 s |
| 8 | `08_sensitivity_labels.py` | Label-robustness comparison: table 27 | 5 s |
| 9 | `09_optimism.py` | Apparent vs nested-CV performance: table 29 | ~2 min |
| 10 | `10_ebm_shapes.py` | Readable low-cost rule: tables 28/28-summary/28-notes, fig R13 | ~9 min |
| 11 | `11_continuous_recovery.py` | Continuous-value recovery: tables 30–34, fig R14, `ckd_recovered_continuous.csv` | ~1 min |
| 12 | `12_binned_vs_continuous.py` | Model-level binned-vs-continuous comparison: table 35 | ~25 min |
| 6 | `06_write_report.py` | `reports/research_report.md` | 2 s |
| — | `make_notebook.py` | `notebooks/01_analysis_walkthrough.ipynb` (37 cells) | 1 s |

Stages 11 and 12 are **gated**: each refuses to run unless the provenance
report classifies the source release SAME-SOURCE, because recovering one
cohort's values from another is only legitimate when they are the same
patients.

Stage 3 is the only expensive step. It fits roughly 100,000 models:
6 feature configurations x 6 model families x 3 calibration methods x 5 repeats
x 5 outer folds = **2,700 outer folds**, each of which runs a 4-fold inner grid
search, a refit, a calibration wrapper and a separate threshold-selection
cross-validation.

### 1.3 Module responsibilities

21 modules, 8,613 lines of Python across `src/`, `scripts/` and `tests/`.

**`src/ckd/data/` — from raw file to modelling matrix**

| Module | Responsibility |
|---|---|
| `load.py` | Reads every cell as a string with `keep_default_na=False`, splits off exactly the two metadata rows, attaches `source_csv_line` provenance (patients start at line 4), computes the SHA-256 of the raw file. |
| `bins.py` | Parses one interval label into `(lower, upper, representative)`. Midpoint for closed bins; the finite edge for open-ended bins. |
| `clean.py` | Applies the missing-token policy, encodes every column to its representative value, codes the target, records every missing and unparsed cell with its CSV line. |
| `quality.py` | The 12-section data-quality audit: provenance, class distribution, missing values, bin anomalies, implausible values, duplicates, near-constant features, outcome proxies, the `class`/`affected`/`stage`/`grf` relationships, clinical inconsistencies, the `" p "` investigation, variable inventory. Also `cramers_v()` and `audit_bin_monotonicity()`. |

**`src/ckd/features/` — what may be used, and what may never be**

| Module | Responsibility |
|---|---|
| `configs.py` | One `VariableSpec` per column (tier, description, rationale, `uncertain` flag, uncertainty note); the prohibited-column sets; the six feature configurations. |
| `encoders.py` | `LeakageGuard` (raises `LeakageError`), `ColumnSelector`, `assert_no_target_correlation_leak()`. |

**`src/ckd/models/` — fitting**

| Module | Responsibility |
|---|---|
| `zoo.py` | Estimator factories and their deliberately small search spaces; availability reporting for optional dependencies. |
| `pipeline.py` | `build_pipeline()` and `assert_pipeline_is_clean()`. |
| `nested_cv.py` | The repeated nested CV engine; deterministic seed derivation; training-fold-only threshold selection. |

**`src/ckd/evaluation/` — measuring**

| Module | Responsibility |
|---|---|
| `metrics.py` | Discrimination, classification and calibration metrics; separation detection. |
| `thresholds.py` | Threshold policies (prespecified, target-sensitivity, Youden) and decision-curve net benefit. |
| `bootstrap.py` | Per-repeat and per-fold aggregation, stratified patient-level bootstrap CIs, quantile-binned calibration curves with Wilson intervals. |
| `importance.py` | Permutation and SHAP importance, Kendall's W, pairwise Jaccard of top-k sets, top-k selection frequency. |
| `spectrum.py` | Case-mix (spectrum) analysis and univariate separability. Added mid-project — see §2.2. |
| `plots.py` | The house figure style (Okabe–Ito colourblind-safe palette, 300 dpi PNG + PDF). |

### 1.4 The six feature configurations

| Configuration | n | Valid? | Contents |
|---|---:|---|---|
| `leaky_model` | 28 | **No** | Full valid set **plus `grf`, `stage`, `affected`**. Exists only to quantify leakage. Labelled INVALID in every table and figure. |
| `full_valid_model` | 25 | Yes | Everything that is not the outcome, a copy of it, or a post-diagnosis derivative. The honest upper reference. |
| `low_cost_model` | 11 | Yes | History, examination and a urine reagent strip. No venepuncture, no microscope, no analyser. **The primary question.** |
| `laboratory_model` | 14 | Yes | Blood chemistry, full blood count and urine microscopy. The comparator. |
| `clinical_only_model` | 8 | Yes | *Sensitivity analysis.* History and examination alone — isolates how much of the low-cost result comes from the dipstick. |
| `low_cost_plus_urine_micro_model` | 15 | Yes | *Sensitivity analysis.* Tests whether the judgement call that put microscopy outside the low-cost tier changes anything. |

The last two were **added** because "low cost" has a genuine tier boundary
(urine microscopy needs a microscope and a technician, but no analyser and no
venepuncture) that should not be decided silently. Machine-checked invariants:
`clinical_only` is a strict subset of `low_cost`, which is a strict subset of
`full_valid`; and `low_cost` union `laboratory` equals `full_valid` exactly.

Ten variables are flagged `uncertain` rather than given an invented
interpretation, because the dataset ships without a data dictionary. `ane`
matters most: it is tiered conservatively as `blood_lab` (and so excluded from
`low_cost`) because its provenance could not be established. A test confirms it
is *not* a deterministic function of `hemo`. Erring this way can only
**understate** low-cost performance.

### 1.5 The six design decisions that carry the study

**(a) The bin encoding is stateless — leakage-proof by construction.**
`representative()` is a pure function of one cell's text. It uses no
cross-row statistics and no outcome information, so it cannot transfer
information between folds. This is what makes it legitimate to encode once,
before splitting. Two tests assert the property directly, including one that
re-encodes arbitrary row subsets and checks the values are unchanged.

**(b) Leakage is prevented by exception, in three independent layers.**

1. *Declaration.* `configs.py` names the prohibited columns and each
   configuration's exact contents.
2. *Runtime guard.* Every pipeline begins
   `outcome_guard -> leakage_guard -> select -> impute -> [scale] -> clf`. The
   guard raises at **fit and at transform** — so a prohibited column smuggled in
   at prediction time also fails. It **raises rather than drops**: silently
   dropping would hide a bug in the caller. The invalid configuration must opt
   in explicitly via an `allow` list, and even then the raw `class` column can
   never pass.
3. *Assertion before work.* `assert_pipeline_is_clean()` runs on all 36
   pipelines before stage 3 does anything.

The runner independently restricts the frame to the columns a configuration
declares, so the guard acts as a *second* check on the caller rather than as
the only one. The key adversarial test hands a valid pipeline the **whole**
dataframe including `affected`, `stage` and `grf` and requires it to **refuse**
— not to quietly select its own subset.

**(c) Every reported number descends from one file.** Stage 3 writes 108,000
out-of-fold predictions to `cv_predictions.csv.gz`. Every metric, interval,
curve, figure and sentence in the report is derived from that table, so results
cannot drift apart from the cross-validation that produced them. Stage 6 then
injects each number into the report text from `reports/tables/`, so the prose
cannot drift either.

**(d) The screening threshold never sees test data.** The 0.50 threshold is
prespecified in `config/experiment.yaml`. The 90%-sensitivity screening
threshold is chosen from a *separate* cross-validated prediction on the outer
**training** fold. The outer test fold is touched exactly once, by
`predict_proba`.

**(e) Calibration statistics are computed per repeat, not on averaged
probabilities.** Averaging a patient's probability across five repeats shrinks
it toward the centre, which biases the calibration slope upward. This was found
and corrected during the build; the statistics are now computed within each
repeat and summarised, and calibration curves stack predictions rather than
averaging them.

**(f) Complete separation is detected, not optimised through.** Under
separation the recalibration slope has no maximum-likelihood solution — the
likelihood increases without bound. An optimiser still returns a number (values
in the hundreds were observed before the check was added).
`separates_perfectly()` now detects this and the slope is reported as
**undefined**. 2 of 75 valid cells are affected, including the model the
mechanical selection rule picks.

### 1.6 The test suite — 227 tests, what each file guarantees

| File | Guarantees |
|---|---|
| `test_data_loading.py` | Exactly two metadata rows removed and they really are metadata; 200 patients remain; provenance line numbers contiguous; bin parsing correct on known labels; **the encoding is stateless and uses no cross-row information**; bin representatives are non-decreasing within each feature; the `" p "` token became missing; exactly one missing cell; the raw file is untouched. |
| `test_leakage.py` | No valid configuration declares a prohibited column; `affected`/`stage`/`grf` appear in no valid configuration; the guard raises at fit, at transform, and on unnamed arrays; an exemption cannot unlock `class`; **a valid pipeline handed the full frame must refuse**; and `TestNoPreprocessingOnFullData` confirms imputer and scaler statistics differ between folds and match the training fold rather than the complete dataset. |
| `test_encoders.py` | Every column has a specification with a rationale; uncertain variables carry an uncertainty note and certain ones do not; `ane` is conservatively tiered and is *not* a deterministic function of `hemo`; the configuration subset and partition invariants hold; spectrum subgroups cover all stages and retain all non-CKD patients. |
| `test_metrics.py` | Every metric matches scikit-learn; confusion-matrix identities; calibration slope behaves correctly on synthetic over-, under- and mis-calibrated data; ECE approximately 0 under perfect calibration; threshold policies achieve their targets; and `TestSeparationHandling` covers the undefined-slope case. |
| `test_pipelines.py` | **No neural-network model is registered**; search spaces stay small enough for the sample size; guards come before everything else in every pipeline; each patient is predicted exactly once per repeat; outer folds are stratified; fold seeds are distinct, deterministic and in range. |
| `test_reproducibility.py` | Cleaning is deterministic; identical seeds give identical predictions and different seeds give different partitions; raw and top-level copies agree; the manifest matches the configuration; **no prohibited column name appears in the predictions**; and `TestReportedResultsArePlausible` requires the dummy baseline to be at chance, the leaky configuration to be at least as good as valid ones, CIs to bracket their estimates, and — critically — **near-perfect valid performance to be documented rather than unexplained**. |

That last test deserves a note. When the valid full model reached ROC-AUC
1.0000, the naive tripwire ("perfection means a bug") fired. Rather than
weaken it to pass, it was **rewritten to require the explanation to exist on
disk**: the test now fails if the case-mix analysis (tables 21–23) is removed.

---

## Part 2 — What the study establishes

### 2.1 Machine-checked guarantees

These are not claims about kidney disease. They are claims about the analysis,
and each is enforced by code that fails loudly rather than by a promise in
prose.

| # | Guarantee | How it is enforced |
|---|---|---|
| G1 | No prohibited column entered any valid pipeline | 3 independent layers; final audit over every pipeline actually run: **36 checked, 0 violations** |
| G2 | No preprocessing was fitted on the complete dataset | `TestNoPreprocessingOnFullData` shows imputer and scaler statistics differ between folds and equal the training-fold values |
| G3 | The encoding cannot transfer information between folds | Statelessness tested directly by re-encoding row subsets |
| G4 | The raw file was never modified | SHA-256 checked on every load; original and `data/raw/` copy both `f24075f4...4ea84a` |
| G5 | The whole analysis is deterministic | One master seed (20240517) derives all fold and model seeds; identical seeds give identical predictions, different seeds give different partitions |
| G6 | Every metric is correct | Each checked against scikit-learn, plus identity and edge-case tests |
| G7 | No neural network was used and no search space was oversized | Asserted in `test_pipelines.py` |
| G8 | The prediction artefact is internally consistent | 108,000 rows; each patient predicted exactly once per repeat per cell; labels consistent across all cells; class balance preserved |
| G9 | No number in the report was transcribed by hand | Stage 6 injects every value from `reports/tables/` |

### 2.2 Answers to the research questions

All figures below are out-of-fold, from repeated nested cross-validation
(5 outer x 4 inner x 5 repeats = 25 independent outer test sets), with
patient-level stratified bootstrap intervals (2,000 resamples).

**Primary question — can a low-cost model predict CKD with useful sensitivity
and calibrated risk?** On this sample, yes. The 11-variable low-cost model
(SVM RBF, isotonic) reaches **ROC-AUC 0.993 (95% CI 0.983–0.9997)**, sensitivity
0.969, specificity 1.000, NPV 0.947, Brier 0.0192, calibration slope 1.26. It
uses no venepuncture, no microscope and no analyser. **This is a feasibility
signal, not screening performance** — see the case-mix finding below.

**SQ1 — how much does leakage inflate performance?** To the ceiling, from any
starting point. The audit compares uncalibrated cells only, so the comparison
isolates the feature set rather than the calibration method:

| Configuration | Best uncalibrated ROC-AUC | Headroom below 1.0 | Fraction of headroom closed by leakage |
|---|---:|---:|---:|
| `leaky_model` (INVALID) | 1.0000 | — | — |
| `full_valid_model` | 1.0000 | 0.0000 | undefined (already saturated) |
| `laboratory_model` | 0.9951 | 0.0049 | — |
| `low_cost_model` | 0.9921 | 0.0079 | **100%** |
| `clinical_only_model` | 0.9526 | 0.0474 | **100%** |

Absolute inflation against the full valid model is +0.0000, which read naively
would say "leakage does nothing". It is the wrong reading: the valid model is
already at the ceiling. Measured against baselines that are *not* saturated,
leakage closes **100% of the remaining error** — for an 11-variable set and an
8-variable set alike. This reframing is why table 23 exists.

**SQ2 — can a reduced-feature model match a laboratory model?** Effectively
yes, on this sample. Like-for-like (uncalibrated): 0.9921 low-cost vs 0.9951
laboratory. Best-calibrated: 0.9933 vs 0.9952. The difference is far inside
either confidence interval. Two further results sharpen this:

- Adding urine microscopy changes **nothing** (0.9919 vs 0.9921) — a null
  result, reported as such, and it supports the cost-tiering judgement.
- Removing the dipstick costs a lot: `clinical_only` falls to 0.9526 with
  sensitivity 0.891. **The urine reagent strip carries most of the low-cost
  signal**, which is the practically useful part of this finding.

**SQ3 — are the probabilities calibrated?** Median across all valid
configuration x model cells:

| Calibration | Brier | Slope (ideal 1) | Intercept (ideal 0) | ECE |
|---|---:|---:|---:|---:|
| Uncalibrated | 0.037 | 1.83 | 0.27 | 0.046 |
| Platt (sigmoid) | 0.036 | 2.25 | 0.05 | 0.072 |
| Isotonic | **0.029** | **1.71** | 0.11 | **0.028** |

Two findings worth flagging:

- **Isotonic did not overfit**, contrary to the expectation stated in Methods
  *before* the run. Reported as observed, with the caveat that the comparison
  rests on 25 outer folds.
- **Every identified slope is above 1** — the models are systematically
  *under*-confident, the opposite of the usual small-sample pattern. This
  follows from near-separability: regularised models trained on 160 patients
  emit middling probabilities for patients the data actually separate cleanly.

**SQ4 — are the important predictors stable?** Only for the small model.

| Configuration | Kendall's W | Mean pairwise Jaccard (top 5) | Features in top 5 in >=80% of folds |
|---|---:|---:|---|
| `low_cost_model` (SVM) | **0.694** | 0.674 | `sg`, `al`, `dm`, `htn` |
| `full_valid_model` (SVM) | 0.471 | 0.341 | `sg` only |

Twenty-five correlated laboratory variables substitute for one another between
folds; eleven cheap ones do not. SHAP and permutation importance agree
(Spearman 0.905) where both are computable, and where no exact SHAP explainer
applies (the RBF SVM) that is recorded in `table_20_importance_notes.txt`
rather than papered over.

**SQ5 — how uncertain is all of this?** Enough to matter. Bootstrap intervals
for ROC-AUC span roughly 0.02, so no difference between valid configurations is
resolvable. Events per candidate predictor is **2.88**, far below accepted
minimums. Hyper-parameter selection is itself unstable across folds (the EBM
learning rate takes its modal value in only 52% of folds), which is reported
rather than hidden by quoting a single "best" configuration.

**The finding nobody asked for — case mix.** With every prohibited column
removed and the guards proving it, the full valid model still separated all 200
patients out of fold: on repeat-averaged predictions the highest non-CKD
probability is 0.346 and the lowest CKD probability is 0.505, so no threshold
misclassifies anyone. (Per repeat the margin is thinner — 4 of the 5 repeats
separate outright, which is why this cell's calibration slope is reported as
undefined.) Rather than accept the result, the cause was investigated and
turned out to be the sample:

- **107 of 128 CKD patients (84%) are stage s3–s5.** Only ~21 are early-stage.
- **Haemoglobin alone reaches univariate ROC-AUC 0.968** (CKD 6.10–14.55, non-CKD
  11.95–16.50 g/dL); `pcv` 0.949; `sg` 0.889. No multivariable learning is
  needed to separate groups already this far apart.
- This is the profile of a case-control-like contrast between established
  kidney failure and comparatively healthy controls — **not** a consecutive
  screening series, in which most true cases are early and biochemically near
  normal.
- Re-evaluating the *same* out-of-fold predictions within stage-defined
  subgroups (`stage` used only to partition, never as a predictor):

| Configuration | Sensitivity, all patients | Sensitivity, early CKD (s1–s2, n=21) |
|---|---:|---:|
| Low-cost | 0.969 | 0.905 |
| Laboratory | 0.969 | 0.857 |
| Clinical only | 0.891 | **0.762** |

ROC-AUC barely moves in these subgroups (low-cost 0.993 -> 0.983) while
sensitivity falls materially. That dissociation is the point: **a near-ceiling
AUC can coexist with materially worse case detection, and no aggregate metric
reveals it.**

Two independent mechanisms — target leakage and case-mix spectrum — push
measured performance toward 1.0 on this dataset, and neither is clinical
usefulness. A published result of "99% accuracy" on this data has probably met
one or both and cannot tell them apart without exactly this analysis.

### 2.3 Findings reported as negative, unstable or inconclusive

Listing these explicitly, because a study that reports only its wins is not
reporting.

1. Adding urine microscopy to the low-cost set changes nothing (0.9919 vs
   0.9921) — a null result.
2. Feature ranking is only moderately reproducible for the full model
   (Kendall's W = 0.471). Any "top predictors of CKD" narrative from a single
   fit of this data would be an artefact of one partition.
3. Hyper-parameter selection is unstable across folds.
4. 2 of 75 valid cells have an **undefined** calibration slope, including the
   model the prespecified rule selects.
5. Isotonic beating Platt contradicts the a-priori expectation stated in
   Methods; reported as observed, not quietly dropped.
6. Ten variables could not be interpreted from the file and are flagged
   uncertain rather than assigned an invented clinical meaning.
7. Four patients are labelled `notckd` while carrying stage s3–s5 and an eGFR
   below 60 (CSV lines 12, 18, 52, 123). Reported, **not corrected** — there is
   no external source of truth to arbitrate.
8. One `grf` cell contains the token `" p "` (CSV line 183). Investigated,
   documented, treated as missing and imputed inside training folds only. The
   raw file is untouched.

### 2.4 What is explicitly *not* established

- **Nothing about clinical utility.** No model here is validated for any
  clinical purpose.
- **Nothing causal.** Feature importance describes how a fitted model uses a
  column given the other columns present. It is not an effect size.
- **Nothing about generalisability.** Every estimate is internal to 200
  patients from one hospital. Internal cross-validation systematically
  overstates performance in a new population.
- **Nothing about screening performance.** The case-mix analysis shows the
  headline figures overstate it, most for the early-stage patients a screening
  programme exists to find.
- **Nothing about subgroup fairness.** With no sex, ethnicity, socioeconomic or
  location variable, differential performance cannot be checked at all. That is
  a reason for caution, not silence.
- **Nothing about chronicity.** CKD is defined by abnormality persisting beyond
  three months; a cross-sectional record cannot establish that, so the outcome
  label itself rests on information absent from the file.
- **Nothing that transfers across prevalence.** PPV and NPV hold only at this
  sample's 64.0% prevalence.

### 2.5 Why this counts as defensible research

The integrity constraints were converted into mechanisms rather than
intentions:

| Constraint | Mechanism |
|---|---|
| Do not fabricate results | Every number injected from generated tables; the report cannot drift from the computation |
| Do not fabricate references | 10 references, each with a DOI or stable arXiv identifier |
| Do not fabricate variable definitions | 10 variables flagged uncertain instead of interpreted |
| Do not tune on evaluation folds | Tuning, calibration and threshold selection all inside training folds; outer test folds touched once |
| Do not select on accuracy | Selection rule fixed in advance on discrimination with a Brier tie-break; sensitivity, NPV and calibration are what is examined |
| Report negative findings honestly | Section 2.3 above |
| Label exploratory work | Decision curves, univariate screens, correlation structure and the case-mix subgroups all carry an EXPLORATORY label and fed no modelling decision |
| Do not describe this as a diagnostic tool | Stated in the README, the report abstract, the ethics section and this document |

Four things were found and corrected rather than shipped: calibration computed
on repeat-averaged probabilities (biases the slope upward); a calibration slope
of 476.7 emitted under separation (the MLE does not exist); a subgroup
comparison that silently used a different model per subgroup; and — found while
writing this document — the abstract and conclusion describing the case-mix
analysis as "pre-planned" when section 5.5 correctly labels it post hoc. It was
added mid-project after the perfect-AUC result, so "post hoc" is the accurate
label; `scripts/06_write_report.py` was corrected and the report regenerated.

### 2.6 The model this study best supports

The prespecified rule picks `full_valid_model` / SVM (RBF) / isotonic —
ROC-AUC 1.000. **That pick is reported but not endorsed**, because the same
analyses that produced it also disqualify it: it separates the classes
completely so its calibration slope is not identified, its importances are
unstable (W = 0.471), and it needs a full laboratory panel — the exact
constraint the study set out to relax.

**Best supported for the question actually asked: the low-cost model — SVM
(RBF) with isotonic calibration on 11 history, examination and urine-dipstick
variables.** ROC-AUC 0.993 (95% CI 0.983–0.9997) against 0.995 for the
14-variable laboratory panel; sensitivity 0.969; NPV 0.947; Brier 0.0192; an
*identified* calibration slope of 1.26; and reproducible importances. It is the
only candidate whose probability scale can be checked and whose important
variables survive resampling.

This is a statement about which result is better *evidenced*. It is not a
recommendation to use anything.

---

## Part 3 — Next course of action

> **Status note (2026-08-22).** This roadmap was written before the study was
> reframed. Most of it has since been executed, and one item was answered in
> a way that changed the paper. Current status of every item:
>
> | Item | Status | Where |
> |---|---|---|
> | N1 external validation | **Blocked, and the reason is a finding.** The intended cohort (UCI-2015) turned out to share our patients; the provenance gate now blocks it. Genuine external validation needs icddr,b (restricted). | §5.14, `reports/external/provenance_report.md` |
> | N2 prospective series | Not started; still the only route to a clinical claim | — |
> | N3 recover continuous values | **Done**, and better than planned: same-cohort rather than cross-cohort | §5.15–5.16, stages 11–12 |
> | N4 label robustness | **Done**; produced a finding that qualifies the low-cost claim | §5.13, stage 8 |
> | N5 encoding robustness | **Done** | §5.17, stage 14 |
> | N6 whole-procedure bootstrap | **Done** | §5.18, stage 15 |
> | N7 operating point as a decision | **Deliberately not done** — see below | — |
> | N8 readable low-cost rule | **Done**; found a suppression term | §5.12, stage 10 |
> | N9 calibration experiment | **Done** (20 repeats; beta calibration not added) | §5.19, stage 16 |
> | N10 optimism | **Done** | §5.11, stage 9 |
> | N11 TRIPOD+AI | **Done** | Appendix A, stage 13 |
> | N12 archive and pin | **Done** | `requirements.lock.txt`, `scripts/make_archive.py`, `CITATION.cff` |
> | N13 continuous integration | **Done** | `.github/workflows/ci.yml` |
> | N14 pre-registration | Not started (depends on N2) | — |
>
> **Why N7 was dropped.** It asked for a referral-capacity constraint to turn
> the operating point into a deployment decision. That made sense when the
> study was framed as screening feasibility. After the reframe it would
> reintroduce exactly the deployment framing the paper removes, on a sample
> that is not a screening series — and its methodological content (derive the
> threshold inside training folds) is already design decision (d). Dropped
> deliberately rather than silently.
>
> Two items were added that this roadmap did not anticipate, both consequences
> of the provenance finding: the prior-work survey (stage 7) and the
> pseudo-external-validation analysis (§5.14).

Ordered by how much each would change what the study can claim, not by effort.

### Tier 1 — changes the scientific standing

These address the limitations that currently cap every conclusion. Nothing in
Tier 2 or 3 substitutes for them.

**N1. External validation on an independent cohort.**
*Why:* Every estimate is internal. This is limitation 2 and it is the single
largest gap.
*What to do:* Obtain a second CKD dataset, map its variables onto this
schema, and evaluate models trained here on it without refitting — then the
reverse direction. Report discrimination **and** calibration; expect the slope
to degrade more than the AUC, because that is what usually happens.
*Candidate:* the older and larger UCI "Chronic Kidney Disease" release (Rubini,
Soundarapandian and Eswaran; ~400 records, **continuous** values), which shares
this file's variable names. **Verify provenance before using it**: because the
column names are identical, this v2 file may be a derived or overlapping
re-release rather than an independent sample, and validating on an overlapping
cohort would be worse than not validating at all. Check record-level overlap
explicitly and report the check.
*Deliverable:* `src/ckd/data/external.py`, `scripts/07_external_validation.py`.
*Effort:* medium; data acquisition and provenance checking dominate.

**N2. A prospective consecutive screening series.**
*Why:* This is the only thing that converts a feasibility signal into evidence.
It fixes case-mix bias (limitation 4), selection bias (3), chronicity (9) and
prevalence dependence (10) simultaneously.
*Design essentials:*
- Consecutive attenders or a community sample — **not** a case-control contrast.
- The reference standard applied to **everyone**, not only to those the model
  flags, or verification bias replaces spectrum bias.
- eGFR **and** urine albumin-creatinine ratio, repeated at least three months
  apart, so the chronicity criterion is actually met.
- Deliberate early-stage enrichment: target at least half the cases at s1–s2.
- Record sex at minimum, plus enough demographics to check subgroup
  performance.
- Sample size: as a rule of thumb, 10 events per candidate predictor for 11
  predictors needs >=110 cases; at a realistic community prevalence near 10%
  that is roughly 1,100 participants. Do a formal Riley et al. calculation
  before committing — the rule of thumb is an anchor, not an answer.
- Pre-register the protocol and the analysis plan before collection starts.
*Effort:* large. It is also the only route to a clinical claim.

**N3. Recover continuous measurements.**
*Why:* Limitation 5. Everything here rests on interval midpoints and truncated
open-ended tails; the effective precision of every variable is unknown.
*What it unlocks:* checking whether the representative-value encoding costs
performance, fitting calibration on a proper scale, and defining thresholds in
clinically meaningful units (mg/dL, g/dL) rather than bin indices.
*Effort:* small if a continuous source exists; otherwise it rides on N2.

### Tier 2 — strengthens the current analysis, using data already in hand

All of these are runnable in this repository this week and each closes a
specific reviewer question.

**N4. Label-robustness sensitivity analysis.** Re-run the headline cells (a)
excluding the four internally inconsistent records (CSV lines 12, 18, 52, 123)
and (b) with their labels flipped. If conclusions move, readers need to know;
if they do not, limitation 8 is largely answered. *Cost:* ~10 minutes if
restricted to `low_cost`/`full_valid`/`laboratory` x SVM/RF x none/isotonic.

**N5. Encoding-robustness analysis.** Re-run with three alternative encodings:
bin index (pure ordinal rank), one-hot per bin, and rank-normalised. This tests
whether the midpoint-representative choice drives any result, and directly
addresses limitation 5 and 11 without needing new data. *Cost:* one extra stage
3 pass per encoding on a restricted grid.

**N6. Bootstrap the whole nested-CV procedure.** Current intervals resample
patients over the pooled out-of-fold predictions, so they capture patient
sampling but hold the fitted models fixed — they understate total uncertainty.
Refitting inside each resample gives an honest interval. *Cost:* expensive;
restrict to 3–4 headline cells and ~200 resamples.

**N7. Turn the operating point into a decision.** Decision-curve net benefit is
currently exploratory. Specify a real constraint — the referral capacity a
clinic can absorb — derive the threshold that respects it **inside training
folds**, and report the confusion matrix there. That is the number a programme
would actually act on, and it makes the missed-case versus unnecessary-referral
trade-off concrete rather than rhetorical.

**N8. Publish a readable low-cost rule alongside the SVM.** The selected model
is an RBF SVM — accurate but opaque, which sits awkwardly with "explainable" in
the title. The low-cost **EBM (a pure GAM, `interactions=0`)** reaches ROC-AUC
0.987 with isotonic calibration and a slope of **1.01** — statistically
indistinguishable from the SVM's 0.993, better calibrated, and it yields
per-variable shape functions a clinician can read and challenge. Add EBM shape
extraction to stage 5 and report an explicit "best-discriminating vs
best-explainable at equal cost" comparison. *Cost:* low; the predictions
already exist.

**N9. A dedicated calibration experiment.** The isotonic-beats-Platt result is
surprising and rests on 25 folds. Raise repeats to ~20 for a restricted grid,
and add beta calibration as a third method. *Cost:* low on a restricted grid.

**N10. Quantify optimism explicitly.** Report apparent (resubstitution) minus
nested-CV performance per cell. One extra fit per cell, and it gives readers a
direct measure of how much internal validation is already correcting for.

### Tier 3 — dissemination and engineering

**N11. Complete a TRIPOD+AI checklist** (reference 3 in the report) and include
it as an appendix table. Journals increasingly require it, and the study
already satisfies most items — the checklist mainly needs assembling.

**N12. Archive with a DOI and pin the environment.** `requirements.txt` is
pinned to the reference run, but there is no lock file or container. Add one,
then archive the repository (Zenodo or equivalent) so the analysis is citable
at a fixed state.

**N13. Continuous integration.** The repository is now under git. Add a
workflow that runs `pytest` plus stages 1, 2, 4 and 6 on every commit, with
stage 3 predictions cached as an artefact. The leakage guarantees are only
guarantees while the tests actually run.

**N14. Pre-register the prospective study** (N2) before any data is collected.

### Deliberately *not* recommended

Stating these prevents well-meant regressions:

- **No deep learning.** With 72 minority events, a neural network cannot be
  estimated reliably, and it would reintroduce exactly the over-fitting this
  study is built to detect. A test currently enforces this.
- **No chasing the last 0.007 of AUC.** Bootstrap intervals span ~0.02. Any
  tuning that closes that gap is fitting noise, and this dataset is already at
  the ceiling for reasons that have nothing to do with model quality.
- **No SMOTE or other synthetic oversampling.** The imbalance is mild (64/36)
  and synthesising minority patients at n=200 would manufacture structure.
- **Do not "clean" the four inconsistent records away.** They are evidence
  about label quality. Handle them as a sensitivity analysis (N4), not a
  deletion.
- **Never quote the leaky model outside the audit.** It is labelled INVALID in
  every table and figure for a reason.
- **Do not enable EBM interaction terms** at this sample size — pairwise terms
  would be estimated from a few dozen events each.
- **Do not select a final model on accuracy**, and do not replace the
  prespecified rule after seeing results. If the rule is wrong, say so and
  explain why, as section 5.10 of the report does.

### Suggested sequencing

| Order | Actions | Rationale |
|---|---|---|
| 1 | N4, N5, N8, N11 | Cheap, this week, and each closes a predictable reviewer question |
| 2 | N7, N9, N10, N12, N13 | Strengthen the methodology and make the repository citable and self-checking |
| 3 | N1 | The first genuine test of whether any of this transfers |
| 4 | N6 | Worth the compute only once N1 shows the result survives at all |
| 5 | N14 then N2, with N3 | The only path to a clinical claim |

**The honest summary:** Tier 2 and Tier 3 improve a feasibility study. Only
Tier 1 changes what it is allowed to conclude.
