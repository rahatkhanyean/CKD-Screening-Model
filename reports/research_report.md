# Three mechanisms inflate reported performance on a widely used public CKD benchmark

**A leakage-controlled, case-mix-aware re-analysis**

*Generated 2026-08-22 by `scripts/06_write_report.py`. Every number in this document is injected directly from the computed tables in `reports/tables/`; none is transcribed by hand.*

---

> ### Study-type and safety statement
>
> This is a **methodological re-analysis** of a public dataset, not a clinical study. Its object is the measurement process: how much of the near-perfect performance routinely reported on this benchmark is produced by properties of the data rather than by predictive ability. It is **not a diagnostic tool**, and nothing in it should be used to make decisions about any patient. No claim is made about clinical utility, causality, deployment readiness or generalisability to any population. Where models are compared, the comparison is evidence about the benchmark, not a recommendation to deploy anything.

---

## Abstract

**Background.** Machine-learning studies on the public UCI chronic kidney disease (CKD) benchmarks routinely report accuracy at or near 100%. In a structured survey of 13 such studies, every independently verified headline metric on the diagnostic task sits at or above 99%, 9 are coded as using serum creatinine or eGFR as inputs, and 1 has a validation cohort that is genuinely external. Such numbers are more often a property of the data than evidence of clinical usefulness.

**Objective.** To identify and quantify the mechanisms that produce near-perfect measured performance on this benchmark, and to establish what predictive signal remains once each is controlled or measured.

**Methods.** 200 patient records (128 CKD, 72 non-CKD) from the UCI 'Risk Factor Prediction of Chronic Kidney Disease' release. All continuous variables were already discretised into interval bins in the released file. Six feature configurations were compared, including one **deliberately invalid** set containing an exact copy of the outcome (`affected`), the post-diagnosis stage label (`stage`) and the diagnostic eGFR quantity (`grf`). 6 model families (dummy, penalised logistic regression, random forest, support vector machine, XGBoost, explainable boosting machine) were evaluated under 5-fold outer / 4-fold inner nested stratified cross-validation repeated 5 times (25 outer test sets). Imputation, scaling, tuning, calibration and threshold selection were performed strictly within training folds, and prohibited columns were blocked programmatically by a guard that raises at fit and transform time. Uncalibrated, Platt and isotonic probabilities were compared. Confidence intervals come from a stratified patient-level bootstrap (2000 resamples).

**Results.** The invalid leaky configuration reached ROC-AUC 1.000 (1.000-1.000). The best valid configuration reached 1.000 (1.000-1.000), i.e. the valid model is itself at the discrimination ceiling, so the gain from leakage measured against it is arithmetically near zero. Measured against baselines that still have headroom, leakage closes 100% of the remaining error for the low-cost set and 100% for the clinical-only set: it takes any configuration to 1.0. The low-cost configuration (history, examination and urine dipstick; 11 variables) achieved ROC-AUC 0.992 (0.981-0.999) versus 0.995 (0.988-0.999) for the laboratory-only configuration (14 variables). At the prespecified threshold of 0.50 the low-cost model reached sensitivity 0.961 (0.922-0.992) and negative predictive value 0.935 (0.878-0.986) (all figures in this paragraph: best model per configuration, uncalibrated pooled out-of-fold predictions; the prespecified selection rule, which also considers calibrated cells, is reported in the results). The calibration slope of the best valid model is **not identified**: it separates the classes completely, so the logistic recalibration has no maximum-likelihood solution (intercept 0.31, Brier 0.0036); the identified slope of the best low-cost model is 1.26. Feature-importance rankings were weakly concordant across the 25 outer folds (Kendall's W = 0.471).

**Mechanism 2, case mix, is larger than mechanism 1.** Even without any prohibited column, valid configurations sit close to the discrimination ceiling, and a post hoc case-mix analysis explains why: 107 of the 128 CKD patients (84%) are staged s3-s5, and haemoglobin alone separates the groups with univariate ROC-AUC 0.968. This is a contrast between advanced disease and comparatively healthy controls, not a screening series. Restricted to early CKD (s1-s2, n = 21) versus non-CKD, sensitivity at the prespecified threshold falls from 0.969 to 0.905 for the low-cost model and from 0.891 to 0.762 for the clinical-only model.

**Mechanism 3: the benchmark's two releases are not independent cohorts.** A record-level provenance check registered before any external claim found that all 200 patients in this file match into the 2015 UCI CKD release (maximum bipartite match fraction 1.000 against a permutation null of 0.003 +/- 0.003), 187 of them uniquely, with 0 contradictions across 10 categorical variables that took no part in the matching. This file is a discretised subset re-release of that dataset. 3 of the surveyed studies validate models across the two releases as though they were independent cohorts, and one merges them into a single training set.

**Conclusions.** Three distinct mechanisms drive measured performance on this benchmark toward 1.0, and none of them is predictive ability: target leakage, a case mix dominated by advanced disease, and validation cohorts that share their patients with the training data. Case mix is the largest of the three and the least often controlled. Under leakage-controlled validation a variable set costing no venepuncture retains much of the discrimination available from laboratory measurements, and an additive model matches it with an identified probability scale - but no figure here estimates screening performance, because this sample is not a screening series. These are internal, 200-patient, single-centre estimates; the study has no external validation and cannot obtain one from the sources examined. What it offers instead is a reusable procedure: declare prohibited columns, enforce them programmatically, test the case mix before believing the discrimination, and check that an external cohort is external.

## 1. Introduction

Chronic kidney disease affects roughly 9% of the world's population and caused an estimated 1.2 million deaths in 2017, with the burden falling disproportionately on regions where diagnostic laboratory capacity is limited [1]. CKD is defined by abnormalities of kidney structure or function present for more than three months, operationalised principally through estimated glomerular filtration rate (eGFR) and albuminuria [2]. Because early CKD is largely asymptomatic, detection depends on testing rather than on presentation, which makes the cost and availability of the test a first-order determinant of who gets diagnosed.

That clinical importance is why the public UCI CKD datasets have become standard benchmarks: they are small, tidy, and answer a question that matters. It is also why the results reported on them deserve scrutiny. A benchmark on which almost every method reports near-perfect accuracy has stopped discriminating between methods, and the interesting question becomes what, in the data, is producing the ceiling.

A secondary question follows once the ceiling is understood: how much of the discriminative information in a CKD assessment is carried by variables that cost almost nothing to obtain - history, physical examination, and a urine reagent strip - relative to variables that require venepuncture and a laboratory analyser?

There is a well-documented hazard in answering either question with machine learning on small clinical datasets, and it is not hypothetical for this data. A structured survey of prior studies on this dataset and its parent release (`reports/tables/table_26_prior_work.csv`; 13 studies, 3 independently verified so far, the coding of the remainder attributed to the comparison table of Kabir et al. [11] pending hand verification) finds 2 of the 3 independently verified headline metrics at or above 99% accuracy - 99.16% [12], and 99.5% with cross-dataset validation reaching 100% [13]; the third is an external sensitivity from the one genuinely external study [11] - with 9 of 13 studies coded as using serum creatinine or eGFR as inputs, and exactly 1 with a validation cohort that is genuinely external. Such results are rarely evidence of clinical usefulness; far more often they indicate that a predictor encodes the outcome. This dataset contains three such columns, and one of them is an exact copy of the label. A study that does not control for this cannot distinguish learning from lookup.

A further hazard is specific to this dataset's published record: 3 of the surveyed studies validate models across the 2015 UCI release and this file (its 2020 re-release) as if they were independent cohorts, and one merges them into a single training set. The provenance analysis in `reports/external/provenance_report.md` shows the two releases share their patients record-for-record, so those validations were performed on the training population - a point developed in the discussion.

This work therefore treats leakage control, honest validation and calibration as the primary objects of study, and treats predictive performance as something to be measured carefully rather than maximised. Reporting follows the spirit of TRIPOD+AI [3].

## 2. Research questions

**Primary.** Why do machine-learning studies on this benchmark routinely report accuracy at or near 100%, and what remains once the mechanisms responsible are controlled or measured?

Three candidate mechanisms are examined, in the order they were found:

1. **Target leakage.** How much does including outcome-derived columns inflate measured performance? (Section 5.2)
2. **Case-mix (spectrum) composition.** How much of the remaining performance is explained by *which patients* the sample contains rather than by what the model learned? (Section 5.5)
3. **Pseudo-external validation.** Are the cohorts used to validate models on this benchmark actually independent of the cohorts used to train them? (Section 5.14)

**Supporting questions**, addressed because they are needed to interpret the three mechanisms rather than for their own sake:

4. Under leakage-controlled validation, how much discrimination survives when expensive laboratory variables are removed - i.e. what does a cost-stratified feature set actually cost? (Sections 5.3, 5.12)
5. Are predicted probabilities calibrated, and is the probability scale even identified? (Section 5.6)
6. Are the identified important predictors stable across resampling? (Section 5.7)
7. How uncertain is any of this, given 200 patients? (Sections 5.8, 5.11, 5.13)

## 3. Dataset

### 3.1 Source and provenance

The analysis uses `ckd-dataset-v2.csv`, 200 patient records distributed by the UCI Machine Learning Repository as *Risk Factor Prediction of Chronic Kidney Disease* (creators Md. Ashiqul Islam and Shamima Akter), under CC BY 4.0 [4]. Its documentation states that the records were collected at Enam Medical College, Savar, Dhaka, Bangladesh.

**That documented provenance is contradicted by the data itself.** Section 5.14 reports a record-level check, run before any external claim was permitted, which finds every one of these patients present in the 2015 UCI CKD release (documented as Apollo Hospitals, India), 187 of them matching a single record uniquely and agreeing on ten categorical variables that took no part in the matching. The file analysed here is a discretised subset re-release of that earlier dataset. We report the measurement and take no view on how the discrepancy arose; readers should treat the stated collection site, country and year as unverified, and this study makes no claim that rests on them.

- SHA-256 of the analysed file: `f24075f420b0f271bfddf3844a40061dbe9e2cb1c2336daa64463122344ea84a`
- Shape as read (header excluded): 202 x 29
- After removing 2 metadata rows: **200 patients x 29 columns**

The two rows immediately below the header are metadata, not observations: the first contains only the token `discrete` in every column, and the second declares `class` as the target and `meta` for `age`. Both are removed in exactly one place in the codebase (`src/ckd/data/load.py`), and the classification is verified rather than assumed.

### 3.2 The decisive property: the data are already discretised

Every continuous variable in the released file has been binned into interval strings before publication - `sg` as `1.019 - 1.021`, `bgr` as `< 112`, `grf` as `>= 227.944`. **The original measurements are not recoverable.** This information loss is a property of the published dataset, not a modelling choice, and it constrains everything that can be concluded from it. In particular, no analysis of this file can recover the resolution that a real eGFR or creatinine value would provide, and the effective measurement precision of every predictor is unknown.

Each interval label is mapped to a single representative number: the midpoint for a closed bin, and the finite edge for an open-ended one. The mapping is a pure function of one cell's text - it uses no cross-row statistics and no outcome information - so it cannot transfer information between cross-validation folds, which is why it is the only transformation applied before splitting.

### 3.3 Outcome and class balance

The outcome is `class` (`ckd` / `notckd`): **128 CKD (64.0%) and 72 non-CKD (36.0%)**. The minority class provides 72 events; with 25 candidate predictors in the full valid configuration this is 2.88 events per predictor, far below both the traditional rule of thumb of 10 and the requirements of modern sample-size calculations for prediction models [5]. The study is therefore underpowered by construction, and is presented as a feasibility and methodology exercise rather than as model development.

### 3.4 Data quality

The full audit is in `reports/data_quality_report.md`. Summary:

| Check | Result |
|---|---|
| Missing cells after cleaning | 1 |
| Exact duplicate records | 0 |
| Duplicate predictor patterns | 0 |
| Near-constant features (modal >= 90%) | 2 |
| Bin-definition anomalies | 1 |
| Clinical inconsistencies flagged | 2 |
| Variables with uncertain interpretation | 10 |

- **`ba`** is near-constant: 94.5% of patients share the value `0`.
- **`pot`** is near-constant: 98.5% of patients share the value `< 7.31`.
- **`su`** has internally inconsistent published bin edges (overlaps: []; tied representatives: [['4 - 4', '>= 4']]), so its ordinal encoding contains ties. Reported, not silently repaired.
- **`pot`** contains 1 value(s) in band `7.31 - 11.72`: Serum potassium far outside any survivable range (normal 3.5-5.1 mEq/L); almost certainly a recording or unit error.
- **`pot`** contains 1 value(s) in band `38.18 - 42.59`: Serum potassium far outside any survivable range (normal 3.5-5.1 mEq/L); almost certainly a recording or unit error.
- **`pot`** contains 1 value(s) in band `>= 42.59`: Serum potassium far outside any survivable range (normal 3.5-5.1 mEq/L); almost certainly a recording or unit error.

- **Patients labelled 'notckd' but assigned an advanced CKD stage (s3-s5)** - 4 patients (CSV lines [12, 18, 52, 123]). Either the outcome label or the stage assignment is wrong for these records. Reported, not corrected: we have no external source of truth to arbitrate.
- **Patients labelled 'notckd' whose eGFR band is below 60 mL/min/1.73m2** - 4 patients (CSV lines [12, 18, 52, 123]). An eGFR below 60 sustained over three months is itself a CKD-defining criterion, so these labels are internally inconsistent with the eGFR column. Chronicity cannot be verified from a single cross-sectional record, which is one possible benign explanation.

#### The `" p "` value in `grf`

Exactly 1 cell in the eGFR column contains the token `" p "` (the letter p with surrounding whitespace), which is not an interval label.
- CSV line **183**, column `grf`, raw value `' p '`
- The affected patient is labelled `ckd` and staged `s5`; every other patient at that stage falls in the eGFR band(s) `['< 26.6175']`.

The most parsimonious reading is a data-entry error. Because the true value cannot be recovered, the cell is treated as **missing** and imputed using the median of the relevant training fold only. The raw file is preserved unmodified at `data/raw/`. Note that `grf` is excluded from every clinically valid configuration, so this defect can affect only the deliberately invalid leaky model.

### 3.5 Leakage structure

- **`affected` is an exact one-to-one copy of the outcome** (verified across all 200 rows; Cramer's V = 1.0). It is the target, renamed.
- **`stage`** is a post-diagnosis staging label. Cramer's V with the outcome = 0.7548. Proportion CKD by stage:

  | stage | n | proportion CKD |
  |---|---:|---:|
  | s1 | 54 | 0.167 |
  | s2 | 35 | 0.343 |
  | s3 | 31 | 1.000 |
  | s4 | 45 | 0.911 |
  | s5 | 35 | 1.000 |

- **`grf`** is eGFR, the quantity from which `stage` is banded and by which CKD is defined (Cramer's V with `stage` = 0.7351). Using it to predict CKD approaches using the diagnostic criterion as a predictor.

Stages s3 and s5 are 100% CKD. Any model given these columns is performing a lookup, not a prediction.

### 3.6 Variables whose meaning could not be established

The released dataset ships **no data dictionary** [4]. Rather than inventing clinical interpretations, variables whose meaning or provenance cannot be settled from the file are flagged explicitly:

- **`bp (Diastolic)`** (exam) - The column is named for diastolic pressure but ships as a binary 0/1 flag, so the underlying mmHg value and the threshold that produced the flag are both unrecoverable. The direction of the coding (1 = elevated) is inferred from its association with 'bp limit' and 'htn', not from documentation.
- **`bp limit`** (exam) - Undocumented. Empirically it refines 'bp (Diastolic)': every patient with bp (Diastolic)=0 has bp limit=0, while bp (Diastolic)=1 splits across all three levels. It is therefore most likely a graded severity band, but the mmHg cut-points are unknown and are NOT reconstructed here.
- **`dm`** (history) - A first-time diabetes diagnosis requires a glucose assay, so in a population with poor prior healthcare contact this variable would be less freely available than assumed here.
- **`cad`** (history) - Establishing CAD de novo is expensive. Its value as a *screening* predictor depends on patients already knowing the diagnosis.
- **`appet`** (history) - Polarity (whether 1 denotes poor or good appetite) is not documented in the released file.
- **`ane`** (blood_lab) - IMPORTANT AND CONSEQUENTIAL. Anaemia can also be recorded clinically (conjunctival or palmar pallor) at no cost. We verified that 'ane' is NOT a deterministic function of the 'hemo' bins (for example 25 patients with hemo 10-11.3 are ane=0 while 3 are ane=1), so it is not a pure re-coding of haemoglobin. Because provenance cannot be established from the file, it is EXCLUDED from the low-cost configuration. This is the conservative choice: it can only understate low-cost performance, never inflate it.
- **`su`** (urine_dip) - The published bin edges for this column are internally inconsistent and overlapping ('1 - 2' vs '2 - 2'; '3 - 4' vs '4 - 4' vs '>= 4'), so its ordinal encoding contains ties. Reported as a data-quality defect; not silently repaired.
- **`rbc`** (urine_micro) - Could in principle refer to a blood-count red-cell abnormality rather than urinary red cells, but its position among the urinalysis columns and the presence of a separate 'rbcc' (red blood cell count) column make urinary microscopy the far more likely reading.
- **`sc`** (blood_lab) - Serum creatinine is the input to the eGFR equation that defines 'grf' and 'stage'. It is retained in the valid laboratory configuration because it is a genuine, routinely measured screening analyte rather than a diagnostic label, but it is the variable most likely to behave as a near-proxy for the outcome, and results are interpreted with that in mind.
- **`pot`** (blood_lab) - Near-constant: 197 of 200 patients fall in the single bin '< 7.31'. The upper bins ('38.18 - 42.59', '>= 42.59') lie far outside any survivable serum potassium range and are almost certainly recording errors. Flagged, retained as-is, and carries essentially no usable information.

The most consequential of these is `ane`. Anaemia can be recorded clinically at no cost, or read from a full blood count. We verified that `ane` is **not** a deterministic function of the `hemo` bins, so it is not a pure recoding of haemoglobin - but its provenance remains unresolved. It is therefore **excluded from the low-cost configuration**, which is the conservative choice: it can only understate low-cost performance, never inflate it.

## 4. Methods

### 4.1 Feature configurations

Cost tiering is an explicit, auditable judgement recorded in `src/ckd/features/configs.py` with a written rationale per variable, not a fact extracted from the file. Six configurations were compared:

| Configuration | k | Valid? | Contents |
|---|---:|---|---|
| `leaky_model` | 28 | **NO** | `age`, `bp (Diastolic)`, `bp limit`, `htn`, `dm`, `cad`, `appet`, `pe`, `sg`, `al`, `su`, `rbc`, `pc`, `pcc`, `ba`, `ane`, `bgr`, `bu`, `sc`, `sod`, `pot`, `hemo`, `pcv`, `rbcc`, `wbcc`, `grf`, `stage`, `affected` |
| `full_valid_model` | 25 | yes | `age`, `bp (Diastolic)`, `bp limit`, `htn`, `dm`, `cad`, `appet`, `pe`, `sg`, `al`, `su`, `rbc`, `pc`, `pcc`, `ba`, `ane`, `bgr`, `bu`, `sc`, `sod`, `pot`, `hemo`, `pcv`, `rbcc`, `wbcc` |
| `low_cost_model` | 11 | yes | `age`, `bp (Diastolic)`, `bp limit`, `htn`, `dm`, `cad`, `appet`, `pe`, `sg`, `al`, `su` |
| `laboratory_model` | 14 | yes | `rbc`, `pc`, `pcc`, `ba`, `ane`, `bgr`, `bu`, `sc`, `sod`, `pot`, `hemo`, `pcv`, `rbcc`, `wbcc` |
| `clinical_only_model` | 8 | yes | `age`, `bp (Diastolic)`, `bp limit`, `htn`, `dm`, `cad`, `appet`, `pe` |
| `low_cost_plus_urine_micro_model` | 15 | yes | `age`, `bp (Diastolic)`, `bp limit`, `htn`, `dm`, `cad`, `appet`, `pe`, `sg`, `al`, `su`, `rbc`, `pc`, `pcc`, `ba` |

- `leaky_model` is **deliberately invalid**. It exists only to quantify leakage-driven inflation and its results must never be read as evidence of clinical usefulness.
- `clinical_only_model` and `low_cost_plus_urine_micro_model` are **flagged sensitivity analyses**. They exist because the boundary of 'low cost' is a genuine judgement call - specifically, whether urine microscopy (non-invasive but requiring a microscope and a technician) belongs inside it. Rather than deciding that silently, both sides of the boundary are reported.

### 4.2 Models

| Model | Notes |
|---|---|
| Dummy (prevalence) | Baseline: always predicts the training-fold prevalence. |
| Logistic regression (L2) | Regularised linear baseline. Because the predictors are ordinal bin representatives, this assumes an approximately linear effect of the bin scale on the log-odds. |
| Random forest | 300 trees; 8 candidates, kept small because inner folds hold ~32 patients. |
| Support vector machine (RBF) | Requires scaling; scaler is fitted inside training folds only. |
| XGBoost | Shallow trees (depth 2-3) to limit variance at n=200. |
| Explainable boosting machine (GAM) | Main effects only (interactions=0); shape functions are directly readable. |

**Neural networks and deep learning were deliberately excluded.** With 72 minority-class events and at most 25 predictors, a deep model cannot be estimated reliably, and fitting one would invite exactly the over-fitting this study is designed to detect.

XGBoost was used rather than CatBoost. Both were installed and benchmarked; XGBoost fitted roughly three times faster on this data, which made the full repeated nested design feasible. The choice is about compute, not expected accuracy. The explainable boosting machine was configured with `interactions=0` (a pure generalised additive model), `outer_bags=6` and `max_rounds=1500`: the library defaults cost ~8.7 s per fit here while adding capacity this sample cannot support. Both are stated as compute/regularisation trade-offs, not as neutral defaults.

Search spaces were kept small (at most 8 candidates per model) because inner folds hold roughly 32 patients; a large grid searched on folds that size selects on noise.

### 4.3 Validation design

| Element | Value |
|---|---|
| Outer loop | 5-fold stratified CV |
| Inner loop | 4-fold stratified CV |
| Repeats | 5 |
| Independent outer test sets | 25 |
| Inner selection metric | roc_auc |
| Master seed | 20240517 |
| Grid size | 6 configurations x 6 models x 3 calibration methods |
| Runtime | 71.89 min |

5 repeats were chosen as a documented compromise: 25 outer test sets stabilise the mean of fold-level metrics while keeping the full grid runnable in about 72 minutes on 12 CPU cores. Increasing repeats reduces cross-validation partition noise but **cannot** reduce the dominant source of uncertainty here, which is the 200-patient sample itself; that is quantified separately by patient-level bootstrap.

Within each outer fold, in order: hyper-parameters are selected by grid search over the inner folds of the outer training data; the selected pipeline is refitted on the outer training data; probability calibration is fitted on the outer training data via `CalibratedClassifierCV` with its own internal stratified splits; the screening threshold is chosen from a further cross-validated prediction on the outer training data. Only then is `predict_proba` called on the outer test fold. Nothing - imputation, scaling, tuning, calibration, threshold - is computed on the outer test fold or on the complete dataset [6].

A single train/test split was not used as the principal evaluation. SMOTE was not used: the class ratio is roughly 1.8:1, which does not warrant synthetic oversampling, and introducing it would add a resampling artefact without a specific question to justify it.

### 4.4 Leakage prevention, enforced programmatically

Three independent layers, each covered by automated tests:

1. **Declaration.** Prohibited columns (`class`, `affected`, `stage`, `grf`) and the contents of every configuration are declared in code.
2. **Runtime guard.** Every pipeline begins with a `LeakageGuard` that raises `LeakageError` at *fit and at transform time* if a prohibited column is present. It raises rather than dropping the column, so a bug in calling code cannot be silently absorbed. The invalid configuration must opt in explicitly, and even then the raw outcome column can never pass.
3. **Pre-flight assertion.** `assert_pipeline_is_clean` is run over all 36 pipelines before any fitting begins.

The key adversarial test hands a valid pipeline the **entire** dataframe including `affected`, `stage` and `grf` and asserts that it **refuses**. A separate test confirms that scaler and imputer statistics differ between folds, i.e. that preprocessing was not fitted on complete data.

### 4.5 Evaluation

Reported: ROC-AUC, PR-AUC, sensitivity/recall for CKD, specificity, precision (PPV), negative predictive value, F1, balanced accuracy, Brier score, calibration slope and intercept, expected calibration error, and the confusion matrix at two operating points.

- **Primary threshold: 0.5**, prespecified in `config/experiment.yaml` before any result was computed.
- **Screening threshold**: the highest threshold still achieving 90% sensitivity on inner cross-validated predictions **of the outer training fold**, recomputed independently in every fold. It never sees outer-test outcomes.

Calibration slope and intercept follow the standard logistic recalibration framework [7]: the slope is the coefficient of `logit(p)` in an unpenalised logistic refit, and the intercept is obtained with `logit(p)` as a fixed offset. Slope < 1 indicates predictions that are too extreme. Where the predictions separate the classes completely the maximum-likelihood slope does not exist, and it is reported as undefined rather than as whatever value an optimiser happened to reach.

**Caution stated in advance about isotonic calibration.** Isotonic regression fits a free-form monotone step function and is therefore far more flexible than Platt scaling. With roughly 160 patients available inside each training fold, it has enough freedom to follow noise, and the usual expectation at this sample size is that it will overfit and perform worse than the parametric alternative. It is included precisely so that this expectation is tested rather than assumed, and the result is reported in section 5.6 whichever way it falls.

Uncertainty is reported two ways, deliberately kept separate: the spread across the 25 outer folds and across the 5 repeats (cross-validation partition variability), and a stratified patient-level bootstrap (2000 resamples, 95% percentile intervals). Conflating them would overstate precision.

A decision-curve net-benefit analysis [8] is reported as an **exploratory** supplement.

## 5. Results

### 5.1 Baseline

The dummy classifier, which always predicts the training-fold prevalence, achieved ROC-AUC 0.472 at best - chance, as required. Every result below should be read against that floor and against the 64.0% prevalence.

### 5.2 Leakage audit (secondary question 1)

Best model per configuration, uncalibrated, pooled out-of-fold predictions, with 95% bootstrap confidence intervals:

| Configuration | k | Best model | ROC-AUC (95% CI) | PR-AUC | Sensitivity | Specificity | Brier |
|---|---:|---|---|---|---|---|---|
| Leaky (INVALID) | 28 | XGBoost | 1.000 (1.000-1.000) | 1.000 | 1.000 | 1.000 | 0.000 |
| Full valid | 25 | SVM (RBF) | 1.000 (1.000-1.000) | 1.000 | 0.977 | 1.000 | 0.007 |
| Laboratory | 14 | SVM (RBF) | 0.995 (0.988-0.999) | 0.997 | 0.969 | 0.972 | 0.028 |
| Low-cost + urine microscopy | 15 | Random forest | 0.992 (0.980-0.999) | 0.996 | 0.961 | 1.000 | 0.033 |
| Low-cost | 11 | Random forest | 0.992 (0.981-0.999) | 0.996 | 0.961 | 1.000 | 0.030 |
| Clinical only | 8 | EBM (GAM) | 0.953 (0.919-0.979) | 0.979 | 0.914 | 0.986 | 0.058 |

The invalid configuration reaches ROC-AUC 1.000 with sensitivity 1.000 and specificity 1.000. Measured against the best valid configuration the gain is only 0.000, but that is **not** evidence that leakage is unimportant here: the valid configuration is itself at 1.000, so there is essentially no headroom left for leakage to exploit. Section 5.5 explains why the valid model is already at the ceiling, and the table below re-measures the effect against baselines that are not saturated.

**Why apparently perfect performance signals leakage rather than usefulness.** `affected` reproduces the outcome exactly, so any model given it needs only to copy one column; `stage` is assigned after diagnosis and is 100% CKD in stages s3 and s5; `grf` is the eGFR value from which staging is derived and by which CKD is defined. A model using them is not forecasting an unknown state from clinical signs - it is reading back a conclusion that was already recorded. Such a model would have nothing to contribute at the moment of screening, because at that moment none of these three quantities exists yet. The practical test is temporal: a predictor that could not have been measured *before* the diagnosis cannot support screening, however well it scores.

This is directly relevant to the published record on this dataset, where accuracies at or near 100% are commonly reported.

**An important caveat on how to read the inflation number.** The absolute ROC-AUC gain from leakage looks modest here, but that is a ceiling artefact, not evidence that leakage is harmless: the valid baseline is already close to 1.0, so there is almost no room left to gain. Measured against baselines that are not saturated:

| Baseline configuration | Baseline ROC-AUC | Headroom to 1.0 | Absolute inflation | Share of headroom closed by leakage |
|---|---:|---:|---:|---:|
| Full valid | 1.000 | 0.000 | 0.000 | undefined (no headroom) |
| Low-cost | 0.992 | 0.008 | 0.008 | 100.0% |
| Clinical only | 0.953 | 0.047 | 0.047 | 100.0% |

The honest summary is that leakage takes any configuration to the ceiling. Where a valid model already sits at the ceiling for other reasons - which, as section 5.5 shows, is the case here - the gain is arithmetically small while the epistemic problem is unchanged.

### 5.3 Low-cost versus laboratory (primary question, secondary question 2)

Best model per configuration, **uncalibrated**, pooled out-of-fold predictions (the same cells as the leakage audit above). The prespecified selection rule in section 5.9 instead selects over all (model x calibration) cells, so its chosen model and its numbers can differ slightly from this table's; every quoted figure names its cell for that reason:

| Configuration | k | Best model | ROC-AUC (95% CI) | Sensitivity | Specificity | NPV | PPV | Brier |
|---|---:|---|---|---|---|---|---|---|
| Full valid | 25 | SVM (RBF) | 1.000 (1.000-1.000) | 0.98 (0.95-1.00) | 1.00 (1.00-1.00) | 0.96 (0.91-1.00) | 1.00 (1.00-1.00) | 0.007 |
| Laboratory | 14 | SVM (RBF) | 0.995 (0.988-0.999) | 0.97 (0.94-0.99) | 0.97 (0.93-1.00) | 0.95 (0.89-0.99) | 0.98 (0.96-1.00) | 0.028 |
| Low-cost + urine microscopy | 15 | Random forest | 0.992 (0.980-0.999) | 0.96 (0.92-0.99) | 1.00 (1.00-1.00) | 0.94 (0.88-0.99) | 1.00 (1.00-1.00) | 0.033 |
| Low-cost | 11 | Random forest | 0.992 (0.981-0.999) | 0.96 (0.92-0.99) | 1.00 (1.00-1.00) | 0.94 (0.88-0.99) | 1.00 (1.00-1.00) | 0.030 |
| Clinical only | 8 | EBM (GAM) | 0.953 (0.919-0.979) | 0.91 (0.86-0.96) | 0.99 (0.96-1.00) | 0.87 (0.80-0.93) | 0.99 (0.97-1.00) | 0.058 |

The low-cost set (11 variables, no venepuncture) reaches ROC-AUC 0.992 against 0.995 for the laboratory set (14 variables) - a gap of 0.003. The bootstrap confidence intervals overlap substantially, so this comparison is not statistically resolved at n=200; the study cannot distinguish the two configurations reliably.

Dropping urine testing entirely (`clinical_only_model`, 8 variables) gives ROC-AUC 0.953, a loss of 0.039 relative to the low-cost set. Adding urine microscopy to the low-cost set gives 0.992, a change of -0.0002. The judgement call about where urine microscopy belongs therefore does not materially change the conclusions.

### 5.4 Screening operating points

Low-cost model (SVM (RBF), Isotonic), pooled out-of-fold predictions for all 200 patients:

| Operating point | Threshold | TP | FP | TN | FN | Sensitivity | Specificity | PPV | NPV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Prespecified (0.50) | 0.500 | 124 | 0 | 72 | 4 | 0.97 | 1.00 | 1.00 | 0.95 |
| Screening (training-fold selected) | 0.967 | 118 | 0 | 72 | 10 | 0.92 | 1.00 | 1.00 | 0.88 |

At the prespecified threshold the low-cost model misses **4 of 128 CKD patients** while referring **0 of 72 non-CKD patients** unnecessarily. Moving to the training-fold-selected screening threshold changes this to 10 missed and 0 unnecessary referrals.

For a screening instrument the relevant question is not which threshold maximises accuracy but what exchange rate between missed cases and unnecessary referrals is acceptable in the intended setting - a judgement that depends on referral capacity and on the consequences of delayed diagnosis, neither of which this dataset contains. Figure R7 presents the full trade-off curve rather than a single recommended threshold.

**Negative predictive value is prevalence-dependent.** The values above are computed at the 64.0% CKD prevalence of this hospital-based sample. In a community screening population, where CKD prevalence would be far lower, NPV would be substantially higher and PPV substantially lower for the same model. None of the predictive values reported here transfer to a different prevalence.

### 5.5 Case mix: why the valid models are already near the ceiling (EXPLORATORY, post hoc)

The valid configurations reach discrimination close to 1.0 without any prohibited column. That needs explaining before it is treated as good news, and the explanation is in the composition of the sample.

**Finding 1: single predictors already separate the groups.** `hemo` alone achieves univariate ROC-AUC 0.968. Its value ranges are 6.10-14.55 in CKD patients and 11.95-16.50 in non-CKD patients; only 47% of patients fall in the range the two classes share. No multivariable learning is required to separate groups that are already this far apart.

| Predictor | Univariate ROC-AUC | CKD range | Non-CKD range | Patients in shared range |
|---|---:|---|---|---:|
| `hemo` | 0.968 | 6.10-14.55 | 11.95-16.50 | 47% |
| `pcv` | 0.949 | 17.90-47.15 | 39.35-49.10 | 51% |
| `sg` | 0.888 | 1.01-1.02 | 1.02-1.02 | 58% |
| `rbcc` | 0.876 | 2.69-7.41 | 4.17-6.53 | 84% |
| `al` | 0.828 | 0.00-4.00 | 0.00-0.00 | 58% |
| `htn` | 0.805 | 0.00-1.00 | 0.00-0.00 | 61% |
| `sod` | 0.798 | 118.00-158.00 | 135.50-150.50 | 86% |
| `bu` | 0.781 | 48.10-352.90 | 48.10-67.15 | 80% |

**Finding 2: the cohort is dominated by advanced disease.** 107 of the 128 CKD patients (84%) are staged s3-s5. Only about 21 are early-stage. That is the profile of a case-control-like contrast between established, moderate-to-severe CKD and comparatively healthy controls - not of a consecutive screening series, in which most true cases would be early, asymptomatic and biochemically near-normal. Anaemia, low haematocrit and abnormal urine concentration are consequences of established kidney disease; they are exactly the features that are *absent* in the patients a screening programme most needs to identify.

**Finding 3: performance degrades in the screening-relevant subgroup.** Re-evaluating the *same* out-of-fold predictions within stage-defined subgroups (`stage` is used only to partition patients for evaluation and never entered any model as a predictor):

| Configuration | Subgroup | n CKD | ROC-AUC | Sensitivity @ 0.5 | NPV |
|---|---|---:|---:|---:|---:|
| Full valid | All patients (as sampled) | 128 | 1.000 | 0.98 | 0.96 |
| Full valid | Advanced CKD (s4-s5) | 76 | 1.000 | 0.99 | 0.99 |
| Full valid | Early CKD (s1-s2) | 21 | 1.000 | 0.90 | 0.97 |
| Laboratory | All patients (as sampled) | 128 | 0.995 | 0.97 | 0.95 |
| Laboratory | Advanced CKD (s4-s5) | 76 | 0.997 | 0.99 | 0.99 |
| Laboratory | Early CKD (s1-s2) | 21 | 0.983 | 0.86 | 0.96 |
| Low-cost | All patients (as sampled) | 128 | 0.993 | 0.97 | 0.95 |
| Low-cost | Advanced CKD (s4-s5) | 76 | 0.993 | 0.97 | 0.97 |
| Low-cost | Early CKD (s1-s2) | 21 | 0.983 | 0.90 | 0.97 |
| Clinical only | All patients (as sampled) | 128 | 0.954 | 0.89 | 0.84 |
| Clinical only | Advanced CKD (s4-s5) | 76 | 0.957 | 0.92 | 0.92 |
| Clinical only | Early CKD (s1-s2) | 21 | 0.919 | 0.76 | 0.93 |

Sensitivity is lower in the early-CKD subgroup for 4 of the 4 configurations examined. The largest decline is for Clinical only, from 0.891 across all patients to 0.762 among early-stage CKD - so roughly 24% of exactly the cases a screening programme exists to find are missed at the prespecified threshold, while its ROC-AUC in that subgroup is still 0.919. The dissociation matters: a near-ceiling AUC can coexist with materially worse case detection, and no aggregate metric reveals it.

The full valid configuration is the exception: it separates the classes completely in **every** subgroup, including early CKD (ROC-AUC 1.000, sensitivity 0.905). That is not reassurance. With only 21 early-stage cases, and with a comparison group of 72 patients who were apparently healthy enough to be labelled non-CKD despite presenting at a tertiary hospital, perfect separation says more about how far apart the two groups are in this sample than about the difficulty of the screening task.

These subgroup analyses are **exploratory and post hoc**, and the early CKD subgroup contains only 21 cases, so every estimate in it is imprecise and none of them would survive a demand for statistical significance. They are reported because the direction of the effect is consistent across the cheaper configurations, because the underlying case-mix imbalance is a hard fact about the sample rather than an inference, and because omitting the analysis would leave the headline numbers looking far more like screening performance than they are.

### 5.6 Calibration (secondary question 3)

Median across all valid configuration x model cells:

| Calibration | Brier | Slope (ideal 1) | Intercept (ideal 0) | ECE |
|---|---:|---:|---:|---:|
| Uncalibrated | 0.037 | 1.83 | 0.27 | 0.046 |
| Platt / sigmoid | 0.036 | 2.25 | 0.05 | 0.072 |
| Isotonic | 0.029 | 1.71 | 0.11 | 0.028 |

By median Brier score the best-performing calibration method across valid cells was **Isotonic**.

Isotonic regression did not underperform Platt scaling here (median Brier 0.029 vs 0.036). This is somewhat contrary to the usual expectation at n=200 and should be treated with caution: with only 25 outer folds the comparison is itself noisy.

**2 of 75 valid cells have an undefined calibration slope.** These are cells whose predictions separate the two classes completely, so the logistic recalibration model has no maximum-likelihood solution - the likelihood keeps increasing as the coefficient grows. An optimiser will still return a number (we observed values in the hundreds before adding the check), but it measures where the solver stopped, not calibration. Those cells are reported as undefined rather than given a spurious value.

The model selected by the headline rule (Full valid, SVM (RBF), Isotonic) is one of those cases: it separates the classes completely, so its calibration slope is not identified. Its Brier score is 0.0036 and its calibration intercept 0.31. Perfect separation is precisely the situation in which the probability scale is least trustworthy: the model has no data telling it what a genuinely ambiguous patient looks like, so its intermediate probabilities are extrapolation. For a screening application this is a reason to prefer a model that does *not* separate perfectly.

The low-cost model (SVM (RBF), Isotonic) does have an identified slope of 1.26 with intercept 0.06. A slope above 1 means its predictions are too conservative - it under-uses its own signal, which at this sample size is the safer direction of error.

Across valid configurations every identified slope is above 1, i.e. the models are systematically **under**-confident rather than over-confident. That is the opposite of the usual small-sample pattern and follows from the same near-separability discussed in section 5.5: regularised models trained on 160 patients produce middling probabilities for patients the data actually separate cleanly.

### 5.7 Feature importance and stability (secondary question 4)

For the selected model (Full valid, SVM (RBF)), across 25 outer folds:

- Kendall's W (concordance of the full ranking) = **0.471**
- Mean pairwise Jaccard overlap of the top-5 sets = **0.341**

| Feature | Tier | Mean permutation importance | SD | Mean rank | Top-5 frequency |
|---|---|---:|---:|---:|---:|
| `ba` | urine_micro | 0.0156 | 0.0171 | 6.5 | 0.60 |
| `bgr` | blood_lab | 0.0136 | 0.0141 | 6.5 | 0.52 |
| `sg` | urine_dip | 0.0134 | 0.0090 | 4.3 | 0.80 |
| `hemo` | blood_lab | 0.0120 | 0.0079 | 4.8 | 0.68 |
| `pcv` | blood_lab | 0.0096 | 0.0080 | 6.0 | 0.52 |
| `bu` | blood_lab | 0.0095 | 0.0124 | 9.2 | 0.44 |
| `sc` | blood_lab | 0.0093 | 0.0142 | 11.2 | 0.36 |
| `su` | urine_dip | 0.0079 | 0.0155 | 9.3 | 0.20 |
| `pot` | blood_lab | 0.0055 | 0.0071 | 14.1 | 0.32 |
| `sod` | blood_lab | 0.0045 | 0.0042 | 10.2 | 0.20 |
| `wbcc` | blood_lab | 0.0040 | 0.0062 | 13.6 | 0.16 |
| `rbcc` | blood_lab | 0.0035 | 0.0034 | 10.9 | 0.04 |

Features reaching the top 5 in at least 80% of folds: `sg`.

Features that appear in the top 5 only intermittently (20-80% of folds) - i.e. features whose apparent importance is not reproducible: `ba`, `bgr`, `hemo`, `pcv`, `bu`, `sc`, `pot`.

**The low-cost configuration is markedly more stable than the full one.** For Low-cost (SVM (RBF)), features reaching the top 5 in at least 80% of folds: `sg`, `al`, `dm`, `htn`.

| Feature | Tier | Mean permutation importance | SD | Top-5 frequency |
|---|---|---:|---:|---:|
| `sg` | urine_dip | 0.1512 | 0.0374 | 1.00 |
| `al` | urine_dip | 0.0814 | 0.0449 | 1.00 |
| `dm` | history | 0.0494 | 0.0335 | 0.96 |
| `htn` | history | 0.0405 | 0.0372 | 0.96 |
| `appet` | history | 0.0152 | 0.0194 | 0.16 |
| `su` | urine_dip | 0.0116 | 0.0148 | 0.20 |
| `pe` | exam | 0.0098 | 0.0124 | 0.16 |
| `cad` | history | 0.0096 | 0.0220 | 0.08 |

This is the expected pattern and a reassuring one: with 25 highly correlated predictors the model can substitute one laboratory variable for another between folds, so no single one dominates reliably. With 11 largely non-redundant predictors the ranking settles down. It also means the low-cost model is the more *interpretable* of the two, independently of which discriminates better.

**SHAP versus permutation importance.** The model selected by the headline rule (SVM (RBF)) admits no exact SHAP explainer, so SHAP was additionally computed for the best SHAP-capable valid model (Full valid, Logistic regression). The two measures agree closely (Spearman rho = 0.905 across features), which is evidence that the ranking reflects the model rather than the quirks of one attribution method. Where a technique did not apply it was recorded as such rather than silently omitted; see `reports/tables/table_20_importance_notes.txt`.

**Interpretation limits.** These quantities describe how a fitted model uses a column given the other columns present. They are not effect sizes, and they are **not causal**. A variable can rank highly because it proxies something else in the dataset, and a genuinely important variable can rank low if a correlated variable absorbs its signal. Permutation importance was computed on outer test folds; because it is used only for reporting, and never to select features, models or thresholds, it does not contaminate the validation.

### 5.8 Uncertainty (secondary question 5)

| Configuration | ROC-AUC | 95% bootstrap CI | CI width | SD across repeats |
|---|---:|---|---:|---:|
| Full valid | 1.000 | (1.000-1.000) | 0.000 | 0.0000 |
| Laboratory | 0.995 | (0.988-0.999) | 0.011 | 0.0013 |
| Low-cost + urine microscopy | 0.992 | (0.980-0.999) | 0.020 | 0.0034 |
| Low-cost | 0.992 | (0.981-0.999) | 0.019 | 0.0029 |
| Clinical only | 0.953 | (0.919-0.979) | 0.061 | 0.0045 |

Bootstrap intervals for ROC-AUC span roughly **0.02** on average. Differences between valid configurations smaller than that are not resolvable with 200 patients. The spread across repeats is much smaller than the bootstrap interval, which is the expected pattern and an important one: it shows that the uncertainty here is driven by the **sample**, not by the cross-validation partitioning. Running more repeats would tighten the former and do nothing about the latter.

The bootstrap intervals themselves are **optimistically narrow**. They resample the same 200 patients that were used to fit the models whose predictions are being resampled, so they capture estimation noise but not the variation that would arise in a genuinely new population.

### 5.9 Model selection

Selection rule, fixed in advance: among cells with a valid feature configuration and a non-dummy model, take the highest mean ROC-AUC, breaking ties on the lower Brier score. Applied mechanically, this selects:

- **Overall best valid model:** Full valid / SVM (RBF) / Isotonic - ROC-AUC 1.000, sensitivity 1.00, NPV 1.00, Brier 0.004
- **Best low-cost model:** SVM (RBF) / Isotonic - ROC-AUC 0.993, sensitivity 0.97, NPV 0.95, Brier 0.019

Selection was **not** made on accuracy. Because the intended use is screening, sensitivity, negative predictive value and calibration were the quantities examined, with discrimination used only as the ranking criterion and calibration as the tie-break.

Hyper-parameter selection was itself unstable across folds, which is worth recording:

| Model | Hyper-parameter | Modal value | Selection frequency | Distinct values chosen |
|---|---|---|---:|---:|
| EBM (GAM) | `learning_rate` | `0.01` | 0.52 | 2 |
| Logistic regression | `C` | `0.01` | 0.96 | 2 |
| Logistic regression | `class_weight` | `None` | 1.00 | 1 |
| Random forest | `class_weight` | `None` | 0.64 | 2 |
| Random forest | `max_depth` | `3` | 0.80 | 2 |
| Random forest | `max_features` | `sqrt` | 1.00 | 1 |
| Random forest | `min_samples_leaf` | `1` | 1.00 | 1 |
| SVM (RBF) | `C` | `0.1` | 1.00 | 1 |
| SVM (RBF) | `class_weight` | `None` | 0.92 | 2 |
| SVM (RBF) | `gamma` | `scale` | 1.00 | 1 |
| XGBoost | `max_depth` | `2` | 0.76 | 2 |
| XGBoost | `colsample_bytree` | `0.8` | 1.00 | 1 |
| XGBoost | `learning_rate` | `0.05` | 0.88 | 2 |
| XGBoost | `reg_lambda` | `5.0` | 0.92 | 2 |
| XGBoost | `subsample` | `0.8` | 1.00 | 1 |

### 5.10 Which model is best supported, and for what

The mechanical rule selects Full valid / SVM (RBF) / Isotonic because it has the highest discrimination. Taken on its own that is a misleading recommendation, and the analyses above say why:

- It separates the classes **completely**, so its calibration slope is not identified at all. A screening tool is used by acting on a probability, and this model's probability scale cannot be checked.
- Its feature importances are unstable (Kendall's W = 0.471; only 1 feature reaches the top 5 in 80% of folds), because 25 correlated laboratory variables can substitute for one another between folds.
- It requires venepuncture and a full laboratory panel, which is exactly the constraint the study set out to relax.

**Judged against the question this study actually asks, the better-supported configuration is the low-cost one** (SVM (RBF), Isotonic): ROC-AUC 0.993 against 1.000, a difference far inside the confidence intervals of either; an identified calibration slope of 1.26; markedly more stable importances (Kendall's W = 0.694, with `sg`, `al`, `dm`, `htn` in the top 5 of at least 80% of folds); and 11 variables obtainable from history, examination and a urine reagent strip.

This preference is a statement about which result is better *evidenced*, not a recommendation to use anything. Neither model is validated for any clinical purpose, and section 5.5 shows that both are evaluated on a sample whose case mix flatters them.

### 5.11 Apparent versus nested-CV performance (optimism)

Fitting the identical selection procedure (inner grid search on ROC-AUC) on the complete dataset and evaluating on that same data gives the *apparent* performance; the difference from the pooled nested out-of-fold estimate is the optimism that internal validation is already correcting for:

| Configuration | Model | Apparent ROC-AUC | Nested ROC-AUC | Optimism | Apparent Brier | Nested Brier |
|---|---|---:|---:|---:|---:|---:|
| Low-cost | SVM (RBF) | 0.9964 | 0.9909 | +0.0055 | 0.0173 | 0.0210 |
| Low-cost | Random forest | 0.9993 | 0.9921 | +0.0072 | 0.0113 | 0.0301 |
| Full valid | SVM (RBF) | 1.0000 | 1.0000 | +0.0000 | 0.0041 | 0.0069 |
| Full valid | Random forest | 1.0000 | 0.9996 | +0.0004 | 0.0013 | 0.0168 |
| Laboratory | SVM (RBF) | 0.9959 | 0.9951 | +0.0008 | 0.0399 | 0.0276 |
| Laboratory | Random forest | 1.0000 | 0.9942 | +0.0058 | 0.0050 | 0.0328 |

The largest ROC-AUC optimism across these cells is +0.0072. That optimism is this small for the same reason the valid models sit near the ceiling (section 5.5): the case mix leaves little room for resubstitution to exaggerate. On a harder problem the same procedure would show a much larger gap, so the small values here should be read as further evidence about the sample, not as evidence that validation discipline was unnecessary.

One asymmetry is worth recording: resubstitution is guaranteed to flatter *rank order* (the AUC optimism above is non-negative in every cell), but not the *probability scale*. In 1 cell(s) the apparent Brier score is actually worse than the nested one (Laboratory x SVM (RBF) (0.0399 vs 0.0276)), because the averaged out-of-fold probabilities are better placed on the probability scale than a single resubstitution fit's. Discrimination and calibration do not inflate together, which is one more reason the two must be reported separately.

### 5.12 A readable rule at the same cost (EBM versus SVM)

The model the prespecified rule selects is an RBF SVM, which is accurate and opaque. At identical cost - the same 11 history, examination and dipstick variables - the explainable boosting machine is a pure additive model (`interactions=0`) whose per-variable shape functions can be read and challenged:

| Model | Calibration | ROC-AUC | Brier | Calibration slope | Sensitivity | Readable shape functions |
|---|---|---:|---:|---:|---:|---|
| EBM (GAM) | Uncalibrated | 0.988 | 0.0270 | 0.813 | 0.961 | yes |
| EBM (GAM) | Isotonic | 0.987 | 0.0276 | 1.009 | 0.961 | yes |
| SVM (RBF) | Uncalibrated | 0.991 | 0.0210 | 2.227 | 0.969 | no |
| SVM (RBF) | Isotonic | 0.993 | 0.0192 | 1.257 | 0.969 | no |

The EBM gives up 0.0066 ROC-AUC against the SVM - far inside the bootstrap intervals of either - and buys with it a calibration slope of 1.009 against 1.257, i.e. a probability scale that needs essentially no correction. Where the study must choose between the best-discriminating model and the best-explainable one at equal cost, the evidence does not force the trade-off it is usually assumed to: on this sample the readable model is also the better-calibrated one.

Figure R13 plots the fitted shape functions, averaged over the 25 outer folds. Their directions are clinically coherent and, for `appet`, independently confirm a coding polarity that the released file never documented (see section 3.6): low urine specific gravity raises risk (impaired concentrating ability), albuminuria raises it steeply, and diabetes, hypertension, oedema and poor appetite all push the same way.

**One term must not be read on its own.** `bp (Diastolic)` runs opposite to its own marginal association (shape Spearman -0.984 against signed univariate ROC-AUC 0.553). That is a suppression effect rather than an error: the binary diastolic flag sits beside `bp limit` and `htn`, which carry the hypertension signal, so its residual contribution changes sign. It is recorded in `table_28_ebm_shape_notes.txt` and is a concrete illustration of the interpretation limit stated in section 5.7: an additive model is readable term by term only where its terms are not proxies for one another.

### 5.13 Label robustness (the four internally inconsistent records)

Four patients are labelled `notckd` while carrying an advanced CKD stage and an eGFR below 60 (CSV lines 12, 18, 52, 123). Because there is no external source of truth to arbitrate, they are kept in the primary analysis and interrogated here instead: the headline cells were re-run with those records **excluded**, and again with their labels **flipped** to `ckd`.

| Configuration | Model | Calibration | ROC-AUC (primary) | Excluded (delta) | Flipped (delta) |
|---|---|---|---:|---:|---:|
| Low-cost | SVM (RBF) | Uncalibrated | 0.991 | 0.992 (+0.0010) | 0.967 (-0.0241) |
| Low-cost | SVM (RBF) | Isotonic | 0.993 | 0.994 (+0.0008) | 0.971 (-0.0220) |
| Low-cost | Random forest | Uncalibrated | 0.992 | 0.990 (-0.0025) | 0.974 (-0.0184) |
| Full valid | SVM (RBF) | Uncalibrated | 1.000 | 1.000 (+0.0000) | 0.995 (-0.0051) |
| Full valid | SVM (RBF) | Isotonic | 1.000 | 1.000 (+0.0000) | 0.995 (-0.0053) |
| Laboratory | SVM (RBF) | Uncalibrated | 0.995 | 0.995 (+0.0002) | 0.994 (-0.0016) |
| Laboratory | SVM (RBF) | Isotonic | 0.994 | 0.995 (+0.0013) | 0.992 (-0.0022) |
| Laboratory | Random forest | Uncalibrated | 0.994 | 0.995 (+0.0012) | 0.993 (-0.0017) |

**Excluding the four records changes nothing.** The largest absolute ROC-AUC change across the cells is 0.0025, an order of magnitude inside the bootstrap intervals, and no cell's sensitivity moves by more than 0.0078. On the deletion reading, limitation 8 is answered: these records are not driving any conclusion.

**Flipping them does not, and the asymmetry is informative.** Treating the stage and eGFR columns as correct costs the low-cost configuration between 0.0184 and 0.0241 ROC-AUC, while the laboratory configuration loses at most 0.0022. The low-cost loss is comparable to the width of the bootstrap intervals themselves, so on this reading label quality is *not* a negligible source of uncertainty for the cheap model.

The mechanism is worth stating plainly, because it is the study's own primary claim placed under stress. These four patients are precisely the ones whose history, examination and dipstick findings look unremarkable while their laboratory values indicate advanced kidney disease. If their `notckd` labels are the errors, then they are exactly the patients a low-cost instrument would miss - and the laboratory panel would not. Four records cannot settle that, but they point the same way as the case-mix analysis in section 5.5: the low-cost result is strongest exactly where the disease is already advanced enough to show up without a laboratory.

### 5.14 Pseudo-external validation: the benchmark's two releases share their patients (mechanism 3)

The two mechanisms above concern a single dataset. The third concerns how this dataset is used in the literature. Several published studies validate models trained on the 2015 UCI CKD release against this file - distributed separately, with a different stated collection site, sample size and year - and present the result as external validation. That inference requires the two releases to contain different patients.

They do not. Every candidate dataset registered for this study was put through a record-level provenance check before any external claim was permitted (`reports/external/provenance_report.md`). Each of this file's patients is represented as a vector of intervals over the shared variables; a record in a candidate dataset is *compatible* with a patient when the outcome matches and every shared value falls inside that patient's interval. The observed maximum bipartite matching is then calibrated against column permutations that preserve every marginal distribution while destroying cross-variable structure.

| Quantity | Value |
|---|---|
| Patients in this file matched into the 2015 release | 200 / 200 |
| Maximum bipartite match fraction | 1.000 |
| Permutation null (mean +/- SD) | 0.003 +/- 0.003 |
| Permutation null (maximum observed) | 0.015 |
| Verdict | **SAME-SOURCE** |

Matching on intervals could in principle be coincidence, so the pairing was checked against evidence it had no access to. 187 of the 200 patients are compatible with exactly **one** record in the 2015 release. Across those pairs, 10 categorical variables that took no part in the matching agree in 1763 comparisons with **0 contradictions**. A coincidental alignment does not reproduce ten unused variables.

The conclusion is that this file is a discretised subset re-release of the 2015 dataset. Its documented provenance - a different country, hospital and year - cannot be reconciled with record-level identity; we report the measurement and take no view on how the discrepancy arose.

The consequence for the published record is direct. Of the 13 studies surveyed in section 1, **3** validate across these two releases as though they were independent cohorts, and one of those merges them into a single training set before reporting accuracy. On the evidence above, those procedures evaluate models on their own training population. This is not a criticism of the authors: nothing in either dataset's documentation indicates the overlap, and the present study registered the 2015 release as its own primary external-validation candidate before the check refused it.

Two implications follow for this report. First, the study has no external validation and cannot acquire one from these sources; every estimate here is internal, and the gate that produced this finding also blocks the 2015 release from every external-validation table (`tests/test_external.py::TestProvenanceGate`). Second, because the overlap is with a *continuous-valued* release of the same patients, the information destroyed by pre-discretisation can be recovered for this cohort and measured directly - a comparison that is possible precisely because the cohorts are not independent.

### 5.15 What the published binning destroyed (and what the file supplied in place of missing data)

Section 5.14 established that these patients appear, with their measurements intact, in a continuous-valued release. That makes a normally unanswerable question answerable: what did the interval encoding cost? Because the comparison is between two representations of **the same 187 patients**, there is no population shift, no case-mix difference and no sampling variation to confound it. Each variable below is scored on the patients whose value the source actually recorded, so the comparison isolates the encoding.

| Variable | n | ROC-AUC binned | ROC-AUC continuous | Lost to binning |
|---|---:|---:|---:|---:|
| `sc` | 177 | 0.680 | 0.946 | +0.2662 |
| `pot` | 150 | 0.518 | 0.588 | +0.0700 |
| `sod` | 150 | 0.806 | 0.820 | +0.0144 |
| `wbcc` | 141 | 0.609 | 0.620 | +0.0112 |
| `bu` | 176 | 0.786 | 0.794 | +0.0077 |
| `pcv` | 156 | 0.970 | 0.975 | +0.0042 |
| `rbcc` | 129 | 0.928 | 0.929 | +0.0013 |
| `sg` | 165 | 0.913 | 0.913 | +0.0000 |
| `su` | 164 | 0.639 | 0.639 | +0.0000 |
| `al` | 164 | 0.881 | 0.881 | +0.0000 |
| `hemo` | 165 | 0.985 | 0.985 | -0.0002 |
| `age` | 184 | 0.652 | 0.651 | -0.0006 |
| `bgr` | 171 | 0.727 | 0.720 | -0.0071 |

**The loss is concentrated in one variable, and it is the diagnostically decisive one.** For every variable except `sc` the encoding costs at most 0.0700 ROC-AUC - the bins are fine enough to preserve the signal. Serum creatinine loses 0.2662, falling from 0.946 to 0.680. The published bins show why:

| Published bin | n | Continuous span (mg/dL) | Fraction CKD |
|---|---:|---|---:|
| `< 3.65` | 137 | 0.5 - 3.6 | 0.52 |
| `3.65 - 6.8` | 21 | 3.9 - 6.7 | 1.00 |
| `6.8 - 9.95` | 9 | 7.1 - 9.7 | 1.00 |
| `9.95 - 13.1` | 4 | 10.2 - 12.8 | 1.00 |
| `13.1 - 16.25` | 4 | 13.4 - 15.2 | 1.00 |
| `16.25 - 19.4` | 1 | 18.1 - 18.1 | 1.00 |
| `>= 28.85` | 1 | 32.0 - 32.0 | 1.00 |

The first bin absorbs 137 of 177 patients and spans 0.5 to 3.6 mg/dL - from unambiguously normal (0.6-1.2) through severe renal impairment. Every other bin is 1.00 CKD or higher. In this release serum creatinine is therefore not a graded measurement but a coarse flag that fires only once creatinine is already extreme, and inside the bin holding most of the cohort it carries almost no information (0.52 CKD).

This has a consequence for how the leakage boundary should be read. This study excluded `grf` (eGFR) as a post-diagnosis derivative but retained serum creatinine, on the grounds that it is a routinely measured analyte rather than a diagnostic label. On the released data that judgement is comfortable, because binning has flattened the variable. On the underlying measurements it is much less so: creatinine alone reaches ROC-AUC 0.946, which is close to the quantity that defines the outcome. A study using the continuous release would need to defend that inclusion far more carefully than one using this file - and would not know it from this file alone.

**A second property of the release surfaces at the same time.** The analysed file contains exactly one missing cell. Its source contains a great many: across the recoverable variables, 339 cells that the source leaves blank carry a value here. For **all 13 of 13** variables, every one of those cells was filled with a *single constant* - and in each case that constant is the clinically normal range:

| Variable | Cells filled | Value supplied | Distinct values used | Odds of being unobserved, CKD vs non-CKD |
|---|---:|---|---:|---:|
| `age` | 3 | `51 - 59` | 1 | infinite |
| `rbcc` | 58 | `4.46 - 5.05` | 1 | 31.2 |
| `wbcc` | 46 | `7360 - 9740` | 1 | 20.5 |
| `pcv` | 31 | `37.4 - 41.3` | 1 | 11.2 |
| `pot` | 37 | `< 7.31` | 1 | 4.9 |
| `sod` | 37 | `133 - 138` | 1 | 4.9 |
| `al` | 23 | `< 0` | 1 | 4.6 |
| `su` | 23 | `< 0` | 1 | 4.6 |
| `sg` | 22 | `1.019 - 1.021` | 1 | 4.3 |
| `hemo` | 22 | `11.3 - 12.6` | 1 | 3.0 |
| `bgr` | 16 | `112 - 154` | 1 | 1.9 |
| `sc` | 10 | `< 3.65` | 1 | 0.9 |
| `bu` | 11 | `48.1 - 86.2` | 1 | 0.7 |

The missingness is not random. A patient with CKD is 5 to 31 times more likely to have these measurements absent from the source, which is what one would expect when tests are ordered selectively. Filling every such cell with the normal value therefore assigns normal-looking laboratory results to precisely the patients most likely to be ill, and does so invisibly: nothing in the released file distinguishes a measured normal result from a supplied one.

**Its direction is worth stating plainly, because it runs against this report's own thesis.** Every other mechanism examined here inflates measured performance. This one deflates it: substituting normal values for the sickest patients makes the classes harder to separate, not easier. Comparing each variable's separability across all pinned patients against the observed subset suggests an attenuation of up to 0.0652 ROC-AUC (largest for `al`), though that comparison is across different patient subsets and should be read as indicative rather than exact. The practical implication is not that the benchmark is harder than it looks - it is that a third of some columns are not measurements at all, which no analysis of the released file can discover.

**Two undocumented variables are resolved as a by-product.** Section 3.6 flags `bp (Diastolic)` and `bp limit` as variables whose coding could not be established from the file. Against the recovered measurements they read directly:

| Encoded column | Level | n | Diastolic range (mmHg) | Median |
|---|---|---:|---|---:|
| `bp (Diastolic)` | 0 | 82 | 50 - 70 | 70 |
| `bp (Diastolic)` | 1 | 101 | 60 - 120 | 80 |
| `bp limit` | 0 | 85 | 50 - 110 | 70 |
| `bp limit` | 1 | 54 | 80 - 80 | 80 |
| `bp limit` | 2 | 44 | 90 - 120 | 90 |

`bp (Diastolic)` is an indicator for diastolic pressure at or above 80 mmHg, and `bp limit` bands the same measurement into at-or-below 70 / exactly 80 / at-or-above 90. Both readings hold for every pinned patient but a handful (one record coded 1 at 60 mmHg; two coded 0 at 100 and 110 mmHg), which are further instances of the data-entry noise documented in section 3.4. The uncertainty flags on these variables can now be removed - but only because a second release of the same patients existed.

## 6. Discussion

This study set out to measure one failure mode and found three. Each pushes measured performance towards 1.0, none of them is predictive ability, and they compound: controlling any one still leaves the others free to produce a near-perfect number.

The first is **target leakage**. Including an exact copy of the outcome, a post-diagnosis stage label and the diagnostic eGFR value takes every configuration to ROC-AUC 1.000 - closing 100% of the headroom regardless of how good the honest baseline was. These columns sit in the released file with no marking to indicate that they are post-diagnosis, and any pipeline that selects features by association with the outcome will pick them up. The lesson is not that previous analyses were careless in some unusual way; it is that the trap is built into the dataset.

The second is **case mix**, and on this dataset it turns out to matter more. Even with every prohibited column removed, the full valid configuration reaches ROC-AUC 1.000 - it separates all 200 patients out of fold. We verified this is not a residual leak: the pipelines are guarded, the runner restricts columns to those each configuration declares, and an automated test re-checks the perfect cells against the prohibited list. It is instead a genuine property of the sample, and section 5.5 shows what produces it.

On the primary question, the low-cost variable set - history, examination and a urine reagent strip - achieves ROC-AUC 0.992 under leakage-controlled nested validation. That is well above chance and within the confidence interval of the laboratory-based configuration. Given that the low-cost set requires no venepuncture, no analyser and no microscope, the finding is **encouraging as a feasibility signal**. It is not evidence that such a model would work as a screening instrument.

The case-mix analysis is what forces that distinction, and it turned out to matter more than the leakage audit. Leakage is the failure mode this study set out to measure; case mix is the failure mode the data revealed. Because 84% of the CKD patients here have moderate-to-severe disease, the models are largely being asked to distinguish established kidney failure from health - a task on which haemoglobin alone scores 0.968. A screening instrument faces a different and much harder problem, and the early-stage subgroup shows the models performing materially worse on it. An aggregate ROC-AUC near 1.0 on this sample should therefore be read as a statement about the sample, not about the method.

The third mechanism is **pseudo-external validation**, and it is the one that cannot be fixed by analysing this dataset more carefully. Section 5.14 shows that the two releases treated in the literature as independent cohorts share their patients record-for-record. A study that trains on one and validates on the other has measured resubstitution performance with extra steps, and will see exactly what it expects to see: near-perfect transfer. Unlike leakage, this failure is invisible from inside either file - both look like well-formed, separately documented datasets - which is why a provenance check belongs before the modelling, not after a surprising result.

Together these reframe what a 'high-performing' published result on this benchmark means. Three distinct mechanisms - target leakage, case-mix spectrum, and non-independent validation cohorts - each push measured performance towards the ceiling, and none has anything to do with clinical usefulness. A study reporting 99% accuracy on this data has probably encountered at least one, and cannot distinguish them without the leakage audit, subgroup analysis and provenance check reported here. The three are also ordered by how easy they are to miss: leakage is visible in the column list, case mix requires looking at who is in the sample, and cohort overlap requires comparing two datasets nobody had reason to suspect were one.

Note that the best valid model overall used the Full valid configuration rather than the low-cost one. The low-cost result should therefore be read as 'a substantial fraction of the achievable signal at a fraction of the cost', not as 'no loss from dropping laboratory tests'.

Calibration deserves particular emphasis because it is what makes a predicted probability usable for a referral decision. A model with good discrimination but a calibration slope well below 1 will systematically overstate risk in the patients it is most confident about - exactly the patients whose management would change. The comparison here also illustrates a general point about small samples: the more flexible calibration method is not the better one when there are only a few hundred observations to fit it with.

The stability analysis is, in a sense, the most honest part of the study. With Kendall's W of 0.471 across 25 outer folds, the feature ranking is only moderately reproducible. Any narrative that named the 'top predictors of CKD' from a single fit of this dataset would be reporting an artefact of one partition.

## 7. Limitations

These are not boilerplate. Each one materially constrains what the results above can support.

1. **Sample size.** 200 patients, 72 in the minority class. Bootstrap intervals for ROC-AUC span roughly 0.02, so differences between configurations smaller than that are not resolvable. Events per candidate predictor (2.88) is far below accepted minimums for prediction-model development [5].
2. **No external validation, and none obtainable from the obvious source.** Every estimate is internal to these 200 patients. Internal cross-validation systematically overstates the performance a model would show in a new population, and no correction applied here changes that. The natural remedy - validating against the larger 2015 UCI release, which shares this file's variable vocabulary - is unavailable: section 5.14 shows the two share their patients. The provenance gate blocks that dataset from every external-validation table, so this limitation is enforced rather than merely stated.
3. **Hospital-based, single-centre sampling.** These patients are not a random sample of any population. The 64.0% CKD prevalence is a property of who was recruited, not of any community. Selection bias is likely and its direction is unknown.
4. **Case-mix (spectrum) bias - the most consequential limitation.** 107 of 128 CKD patients (84%) have stage s3-s5 disease, so the dataset largely contrasts established kidney failure with comparatively healthy controls. Discrimination measured on such a sample is systematically optimistic for screening use. Section 5.5 shows sensitivity for the low-cost model falling from 0.969 overall to 0.905 among early-stage CKD patients, and the clinical-only model from 0.891 to 0.762. **No headline figure in this report should be read as an estimate of screening performance.**
5. **Pre-discretised predictors.** The published file contains only binned intervals. Real continuous values are unrecoverable, open-ended tail bins are compressed to their finite edge, and the effective measurement precision of every variable is unknown. A model built on continuous measurements might perform differently in either direction.
6. **No data dictionary.** The dataset ships without variable definitions [4]. 10 variables have interpretations that could not be settled from the file and are flagged as uncertain rather than resolved by assumption. The tiering of `ane` in particular changes what 'low cost' means, and was decided conservatively.
7. **Minimal demographic information.** Age is present, in bands. There is no sex, no ethnicity, no socioeconomic indicator, no comorbidity detail beyond three binary flags. Subgroup performance therefore cannot be assessed at all, and undetected differential performance across groups is entirely possible.
8. **Label quality.** 4 patients are labelled `notckd` while carrying an advanced CKD stage and an eGFR below 60. Either the label or the staging is wrong for those records, and there is no external source of truth to arbitrate. Section 5.13 quantifies both readings: **deleting** them changes nothing (largest |delta ROC-AUC| 0.0025), but **trusting the staging instead of the label** costs the low-cost configuration up to 0.0241 ROC-AUC against at most 0.0022 for the laboratory configuration. Label noise is therefore not a negligible source of uncertainty for the cheap model specifically, and this limitation is only half answered.
9. **Cross-sectional data, no chronicity.** CKD is defined by abnormalities persisting beyond three months [2]. A single record cannot establish chronicity, so the outcome label itself rests on information not present in the file.
10. **Prevalence-dependent metrics.** PPV and NPV reported here hold only at this sample's 64.0% prevalence and do not transfer to a community screening setting.
11. **Linearity assumption for the linear models.** Ordinal bin representatives are entered on their original scale, so logistic regression and the SVM assume an approximately monotone, roughly linear relationship with the log-odds across bins. The tree ensembles and the EBM do not make this assumption.
12. **Compute-driven modelling choices.** Search spaces, forest size and EBM capacity were constrained to keep the repeated nested design feasible. These are documented trade-offs; a larger search might find better configurations, though at this sample size it would also select on noise more aggressively.
13. **Exploratory analyses.** The decision-curve net benefit, the univariate association screen and the correlation structure are labelled exploratory and were not used for any modelling decision. They should not be read as confirmatory findings.

## 8. Ethical considerations

**This is not a diagnostic tool and must not be used as one.** No model in this repository is validated for any clinical purpose. Presenting it to a patient or clinician as a screening instrument would be unsafe.

**Consequences of error are asymmetric and are borne by patients.** A false negative in CKD screening means a patient with progressive kidney disease is reassured and not followed up; at the prespecified threshold the low-cost model produced 4 such errors among 128 CKD patients in cross-validation. A false positive means an unnecessary referral, which in a resource-constrained setting consumes capacity that another patient needed. Neither error is neutral, and the balance between them is a clinical and policy decision, not a modelling one.

**Data governance.** The analysis uses de-identified records already released for research under CC BY 4.0 [4]. No attempt is made to re-identify any individual. The raw file is preserved unmodified and all derived artefacts are written separately. Findings about individual records (for instance the four internally inconsistent labels) are reported by CSV line number, which refers to a position in a public de-identified file and not to any identifiable person.

**Equity.** A low-cost screening model is attractive precisely because it could extend detection to populations without laboratory access. That same argument makes undetected differential performance especially harmful: a model that works less well for a subgroup would concentrate its errors on the people the approach is meant to help. This dataset contains too little demographic information to check for that, which is a reason for caution rather than an excuse for silence.

**Transparency about the leaky model.** A deliberately invalid configuration is included and reports near-ceiling performance. It is labelled INVALID in every table and figure that shows it. It exists to demonstrate a hazard, and quoting its numbers outside that context would be a misrepresentation.

## 9. Reproducibility statement

- **Seed.** A single master seed (`20240517`) deterministically derives every fold seed and model seed. `tests/test_reproducibility.py` asserts that identical seeds produce identical predictions and that different seeds produce different partitions.
- **Input integrity.** The raw file is checksummed on every load (SHA-256 `f24075f420b0f271bfddf3844a40061dbe9e2cb1c2336daa64463122344ea84a`) and is never written to.
- **Single source of results.** All 108,000 out-of-fold predictions are written to `data/processed/cv_predictions.csv.gz`. Every table, figure and number in this report is derived from that one file, so results cannot drift apart from the cross-validation that produced them.
- **Run manifest.** `data/processed/cv_predictions_manifest.json` records the design, the seed, the models actually run, the models unavailable in this environment, and the runtime (71.89 min).
- **This document is generated.** Every quantity above is injected from `reports/tables/` by `scripts/06_write_report.py`; nothing is transcribed by hand.
- **Environment.** Dependencies are pinned in `requirements.txt`.

Exact commands:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest tests/ -q
python scripts/run_all.py
```

## 10. Conclusion

Machine-learning results on this benchmark cluster at the discrimination ceiling. This re-analysis identifies three mechanisms that put them there, none of which is predictive ability.

**Target leakage** is the visible one: the file ships an exact copy of the outcome, a post-diagnosis stage label and the diagnostic eGFR value, and including them takes every configuration to ROC-AUC 1.000. **Case mix** is the larger one: with every prohibited column removed and leakage prevented by a guard that raises rather than drops, the full valid configuration still separates all 200 patients out of fold, because 107 of 128 cases (84%) carry stage s3-s5 disease and haemoglobin alone reaches univariate ROC-AUC 0.968. **Non-independent validation cohorts** are the least visible: the two releases the literature treats as separate share their patients record-for-record, so published cross-dataset validations between them measure resubstitution.

The prespecified selection rule, which ranks on discrimination alone, picks **SVM (RBF) on the Full valid configuration with Isotonic probabilities** (ROC-AUC 1.000, sensitivity 1.00, NPV 1.00, Brier 0.0036). We report that pick because the rule was fixed in advance, but we do not endorse it: as section 5.10 sets out, that model separates the classes completely, so its calibration slope is not identified, its feature importances are unstable (Kendall's W = 0.471), and it needs every valid variable in the dataset, the full laboratory panel included.

**The best-supported model for the question this study asks is the low-cost one: SVM (RBF) with Isotonic probabilities on 11 history, examination and urine-dipstick variables** - ROC-AUC 0.993 (SVM (RBF), Isotonic) against 1.000 for the selected Full valid model (SVM (RBF), Isotonic), a difference well inside either confidence interval; sensitivity 0.97, NPV 0.95, Brier 0.0192, and an identified calibration slope of 1.26. It is the only candidate whose probability scale can be checked and whose important variables are reproducible across folds.

A post hoc case-mix analysis then showed that even the leakage-controlled figures overstate screening ability: 84% of the CKD patients have stage s3-s5 disease, haemoglobin alone discriminates at ROC-AUC 0.968, and among early-stage CKD patients the low-cost model's sensitivity falls from 0.969 to 0.905 while its ROC-AUC stays above 0.98. Two independent mechanisms - target leakage and case-mix spectrum - push measured performance towards the ceiling on this dataset, and neither reflects clinical usefulness.

What this study supports is a methodological claim in two parts. First, about this benchmark: the numbers reported on it are largely explained by three properties of the data, the largest of which - case mix - is rarely examined, and the least visible of which - cohort overlap - invalidates the cross-dataset validations that would otherwise be its strongest evidence. Second, about method: every one of the three was found by a check that can be run before any model is fitted. Declare the prohibited columns and enforce them with something that raises rather than drops; look at the stage distribution and the univariate separability before believing a discrimination figure; and verify that a validation cohort is actually a different set of patients. Each check is cheap; each one here changed a conclusion.

What the study does not support is any claim about clinical utility, causality, deployment readiness or generalisability. The estimates are internal to 200 patients, with pre-discretised predictors, a case mix dominated by advanced disease, no demographic detail beyond age bands, four records whose labels contradict their own staging, and confidence intervals wide enough to encompass materially different conclusions. The cost-stratified comparison reported here says something about how much discrimination survives dropping laboratory variables **on this sample**; it says nothing about screening, because a sample in which 84% of cases have moderate-to-severe disease is not a screening population. **Prospective validation in a genuine screening series would be required** before any clinical interpretation is warranted, and no dataset examined here can substitute for it.

## References

1. GBD Chronic Kidney Disease Collaboration. Global, regional, and national burden of chronic kidney disease, 1990-2017: a systematic analysis for the Global Burden of Disease Study 2017. *Lancet*. 2020;395(10225):709-733. doi:[10.1016/S0140-6736(20)30045-3](https://doi.org/10.1016/S0140-6736(20)30045-3)
2. Kidney Disease: Improving Global Outcomes (KDIGO) CKD Work Group. KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease. *Kidney Int*. 2024;105(4S):S117-S314. doi:[10.1016/j.kint.2023.10.018](https://doi.org/10.1016/j.kint.2023.10.018)
3. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*. 2024;385:e078378. doi:[10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378)
4. Islam MA, Akter S. Risk Factor Prediction of Chronic Kidney Disease [dataset]. UCI Machine Learning Repository; 2020. Licensed CC BY 4.0. doi:[10.24432/C5WP64](https://doi.org/10.24432/C5WP64)
5. Riley RD, Snell KIE, Ensor J, et al. Minimum sample size for developing a multivariable prediction model: PART II - binary and time-to-event outcomes. *Stat Med*. 2019;38(7):1276-1296. doi:[10.1002/sim.7992](https://doi.org/10.1002/sim.7992)
6. Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*. 2006;7:91. doi:[10.1186/1471-2105-7-91](https://doi.org/10.1186/1471-2105-7-91)
7. Van Calster B, Nieboer D, Vergouwe Y, De Cock B, Pencina MJ, Steyerberg EW. A calibration hierarchy for risk models was defined: from utopia to empirical data. *J Clin Epidemiol*. 2016;74:167-176. doi:[10.1016/j.jclinepi.2015.12.005](https://doi.org/10.1016/j.jclinepi.2015.12.005)
8. Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. *Med Decis Making*. 2006;26(6):565-574. doi:[10.1177/0272989X06295361](https://doi.org/10.1177/0272989X06295361)
9. Lundberg SM, Lee S-I. A unified approach to interpreting model predictions. In: *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*. 2017:4765-4774. arXiv:[1705.07874](https://arxiv.org/abs/1705.07874)
10. Nori H, Jenkins S, Koch P, Caruana R. InterpretML: a unified framework for machine learning interpretability. 2019. arXiv:[1909.09223](https://arxiv.org/abs/1909.09223)
11. Kabir MA, Munira S, Azad DT, Ikram SM, Sarker MHR, Hanifi SMA. Community-based early-stage chronic kidney disease screening using explainable machine learning for low-resource settings. *Int J Med Inform*. 2026. arXiv:[2601.01119](https://arxiv.org/abs/2601.01119)
12. Islam MA, Majumder MZH, Hussein MA. Chronic kidney disease prediction based on machine learning algorithms. *J Pathol Inform*. 2023;14:100189. doi:[10.1016/j.jpi.2023.100189](https://doi.org/10.1016/j.jpi.2023.100189)
13. Hossain MF, Diya ST, Khan R. ACD-ML: Advanced CKD detection using machine learning: a tri-phase ensemble and multi-layered stacking and blending approach. *Comput Methods Programs Biomed Update*. 2025;7:100173. doi:[10.1016/j.cmpbup.2024.100173](https://doi.org/10.1016/j.cmpbup.2024.100173)

*No reference above was generated without verification; each has a DOI or a stable arXiv identifier. The prior-work survey table additionally distinguishes, per study, which facts were verified from full text or abstract and which are attributed to the comparison table of [11] pending hand verification.*

## Appendix A. Generated artefacts

**Figures** (`reports/figures/`, 300 dpi PNG and PDF)

- `fig_e1_cohort_composition.png`
- `fig_e2_univariate_association.png`
- `fig_e3_leakage_structure.png`
- `fig_e4_predictor_correlation.png`
- `fig_e5_configuration_composition.png`
- `fig_r10_best_valid_importance_stability.png`
- `fig_r10_low_cost_importance_stability.png`
- `fig_r10_shap_capable_importance_stability.png`
- `fig_r11_shap_capable_shap_vs_permutation.png`
- `fig_r12_spectrum_effect.png`
- `fig_r13_ebm_shapes.png`
- `fig_r14_binning_cost.png`
- `fig_r1_auc_heatmap.png`
- `fig_r2_leakage_audit.png`
- `fig_r3_screening_metrics.png`
- `fig_r4_roc_pr_curves.png`
- `fig_r5_calibration_curves.png`
- `fig_r6_calibration_methods.png`
- `fig_r7_threshold_tradeoff.png`
- `fig_r8_confusion_matrices.png`
- `fig_r9_uncertainty.png`

**Tables** (`reports/tables/`)

- `table_00_dataset_summary.csv`
- `table_01_data_dictionary.csv`
- `table_02_feature_configurations.csv`
- `table_03_univariate_association.csv`
- `table_04_collinear_pairs.csv`
- `table_05_level_distribution_by_outcome.csv`
- `table_06_variable_summary.csv`
- `table_07_hyperparameter_selection.csv`
- `table_08_metrics_per_repeat.csv`
- `table_09_metrics_per_outer_fold.csv`
- `table_10_metrics_summary_across_repeats.csv`
- `table_11_bootstrap_confidence_intervals.csv`
- `table_12_headline_results.csv`
- `table_13_leakage_audit.csv`
- `table_14_calibration_results.csv`
- `table_15_calibration_curve_points.csv`
- `table_16_selected_models.csv`
- `table_17_threshold_tradeoff_low_cost.csv`
- `table_18_confusion_matrices.csv`
- `table_19_best_valid_importance_per_fold.csv`
- `table_19_low_cost_importance_per_fold.csv`
- `table_19_shap_capable_importance_per_fold.csv`
- `table_20_importance_stability.csv`
- `table_21_spectrum_analysis.csv`
- `table_22_univariate_separability.csv`
- `table_23_leakage_ceiling_analysis.csv`
- `table_24_provenance.csv`
- `table_24_provenance_agreement.csv`
- `table_24_provenance_evidence.csv`
- `table_26_prior_work.csv`
- `table_26_prior_work_summary.csv`
- `table_27_label_robustness.csv`
- `table_28_ebm_shape_functions.csv`
- `table_28_ebm_shape_summary.csv`
- `table_28_ebm_vs_svm.csv`
- `table_29_optimism.csv`
- `table_30_recovery_coverage.csv`
- `table_31_binning_cost.csv`
- `table_32_imputation_audit.csv`
- `table_33_sc_bin_structure.csv`
- `table_34_blood_pressure_recovery.csv`
