# Project documentation

**Three mechanisms inflate reported performance on a widely used public CKD
benchmark: a leakage-controlled, case-mix-aware re-analysis.**

*Last revised 2026-08-22. This document describes the work as it now stands.
It replaces an earlier version written when the study was framed as a
low-cost screening feasibility study; §0.1 explains why that framing was
abandoned, because the reason is itself one of the findings.*

---

## Contents

- [Part 0 — What this work is about](#part-0--what-this-work-is-about)
- [Part 1 — What was built](#part-1--what-was-built)
- [Part 2 — What the work establishes](#part-2--what-the-work-establishes)
- [Part 3 — Limitations](#part-3--limitations)
- [Part 4 — Future work](#part-4--future-work)
- [Part 5 — Practical guide](#part-5--practical-guide)

---

## Part 0 — What this work is about

### 0.1 The question, and how it changed

The project began as an ordinary applied-ML study: *can a small, explainable
model predict CKD status from inexpensive information — history, examination,
a urine dipstick — with useful sensitivity and calibrated risk?* It was built
carefully, with programmatically enforced leakage control and repeated nested
cross-validation.

Careful construction produced an uncomfortable result. **Every valid model sat
at the discrimination ceiling.** The full valid configuration separated all
200 patients out of fold, with no prohibited column anywhere near it. A
tripwire test — "perfect performance means a bug" — fired, and the
investigation into why it fired became the study.

The answer was that the ceiling is a property of the *data*, not of the
models, and that three distinct mechanisms put it there. None of them is
predictive ability. That reframing is the work:

> **Why do machine-learning studies on this benchmark routinely report
> accuracy at or near 100%, and what remains once the mechanisms responsible
> are controlled or measured?**

The original screening question survives only as a supporting result about
cost-stratified feature sets. It cannot be answered on this data, and the
report says so in the sections that would otherwise be its headline.

### 0.2 The three mechanisms

Ordered as the report orders them — **by how easy each is to miss**, which is
the argument that makes this a general contribution rather than a complaint
about one dataset.

| # | Mechanism | What it is | How it is found | Effect |
|---|---|---|---|---|
| 1 | **Target leakage** | The file ships an exact copy of the outcome (`affected`), a post-diagnosis stage label (`stage`), and the diagnostic eGFR value (`grf`) | Visible in the column list — if you look | Takes **any** configuration to ROC-AUC 1.000 |
| 2 | **Case-mix (spectrum) composition** | 84% of CKD cases carry stage 3–5 disease; haemoglobin alone reaches univariate ROC-AUC 0.968 | Requires examining *who is in the sample*, not the columns | The larger effect; puts valid models at the ceiling unaided |
| 3 | **Pseudo-external validation** | The 2015 and 2020 UCI releases, treated in the literature as independent cohorts, share their patients record-for-record | Requires comparing two datasets nobody suspected were one | Makes cross-dataset validation measure resubstitution |

Mechanism 3 was not anticipated. This study registered the 2015 release as
its *own* primary external-validation candidate; the provenance gate built to
check it refused it.

### 0.3 The dataset, and two facts that shape everything

`ckd-dataset-v2.csv` — 200 patients, 29 columns, preceded by two metadata rows
that are not patients. Outcome `class` ∈ {ckd, notckd}: 128 CKD (64%), 72 not.
Distributed by UCI as *Risk Factor Prediction of Chronic Kidney Disease*
(Islam & Akter, doi:10.24432/C5WP64, CC BY 4.0).

**Fact 1 — every continuous variable was discretised before release.** `sg`
appears as `1.019 - 1.021`, `bgr` as `< 112`. This is a property of the
published data, not a modelling choice.

**Fact 2 — the documented provenance is contradicted by the data.** UCI
documents collection at Enam Medical College, Savar, Bangladesh. A
record-level check finds all 200 of these patients inside the 2015 UCI release
(documented as Apollo Hospitals, India). Treat the stated site, country and
year as unverified; no claim in this work rests on them.

### 0.4 What makes the provenance finding solid

It rests on three independent kinds of evidence, and the third is the one that
makes coincidence implausible:

1. **Containment matching.** Each of our patients is a vector of intervals
   over 13 shared variables. A source record is *compatible* if the outcome
   matches and every value falls inside the corresponding interval. Maximum
   bipartite matching: **1.000** of our patients matched, against a
   permutation null of **0.003 ± 0.003** (max 0.015 over 50 shuffles).
2. **Uniqueness.** **187 of 200** patients are compatible with exactly *one*
   source record; the maximum anywhere is two.
3. **Held-out confirmation.** Across those 187 pairs, **10 categorical
   variables that took no part in the matching agree with 0 contradictions**
   (~1,500 opportunities): `rbc`/`pc` 1↔abnormal, `pcc`/`ba` 1↔present,
   `htn`/`dm`/`cad`/`pe`/`ane` 1↔yes, `appet` 1↔poor.

A coincidental alignment does not reproduce ten unused variables. The verdict
rule was fixed in code *before* any external result existed
(`ckd.data.provenance.classify`), and the registry's negative control — the
byte-identical file — is caught as SAME-SOURCE by the same rule.

**Deliberate restraint:** the report states the measurement and the
inconsistency with the documented provenance, and takes no view on how the
discrepancy arose.

---

## Part 1 — What was built

### 1.1 Scale

| | |
|---|---|
| Python (src + scripts + tests) | ~14,500 lines across 25 `src/` modules |
| Pipeline stages | 14 numbered scripts + `run_all.py` |
| Automated tests | **380**, all passing, none skipped |
| Generated tables | 50 CSVs, all report numbers injected from them |
| Generated figures | 21 (300 dpi PNG + PDF) |
| Commits | 23 from tag `reference-run-2026-08-22` |

### 1.2 The pipeline

`scripts/run_all.py` runs stages in order and stops at the first failure. All
reusable logic lives in `src/ckd/`; scripts orchestrate, they do not implement.
Stage numbers reflect the order stages were *written*; `run_all.py` is the
authority on execution order.

| # | Script | Produces | Runtime |
|---|---|---|---|
| 0 | `00_provenance.py` | Provenance gate; tables 24/24e/24a, `provenance_report.md` | ~30 s |
| 1 | `01_prepare_data.py` | Cleaned + encoded data, bin map, data dictionary, quality report | 2 s |
| 2 | `02_eda.py` | Figures E1–E5, tables 00–06 (EXPLORATORY) | 7 s |
| 3 | `03_nested_cv.py` | 108,000 out-of-fold predictions + manifest; `--label-variant` for robustness runs | 72 min |
| 4 | `04_evaluate.py` | Tables 08–18, 21–23; figures R1–R9, R12 | 3 min |
| 5 | `05_importance_stability.py` | Tables 19–20; figures R10 (×3), R11 | 30 min |
| 7 | `07_literature.py` | Prior-work survey; tables 26, 26-summary | 1 s |
| 8 | `08_sensitivity_labels.py` | Label robustness; table 27 | 5 s |
| 9 | `09_optimism.py` | Apparent vs nested-CV; table 29 | 2 min |
| 10 | `10_ebm_shapes.py` | Readable rule; tables 28/-summary/-notes, fig R13 | 9 min |
| 11 | `11_continuous_recovery.py` | Value recovery; tables 30–34, fig R14 | 1 min |
| 12 | `12_binned_vs_continuous.py` | Model-level representation test; table 35 | 25 min |
| 13 | `13_tripod_checklist.py` | TRIPOD+AI appendix; table 36 | 1 s |
| 14 | `14_encoding_robustness.py` | Four encodings; table 37 | ~50 min |
| 15 | `15_procedure_bootstrap.py` | Honest intervals; table 38 | ~25 min |
| 16 | `16_calibration_experiment.py` | Calibration at 20 repeats; table 39 | 1 min |
| 6 | `06_write_report.py` | `reports/research_report.md` (907 lines) | 2 s |

Stages 11 and 12 are **gated**: they refuse to run unless the provenance
verdict is SAME-SOURCE, because recovering one cohort's values from another is
only legitimate when they are the same patients.

### 1.3 The design decisions that carry the study

**(a) The bin encoding is stateless.** Each interval label maps to a number by
a pure function of that one cell's text — no cross-row statistics, no outcome
information. That is why it may legitimately be applied before splitting.
Tested by re-encoding arbitrary row subsets and by reversing row order.

**(b) Leakage is prevented by exception, in three independent layers.**
Declaration (`configs.py` names every prohibited column); a runtime
`LeakageGuard` that **raises at fit *and* transform** — and raises rather than
drops, so a caller's bug cannot be silently absorbed; and
`assert_pipeline_is_clean()` over every pipeline before any work. The key
adversarial test hands a valid pipeline the *whole* dataframe and requires it
to refuse.

**(c) Every reported number descends from one file.** 108,000 predictions →
tables → report text, injected by stage 6. Prose cannot drift from results.

**(d) Thresholds never see test data.** The 0.50 threshold is prespecified in
config; the screening threshold comes from a separate cross-validated
prediction on the outer *training* fold.

**(e) Calibration statistics are computed per repeat.** Averaging a patient's
probability across repeats shrinks it toward the centre and biases the slope
upward. Found and corrected during the build.

**(f) Complete separation is detected, not optimised through.** Under
separation the recalibration slope has no maximum-likelihood solution; an
optimiser still returns a number (values in the hundreds were observed).
`separates_perfectly()` detects this and the slope is reported **undefined**.

**(g) Group-aware folds for resampled data.** A bootstrap sample contains the
same patient many times; outer folds within a resample are formed over
*patients*, so no model is evaluated on a patient it trained on. Off by
default, leaving the reference path byte-identical.

### 1.4 Machine-checked guarantees

Claims about the *analysis*, each enforced by code that fails loudly.

| # | Guarantee | Enforcement |
|---|---|---|
| G1 | No prohibited column entered any valid pipeline | 3 layers + audit over every pipeline run |
| G2 | No preprocessing fitted on complete data | Imputer/scaler statistics differ between folds and match the training fold |
| G3 | Encoding cannot transfer information between folds | Statelessness tested per encoding |
| G4 | The raw file was never modified | SHA-256 on every load; CI re-verifies |
| G5 | The analysis is deterministic | One master seed derives all fold and model seeds |
| G6 | Every metric is correct | Checked against scikit-learn plus edge cases |
| G7 | No neural network, no synthetic oversampling | Asserted in `test_pipelines.py` |
| G8 | The prediction artefact is internally consistent | Each patient predicted once per repeat per cell |
| G9 | No number in the report was transcribed by hand | Stage 6 injects everything |
| G10 | No non-INDEPENDENT dataset can support an external claim | `TestProvenanceGate` scans external tables |
| G11 | Report numbers match their source tables | `test_report.py` re-derives them |

**The honest limit of G5.** Identical seeds reproduce identical predictions
*within* an environment. Across an environment rebuild from the same pinned
versions, agreement is to **~1.1e-16**, not bit-for-bit, because summation
order depends on the installed BLAS. Verified pre-existing against unmodified
sources. The regression gate asserts 1e-12 rather than claiming exactness the
project cannot deliver.

---

## Part 2 — What the work establishes

### 2.1 The three mechanisms, quantified

**Mechanism 1 — leakage.** The invalid configuration reaches ROC-AUC 1.000
with sensitivity and specificity both 1.000. Measured against baselines with
headroom, leakage closes **100%** of remaining error. Measured against the
full valid configuration the gain is ~0 — because that baseline is *already*
at the ceiling, which is mechanism 2.

**Mechanism 2 — case mix.** With every prohibited column removed and leakage
prevented by a guard that raises, the full valid configuration still separates
all 200 patients out of fold. 107 of 128 cases (84%) are stage s3–s5;
haemoglobin alone reaches univariate ROC-AUC 0.968. Restricted to early CKD
(s1–s2, n=21), low-cost sensitivity falls **0.969 → 0.905** and clinical-only
**0.891 → 0.762**.

**Mechanism 3 — pseudo-external validation.** See §0.4. Of 13 surveyed
studies, **3** validate across the two releases as independent cohorts and one
merges them into a single training set.

### 2.2 Findings from the recovered measurements

Because the overlap is with a *continuous-valued* release of the same
patients, the information destroyed by pre-discretisation is measurable
within-cohort — no population shift, no case-mix difference.

- **Binning is mostly benign but catastrophic in one place.** Every
  recoverable variable loses ≤0.07 univariate ROC-AUC — except **serum
  creatinine, which loses 0.2662** (0.946 → 0.680). The bin `< 3.65` absorbs
  137 of 177 patients and spans 0.5–3.6 mg/dL: normal through severe
  impairment, at 52% CKD, while every other bin is 100% CKD.
- **Consequence for this study's own leakage boundary.** We excluded `grf` but
  retained `sc`. That call is comfortable *only because binning flattened the
  variable*. On the real measurements `sc` alone reaches 0.946 — close to the
  quantity that defines the outcome. A study on the continuous release would
  have to defend that inclusion far harder, and could not know it from this
  file.
- **The release imputed missing data with constants.** 339 cells the source
  leaves blank carry values here, and for **all 13** variables every such cell
  received a *single* constant — in each case the clinical normal range.
  Missingness is strongly outcome-associated (OR 3–31), so normal values were
  supplied to the patients most likely to be ill, invisibly.
  **This mechanism deflates rather than inflates** — reported because it is
  true, not because it supports the thesis.
- **Nothing changes at the model level.** Same patients, both
  representations, identical seeds: largest difference **0.0122 ROC-AUC**, an
  order of magnitude inside the intervals. That null is the interesting part —
  you can destroy the resolution of the most diagnostic analyte in the dataset
  and the models do not notice, because haemoglobin and PCV already carry the
  information.
- **Two `uncertain` variables resolved.** `bp (Diastolic)` is a diastolic ≥80
  mmHg indicator; `bp limit` bands ≤70 / exactly 80 / ≥90. Three further
  data-entry inconsistencies found.

### 2.3 Robustness

| Question | Answer | Where |
|---|---|---|
| Do the 4 contradictory labels drive anything? | Deleting them: no (\|ΔAUC\| ≤ 0.0025). **Flipping** them costs low-cost up to **0.0241** vs 0.0022 for laboratory | §5.13 |
| Is the midpoint encoding load-bearing? | No. Four encodings incl. one-hot (86 columns); largest deviation **0.0101** | §5.17 |
| How much is internal validation correcting for? | Largest optimism **+0.0072** — itself evidence of the ceiling | §5.11 |
| Are the reported intervals honest? | No — the procedure-level intervals are **1.7–2.7× wider** | §5.18 |
| Does isotonic really beat Platt at n=200? | Yes: **105/120** paired wins (88%) at 100 outer test sets | §5.19 |

The label-robustness asymmetry is the study's own primary claim under stress:
the 4 disputed patients are exactly those whose cheap findings look
unremarkable while their labs indicate advanced disease. If their `notckd`
labels are the errors, they are the patients a low-cost instrument would miss.

### 2.4 Supporting result: cost-stratified feature sets

Under leakage-controlled validation, 11 history/examination/dipstick variables
reach ROC-AUC 0.992 (95% CI 0.981–0.999) against 0.995 for the 14-variable
laboratory panel. At identical cost the **EBM** gives up only 0.0066 AUC to
the SVM while having a calibration slope of **1.009 vs 1.257** — on this
sample the readable model is also the better-calibrated one, so the usual
accuracy-versus-interpretability trade-off does not bind. One term
(`bp (Diastolic)`) is a suppression effect and is documented as not readable
alone.

**This is a statement about cost tiers on this case mix. It is not a screening
result, and the report never presents it as one.**

### 2.5 Findings reported as negative, unstable or inconclusive

Recorded because a study that only reports what worked is not reporting.

- Feature-importance rankings are only **weakly** concordant (Kendall's W
  0.471 for the selected model). Any "top predictors of CKD" narrative from a
  single fit would be an artefact of one partition.
- The selected model's calibration slope is **not identified** — it separates
  the classes completely.
- The low-cost vs laboratory difference is **not statistically resolved**, and
  the honest intervals make that worse, not better.
- Constant imputation **deflates** separability — against the thesis.
- My own prediction that ungrouped bootstrap folds would inflate replicates
  **failed** when tested (grouped 0.9923 vs ungrouped 0.9876). The grouping
  requirement stands on definitional grounds instead.

---

## Part 3 — Limitations

Each materially constrains what the work can support. Several are now
*quantified* rather than asserted.

### 3.1 Limitations of the data

1. **Sample size.** 200 patients, 72 minority events, **2.88 events per
   candidate predictor** — far below accepted minimums for prediction-model
   development. The study is underpowered by construction and is presented as
   a measurement exercise, not model development.
2. **No external validation, and none obtainable from the obvious source.**
   Every estimate is internal. The natural remedy — the larger 2015 release —
   is disqualified because it shares these patients. The gate **enforces**
   this rather than merely stating it.
3. **Hospital-based, single-centre sampling.** Not a random sample of any
   population. The 64% prevalence is a property of recruitment. Selection bias
   is likely; its direction is unknown.
4. **Case-mix bias — the most consequential.** 84% of cases are stage s3–s5,
   so the data largely contrast established kidney failure with comparatively
   healthy controls. **No figure in this work estimates screening
   performance.**
5. **Pre-discretised predictors.** Now bounded: ≤0.07 univariate AUC lost
   everywhere except serum creatinine (0.2662), and no multivariable
   conclusion changed. But effective measurement precision remains unknown for
   variables the recovery could not reach.
6. **Constant imputation by the data provider.** 339 cells are supplied
   values, not observations, with no missingness indicator. Discoverable only
   because a second release existed.
7. **No data dictionary.** 10 variables were flagged `uncertain` rather than
   given invented interpretations; 2 are now resolved. The tiering of `ane` in
   particular changes what "low cost" means and was decided conservatively —
   a choice that can only *understate* low-cost performance.
8. **Minimal demographics.** Age in bands only. No sex, ethnicity, or
   socioeconomic indicator. **Subgroup and fairness analysis is impossible**,
   and undetected differential performance is entirely possible.
9. **Label quality.** 4 records are internally contradictory. Deleting them
   changes nothing; trusting the staging instead costs the low-cost
   configuration up to 0.0241 AUC. Limitation **half-answered**, not answered.
10. **Cross-sectional data, no chronicity.** CKD is defined by abnormality
    persisting beyond three months. A single record cannot establish that, so
    the outcome label itself rests on information not in the file.
11. **Prevalence-dependent metrics.** PPV and NPV hold only at this sample's
    64% prevalence and do not transfer.

### 3.2 Limitations of the analysis

12. **Linearity assumption** for logistic regression and the SVM: ordinal bin
    representatives enter on their original scale. Tree ensembles and the EBM
    do not make this assumption.
13. **Compute-driven modelling choices.** Search spaces, forest size and EBM
    capacity were constrained to keep the repeated nested design feasible.
14. **Exploratory analyses** (decision-curve net benefit, univariate screens,
    correlation structure) are labelled EXPLORATORY and were used for no
    modelling decision.
15. **Reproducibility across environments** is to ~1e-16, not bit-for-bit
    (§1.4).

### 3.3 Limitations of the literature survey

16. **Only 3 of 13 studies independently verified.** The remaining 10 carry
    `todo_hand_check=yes` because Wiley/ScienceDirect/BSPC paywalls blocked
    automated verification; their coding is *attributed* to a published
    comparison table, not confirmed. The report states this split rather than
    implying full verification.
17. **The survey is not systematic.** It is a targeted sample seeded from one
    paper's reference list, sufficient to establish that near-perfect results
    and pseudo-external validation occur, but not to estimate how often.

### 3.4 What this work does not claim

- No claim about **clinical utility, causality, deployment readiness, or
  generalisability**.
- No claim that any model here should be used on any patient.
- No claim about **how** the provenance discrepancy arose — only that the
  records coincide.
- No claim that the surveyed authors were careless. Nothing in either
  dataset's documentation reveals the overlap; this study fell into the same
  trap and was caught by its own gate.

---

## Part 4 — Future work

Ordered by how much each would change what the work can claim.

### Tier 1 — changes the scientific standing

**F1. Obtain a genuinely independent cohort. `BLOCKED — needs a person.`**
The only identified candidate is the icddr,b / TangailBD community cohort
(Kabir et al.): n=284, **68% of cases early-stage** (s1=25, s2=51, s3=36),
CKD-EPI eGFR + ACR with chronicity confirmed at 3 months — a genuine screening
series, and the exact case mix this study says is missing. **Restricted;
requests to `habibur.rahman@icddrb.org`.** Until then, the spectrum prediction
stands as a *registered, testable claim* rather than a result.

**F2. The case-mix re-test.** With F1 in hand, the single most valuable
experiment: does low-cost sensitivity collapse on early-stage patients as §5.5
predicts? This is the paper's central claim facing data that could refute it.

**F3. A prospective consecutive screening series.** The only route to a
clinical claim. Design essentials: consecutive or community sampling (not
case-control); reference standard applied to *everyone*; eGFR **and** ACR
repeated ≥3 months apart so chronicity is actually established; deliberate
early-stage enrichment (target ≥50% of cases at s1–s2); sex recorded at
minimum; ~1,100 participants at a realistic community prevalence for 10
events per predictor; protocol pre-registered before collection.

**F4. Complete the literature survey properly.** Convert the targeted sample
into a systematic one with defined search terms and inclusion criteria, and
hand-verify the 10 unverified rows. This turns "these failures occur" into
"these failures occur at rate X" — a materially stronger claim, and the
cheapest Tier-1 item. Needs institutional access.

### Tier 2 — strengthens the current analysis

**F5. Extend the provenance check to other benchmarks.** The machinery
(`ckd.data.provenance`) is dataset-agnostic: interval-containment matching
with a permutation null and held-out categorical confirmation. Running it
across commonly paired UCI medical datasets would establish whether
pseudo-external validation is peculiar to CKD or endemic. **This is the most
promising generalisation of the work.**

**F6. Frozen-transfer and retraining on the registered INDEPENDENT sources.**
`th_uae` (491 patients, prospective incident CKD — a task shift, honest usage
is sensitivity-only on the 56 confirmed cases), `birdem` (400 diabetics,
longitudinal, grouped CV mandatory, thin schema overlap), `mimic_iv_demo`
(ETL required, appendix only). Each is registered, checksummed and classified;
none substitutes for F1.

**F7. Beta calibration** as a third method, requiring a custom
`CalibratedClassifierCV` wrapper. Deferred as low-value: the question worth
answering (does the isotonic ordering survive resampling?) is answered.

**F8. Sensitivity of the provenance verdict** to the containment variable set
and to the compatibility tolerance. The verdict is not close to its threshold
(1.000 vs a null of 0.003), so this is due diligence rather than doubt.

### Tier 3 — dissemination

**F9. Fill in `CITATION.cff`** — author name, affiliation, ORCID, repository
URL. Deliberately left unfilled rather than guessed.

**F10. Archive with a DOI.** `scripts/make_archive.py` produces a citable zip
with a per-file SHA-256 manifest and refuses a dirty tree. Needs a Zenodo (or
equivalent) deposit.

**F11. Target a venue.** The work now fits a methods/cautionary venue —
*BMC Medical Informatics and Decision Making*, *Diagnostic and Prognostic
Research*, *JMIR Medical Informatics*, or an ML4H/MLHC findings track. The
code would also stand alone at JOSS.

**F12. Notify the dataset custodians.** The provenance discrepancy and the
undocumented constant imputation are both material to anyone using either
release. A note to the UCI curators would be the responsible step — a
judgement call for the author, not something this analysis should decide.

### Explicitly not recommended

Stated to prevent well-meant regressions:

- **No deep learning.** 72 minority events cannot estimate one, and a test
  enforces this.
- **No SMOTE or synthetic oversampling.** Imbalance is mild (64/36);
  synthesising patients at n=200 manufactures structure.
- **No chasing the last 0.007 of AUC.** Intervals span ~0.02 — and the honest
  ones span 0.03–0.04.
- **Do not "clean away" the 4 contradictory records.** They are evidence about
  label quality; handle as sensitivity analysis.
- **Never quote the leaky model outside the audit.**
- **Do not enable EBM interaction terms** at this sample size.
- **Do not replace the prespecified selection rule after seeing results.** If
  it is wrong, say so and explain why — as §5.10 does.
- **Do not add a referral-capacity operating point.** It would reintroduce the
  deployment framing this work removed, and its methodological content is
  already design decision (d).

---

## Part 5 — Practical guide

### 5.1 Reproducing the analysis

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash
python -m pip install -r requirements.txt   # or requirements.lock.txt
python -m pytest tests/ -q                  # 380 tests, ~3 min
python scripts/run_all.py                   # full pipeline
```

`run_all.py --skip-cv` reuses committed predictions and skips the expensive
stages. The complete run is dominated by stage 3 (~72 min) plus stages 12, 14
and 15.

### 5.2 Where to look for what

| You want | Read |
|---|---|
| The argument | `reports/research_report.md` |
| The provenance evidence | `reports/external/provenance_report.md`, tables 24/24e/24a |
| Objection → artifact map | `REVIEWER_RESPONSE.md` |
| Decision history and corrections | `WORKLOG.md` |
| What may never enter a model | `src/ckd/features/configs.py` |
| Why a guarantee holds | the test that enforces it, `tests/` |

### 5.3 If you extend this work

1. **Add a table, then inject it.** Never hand-write a number into prose.
2. **Add tests with the code.** Analysis code without tests is not done here.
3. **Run the provenance gate before any external claim.** It exists because
   this project needed it and did not know that until it ran.
4. **Record failures.** `WORKLOG.md` contains predictions that turned out
   wrong; that is the point of it.
