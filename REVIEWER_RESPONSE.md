# Reviewer objections → the artifact that addresses each

Every row points at a committed artifact. Where an objection is **not**
fully answered, the row says so; nothing here claims a fix that does not
exist on disk.

Status vocabulary:

- **Resolved** — the objection no longer applies to the current work.
- **Answered** — addressed with evidence, though the underlying constraint
  remains and is stated in the report.
- **Open** — not addressed yet; the plan says when.
- **Blocked** — cannot be addressed without data we do not have.

---

## The framing objections

| # | Objection | Status | Artifact |
|---|---|---|---|
| 1 | *"The title promises 'Low-Cost CKD Screening in Bangladesh' but the data cannot deliver a screening claim."* | **Resolved** | Retitled to the mechanisms thesis; screening framing removed from every claim; study type restated as a methodological re-analysis. `scripts/06_write_report.py`, commit `07713b8`. Enforced by `tests/test_report.py::TestReframedFraming`. |
| 2 | *"The primary result is disavowed by the paper's own limitations section."* | **Resolved** | The low-cost comparison is now a *supporting* result about cost-stratified feature sets (§5.3, §5.12), not the headline. The headline is the three mechanisms, which the limitations support rather than undercut. |
| 3 | *"'Bangladesh' provenance."* | **Resolved, and became a finding** | §3.1 reports the documented provenance and states that the data contradict it; readers are told to treat site, country and year as unverified. Evidence: `reports/external/provenance_report.md`. |

## The evidentiary objections

| # | Objection | Status | Artifact |
|---|---|---|---|
| 4 | *"You critique a literature you never cite — the '~100% accuracy' claim has no citation."* | **Answered** | `data/literature/prior_work.csv` (13 studies, per-row verification provenance) → `table_26_prior_work{,_summary}.csv` → injected Introduction. Refs [11][12][13] added. **Caveat:** 3 studies independently verified; 10 carry `todo_hand_check=yes` because paywalls (Wiley, ScienceDirect, BSPC) blocked automated verification. The report states the split rather than implying full verification. |
| 5 | *"The leakage contribution self-destructs: the honest baseline is already 1.000, so measured inflation is 0.000."* | **Answered, and reframed** | Conceded explicitly in §5.2 and the abstract. The response is not a rescue metric but a second mechanism: §5.5 shows *why* the valid baseline is at the ceiling (case mix), which is now the larger finding. The headroom table remains, labelled as what it is. |
| 6 | *"n = 200, 2.88 events per predictor — no journal accepts prediction-model development at that size."* | **Answered by scope change** | The work no longer claims model development. It claims measurement of three benchmark properties, for which n = 200 is the object of study rather than a limitation to apologise for. Sample-size limitation retained verbatim (limitation 1). |
| 7 | *"No new data; re-analysis of a public dataset with dozens of existing papers."* | **Answered** | The re-analysis produced findings not in those papers: the cohort-overlap result (§5.14), the case-mix decomposition (§5.5), the label-noise asymmetry (§5.13), and the identified-vs-unidentified calibration slope distinction (§5.6). |

## The internal-quality objections

| # | Objection | Status | Artifact |
|---|---|---|---|
| 8 | *"The abstract literally reads 'slope n/a' — a template artifact."* | **Resolved** | Fixed; abstract now states non-identification explicitly. `tests/test_report.py::TestNoTemplateArtifacts` fails the build if any unresolved token or bare `nan` reappears. |
| 9 | *"Low-cost AUC appears as 0.992 and 0.993 without reconciliation."* | **Resolved** | §5.3 gains a "Best model" column and an uncalibrated-provenance caption; every quoted figure names its (config × model × calibration) cell. `tests/test_report.py::TestHonestAttribution`. |
| 10 | *"The conclusion says '0.993 against 1.000 for the full laboratory panel' — 1.000 is Full valid, not Laboratory."* | **Resolved** | Each quoted AUC now names its own configuration; a test asserts the selected-model AUC is never captioned as laboratory. |
| 11 | *"'Explainable' in the title, but the selected model is an RBF SVM."* | **Answered** | §5.12 adds the EBM at identical cost: 0.0066 AUC lower, calibration slope 1.009 vs 1.257, with readable shape functions (fig R13). One suppression term is documented as not readable alone (`table_28_ebm_shape_notes.txt`) rather than glossed. |
| 12 | *"Label noise (4 contradictory records) is acknowledged but not quantified."* | **Answered** | §5.13: exclusion changes nothing (|Δ AUC| ≤ 0.0025); flipping costs the low-cost set up to 0.0241 vs 0.0022 for laboratory. Limitation 8 rewritten from "answered" to **half-answered** — the honest reading. |
| 13 | *"How much is internal validation already correcting for?"* | **Answered** | §5.11: apparent-minus-nested optimism per headline cell, largest +0.0072, read as further evidence of the ceiling rather than as licence to skip validation. |
| 18 | *"Your confidence intervals hold the fitted models fixed, so they understate uncertainty — you say so yourself."* | **Answered** | §5.18: 200 resamples per cell, each re-running the entire nested procedure. Honest intervals are 1.7–2.7× wider. Outer folds within a resample are grouped by patient so no model is tested on a patient it trained on. |
| 19 | *"The isotonic-beats-Platt result rests on 25 folds and is contrary to expectation."* | **Answered** | §5.19: repeated at 20 repeats / 100 outer test sets. Isotonic wins 105 of 120 paired comparisons (88%); median slope 1.15 vs 2.38. The ordering survives. |
| 20 | *"The interval-midpoint encoding is an arbitrary modelling choice."* | **Answered** | §5.17: four encodings including one-hot (86 columns on 200 patients); largest deviation 0.0101 ROC-AUC, inside the intervals. With §5.16 (continuous values) the representation question is closed. |

## What remains open or blocked

| # | Objection | Status | Position |
|---|---|---|---|
| 14 | *"No external validation."* | **Blocked** | The obvious source (UCI-2015) turned out to share our patients, so it is disqualified — and the provenance gate now enforces that mechanically. Genuine external validation requires the icddr,b/TangailBD cohort (restricted; request to `habibur.rahman@icddrb.org`). Until then the spectrum prediction stands as a registered, testable claim, and the report says so. |
| 15 | *"Pre-discretised predictors — the model isn't reproducible on real measurements."* | **Answered** | §5.15–5.16: values recovered for 187 patients from the overlapping continuous release. Binning costs ≤0.07 univariate AUC everywhere except serum creatinine (0.2662), and changes **no** multivariable conclusion (largest difference 0.0122). Also surfaced that the release supplied constant "normal" values for 339 missing cells whose absence is outcome-associated. |
| 16 | *"Single dataset."* | **Partly open** | Three additional datasets are registered, checksummed and classified INDEPENDENT (`th_uae`, `birdem`, `mimic_iv_demo`), with their design caveats recorded (task shift; longitudinal/grouped-CV requirement; ETL required). Phases 3a–3c will use them; none is a substitute for #14. |
| 17 | *"Prior-work coding not fully verified."* | **Open** | 10 of 13 rows flagged `todo_hand_check`. Requires institutional access to paywalled full texts. |

---

*Generated as part of Phase 2. Commit history: `git log --oneline` from tag
`reference-run-2026-08-22`.*
