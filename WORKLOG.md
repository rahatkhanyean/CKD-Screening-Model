# WORKLOG

Newest entries at the top. One entry per work session / phase milestone.

---

## 2026-08-22 — Phase 2 COMPLETE: the paper is now a three-mechanisms methods paper

Commit `07713b8`. Full suite **324 passed**.

**Title:** *"Three mechanisms inflate reported performance on a widely used
public CKD benchmark — a leakage-controlled, case-mix-aware re-analysis."*
Chosen without waiting on the user (they were away); trivially changeable,
since the title is a single line in `06_write_report.py`.

**What changed.** The thesis went from two mechanisms to three once Phase 0's
provenance result was folded in as a finding rather than a blocker:

1. Target leakage (visible in the column list)
2. Case mix (requires looking at who is in the sample) — still the largest
3. Pseudo-external validation (requires comparing two datasets nobody
   suspected were one) — **new §5.14**

The Discussion now orders them by how easy each is to miss, which is the
argument that makes the paper a contribution rather than a complaint.

**Framing removals.** Screening/Bangladesh framing is gone from every claim.
§3.1 now reports the documented provenance *and* states the contradiction,
telling readers to treat site/country/year as unverified. Limitation 2 is
rewritten to "no external validation **and none obtainable from the obvious
source**" — enforced by the gate, not merely asserted.

**Framing is now test-enforced.** `TestReframedFraming` (5 tests): title
carries the thesis; all three mechanisms have result sections; §3.1 states
the contradiction; no unqualified "screening performance" claim survives
(sentence-level check, after a first window-based version produced a false
positive on a correctly-qualified sentence); "Bangladesh" may appear only in
provenance context.

**Also delivered:** `REVIEWER_RESPONSE.md` — 17 objections mapped to
artifacts, with four rows honestly marked Open or Blocked rather than
claimed as fixed.

**Note on #4 (the literature table):** 3 of 13 studies independently
verified; 10 flagged `todo_hand_check` because Wiley/ScienceDirect/BSPC
paywalls blocked automated verification. The Introduction states the split
rather than implying full verification. This needs institutional access or
a hand pass.

**Next:** Phase 3 is gated only on the icddr,b request for 3e; 3b′ (the
dataset-parameterized refactor) and 3d (continuous-value recovery, now a
same-cohort comparison) can start immediately.

---

## 2026-08-22 — Phase 1 COMPLETE (1a–1e). Two new substantive findings.

Commits `2b93315` (1a), `56872c1` (1b), `45e8f10`+`b883aec` (1e),
`061f123` (1c), `3e8d23a` (1d). Full suite **319 passed** (was 227).

**1a — report inconsistencies fixed, then made unrepeatable.** The
abstract's `slope n/a` now states non-identification explicitly; §5.3
gains a "Best model" column and an uncalibrated-provenance caption so
0.992-vs-0.993 is self-explaining; the conclusion no longer calls the
Full-valid AUC "the full laboratory panel". `tests/test_report.py` (12)
re-derives every headline number from its source table and fails on
unresolved template tokens.

**1b — the uncited claim is now counted evidence.**
`data/literature/prior_work.csv`: 13 studies, per-row verification
provenance. 3 independently verified; 10 flagged `todo_hand_check`.
Introduction now reads: 2 of 3 verified headline metrics ≥99%, 9/13 coded
as using SC/eGFR, exactly **1** genuinely external validation — plus a new
paragraph tying the 3 pseudo-external cross-dataset studies to the
Phase 0 provenance finding. Paywalls (Wiley, ScienceDirect, BSPC) blocked
automated verification of several; those rows say so rather than guess.

**1e — optimism is small, and that is itself evidence.** Largest ROC-AUC
optimism across six headline cells is **+0.0072**. Reported as another
symptom of the case-mix ceiling, not as a licence to skip validation.
Found and fixed an aggregation mismatch (initial version pooled
predictions differently from the published tables) by reusing
`pooled_predictions`. Also recorded an asymmetry worth keeping:
resubstitution flatters rank order but **not** the probability scale —
laboratory×SVM has apparent Brier 0.0399 vs nested 0.0276.

**1c — NEW FINDING that qualifies the study's own primary claim.**
Excluding the 4 inconsistent records changes nothing (largest |ΔAUC|
0.0025). **Flipping** their labels costs the low-cost configuration up to
**0.0241 ROC-AUC** — comparable to the bootstrap CI width — while the
laboratory configuration loses at most 0.0022. Mechanism: these are
exactly the patients whose cheap findings look unremarkable while their
labs indicate advanced disease. If their `notckd` labels are the errors,
they are the patients a low-cost instrument would miss. Limitation 8
rewritten from "answered" to **half-answered**.

**1d — the readable rule, and a suppression term.** At identical cost the
low-cost EBM gives up only 0.0066 AUC to the SVM but has calibration slope
**1.009 vs 1.257** — the readable model is also the better-calibrated one,
so the usual accuracy-vs-interpretability trade-off does not bind here.
Shape functions are clinically coherent and independently confirm the
`appet` polarity the released file never documented (agreeing with the
Phase 0 provenance mapping). **One term must not be read alone:**
`bp (Diastolic)` runs opposite its marginal association (ρ −0.984 vs
signed AUC 0.553) — a suppression effect beside `bp limit`/`htn`,
documented in `table_28_ebm_shape_notes.txt` and discussed in §5.12.

**Engineering.** `03_nested_cv.py --label-variant` requires a non-default
`--out`, and `table_07` is now written only by the primary run, so no
variant or restricted run can overwrite reference artefacts. `run_all.py`
now runs 00 → 01 → 02 → 03 (+2 variants) → 04 → 05 → 07 → 08 → 09 → 10 →
06. Two Google-Drive/git incidents: a stale `packed-refs.lock` (cleared;
`fsck` clean, commit intact) — the predicted sync symptom, mitigated by
frequent commits as planned.

**Open for Phase 2:** restructure the report around the two-mechanisms
thesis (now arguably *three*: leakage, case mix, pseudo-external
validation) and retitle. The icddr,b data request remains the single most
valuable outstanding item.

---

## 2026-08-22 — Phase 0 COMPLETE. Provenance gate verdict: the benchmark's "two datasets" are one

**The finding.** `scripts/00_provenance.py` (new stage, gated by
`tests/test_external.py::TestProvenanceGate`) classified `uci2015` as
**SAME-SOURCE** with the internal file:

- Interval-containment matching (13 shared variables, outcome required to
  match, missing = compatible): **maximum bipartite match fraction 1.000**
  — every one of our 200 patients fits inside the 400 UCI-2015 records —
  against a 50-shuffle permutation null of **0.003 ± 0.003 (max 0.015)**.
- **187 of 200 patients pin to exactly one** UCI-2015 record (max 2
  partners anywhere).
- Held-out confirmation: across those 187 pairs, **10 categorical
  variables that played no part in the matching agree with 0
  contradictions** (~1,500 opportunities): rbc/pc 1↔abnormal,
  pcc/ba 1↔present, htn/dm/cad/pe/ane 1↔yes, appet 1↔poor.
- `uci2023_v2` (the registry's negative control): byte-identical,
  SAME-SOURCE, caught as designed.

**What this means.** The v2 file — documented by UCI as collected at Enam
Medical College, Bangladesh, 2020 — is at record level a discretised subset
re-release of the 2015 Apollo Hospitals (India) dataset. Consequences:

1. **uci2015 can never serve as external validation** for this study; the
   gate test now enforces that mechanically (any external table naming a
   non-INDEPENDENT dataset fails CI).
2. **Published cross-dataset validations between UCI-2015 and "UCI-2023"**
   (at least refs [49][50][51] in Kabir et al., and Kabir et al.'s own
   external validation) **evaluated on the same patients.** Pseudo-external
   validation joins target leakage and case-mix as a third inflation
   mechanism for the reframed paper — likely headline material.
3. **Phase 3d improves**: the unique pins allow deterministic recovery of
   continuous values for 187+ internal patients, so binned-vs-continuous
   becomes a same-cohort comparison with no population-shift confound.
4. **Genuine external validation now rests on icddr,b** (TangailBD) —
   the data request (habibur.rahman@icddrb.org) is the program's single
   most valuable outstanding item.
5. Two documented variable uncertainties are resolved empirically (`appet`
   1 = poor; urine flags 1 = abnormal/present), and a new data-quality
   defect surfaced: **v2 coded UCI-2015 missing values as 0** (e.g. 67
   missing `rbc` → 0), i.e. silent missing-as-normal imputation in the
   released file — this plausibly explains `ba`'s near-constancy and must
   be reported wherever those variables are interpreted (feeds Phase 1/2).

**Caveat recorded with the claim.** "SAME-SOURCE" is a statement about
record-level identity, not about who mislabelled what; the report presents
the evidence and the inconsistency with the documented provenance without
attributing intent.

**Artifacts.** `reports/external/provenance_report.md`;
`reports/tables/table_24_provenance{,_evidence,_agreement}.csv`;
verdict rule fixed in `src/ckd/data/provenance.py::classify` before any
external result existed.

**New code + tests.** `src/ckd/data/external.py` (registry, checksums,
loaders, per-source prohibited sets, dataset seeds),
`src/ckd/data/provenance.py` (containment, null calibration, forensics,
held-out agreement), `scripts/00_provenance.py`,
`config/external_datasets.yaml`, `tests/test_external.py` (35),
`tests/test_provenance.py` (14). **Full suite: 276 passed** (was 227).
openpyxl==3.1.5 added to the environment for the TH xlsx (to be added to
requirements.txt in the Phase-1 commit that touches it).

---

## 2026-08-22 — Phase 0.2: external-dataset acquisition and a decisive literature find

**The decisive find.** Kabir et al. 2026, *"Community-Based Early-Stage
Chronic Kidney Disease Screening using Explainable Machine Learning for
Low-Resource Settings"* (arXiv:2601.01119, Int J Med Inform / Elsevier;
PubMed 41946122) — read in full. It matters to this project four ways:

1. **It independently confirms our reframe thesis in print.** Their Related
   Work states the vast majority of ML-CKD studies use UCI-2015 and that
   including serum creatinine / eGFR as inputs "introduces a form of
   *information leakage* and *circular reasoning*." Strong citation for the
   reframed Introduction.
2. **Its Table 1 is a seed for our prior-work table (1b):** 19 studies
   [33]–[51] coded for dataset used, SC/eGFR use, and external validation,
   with full citations. We will re-verify each paper ourselves before
   entering yes/no values (guardrail 7) but the citation list is a running
   start.
3. **It confirms "UCI-2023" IS our dataset:** described as Enam Medical
   College, Savar; 200 patients; 128 CKD / 72 non-CKD; stages 1–5 annotated;
   citation = Islam & Akter, doi:10.24432/C5WP64 — identical to our file's
   provenance. A third party independently documents the identity our
   provenance gate must prove; `uci2023_v2` stays registered as a
   SAME-SOURCE negative control.
4. **It supplies a third public external candidate.** Their TH dataset =
   Al-Shamsi, Regmi & Govender, PLOS ONE 2018 (doi:10.1371/journal.pone.0199920),
   Tawam Hospital, Al-Ain, UAE: 491 adults at high cardiovascular risk, 56
   with confirmed CKD s3–5; the 435 "non-CKD" were NOT screened for
   early-stage CKD, so negatives are unreliable — Kabir used only the 56
   confirmed cases for external sensitivity. Late-stage-enriched: the mirror
   image of the icddr,b cohort, useful for the spectrum analysis.

**icddr,b / TangailBD (the 3e target).** Community cohort, Mirzapur DSS,
Tangail: 872 completed, final analytic n=284 (112 CKD: s1=25, s2=51, s3=36 —
**68% of cases early-stage**; 172 non-CKD; s4 n=5 and s5 n=1 excluded).
CKD-EPI eGFR + ACR, chronicity confirmed at 3 months — a genuine screening
series, exactly what our case-mix re-test needs. **Not publicly available**
(icddr,b ethics); data availability statement: requests to
`habibur.rahman@icddrb.org`. → USER ACTION required for Phase 3e; recorded
as `UNAVAILABLE(restricted; request via email)` in the registry.

**Acquired today (all checksummed):**

| Dataset | File | SHA-256 |
|---|---|---|
| uci2015 (UCI id 336, doi:10.24432/C5G020, CC BY 4.0) | `data/external/uci2015/raw/data.csv` (401 lines = 400 records, CONTINUOUS values) | `0eeea8d1…8109e694` |
| uci2015 original archive | `…/chronic_kidney_disease.zip` (contains a .rar; data.csv fetched from UCI's csv endpoint instead) | `3f0d0e5b…a26a585` |
| th_uae (PLOS ONE S1 Dataset, CC BY 4.0) | `data/external/th_uae/raw/pone.0199920.s002.xlsx` | `cdc0e009…b76c58` |
| mimic_iv_demo (PhysioNet, open) | `data/external/mimic_iv_demo/raw/mimic-iv-clinical-database-demo-2.2.zip` | `97301a03…afeb8ec` |

First-glance uci2015 provenance signal: 25 columns whose names (`sg, al,
su, bgr, bu, sc, sod, pot, hemo, pcv, wbcc, rbcc, htn, dm, cad, appet, pe,
ane, class`) overlap ours almost exactly; ours adds `affected`, `grf`,
`stage`, `bp limit` and discretised `age`/`bp`. India (Apollo) vs Bangladesh
(Enam) origin claims, 400 vs 200 records, 250/150 vs 128/72. Overlap check
in 0.3 will decide INDEPENDENT vs derived formally.

**BIRDEM — located and acquired.** Anan et al., *Data in Brief* 2025
(doi:10.1016/j.dib.2025.112414); data on Mendeley
(doi:10.17632/hjkzgbxgv5.2, CC BY 4.0).
`data/external/birdem/raw/DiabeticCKD_dataset.csv`, SHA-256
`9cef0eb4…92793618` — **verified against Mendeley's own published hash**.
Two structural findings that gate its use:
1. **Longitudinal, not cross-sectional: 3,999 data rows for 400 patients**
   (~10 yearly rows each, `Patient ID` repeats). Any CV on it MUST group by
   patient or fold contamination is guaranteed. Registry gets a mandatory
   `grouped_split_by: Patient ID` flag.
2. All 400 are diabetic (so `dm` is constant), and the variables are
   demographic/lifestyle only — no labs, no dipstick. Harmonized overlap
   with our schema is thin (age, htn, heart disease≈cad, appet-like
   lifestyle vars). Confirms its role: sensitivity analysis only.

**TH structure inspected** (491×25): `EventCKD35` outcome with
`TimeToEventMonths` — it is a **prospective incident-CKD cohort**, not a
cross-sectional diagnostic sample. Baseline `CreatnineBaseline`/
`eGFRBaseline` are temporally legitimate predictors there (measured before
the outcome), unlike our `grf`. But validating our *prevalent-CKD
detection* models against *incident-CKD-by-36-months* labels is a task
shift, and the 435 non-events are not verified non-CKD cross-sectionally.
Registry notes must state this; Kabir et al. used only the 56 confirmed
cases (sensitivity-only external check), which is the honest usage.

---

## 2026-08-22 — Phase 0.1 GATE PASSED: environment restored, 227/227 green

- `.venv` rebuilt from `requirements.txt`; every pinned version installed
  exactly (numpy 2.5.2, pandas 3.0.5, scipy 1.18.0, scikit-learn 1.9.0,
  matplotlib 3.11.1, PyYAML 6.0.3, xgboost 3.4.1, catboost 1.2.10,
  shap 0.52.0, interpret 0.7.8, pytest 9.1.1) on CPython 3.14.5.
- **Full suite: 227 passed in 166.38s.** The 213-vs-227 discrepancy is
  resolved: 14 tests were parametrized over the optional model deps
  (xgboost/interpret/shap/catboost) missing from the ambient interpreter.
  The documented count was correct for the reference environment.
- Gate satisfied: source changes may now begin (Phase 0.2 registry).

---

## 2026-08-22 — Phase 0.1: baseline integrity (in progress)

**Done**

- PLAN.md written and approved (decisions: start Phase 0; keep `.git` in
  place under Google Drive sync, corruption risk accepted and mitigated by
  frequent commits; network downloads of UCI-2015 + MIMIC-IV demo approved).
- `git init`; baseline commit `093f2f2` of the exact on-disk reference-run
  state; tagged `reference-run-2026-08-22`. `.gitattributes` added with
  `* -text` so git can never rewrite line endings of the checksummed raw CSV
  (guarantee G4 would otherwise be breakable by a checkout).
- Raw-file SHA-256 re-verified: `data/raw/ckd-dataset-v2.csv` and the
  top-level copy both hash to `f24075f4...4ea84a`, matching the manifest.

**Found**

- The project was NOT under git despite docs/PROJECT_DOCUMENTATION.md N13
  saying otherwise ("The repository is now under git" was aspirational).
- The reference environment is missing: ambient CPython 3.14.5 has
  pandas 2.3.3 / numpy 2.4.4 (pins: 3.0.5 / 2.5.2) and no xgboost /
  interpret / shap / catboost, while the run manifest records
  `models_unavailable: {}`. Rebuilding `.venv` from requirements.txt.
- 213 tests collect in the ambient env vs 227 documented. Hypothesis:
  parametrization shrinkage from missing optional model deps. To be
  confirmed against the pinned env before any source change.

**Open**

- Full pytest run in the pinned env (pending pip install).
- Phase 0.2/0.3: registry, downloads, provenance report.
