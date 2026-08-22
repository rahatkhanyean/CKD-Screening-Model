# WORKLOG

Newest entries at the top. One entry per work session / phase milestone.

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
