# PLAN — From feasibility study to cautionary methods paper

**Working thesis of the reframed paper:** two independent mechanisms — target
leakage and case-mix spectrum bias — both push measured performance toward 1.0
on a widely-used public CKD benchmark, and the second matters more. The
low-cost comparison becomes a supporting methods contribution.

Each task lists **files**, **tests**, and **gates** (what must be true before
the next thing runs). Nothing below runs a full nested-CV job until Phase 0's
provenance gate passes; restricted grids are used first everywhere.

---

## Phase 0 — Baseline integrity + provenance gate

### 0.1 Restore the reference environment and baseline (prerequisite to everything)

Findings from pre-planning that must be resolved first:

- The project is **not under git** (`git status`: not a repository). Only
  `.gitignore` exists. The working method requires per-phase commits.
- The ambient interpreter (CPython 3.14.5) does **not** match
  `requirements.txt` pins (pandas 2.3.3 vs pinned 3.0.5, numpy 2.4.4 vs
  2.5.2) and lacks `xgboost`, `interpret`, `shap`, `catboost`. The reference
  run's manifest shows `models_unavailable: {}`, so it ran in a venv that is
  no longer present here.
- **213 tests collect** in the ambient environment vs the documented 227 —
  expected to be parametrization shrinkage from the missing optional models;
  must be confirmed, not assumed.

Tasks:
- `git init`; initial commit of the exact current state (code + reports +
  processed artefacts + raw CSV) so every later change is diffable against
  the reference run. Tag `reference-run-2026-08-22`.
- Create `.venv` from `requirements.txt`; run the **full** test suite; record
  collected/passed counts in WORKLOG.md. Baseline must be green (227
  expected) before any code change.
- Verify raw-file SHA-256 against the manifest (`f24075f4…4ea84a`).
- Create `WORKLOG.md` with the baseline entry.

Files: `WORKLOG.md` (new), `.venv/` (untracked).
Gate: full suite green in the pinned environment. **No source change before
this passes.**

Caution to record (not a blocker): the repo lives under Google Drive sync
(`G:\My Drive\...`). Git and Drive sync can conflict on `.git` internals; the
worklog will note this and recommend pausing sync during heavy stages or
relocating the repo — proceeding here as instructed either way.

### 0.2 External-dataset registry and harmonization layer

Files (new): `src/ckd/data/external.py`, `config/external_datasets.yaml`,
`tests/test_external.py`.

- `DatasetRecord` registry: id, citation, expected n / class balance /
  columns, acquisition status (`OBTAINED` / `UNAVAILABLE(reason)`), license,
  local path under `data/external/<id>/raw/`, SHA-256 once obtained.
- Harmonization layer: per-dataset column map onto the internal schema
  (shared feature names, unit reconciliation, target coding to {0,1}),
  mirroring the `VariableSpec` pattern — any variable that cannot be mapped
  with confidence is flagged `uncertain` with a note, never guessed.
  Conservative mapping may only ever UNDERSTATE low-cost performance.
- **Per-dataset prohibited sets** declared in the registry (UCI-2015 has no
  `affected`/`stage`/`grf`, but any eGFR- or stage-derived column in any
  source joins that source's `POST_DIAGNOSIS` set). `LeakageGuard` already
  accepts a `forbidden` parameter, so the extension is declarative — plus
  tests that the guard actually receives each source's set.

Candidates to register (verify availability; record failures rather than
assuming):

| id | Expectation | Role if INDEPENDENT |
|---|---|---|
| `uci2015` | UCI "Chronic Kidney Disease" (Rubini, Soundarapandian, Eswaran; Apollo Hospitals, India; 400 records; CONTINUOUS values; CC BY 4.0) | Primary external validation + continuous-value re-run (3d) |
| `uci2023_v2` | "UCI-2023 / Enam Medical College" 200-record release — **almost certainly our own file re-cited**; registered precisely to *prove* SAME-SOURCE and close the door on it | None (negative control for the gate) |
| `icddrb_kabir` | icddr,b community Bangladesh cohort (Kabir et al.); community/early-stage-enriched | Case-mix re-test (3e) — the single most important external result |
| `birdem` | BIRDEM diabetic CKD, Dhaka, 400 diabetics, longitudinal | Sensitivity analysis only |
| `mimic_iv_demo` | MIMIC-IV Clinical Database Demo, PhysioNet, ~100 patients, no credentialing | Distribution stress-test appendix only |

Acquisition: `uci2015` and `mimic_iv_demo` are open downloads (network
permitting). `icddrb_kabir` and `birdem` likely require author contact — if
so, the registry marks them `UNAVAILABLE`, the provenance report says so, and
Phase 3e is explicitly blocked-on-data rather than silently dropped.
**Open question 3.**

### 0.3 Record-level overlap check → provenance report

File (new): `scripts/00_provenance.py` →
`reports/external/provenance_report.md` plus
`reports/tables/table_24_provenance.csv` (the report-from-tables invariant
holds for provenance numbers too).

Evidence collected per candidate, in increasing order of specificity:

1. **Cohort arithmetic**: n, class balance (ours: 200; 128/72), stage
   distribution where present.
2. **Schema fingerprint**: column-name overlap (the v2 file shares UCI-2015's
   abbreviations — suggestive, not conclusive).
3. **Distributional**: per shared variable, empirical distribution vs ours;
   for binned-vs-continuous pairs, **bin-boundary forensics** — do our
   published bin edges (e.g. `>= 227.944`) coincide with quantiles or
   observed values of the candidate's continuous variable? Exact
   floating-point edge matches are strong evidence of derivation.
4. **Record-level**: canonicalized row hashing and near-duplicate matching on
   shared columns (exact for binned-vs-binned; interval-containment matching
   for binned-vs-continuous: a candidate row "matches" if every continuous
   value falls inside the corresponding bin of one of our rows — report the
   maximum bipartite match fraction, with row-order and duplicate-hash checks).
5. **Verdict**: INDEPENDENT / OVERLAPPING / SAME-SOURCE, with the evidence
   table, written into the report.

Test (new — the gate itself): `tests/test_external.py::TestProvenanceGate`
fails if any dataset not classified INDEPENDENT appears in any
external-validation output table (reads verdicts from
`table_24_provenance.csv`, scans all `table_*_external*`/frozen-transfer/
cross-dataset tables for dataset ids).

Gate for Phase 3: provenance report exists, verdicts committed, gating test
green.

---

## Phase 1 — Cheap, high-value (internal data only; runs in parallel with 0.2–0.3 once 0.1 is done)

### 1a. Fix internal report inconsistencies

Files: `scripts/06_write_report.py`, possibly `scripts/04_evaluate.py`
(upstream provenance columns); `tests/test_report.py` (new).

- **Abstract "slope n/a"**: `scripts/06_write_report.py` line ~272 injects
  `fmt(best['calibration_slope'], 2)`, which renders `n/a` when the slope is
  undefined under complete separation. Fix: the abstract mirrors §5.6's
  handling — "calibration slope not identified (complete separation); Brier
  0.0036" — injected from the same table, never hand-written.
- **0.992 vs 0.993 low-cost AUC**: §5.2/5.3 report the best *uncalibrated*
  cell (random forest) while §5.9/5.10 report SVM+isotonic. Every table cell
  and injected sentence gains explicit (config × model × calibration)
  provenance so no two numbers can look contradictory; where both appear,
  the text states why they differ.
- **Conclusion error**: "0.993 against 1.000 for the full laboratory panel"
  — 1.000 is Full valid; Laboratory is 0.995. Fix the injection to name the
  configuration it actually quotes.
- Tests: scan the generated report for (i) unresolved template tokens and
  `n/a` outside allowed contexts, (ii) headline numbers equal to their
  source-table values (parse the markdown, compare against the CSVs),
  (iii) no Full-valid number attributed to "laboratory".

### 1b. Prior-work literature table (the uncited motivating claim)

Files: `data/literature/prior_work.csv` (new), `scripts/07_literature.py`
(new; renders → `reports/tables/table_26_prior_work.csv` + markdown table),
Introduction rewrite in stage 6; `tests/test_literature.py` (new: schema,
allowed values, no empty cells — `unclear` is a value, blank is a bug).

Columns: citation, doi_or_url, dataset_version (v2-binned /
uci2015-continuous / other), reported_headline_metric, metric_value,
used_affected / used_stage / used_grf (yes/no/unclear),
did_external_validation, validation_design, notes.

Method: use web search to identify papers using this dataset (UCI citation
lists, scholar trails); populate only what is verifiable from the paper's
text; every uncertain cell is `unclear`; a clearly-marked TODO block lists
rows needing hand confirmation. The Introduction's "~100% accuracy is
frequently reported" claim is rewritten to cite the table: "of N papers
surveyed, k report ≥0.99 accuracy; of those, m used at least one
post-diagnosis column and none validated externally" — all injected. This
converts the paper's weakest point into its empirical opening.

### 1c. Label-robustness (N4)

Files: prefer a `--label-variant` flag on stage 3 over a separate script so
the CV engine stays single-sourced (decided during the 3b′ refactor design);
output `table_27_label_robustness.csv`; injected report section.

Grid: {low_cost, full_valid, laboratory} × {SVM, RF} × {none, isotonic},
same 5×5×4 design, seeds derived from master. Variants: (a) exclude CSV
lines {12, 18, 52, 123}; (b) flip their labels. Report deltas vs headline
cells. Minutes of compute; runs after its design checks, before any
external work. Tests (`tests/test_sensitivity.py`): variants touch exactly
those four lines; main artefacts untouched; records are never silently
dropped from the primary tables.

### 1d. Interpretable rule at equal cost (N8)

Files: extend `scripts/05_importance_stability.py` (EBM shape-function
extraction for low_cost × EBM × isotonic),
`src/ckd/evaluation/importance.py` (shape export helper), figure
`fig_r13_ebm_shapes`, `table_28_ebm_vs_svm.csv`; report gains an explicit
"best-discriminating (SVM 0.993) vs best-explainable at equal cost
(EBM 0.987, slope 1.01)" subsection. Predictions already exist in
`cv_predictions.csv.gz`; only shape extraction needs (cheap) fits.
Test: shapes are direction-consistent with the univariate association for
the four stable features (`sg`, `al`, `dm`, `htn`), or the discrepancy is
recorded in a notes file (mirror `table_20_importance_notes.txt`).
Guardrail preserved: `interactions=0` — the existing no-interaction rule.

### 1e. Optimism accounting (N10)

Files: small `scripts/09_optimism.py` (one resubstitution fit per headline
cell), `table_29_optimism.csv`, injected paragraph reporting apparent minus
nested-CV performance per cell. Test: apparent ≥ nested-CV AUC in every
cell — if not, something is broken and the test should scream.

Phase-1 exit: full suite green (baseline + new tests), commit, WORKLOG entry.

---

## Phase 2 — Reframe the report around the two-mechanisms thesis

Files: `scripts/06_write_report.py` (restructure), `README.md`, docs.

- New working title, to be finalised with you (open question 7):
  *"Two mechanisms inflate reported performance on a widely used public CKD
  dataset: a leakage-controlled, case-mix-aware re-analysis."*
- Structure: (1) literature table as the empirical opening (1b);
  (2) leakage mechanism + headroom analysis (tables 13, 23) as Result 1;
  (3) spectrum/case-mix analysis (tables 21–22) elevated to Result 2 — the
  "near-ceiling AUC coexists with materially worse early-stage sensitivity"
  dissociation becomes the central finding; (4) low-cost vs laboratory as a
  supporting result about cost-stratified feature sets, not deployment;
  (5) robustness block (1c/1e, later 4a) as Result 4.
- Bangladesh/screening deployment framing removed from all claims; the
  screening question survives only as motivation for the cost tiers.
- Existing guardrails hold: leaky model quoted only inside the audit; the
  prespecified selection rule reported unchanged alongside the §5.10-style
  critique; every number injected from tables.
- Tests: 1a's report-consistency tests re-run against the new structure;
  add a check that no headline number is asserted as "screening
  performance."

Exit: regenerate report end-to-end (stages 4→6; no new CV), suite green,
commit.

---

## Phase 3 — External validation AND retraining (gated on Phase 0 verdicts)

### 3b′ (first): dataset-parameterized pipeline refactor

The engineering spine of Phase 3, so it precedes 3a despite the numbering.

Files: `src/ckd/data/load.py`/`clean.py` (accept a `DataSource`),
`scripts/03_nested_cv.py` (`--dataset` flag, default `internal_v2`),
`src/ckd/features/configs.py` (configurations become per-dataset views over
the harmonized schema; internal behaviour byte-identical),
`config/experiment.yaml` (per-dataset sections); CV engine itself unchanged.

- Seed derivation: `dataset_seed = stable_hash(master_seed, dataset_id)` —
  deterministic, documented, tested (same master seed → identical
  predictions per dataset; different datasets → different partitions).
- Guard: each source's prohibited set flows from the registry into
  `build_pipeline`; `assert_pipeline_is_clean` runs per source; adversarial
  test hands each valid external pipeline that source's post-diagnosis
  columns and requires refusal.
- **Regression gate: after the refactor, a restricted-grid re-run on
  `internal_v2` must reproduce the reference predictions bit-for-bit**
  (determinism tests extended to assert against a stored fingerprint of
  `cv_predictions.csv.gz`). No external run until this passes.

### 3a. Frozen transfer (both directions)

Files: `scripts/10_external_validation.py`, `table_30_frozen_transfer.csv`.
Train on internal data (inner-CV-tuned per the reference design), predict
the external set without refitting; then the reverse. Report discrimination
AND calibration (slope/intercept/Brier); the report registers in advance the
framework's prediction that the slope degrades more than the AUC. Harmonized
shared features only; PPV/NPV restated at external prevalence.

### 3b. Retrain from scratch on each INDEPENDENT dataset

Same engine, same guards, seeds from master via dataset hash; nested CV on
the external source. Restricted grid first; full grid only after provenance
and design checks (the standing rule). `table_31_external_nested_cv.csv`,
fully comparable to internal tables, injected.

### 3c. Cross-dataset training

Train on the harmonized union (shared features only), evaluate held-out
**per source** — never a pooled number that hides source shift.
`table_32_cross_dataset.csv`. Test: the table must contain one row per
source and no pooled row.

### 3d. Continuous-value re-run (N3; rides on 3b for uci2015)

Encode uci2015 two ways: (i) raw continuous values; (ii) our bin-midpoint
encoding applied through the v2 bin edges. The difference isolates what
pre-discretisation cost. Thresholds restated in clinical units (mg/dL,
g/dL) for the continuous fit. `table_33_binning_cost.csv`.

### 3e. Case-mix re-test (the most important external result)

On a community/early-stage-enriched source (icddr,b if obtainable):
re-evaluate whether low-cost sensitivity falls on early-stage patients as
the spectrum analysis predicts (internal: 0.969 → 0.905). Headline
placement in the reframed paper. If the data cannot be obtained, the report
states that plainly and the prediction stands as a registered, testable
claim rather than a result. `table_34_casemix_retest.csv`.

Exit: full suite green (now including the provenance gate + external
tests), commit, WORKLOG, report regenerated.

---

## Phase 4 — Deepen the internal analysis

- **4a Encoding robustness (N5):** bin-index, one-hot-per-bin, and
  rank-normalised encodings on a restricted grid; stage-3 `--encoding`
  flag; `table_35_encoding_robustness.csv`. Tests: each encoding passes the
  same statelessness property test as the midpoint encoding (re-encode
  arbitrary row subsets → unchanged).
- **4b Operating point → decision (N7):** specify a referral-capacity
  constraint (open question 6 — I will propose a sourced scenario value,
  flagged as a scenario, not a recommendation); derive the threshold INSIDE
  training folds; report the confusion matrix there.
  `table_36_capacity_threshold.csv`.
- **4c Whole-procedure bootstrap (N6):** refit inside each resample,
  ~200 resamples × 3–4 headline cells. **Gated on Phase 3 showing the
  result survives externally at all.** Background-run, cached, resumable.
- **4d Calibration experiment (N9):** repeats → ~20 on a restricted grid;
  add beta calibration as a third method (pinned `betacal` if installable
  in the reference env, else a direct implementation in
  `evaluation/metrics.py` tested against synthetic calibration maps);
  re-examine isotonic-beats-Platt. `table_37_calibration_experiment.csv`.

---

## Phase 5 — Dissemination / engineering

- **5a** TRIPOD+AI checklist as a generated appendix table
  (`table_38_tripod_ai.csv`), each row naming the section/artefact that
  satisfies the item; `not-satisfied` is an allowed, visible value.
- **5b** Environment pinning: `requirements.lock.txt` (full freeze) +
  optional Dockerfile; `scripts/make_archive.py` for a Zenodo-ready archive
  (clean-tree check, tag, zip, checksums).
- **5c** CI: pytest + stages 1, 2, 4, 6 on every commit, stage-3 predictions
  cached as an artifact keyed by manifest hash (assumes a GitHub remote —
  open question 2).
- **REVIEWER_RESPONSE.md**: objection → artifact map (title framing → Phase 2;
  uncited motivating claim → 1b; n=200 → reframed contribution + 3b; single
  dataset → 3a–3d; internal inconsistencies → 1a; "explainable" vs RBF SVM
  → 1d; label noise → 1c; optimism → 1e; spectrum claim untested externally
  → 3e or registered prediction).

---

## New-tests summary (no new module is done without its tests)

| Area | Test file | Key assertions |
|---|---|---|
| Provenance gate | `tests/test_external.py` | non-INDEPENDENT dataset can never appear in an external-validation table; registry schema; harmonization maps only declared columns; per-source guard refusal |
| Report consistency | `tests/test_report.py` | no unresolved tokens; headline numbers equal source-table values; provenance-labelled cells |
| Literature table | `tests/test_literature.py` | schema; no blank cells; `unclear` only as an explicit value |
| Refactor regression | extend `tests/test_reproducibility.py` | internal predictions bit-identical pre/post refactor; per-dataset seeds deterministic and distinct |
| Label variants | `tests/test_sensitivity.py` | variants touch exactly CSV lines {12,18,52,123}; primary artefacts untouched |
| Encodings | extend `tests/test_data_loading.py` | statelessness per encoding |
| Beta calibration | extend `tests/test_metrics.py` | recovers known synthetic calibration maps |

## Sequencing & gates

```
0.1 baseline+git ─► 0.2/0.3 provenance ─► 3b′ refactor ─► 3a…3e ─► 4c
       │
       └─► 1a…1e (parallel with 0.2/0.3) ─► 2 reframe ─► 4a, 4b, 4d ─► 5
```

Full suite + commit + WORKLOG entry at every phase boundary. Expensive runs
only after their phase's design checks pass; restricted grids first, always.

## Open questions (⛔ = genuinely blocking; others proceed under the stated assumption)

1. **Environment (assumption):** I will create `.venv` from
   `requirements.txt` and treat it as the only environment for all runs. If
   pinned versions fail to install on this machine (Python 3.14 wheels), I
   report the exact failures before substituting anything.
2. **Git remote / CI:** no remote exists. Phase 5c assumes GitHub — confirm
   before Phase 5. Local git starts immediately regardless.
3. **⛔ Data acquisition (blocks 3e only):** icddr,b (Kabir et al.) and
   BIRDEM are likely not open downloads. If web search confirms that, the
   case-mix re-test needs you to obtain them (author contact or
   institutional access). Everything else proceeds; 3e is the only task
   that may end as a registered prediction instead of a result.
4. **Network downloads (assumption):** I will download UCI-2015 and the
   MIMIC-IV demo into `data/external/` with recorded SHA-256s (both are
   openly licensed). Flag now if this machine must stay offline.
5. **227 vs 213 tests (verified in 0.1):** expected to be optional-dependency
   parametrization; if the pinned env still collects ≠ 227, I diff the
   collection list and report before proceeding.
6. **Referral-capacity value (4b):** I will propose a sourced scenario value;
   flag if you have a setting-specific number.
7. **Final title (Phase 2):** I will draft 2–3 options in the WORKLOG; the
   reframe does not wait on the wording.
8. **Google Drive + git caution:** `.git` under Drive sync is a known
   corruption risk; say the word to relocate the repo first, otherwise I
   proceed in place and note it.
