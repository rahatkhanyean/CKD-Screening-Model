# Change log — Q1 revision (2026-08-24)

Base commit `ab32757`. Every materially revised passage, new analysis and new
artefact. **No previously reported number was altered by hand**; where a value
changed it is because a corrected analysis produced it, and the change is
noted.

Legend — **NEW**: analysis that did not exist before · **FIX**: corrects a
defect · **WEAKEN**: claim narrowed to match evidence · **DOC**:
documentation or infrastructure.

---

## A. New analyses

| # | Analysis | Script | Output | Finding |
|---|---|---|---|---|
| A1 | **NEW** Provenance sensitivity suite | `21_provenance_sensitivity.py` | `table_44`, `table_45`, `table_46`, `fig_provenance_sensitivity` | Match survives outcome exclusion (182 unique pins), leave-one-out (133–187), imputation-variable removal (131), tolerance 0–10⁻² (179–187) |
| A2 | **FIX** Copula null replacing a degenerate one | `provenance_sensitivity.py::copula_null` | `table_46` | 0.056 ± 0.014 vs observed 1.000 (~67 SD; see G8). Replaces a whole-row permutation null that was invalid by construction |
| A3 | **NEW** Matched-pair audit | `21_provenance_sensitivity.py` | `table_45` | 12 reproducibly sampled pairs for manual inspection |
| A4 | **NEW** Feature registry | `22_build_feature_registry.py` | `data/feature_registry.csv`, `table_47` | 29 variables × 18 fields; 4 variables barred outright |
| A5 | **NEW** Subgroup metrics with intervals | `23_casemix_reanalysis.py` | `table_48` | Early-stage sensitivity 0.905 (0.762–1.000) vs 0.969 (0.937–0.992): **not resolved** |
| A6 | **NEW** Registry-driven feature restriction | `23_casemix_reanalysis.py` | `table_49` | Removing incorporation inputs: 1.000. Removing consequences: 0.998. Both: 0.991. Pre-index only: 0.994 |
| A7 | **FIX** Nested single-feature baseline | `informativeness.py::nested_best_single_auc` | `table_50` | Lift unchanged to 4 dp; selection stable (1 feature, 100% of folds) |
| A8 | **NEW** Bootstrap convergence | `25_bootstrap_convergence.py` | `table_51`, `fig_bootstrap_convergence` | Endpoint SD 0.003 by 175 resamples vs interval width 0.038 |
| A9 | **DOC** LaTeX fragment generation | `24_make_latex_tables.py` | `paper/generated/*.tex` | Manuscript tables generated, not typed |

## B. Manuscript changes

| # | Location | Tag | Change |
|---|---|---|---|
| B1 | Abstract | WEAKEN | Rewritten to structured form (Background / Objective / Methods / Results / Limitations / Conclusion). States 200 unique patients; five evaluations per patient; labels early-stage analysis post hoc and underpowered with its interval; labels MIMIC an exploratory stress test with its interval; removes "none of which is predictive ability"; labels the framework not validated |
| B2 | Index Terms | DOC | Added `transportability` |
| B3 | §I-A Contributions | WEAKEN | Transportability probe and audit framework restated as stress test and proof of concept |
| B4 | §II title and opening | WEAKEN | "Related Work" → "**A Targeted Illustrative Review**"; non-systematic sampling disclosed; ten unverified studies support no quantitative claim |
| B5 | §II Table I | WEAKEN | Every row labelled *verified* or *attributed*; caption states no prevalence should be inferred |
| B6 | §I intro | WEAKEN | Literature claim narrowed from prevalence to existence |
| B7 | §III-A **new** | NEW | Intended-use specification: population, setting, index time, information at index, outcome, task type, decision. Records that index time, per-patient availability and chronicity are unidentifiable; reframes the work as a benchmark audit |
| B8 | §IV-B **new** | NEW | Feature-availability and leakage taxonomy, with generated Table II |
| B9 | §IV-C | FIX | "25 independent outer test sets" removed. States effective sample is 200 unique patients, five evaluations each, repeated evaluations do not increase sample size, 108,000 rows are an artefact size |
| B10 | §V-C | WEAKEN | Subgroup table with intervals; states the early-stage decline is not statistically resolved; 21-case denominator throughout |
| B11 | §V-C Fig. 1 caption | WEAKEN | "systematically optimistic" → "appears optimistic"; states 21 cases and exploratory status |
| B12 | §V-D **new** | NEW | Which information pathway produces the ceiling; adopts "cannot be cleanly separated" wording; states high discrimination is real, not absence of predictive ability |
| B13 | §V-E | NEW/WEAKEN | Copula null; sensitivity table and figure; saturation caveat about match fraction; claim weakened to "discretised subset or re-release"; states maintainers not contacted |
| B14 | §V-G | DOC | Full bootstrap specification (sampling unit, stratification, fold construction, seeds, failures) and convergence analysis with the 1,000-resample limitation stated |
| B15 | §VI title and opening | WEAKEN | "A Proposed Diagnostic" → "**A Proof-of-Concept Pre-Modelling Audit Framework**"; states not validated; thresholds moved to supplement |
| B16 | §VI-A | FIX | Nested selection of the single-feature baseline described and its null result reported |
| B17 | §VI closing | WEAKEN | Verdict language removed; quantities reported instead of flags |
| B18 | Reproducibility | FIX | 108,000 relabelled as repeated evaluations of 200 patients |

## C. New tests

| Test file | Count | Guards |
|---|---|---|
| `tests/test_registry.py` | 18 | Registry-derived configurations reproduce `configs.py` exactly; taxonomy content; evidence and uncertainty notes present |
| `tests/test_manuscript_consistency.py` | 25 | Input hashes; manuscript numbers vs generated tables; no TODO/internal labels (author placeholder excepted and required to be marked); prohibited variables absent; no non-independent dataset in external tables; figures/tables exist; no dangling refs; no uncited bibliography; no hardcoded section refs. Resolves `\input` so generated fragments are checked. Also extracts text from every included figure with `pdftotext` and fails if a figure draws an internal label inside itself |

## D. New infrastructure

| Artefact | Purpose |
|---|---|
| `reproduce.py` | Portable one-command pipeline; verified on the reference environment |
| `Makefile` | POSIX equivalent (**not** verified — `make` is unavailable on the reference machine) |
| `paper/supplement.tex` / `.pdf` | Registry detail, full sensitivity results, candidate thresholds, synthetic-data discussion, reproducibility, checklist |
| `paper/generated/` | LaTeX fragments emitted from CSVs, including `numbers.tex`, which defines the values quoted inline as macros so prose cannot detach from data |
| `REVISION_PLAN.md` | Working plan, claims map, defect register (D1–D8) |
| `.gitattributes` | Normalises generated text results to LF, so a re-run on another platform produces a diff only where a value actually changed |

## E. Findings that changed

| Finding | Before | After | Cause |
|---|---|---|---|
| Provenance null | 0.003 ± 0.004 (column permutation only) | Additionally 0.056 ± 0.014 (copula, 200 draws) | A2 — stricter null |
| Early-stage sensitivity | 0.905, no interval | 0.905 (0.762–1.000), not resolved vs 0.969 (0.937–0.992) | A5 |
| Multivariable lift | Non-nested selection | Nested; **values unchanged** | A7 |
| Case mix as "largest mechanism" | Asserted | "Principal explanation… cannot be cleanly separated" | A6 |

## G. Fixes from the final QA pass

Found by inspecting the built PDF and by re-running the pipeline end to end,
after the substantive revision was otherwise complete.

| # | Tag | Change |
|---|---|---|
| G1 | **FIX** | `paper/figures/` --- the directory the LaTeX build reads --- was populated by hand and refreshed by no stage. A re-run regenerated figures into `reports/figures/` and then built the manuscript from the previous copies. `sync_figures()` in `reproduce.py` and a `syncfigs` Makefile target now copy every included figure before either build and fail if one is missing |
| G2 | **FIX/WEAKEN** | Fig. 1 drew the internal title *"Figure R12. Case-mix analysis: why near-ceiling discrimination here is not a screening result"* inside the graphic --- an internal label the manuscript's own numbering contradicts, and a claim stronger than the weakened wording of §V-D. Retitled at source to *"Case-mix analysis: discrimination and sensitivity by disease severity (EXPLORATORY)"*; interpretation now lives only in caption and text |
| G3 | **DOC** | New test extracts text from every included figure and fails on an embedded internal label --- the check that would have caught G2 |
| G4 | **FIX** | Supplement spilled two lines onto a blank third page; two paragraphs tightened at source. Two clean pages, no overfull boxes |
| G5 | **DOC** | `.gitattributes` added after confirming that all 27 regenerated tables and processed datasets are byte-identical to the committed copies, and that the entire apparent diff was line endings |
| G6 | **WEAKEN** | The **Conclusion still contained two claims the body had already retracted**: that case mix "is the larger effect" (§V-D says its contribution cannot be cleanly separated) and that the two releases "share their patients record-for-record" (§V-E explicitly disclaims exact raw-value identity). Both rewritten to match the body: case mix "appears to be the principal explanation, though its contribution cannot be cleanly separated"; the releases "draw on the same patients --- all 200 analysed records match into the 2015 release, 187 of them uniquely" |
| G7 | **DOC** | New test class `TestRetractedClaimsStayRetracted` greps manuscript and supplement for seven phrases retracted during this revision, each with the reason. G6 went undetected through an entire revision cycle because nothing checked that the Conclusion agreed with the Results |
| G8 | **FIX** | **A stale headline number.** The copula-null statistics were quoted from a 100-draw run while the stage default is 200. Manuscript said $0.055 \pm 0.013$, maximum over 100 draws, 73 SD from the observed value; the pipeline produces $0.056 \pm 0.014$, 200 draws, 67 SD. Corrected --- **not by retyping**: the values are now generated into `paper/generated/numbers.tex` and the manuscript quotes macros |
| G9 | **FIX** | The test that should have caught G8 asserted only that the formatted value was a *substring* of the manuscript, and passed because an unrelated 0.056 in §VI matched. Replaced with a field-by-field comparison of the generated fragment against its source CSV, and confirmed to fail when a value is perturbed |
| G10 | **FIX** | Figure axis labels and titles carried literal markdown backticks (`sc`, `pot`) into the typeset PDF, in Figs. 1, 4 and 5. Fixed at source in `scripts/04_evaluate.py`, `10_ebm_shapes.py` and `11_continuous_recovery.py`, which now set a monospace font instead of marking up the text. Fig. 4's title was also ungrammatical and is reworded |
| G11 | **FIX** | Supplement Table S2 promised "the complete leave-one-variable-out results" and shipped a stub pointing at the CSV. Now generated in full from `table_44` by `scripts/24_make_latex_tables.py` |
| G12 | **FIX** | Supplement gave `make reproduce` as the one-command path while `make` is unverified on the reference machine. Changed to `python reproduce.py`, with the Makefile named as an unverified convenience |
| G13 | **FIX** | Reproducibility statement said 416 automated tests; the suite holds 459. `reproduce.py` now writes the pytest inventory to `table_52_test_inventory.csv` and the manuscript quotes a generated `TestCount` macro, so the count cannot go stale |
| G14 | **WEAKEN** | Discussion said the three mechanisms are "none is predictive ability" and the Conclusion ranked them ("the largest of which"). Both retracted elsewhere in the revision; both now rewritten and added to the retracted-phrase guard |

## F. Findings that did not change

Leakage audit (all configurations → 1.000); case-mix composition (107/128 at
s3–s5; haemoglobin 0.968); match fraction 1.000 with 187 unique pins and zero
contradictions on ten held-out variables; binning cost for creatinine
(0.2662); 339 constant-imputed cells; binned-vs-continuous null result
(0.0122); encoding robustness (0.0101); transportability probe
(0.699–0.708); saturation (10% of training data).
