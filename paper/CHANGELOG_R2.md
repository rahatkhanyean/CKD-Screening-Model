# Change log — second revision (2026-08-25)

Base commit `47e72fe`. **No number was altered by hand.** Where a value
changed it is because a corrected analysis produced it, and the change is
recorded below.

Legend — **NEW**: analysis that did not exist · **FIX**: corrects a defect ·
**WEAKEN**: claim narrowed to match evidence · **DOC**: documentation or
infrastructure.

---

## A. The clinical correction

| # | Tag | Change |
|---|---|---|
| A1 | **FIX** | Feature registry re-derived against KDIGO 2024. `al` (urine albumin) moved from `legitimate_pre_index` to `diagnostic_criterion_input`: albuminuria (ACR ≥ 30 mg/g) is the first-listed marker of kidney damage. `bu` (blood urea) moved from `incorporation_risk` to `correlated_renal_biomarker`: it appears nowhere in the KDIGO definition. Sediment findings (`rbc`, `pc`, `pcc`, `ba`) separated as `possible_criterion_sediment`; `sg`, `sod`, `pot` as correlated biomarkers |
| A2 | **FIX** | Consequence of A1: the previous "incorporation removed" experiment never removed the albuminuria limb, so the claim it supported was unsupported. Re-run correctly, the conclusion holds — removing both KDIGO limbs and the sediment findings leaves ROC-AUC 1.000 |
| A3 | **FIX** | Consequence of A1: "legitimate pre-index" shrinks from 11 variables to 5. Performance on that set falls from a reported 0.994 to **0.930 (0.894–0.962)**, and a 3-variable set gives 0.881 (0.833–0.923). Neither is saturated. The old 0.994 was an artefact of misclassifying urinalysis and renal chemistry as pre-index |
| A4 | **NEW** | `incorporation_variables(strict=)` in `ckd.features.registry`, giving both a narrow (criterion inputs only) and a broad (plus sediment) reading, so the conclusion does not depend on where an arguable boundary is drawn |

## B. New analyses

| # | Analysis | Script | Output | Finding |
|---|---|---|---|---|
| B1 | **NEW** Ten registry-derived configurations | `23_casemix_reanalysis.py` | `table_49` | Incorporation removal: 1.000. Consequences: 0.998. Both: 0.995. Pre-index: 0.930. Minimal: 0.881 |
| B2 | **NEW** Case-mix re-weighting simulation | `26_casemix_simulation.py` | `table_54`, `fig_casemix_simulation` | 16% → 80% early-stage moves AUC 0.993 → 0.987 (low-cost); full valid unmoved at 1.000 |
| B3 | **NEW** Severity-balanced resampling without replacement | `26_casemix_simulation.py` | `table_55` | 9 cases per stage; 0.991 / 0.990 / 1.000 |
| B4 | **NEW** Copula-null dependence diagnostics | `provenance_sensitivity.py::copula_null_diagnostics` | `table_53` | Mean \|real − null\| Spearman 0.031 over 78 pairs; 0 synthetic values off support |
| B5 | **NEW** Authoritative null summary with exceedance bounds | `21_provenance_sensitivity.py` | `table_46_provenance_null_summary.csv` | 0 of 200 draws reach the observed value for either null; *p* ≤ 0.005 |
| B6 | **NEW** PR-AUC on the external cohort | `19_external_validation.py` | `table_42` | 0.313–0.326 against a no-skill baseline of 0.213 |
| B7 | **NEW** Bootstrap checkpoints and restricted cells | `15_procedure_bootstrap.py` | `table_56` | Checkpoints at 100/200/500/1000; conservative pre-index and no-incorporation cells added |
| B8 | **NEW** Specificity and Brier intervals | `23_casemix_reanalysis.py` | `table_49` | Completes the metric set the review asked for |

## C. Corrections to method

| # | Tag | Change |
|---|---|---|
| C1 | **FIX** | Copula null mapped synthetic values back with linear interpolation, placing them between the release's discrete bin representatives where no interval can contain them. This depressed the null mechanically. Now `method="nearest"`; the null **rose** from 0.0563 to 0.0586, the conservative direction |
| C2 | **FIX** | Whole-row permutation removed from the table of nulls and explained in the supplement as a property of the estimator. A test asserts it cannot return as an inferential null |
| C3 | **FIX** | External bootstrap intervals no longer described as excluding the internal estimate "by a wide margin"; no paired comparison was performed and the text now says so |

## D. Manuscript changes

| # | Location | Tag | Change |
|---|---|---|---|
| D1 | Abstract | WEAKEN | "frequently report" removed; results rewritten around both KDIGO limbs and the 0.930 pre-index figure; conclusion adopts the reviewer's wording |
| D2 | §III-D taxonomy | FIX | Rewritten around the KDIGO definition, with the dipstick-versus-ACR caveat stated explicitly |
| D3 | §V-D | FIX/WEAKEN | "Two candidate explanations are ruled out" → "the selected variables representing those pathways were not individually necessary" |
| D4 | §V-D (new) | NEW | "Does the severity mix explain the ceiling?" reporting the simulation and its limits |
| D5 | §V-K | WEAKEN | Three named overstatements rewritten; PR-AUC paragraph added |
| D6 | §VII-A | — | Synthetic-data section reduced from 53 lines to one future-work paragraph |
| D7 | Conclusion | WEAKEN | Ranking removed; reviewer's wording adopted; the 0.930 boundary result added |
| D8 | Reproducibility | DOC | Data-availability, code-availability, ethics and registration statements added |
| D9 | Table I | FIX | Now generated from the extraction sheet; splits verified from attributed. The claim of three cross-release studies becomes one verified plus two attributed |
| D10 | Tables VII, XII | DOC | Generated from CSVs rather than typed |
| D11 | throughout | WEAKEN | "honest uncertainty" → "whole-procedure uncertainty" |

## E. New tests

| Class | Guards |
|---|---|
| `TestCorrectedTaxonomyReachesTheManuscript` | Restricted sets actually remove both KDIGO limbs; manuscript no longer names blood urea as a criterion; albuminuria is discussed |
| `TestMatchingNullsAreLabelledAsImplemented` | Only implemented nulls are reported; row permutation is not offered as a null; the copula null preserves dependence and support |
| `TestLiteratureClaimsRestToVerifiedStudies` | No prevalence language; the cross-release claim matches the verified count |
| `TestTaxonomyContent` (extended) | Both KDIGO limbs flagged; blood urea excluded from both incorporation readings; strict ⊂ broad |

## F. Findings that changed

| Finding | Before | After | Cause |
|---|---|---|---|
| Pre-index-only discrimination | 0.994 (11 variables) | **0.930 (0.894–0.962)**, 5 variables | A1/A3 — misclassification corrected |
| Incorporation-removed discrimination | 1.000, but albuminuria retained | 1.000, both limbs removed | A2 |
| Copula null | 0.0563 ± 0.0140 | 0.0586 ± 0.0142 | C1 — discrete support preserved |
| SD distance of observed from null | 73, then 67 | 66 | C1 |
| Cross-release studies asserted | 3 | 1 verified, 2 attributed | Verification attempted; both paywalled |
| External evaluation metrics | ROC-AUC only | ROC-AUC + PR-AUC vs baseline | B6 |

## G. Findings that did not change

Leakage audit (all configurations → 1.000); case-mix composition (107/128 at
s3–s5; haemoglobin 0.968); match fraction 1.000 with 187 unique pins and zero
contradictions on ten held-out variables; binning cost for creatinine
(0.2662); 339 constant-imputed cells; transportability probe ROC-AUC
(0.699–0.708); saturation at 10% of training data.
