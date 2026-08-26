# Reviewer-resolution matrix

**Base commit** `47e72fe` · **Opened** 2026-08-25

## Baseline established before any edit

Verified on the clean tree at `47e72fe`, immediately before the first change:

| Check | Result |
|---|---|
| `pytest tests/` | 459 passed, 0 failed, 0 errors, 0 skipped |
| `python reproduce.py` | REPRODUCTION COMPLETE (exit 0) |
| `ieee_paper.pdf` | 14 pages, 0 overfull, 0 undefined refs/citations |
| `supplement.pdf` | 3 pages, 0 overfull, 0 undefined refs/citations |
| Determinism | all 27 regenerated tables/datasets byte-identical to committed copies |
| Repository instructions | no `AGENTS.md`, `CLAUDE.md` or equivalent present |

## Matrix

| # | Reviewer issue | Scientific implication | Code files | Manuscript sections | New analysis | Verification | Status |
|---|---|---|---|---|---|---|---|
| 2 | Taxonomy mis-assigns `bu` and `al` | **The central leakage claim is mis-specified.** KDIGO defines CKD by the GFR limb *or* a kidney-damage marker, of which albuminuria is first-listed; blood urea is not a criterion at all. Classifying `bu` as incorporation risk and `al` as an ordinary predictor inverts both. Any "incorporation removed" claim was false: the albuminuria limb stayed in every model. | `scripts/22_build_feature_registry.py`, `src/ckd/features/registry.py`, `scripts/23_casemix_reanalysis.py` | §III-D taxonomy, §IV-B, Table III, §V-D | Re-run nested design over 10 registry-derived configurations | KDIGO 2024 primary source; registry test; regenerated `table_49` | **Done** |
| 3 | Case-composition claim overreaches | Feature removal shows non-necessity, not causal ranking | `scripts/23_casemix_reanalysis.py`, new severity analyses | §V-D, Conclusion, Abstract | Stage-specific, severity-balanced, standardised, simulation | Regenerated `table_49`, `table_54`, `table_55`; retracted-phrase test | **Done** |
| 4 | Matching-null inconsistency | A null that flatters the result would weaken the provenance claim | `src/ckd/data/provenance_sensitivity.py`, `scripts/21_provenance_sensitivity.py` | §V-E, Fig. 2--3, supplement | Copula-null dependence diagnostics; discrete-support fix | `table_53` diagnostics: mean \|real-null\| Spearman 0.031 over 78 pairs; 0 values off support | **Done** |
| 5 | Bootstrap under-specified and under-powered | Monte Carlo error unquantified | `scripts/15_procedure_bootstrap.py`, `scripts/25_bootstrap_convergence.py` | §V-G | Checkpoints at 100/200/500 (`table_56`); restricted cells added | Run at 500 launched; see verification report | **Partial** |
| 6 | Literature prevalence claims unsupported | Framing rests on unverified studies | `scripts/07_literature.py` | §I, §II, Table I | Verification attempted on 2 paywalled studies (both HTTP 403, recorded); counts split verified vs attributed | Literature test | **Done (fallback)** |
| 7 | Bibliography too thin | Claims lack primary support | `paper/ieee_paper.tex` | throughout | KDIGO 2024 verified and cited at the taxonomy | Citation test | **Partial** |
| 8 | MIMIC over-claimed | Exploratory result read as validation | `scripts/19_external_validation.py` | §V-K → supplement | PR-AUC added (0.313--0.326 vs 0.213 baseline); three overstatements rewritten | Phrase test | **Done** |
| 9 | Absolute claims remain | Evidence-calibration | manuscript | throughout | none | Retracted-phrase test | **Done** |
| 10 | Estimand terminology conflated | Reader cannot tell which uncertainty is which | `scripts/04_evaluate.py` | Methods, all tables | Estimands separated in captions; PR-AUC added | Consistency test | **Partial** |
| 11 | Manuscript too long | Journal limit | `paper/*.tex` | Synthetic-data section reduced to one future-work paragraph; full tables moved to supplement | Page count | **Partial** (15 pp) |
| 12 | Figures/tables and submission metadata | Submission readiness | plotting scripts | figures | Tables generated not typed; submission statements added; author checklist written | Figure tests | **Partial** |
| 13 | Tests do not guard the new claims | Regression risk | `tests/` | — | 9 new tests across taxonomy, nulls, literature | pytest | **Done** |
| 14 | Final verification | Agreement across artefacts | all | all | full pipeline | pytest + build + page inspection | Pending |

## Notes on scope

Items are marked **Done** only when code, regenerated results, manuscript text
and a passing test agree. "In progress" means the code and results exist but
the manuscript has not yet been brought into line.
