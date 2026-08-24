# Q1 revision — working plan and claims map

**Started** 2026-08-24 from commit `ab32757`.
**Baseline reproduction:** 416 tests pass; manuscript builds clean (11 pp,
0 overfull boxes, 0 undefined references). Environment: CPython 3.14.5,
numpy 2.5.2, pandas 3.0.5, scikit-learn 1.9.0, scipy 1.18.0, Windows 11.
Raw-data SHA-256 verified: `f24075f4…` (internal), `0eeea8d1…` (UCI-2015),
`97301a03…` (MIMIC-IV demo).

No `AGENTS.md` or equivalent repository instruction file exists.

---

## Scope decision, stated up front

The brief specifies 14 phases. Several are achievable at high quality in
this pass; two are not achievable honestly with the data and access
available, and the brief itself prescribes the fallback in those cases.
This plan states which is which **before** the work starts, so the final
report can be checked against it.

| Phase | Verdict | Note |
|---|---|---|
| 1 Inspect/reproduce | **Do** | Complete |
| 2 Scientific framing | **Do** | Intended-use statement; language audit |
| 3 Leakage taxonomy | **Do** | Machine-readable registry driving configs |
| 4 Provenance sensitivity | **Do** | The paper's key contribution; full suite |
| 5 Case mix / saturation | **Do** | Subgroup CIs; nested single-feature selection |
| 6 CV correctness | **Do** | Language fix + 1000-resample bootstrap |
| 7 Transportability probe | **Do** | Cohort documentation + tests |
| 8 Systematic review | **Fallback** | Paywalls block full-text verification of 10/13. Per brief: rename "targeted illustrative review", narrow abstract/intro claims, exclude unverified studies from quantitative claims |
| 9 Multi-dataset diagnostic | **Partial** | Run the diagnostic on the additional registered datasets we hold; if coverage is thin, rename to proof-of-concept per brief |
| 10 Restructure | **Do** | Main paper + supplement split |
| 11 Abstract | **Do** | Conservative rewrite |
| 12 Figures/tables | **Do** | Regenerate; captions carry n and exploratory labels |
| 13 Reproducibility | **Do** | Makefile, manifests, consistency tests |
| 14 QA | **Do** | Full pipeline + visual inspection |

---

## Defects found in Phase 1 (before any edit)

| # | Defect | Location | Severity |
|---|---|---|---|
| D1 | "25 **independent** outer test sets" — the 25 outer test sets are 5 repartitions of the same 200 patients; each patient is evaluated 5 times. They are not independent samples. | `ieee_paper.tex` §IV-C | **High** — misstates the effective sample |
| D2 | "108,000 out-of-fold predictions" presented in Methods and Reproducibility without stating these are repeated evaluations of 200 unique patients | `ieee_paper.tex` §IV-C, Reproducibility | **High** — invites reading as sample size |
| D3 | Multivariable lift selects the best single predictor on the same data used to evaluate the model, with no nested selection | `informativeness.py::best_single_predictor` | **Medium** — biases the *baseline* upward, so reported lift is conservative; still needs correcting |
| D4 | Whole-procedure bootstrap uses 200 resamples | `15_procedure_bootstrap.py` | **Medium** — brief requests ≥1000 |
| D5 | Provenance verdict rests on one matching configuration; no leave-one-variable-out, no outcome-excluded matching, no tolerance sensitivity | `provenance.py` | **High** — key contribution under-defended |
| D6 | Subgroup metrics reported without confidence intervals | `table_21_spectrum_analysis.csv` | **Medium** |
| D7 | Feature configurations hand-maintained in `configs.py`, not derived from a leakage registry | `features/configs.py` | **Medium** |
| D8 | Diagnostic protocol thresholds calibrated on one dataset, presented with named thresholds | §VI | **Medium** |

---

## Claims map

Every quantitative manuscript claim, its source, and verification status.
"Verified" = re-derived from the committed table by
`scratchpad/verify_claims.py` and found verbatim in the `.tex`.

| Claim | Dataset | Script / function | Result file | Manuscript | Status |
|---|---|---|---|---|---|
| Leaky config ROC-AUC 1.000 | internal | `03_nested_cv` → `04_evaluate` | `table_13_leakage_audit` | §V-A, Tab. II | Verified |
| Leakage closes 100% of headroom | internal | `04_evaluate` | `table_23_leakage_ceiling` | §V-A | Verified |
| 84% of cases stage s3–s5 | internal | `01_prepare_data` | `data_quality_report.json` | §V-B | Verified |
| Haemoglobin univariate AUC 0.968 | internal | `spectrum.py` | `table_22_univariate_separability` | §V-B, Tab. III | Verified |
| Early-stage sensitivity 0.969→0.905 (n=21) | internal | `spectrum.py` | `table_21_spectrum_analysis` | §V-B | Verified; **CIs to add (D6)** |
| Containment match fraction 1.000 vs null 0.003±0.003 | internal + UCI-2015 | `provenance.py` | `table_24_provenance` | §V-C, Fig. 2 | Verified; **sensitivity suite to add (D5)** |
| 187/200 unique pins, 0 contradictions / 10 held-out vars | internal + UCI-2015 | `provenance.py` | `table_24_provenance_agreement` | §V-C | Verified |
| sc binning cost 0.266 | internal + UCI-2015 | `recovery.py` | `table_31_binning_cost` | §V-D, Tab. IV | Verified |
| 339 cells constant-imputed | internal + UCI-2015 | `recovery.py` | `table_32_imputation_audit` | §V-E | Verified |
| Binned vs continuous max Δ 0.0122 | internal | `12_binned_vs_continuous` | `table_35` | §V-F | Verified |
| Procedure-level CIs 1.7–2.7× wider | internal | `15_procedure_bootstrap` | `table_38` | §V-G, Tab. V | Verified; **rerun at 1000 (D4)** |
| Encoding max deviation 0.0101 | internal | `14_encoding_robustness` | `table_37` | §V-H, Tab. VI | Verified |
| External 0.997→0.699–0.708 (95% CI 0.59–0.80) | internal + MIMIC | `19_external_validation` | `table_42` | §V-J, Tab. VII | Verified |
| Multivariable lift (lab +0.026) | internal | `20_benchmark_diagnostics` | `table_43_bid_lift` | §VI-A, Tab. VIII | Verified; **nested selection to add (D3)** |
| Saturation: 90% of gain at 10% | internal | `20_benchmark_diagnostics` | `table_43_bid_saturation` | §VI-B, Fig. 5 | Verified |
| Standardisation: AUC ≤0.008, sens ≤0.056 | internal | `20_benchmark_diagnostics` | `table_43_bid_standardisation` | §VI-C | Verified |
| Literature: 13 studies, 3 verified, 1 external | literature | `07_literature` | `table_26_prior_work` | §II, Tab. I | **Partly verified — 10/13 unverifiable (Phase 8)** |

---

## Execution order

1. Phase 6 (statistical correctness) — highest scientific priority
2. Phase 4 (provenance sensitivity) — defends the key contribution
3. Phase 5 (case mix, CIs, nested lift)
4. Phase 3 (feature registry)
5. Phase 9 (multi-dataset diagnostic)
6. Phase 7 (probe documentation)
7. Phase 8 (review honesty fallback)
8. Phases 2, 10, 11, 12 (manuscript)
9. Phase 13 (reproducibility engineering)
10. Phase 14 (QA)
