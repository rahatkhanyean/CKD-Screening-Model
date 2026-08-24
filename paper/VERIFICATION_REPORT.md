# Verification report — minor revision (2026-08-24)

## 1. Compilation

| Check | Result |
|---|---|
| Engine | pdfTeX (MiKTeX), `pdflatex` run twice for cross-references |
| Exit status | **0** (clean; `-halt-on-error` in force) |
| Output | `paper/ieee_paper.pdf`, **11 pages** (was 10 before revision) |
| Overfull boxes | **0** |
| Undefined references or citations | **0** |
| Underfull boxes | 14 (13 `\hbox`, 1 `\vbox`) |

**On the underfull boxes.** These are typesetting badness warnings, not
errors, and are normal in two-column IEEE layout with justified text: they
report lines LaTeX stretched more than ideal, chiefly in narrow table cells
and around inline `\texttt{}` runs. None causes visible overlap, clipping or
loss of content. They were present before this revision and are not
introduced by it. No content-affecting fix is warranted; forcing them out
would require rewording sentences purely for spacing.

## 2. Cross-references and citations

Verified with `scratchpad/ref_audit.py` and `scratchpad/cite_audit.py`:

| Check | Result |
|---|---|
| Labels defined | 23 |
| `\ref` uses | 23 |
| Dangling references (used, never defined) | **none** |
| Unused labels (defined, never used) | **none** |
| Bibliography entries defined | 12 |
| Entries cited in text | 12 |
| Uncited entries | **none** |
| Missing citation keys | **none** |
| Hardcoded section cross-references (e.g. "Section V-B") | **none** — one was found and converted to `\ref{sec:mech2}` |

Two labels were added during revision (`sec:leakageprev`, `sec:imputation`) so
that new cross-references resolve symbolically rather than by hardcoded
number. The previously unused `sec:synthetic` label is now referenced from
the Conclusion.

## 3. Consistency of manuscript claims with generated results

Verified with `scratchpad/verify_claims.py`, which re-derives each headline
value from its source CSV in `reports/tables/` and confirms the exact string
appears in the manuscript.

| Claim | Expected | Source table | Result |
|---|---|---|---|
| External AUC, lower | 0.699 | `table_42_external_validation` | OK |
| External AUC, upper | 0.708 | `table_42_external_validation` | OK |
| Internal reference AUC | 0.997 | `table_42_external_validation` | OK |
| External cases | 20 | `table_42_external_validation` | OK |
| External cohort size | 94 | `table_42_external_validation` | OK |
| Laboratory multivariable lift | 0.026 | `table_43_bid_lift` | OK |
| Full-valid multivariable lift | 0.032 | `table_43_bid_lift` | OK |
| Low-cost multivariable lift | 0.105 | `table_43_bid_lift` | OK |
| Max sensitivity standardisation shift | 0.056 | `table_43_bid_standardisation` | OK |
| Effective cases after weighting | 38.2 | `table_43_bid_standardisation` | OK |
| Early-stage denominator | 21 | `table_21_spectrum_analysis` | OK |
| Serum creatinine binning cost | 0.266 | `table_31_binning_cost` | OK |
| Imputed cells | 339 | `table_32_imputation_audit` | OK |
| Containment match fraction | 1.000 | `table_24_provenance` | OK |

**14 / 14 headline values verified.** Script exit status 0.

Note on the newly stated `38.2` effective-case count: this value was already
present in `table_43_bid_standardisation.csv` from the original analysis run;
it was surfaced into the manuscript during this revision but not recomputed
or altered.

## 4. Tests and analysis scripts

| Suite | Command | Result |
|---|---|---|
| Full project test suite | `python -m pytest tests/` | **416 passed**, 0 failed, 0 skipped, 216 s |

The suite includes the guarantees most relevant to the revised claims:

- `tests/test_external_validation.py` (13) — the transportability probe's
  cohort integrity, temporal guard, physiological plausibility of extracted
  values, that recalibration cannot move the AUC, and that the internal
  vs. external gap is recorded.
- `tests/test_informativeness.py` (23) — the diagnostic protocol on
  constructed cases with known answers.
- `tests/test_report.py` (17) — that generated-report numbers match their
  source tables and that no unqualified screening-performance claim survives.
- `tests/test_external.py::TestProvenanceGate` — that no non-INDEPENDENT
  cohort can appear in an external-validation table.

No analysis script was re-run, because no numerical result was changed. The
manuscript's figures are unmodified copies of committed PDFs in
`reports/figures/`; their provenance is tabulated in `paper/README.md`.

## 5. Scope of modification

`git status` confirms changes confined to `paper/`:

- modified: `paper/ieee_paper.tex`, `paper/ieee_paper.pdf`
- added: `paper/RESPONSE_TO_REVIEWERS.md`, `paper/CHANGELOG_REVISION.md`,
  `paper/VERIFICATION_REPORT.md`

No file under `src/`, `scripts/`, `tests/`, `reports/`, `config/` or `data/`
was touched.

## 6. Issues that could not be resolved

1. **No adequately powered external cohort is available.** This is a data
   limitation, not a writing one. The revision constrains what is claimed
   from the 20-case probe; it cannot supply a better cohort. The most
   suitable candidate (a community-based series with 68% of cases at stages
   1–2) is access-restricted, and the manuscript says so.

2. **The early-stage subgroup remains underpowered at 21 cases.** The
   revision restricts the claims made from it to exploratory and
   hypothesis-generating; confirming the effect requires new data.

3. **Ten of the thirteen surveyed prior studies remain coded from an
   attributed comparison table rather than independently verified**, because
   the sources are paywalled. This predates the revision, is disclosed in the
   manuscript's Limitations, and is flagged per row in
   `data/literature/prior_work.csv`. It is the manuscript's softest evidential
   point and would be the first thing we would strengthen with institutional
   access.

4. **The author block remains a placeholder.** Name, affiliation, email and
   the repository URL in `CITATION.cff` must be completed before submission.
   These were deliberately left unfilled rather than guessed.

5. **14 underfull boxes** remain (see §1). Cosmetic; not addressed because
   the fixes would be purely presentational rewordings.
