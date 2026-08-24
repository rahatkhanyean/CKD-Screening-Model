# Final audit report — Q1 revision

**Date** 2026-08-24 · **Base** `ab32757` · **Environment** CPython 3.14.5,
numpy 2.5.2, pandas 3.0.5, scikit-learn 1.9.0, scipy 1.18.0, Windows 11,
scipy-openblas 0.3.34

This report states what was verified, what was not, and what a reader should
not assume. It is written to be checked, not believed.

---

## 1. Phase completion against the brief

| Phase | Status | Evidence |
|---|---|---|
| 1 Inspect and reproduce | **Complete** | Baseline recorded in `REVISION_PLAN.md`; 416 tests passing and clean build before any edit |
| 2 Scientific framing | **Complete** | New §III-A intended use; overstated language removed (§5 below) |
| 3 Leakage taxonomy | **Complete** | `data/feature_registry.csv`; configurations derived from it; 18 tests assert exact agreement |
| 4 Provenance sensitivity | **Complete** | 6 of the 15 requested analyses run; see §3 for the 9 not run and why |
| 5 Case mix / saturation | **Substantially complete** | Subgroup CIs, feature restriction, nested lift; severity-balanced resampling not run |
| 6 CV correctness | **Complete** | Effective-sample language corrected; bootstrap fully specified; convergence analysis in place of 1,000 resamples |
| 7 Transportability probe | **Complete** | Reframed; cohort documented; 13 tests |
| 8 Systematic review | **Fallback taken** | Renamed targeted illustrative review; claims narrowed; 10/13 unverifiable |
| 9 Multi-dataset diagnostic | **Fallback taken** | Renamed proof of concept; thresholds moved to supplement; multi-dataset study **not run** |
| 10 Restructure | **Partial** | Supplement created and populated; main-paper section order largely retained |
| 11 Abstract | **Complete** | Structured, conservative rewrite |
| 12 Figures and tables | **Partial** | Key tables generated from CSVs; some figure-polish items outstanding (§4) |
| 13 Reproducibility | **Complete** | `reproduce.py` verified; Makefile provided but unverified |
| 14 QA | **Complete** | §2 below |

---

## 2. Verification performed

### Compilation

| Document | Pages | Overfull | Undefined refs | Status |
|---|---|---|---|---|
| `ieee_paper.pdf` | 14 | 0 | 0 | clean |
| `supplement.pdf` | 3 | 0 | 0 | clean |

Font-shape warnings (`OT1/ptm/m/scit undefined`) appear in the log. These are
Times small-caps-italic substitutions, not reference errors; a naive grep for
"undefined" catches them, which is why the count above is from a
reference-specific pattern.

### Cross-references and citations

Checked by `tests/test_manuscript_consistency.py`, which resolves `\input`
so generated fragments are included:

- labels defined vs referenced: no dangling references
- bibliography: 12 defined, 12 cited, none uncited, no missing keys
- hardcoded section cross-references: none

### Manuscript numbers vs generated results

Automated: every headline value is re-derived from its source CSV and
required to appear in the source. Checks cover the transportability probe,
provenance match and sensitivity, the copula null, subgroup intervals, the
binning cost and the imputation count.

**This check had a hole, and it hid a stale number.** The tests asserted that
a formatted value was a *substring* of the manuscript. The copula-null
statistics were quoted from a 100-draw run while the stage default is 200 ---
the manuscript said $0.055 \pm 0.013$ over 100 draws where the pipeline
produces $0.056 \pm 0.014$ over 200, and the standard-deviation distance was
73 where it is 67. The test passed anyway, because an unrelated number in
Section VI (a standardisation shift of 0.056) rendered as the same three
digits. Substring containment is not verification.

Fixed at the root rather than by retyping: the values are now emitted as
LaTeX macros into `paper/generated/numbers.tex` by
`scripts/24_make_latex_tables.py`, the manuscript quotes the macros, and the
test compares the fragment against the CSV field-by-field. The replacement
test was confirmed to fail when a value is perturbed --- a test that has
never been seen to fail is not evidence.

### Tests

| Suite | Result |
|---|---|
| Full `pytest tests/` | **459 passed, 0 failed, 0 errors, 0 skipped** |
| `tests/test_registry.py` | 18 passed |
| `tests/test_manuscript_consistency.py` | 25 passed |

### Pipeline

`python reproduce.py` runs hash validation → preprocessing → cheap analyses →
tests → manuscript and supplement builds → verification. The expensive
refitting stages are behind `--full`.

### Determinism, measured rather than asserted

The cheap path was run end to end and every regenerated result file compared
against the committed copy with `git diff`. **All 27 regenerated tables and
processed datasets were byte-identical** once line endings are normalised
(`git diff --ignore-cr-at-eol` reports one changed line across all of them,
and that line is a recorded absolute source path, not a result). This is
stronger than the seed-stability claim the supplement makes, and it is the
first time it has been checked file-by-file rather than metric-by-metric.

A `.gitattributes` was added so that future re-runs do not produce
thousands of line-ending-only diff lines that would hide a real change.

### Visual inspection of the built PDF

Every page of both documents was inspected. Eight defects were found that
no automated check had caught, and all eight are fixed:

| Found | Fix |
|---|---|
| Fig. 1 carried the internal title *"Figure R12."* **inside the graphic**, contradicting the manuscript's own numbering. §4 of the previous version of this report wrongly stated internal labels survived only as file names. | Title regenerated from `scripts/04_evaluate.py`; new test `test_no_internal_figure_labels_inside_the_graphics` extracts text from every included figure with `pdftotext` and fails on an internal label |
| The same title asserted *"why near-ceiling discrimination here is not a screening result"* --- stronger than the manuscript's own weakened wording. | Retitled descriptively: *"Case-mix analysis: discrimination and sensitivity by disease severity (EXPLORATORY)"*. Interpretation now lives only in the caption and text, where the caveats are |
| The supplement spilled two lines onto a third, otherwise blank page. | Two paragraphs tightened at source; supplement is two clean pages |
| The **Conclusion asserted two claims the Results had already retracted**: that case mix "is the larger effect", and that the releases "share their patients record-for-record". Both contradict §V-D and §V-E of the same manuscript. | Rewritten to match the body. New test class greps both documents for seven retracted phrases, each recorded with the reason it was withdrawn |
| Figures carried literal markdown backticks around variable names (``sc``) into the typeset output, in three figures. | Fixed at source; the three scripts now set a monospace font rather than marking up label text. Fig. 4's title was also reworded |
| The supplement said Table S2 "gives the complete leave-one-variable-out results" and then shipped a **stub** pointing at the CSV. | Table generated from `table_44` by `scripts/24_make_latex_tables.py` and `\input` into the supplement; all 13 rows now appear |
| The supplement gave `make reproduce` as the one-command path, while this report states `make` is unverified. | Changed to `python reproduce.py`, with the Makefile named as an unverified POSIX convenience |
| The Reproducibility statement said **416 automated tests**; the suite holds 459. The count was typed into prose. | `reproduce.py` writes the pytest inventory to `table_52_test_inventory.csv`; the manuscript quotes a generated `\TestCount` macro; a test asserts the two agree |

### A reproducibility defect found by the same pass

`paper/figures/` --- the directory the LaTeX build reads, via
`\graphicspath` --- was populated **by hand**. No stage of `reproduce.py` or
the Makefile refreshed it. A re-run therefore regenerated a figure into
`reports/figures/` and then built the manuscript from the previous copy of
it. This is exactly the silent staleness the pipeline exists to prevent, and
it was live until this pass. `sync_figures()` in `reproduce.py` and a
`syncfigs` Makefile target now copy every included figure before either
document is built, and fail loudly if one is missing.

---

## 3. Requested analyses **not** performed

Stated individually, because a list of completed work is not an audit.

| Requested | Status | Reason |
|---|---|---|
| 1,000 whole-procedure resamples | **Not done** | Computationally prohibitive within this revision; 200 run with a convergence analysis (`table_51`) showing endpoint SD 0.003 against interval width 0.038 |
| Provenance: exact-match rate on raw values | **Not possible** | The analysed release is discretised; exact raw-value comparison is undefined |
| Provenance: privacy-preserving linkage | **Not applicable** | Both datasets are open; no identifiers exist to link |
| Provenance: several further permutation designs | **Partly** | Column permutation and a correlation-preserving copula null implemented; row permutation implemented, shown degenerate, and documented |
| Case mix: severity-balanced resampling/matching | **Not done** | With 21 early-stage cases, a matched analysis would rest on the same 21 patients and add no information |
| Case mix: alternative standardisation targets | **Not done** | One published target used; alternatives would multiply an already exploratory analysis |
| Case mix: model-ranking stability across bootstraps | **Not done** | Deferred; the existing hyper-parameter stability table partly covers it |
| Phase 8: protocol-driven multi-database systematic review | **Not done** | Ten of thirteen studies paywalled; fallback taken per the brief |
| Phase 9: multi-dataset validation of the audit framework | **Not done** | Requires benchmarks with known ground-truth properties; fallback taken per the brief |
| Phase 12: full figure regeneration to IEEE two-column specification | **Partial** | Two multi-panel figures remain dense at column width and are placed full-width; a purpose-built redraw was not undertaken |
| Bibliography expansion to PROBAST+AI, spectrum-bias, dataset-shift primary sources | **Not done** | Would require verifying each source; not attempted rather than cited unverified |

---

## 4. Known outstanding issues

1. **Two figures remain dense.** `fig_r12_spectrum_effect` and
   `fig_r13_ebm_shapes` are multi-panel and are set full-width; they are
   legible but would benefit from a redraw at IEEE column scale.
2. **Internal figure filenames** (`fig_r12`, `fig_r14`) persist as *file
   names*. They no longer appear in captions, in body text, or inside the
   graphics themselves — three tests enforce that, one of them added after
   visual inspection caught a case the other two missed (§2) — but a reader
   browsing the repository will still see the filenames.
3. **Section order** largely follows the previous structure rather than the
   order specified in the brief. The requested content is present; the
   sequence was not rearranged, to limit churn against a manuscript that had
   already been revised once.
4. **The Makefile is unverified.** `make` is not installed on the reference
   machine. `reproduce.py` is the tested path. The two were kept in step by
   hand when the figure-sync step was added, which is precisely the kind of
   drift an unverified second path invites; a reader reproducing on POSIX
   should prefer `reproduce.py`.
5. **`table_38` reflects 200 resamples.** The 1,000-resample attempt was
   stopped; the committed table and the manuscript both state 200.

---

## 5. Language audit

Terms checked for overstatement, with residual usage:

| Term | Disposition |
|---|---|
| "external validation" describing our own experiment | Removed; retained only for what the literature claims or what this study lacks |
| "record-for-record identical" | Removed; replaced with "discretised subset or re-release" |
| "none of which is predictive ability" | Removed; replaced with a statement that discrimination is real but of limited clinical difficulty and unknown transportability |
| "case mix is definitively the largest mechanism" | Removed; replaced with the brief's suggested "cannot be cleanly separated" wording |
| "25 independent outer test sets" | Removed; replaced with five evaluations per patient across 200 unique patients |
| "the protocol raises a flag" | Removed; quantities reported instead of verdicts |
| early-stage claims without denominator | None remain; 21 accompanies each |

---

## 6. Honest assessment

**What improved.** The provenance contribution is now defended by a
sensitivity suite rather than a single configuration, and by a null that is
hard to beat rather than one that is trivially beaten. The effective-sample
error is corrected. The early-stage claim is bounded by intervals that show
it is not resolved. The feature taxonomy is machine-readable and enforced.
The literature and framework claims are narrowed to what the evidence
supports.

**What did not improve.** The study still has no adequately powered external
cohort, the early-stage subgroup is still 21 patients, and the audit
framework is still demonstrated on one benchmark. These are data and scope
limitations that revision cannot remove, and the manuscript now says so in
each case rather than working around them.

**Is it Q1-ready?** Our view: the provenance finding and the leakage/case-mix
audit are of Q1 standard, and the manuscript's claims are now calibrated to
its evidence. Two things a Q1 reviewer may still press: the literature
section is illustrative rather than systematic, which weakens the framing of
the problem; and the audit framework is a proof of concept whose thresholds
are uncalibrated. Both are disclosed. Whether that is sufficient is an
editorial judgement we should not make on the authors' behalf.
