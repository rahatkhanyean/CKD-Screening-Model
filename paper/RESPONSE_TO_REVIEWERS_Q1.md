# Response to Reviewers — Q1 revision

**Manuscript:** *Three Mechanisms Inflate Reported Performance on a Widely
Used Public Chronic Kidney Disease Benchmark*

**Revision date:** 2026-08-24 · **Base commit:** `ab32757` ·
**Revision commits:** `b65cfe2` and following

---

We are grateful for a review that went well beyond prose. Several comments
identified genuine statistical defects, not presentational ones, and
addressing them changed results and wording rather than wording alone. Two
points are worth stating at the outset because they set the tone of this
response.

First, **we found and corrected an invalid analysis of our own while acting
on the provenance comment**: the null distribution we had implemented for the
record-matching statistic was degenerate. It is now replaced and the failure
documented, because a reader is entitled to know which nulls do not work.

Second, **we have not claimed to resolve what we could not**. Two requested
analyses were not feasible with the data and access available; both are
reported as fallbacks the brief anticipated, with the affected claims
narrowed accordingly rather than defended.

---

## 1. Effective sample size and the repeated cross-validation

> *Treat the 200 unique patients — not the number of repeated predictions —
> as the effective observational sample. Replace wording such as "25
> independent outer test sets."*

**This was a real error and we thank the reviewer for catching it.** The 25
outer test folds are five repartitions of the same 200 patients; each patient
is evaluated once per repeat. They are not independent test sets, and
describing them as such misstated the evidence.

**Action.** Methods (§IV-C) now open by stating that the effective sample is
200 unique patients, that the design produces *five out-of-fold evaluations
per patient*, that repeated evaluations do not increase effective sample
size, and that per-patient probabilities are averaged before any metric is
computed. The "108,000 predictions" figure is retained only as an artefact
size and is explicitly labelled as repeated evaluations of 200 patients, in
both Methods and the Reproducibility statement.

**Audit of the implementation.** We verified — and the test suite enforces —
that preprocessing, imputation, scaling, hyper-parameter selection,
calibration and threshold selection are all fold-contained, and that
duplicated patients in resampled data remain within a fold.

**Remaining limitation.** None; this is a correction, not a constraint.

---

## 2. Whole-procedure bootstrap: specification and resample count

> *Use more than 200 resamples, preferably 1,000. If prohibitive, run a
> justified convergence analysis and report the limitation.*

**Action.** The procedure is now fully specified in §V-G: sampling unit is
the patient; stratified on outcome; folds formed over patients within each
resample; hyper-parameter selection and calibration re-run inside every
resample; percentile intervals; seeds derived from the master seed and
resample index; failed resamples counted (none failed).

**On the count — we did not achieve 1,000.** A 1,000-resample run was
launched and abandoned after it became clear it would require several hours
on the available hardware while competing with the other analyses in this
revision. Rather than report a number we did not compute, we added a
convergence analysis (new stage 25, `table_51`): drawing increasing subsets
of the completed resamples, the lower interval endpoint for the low-cost SVM
moves from 0.965 at 25 resamples to 0.960 at 200, with the endpoint's
sampling standard deviation falling to 0.003 by 175 — an order of magnitude
below the interval width of 0.038.

**Remaining limitation, stated in the manuscript.** The interval is stable at
the count used; a larger run would tighten it. This is recorded as a
computational limitation, not a resolved question.

---

## 3. Provenance analysis: extensive sensitivity required

> *The patient-overlap result is the most important contribution and must be
> subjected to extensive sensitivity analysis… Do not use "exactly identical
> patients" unless exact raw values establish this.*

**Action.** New module `provenance_sensitivity.py` and stage 21. Results
(§V-E, Table VI, Fig. 4; full detail in the supplement):

| Condition | Match | Unique pins |
|---|---|---|
| Primary (13 variables, outcome used) | 1.000 | 187 |
| **Outcome excluded** | 1.000 | 182 |
| Imputation-affected variables removed | 1.000 | 131 |
| Leave-one-variable-out (range) | 1.000 | 133–187 |
| Tolerance 0 → 10⁻² | 1.000 | 179–187 |
| Negative control (synthetic) | 0.005 | 1 |

**The null was wrong and is now corrected.** We implemented a whole-row
permutation null and observed a match fraction of 1.000 with *zero variance*.
The reason is structural: maximum bipartite matching depends only on the
multiset of candidate records, not their order, so row permutation leaves the
statistic unchanged by construction. That null is invalid. We replaced it
with a **copula null** preserving marginals and correlations while destroying
record identity, which yields 0.056 ± 0.014 over 200 draws (max 0.105) against an observed
1.000 — roughly 67 standard deviations. The degenerate check is retained in
code so the claim is verifiable.

**A caution about the statistic itself**, now in the manuscript: match
fraction *saturates*. A four-variable subset also returns 1.000 — but with
zero unique pins and up to 19 partners per record. Uniqueness, not match
fraction, carries the argument.

**Language weakened as requested.** We no longer write "record-for-record."
The claim is now: *"the data provide strong record-level evidence that the
2020 release is a discretised subset or re-release of patients represented in
the 2015 release."* We explicitly do not claim exact raw-value identity,
which discretisation makes untestable, and we state that the repository
maintainers **have not been contacted**, so nothing is presented as an
officially confirmed provenance correction.

**Also added.** A reproducibly sampled 12-pair matched audit table
(`table_45`) that a reader can regenerate and inspect.

---

## 4. Case mix: do not claim it is definitively the largest mechanism

> *Avoid claiming case mix is definitively the largest mechanism unless there
> is a valid quantitative decomposition.*

**Action.** New §V-D uses the feature registry to remove each information
pathway in turn and re-run the nested design:

| Feature set | k | ROC-AUC (95% CI) |
|---|---|---|
| Full valid (reference) | 25 | 1.000 (1.000–1.000) |
| Minus consequences of advanced disease | 21 | 0.998 (0.995–1.000) |
| Minus diagnostic-criterion inputs | 23 | 1.000 (1.000–1.000) |
| Minus both | 19 | 0.991 (0.975–1.000) |
| Pre-index available only | 11 | 0.994 (0.986–0.999) |

This rules out two candidate explanations — incorporation is *not* what
produces the ceiling — but the sets are heavily redundant, so it is not a
clean decomposition. We therefore adopt the reviewer's suggested wording
verbatim: **"case composition appears to be the principal explanation for
valid models reaching the ceiling, but its contribution cannot be cleanly
separated from diagnostic incorporation and feature redundancy in this
dataset."**

**The Conclusion had not been brought into line, and we found it only on a
final read of the built PDF.** It still said case mix "is the larger effect"
--- contradicting the section above it. It now carries the same qualified
statement, and a test greps both documents for this and six other retracted
phrases so that a later edit cannot quietly restore one.

---

## 5. Do not describe case-mix separability as "no predictive ability"

**Action.** That phrasing is removed throughout. The manuscript now states
explicitly (§V-D): *"Reaching 0.99 on these variables is real internal
discrimination, not an artefact and not an absence of predictive ability.
What the case-mix analysis questions is its clinical difficulty, relevance
and transportability."* The abstract's conclusion adopts the reviewer's
suggested framing about invalid information pathways, non-independent
evaluation and separable case composition.

---

## 6. Early-stage subgroup: exploratory, underpowered, denominator stated

**Action.** Subgroup metrics are now reported with stratified bootstrap
intervals (new `table_48`, Table IV), which materially changed how the result
reads:

| Subgroup | Cases | ROC-AUC (95% CI) | Sensitivity (95% CI) |
|---|---|---|---|
| All patients | 128 | 0.993 (0.983–1.000) | 0.969 (0.937–0.992) |
| Early, s1–s2 | 21 | 0.983 (0.954–1.000) | 0.905 (0.762–1.000) |

**The apparent decline is not statistically resolved** — the intervals
overlap across most of their range — and the manuscript now says so in §V-C,
in the figure caption, in Limitations and in the abstract. The 21-case
denominator accompanies every mention, including the standardisation
discussion in §VI-C, whose weights concentrate on that stratum (effective
case count 38.2).

We also note that the case-mix argument does not depend on this subgroup: the
imbalance (107 of 128 cases at s3–s5) is established by counting.

---

## 7. Transportability probe

**Action.** Retained and reframed as an exploratory, clinically mismatched
stress test throughout — never as external validation. §V-J states the three
conditions a validation cohort would need and that this one meets none;
enumerates six mechanisms that could produce the observed difference, with
internal optimism named as a seventh the design cannot isolate; and states
that no performance claim for any population should be drawn from it. Cohort
construction, label definition and timing, the temporal guard (8,754 values
dropped), unit conversions and exclusions are documented, with 13 dedicated
tests including physiological-plausibility checks that would catch a silent
unit error.

---

## 8. Literature review

> *If a full systematic review is not feasible, explicitly rename the section
> "targeted illustrative review" and substantially narrow the claims.*

**We took the fallback, because the review is not feasible as specified.**
Ten of the thirteen studies are paywalled and could not be verified from
source text; a systematic review additionally requires protocol-driven
multi-database searching and double screening that we did not perform.

**Action.** §II is renamed *"Related Work: A Targeted Illustrative Review"*
and opens by disclosing the sampling method and its limits. The manuscript
now states that **the ten unverified studies support no quantitative claim**;
Table I labels every row *verified* or *attributed*; and abstract and
introduction claims are narrowed from prevalence ("9 of 13 studies use…") to
existence ("such results exist in the published record"). The internal label
`todo_hand_check` no longer appears in the manuscript, and a test enforces
that.

**Remaining limitation.** Substantial and stated: we cannot say how common
these practices are.

---

## 9. Benchmark diagnostic must not be presented as validated

**Action.** Renamed *"A Proof-of-Concept Pre-Modelling Audit Framework"*
(§VI). The section now opens by stating it is not a validated instrument,
that multi-dataset evaluation with known ground truth has **not** been
performed, and that until it is, no threshold should be treated as a
criterion. Candidate flagging values are moved to the supplement. The
negative standardised-AUC result is reported prominently, with its
dataset-specific explanation and an explicit disclaimer against reading it as
a general property of AUC.

**Also corrected.** The single-feature baseline is now selected *inside*
training folds (`nested_best_single_auc`, `table_50`). The lift values are
unchanged to four decimal places, because selection is completely stable — a
single column is chosen in 100% of folds — so selection optimism was a real
concern in principle that turns out to be empty here. We report it either
way.

**Remaining limitation.** Multi-dataset validation of the framework is not
done. This is the main reason the contribution is labelled proof-of-concept.

---

## 10. Leakage taxonomy and feature registry

**Action.** `data/feature_registry.csv` records 18 fields for each of 29
variables. `src/ckd/features/registry.py` derives the prohibited sets and
configuration membership from it, and `tests/test_registry.py` (18 tests)
asserts the derivation reproduces the configurations used in the reference
analysis **exactly**, so a classification cannot drift from the configuration
depending on it. A publication-quality taxonomy table (Table II) is
*generated* from the registry, not typed. The taxonomy separates direct
outcome copies, deterministic derivatives, post-diagnosis variables,
incorporation risk, consequences of advanced disease, legitimate pre-index
predictors and uncertain provenance.

---

## 11. Intended use

**Action.** New §III-A specifies population, setting, index time, information
at index, outcome, task type and supported decision — and records that
**three of these are unidentifiable** from the release: index time,
per-patient availability, and chronicity. The manuscript concludes that the
dataset cannot support a uniquely defined intended use and reframes itself
explicitly as a methodological benchmark audit rather than a
prediction-model study.

---

## 12. Reproducibility engineering

**Action.** `reproduce.py` (portable, verified on the reference environment)
and a `Makefile` (POSIX) run: hash validation → preprocessing → analyses →
tests → table regeneration → manuscript and supplement builds → verification
that manuscript numbers match generated results.
`tests/test_manuscript_consistency.py` (25 tests) detects LaTeX numbers
differing from generated outputs, missing figures or tables, non-independent
datasets in external-validation tables, prohibited variables in valid
configurations, placeholder author information, unresolved TODO markers,
broken citations, dangling references and changed dataset hashes.

**A defect the first version of this pipeline had, and how it was found.** The
LaTeX build reads `paper/figures/`, while the analysis stages write
`reports/figures/`. Nothing refreshed the former: a re-run could regenerate a
figure and then build the manuscript from the previous copy of it. We found
this only by inspecting the built PDF page by page and noticing a figure that
did not match its regenerated source. `reproduce.py` now syncs figures before
either build and fails if one is missing, and a new test extracts text from
every included figure so that an embedded internal label cannot survive
again. We report this because a reproducibility claim that has never been
tested end to end is not yet a reproducibility claim.

**Determinism, checked file-by-file.** Running the cheap path end to end and
diffing every regenerated output against the committed copy, all 27
regenerated tables and processed datasets are byte-identical; the only
changed line in the whole set is a recorded absolute source path.

---

## What we have not resolved

1. **No adequately powered, setting-matched external cohort.** A data
   limitation the writing cannot fix. The most suitable candidate is
   access-restricted.
2. **1,000 whole-procedure resamples.** Not achieved; 200 with a convergence
   analysis is what we report.
3. **Systematic literature review.** Not feasible; the section is renamed and
   its claims narrowed.
4. **Multi-dataset validation of the audit framework.** Not performed; the
   contribution is labelled proof-of-concept.
5. **Author information.** The author block remains a marked placeholder,
   which only the authors can complete.
