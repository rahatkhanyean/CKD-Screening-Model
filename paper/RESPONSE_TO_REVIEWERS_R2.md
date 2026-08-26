# Response to reviewers — second revision

**Manuscript:** *Three Mechanisms Inflate Reported Performance on a Widely
Used Public Chronic Kidney Disease Benchmark*

**Base commit:** `47e72fe` · **Date:** 2026-08-25

---

We are grateful for a review that went to the clinical substance rather than
the presentation. One comment identified an error that had inverted a central
claim, and correcting it changed both what the paper can assert and what it
must retract. We report that first, because it is the most consequential
thing in this revision.

---

## 1. The diagnostic-incorporation taxonomy was wrong, in both directions

> *The current taxonomy may incorrectly treat blood urea and serum creatinine
> as diagnostic-criterion variables while treating albuminuria as an ordinary
> legitimate predictor.*

**The reviewer is right, and the error was worse than a mislabel.**

We verified the classification against the KDIGO 2024 guideline rather than
against intuition. CKD is defined by a GFR below 60 mL/min/1.73 m² **or** a
marker of kidney damage, of which albuminuria (ACR ≥ 30 mg/g) is the
first listed; the damage markers also include urine sediment abnormalities
and persistent haematuria. **Blood urea appears nowhere in the definition** —
neither limb, neither list.

The registry we shipped had it backwards:

| Variable | Was | Is now | Why |
|---|---|---|---|
| `al` (urine albumin) | `legitimate_pre_index` | `diagnostic_criterion_input` | The albuminuria limb of the KDIGO definition |
| `bu` (blood urea) | `incorporation_risk` | `correlated_renal_biomarker` | Not a KDIGO criterion; confounded by protein intake, catabolism, hydration, GI bleeding |
| `rbc`, `pc`, `pcc`, `ba` | mixed | `possible_criterion_sediment` | Sediment abnormalities and haematuria are damage markers; pyuria and bacteriuria indicate infection |
| `sc` | `incorporation_risk` | `diagnostic_criterion_input` | Unchanged in substance; renamed for precision |
| `sg`, `sod`, `pot` | mixed | `correlated_renal_biomarker` | Track renal function; not criteria |

**The consequence was that a claim in the paper was false.** The previous
manuscript said that removing "the variables that enter the diagnostic
criterion — serum creatinine and blood urea" left discrimination unchanged,
and concluded that incorporation is not what produces the ceiling. That test
never removed the albuminuria limb. The albuminuria limb stayed in every
model described as free of incorporation.

**Action.** The registry is corrected with the KDIGO source recorded per
variable, the configurations are derived from it, and we re-ran the full
nested design over **ten** registry-derived configurations. Three tests now
guard the correction, including one asserting that a configuration claiming
to remove incorporation retains no incorporation variable.

**The corrected test gives the same answer, and now it means something.**
Removing both KDIGO limbs *and* the sediment findings leaves ROC-AUC at
1.000. So the conclusion survives — but it is now supported by an experiment
that actually performed the removal.

**A caveat we now state explicitly.** The released `al` column is a
semiquantitative reagent-strip grade, not a measured ACR. Dipstick protein is
a crude, concentration-dependent proxy and is not interchangeable with a
quantified ratio, so its incorporation risk is real but weaker than a
measured ACR would carry. That is why it has its own restricted
configuration rather than being pooled with creatinine.

**And the correction revealed something the old taxonomy had hidden.** The
old registry counted 11 variables as "legitimate pre-index"; the corrected
one counts 5. On those five — age, hypertension, pedal oedema, random blood
glucose, white-cell count — discrimination falls to **0.930 (0.894–0.962)**,
and on a three-variable history-and-examination set to **0.881
(0.833–0.923)**. Neither is saturated. The previous manuscript reported
0.994 for "pre-index available only" and drew comfort from it. That number
was an artefact of misclassifying urinalysis and renal chemistry as
pre-index information.

---

## 2. Case composition: the claim did not survive testing

> *Determine whether the available data support any causal ranking of the
> mechanisms. If not, revise the central conclusion to…*

**Action.** We implemented the strongest case-composition analyses the data
permit, including the two that were missing: severity-balanced resampling and
a simulation varying the case-severity mix (new stage 26, `table_54`,
`table_55`).

Holding the fitted models fixed and re-weighting the out-of-fold predictions
to target severity distributions:

| Target mix | Early frac. | Low-cost AUC | Laboratory AUC | Full valid AUC |
|---|---|---|---|---|
| As sampled (benchmark) | 0.16 | 0.993 | 0.994 | 1.000 |
| Community screening series | 0.68 | 0.989 | 0.985 | 1.000 |
| Early-heavy | 0.80 | 0.987 | 0.984 | 1.000 |
| Advanced-heavy | 0.00 | 0.994 | 0.997 | 1.000 |
| Severity-balanced, no replacement | — | 0.991 | 0.990 | 1.000 |

**Re-weighting severity from 16% to 80% early-stage cases moves ROC-AUC by
less than 0.01.** Sensitivity is more responsive (0.969 → 0.923 and 0.879),
but discrimination is not. On this evidence the severity mix is not what
holds the benchmark at the ceiling, and the previous claim that case mix is
"the principal explanation" is not supported.

**We adopt the reviewer's wording verbatim**, in §V-D, the abstract and the
conclusion:

> "The persistence of near-ceiling performance across restricted feature sets
> is consistent with unusually separable cohort composition, although its
> contribution cannot be isolated from residual incorporation, proxy
> information, feature redundancy, and unknown recruitment mechanisms."

"Ruled out" is replaced throughout with the precise statement that the
selected variables representing those pathways were **not individually
necessary** for near-ceiling performance.

**Limits of the simulation, stated in the paper.** It holds the models fixed,
so it answers what we would have *measured* on a differently composed cohort,
not what a model trained on that cohort would achieve. Its support is thin:
the benchmark holds 9 stage-1 and 12 stage-2 cases, so an early-heavy target
reuses a handful of patients up to 16 times. Both the distinct-patient count
and the maximum reuse are reported for every scenario. It is a resampling
exercise, not prospective evidence.

---

## 3. Matching nulls: named, audited, and one of them corrected

> *Verify whether it actually preserves the relevant dependence structure.
> Report diagnostic comparisons between real and null-generated data.*

**We had never checked this.** Doing so found a defect.

The copula null mapped synthetic values back through `np.quantile` with
linear interpolation. That places synthetic values *between* the release's
discrete bin representatives, where no published interval can contain them —
depressing the null for a purely mechanical reason and flattering the
observed result. Mapping is now nearest-value, so every synthetic value is
one the column actually takes. **The null rose** (0.0563 → 0.0586), which is
the conservative direction.

**Diagnostics, now reported** (`table_53`, supplement Table S3):

| Quantity | Value |
|---|---|
| Variable pairs compared | 78 |
| Mean \|real − null\| Spearman | 0.031 |
| Largest single discrepancy | 0.168 (`su`~`bgr`) |
| Real correlation range | −0.73 to +0.87 |
| Synthetic values off observed support | 0 |

The null reproduces the cohort's shape closely while destroying record
identity, which is what makes exceeding it informative.

**Nulls are now named consistently everywhere.** A single authoritative
summary (`table_46_provenance_null_summary.csv`) records, per null, the
description, draws, mean, SD, max, exceedances and an exceedance bound
(0 of 200 in both cases, so *p* ≤ 0.005). **The whole-row permutation is no
longer presented as a competing null**: it is explained in the supplement as
a property of the estimator, since bipartite matching is invariant to
candidate row order. A test asserts it cannot reappear in the table of nulls.

The conservative conclusion is retained unchanged, and we still do not claim
exact raw-record identity or official provenance confirmation.

---

## 4. Whole-procedure bootstrap

**Action.** The procedure was already fully specified; we extended it. Cells
now include two restricted configurations — the conservative pre-index set
and the no-incorporation-or-consequences set — because after the taxonomy
correction the conservative set is the only configuration that is *not*
saturated, so it is the only one whose interval width carries information.
Checkpoint estimates at 100, 200, 500 and 1000 are emitted to
`table_56_bootstrap_checkpoints.csv`.

**We report the count honestly in the verification report rather than
claiming a target we did not reach.** Promotional wording ("honest
uncertainty") is replaced with "whole-procedure uncertainty" throughout.

---

## 5. Literature: verification attempted, fallback taken, claim narrowed

**We attempted the preferred approach and it failed on access.** Direct
verification of the two studies whose designs the manuscript relied on was
attempted on 2026-08-25: both publisher pages returned HTTP 403. The attempt
and its outcome are recorded per study in the extraction sheet
(`verification_attempt` column), so the audit trail shows this was tried, not
assumed.

**The claim was overstated and is now corrected.** The manuscript said "three
validate across the two releases … and one merges them". Only **one** study
is verifiable from its own source text. It now says so, attributes the other
two explicitly, and Table I is **generated** from the extraction sheet rather
than typed — which is how the "three" survived in the first place.

"Frequently report accuracy at or near 100%" is removed from the abstract.
No prevalence language remains; a test enforces this.

---

## 6. MIMIC: reduced, reframed, and given the metric that matters

**Action.** All three named overstatements are rewritten:

- "The binding constraint is discrimination of 0.70" → the passage now
  explains that recalibration leaving no prediction above 0.50 means 0.50 is
  no longer an appropriate operating threshold, and that no threshold can be
  chosen without a stated decision and a decision-curve analysis, neither of
  which this study provides.
- "Prevalence is the obvious culprit" → the reviewer's wording, naming case
  mix, measurement, missingness, setting and label definition alongside it.
- "It bounds the direction" → "The analysis provides exploratory evidence
  that near-perfect internal discrimination does not persist under this
  particular cross-setting transfer."

**We also removed an improper inference.** The paper said the bootstrap
intervals "exclude the internal estimate by a wide margin". No paired
comparison was performed, so that is not a test; the text now says so.

**PR-AUC is added, and it changes the picture.** Prevalence differs between
the cohorts (0.64 internally, 0.21 externally), so ROC-AUC is invariant to
exactly the shift a deployer would feel. External PR-AUC is **0.313–0.326
against a no-skill baseline of 0.213**, with intervals reaching down to 0.25
— a far smaller improvement over chance than ROC-AUC 0.70 suggests.

---

## 7. Absolute claims

Every term the reviewer listed was searched and each occurrence evaluated.
Applied: "no multivariable learning is required" → "single predictors already
provide unusually high discrimination"; "the benchmark cannot distinguish
methods" → "provides little practical headroom for distinguishing the
evaluated methods under this design"; "measure resubstitution" →
"non-independent evaluations on overlapping patients"; "learned nothing
transferable" → "whose performance need not transfer". A test class greps
both documents for nine retracted phrases, each recorded with the reason.

---

## 8. Submission readiness

Data-availability, code-availability, ethics and registration statements are
added, written from facts in the repository — licences, checksums, the
absence of preregistration, and the commit ordering that fixed the
model-selection rule before results were computed.

Items only the authors can supply — names, funding, competing interests,
contributions, and the archived repository DOI — are left as marked
`TODO(author)` placeholders and enumerated in
`paper/AUTHOR_ACTION_CHECKLIST.md`, outside the manuscript. A test fails if a
placeholder is present without being marked.

---

## What we have not resolved

1. **The manuscript is 15 pages**, not shorter. The synthetic-data section is
   reduced to one future-work paragraph and the full tables now live in the
   supplement, but the corrected taxonomy and the new severity analyses added
   material. The checklist names, in order, the sections that should go next
   if the target venue is stricter.
2. **A protocol-driven systematic review was not performed.** Ten of thirteen
   studies remain inaccessible.
3. **No adequately powered external cohort exists.** Unchanged, and not
   fixable by revision.
4. **The audit framework remains a proof of concept.**
