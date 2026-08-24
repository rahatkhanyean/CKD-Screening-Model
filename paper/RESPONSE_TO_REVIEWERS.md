# Response to Reviewers

**Manuscript:** *Three Mechanisms Inflate Reported Performance on a Widely Used
Public Chronic Kidney Disease Benchmark*

**Revision:** minor revision, submitted 2026-08-24

---

We thank the reviewer for a careful and constructive reading, and in
particular for identifying two places where the manuscript's language ran
ahead of what the evidence supports. Both comments were, in our view,
correct, and addressing them has made the paper more accurate rather than
merely more cautious. We have not weakened either experiment; we have
restated what each can and cannot establish.

All section numbers below refer to the revised manuscript. Every numerical
value in the paper continues to be generated from a committed table in
`reports/tables/`; no number was changed in this revision, and an automated
check confirms that all fourteen headline values still match their source
tables (see `VERIFICATION_REPORT.md`).

---

## Comment 1 — External-validation limitation

> *The MIMIC-IV external-validation cohort contains only 20 cases and
> represents a substantially different critical-care population. Revise the
> manuscript to emphasize that the observed external-performance difference
> cannot be attributed solely to internal optimism… Describe it as a stress
> test or transportability probe rather than definitive external validation.*

**Response.** We agree, and have adopted the reviewer's framing throughout.
The experiment is now consistently described as a **transportability probe**
or stress test, never as an external validation, and the manuscript now
enumerates explicitly the mechanisms that could produce the observed
difference rather than leaving the reader to infer them.

We would like to record that we consider the reviewer's characterisation
more accurate than our own. Our previous phrasing — "discrimination does not
transfer", "do not survive contact with different patients" — asserted a
general property when the evidence supports a conditional statement about one
cohort.

**Sections changed:** Abstract; Contributions (Section I-A); Section V-J
(retitled); Section VII (Discussion); Section VIII (Limitations); Section IX
(Conclusion); Index Terms.

**Revisions made:**

1. **Abstract.** The result is introduced as "a *transportability probe*
   rather than a definitive external validation", and followed immediately by:
   "With 20 cases in a markedly different clinical setting, this difference
   cannot be attributed to internal optimism alone: small-sample uncertainty,
   case mix, feature definition, missingness and genuine distribution shift
   all plausibly contribute."

2. **Contributions.** The item previously reading "We supply the **external
   validation** the benchmark's own literature lacks" now reads "We report a
   **transportability probe**…", and states that it is "a stress test rather
   than a definitive external validation".

3. **Section V-J retitled** from "External validation" to "A transportability
   probe on an independent cohort". A new opening paragraph states the three
   conditions a validation cohort would need to satisfy — adequate power, a
   population in which the model is intended to operate, and labelling by the
   same standard — and notes that this cohort satisfies none of them.

4. **The headline claim is now conditional.** "Discrimination does not
   transfer" has become "Discrimination falls sharply on the new cohort",
   followed by: "The inference this supports is a conditional one:
   performance measured near the ceiling on this benchmark is not, on this
   evidence, a reliable guide to performance on a different patient
   population. It does not by itself establish how much of the internal
   figure was optimism, and we quantify neither."

5. **A six-item enumeration** now replaces the single paragraph that
   previously discussed optimism versus population shift, covering
   small-sample uncertainty (with the 0.59–0.80 interval), case-mix
   differences, measurement and feature definition, missingness differences,
   clinical setting, and genuine population/distribution shift. Internal
   optimism is named as a seventh candidate for which the study offers only
   indirect support, and the passage closes: "Attributing the fall to any
   single mechanism, optimism included, would overstate what a 20-case
   comparison can support."

6. **The experiment's value is preserved explicitly.** A closing paragraph
   states that it remains the only evidence in the study drawn from unseen
   patients, the only such evidence available once the provenance gate has
   excluded the alternative, and that its direction is unambiguous and
   consistent across three model families.

7. **Limitations** now opens "the study has **no adequately powered external
   validation**", describes what it does have, and states that no performance
   claim for any population should be drawn from it.

8. **Discussion and Conclusion** were revised to match, including replacing
   "what remains does not transfer" with "what remains did not transfer to
   the one independent cohort we could test".

**Status:** we believe this comment is addressed. The claim now made is
strictly weaker than the evidence, which we regard as the correct direction
of error.

---

## Comment 2 — Early-stage CKD subgroup

> *The early-stage CKD conclusion is based on only 21 cases and arises from a
> post hoc subgroup analysis. Revise all related claims so they are strictly
> characterized as exploratory, hypothesis-generating, statistically
> underpowered, and requiring confirmation… Report the relevant denominator
> alongside the claim wherever practical.*

**Response.** We agree. The manuscript previously carried a single hedging
sentence in Section V-B; the qualification is now attached to every place the
subgroup is used, and the denominator (21 cases) is stated alongside each
claim.

**Sections changed:** Abstract; Section V-B; Fig. 1 caption; Section VI-C;
Section VIII (Limitations).

**Revisions made:**

1. **Abstract.** Now reads: "in an exploratory, post hoc and statistically
   underpowered subgroup of the 21 early-stage cases, sensitivity falls from
   0.969 to 0.905, a hypothesis-generating signal requiring confirmation in a
   larger prospectively defined cohort."

2. **Section V-B** gains a dedicated paragraph in bold — "These early-stage
   figures are exploratory and must not be read as estimates" — stating that
   the subgroup contains 21 cases, that the analysis was defined post hoc
   after the ceiling was observed, that no formal comparison is made or
   implied, and that a difference of this size "would not survive a demand
   for statistical significance". It then gives the three reasons the numbers
   are reported at all, and closes: "We draw no causal conclusion and make no
   claim that the effect size generalises beyond this sample."

3. **Fig. 1 caption** softened from "is systematically optimistic relative to
   the early-stage patients" to "appears optimistic…; the early subgroup
   contains only 21 cases, so these comparisons are exploratory and
   hypothesis-generating rather than estimates."

4. **Section VI-C.** The early-stage subgroup AUC of 0.983, used to explain
   why standardised AUC moves so little, now carries "on 21 cases and
   therefore imprecise", and the standardised quantities are explicitly
   reported as exploratory with their effective case count (38.2) stated.

5. **Limitations** gains a dedicated paragraph stating that the subgroup
   analysis is exploratory and underpowered, that its role is to illustrate a
   case-mix imbalance established independently by arithmetic (107 of 128
   cases at stages 3–5) rather than to estimate early-stage performance, and
   that confirmation requires a larger, prospectively defined cohort.

**Status:** we believe this comment is addressed. We note that the case-mix
imbalance itself — the mechanism the paper relies on — is established by
counting, not by the subgroup analysis, so the paper's argument does not rest
on the underpowered comparison.

---

## Additional revision: synthetic-data evaluation

At the editor's request we have added **Section VII-A, "Extension to
synthetic tabular and synthetic patient data"**, connecting the paper's
safeguards to synthetic-data evaluation pipelines.

We have been careful to distinguish what already exists in the released
repository from what does not. **Seven** of the safeguards discussed are
implemented here in a form adapted to real-data provenance: cryptographic
dataset fingerprints, exact-row and near-duplicate detection, entity-level
overlap checks, feature–target leakage scans, post-outcome variable
declaration, fold-local preprocessing, provenance manifests, and a
fail-closed gate. **Two** are specific to synthetic data and are explicitly
identified as *not implemented*: membership-inference/memorisation checks and
nearest-neighbour distance-to-closest-record diagnostics. The section states
plainly that "the present work provides the provenance half of the problem,
not the memorisation half."

The section is framed as a broader implication, not as an experiment
performed. It closes with a caution we think important: the most consequential
mechanism in this study was case mix, which no hash, fingerprint or duplicate
scan would detect — so provenance gating and informativeness diagnostics are
complementary rather than alternatives.

---

## Terminology audit

Following the reviewer's comment 8, we audited the manuscript for
over-strong usage of "external validation", "generalisation",
"transportability" and "early-stage". Results:

| Term | Before | After |
|---|---|---|
| "external validation" describing our own experiment | 4 uses | 0 (replaced by "transportability probe"/"stress test"; the term is retained only where it describes what the *literature* claims, or what the study lacks) |
| "does not transfer" / "do not survive contact" | 3 uses | 0 (replaced by conditional phrasing) |
| "systematically optimistic" (Fig. 1) | 1 use | 0 |
| early-stage claims without a stated denominator | 4 | 0 |
| hardcoded section cross-references | 1 | 0 (converted to `\ref`) |

---

## What we have not claimed to resolve

We do not claim that the transportability limitation is *eliminated*. It is a
property of the available data, not of the writing: no adequately powered,
setting-matched external cohort is accessible to us. The revision makes the
limitation explicit and prevents over-reading; it does not remove it. The
most suitable candidate cohort — a community-based series with 68% of cases
at stages 1–2 — remains access-restricted, and we say so.

Similarly, the early-stage analysis remains underpowered after revision. We
have constrained what is claimed from it rather than strengthened the
evidence, which would require new data.
