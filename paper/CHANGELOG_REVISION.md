# Change log — minor revision (2026-08-24)

Every materially revised passage in `paper/ieee_paper.tex`, in document
order. Line numbers refer to the revised file. **No numerical result was
changed**; all edits are to framing, qualification and cross-referencing.

Legend — **C1**: reviewer comment 1 (external validation);
**C2**: reviewer comment 2 (early-stage subgroup);
**ED**: editor request (synthetic data); **HK**: housekeeping.

---

| # | Location | Tag | Change |
|---|---|---|---|
| 1 | Abstract, case-mix sentence | C2 | "restricted to early-stage disease, sensitivity falls from 0.969 to 0.905" → "in an exploratory, post hoc and statistically underpowered subgroup of **the 21 early-stage cases**… a hypothesis-generating signal requiring confirmation in a larger prospectively defined cohort". |
| 2 | Abstract, transfer sentence | C1 | Result reframed as "a *transportability probe* rather than a definitive external validation"; "so the near-perfect internal figures do not survive contact with different patients" replaced by an explicit statement that the difference "cannot be attributed to internal optimism alone", listing five contributing mechanisms. |
| 3 | Index Terms | C1 | Added `transportability`. |
| 4 | §I-A Contributions, item 5 | C1 | "We supply the **external validation** the benchmark's own literature lacks" → "We report a **transportability probe**…", with "a stress test rather than a definitive external validation" and a note that the several possible mechanisms are made explicit. |
| 5 | §IV-B heading | HK | Added `\label{sec:leakageprev}` so §VII-A can cross-reference it instead of hardcoding "Section IV-B". |
| 6 | §V-B, case-mix paragraph | C2 | "$n=21$" removed from the inline parenthesis and promoted to a dedicated bolded paragraph (next row). |
| 7 | §V-B, **new paragraph** | C2 | "**These early-stage figures are exploratory and must not be read as estimates.**" States the 21-case denominator, post hoc specification, absence of formal inferential comparison, and that a difference of this size "would not survive a demand for statistical significance"; gives three reasons for reporting them; closes "We draw no causal conclusion and make no claim that the effect size generalises beyond this sample." |
| 8 | Fig. 1 caption | C2 | "is **systematically optimistic** relative to the early-stage patients" → "**appears** optimistic…; the early subgroup contains only 21 cases, so these comparisons are exploratory and hypothesis-generating rather than estimates." |
| 9 | §V-E heading | HK | Added `\label{sec:imputation}` for cross-reference from the missingness bullet in §V-J. |
| 10 | §V-J title | C1 | "External validation" → "A transportability probe on an independent cohort". |
| 11 | §V-J, **new opening paragraph** | C1 | States the three conditions a validation cohort would need (adequate power, intended-use population, same labelling standard) and that this cohort satisfies none; distinguishes what the probe *can* test. |
| 12 | §V-J, headline claim | C1 | "**Discrimination does not transfer.**" → "**Discrimination falls sharply on the new cohort.**" Closing sentence replaced with a conditional inference: "…is not, on this evidence, a reliable guide to performance on a different patient population. It does not by itself establish how much of the internal figure was optimism, and we quantify neither." |
| 13 | §V-J, optimism paragraph | C1 | Single paragraph replaced by "**The difference cannot be attributed to internal optimism alone.**" plus a six-item `itemize`: small-sample uncertainty (with interval), case mix, measurement/feature definition, missingness differences (cross-referencing §V-E), clinical setting, distribution shift. Internal optimism named as a seventh candidate with only indirect support. |
| 14 | §V-J, **new closing paragraph** | C1 | Preserves the experiment's value: only evidence from unseen patients, only such evidence the gate permits, direction unambiguous across three model families. |
| 15 | §VI-C, early-stage explanation | C2 | Subgroup AUC 0.983 now qualified "on 21 cases and therefore imprecise"; added that standardised quantities inherit that imprecision, are reported as exploratory, carry an effective case count of 38.2, and support only the qualitative contrast between the two metrics. |
| 16 | §VII, external paragraph | C1 | "converts that argument from a prediction into an observation" → "lends that argument empirical support, within limits we state rather than minimise"; lists the seven candidate mechanisms; adds that establishing *why* requires an adequately powered cohort from the intended setting. |
| 17 | §VII-A **new subsection** | ED | "Extension to synthetic tabular and synthetic patient data" (~5 paragraphs). Distinguishes **seven implemented** safeguards from **two not implemented** (membership inference; nearest-neighbour / distance-to-closest-record). Frames the analogy as an extension, not an experiment. Closes with the caution that case mix — the study's largest mechanism — is undetectable by any hash or duplicate scan. |
| 18 | §VIII Limitations, opening | C1 | "**external validation rests on one small, confounded cohort**" → "the study has **no adequately powered external validation**"; enumerates the seven mechanisms it cannot apportion among; adds "No performance claim for any population should be drawn from it." |
| 19 | §VIII Limitations, **new paragraph** | C2 | "**The early-stage subgroup analysis is exploratory and underpowered.**" 21 cases, post hoc, no inferential comparison; states its role is to illustrate an imbalance established by arithmetic (107/128 at s3–s5); requires confirmation in a larger prospectively defined cohort; extends the caution to §VI's standardised quantities. |
| 20 | §IX Conclusion, transfer paragraph | C1 | Reframed as a transportability probe; "indicates what those mechanisms *may* cost"; lists contributing mechanisms; retains "It bounds the direction rather than the magnitude". |
| 21 | §IX Conclusion, methodological claim | C1 | "what remains does not transfer" → "what remains did not transfer to the one independent cohort we could test". |
| 22 | §IX Conclusion, closing | ED/HK | Added forward reference to §VII-A, resolving the previously unused `sec:synthetic` label. |
| 23 | §VI-C, standardisation text | HK | Hardcoded "Section~V-B" replaced with `\ref{sec:mech2}`. |
| 24 | Reproducibility | HK | Stale "380 automated tests" → "416", with the provenance gate and external-cohort temporal guard named. |

---

## Files added

| File | Purpose |
|---|---|
| `paper/RESPONSE_TO_REVIEWERS.md` | Point-by-point response |
| `paper/CHANGELOG_REVISION.md` | This document |
| `paper/VERIFICATION_REPORT.md` | Compilation, cross-reference, citation and test verification |

## Files not modified

No file outside `paper/` was modified in this revision. In particular, no
change was made to `src/`, `scripts/`, `tests/`, `reports/tables/`,
`reports/figures/`, `config/` or `data/`. The manuscript's numerical content
is unchanged, so no regeneration of results was required or performed.
