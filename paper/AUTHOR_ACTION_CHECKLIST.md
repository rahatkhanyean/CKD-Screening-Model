# Author-action checklist

Items only the authors can supply. Each is marked `TODO(author)` in the
manuscript source, and `tests/test_manuscript_consistency.py` fails if a
placeholder is present without being marked, so none can be submitted by
accident.

| # | Item | Where | Why we did not fill it |
|---|---|---|---|
| A1 | Author names, affiliations, ORCIDs, corresponding-author email | `ieee_paper.tex` and `supplement.tex` title blocks | Inventing author identities would be fabrication |
| A2 | Funding statement (or an explicit "no funding") | Reproducibility section | Not derivable from the repository |
| A3 | Competing-interest declaration | Reproducibility section | Not derivable from the repository |
| A4 | Author contributions (CRediT taxonomy) | Reproducibility section | Requires knowing who did what |
| A5 | Archived repository DOI (Zenodo/figshare) and the public URL | Code-availability statement | No repository has been published; citing a URL that does not resolve would be fabrication |
| A6 | Target journal, and confirmation the page/word limit is met | — | The manuscript is currently **15 pages** in IEEEtran conference format plus a 4-page supplement; see the note below |
| A7 | Ethics-committee reference, if the target journal requires one even for public de-identified data | Ethics statement | Institution-specific |

## Notes

**A5 is the one that changes a claim.** Until the repository is archived and
its DOI minted, the manuscript says "the accompanying repository" rather than
naming a location. Do not replace this with a bare GitHub URL at submission
if the repository will later move; a DOI is what makes the reproducibility
claim durable.

**A6, page limit.** The main text is 15 pages. If the target venue is
stricter, the sections most readily moved to the supplement, in order of how
little the argument depends on them, are: the EBM shape-function discussion
and Fig. 5; the robustness table (Table XI); the discretisation-cost
subsections; and the audit-framework subsections beyond its statement of
scope. Each is already self-contained.

**What we did fill in.** Data availability, code availability, ethics and
registration statements are written from facts in the repository — licences,
checksums, the absence of preregistration, and the commit ordering that fixed
the model-selection rule before results were computed. They are not
placeholders and need no author input, though the authors should of course
check them.
