# FINDINGS — CKD dataset provenance & integrity audit

**Last updated:** 2026-08-26 · **Scope:** UCI id=336, UCI id=857 ·
**All artefacts:** `audit/raw/` (checksummed), `audit/outputs/`,
`audit/scripts/`, `audit/notes/`

---

## Answer to the core question

> *Was the CKD data from Enam Medical College, Bangladesh ever used in
> published research, and how does it relate to the UCI CKD releases?*

**Yes, and the answer has an unexpected component.**

1. **UCI id=857 *is* the Enam Medical College data.** Its dataset card states
   it verbatim.
2. **The paper that introduced it has been retracted by IEEE** — and UCI does
   not say so. The retraction is one of at least 11 in the same proceedings,
   so it likely reflects a venue-level action rather than a finding about
   this dataset. **The reason was not established** and must not be guessed.
3. **At least one independent paper uses it and names the hospital.**
4. **Its 200 patients also appear in id=336**, which documents a *different
   country*. That is a contradiction this audit measures but cannot resolve.

---

## Confirmed facts (stated in a source, quote captured)

| # | Fact | Source |
|---|---|---|
| F1 | id=857 is "real Bangladeshi patient data… collected from Enam Medical College, Savar, Dhaka, Bangladesh" | UCI dataset card, id=857 |
| F2 | id=857's own citation is Islam et al., ICISS 2020, `10.1109/ICISS49785.2020.9315878` | UCI card citation field |
| F3 | **That paper is retracted.** OpenAlex `is_retracted: true`; Crossref (IEEE's own deposit) titles it "Retracted: …" | OpenAlex `W3127397372`, Crossref |
| F4 | **UCI's card carries no retraction notice.** The string "retract" occurs nowhere in the page HTML, and the card reproduces the title without the "Retracted:" prefix | fetched page HTML |
| F4b | **The retraction is not isolated.** 11 of 200 sampled ICISS 2020 proceedings articles (5.5%) carry a "Retracted:" title prefix, the target paper among them | Crossref, `retraction_context.csv` |
| F5 | id=336's source is "Dr.P.Soundarapandian… Apollo Hospitals, Managiri, Madurai Main Road, Karaikudi, Tamilnadu, India" | legacy `chronic_kidney_disease.info.txt` |
| F6 | **That statement is only in the legacy archive.** "Apollo" and "India" occur nowhere in the modern card or the `ucimlrepo` metadata object | programmatic check |
| F7 | All 200 id=857 patients match into id=336; 187 uniquely; null never exceeds 0.015 over 200 draws | `overlap_report.md` |
| F8 | Ten variables excluded from matching agree on **1763/1763** comparisons, zero contradictions | `overlap_report.md` |
| F9 | A 2024 paper (JCST, `10.59796/jcst.v15n1.2025.76`) states it trained on "200 patients from Enam Medical College… obtained from the UCI machine learning repository" | Crossref-verified |

## Recorded absences (checked, and not found)

| Fact | id=336 | id=857 |
|---|---|---|
| Ethics / IRB approval | not stated | not stated |
| Informed consent | not stated | not stated |
| Collection date range | "nearly 2 months", no dates | not stated |
| Any mention of the other release | not stated | not stated |
| Retraction notice | n/a | **not stated** |

---

## The central tension

Two documented facts cannot both be straightforwardly true:

* id=336 documents **Apollo Hospitals, Karaikudi, Tamilnadu, India** (2015).
* id=857 documents **Enam Medical College, Savar, Dhaka, Bangladesh** (2020).
* Their patient records correspond at 1.000 match, 187 unique pins, with
  zero contradictions on ten held-out variables.

This audit **measures** the correspondence and **takes no view** on how it
arose. Candidate explanations — re-release under new documentation, an error
in one of the provenance statements, an undisclosed derivation — are not
distinguishable from the data, and I decline to guess between them. The
question belongs to the repository maintainers and the dataset authors, who
have **not** been contacted.

---

## Open questions

1. **Why was the native paper retracted?** Not established. The IEEE
   retraction notice was not retrieved; a targeted web search surfaced no
   notice. What *is* established is that at least 11 of 200 sampled papers
   in the same proceedings carry the same prefix, which points toward a
   venue-level action. It would be an error to read the flag as a judgement
   on the CKD data. *Highest-value next step: obtain the IEEE notice.*
2. **Does the retraction bear on the dataset's validity?** Unknown, and not
   inferable from the flag.
3. **How many papers use id=857 without naming the hospital?** Probably most.
   Abstract-level screening cannot tell (see dead end 2).
4. **Which release does each citing paper actually use?** 65 of 69 are
   "unclear" from abstracts alone.
5. **Is either provenance statement in error, and which?** Unresolved.

## Dead ends

1. **Semantic Scholar API** — HTTP 429 rate-limit. OpenAlex used instead.
2. **Abstract-level screening for Methods-level facts.** OpenAlex supplies
   title and abstract, not full text. Whether a paper used serum creatinine,
   or validated across releases, usually lives in the Methods. 65/69 rows in
   `literature.csv` are honestly "unclear" as a result. Resolving them needs
   full-text retrieval, much of it paywalled.
3. **Two arXiv CKD papers** scanned for Enam: no mention.
4. **`opendatabay.com` mirror** of the Bangladeshi renal dataset: no Enam
   mention in the fetched page.

## Two corrections I made to my own method

Recorded because both inflated a headline count before I caught them:

* **Task 2.** My first classifier treated "patients" and "records" as
  evidence of a dataset reference, producing **7** `explicit_link` rows.
  Six were Enam Medical College's *own clinical papers* on kidney topics —
  a UTI series, an AKI ICU study, a diabetic nephropathy cohort. Naming a
  hospital and studying kidneys is not using the UCI dataset. Tightened to
  require a repository cue; count fell 7 → 1 (+ the card itself).
* **Task 4.** My first pass flagged **3** papers as non-independent
  cross-release designs on the word "merged" alone. One was a breast-cancer
  study; one a kidney-cancer survey. Tightened to require a release
  reference too; count fell 3 → 0.

---

## Artefact index

| File | Contents |
|---|---|
| `outputs/provenance.csv` | 26 provenance facts with verbatim quotes (Task 1) |
| `outputs/enam_trace.csv` | 264 screened records with verdicts (Task 2) |
| `outputs/openalex_trace_raw.csv` | 270 raw OpenAlex hits |
| `outputs/overlap_report.md` | Task 3 method, statistics, interpretation |
| `outputs/overlap_summary.json` | Machine-readable overlap statistics |
| `outputs/overlap_null_draws.csv` | 200 null draws |
| `outputs/overlap_per_patient.csv` | Per-patient partner counts and pairing |
| `outputs/literature.csv` | 69 citing works (Task 4) |
| `outputs/retraction_context.csv` | 200 ICISS 2020 articles, retraction prefix flagged |
| `raw/checksums.csv` | SHA-256 of every downloaded file |
| `raw/page_fetches.csv` | HTTP status of every page fetch |
| `notes/task1_findings.md`, `notes/task2_findings.md` | Per-task notes |
