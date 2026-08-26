# Task 2 findings — Enam Medical College trace

**Artefacts:** `audit/outputs/enam_trace.csv` (264 rows),
`audit/outputs/openalex_trace_raw.csv` (270 rows), raw JSON in
`audit/raw/openalex/`

## Answer to the core question

**Yes — the Enam Medical College CKD data has been used in published
research, at two levels, and one of them is retracted.**

### Level 1: the dataset creators' own paper — RETRACTED

The UCI card's own citation field names:

> "M. A. Islam, S. Akter, M. S. Hossen, S. A. Keya, S. A. Tisha and S.
> Hossain, 'Risk Factor Prediction of Chronic Kidney Disease based on Machine
> Learning Algorithms,' 2020 3rd International Conference on Intelligent
> Sustainable Systems (ICISS), Thoothukudi, India, 2020, pp. 952-957, doi:
> 10.1109/ICISS49785.2020.9315878."

Resolving that DOI returns, from two independent authorities:

| Source | Title returned | Flag |
|---|---|---|
| OpenAlex (`W3127397372`) | "**Retracted:** Risk Factor Prediction of Chronic Kidney Disease based on Machine Learning Algorithms" | `is_retracted: true` |
| Crossref (IEEE's own deposit) | "**Retracted:** Risk Factor Prediction of Chronic Kidney Disease based on Machine Learning Algorithms" | publisher: IEEE |

**The paper that introduced this dataset has been retracted by IEEE.** It
carries 70 citations in OpenAlex and 52 in Crossref.

**The UCI dataset card does not say so.** A programmatic search of the
fetched page HTML finds no occurrence of "retract" in any form, and the
card's citation field reproduces the title *without* the "Retracted:" prefix
that both OpenAlex and Crossref carry. The dataset remains live, last
updated 8 March 2024.

I have **not** established *why* it was retracted. The retraction notice
itself was not retrieved, and no reason should be inferred from the flag
alone. That is the single most important open question from this task.

### Level 2: independent third-party use — one explicit link found

> "This study involved training an open-source clinical dataset of 200
> patients from Enam Medical College, comprising 28 clinical features,
> obtained from the UCI machine learning repository."

— *Optimizing Chronic Kidney Disease Prediction: A Machine Learning Approach
with Minimal Diagnostic Predictors*, Journal of Current Science and
Technology (Rangsit University), DOI `10.59796/jcst.v15n1.2025.76`,
published 2024-12-24 (Crossref-verified).

This paper both cites the native paper and names the hospital, with the
patient count (200) and feature count (28) matching id=857 exactly.

## Verdict counts

| Verdict | n |
|---|---|
| no_match | 192 |
| mentions_enam_only | 64 |
| mentions_both_no_link | 5 |
| **explicit_link** | **2** |
| inaccessible | 1 |

The 64 `mentions_enam_only` rows are overwhelmingly clinical papers from
*Journal of Enam Medical College* — the institution's own journal — on
unrelated topics. They establish that the hospital publishes; they say
nothing about the dataset.

## A correction I made to my own method

My first classifier counted "patients" and "records" as evidence of a
dataset reference. That produced **seven** `explicit_link` rows, of which six
were Enam Medical College's own clinical papers on kidney topics — a urinary
tract infection series, an AKI ICU outcome study, a diabetic nephropathy
cohort. Naming a hospital and studying kidneys is not the same as using the
UCI dataset. The cue list now requires a repository or ML-dataset term, and
the count fell from seven to one (plus the UCI card itself, classified
directly as primary documentation). The loose version is preserved in the
script's comments so the tightening is auditable.

## What remains uncertain

1. **Why the native paper was retracted.** Not established. The IEEE
   retraction notice was not retrieved.
2. **Whether more third-party papers use the data without naming the
   hospital.** Very likely: most papers cite the UCI dataset, not the
   hospital. The 69 citing works were scanned by title and abstract only —
   OpenAlex does not give full text, so a paper naming Enam only in its
   Methods section would not be caught.
3. **Whether the retraction affects the dataset's validity.** Unknown, and
   not inferable from the retraction flag.

## Natural next step

Task 3: the record-level overlap test between the two releases. Note that
Task 1 established the two releases document *different countries* —
Tamilnadu, India for id=336 and Savar, Bangladesh for id=857 — which makes
the overlap question sharper, not softer.
