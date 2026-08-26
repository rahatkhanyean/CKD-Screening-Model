# Task 1 findings — provenance extraction

**Run:** 2026-08-26 · **Artefacts:** `audit/raw/` (checksummed),
`audit/outputs/provenance.csv` (23 facts)

## Confirmed, with verbatim quotes

**id=857 names Enam Medical College explicitly.** From the dataset card's
summary field, present verbatim in the fetched page HTML:

> "This dataset is real Bangladeshi patient data. The dataset is collected
> from Enam Medical College, Savar, Dhaka, Bangladesh."

**id=857 has a native publication by its own creators**, given in the card's
citation field:

> "M. A. Islam, S. Akter, M. S. Hossen, S. A. Keya, S. A. Tisha and S.
> Hossain, 'Risk Factor Prediction of Chronic Kidney Disease based on Machine
> Learning Algorithms,' 2020 3rd International Conference on Intelligent
> Sustainable Systems (ICISS), Thoothukudi, India, 2020, pp. 952-957, doi:
> 10.1109/ICISS49785.2020.9315878."

This already answers the core question in the affirmative at the weakest
useful strength: **the Enam Medical College data has been used in at least
one published paper — the dataset creators' own.** Task 2 tests whether
anyone else has.

**id=336 names Apollo Hospitals — but not where you would look for it.**

> "(a) Source: Dr.P.Soundarapandian.M.D.,D.M (Senior Consultant
> Nephrologist), Apollo Hospitals, Managiri, Madurai Main Road, Karaikudi,
> Tamilnadu, India."

This appears **only** in `chronic_kidney_disease.info.txt`, inside the legacy
`Chronic_Kidney_Disease.rar` under `machine-learning-databases/00336/`. It is
**not** on the modern dataset page and **not** in the `ucimlrepo` metadata
object: a programmatic check found no occurrence of "Apollo", "India",
"ethic", "IRB", "consent" or "approval" anywhere in that object. Anyone
citing the modern card alone has no documented hospital or country.

## Recorded absences (these are findings, not gaps)

| Fact | id=336 | id=857 |
|---|---|---|
| Ethics / IRB approval | not stated | not stated |
| Informed consent | not stated | not stated |
| Collection date range | "nearly 2 months", no dates | not stated |
| Any relationship to the other release | not stated | not stated |

Neither dataset's documentation mentions the other. **Any link between the
two releases has to come from record-level evidence (Task 3), not from
documentation.**

## What remains uncertain

1. Whether "July 2015" in id=336 is a collection date, a donation date or a
   compilation date. The file does not say, so it is recorded as
   `date_stated`, not as a collection range.
2. Whether the id=857 data was collected prospectively at Enam Medical
   College or assembled from records. The card says only "collected from".
3. Whether any ethics approval exists but went undocumented. Absence of a
   statement is not evidence of absence of approval, and is recorded as
   `unknown`, not as "no approval".

## Natural next step

Task 2: search for third-party papers naming Enam Medical College alongside
CKD data, and for papers citing either DOI while naming a specific
Bangladeshi hospital.
