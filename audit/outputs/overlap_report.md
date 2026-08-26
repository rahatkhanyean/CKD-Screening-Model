# Task 3 — record-level overlap between UCI id=857 and id=336

**Code:** `audit/scripts/07_overlap_check.py` ·
**Inputs:** `audit/raw/uci_336_X.csv`, `uci_857_X.csv` (+ targets), checksummed
in `audit/raw/checksums.csv` ·
**Outputs:** `overlap_summary.json`, `overlap_null_draws.csv`,
`overlap_per_patient.csv`, `overlap_console.txt`

## Method

id=857 stores every continuous variable as an interval string; id=336 stores
raw values. Each id=857 patient is therefore a box in 13 dimensions and each
id=336 record is a point. A record is **compatible** with a patient when the
outcome label matches and every shared value lies inside the corresponding
interval. Missing values on either side count as compatible.

Two encodings were undone first:

* **Excel date-mangling.** Bin labels such as `1-Jan` and `20-Dec` are
  Excel's rendering of `1-1` and `12-20`. Decoded by the rule
  `"D-Mon" → "M - D"`, not dropped.
* **Open bins.** `< X` and `≥ X` become half-infinite intervals.

Containment variables (13): `age, sg, al, su, bgr, bu, sc, sod, pot, hemo,
pcv, wbcc, rbcc`. All 2,600 interval cells parsed; none were unparseable.

## Results

| Statistic | Value |
|---|---|
| id=857 patients | 200 (128 CKD) |
| id=336 records | 400 (250 CKD) |
| **Maximum bipartite match fraction** | **1.0000** — all 200 patients match to distinct records |
| **Unique pins** (patients compatible with exactly one record) | **187 / 200** |
| Maximum partners for any patient | 2 |
| Mean partners | 1.06 |
| Patients with no compatible record | 0 |

### Null calibration

Each candidate column was permuted independently, preserving every marginal
while destroying cross-variable structure; 200 draws.

| | Value |
|---|---|
| Null mean | 0.0025 |
| Null SD | 0.0036 |
| Null maximum | 0.0150 |
| Draws at or above the observed 1.0000 | **0 of 200** (*p* ≤ 0.005) |

The observed value is not approached by any draw. Note that the largest null
draw matches 3 of 200 patients; the observed result matches all 200.

**Row-order permutation was deliberately not used.** Maximum bipartite
matching depends only on the multiset of candidate records, so permuting rows
leaves the statistic unchanged by construction. It would be a degenerate
null, not a weak one.

### Cross-check on variables the matcher never saw

Ten categorical variables (`rbc, pc, pcc, ba, htn, dm, cad, appet, pe, ane`)
took no part in matching. For the 187 uniquely pinned patients, their values
were compared against the pinned id=336 record:

**1,763 of 1,763 comparisons agree. Zero contradictions.**

| Variable | Agree | Disagree |
|---|---|---|
| `ane`, `appet`, `cad`, `dm`, `htn`, `pe` | 186 each | 0 |
| `ba`, `pcc` | 184 each | 0 |
| `pc` | 159 | 0 |
| `rbc` | 120 | 0 |

Perfect agreement on ten variables that played no part in establishing the
correspondence is the strongest single piece of evidence here. A matcher
tuned on 13 variables has no mechanism to force agreement on ten others.

## Interpretation

Stated at the strength the evidence supports:

> **The data provide strong record-level evidence that the 200 patients in
> UCI id=857 are also represented in UCI id=336.** Every id=857 patient is
> compatible with at least one id=336 record; 187 are compatible with exactly
> one; ten variables excluded from the matching agree without a single
> contradiction; and a marginal-preserving null never exceeds 0.015.

What this does **not** establish:

* **Not exact record identity.** id=857 is discretised, so raw-value equality
  is untestable by construction. "Compatible with" is not "identical to".
* **Not a direction.** The evidence says the two releases draw on overlapping
  patients. It does not say which came first, whether one was derived from
  the other, or by whom.
* **Not an explanation.** Task 1 established that the two releases document
  *different countries* — Tamilnadu, India (id=336) and Savar, Bangladesh
  (id=857). This analysis cannot reconcile that. It measures the data; it
  takes no view on how the discrepancy arose.
* **Not a finding about the repository maintainers.** They have not been
  contacted, and nothing here is an officially confirmed provenance
  correction.

## Consequence for the literature

Any study that trains on one release and evaluates on the other is, on this
evidence, evaluating on patients overlapping its own training set. That is
non-independent evaluation, and any accuracy so obtained should not be read
as external validation. Task 4 identifies which studies do this.
