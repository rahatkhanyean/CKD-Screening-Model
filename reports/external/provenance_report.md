# External-dataset provenance report

*Generated 2026-08-22 by `scripts/00_provenance.py`. Every number below is read from `reports/tables/table_24_provenance.csv` and `table_24_provenance_evidence.csv`; the verdict rule is the mechanical one in `src/ckd/data/provenance.py::classify`, fixed before any external result existed.*

**Gate.** Only datasets classified INDEPENDENT with a completed or not-applicable record check may appear in an external-validation table. `tests/test_external.py::TestProvenanceGate` enforces this mechanically. A RESTRICTED dataset re-enters this stage when its data arrives; its verdict is documentary until then and it remains blocked (`record_check = pending-data`).

## Verdicts

| Dataset | Verdict | Record check | Design | n (pos/neg) | Evidence |
|---|---|---|---|---|---|
| `birdem` | **INDEPENDENT** | not-applicable-no-shared-schema | longitudinal-repeated | 4000 (185/215) | no shared record-level basis; documentary evidence only |
| `icddrb_kabir` | **INDEPENDENT** | pending-data | cross-sectional-diagnostic | 284 (112/172) | no shared record-level basis; documentary evidence only |
| `mimic_iv_demo` | **INDEPENDENT** | not-applicable-no-shared-schema | ehr-extract | 100 | no shared record-level basis; documentary evidence only |
| `th_uae` | **INDEPENDENT** | not-applicable-no-shared-schema | prospective-incident | 491 (56/435) | no shared record-level basis; documentary evidence only |
| `uci2015` | **SAME-SOURCE** | done | cross-sectional-diagnostic | 400 (250/150) | records reproduce the internal cohort: match fraction 1.000 vs null 0.003 +/- 0.003 (max 0.015) over 50 shuffles |
| `uci2023_v2` | **SAME-SOURCE** | done | cross-sectional-diagnostic | 200 (128/72) | byte-identical to the internal raw file |

## Interval-containment matching

Each internal patient is a vector of intervals over 13 shared variables (`age`, `sg`, `al`, `su`, `bgr`, `bu`, `sc`, `sod`, `pot`, `hemo`, `pcv`, `wbcc`, `rbcc`). A candidate row is *compatible* with a patient iff the outcome matches and every shared value falls inside the patient's interval; missing values are treated as compatible, which can only raise the match fraction (the conservative direction for a gate). The observed maximum bipartite matching is calibrated against 50 column-permutation shuffles that preserve every marginal distribution while destroying cross-variable structure.

| Dataset | Match fraction | Null (mean +/- sd, max) | Internal rows with any partner |
|---|---|---|---|
| `uci2015` | 1.000 | 0.003 +/- 0.003, max 0.015 | 200 / 200 |

## Held-out categorical agreement over uniquely pinned pairs

Where the containment relation pins an internal patient to exactly one candidate row, the pair can be checked on shared categorical variables that played **no part** in the matching. Consistency there is held-out confirmation of record identity; a single systematic contradiction would refute it.

| Dataset | Unique pins | Variable | Pairs compared | Candidate missing | Contradictions | Observed mapping |
|---|---|---|---|---|---|---|
| `uci2015` | 187 | `rbc` | 120 | 67 | 0 | 0<->normal (n=98); 1<->abnormal (n=22) |
| `uci2015` | 187 | `pc` | 159 | 28 | 0 | 0<->normal (n=117); 1<->abnormal (n=42) |
| `uci2015` | 187 | `pcc` | 184 | 3 | 0 | 0<->notpresent (n=160); 1<->present (n=24) |
| `uci2015` | 187 | `ba` | 184 | 3 | 0 | 0<->notpresent (n=173); 1<->present (n=11) |
| `uci2015` | 187 | `htn` | 186 | 1 | 0 | 0<->no (n=113); 1<->yes (n=73) |
| `uci2015` | 187 | `dm` | 186 | 1 | 0 | 0<->no (n=120); 1<->yes (n=66) |
| `uci2015` | 187 | `cad` | 186 | 1 | 0 | 0<->no (n=165); 1<->yes (n=21) |
| `uci2015` | 187 | `appet` | 186 | 1 | 0 | 0<->good (n=149); 1<->poor (n=37) |
| `uci2015` | 187 | `pe` | 186 | 1 | 0 | 0<->no (n=151); 1<->yes (n=35) |
| `uci2015` | 187 | `ane` | 186 | 1 | 0 | 0<->no (n=156); 1<->yes (n=30) |

Two side-findings follow from a zero-contradiction mapping and are recorded here because they resolve uncertainties documented in `src/ckd/features/configs.py`: the internal coding polarity of each mapped categorical becomes empirically determined (e.g. `appet` 1 = poor), and candidate values that are *missing* map to internal `0`, i.e. the internal release coded missingness as the negative/normal level — a data-quality fact that must be reported wherever those variables are interpreted.

## Bin-edge forensics

Our file's continuous variables were discretised before release. If the published bin edges coincided with a candidate's observed values or deciles, the bins would plausibly have been derived from that candidate's data.

| Dataset | Variable | Finite edges | Matching observed values | Matching deciles |
|---|---|---|---|---|
| `uci2015` | `age` | 9 | 9 | 2 |
| `uci2015` | `sg` | 8 | 1 | 3 |
| `uci2015` | `al` | 5 | 5 | 4 |
| `uci2015` | `su` | 5 | 5 | 2 |
| `uci2015` | `bgr` | 9 | 3 | 0 |
| `uci2015` | `bu` | 8 | 0 | 0 |
| `uci2015` | `sc` | 7 | 1 | 0 |
| `uci2015` | `sod` | 9 | 4 | 2 |
| `uci2015` | `pot` | 4 | 0 | 0 |
| `uci2015` | `hemo` | 9 | 8 | 1 |
| `uci2015` | `pcv` | 9 | 0 | 0 |
| `uci2015` | `wbcc` | 9 | 0 | 0 |
| `uci2015` | `rbcc` | 9 | 0 | 0 |

