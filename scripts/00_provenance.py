"""Stage 0: provenance gate for every registered external dataset.

For each candidate in ``config/external_datasets.yaml``, collect the
evidence (byte identity, cohort arithmetic, bin-edge forensics, and — where
a shared record-level basis exists — interval-containment matching against
the internal cohort), classify it INDEPENDENT / OVERLAPPING / SAME-SOURCE
by the mechanical rule in ``ckd.data.provenance.classify``, and write:

* ``reports/tables/table_24_provenance.csv``           (verdicts; gates Phase 3)
* ``reports/tables/table_24_provenance_evidence.csv``  (per-variable forensics)
* ``reports/external/provenance_report.md``            (rendered FROM the tables)

The report-from-tables invariant holds here as everywhere: the markdown is
generated from the CSVs, never hand-typed.

Usage
-----
    python scripts/00_provenance.py
"""

from __future__ import annotations

import hashlib
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ckd.config import ensure_dirs, load_config, project_root  # noqa: E402
from ckd.data.clean import clean_dataset  # noqa: E402
from ckd.data.external import LOADERS, load_registry, verify_raw_files  # noqa: E402
from ckd.data.provenance import (  # noqa: E402
    CONTAINMENT_VARIABLES,
    HELDOUT_CATEGORICAL,
    N_NULL_SHUFFLES,
    bin_edge_forensics,
    categorical_agreement,
    classify,
    compatibility_matrix,
    containment_check,
    unique_pins,
)

#: Datasets with a shared record-level basis against the internal cohort
#: (same variable vocabulary at comparable meaning). Everything else gets a
#: documentary verdict, and the report says so explicitly.
RECORD_LEVEL_CANDIDATES = ("uci2015",)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    cfg = load_config()
    root = project_root()
    (tables_dir,) = ensure_dirs(cfg["paths"]["tables_dir"])
    (external_dir,) = ensure_dirs("reports/external")

    registry = load_registry()
    internal = clean_dataset()
    internal_hash = sha256_of(root / cfg["paths"]["raw_csv"])

    verdict_rows: list[dict] = []
    evidence_frames: list[pd.DataFrame] = []
    agreement_frames: list[pd.DataFrame] = []
    containment_details: dict[str, object] = {}

    for dataset_id, record in sorted(registry.items()):
        print(f"[{dataset_id}] status={record.status} design={record.design}")
        if record.status == "OBTAINED":
            verify_raw_files(record)
            print(f"  raw files verified ({len(record.raw_files)})")

        byte_identical = any(f.sha256 == internal_hash for f in record.raw_files)
        containment = None
        record_check = "not-applicable-no-shared-schema"

        if byte_identical:
            record_check = "done"
        elif record.status != "OBTAINED":
            record_check = "pending-data"
        elif dataset_id in RECORD_LEVEL_CANDIDATES:
            result = LOADERS[dataset_id]()
            mappings = {
                k: {b["label"]: (float(b["lower"]), float(b["upper"])) for b in v}
                for k, v in internal.bin_map.items()
            }
            containment = containment_check(
                internal.clean,
                internal.target.to_numpy(),
                result.frame,
                result.target.to_numpy(),
                mappings,
            )
            containment_details[dataset_id] = containment
            record_check = "done"
            forensics = bin_edge_forensics(mappings, result.frame)
            forensics.insert(0, "dataset_id", dataset_id)
            evidence_frames.append(forensics)
            print(
                f"  containment: match={containment.match_fraction:.3f} "
                f"null={containment.null_mean:.3f}+/-{containment.null_sd:.3f} "
                f"(rows with any partner: {containment.rows_with_any_partner}"
                f"/{containment.n_internal})"
            )
            # Held-out confirmation: agreement of categorical variables that
            # played no part in the matching, over uniquely pinned pairs.
            compat = compatibility_matrix(
                internal.clean, internal.target.to_numpy(),
                result.frame, result.target.to_numpy(), mappings,
            )
            pairs = unique_pins(compat)
            agreement = categorical_agreement(internal.clean, result.frame, pairs)
            agreement.insert(0, "dataset_id", dataset_id)
            agreement.insert(1, "n_unique_pins", len(pairs))
            agreement_frames.append(agreement)
            n_contra = int(agreement["n_contradictions"].sum())
            print(
                f"  held-out agreement: {len(pairs)} uniquely pinned pairs; "
                f"{n_contra} contradictions across "
                f"{len(HELDOUT_CATEGORICAL)} categorical variables"
            )

        verdict, reason = classify(byte_identical, containment)
        print(f"  verdict: {verdict} - {reason}")

        exp = record.expected
        verdict_rows.append(
            {
                "dataset_id": dataset_id,
                "verdict": verdict,
                "record_check": record_check,
                "status": record.status,
                "design": record.design,
                "country": record.country,
                "n_rows": exp.get("n_rows"),
                "n_positive": exp.get("n_positive"),
                "n_negative": exp.get("n_negative"),
                "byte_identical_to_internal": byte_identical,
                "containment_match_fraction": (
                    containment.match_fraction if containment else None
                ),
                "containment_null_mean": containment.null_mean if containment else None,
                "containment_null_sd": containment.null_sd if containment else None,
                "containment_null_max": containment.null_max if containment else None,
                "rows_with_any_partner": (
                    containment.rows_with_any_partner if containment else None
                ),
                "reason": reason,
                "doi": record.doi,
            }
        )

    verdicts = pd.DataFrame(verdict_rows)
    verdict_path = tables_dir / "table_24_provenance.csv"
    verdicts.to_csv(verdict_path, index=False)
    print(f"\nwrote {verdict_path}")

    evidence = (
        pd.concat(evidence_frames, ignore_index=True)
        if evidence_frames
        else pd.DataFrame(
            columns=["dataset_id", "variable", "n_finite_edges",
                     "edges_matching_observed_values", "edges_matching_deciles"]
        )
    )
    evidence_path = tables_dir / "table_24_provenance_evidence.csv"
    evidence.to_csv(evidence_path, index=False)
    print(f"wrote {evidence_path}")

    agreement = (
        pd.concat(agreement_frames, ignore_index=True)
        if agreement_frames
        else pd.DataFrame(
            columns=["dataset_id", "n_unique_pins", "variable", "n_pairs_compared",
                     "n_candidate_missing", "n_contradictions", "observed_mapping"]
        )
    )
    agreement_path = tables_dir / "table_24_provenance_agreement.csv"
    agreement.to_csv(agreement_path, index=False)
    print(f"wrote {agreement_path}")

    render_report(external_dir / "provenance_report.md", verdicts, evidence, agreement)
    print(f"wrote {external_dir / 'provenance_report.md'}")
    return 0


def render_report(
    path: Path,
    verdicts: pd.DataFrame,
    evidence: pd.DataFrame,
    agreement: pd.DataFrame,
) -> None:
    """Render the markdown report from the two tables (never hand-typed)."""
    lines: list[str] = []
    w = lines.append
    w("# External-dataset provenance report")
    w("")
    w(f"*Generated {date.today().isoformat()} by `scripts/00_provenance.py`. "
      "Every number below is read from "
      "`reports/tables/table_24_provenance.csv` and "
      "`table_24_provenance_evidence.csv`; the verdict rule is the "
      "mechanical one in `src/ckd/data/provenance.py::classify`, fixed "
      "before any external result existed.*")
    w("")
    w("**Gate.** Only datasets classified INDEPENDENT with a completed or "
      "not-applicable record check may appear in an external-validation "
      "table. `tests/test_external.py::TestProvenanceGate` enforces this "
      "mechanically. A RESTRICTED dataset re-enters this stage when its "
      "data arrives; its verdict is documentary until then and it remains "
      "blocked (`record_check = pending-data`).")
    w("")
    w("## Verdicts")
    w("")
    w("| Dataset | Verdict | Record check | Design | n (pos/neg) | Evidence |")
    w("|---|---|---|---|---|---|")
    for _, r in verdicts.iterrows():
        n = (
            f"{int(r['n_rows'])} ({int(r['n_positive'])}/{int(r['n_negative'])})"
            if pd.notna(r["n_rows"]) and pd.notna(r["n_positive"])
            else (f"{int(r['n_rows'])}" if pd.notna(r["n_rows"]) else "-")
        )
        w(
            f"| `{r['dataset_id']}` | **{r['verdict']}** | {r['record_check']} "
            f"| {r['design']} | {n} | {r['reason']} |"
        )
    w("")
    w("## Interval-containment matching")
    w("")
    done = verdicts[verdicts["containment_match_fraction"].notna()]
    if len(done):
        w("Each internal patient is a vector of intervals over "
          f"{len(CONTAINMENT_VARIABLES)} shared variables "
          f"(`{'`, `'.join(CONTAINMENT_VARIABLES)}`). A candidate row is "
          "*compatible* with a patient iff the outcome matches and every "
          "shared value falls inside the patient's interval; missing values "
          "are treated as compatible, which can only raise the match "
          "fraction (the conservative direction for a gate). The observed "
          "maximum bipartite matching is calibrated against "
          f"{N_NULL_SHUFFLES} column-permutation shuffles that preserve "
          "every marginal distribution while destroying cross-variable "
          "structure.")
        w("")
        w("| Dataset | Match fraction | Null (mean +/- sd, max) | Internal rows with any partner |")
        w("|---|---|---|---|")
        for _, r in done.iterrows():
            w(
                f"| `{r['dataset_id']}` | {r['containment_match_fraction']:.3f} "
                f"| {r['containment_null_mean']:.3f} +/- {r['containment_null_sd']:.3f}, "
                f"max {r['containment_null_max']:.3f} "
                f"| {int(r['rows_with_any_partner'])} / 200 |"
            )
    else:
        w("*No dataset offered a shared record-level basis this run.*")
    w("")
    w("## Held-out categorical agreement over uniquely pinned pairs")
    w("")
    if len(agreement):
        w("Where the containment relation pins an internal patient to exactly "
          "one candidate row, the pair can be checked on shared categorical "
          "variables that played **no part** in the matching. Consistency "
          "there is held-out confirmation of record identity; a single "
          "systematic contradiction would refute it.")
        w("")
        w("| Dataset | Unique pins | Variable | Pairs compared | Candidate missing | Contradictions | Observed mapping |")
        w("|---|---|---|---|---|---|---|")
        for _, r in agreement.iterrows():
            w(
                f"| `{r['dataset_id']}` | {int(r['n_unique_pins'])} | `{r['variable']}` "
                f"| {int(r['n_pairs_compared'])} | {int(r['n_candidate_missing'])} "
                f"| {int(r['n_contradictions'])} | {r['observed_mapping']} |"
            )
        w("")
        w("Two side-findings follow from a zero-contradiction mapping and are "
          "recorded here because they resolve uncertainties documented in "
          "`src/ckd/features/configs.py`: the internal coding polarity of "
          "each mapped categorical becomes empirically determined (e.g. "
          "`appet` 1 = poor), and candidate values that are *missing* map "
          "to internal `0`, i.e. the internal release coded missingness as "
          "the negative/normal level — a data-quality fact that must be "
          "reported wherever those variables are interpreted.")
    else:
        w("*No uniquely pinned pairs this run.*")
    w("")
    w("## Bin-edge forensics")
    w("")
    if len(evidence):
        w("Our file's continuous variables were discretised before release. "
          "If the published bin edges coincided with a candidate's observed "
          "values or deciles, the bins would plausibly have been derived "
          "from that candidate's data.")
        w("")
        w("| Dataset | Variable | Finite edges | Matching observed values | Matching deciles |")
        w("|---|---|---|---|---|")
        for _, r in evidence.iterrows():
            w(
                f"| `{r['dataset_id']}` | `{r['variable']}` | {int(r['n_finite_edges'])} "
                f"| {int(r['edges_matching_observed_values'])} "
                f"| {int(r['edges_matching_deciles'])} |"
            )
    else:
        w("*No forensics computed this run.*")
    w("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
