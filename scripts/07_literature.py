"""Stage 7: render the prior-work survey into report tables.

Input:  ``data/literature/prior_work.csv`` — one row per prior ML study on
        the UCI-2015 dataset and/or its discretised re-release
        ("UCI-2023" / Risk Factor Prediction of CKD, i.e. this study's
        analysed file). Cells state how each fact was verified
        (``verified_from``); anything not independently verified is
        ``unclear`` or attributed in ``notes`` — never guessed.

Output: ``reports/tables/table_26_prior_work.csv``          (the survey)
        ``reports/tables/table_26_prior_work_summary.csv``  (derived counts)

The Introduction's motivating claim in the research report is injected from
the summary table by stage 6, replacing the previously uncited assertion
that near-100% accuracies are common on this data.

Usage
-----
    python scripts/07_literature.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ckd.config import ensure_dirs, load_config, project_root  # noqa: E402

REQUIRED_COLUMNS = [
    "ref_key", "citation", "year", "doi_or_url", "dataset_version",
    "reported_headline_metric", "metric_value",
    "used_affected", "used_stage", "used_grf", "kabir_sc_egfr_coding",
    "did_external_validation", "validation_design",
    "verified_from", "todo_hand_check", "notes",
]

TRI_STATE = {"yes", "no", "unclear", "na"}
VERIFIED_FROM = {"full-text", "abstract", "citation-only"}


def load_prior_work() -> pd.DataFrame:
    path = project_root() / "data" / "literature" / "prior_work.csv"
    table = pd.read_csv(path, dtype=str).fillna("")
    missing = [c for c in REQUIRED_COLUMNS if c not in table.columns]
    if missing:
        raise SystemExit(f"prior_work.csv missing columns: {missing}")
    for column in ("used_affected", "used_stage", "used_grf",
                   "kabir_sc_egfr_coding"):
        bad = set(table[column]) - TRI_STATE
        if bad:
            raise SystemExit(f"{column}: invalid values {bad}")
    bad = set(table["verified_from"]) - VERIFIED_FROM
    if bad:
        raise SystemExit(f"verified_from: invalid values {bad}")
    empties = table[table[REQUIRED_COLUMNS].eq("").any(axis=1)]
    if len(empties):
        raise SystemExit(
            "prior_work.csv has empty cells (use 'unclear', 'na' or a note) "
            f"in rows: {list(empties['ref_key'])}"
        )
    return table


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    numeric = pd.to_numeric(table["metric_value"], errors="coerce")
    verified = table["verified_from"].isin(["full-text", "abstract"])
    uses_v2 = table["dataset_version"].str.contains("uci2023", case=False)
    cross = table["did_external_validation"].str.startswith("pseudo-external")
    rows = [
        ("n_studies_surveyed", int(len(table))),
        ("n_independently_verified", int(verified.sum())),
        ("n_awaiting_hand_check", int((table["todo_hand_check"] == "yes").sum())),
        ("n_with_numeric_headline_metric", int(numeric.notna().sum())),
        ("n_verified_metric_geq_099", int(((numeric >= 0.99) & verified).sum())),
        ("n_kabir_coded_sc_egfr_yes", int((table["kabir_sc_egfr_coding"] == "yes").sum())),
        ("n_using_uci2023_or_merged", int(uses_v2.sum())),
        ("n_cross_dataset_uci2015_uci2023", int(cross.sum())),
        ("n_with_genuinely_external_validation", int(
            (table["did_external_validation"] == "yes").sum())),
    ]
    return pd.DataFrame(rows, columns=["quantity", "value"])


def main() -> int:
    cfg = load_config()
    (tables_dir,) = ensure_dirs(cfg["paths"]["tables_dir"])
    table = load_prior_work()
    table.to_csv(tables_dir / "table_26_prior_work.csv", index=False)
    summary = summarise(table)
    summary.to_csv(tables_dir / "table_26_prior_work_summary.csv", index=False)
    print(f"wrote {tables_dir / 'table_26_prior_work.csv'} ({len(table)} studies)")
    print(f"wrote {tables_dir / 'table_26_prior_work_summary.csv'}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
