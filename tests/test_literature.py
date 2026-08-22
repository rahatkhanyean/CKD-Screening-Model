"""Prior-work survey integrity tests (Phase 1b).

The survey table is the evidence behind the Introduction's motivating
claim, so its discipline is enforced: complete schema, no silent blanks
(``unclear`` and ``na`` are values; an empty cell is a bug), controlled
vocabularies, and honesty markers — a row may carry a numeric headline
metric only if it says where that number was verified from, and every row
not yet fully hand-checked must be flagged as such.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ckd.config import project_root

REQUIRED_COLUMNS = [
    "ref_key", "citation", "year", "doi_or_url", "dataset_version",
    "reported_headline_metric", "metric_value",
    "used_affected", "used_stage", "used_grf", "kabir_sc_egfr_coding",
    "did_external_validation", "validation_design",
    "verified_from", "todo_hand_check", "notes",
]
TRI_STATE = {"yes", "no", "unclear", "na"}
VERIFIED_FROM = {"full-text", "abstract", "citation-only"}


@pytest.fixture(scope="module")
def survey() -> pd.DataFrame:
    path = project_root() / "data" / "literature" / "prior_work.csv"
    assert path.is_file(), "data/literature/prior_work.csv missing"
    return pd.read_csv(path, dtype=str).fillna("")


@pytest.fixture(scope="module")
def rendered() -> pd.DataFrame:
    path = project_root() / "reports" / "tables" / "table_26_prior_work.csv"
    assert path.is_file(), "run scripts/07_literature.py"
    return pd.read_csv(path, dtype=str).fillna("")


class TestSchema:
    def test_all_columns_present(self, survey):
        assert [c for c in REQUIRED_COLUMNS if c not in survey.columns] == []

    def test_no_empty_cells(self, survey):
        empties = survey[survey[REQUIRED_COLUMNS].eq("").any(axis=1)]
        assert empties.empty, (
            "empty cells (use 'unclear'/'na'/a note) in rows: "
            f"{list(empties['ref_key'])}"
        )

    def test_ref_keys_unique(self, survey):
        assert survey["ref_key"].is_unique

    def test_controlled_vocabularies(self, survey):
        for column in ("used_affected", "used_stage", "used_grf",
                       "kabir_sc_egfr_coding"):
            assert set(survey[column]) <= TRI_STATE, column
        assert set(survey["verified_from"]) <= VERIFIED_FROM
        assert set(survey["todo_hand_check"]) <= {"yes", "no"}


class TestHonestyMarkers:
    def test_numeric_metrics_only_when_verified(self, survey):
        """A number may be quoted only from a text we actually saw."""
        numeric = pd.to_numeric(survey["metric_value"], errors="coerce")
        offenders = survey[numeric.notna()
                           & ~survey["verified_from"].isin(["full-text", "abstract"])]
        assert offenders.empty, list(offenders["ref_key"])

    def test_yes_no_leakage_coding_only_when_verified(self, survey):
        """used_affected/stage/grf may say yes/no only for rows verified
        from full text (na is fine anywhere: the column doesn't exist in
        uci2015-only papers)."""
        for column in ("used_affected", "used_stage", "used_grf"):
            decided = survey[survey[column].isin({"yes", "no"})]
            v2_rows = decided[decided["dataset_version"].str.contains("uci2023")]
            offenders = v2_rows[v2_rows["verified_from"] != "full-text"]
            assert offenders.empty, (column, list(offenders["ref_key"]))

    def test_citation_only_rows_are_flagged_for_hand_check(self, survey):
        citation_only = survey[survey["verified_from"] == "citation-only"]
        assert (citation_only["todo_hand_check"] == "yes").all()

    def test_uci2015_only_papers_use_na_for_absent_columns(self, survey):
        uci2015_only = survey[survey["dataset_version"] == "uci2015-continuous"]
        for column in ("used_affected", "used_stage", "used_grf"):
            assert (uci2015_only[column] == "na").all(), column


class TestRenderedTable:
    def test_rendered_matches_source(self, survey, rendered):
        assert len(rendered) == len(survey)
        assert list(rendered["ref_key"]) == list(survey["ref_key"])

    def test_summary_counts_recompute(self, survey):
        path = project_root() / "reports" / "tables" / "table_26_prior_work_summary.csv"
        summary = pd.read_csv(path).set_index("quantity")["value"]
        assert summary["n_studies_surveyed"] == len(survey)
        verified = survey["verified_from"].isin(["full-text", "abstract"])
        assert summary["n_independently_verified"] == int(verified.sum())
        numeric = pd.to_numeric(survey["metric_value"], errors="coerce")
        assert summary["n_verified_metric_geq_099"] == int(
            ((numeric >= 0.99) & verified).sum()
        )
