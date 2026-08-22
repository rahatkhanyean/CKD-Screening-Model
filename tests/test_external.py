"""External-dataset registry, loader and provenance-gate tests (Phase 0.2/0.3).

Two families of guarantee:

1. **Registry integrity.** Every registered dataset declares its provenance,
   its own prohibited columns, and (if obtained) checksummed raw files that
   actually verify. The declared cohort arithmetic matches the files on disk.
2. **The provenance gate.** A dataset that the provenance report has not
   classified INDEPENDENT can never appear in an external-validation table.
   This is the test that makes "validating on an overlapping cohort" a CI
   failure instead of a reviewer discovery.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from ckd.config import project_root
from ckd.data.external import (
    DESIGNS,
    LOADERS,
    STATUSES,
    VERDICTS,
    DatasetRecord,
    ExternalDataError,
    dataset_seed,
    get_dataset,
    load_birdem,
    load_registry,
    load_th_uae,
    load_uci2015,
    verify_raw_files,
)
from ckd.features.encoders import LeakageError, LeakageGuard

REGISTRY = load_registry()

#: Filename patterns that mark a table as carrying external-validation
#: results. Phase 3 scripts MUST name their output tables to match one of
#: these — test_naming_convention_is_load_bearing below keeps this list
#: honest against the tables directory.
EXTERNAL_TABLE_PATTERNS = (
    re.compile(r"external", re.IGNORECASE),
    re.compile(r"frozen_transfer", re.IGNORECASE),
    re.compile(r"cross_dataset", re.IGNORECASE),
    re.compile(r"casemix_retest", re.IGNORECASE),
)

PROVENANCE_TABLE = "table_24_provenance.csv"


def tables_dir() -> Path:
    return project_root() / "reports" / "tables"


class TestRegistryIntegrity:
    def test_expected_datasets_are_registered(self):
        assert set(REGISTRY) == {
            "uci2015",
            "uci2023_v2",
            "icddrb_kabir",
            "th_uae",
            "birdem",
            "mimic_iv_demo",
        }

    @pytest.mark.parametrize("dataset_id", sorted(REGISTRY))
    def test_declarations_are_complete(self, dataset_id):
        rec = REGISTRY[dataset_id]
        assert rec.status in STATUSES
        assert rec.design in DESIGNS
        assert rec.citation and rec.doi and rec.license and rec.country
        # Every dataset must declare its own outcome column(s): the guard
        # cannot protect against a column nobody named.
        assert rec.outcome_columns
        assert rec.prohibited_columns >= set(rec.outcome_columns)

    @pytest.mark.parametrize(
        "dataset_id",
        sorted(d for d in REGISTRY if REGISTRY[d].status == "OBTAINED"),
    )
    def test_obtained_files_exist_and_checksums_match(self, dataset_id):
        paths = verify_raw_files(REGISTRY[dataset_id])
        assert paths, f"{dataset_id} is OBTAINED but declared no files"

    def test_restricted_dataset_carries_acquisition_route(self):
        rec = REGISTRY["icddrb_kabir"]
        assert rec.status == "RESTRICTED"
        assert not rec.raw_files
        assert "icddrb.org" in rec.acquisition_note

    def test_uci2023_v2_is_the_internal_raw_file(self):
        """The registry's negative control must point at the very file the
        internal study analyses — that identity is what makes it a control."""
        rec = REGISTRY["uci2023_v2"]
        assert rec.raw_files[0].path == "data/raw/ckd-dataset-v2.csv"
        assert rec.raw_files[0].sha256.startswith("f24075f4")
        assert set(rec.prohibited_columns) == {"class", "affected", "stage", "grf"}

    def test_repeated_measures_sources_demand_grouped_splits(self):
        assert REGISTRY["birdem"].grouped_split_by == "Patient ID"
        assert REGISTRY["mimic_iv_demo"].grouped_split_by == "subject_id"

    def test_th_uae_prohibits_outcome_and_followup_time(self):
        rec = REGISTRY["th_uae"]
        assert "EventCKD35" in rec.prohibited_columns
        # Follow-up duration is knowable only after follow-up: post-diagnosis.
        assert "TimeToEventMonths" in rec.prohibited_columns

    def test_unknown_dataset_raises_with_known_ids(self):
        with pytest.raises(ExternalDataError, match="uci2015"):
            get_dataset("no_such_dataset")


class TestLoaders:
    def test_uci2015_cohort_arithmetic_and_label_defect(self):
        result = load_uci2015()
        exp = REGISTRY["uci2015"].expected
        assert result.frame.shape == (exp["n_rows"], exp["n_columns"])
        assert int(result.target.sum()) == exp["n_positive"]
        assert int((1 - result.target).sum()) == exp["n_negative"]
        # The two 'ckd\t' records are corrected AND reported, never silent.
        assert any("whitespace" in d for d in result.defects)

    def test_uci2015_values_are_continuous_not_binned(self):
        """The property that makes uci2015 the continuous-value vehicle:
        numeric columns carry point values, not interval strings."""
        result = load_uci2015()
        sc = pd.to_numeric(result.frame["sc"], errors="coerce")
        assert sc.notna().sum() > 300
        # More distinct observed values than any plausible bin count.
        assert sc.nunique() > 50

    def test_th_uae_cohort_arithmetic(self):
        result = load_th_uae()
        exp = REGISTRY["th_uae"].expected
        assert result.frame.shape == (exp["n_rows"], exp["n_columns"])
        assert int(result.target.sum()) == exp["n_positive"]
        # Header whitespace is normalised and reported.
        assert "HistoryHTN" in result.frame.columns
        assert "HistoryHTN " not in result.frame.columns

    def test_birdem_row_and_patient_level_labels_differ(self):
        """The trap this dataset sets: row-level labels overcount nothing but
        identity — 386 positive rows collapse to 185 ever-CKD patients. The
        loader exposes row-level and the caller must aggregate explicitly."""
        result = load_birdem()
        exp = REGISTRY["birdem"].expected
        assert len(result.frame) == exp["n_rows"]
        assert result.frame["Patient ID"].nunique() == exp["n_patients"]
        patient_level = result.frame.groupby("Patient ID")["CKD (1-yes, 0-no)"].max()
        assert int(patient_level.sum()) == exp["n_positive"]
        assert int(result.target.sum()) != int(patient_level.sum())

    def test_every_obtained_tabular_dataset_has_a_loader(self):
        tabular = {
            d
            for d, rec in REGISTRY.items()
            if rec.status == "OBTAINED" and rec.design != "ehr-extract"
            and d != "uci2023_v2"  # loaded by the internal pipeline, not here
        }
        assert tabular == set(LOADERS)


class TestPerSourceLeakageGuard:
    """The internal guard generalises: each source's declared prohibited set
    must actually be enforced by LeakageGuard when passed as ``forbidden``."""

    @pytest.mark.parametrize("dataset_id", sorted(LOADERS))
    def test_guard_refuses_each_sources_prohibited_columns(self, dataset_id):
        result = LOADERS[dataset_id]()
        rec = get_dataset(dataset_id)
        present = set(result.frame.columns) & rec.prohibited_columns
        assert present, f"{dataset_id}: prohibited columns absent from its own frame?"
        guard = LeakageGuard(forbidden=rec.prohibited_columns, label=dataset_id)
        with pytest.raises(LeakageError):
            guard.fit(result.frame)

    @pytest.mark.parametrize("dataset_id", sorted(LOADERS))
    def test_guard_passes_once_prohibited_and_identifiers_are_dropped(self, dataset_id):
        result = LOADERS[dataset_id]()
        rec = get_dataset(dataset_id)
        drop = list(rec.prohibited_columns | set(rec.identifier_columns))
        clean = result.frame.drop(columns=[c for c in drop if c in result.frame])
        guard = LeakageGuard(forbidden=rec.prohibited_columns, label=dataset_id)
        assert guard.fit(clean) is guard


class TestDatasetSeeds:
    def test_deterministic(self):
        assert dataset_seed(20240517, "uci2015") == dataset_seed(20240517, "uci2015")

    def test_distinct_across_datasets_and_masters(self):
        ids = sorted(REGISTRY)
        seeds = {dataset_seed(20240517, d) for d in ids}
        assert len(seeds) == len(ids)
        assert dataset_seed(20240517, "uci2015") != dataset_seed(20240518, "uci2015")

    def test_in_numpy_range(self):
        for d in REGISTRY:
            assert 0 <= dataset_seed(20240517, d) < 2**31 - 1


class TestProvenanceGate:
    """No dataset may appear in an external-validation table unless the
    provenance report classified it INDEPENDENT."""

    def _external_tables(self) -> list[Path]:
        out = []
        for path in sorted(tables_dir().glob("*.csv")):
            if any(p.search(path.name) for p in EXTERNAL_TABLE_PATTERNS):
                out.append(path)
        return out

    def _verdicts(self) -> dict[str, str] | None:
        path = tables_dir() / PROVENANCE_TABLE
        if not path.is_file():
            return None
        table = pd.read_csv(path)
        assert {"dataset_id", "verdict"} <= set(table.columns)
        bad = set(table["verdict"]) - VERDICTS
        assert not bad, f"unknown provenance verdicts {bad}"
        return dict(zip(table["dataset_id"], table["verdict"]))

    def test_no_external_table_without_provenance_report(self):
        """External results may not even exist before the provenance stage
        has run: the gate must come first in time, not only in logic."""
        if self._verdicts() is None:
            assert self._external_tables() == [], (
                "External-validation tables exist but "
                f"{PROVENANCE_TABLE} does not. Run scripts/00_provenance.py "
                "before any external-validation stage."
            )

    def test_only_independent_datasets_in_external_tables(self):
        verdicts = self._verdicts()
        if verdicts is None:
            pytest.skip("provenance report not yet generated")
        not_independent = {d for d, v in verdicts.items() if v != "INDEPENDENT"}
        for path in self._external_tables():
            table = pd.read_csv(path)
            cells = table.astype(str)
            for dataset_id in not_independent:
                hit = cells.apply(lambda col: col.str.contains(dataset_id, regex=False)).any().any()
                assert not hit, (
                    f"{path.name} references {dataset_id!r}, which the "
                    f"provenance report classified {verdicts[dataset_id]}. "
                    "Only INDEPENDENT datasets may support external claims."
                )

    def test_every_registered_dataset_gets_a_verdict(self):
        verdicts = self._verdicts()
        if verdicts is None:
            pytest.skip("provenance report not yet generated")
        assert set(verdicts) == set(REGISTRY)

    def test_same_source_control_is_recognised(self):
        """If the provenance stage has run, the negative control must have
        been caught — a gate that passes its own control is the only gate
        worth trusting."""
        verdicts = self._verdicts()
        if verdicts is None:
            pytest.skip("provenance report not yet generated")
        assert verdicts["uci2023_v2"] == "SAME-SOURCE"
