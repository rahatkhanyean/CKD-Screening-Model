"""Continuous-value recovery tests (Phase 3d).

Recovery is only legitimate because the provenance check established that
the two files describe the same patients; these tests guard both that
precondition and the correctness of the recovered values.

The central correctness property: a recovered continuous value must fall
inside the published interval its own patient carries. If it did not, the
pairing would be wrong and every downstream number meaningless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ckd.config import project_root
from ckd.data.external import load_uci2015
from ckd.data.provenance import (
    CONTAINMENT_VARIABLES,
    compatibility_matrix,
    unique_pins,
)
from ckd.data.recovery import (
    RECOVERABLE,
    bin_structure,
    binning_cost,
    blood_pressure_audit,
    imputation_audit,
    recover,
)

TABLES = project_root() / "reports" / "tables"
PROC = project_root() / "data" / "processed"


@pytest.fixture(scope="module")
def pinned(clean_result):
    source = load_uci2015()
    mappings = {
        k: {b["label"]: (float(b["lower"]), float(b["upper"])) for b in v}
        for k, v in clean_result.bin_map.items()
    }
    y = clean_result.target.to_numpy()
    compat = compatibility_matrix(
        clean_result.clean, y, source.frame, source.target.to_numpy(),
        mappings, CONTAINMENT_VARIABLES,
    )
    pins = unique_pins(compat)
    return source, pins, mappings, y


@pytest.fixture(scope="module")
def recovery(clean_result, pinned):
    source, pins, _mappings, y = pinned
    return recover(clean_result.clean, y, source.frame, pins)


class TestPrecondition:
    def test_recovery_requires_same_source_verdict(self):
        """Guard the reason recovery is allowed at all."""
        path = TABLES / "table_24_provenance.csv"
        if not path.is_file():
            pytest.skip("provenance report not generated")
        verdicts = pd.read_csv(path).set_index("dataset_id")["verdict"].to_dict()
        assert verdicts["uci2015"] == "SAME-SOURCE"

    def test_pins_are_injective(self, pinned):
        _source, pins, _m, _y = pinned
        assert len({j for _i, j in pins}) == len(pins)


class TestRecoveredValuesAreConsistent:
    def test_every_recovered_value_lies_inside_its_published_bin(
        self, clean_result, pinned, recovery
    ):
        """The property that makes the pairing trustworthy."""
        _source, _pins, mappings, _y = pinned
        violations = []
        for variable in RECOVERABLE:
            values = recovery.frame[variable].to_numpy(float)
            labels = clean_result.clean.iloc[recovery.internal_positions][
                variable
            ].to_numpy()
            for value, label in zip(values, labels):
                if not np.isfinite(value):
                    continue
                bounds = mappings[variable].get(str(label))
                if bounds is None:
                    continue
                lower, upper = bounds
                if not (lower - 1e-9 <= value <= upper + 1e-9):
                    violations.append((variable, label, value))
        assert not violations, violations[:10]

    def test_targets_agree_between_files(self, pinned, recovery):
        source, pins, _m, _y = pinned
        source_target = source.target.to_numpy()[[j for _i, j in pins]]
        assert np.array_equal(recovery.target, source_target)

    def test_provenance_flags_match_value_presence(self, recovery):
        for variable in RECOVERABLE:
            finite = np.isfinite(recovery.frame[variable].to_numpy(float))
            flags = recovery.provenance[variable].to_numpy()
            assert list(flags[finite]) == ["observed"] * int(finite.sum())
            assert set(flags[~finite]) <= {"imputed_in_v2"}


class TestBinningCost:
    def test_like_for_like_uses_observed_cells_only(self, X, recovery):
        costs = binning_cost(X, recovery)
        for _, row in costs.iterrows():
            assert row["n_observed"] > 0
            assert row["n_observed"] + row["n_imputed_in_v2"] == len(recovery.target)

    def test_serum_creatinine_is_the_worst_loss(self, X, recovery):
        """The reference-run finding: binning destroys most of the signal in
        the single most diagnostic analyte. If this changes, the report's
        discussion of it must change too."""
        costs = binning_cost(X, recovery).set_index("variable")
        assert costs.loc["sc", "binning_cost"] > 0.2
        assert costs["binning_cost"].idxmax() == "sc"

    def test_most_variables_lose_almost_nothing(self, X, recovery):
        costs = binning_cost(X, recovery)
        small = costs[costs["variable"] != "sc"]["binning_cost"].abs()
        assert (small < 0.1).all(), costs


class TestImputationAudit:
    def test_every_variable_imputed_with_a_single_constant(self, clean_result, recovery):
        audit = imputation_audit(clean_result.clean, recovery)
        assert (audit["n_distinct_labels_assigned"] == 1).all()
        assert (audit["assigned_label_share"] == 1.0).all()

    def test_missingness_is_outcome_associated(self, clean_result, recovery):
        """The property that makes constant imputation consequential rather
        than cosmetic: the cells the release filled in are concentrated in
        one outcome class."""
        audit = imputation_audit(clean_result.clean, recovery)
        strong = audit[audit["n_unobserved_in_source"] >= 30]
        assert len(strong) >= 3
        assert (strong["missingness_odds_ratio_ckd"] > 3).all()


class TestBinStructure:
    def test_first_creatinine_bin_spans_normal_to_severe(self, clean_result, recovery):
        structure = bin_structure(clean_result.clean, recovery, "sc").set_index(
            "published_bin"
        )
        widest = structure["continuous_span"].idxmax()
        row = structure.loc[widest]
        # Normal creatinine is ~0.6-1.2 mg/dL; >3 is severe impairment.
        assert row["continuous_min"] < 1.0
        assert row["continuous_max"] > 3.0
        # And it is the only bin whose outcome mix is not pure.
        assert 0.2 < row["ckd_fraction"] < 0.8


class TestBloodPressureRecovery:
    def test_binary_flag_is_a_diastolic_threshold(self, clean_result, recovery, pinned):
        """Resolves an `uncertain` variable: 'bp (Diastolic)' behaves as a
        'diastolic >= 80 mmHg' indicator."""
        source, _pins, _m, _y = pinned
        audit = blood_pressure_audit(clean_result.clean, recovery, source.frame)
        flag = audit[audit["encoded_column"] == "bp (Diastolic)"].set_index(
            "encoded_level"
        )
        assert flag.loc["0", "diastolic_max_mmhg"] <= 70
        assert flag.loc["1", "diastolic_median_mmhg"] >= 80

    def test_bp_limit_middle_level_is_exactly_eighty(self, clean_result, recovery, pinned):
        source, _pins, _m, _y = pinned
        audit = blood_pressure_audit(clean_result.clean, recovery, source.frame)
        limit = audit[audit["encoded_column"] == "bp limit"].set_index("encoded_level")
        assert limit.loc["1", "diastolic_min_mmhg"] == 80
        assert limit.loc["1", "diastolic_max_mmhg"] == 80


class TestRecoveredArtefact:
    def test_recovered_csv_matches_the_module(self, recovery):
        path = PROC / "ckd_recovered_continuous.csv"
        if not path.is_file():
            pytest.skip("run scripts/11_continuous_recovery.py")
        saved = pd.read_csv(path)
        assert len(saved) == len(recovery.target)
        assert np.array_equal(saved["target"].to_numpy(), recovery.target)
        for variable in RECOVERABLE:
            a = saved[variable].to_numpy(float)
            b = recovery.frame[variable].to_numpy(float)
            assert np.allclose(a, b, equal_nan=True)
