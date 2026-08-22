"""Unit tests for the record-level overlap machinery (Phase 0.3).

The gate is only worth trusting if it demonstrably catches what it must
catch. These tests build synthetic candidate datasets where the truth is
known by construction:

* a *planted re-release* — continuous values drawn inside the internal
  cohort's own bins — must be flagged (match fraction ~1, far above null);
* *independent* data with the same marginals but shuffled cross-variable
  structure must NOT clear the null;
* the mechanical ``classify`` rule must map those evidence patterns to
  SAME-SOURCE / OVERLAPPING / INDEPENDENT as documented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ckd.data.provenance import (
    ContainmentResult,
    bin_edge_forensics,
    categorical_agreement,
    classify,
    compatibility_matrix,
    containment_check,
    interval_bounds,
    matching_fraction,
    unique_pins,
)

VARIABLES = ("v1", "v2", "v3")

MAPPINGS = {
    "v1": {"< 10": (-np.inf, 10.0), "10 - 20": (10.0, 20.0), ">= 20": (20.0, np.inf)},
    "v2": {"< 1": (-np.inf, 1.0), "1 - 2": (1.0, 2.0), ">= 2": (2.0, np.inf)},
    "v3": {"< 100": (-np.inf, 100.0), "100 - 200": (100.0, 200.0), ">= 200": (200.0, np.inf)},
}


def synthetic_internal(n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Internal-style frame of interval labels with a class-linked pattern."""
    target = (rng.random(n) < 0.5).astype(int)
    rows = []
    for t in target:
        if t:
            rows.append({"v1": "< 10", "v2": ">= 2", "v3": "100 - 200"})
        else:
            rows.append({"v1": ">= 20", "v2": "< 1", "v3": "< 100"})
    return pd.DataFrame(rows), target


def planted_rerelease(
    internal: pd.DataFrame, target: np.ndarray, rng: np.random.Generator
) -> tuple[pd.DataFrame, np.ndarray]:
    """Continuous rows drawn INSIDE each internal row's bins: a re-release."""
    def draw(bounds):
        lo, up = bounds
        lo = up - 5.0 if not np.isfinite(lo) else lo
        up = lo + 5.0 if not np.isfinite(up) else up
        return rng.uniform(lo, up)

    rows = [
        {v: draw(MAPPINGS[v][internal.iloc[i][v]]) for v in VARIABLES}
        for i in range(len(internal))
    ]
    return pd.DataFrame(rows), target.copy()


def independent_candidate(n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Same marginal flavour, but values opposing the internal class pattern."""
    target = (rng.random(n) < 0.5).astype(int)
    rows = []
    for t in target:
        if t:  # positive cases here look like the internal NEGATIVES
            rows.append({"v1": rng.uniform(25, 40), "v2": rng.uniform(0, 0.5),
                         "v3": rng.uniform(0, 50)})
        else:
            rows.append({"v1": rng.uniform(0, 5), "v2": rng.uniform(3, 5),
                         "v3": rng.uniform(120, 180)})
    return pd.DataFrame(rows), target


class TestIntervalBounds:
    def test_labels_map_to_bounds_and_missing_is_unbounded(self):
        lower, upper = interval_bounds(pd.Series(["< 10", "10 - 20", None]), MAPPINGS["v1"])
        assert upper[0] == 10.0 and not np.isfinite(lower[0])
        assert (lower[1], upper[1]) == (10.0, 20.0)
        assert not np.isfinite(lower[2]) and not np.isfinite(upper[2])


class TestCompatibility:
    def test_planted_rows_are_compatible_on_the_diagonal(self):
        rng = np.random.default_rng(0)
        internal, target = synthetic_internal(30, rng)
        candidate, ctarget = planted_rerelease(internal, target, rng)
        compat = compatibility_matrix(internal, target, candidate, ctarget,
                                      MAPPINGS, VARIABLES)
        assert bool(np.diag(compat).all())
        assert matching_fraction(compat) == 1.0

    def test_class_mismatch_blocks_compatibility(self):
        rng = np.random.default_rng(1)
        internal, target = synthetic_internal(10, rng)
        candidate, ctarget = planted_rerelease(internal, target, rng)
        compat = compatibility_matrix(internal, target, candidate, 1 - ctarget,
                                      MAPPINGS, VARIABLES)
        assert not np.diag(compat).any()

    def test_missing_candidate_value_is_permissive(self):
        rng = np.random.default_rng(2)
        internal, target = synthetic_internal(5, rng)
        candidate, ctarget = planted_rerelease(internal, target, rng)
        candidate.loc[:, "v2"] = np.nan
        compat = compatibility_matrix(internal, target, candidate, ctarget,
                                      MAPPINGS, VARIABLES)
        assert bool(np.diag(compat).all())


class TestContainmentVerdicts:
    def test_planted_rerelease_is_caught(self):
        rng = np.random.default_rng(3)
        internal, target = synthetic_internal(40, rng)
        candidate, ctarget = planted_rerelease(internal, target, rng)
        result = containment_check(internal, target, candidate, ctarget,
                                   MAPPINGS, VARIABLES, n_shuffles=20, seed=0)
        verdict, reason = classify(byte_identical=False, containment=result)
        assert result.match_fraction == 1.0
        assert verdict == "SAME-SOURCE", reason

    def test_independent_candidate_clears_the_gate(self):
        rng = np.random.default_rng(4)
        internal, target = synthetic_internal(40, rng)
        candidate, ctarget = independent_candidate(60, rng)
        result = containment_check(internal, target, candidate, ctarget,
                                   MAPPINGS, VARIABLES, n_shuffles=20, seed=0)
        verdict, _ = classify(byte_identical=False, containment=result)
        assert verdict == "INDEPENDENT"
        assert result.match_fraction <= result.null_mean + 0.10

    def test_byte_identity_dominates_everything(self):
        verdict, reason = classify(byte_identical=True, containment=None)
        assert verdict == "SAME-SOURCE"
        assert "byte-identical" in reason

    def test_no_record_basis_is_documentary_independent(self):
        verdict, reason = classify(byte_identical=False, containment=None)
        assert verdict == "INDEPENDENT"
        assert "documentary" in reason

    def test_partial_overlap_is_flagged_overlapping(self):
        """Half the candidate rows are planted, half are independent: the
        match fraction clears the null but stays below the SAME-SOURCE bar."""
        rng = np.random.default_rng(5)
        internal, target = synthetic_internal(40, rng)
        planted, ptarget = planted_rerelease(internal.iloc[:20], target[:20], rng)
        indep, itarget = independent_candidate(40, rng)
        candidate = pd.concat([planted, indep], ignore_index=True)
        ctarget = np.concatenate([ptarget, itarget])
        result = containment_check(internal, target, candidate, ctarget,
                                   MAPPINGS, VARIABLES, n_shuffles=20, seed=0)
        verdict, _ = classify(byte_identical=False, containment=result)
        assert verdict == "OVERLAPPING"


class TestHeldOutAgreement:
    def test_unique_pins_returns_only_singleton_rows(self):
        compat = np.array([
            [True, False, False],   # pinned to 0
            [True, True, False],    # two partners: not pinned
            [False, False, True],   # pinned to 2
        ])
        assert unique_pins(compat) == [(0, 0), (2, 2)]

    def test_consistent_mapping_has_zero_contradictions(self):
        internal = pd.DataFrame({"flag": ["1", "0", "1", "0"]})
        candidate = pd.DataFrame({"flag": ["yes", "no", "yes", np.nan]})
        pairs = [(0, 0), (1, 1), (2, 2), (3, 3)]
        table = categorical_agreement(internal, candidate, pairs, ("flag",))
        row = table.iloc[0]
        assert row["n_contradictions"] == 0
        assert row["n_candidate_missing"] == 1
        assert row["n_pairs_compared"] == 3

    def test_contradictions_are_counted(self):
        internal = pd.DataFrame({"flag": ["1", "1", "0"]})
        candidate = pd.DataFrame({"flag": ["yes", "yes", "yes"]})
        pairs = [(0, 0), (1, 1), (2, 2)]
        table = categorical_agreement(internal, candidate, pairs, ("flag",))
        # Majority mapping is 1<->yes; the 0<->yes pair breaks it.
        assert table.iloc[0]["n_contradictions"] == 1


class TestForensics:
    def test_edges_derived_from_candidate_values_are_detected(self):
        candidate = pd.DataFrame({
            "v1": [10.0, 20.0, 5.0, 30.0],
            "v2": [1.0, 2.0, 0.5, 3.0],
            "v3": [100.0, 200.0, 50.0, 300.0],
        })
        forensics = bin_edge_forensics(MAPPINGS, candidate, VARIABLES)
        by_var = forensics.set_index("variable")
        # Every finite edge of MAPPINGS appears verbatim in the candidate.
        for v in VARIABLES:
            assert (
                by_var.loc[v, "edges_matching_observed_values"]
                == by_var.loc[v, "n_finite_edges"]
            )

    def test_unrelated_values_do_not_match_edges(self):
        rng = np.random.default_rng(6)
        candidate = pd.DataFrame({
            "v1": rng.uniform(0.001, 9.999, 50) + 0.0001,
            "v2": rng.uniform(0.001, 0.999, 50) + 0.0001,
            "v3": rng.uniform(0.001, 99.999, 50) + 0.0001,
        })
        forensics = bin_edge_forensics(MAPPINGS, candidate, VARIABLES)
        assert int(forensics["edges_matching_observed_values"].sum()) == 0
