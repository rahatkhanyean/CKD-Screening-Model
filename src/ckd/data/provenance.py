"""Record-level overlap checks between the internal cohort and candidates.

The question this module answers is narrow and decisive: *could a candidate
"external" dataset contain the same patients as ckd-dataset-v2.csv?*
Validating on an overlapping cohort would be worse than not validating at
all, so the answer gates Phase 3 (see ``tests/test_external.py``).

Evidence, in increasing order of specificity
--------------------------------------------
1. **Byte identity.** SHA-256 equality with the internal raw file.
2. **Cohort arithmetic and schema.** n, class balance, shared column names.
3. **Bin-edge forensics.** Our file's continuous variables were discretised
   before release. If its bin edges coincide exactly with values observed
   in a candidate's continuous data, the bins were plausibly derived from
   that data — strong evidence of shared provenance.
4. **Interval-containment matching.** Each internal patient is a vector of
   intervals (one per shared variable). A candidate row *could be* that
   patient if every shared continuous value falls inside the corresponding
   interval and the outcome matches. We compute the maximum bipartite
   matching between internal patients and candidate rows under that
   relation. Because loose intervals make coincidental compatibility easy,
   the match fraction is calibrated against a null distribution obtained by
   independently permuting each candidate column (destroying cross-variable
   structure while preserving marginals). Missing values are treated as
   compatible — permissive in exactly the direction that makes the gate
   HARDER to pass, never easier.

The verdict rule is mechanical and documented in :func:`classify`; the
thresholds are module constants, fixed before any external result exists.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from ..config import project_root

#: Numeric tolerance for "a value sits on a bin edge / inside a bin".
EPS = 1e-9

#: Shared variables used for containment matching: those that are interval
#: strings internally and numeric in uci2015. Shared *categorical* variables
#: (rbc, htn, dm, ...) are deliberately excluded: several have undocumented
#: polarity (see configs.py `uncertain` flags), and dropping constraints only
#: makes rows MORE compatible, i.e. biases toward OVERLAPPING — the safe
#: direction for a gate.
CONTAINMENT_VARIABLES: tuple[str, ...] = (
    "age", "sg", "al", "su", "bgr", "bu", "sc",
    "sod", "pot", "hemo", "pcv", "wbcc", "rbcc",
)

#: Number of column permutations for the null distribution.
N_NULL_SHUFFLES = 50

#: classify(): a match fraction at or above this, after clearing the null,
#: means the candidate can essentially reproduce the internal cohort.
SAME_SOURCE_FRACTION = 0.95

#: classify(): how far above the null the observed fraction must sit before
#: overlap is declared (both an absolute margin and a z-score are required,
#: so that a degenerate null with tiny variance cannot trigger on noise).
OVERLAP_MARGIN = 0.10
OVERLAP_Z = 5.0


@dataclass(frozen=True)
class ContainmentResult:
    """Everything the report needs about one containment comparison."""

    n_internal: int
    n_candidate: int
    variables_used: tuple[str, ...]
    match_fraction: float          # maximum bipartite matching / n_internal
    rows_with_any_partner: int     # internal rows with >= 1 compatible row
    mean_partners_per_row: float
    null_mean: float               # match fraction under column-permutation
    null_sd: float
    null_max: float


def load_bin_mappings(path: Path | None = None) -> dict[str, dict[str, tuple[float, float]]]:
    """Return ``{variable: {label: (lower, upper)}}`` from bin_mappings.json."""
    mpath = path if path is not None else project_root() / "data" / "processed" / "bin_mappings.json"
    raw = json.loads(Path(mpath).read_text(encoding="utf-8"))
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for variable, bins in raw.items():
        out[variable] = {
            b["label"]: (float(b["lower"]), float(b["upper"])) for b in bins
        }
    return out


def interval_bounds(
    labels: pd.Series, mapping: dict[str, tuple[float, float]]
) -> tuple[np.ndarray, np.ndarray]:
    """Map a column of interval labels to (lower, upper) arrays.

    Missing labels become (-inf, +inf): an unknown value is compatible with
    anything, which can only *raise* the match fraction (see module note).
    """
    lower = np.full(len(labels), -np.inf)
    upper = np.full(len(labels), np.inf)
    for i, label in enumerate(labels):
        if isinstance(label, str) and label in mapping:
            lo, up = mapping[label]
            lower[i], upper[i] = lo, up
    return lower, upper


def compatibility_matrix(
    internal: pd.DataFrame,
    internal_target: np.ndarray,
    candidate: pd.DataFrame,
    candidate_target: np.ndarray,
    mappings: dict[str, dict[str, tuple[float, float]]],
    variables: tuple[str, ...] = CONTAINMENT_VARIABLES,
) -> np.ndarray:
    """Boolean matrix: could internal patient i be candidate row j?

    True iff the outcome matches and, for every shared variable, the
    candidate's numeric value lies inside the internal patient's interval
    (closed bounds, EPS tolerance). Missing on either side is compatible.
    """
    n_i, n_c = len(internal), len(candidate)
    compat = np.equal.outer(internal_target.astype(int), candidate_target.astype(int))
    for variable in variables:
        lower, upper = interval_bounds(internal[variable], mappings[variable])
        values = pd.to_numeric(candidate[variable], errors="coerce").to_numpy(dtype=float)
        missing = np.isnan(values)
        inside = (
            (values[None, :] >= lower[:, None] - EPS)
            & (values[None, :] <= upper[:, None] + EPS)
        )
        compat &= inside | missing[None, :]
    assert compat.shape == (n_i, n_c)
    return compat


def matching_fraction(compat: np.ndarray) -> float:
    """Maximum bipartite matching size divided by the number of internal rows."""
    if not compat.any():
        return 0.0
    matching = maximum_bipartite_matching(csr_matrix(compat), perm_type="column")
    return float((matching >= 0).sum()) / compat.shape[0]


def containment_check(
    internal: pd.DataFrame,
    internal_target: np.ndarray,
    candidate: pd.DataFrame,
    candidate_target: np.ndarray,
    mappings: dict[str, dict[str, tuple[float, float]]],
    variables: tuple[str, ...] = CONTAINMENT_VARIABLES,
    n_shuffles: int = N_NULL_SHUFFLES,
    seed: int = 0,
) -> ContainmentResult:
    """Full containment analysis: observed matching plus a permutation null."""
    compat = compatibility_matrix(
        internal, internal_target, candidate, candidate_target, mappings, variables
    )
    observed = matching_fraction(compat)

    rng = np.random.default_rng(seed)
    null_fractions = np.empty(n_shuffles)
    for k in range(n_shuffles):
        shuffled = candidate.copy()
        for variable in variables:
            shuffled[variable] = (
                shuffled[variable].sample(frac=1.0, random_state=rng.integers(2**31 - 1))
                .to_numpy()
            )
        null_compat = compatibility_matrix(
            internal, internal_target, shuffled, candidate_target, mappings, variables
        )
        null_fractions[k] = matching_fraction(null_compat)

    return ContainmentResult(
        n_internal=len(internal),
        n_candidate=len(candidate),
        variables_used=tuple(variables),
        match_fraction=observed,
        rows_with_any_partner=int(compat.any(axis=1).sum()),
        mean_partners_per_row=float(compat.sum(axis=1).mean()),
        null_mean=float(null_fractions.mean()),
        null_sd=float(null_fractions.std(ddof=1)),
        null_max=float(null_fractions.max()),
    )


def bin_edge_forensics(
    mappings: dict[str, dict[str, tuple[float, float]]],
    candidate: pd.DataFrame,
    variables: tuple[str, ...] = CONTAINMENT_VARIABLES,
) -> pd.DataFrame:
    """Do our published bin edges coincide with the candidate's values?

    For each variable: the finite bin edges, how many appear verbatim among
    the candidate's observed values, and how many coincide with the
    candidate's deciles. High counts mean the bins were plausibly derived
    from that data.
    """
    rows = []
    for variable in variables:
        edges = sorted(
            {b for bounds in mappings[variable].values() for b in bounds if math.isfinite(b)}
        )
        values = pd.to_numeric(candidate[variable], errors="coerce").dropna().to_numpy(dtype=float)
        observed = np.unique(values)
        deciles = np.quantile(values, np.linspace(0.1, 0.9, 9)) if len(values) else np.array([])
        n_obs = sum(1 for e in edges if observed.size and np.min(np.abs(observed - e)) <= EPS)
        n_dec = sum(1 for e in edges if deciles.size and np.min(np.abs(deciles - e)) <= 1e-3)
        rows.append(
            {
                "variable": variable,
                "n_finite_edges": len(edges),
                "edges_matching_observed_values": n_obs,
                "edges_matching_deciles": n_dec,
            }
        )
    return pd.DataFrame(rows)


def classify(
    byte_identical: bool,
    containment: ContainmentResult | None,
) -> tuple[str, str]:
    """Mechanical verdict rule. Returns (verdict, reason).

    * Byte identity is conclusive: SAME-SOURCE.
    * A containment match fraction that clears the permutation null (by
      OVERLAP_MARGIN absolutely and OVERLAP_Z standard deviations) signals
      shared records: SAME-SOURCE at >= SAME_SOURCE_FRACTION, else
      OVERLAPPING.
    * Otherwise, or when no record-level comparison is possible (verdicts
      for such datasets rest on documentary evidence recorded separately):
      INDEPENDENT.
    """
    if byte_identical:
        return "SAME-SOURCE", "byte-identical to the internal raw file"
    if containment is None:
        return "INDEPENDENT", "no shared record-level basis; documentary evidence only"
    sd = max(containment.null_sd, 1e-12)
    z = (containment.match_fraction - containment.null_mean) / sd
    clears_null = (
        containment.match_fraction >= containment.null_mean + OVERLAP_MARGIN
        and z >= OVERLAP_Z
    )
    detail = (
        f"match fraction {containment.match_fraction:.3f} vs null "
        f"{containment.null_mean:.3f} +/- {containment.null_sd:.3f} "
        f"(max {containment.null_max:.3f}) over {N_NULL_SHUFFLES} shuffles"
    )
    if clears_null and containment.match_fraction >= SAME_SOURCE_FRACTION:
        return "SAME-SOURCE", f"records reproduce the internal cohort: {detail}"
    if clears_null:
        return "OVERLAPPING", f"record-level structure shared beyond chance: {detail}"
    return "INDEPENDENT", f"matching indistinguishable from chance: {detail}"


#: Shared categorical variables NOT used in containment matching. Agreement
#: on these across uniquely pinned pairs is held-out confirmation: they had
#: no opportunity to influence the match, so consistency is evidence of
#: record identity rather than an artifact of the matcher.
HELDOUT_CATEGORICAL: tuple[str, ...] = (
    "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane",
)


def unique_pins(compat: np.ndarray) -> list[tuple[int, int]]:
    """(internal_row, candidate_row) pairs where exactly one candidate is
    compatible — the rows the data itself pins one-to-one."""
    partners = compat.sum(axis=1)
    return [
        (int(i), int(np.flatnonzero(compat[i])[0]))
        for i in np.flatnonzero(partners == 1)
    ]


def categorical_agreement(
    internal: pd.DataFrame,
    candidate: pd.DataFrame,
    pairs: list[tuple[int, int]],
    variables: tuple[str, ...] = HELDOUT_CATEGORICAL,
) -> pd.DataFrame:
    """Cross-tabulate held-out categorical values over pinned pairs.

    For each variable, every (internal_value, candidate_value) combination
    observed across the pinned pairs is counted. A mapping is *consistent*
    when each candidate value co-occurs with exactly one internal value
    (missing candidate values excluded): e.g. ``1 <-> 'abnormal'`` in every
    pinned pair, never ``1 <-> 'no'``. The ``n_contradictions`` column
    counts pairs that break the majority mapping for their candidate value —
    0 across all variables means the held-out columns confirm identity.
    """
    rows = []
    for variable in variables:
        counts: dict[tuple[str, str], int] = {}
        n_missing = 0
        for i, j in pairs:
            iv = internal.iloc[i][variable]
            jv = candidate.iloc[j][variable]
            if pd.isna(jv):
                n_missing += 1
                continue
            key = (str(iv), str(jv))
            counts[key] = counts.get(key, 0) + 1
        majority: dict[str, str] = {}
        for (iv, jv), n in sorted(counts.items(), key=lambda kv: -kv[1]):
            majority.setdefault(jv, iv)
        n_contradictions = sum(
            n for (iv, jv), n in counts.items() if majority[jv] != iv
        )
        mapping = "; ".join(
            f"{iv}<->{jv} (n={n})" for (iv, jv), n in sorted(counts.items(), key=lambda kv: -kv[1])
        )
        rows.append(
            {
                "variable": variable,
                "n_pairs_compared": sum(counts.values()),
                "n_candidate_missing": n_missing,
                "n_contradictions": n_contradictions,
                "observed_mapping": mapping,
            }
        )
    return pd.DataFrame(rows)
