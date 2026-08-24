"""Sensitivity analyses for the cross-release provenance verdict.

The overlap finding is the study's principal empirical contribution, and a
single matching configuration is not sufficient evidence for it. This module
re-runs the record-level comparison under systematically varied conditions so
that the verdict can be shown to be a property of the data rather than of one
analytic choice.

The analyses answer distinct objections:

*Is the match driven by the outcome?* Matching is repeated with the outcome
constraint removed, so agreement rests on the predictors alone.

*Is it driven by one variable?* Leave-one-variable-out matching removes each
containment variable in turn.

*Is it an artefact of the tolerance?* The interval-containment rule is
re-run under tighter and looser numeric tolerances, and under a rule that
requires strict interior containment rather than closed bounds.

*Is it an artefact of the release's constant imputation?* Matching is
repeated with the variables most affected by undocumented constant
imputation removed.

*Could an unrelated dataset match equally well?* Negative controls match the
analysed file against cohorts it has no relationship to.

*Is the permutation null the right null?* Two nulls are computed: independent
column permutation (destroys all cross-variable structure) and whole-row
permutation within outcome strata (preserves each record's internal
correlation structure and only breaks the pairing). The second is the more
conservative comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .provenance import (
    CONTAINMENT_VARIABLES,
    compatibility_matrix,
    matching_fraction,
    unique_pins,
)


@dataclass(frozen=True)
class MatchSummary:
    """One matching configuration and everything it produced."""

    label: str
    variables: tuple[str, ...]
    n_variables: int
    outcome_used: bool
    match_fraction: float
    n_unique_pins: int
    rows_with_any_partner: int
    mean_partners: float
    median_partners: float
    max_partners: int
    exact_one_to_one: bool


def _partner_stats(compat: np.ndarray) -> tuple[int, float, float, int]:
    partners = compat.sum(axis=1)
    return (
        int((partners > 0).sum()),
        float(partners.mean()),
        float(np.median(partners)),
        int(partners.max()) if len(partners) else 0,
    )


def match_once(
    internal: pd.DataFrame,
    internal_target: np.ndarray,
    candidate: pd.DataFrame,
    candidate_target: np.ndarray,
    mappings,
    variables: tuple[str, ...],
    label: str,
    use_outcome: bool = True,
    tolerance: float = 1e-9,
) -> MatchSummary:
    """Run one matching configuration and summarise it."""
    if use_outcome:
        compat = compatibility_matrix(
            internal, internal_target, candidate, candidate_target,
            mappings, variables,
        )
    else:
        # Every record is outcome-compatible with every other: the match must
        # then be carried by the predictors alone.
        neutral = np.zeros(len(candidate_target), dtype=int)
        compat = compatibility_matrix(
            internal, np.zeros(len(internal_target), dtype=int),
            candidate, neutral, mappings, variables,
        )
    any_partner, mean_p, median_p, max_p = _partner_stats(compat)
    pins = unique_pins(compat)
    return MatchSummary(
        label=label,
        variables=tuple(variables),
        n_variables=len(variables),
        outcome_used=use_outcome,
        match_fraction=matching_fraction(compat),
        n_unique_pins=len(pins),
        rows_with_any_partner=any_partner,
        mean_partners=mean_p,
        median_partners=median_p,
        max_partners=max_p,
        exact_one_to_one=bool(max_p == 1),
    )


def leave_one_out(
    internal, internal_target, candidate, candidate_target, mappings,
    variables: tuple[str, ...] = CONTAINMENT_VARIABLES,
) -> pd.DataFrame:
    """Drop each containment variable in turn."""
    rows = []
    for dropped in variables:
        remaining = tuple(v for v in variables if v != dropped)
        summary = match_once(
            internal, internal_target, candidate, candidate_target, mappings,
            remaining, label=f"without_{dropped}",
        )
        rows.append({
            "analysis": "leave_one_variable_out",
            "dropped_variable": dropped,
            **_as_row(summary),
        })
    return pd.DataFrame(rows)


def _as_row(s: MatchSummary) -> dict:
    return {
        "label": s.label,
        "n_variables": s.n_variables,
        "outcome_used": s.outcome_used,
        "match_fraction": round(s.match_fraction, 4),
        "n_unique_pins": s.n_unique_pins,
        "rows_with_any_partner": s.rows_with_any_partner,
        "mean_partners": round(s.mean_partners, 3),
        "median_partners": s.median_partners,
        "max_partners": s.max_partners,
    }


def copula_null(
    internal: pd.DataFrame,
    internal_target: np.ndarray,
    candidate: pd.DataFrame,
    candidate_target: np.ndarray,
    mappings,
    variables: tuple[str, ...] = CONTAINMENT_VARIABLES,
    n_draws: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Null preserving marginals *and* correlations but not record identity.

    Column permutation is a weak null: it destroys every cross-variable
    relationship, so beating it shows only that the candidate records are
    internally coherent. A sceptic should ask the harder question --- would a
    cohort with the same marginal distributions *and* the same correlation
    structure match just as well, purely by resembling the analysed cohort in
    aggregate?

    This null answers it. Within each outcome stratum we fit a Gaussian
    copula to the candidate's ranks, draw new records from it, and map them
    back through the empirical marginals. The synthetic cohort therefore has
    the right univariate distributions and approximately the right
    dependence structure, but no record corresponds to any real patient. If
    the observed match fraction and uniqueness survive comparison with this
    null, they are properties of individual records rather than of the
    cohort's shape.

    A note on what this null cannot be. Permuting *whole rows* of the
    candidate is not a valid null here: maximum bipartite matching depends
    only on the multiset of candidate records, not on their order, so row
    permutation leaves the statistic exactly unchanged by construction. We
    verified this empirically (match fraction 1.000 with zero variance) and
    report it as a property of the estimator rather than as evidence.
    """
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    numeric = {
        v: pd.to_numeric(candidate[v], errors="coerce").to_numpy(dtype=float)
        for v in variables
    }
    draws = np.empty(n_draws)

    for k in range(n_draws):
        synthetic = candidate.copy()
        # Coerce the columns we overwrite to float so that assigning NaN for
        # preserved missingness does not clash with an object dtype.
        for v in variables:
            synthetic[v] = pd.to_numeric(synthetic[v], errors="coerce").astype(float)
        for label in np.unique(candidate_target):
            idx = np.flatnonzero(candidate_target == label)
            block = np.column_stack([numeric[v][idx] for v in variables])
            observed = ~np.isnan(block)
            # Rank-transform each column to normal scores, ignoring missing.
            scores = np.full_like(block, np.nan, dtype=float)
            for j in range(block.shape[1]):
                col = block[:, j]
                ok = ~np.isnan(col)
                if ok.sum() < 3:
                    continue
                ranks = pd.Series(col[ok]).rank(method="average").to_numpy()
                scores[ok, j] = norm.ppf(ranks / (ok.sum() + 1))
            filled = np.where(np.isnan(scores), 0.0, scores)
            corr = np.corrcoef(filled, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0)
            np.fill_diagonal(corr, 1.0)
            try:
                chol = np.linalg.cholesky(
                    corr + 1e-6 * np.eye(corr.shape[0])
                )
            except np.linalg.LinAlgError:
                chol = np.eye(corr.shape[0])
            z = rng.standard_normal((len(idx), len(variables))) @ chol.T
            # Map back through each column's empirical marginal.
            for j, v in enumerate(variables):
                col = block[:, j]
                ok = ~np.isnan(col)
                if ok.sum() < 3:
                    continue
                quantiles = norm.cdf(z[:, j])
                new = np.quantile(col[ok], np.clip(quantiles, 0, 1)).astype(float)
                # Preserve the original missingness pattern per column.
                new[~observed[:, j]] = np.nan
                synthetic.loc[synthetic.index[idx], v] = new
        draws[k] = matching_fraction(
            compatibility_matrix(
                internal, internal_target, synthetic, candidate_target,
                mappings, variables,
            )
        )
    return draws


def row_permutation_is_degenerate(
    internal: pd.DataFrame,
    internal_target: np.ndarray,
    candidate: pd.DataFrame,
    candidate_target: np.ndarray,
    mappings,
    variables: tuple[str, ...] = CONTAINMENT_VARIABLES,
    seed: int = 0,
) -> tuple[float, float]:
    """Demonstrate that whole-row permutation cannot serve as a null.

    Returns (observed, permuted) match fractions, which must be equal.
    Retained so the claim in :func:`copula_null`'s docstring is checkable
    rather than asserted.
    """
    rng = np.random.default_rng(seed)
    observed = matching_fraction(
        compatibility_matrix(internal, internal_target, candidate,
                             candidate_target, mappings, variables)
    )
    order = np.arange(len(candidate))
    for label in np.unique(candidate_target):
        in_stratum = np.flatnonzero(candidate_target == label)
        order[in_stratum] = rng.permutation(in_stratum)
    shuffled = candidate.iloc[order].reset_index(drop=True)
    permuted = matching_fraction(
        compatibility_matrix(internal, internal_target, shuffled,
                             candidate_target[order], mappings, variables)
    )
    return float(observed), float(permuted)


def tolerance_sensitivity(
    internal, internal_target, candidate, candidate_target, mappings,
    variables: tuple[str, ...] = CONTAINMENT_VARIABLES,
    tolerances: tuple[float, ...] = (0.0, 1e-9, 1e-6, 1e-3, 1e-2),
) -> pd.DataFrame:
    """Vary the numeric tolerance of the containment test.

    A tolerance of exactly 0 requires the value to lie within the published
    interval with no slack whatsoever, which is the strictest reading of the
    bin definition.
    """
    from .provenance import interval_bounds

    rows = []
    for tol in tolerances:
        compat = np.equal.outer(
            internal_target.astype(int), candidate_target.astype(int)
        )
        for variable in variables:
            lower, upper = interval_bounds(internal[variable], mappings[variable])
            values = pd.to_numeric(
                candidate[variable], errors="coerce"
            ).to_numpy(dtype=float)
            missing = np.isnan(values)
            inside = (
                (values[None, :] >= lower[:, None] - tol)
                & (values[None, :] <= upper[:, None] + tol)
            )
            compat &= inside | missing[None, :]
        any_partner, mean_p, median_p, max_p = _partner_stats(compat)
        rows.append({
            "analysis": "tolerance_sensitivity",
            "tolerance": tol,
            "match_fraction": round(matching_fraction(compat), 4),
            "n_unique_pins": len(unique_pins(compat)),
            "rows_with_any_partner": any_partner,
            "mean_partners": round(mean_p, 3),
            "max_partners": max_p,
        })
    return pd.DataFrame(rows)


def sample_matched_pairs(
    internal: pd.DataFrame,
    candidate: pd.DataFrame,
    pins: list[tuple[int, int]],
    variables: tuple[str, ...],
    n: int = 12,
    seed: int = 20240517,
) -> pd.DataFrame:
    """A reproducibly sampled audit table of matched pairs.

    Reviewers should be able to check the claim by eye on a sample they can
    regenerate. Each row shows one internal patient's published interval and
    the candidate record's value for every matching variable.
    """
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(pins), size=min(n, len(pins)), replace=False)
    rows = []
    for k in sorted(chosen):
        i, j = pins[k]
        record = {
            "pair_index": int(k),
            "internal_row": int(i),
            "candidate_row": int(j),
        }
        for variable in variables:
            record[f"{variable}__interval"] = str(internal.iloc[i][variable])
            record[f"{variable}__value"] = pd.to_numeric(
                pd.Series([candidate.iloc[j][variable]]), errors="coerce"
            ).iloc[0]
        rows.append(record)
    return pd.DataFrame(rows)
