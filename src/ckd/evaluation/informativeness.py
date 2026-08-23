"""Benchmark Informativeness Diagnostics (BID).

Motivation
----------
Discrimination metrics answer "how well did this model separate these
patients?" They do not answer the question a benchmark exists to settle:
"does this dataset let me tell good methods from bad ones?" The two come
apart whenever a sample is easy, and this study documents three ways that
happens without being visible in the headline number.

BID is three measurements that a reported ROC-AUC cannot supply, each
targeting one way a benchmark can be uninformative:

``multivariable_lift``
    How much did modelling add over the single best column? A benchmark on
    which every method matches one raw variable cannot rank methods.

``saturation_curve`` / ``fraction_to_reach``
    How much of the training data was needed? A benchmark solved by a
    fraction of its examples cannot reward sample efficiency, and its
    reported numbers say more about the task than the learner.

``standardised_auc``
    What would this discrimination be in a population with a stated case
    mix? Two studies reporting different AUCs on different severity mixes
    are not comparable until this is applied.

Honest positioning
------------------
None of the three components is invented here. Direct standardisation is
standard epidemiological practice; learning curves are long-established in
machine learning; comparing a model against a strong single-feature
baseline is ordinary good practice. What this module contributes is their
composition into a *benchmark-level* diagnostic with explicit reporting
thresholds, motivated by failure modes demonstrated on real data, and
packaged so the checks can be run before a benchmark is trusted rather than
after a surprising result.

The thresholds in :data:`FLAG_THRESHOLDS` are **proposals**. They are set
from the behaviour of one benchmark and should be calibrated across many
before being treated as standards; :func:`flag` reports which were breached
rather than pronouncing a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

#: Proposed reporting thresholds. See the module docstring: these are
#: starting points for discussion, not established cut-offs.
FLAG_THRESHOLDS = {
    "multivariable_lift": 0.05,       # below this, modelling added little
    "fraction_to_reach_90pct": 0.25,  # below this, the task is saturated
    "standardisation_shift": 0.05,    # above this, AUC is case-mix dependent
    "sensitivity_standardisation_shift": 0.05,  # the same, at an operating point
}


def folded_auc(y: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC with the orientation chosen to favour the score.

    A single raw variable has no fitted direction, so reporting
    ``max(auc, 1 - auc)`` gives the baseline the benefit of the doubt. That
    is the conservative choice here: it makes the lift over the baseline
    *smaller*, never larger.
    """
    auc = roc_auc_score(y, score)
    return float(max(auc, 1.0 - auc))


def best_single_predictor(
    X: pd.DataFrame, y: np.ndarray
) -> tuple[str, float]:
    """The most separable single column, and its folded AUC."""
    best_name, best_auc = "", 0.0
    for column in X.columns:
        values = X[column].to_numpy(dtype=float)
        observed = np.isfinite(values)
        if observed.sum() < 10 or len(np.unique(y[observed])) < 2:
            continue
        auc = folded_auc(y[observed], values[observed])
        if auc > best_auc:
            best_name, best_auc = column, auc
    return best_name, best_auc


@dataclass(frozen=True)
class LiftResult:
    model_auc: float
    best_feature: str
    best_feature_auc: float
    lift: float
    lift_ci_low: float
    lift_ci_high: float


def multivariable_lift(
    y: np.ndarray,
    model_prob: np.ndarray,
    X: pd.DataFrame,
    n_boot: int = 2000,
    seed: int = 0,
) -> LiftResult:
    """How much the fitted model adds over the best single column.

    The interval is a paired patient-level bootstrap: both quantities are
    recomputed on each resample, so the interval reflects the uncertainty in
    the *difference* rather than in either term separately.
    """
    name, base_auc = best_single_predictor(X, y)
    model_auc = float(roc_auc_score(y, model_prob))
    feature = X[name].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    draws: list[float] = []
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True),
        ])
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        observed = np.isfinite(feature[idx])
        if observed.sum() < 10 or len(np.unique(yb[observed])) < 2:
            continue
        draws.append(
            float(roc_auc_score(yb, model_prob[idx]))
            - folded_auc(yb[observed], feature[idx][observed])
        )
    return LiftResult(
        model_auc=model_auc,
        best_feature=name,
        best_feature_auc=base_auc,
        lift=model_auc - base_auc,
        lift_ci_low=float(np.percentile(draws, 2.5)),
        lift_ci_high=float(np.percentile(draws, 97.5)),
    )


def weighted_auc(
    y: np.ndarray, score: np.ndarray, weights: np.ndarray
) -> float:
    """AUC as a weighted probability that a case outranks a control.

    Computed directly over case-control pairs so that per-patient weights
    apply exactly, with ties counted as half, matching the rank-based
    definition.
    """
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    sp, sn = score[pos], score[neg]
    wp, wn = weights[pos], weights[neg]
    comparison = (sp[:, None] > sn[None, :]).astype(float)
    comparison += 0.5 * (sp[:, None] == sn[None, :])
    pair_weight = wp[:, None] * wn[None, :]
    total = pair_weight.sum()
    if total <= 0:
        return float("nan")
    return float((comparison * pair_weight).sum() / total)


@dataclass(frozen=True)
class StandardisationResult:
    observed_auc: float
    standardised_auc: float
    shift: float
    target: dict[str, float]
    observed_mix: dict[str, float]
    effective_cases: float


def standardised_auc(
    y: np.ndarray,
    score: np.ndarray,
    severity: pd.Series,
    target_mix: dict[str, float],
) -> StandardisationResult:
    """Re-weight cases to a stated severity mix and recompute AUC.

    Direct standardisation, applied to discrimination. Cases are weighted by
    ``target_share / observed_share`` within their severity stratum;
    controls are unweighted, since the target describes the case mix. The
    result answers "what would this AUC be if the cases had looked like
    *that* population?", which is what makes two studies on different
    severity mixes comparable.

    ``effective_cases`` is Kish's effective sample size for the case
    weights; when it is small the standardised estimate rests on few
    patients and should be read with that in mind.
    """
    severity = pd.Series(severity).astype("object")
    observed = np.ones(len(y), dtype=float)
    case_mask = y == 1
    strata = severity[case_mask]
    counts = strata.value_counts(dropna=True)
    n_cases = float(counts.sum())
    observed_mix = {str(k): float(v / n_cases) for k, v in counts.items()}

    weights = np.ones(len(y), dtype=float)
    for stratum, share in target_mix.items():
        in_stratum = case_mask & (severity.to_numpy() == stratum)
        n_in = int(in_stratum.sum())
        weights[in_stratum] = 0.0 if n_in == 0 else share / (n_in / n_cases)
    # Cases whose stratum the target does not mention contribute nothing.
    unmentioned = case_mask & ~np.isin(severity.to_numpy(), list(target_mix))
    weights[unmentioned] = 0.0

    case_weights = weights[case_mask]
    effective = (
        float(case_weights.sum() ** 2 / np.sum(case_weights ** 2))
        if np.any(case_weights > 0) else 0.0
    )
    obs_auc = weighted_auc(y, score, observed)
    std_auc = weighted_auc(y, score, weights)
    return StandardisationResult(
        observed_auc=obs_auc,
        standardised_auc=std_auc,
        shift=obs_auc - std_auc,
        target=dict(target_mix),
        observed_mix=observed_mix,
        effective_cases=effective,
    )


@dataclass(frozen=True)
class OperatingPointStandardisation:
    observed_sensitivity: float
    standardised_sensitivity: float
    shift: float
    threshold: float
    effective_cases: float


def standardised_sensitivity(
    y: np.ndarray,
    score: np.ndarray,
    severity: pd.Series,
    target_mix: dict[str, float],
    threshold: float = 0.5,
) -> OperatingPointStandardisation:
    """Standardise sensitivity, rather than AUC, to a stated case mix.

    Applying :func:`standardised_auc` to this study's data barely moves the
    estimate, and the reason is instructive rather than reassuring: AUC is a
    rank statistic, so re-weighting which cases are present changes it only
    insofar as it changes the *ordering* of cases against controls. A model
    can therefore keep a near-perfect AUC while missing most of the
    early-stage cases a screening programme exists to find, because those
    cases still outrank the controls - just by less.

    Sensitivity at a fixed operating point has no such invariance: a case
    below the threshold is missed regardless of how it ranks. Standardising
    it is what exposes the case-mix dependence that AUC conceals, which is
    why both are reported and why this one carries the interpretation.
    """
    severity = pd.Series(severity).astype("object")
    case_mask = y == 1
    strata = severity[case_mask]
    counts = strata.value_counts(dropna=True)
    n_cases = float(counts.sum())

    weights = np.zeros(len(y), dtype=float)
    for stratum, share in target_mix.items():
        in_stratum = case_mask & (severity.to_numpy() == stratum)
        n_in = int(in_stratum.sum())
        weights[in_stratum] = 0.0 if n_in == 0 else share / (n_in / n_cases)

    detected = (score >= threshold).astype(float)
    observed = float(detected[case_mask].mean()) if case_mask.any() else float("nan")

    case_weights = weights[case_mask]
    total = case_weights.sum()
    standardised = (
        float((detected[case_mask] * case_weights).sum() / total)
        if total > 0 else float("nan")
    )
    effective = (
        float(total ** 2 / np.sum(case_weights ** 2))
        if np.any(case_weights > 0) else 0.0
    )
    return OperatingPointStandardisation(
        observed_sensitivity=observed,
        standardised_sensitivity=standardised,
        shift=observed - standardised,
        threshold=float(threshold),
        effective_cases=effective,
    )


def fraction_to_reach(
    fractions: np.ndarray, scores: np.ndarray, share: float = 0.90
) -> float:
    """Smallest training fraction reaching ``share`` of the achieved gain.

    Gain is measured above the 0.5 chance floor, so a benchmark that starts
    near the ceiling cannot disguise saturation as strong performance.
    Returns NaN when no fraction reaches the target.
    """
    ceiling = float(np.nanmax(scores))
    if not np.isfinite(ceiling) or ceiling <= 0.5:
        return float("nan")
    threshold = 0.5 + share * (ceiling - 0.5)
    reached = fractions[scores >= threshold]
    return float(np.min(reached)) if len(reached) else float("nan")


def flag(diagnostics: dict[str, float]) -> dict[str, bool]:
    """Which proposed thresholds a set of diagnostics breaches.

    Reports the individual flags rather than a single verdict: the point of
    the protocol is to make the failure modes visible, not to replace one
    over-trusted number with another.
    """
    out: dict[str, bool] = {}
    lift = diagnostics.get("multivariable_lift")
    if lift is not None and np.isfinite(lift):
        out["low_multivariable_lift"] = bool(
            lift < FLAG_THRESHOLDS["multivariable_lift"]
        )
    frac = diagnostics.get("fraction_to_reach_90pct")
    if frac is not None and np.isfinite(frac):
        out["saturated_by_subsample"] = bool(
            frac < FLAG_THRESHOLDS["fraction_to_reach_90pct"]
        )
    shift = diagnostics.get("standardisation_shift")
    if shift is not None and np.isfinite(shift):
        out["case_mix_dependent_auc"] = bool(
            abs(shift) > FLAG_THRESHOLDS["standardisation_shift"]
        )
    sens_shift = diagnostics.get("sensitivity_standardisation_shift")
    if sens_shift is not None and np.isfinite(sens_shift):
        out["case_mix_dependent_sensitivity"] = bool(
            abs(sens_shift) > FLAG_THRESHOLDS["sensitivity_standardisation_shift"]
        )
    return out
