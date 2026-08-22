"""Continuous-value recovery for the analysed cohort (Phase 3d).

The provenance check (``ckd.data.provenance``) established that every
patient in the analysed file is present in the continuous-valued 2015 UCI
release. That makes a question answerable which is normally out of reach:
**what did pre-discretisation actually cost?**

Because the two files describe the *same patients*, the comparison is
within-cohort. There is no population shift, no case-mix difference and no
sampling variation between the binned and continuous versions - the only
thing that differs is the representation. Any difference in measured
performance is therefore attributable to the representation itself.

Recovery is restricted to patients the containment relation pins to
exactly one source record (``unique_pins``); those assignments are the ones
supported by evidence rather than by an arbitrary choice among compatible
partners. Every recovered cell carries a provenance flag:

``observed``      the source records a value; recovery is a lookup.
``imputed_in_v2`` the source records *nothing*, yet the analysed file
                  carries a value. Nothing can be recovered, and the fact
                  that the released file supplied a number anyway is itself
                  reported (see :func:`imputation_audit`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score

#: Variables recoverable as continuous quantities: present in both files,
#: interval-encoded here and numeric in the source.
RECOVERABLE: tuple[str, ...] = (
    "age", "sg", "al", "su", "bgr", "bu", "sc",
    "sod", "pot", "hemo", "pcv", "wbcc", "rbcc",
)

#: Source column holding continuous diastolic blood pressure. The analysed
#: file replaces it with a binary flag plus an undocumented ordinal
#: ('bp limit'), so recovery here resolves a variable that
#: ``features/configs.py`` flags as uncertain.
BP_SOURCE = "bp"


@dataclass(frozen=True)
class RecoveryResult:
    """Recovered continuous data plus its provenance."""

    frame: pd.DataFrame            # continuous values, NaN where unrecoverable
    provenance: pd.DataFrame       # same shape; 'observed' / 'imputed_in_v2'
    target: np.ndarray
    internal_positions: list[int]  # row positions in the analysed file
    source_positions: list[int]    # matched row positions in the source


def recover(
    internal_clean: pd.DataFrame,
    target: np.ndarray,
    source: pd.DataFrame,
    pins: list[tuple[int, int]],
    variables: tuple[str, ...] = RECOVERABLE,
) -> RecoveryResult:
    """Build the recovered continuous frame for uniquely pinned patients."""
    ii = [i for i, _ in pins]
    jj = [j for _, j in pins]
    values: dict[str, np.ndarray] = {}
    flags: dict[str, list[str]] = {}
    for variable in variables:
        column = pd.to_numeric(source.iloc[jj][variable], errors="coerce").to_numpy(float)
        values[variable] = column
        flags[variable] = ["observed" if np.isfinite(v) else "imputed_in_v2"
                           for v in column]
    frame = pd.DataFrame(values, index=range(len(pins)))
    provenance = pd.DataFrame(flags, index=range(len(pins)))
    return RecoveryResult(frame, provenance, target[ii], ii, jj)


def _folded_auc(y: np.ndarray, x: np.ndarray) -> float:
    auc = roc_auc_score(y, x)
    return float(max(auc, 1.0 - auc))


def binning_cost(
    binned: pd.DataFrame,
    recovery: RecoveryResult,
    variables: tuple[str, ...] = RECOVERABLE,
) -> pd.DataFrame:
    """Univariate separability, binned versus continuous, like for like.

    Both columns are evaluated on **the same patients** - those whose value
    the source actually observed - so the difference isolates the cost of
    the interval encoding rather than confounding it with the analysed
    file's imputation of the remainder.
    """
    rows = []
    y = recovery.target
    for variable in variables:
        continuous = recovery.frame[variable].to_numpy(float)
        observed = np.isfinite(continuous)
        if observed.sum() < 10 or len(set(y[observed])) < 2:
            continue
        binned_values = binned[variable].to_numpy(float)[recovery.internal_positions]
        auc_binned = _folded_auc(y[observed], binned_values[observed])
        auc_continuous = _folded_auc(y[observed], continuous[observed])
        auc_binned_all = (
            _folded_auc(y, binned_values) if len(set(y)) > 1 else float("nan")
        )
        rows.append({
            "variable": variable,
            "n_observed": int(observed.sum()),
            "n_imputed_in_v2": int((~observed).sum()),
            "auc_binned_observed_only": round(auc_binned, 4),
            "auc_continuous_observed_only": round(auc_continuous, 4),
            "binning_cost": round(auc_continuous - auc_binned, 4),
            "auc_binned_all_pinned": round(auc_binned_all, 4),
            "attenuation_from_imputed_rows": round(auc_binned - auc_binned_all, 4),
        })
    return pd.DataFrame(rows).sort_values("binning_cost", ascending=False)


def imputation_audit(
    internal_clean: pd.DataFrame,
    recovery: RecoveryResult,
    variables: tuple[str, ...] = RECOVERABLE,
) -> pd.DataFrame:
    """What did the released file put where the source has nothing?

    For each variable: how many pinned patients have no source value, which
    interval label the analysed file assigns them, whether that label is a
    single constant, and whether being unobserved is associated with the
    outcome (Fisher exact). Constant imputation of outcome-associated
    missingness is a property of the released data that no downstream
    analysis can detect from the file alone.
    """
    rows = []
    y = recovery.target
    for variable in variables:
        continuous = recovery.frame[variable].to_numpy(float)
        missing = ~np.isfinite(continuous)
        if missing.sum() == 0:
            continue
        labels = internal_clean.iloc[recovery.internal_positions][variable].to_numpy()
        assigned = pd.Series(labels[missing]).value_counts()
        a = int((missing & (y == 1)).sum())
        b = int((~missing & (y == 1)).sum())
        c = int((missing & (y == 0)).sum())
        d = int((~missing & (y == 0)).sum())
        odds, p = fisher_exact([[a, b], [c, d]])
        rows.append({
            "variable": variable,
            "n_unobserved_in_source": int(missing.sum()),
            "n_distinct_labels_assigned": int(assigned.size),
            "assigned_label": str(assigned.index[0]),
            "assigned_label_share": round(float(assigned.iloc[0] / missing.sum()), 4),
            "unobserved_ckd": a,
            "unobserved_non_ckd": c,
            "missingness_odds_ratio_ckd": (round(float(odds), 3)
                                           if np.isfinite(odds) else float("inf")),
            "missingness_fisher_p": float(f"{p:.3e}"),
        })
    return pd.DataFrame(rows).sort_values("missingness_odds_ratio_ckd", ascending=False)


def bin_structure(
    internal_clean: pd.DataFrame,
    recovery: RecoveryResult,
    variable: str,
) -> pd.DataFrame:
    """What each published interval of one variable actually contains.

    Used to show, concretely, what a coarse bin destroys: the continuous
    span it covers and the outcome mix inside it.
    """
    continuous = recovery.frame[variable].to_numpy(float)
    labels = internal_clean.iloc[recovery.internal_positions][variable].to_numpy()
    y = recovery.target
    rows = []
    for label in pd.unique(labels):
        mask = (labels == label) & np.isfinite(continuous)
        if mask.sum() == 0:
            continue
        values = continuous[mask]
        rows.append({
            "variable": variable,
            "published_bin": str(label),
            "n": int(mask.sum()),
            "continuous_min": float(values.min()),
            "continuous_max": float(values.max()),
            "continuous_span": round(float(values.max() - values.min()), 3),
            "ckd_fraction": round(float(y[mask].mean()), 3),
        })
    return pd.DataFrame(rows).sort_values("continuous_min")


def blood_pressure_audit(
    internal_clean: pd.DataFrame, recovery: RecoveryResult, source: pd.DataFrame
) -> pd.DataFrame:
    """Resolve the undocumented 'bp (Diastolic)' / 'bp limit' coding.

    The analysed file ships a binary flag and an undocumented ordinal in
    place of a measured diastolic pressure. With the source values
    recovered, the mapping can be read off directly rather than inferred.
    """
    bp = pd.to_numeric(source.iloc[recovery.source_positions][BP_SOURCE],
                       errors="coerce").to_numpy(float)
    rows = []
    for column in ("bp (Diastolic)", "bp limit"):
        if column not in internal_clean.columns:
            continue
        levels = internal_clean.iloc[recovery.internal_positions][column].to_numpy()
        for level in sorted(pd.unique(levels), key=str):
            mask = (levels == level) & np.isfinite(bp)
            if mask.sum() == 0:
                continue
            values = bp[mask]
            rows.append({
                "encoded_column": column,
                "encoded_level": str(level),
                "n": int(mask.sum()),
                "diastolic_min_mmhg": float(values.min()),
                "diastolic_max_mmhg": float(values.max()),
                "diastolic_median_mmhg": float(np.median(values)),
            })
    return pd.DataFrame(rows)
