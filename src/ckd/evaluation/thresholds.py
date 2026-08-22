"""Decision-threshold policies.

Two operating points are reported throughout the study:

``primary`` (prespecified)
    A fixed probability threshold of 0.50, declared in ``config/experiment.yaml``
    before any result was computed. It needs no data and therefore cannot be
    optimistically tuned.

``screening`` (transparently selected, training-fold only)
    The lowest threshold whose sensitivity reaches the configured target on
    **inner cross-validated predictions of the outer training fold**. It is
    recomputed independently in every outer fold and never sees outer-test
    outcomes. This mirrors how a threshold would have to be chosen in practice
    and keeps the outer test fold untouched.

A threshold chosen on the evaluation data itself would be optimistically
biased; that variant is deliberately not implemented.
"""

from __future__ import annotations

import numpy as np


def threshold_for_target_sensitivity(
    y_true, y_prob, target_sensitivity: float = 0.90
) -> float:
    """Highest threshold that still achieves at least ``target_sensitivity``.

    Choosing the *highest* such threshold maximises specificity subject to the
    sensitivity constraint. Falls back to 0.0 (flag everyone) when the target
    is unreachable, which is the safe direction for a screening rule.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    if y.sum() == 0:
        return 0.5

    # Candidate thresholds: every distinct predicted probability, plus 0 and 1.
    candidates = np.unique(np.concatenate([[0.0], p, [1.0]]))
    best = 0.0
    for thr in candidates:
        pred = (p >= thr).astype(int)
        tp = np.sum((pred == 1) & (y == 1))
        fn = np.sum((pred == 0) & (y == 1))
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        if sens >= target_sensitivity:
            best = float(thr)
        else:
            # Sensitivity is monotone non-increasing in the threshold, so once
            # the target is lost it cannot be regained.
            break
    return best


def youden_threshold(y_true, y_prob) -> float:
    """Threshold maximising Youden's J (sensitivity + specificity - 1).

    Provided for completeness and used only on training-fold predictions.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    candidates = np.unique(np.concatenate([[0.0], p, [1.0]]))
    best_thr, best_j = 0.5, -np.inf
    for thr in candidates:
        pred = (p >= thr).astype(int)
        tp = np.sum((pred == 1) & (y == 1))
        fn = np.sum((pred == 0) & (y == 1))
        tn = np.sum((pred == 0) & (y == 0))
        fp = np.sum((pred == 1) & (y == 0))
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        j = sens + spec - 1.0
        if j > best_j:
            best_j, best_thr = j, float(thr)
    return best_thr


def net_benefit(y_true, y_prob, threshold: float) -> float:
    """Decision-curve net benefit at one threshold probability.

    ``NB = TP/n - FP/n * (pt / (1 - pt))``. Reported as an exploratory
    supplement: it expresses the trade-off between detecting a case and the
    burden of an unnecessary referral in units of true positives.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    n = len(y)
    if n == 0 or threshold >= 1.0:
        return float("nan")
    pred = (p >= threshold).astype(int)
    tp = np.sum((pred == 1) & (y == 1))
    fp = np.sum((pred == 1) & (y == 0))
    w = threshold / (1.0 - threshold)
    return float(tp / n - (fp / n) * w)


def treat_all_net_benefit(y_true, threshold: float) -> float:
    """Net benefit of referring every patient, the standard reference strategy."""
    y = np.asarray(y_true, dtype=int)
    n = len(y)
    if n == 0 or threshold >= 1.0:
        return float("nan")
    prevalence = y.mean()
    w = threshold / (1.0 - threshold)
    return float(prevalence - (1.0 - prevalence) * w)
