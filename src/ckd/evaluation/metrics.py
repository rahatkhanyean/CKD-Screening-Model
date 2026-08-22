"""Discrimination, classification and calibration metrics.

Conventions
-----------
* ``y_true`` is 1 for CKD (the condition being screened for) and 0 otherwise.
* ``y_prob`` is the predicted probability of CKD.
* Sensitivity/recall, PPV (precision) and NPV are reported for the CKD class,
  because in a screening setting the cost of a missed case dominates.

Calibration slope and intercept follow the standard logistic recalibration
framework: a logistic model is refitted on the linear predictor
``logit(y_prob)``.

* **Slope** is the coefficient of ``logit(y_prob)`` in
  ``logit(P(y=1)) = a + b * logit(y_prob)``. b = 1 is perfect;
  b < 1 indicates over-fitted, too-extreme predictions;
  b > 1 indicates under-fitted, too-conservative predictions.
* **Intercept (calibration-in-the-large)** is ``a`` in
  ``logit(P(y=1)) = a + 1 * logit(y_prob)``, i.e. fitted with the linear
  predictor as an *offset*. a = 0 is perfect; a > 0 means risks are
  systematically under-estimated.

Both are fitted by direct maximum likelihood (Newton / L-BFGS on the exact
log-likelihood) rather than by a penalised solver, because any penalty would
bias the very quantity being measured.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from scipy.special import expit
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

#: Probabilities are clipped before taking logits so that a confident 0 or 1
#: prediction does not produce an infinite linear predictor.
EPS = 1e-6


def logit(p: np.ndarray | Sequence[float], eps: float = EPS) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


#: Above this magnitude the recalibration slope is treated as unidentified.
#: A slope of, say, 400 does not mean "400 times too conservative"; it means
#: the likelihood was still climbing when the optimiser stopped.
MAX_IDENTIFIABLE_SLOPE = 25.0


def _both_classes_present(y_true: np.ndarray) -> bool:
    return len(np.unique(y_true)) == 2


def separates_perfectly(y_true: np.ndarray, score: np.ndarray) -> bool:
    """True when ``score`` orders the two classes with no overlap at all.

    Complete separation means the logistic maximum-likelihood estimate does
    not exist, so any calibration slope reported from it would be an artefact
    of where the optimiser happened to stop.
    """
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    if len(np.unique(y)) < 2:
        return False
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return False
    return bool(neg.max() < pos.min() or pos.max() < neg.min())


@dataclass
class ThresholdMetrics:
    """Metrics that depend on a decision threshold."""

    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    sensitivity: float
    specificity: float
    precision: float
    npv: float
    f1: float
    balanced_accuracy: float
    accuracy: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def threshold_metrics(
    y_true: Sequence[int], y_prob: Sequence[float], threshold: float
) -> ThresholdMetrics:
    """Confusion matrix and all threshold-dependent metrics."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    pred = (p >= threshold).astype(int)

    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))

    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    f1 = (2 * prec * sens / (prec + sens)) if (prec and sens and prec + sens > 0) else 0.0
    bal = np.nanmean([sens, spec])
    acc = (tp + tn) / len(y) if len(y) else np.nan

    return ThresholdMetrics(
        threshold=float(threshold),
        tp=tp, fp=fp, tn=tn, fn=fn,
        sensitivity=float(sens),
        specificity=float(spec),
        precision=float(prec) if prec == prec else float("nan"),
        npv=float(npv) if npv == npv else float("nan"),
        f1=float(f1),
        balanced_accuracy=float(bal),
        accuracy=float(acc),
    )


def _newton_logistic(
    design: np.ndarray, y: np.ndarray, beta0: np.ndarray,
    max_iter: int = 60, tol: float = 1e-9,
) -> np.ndarray | None:
    """Unpenalised logistic MLE by Newton-Raphson (IRLS).

    Used instead of a general-purpose optimiser because the calibration
    statistics are recomputed for every bootstrap resample: this converges in a
    handful of exact Newton steps and is roughly two orders of magnitude faster
    than a quasi-Newton solver, with no penalty term that could bias the very
    quantity being measured.

    Returns ``None`` if the Hessian becomes numerically singular, which happens
    under complete separation - a case where the coefficient is genuinely not
    identified rather than merely hard to compute.
    """
    beta = beta0.astype(float).copy()
    for _ in range(max_iter):
        mu = expit(design @ beta)
        w = mu * (1.0 - mu)
        if not np.any(w > 1e-12):
            return None
        grad = design.T @ (y - mu)
        hess = (design * w[:, None]).T @ design
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(step)):
            return None
        # Step-halving keeps the iteration stable when the data are close to
        # separated and a full Newton step would overshoot badly.
        scale = 1.0
        if np.max(np.abs(step)) > 10.0:
            scale = 10.0 / np.max(np.abs(step))
        beta = beta + scale * step
        if np.max(np.abs(scale * step)) < tol:
            break
    return beta if np.all(np.isfinite(beta)) else None


def calibration_slope_intercept(
    y_true: Sequence[int], y_prob: Sequence[float]
) -> tuple[float, float]:
    """Return ``(slope, intercept)`` from logistic recalibration.

    The slope comes from an unpenalised two-parameter logistic fit of ``y`` on
    ``logit(p)``; the intercept comes from a one-parameter fit with ``logit(p)``
    as a fixed offset. Returns ``(nan, nan)`` when only one outcome class is
    present, since neither quantity is identified in that case.
    """
    y = np.asarray(y_true, dtype=float)
    lp = logit(y_prob)

    if not _both_classes_present(y.astype(int)):
        return float("nan"), float("nan")
    if np.allclose(lp, lp[0]):
        # No variation in the linear predictor: the slope is not identified.
        return float("nan"), float("nan")

    # --- slope: logit(P(y=1)) = a + b * lp ---
    # Under COMPLETE SEPARATION the maximum-likelihood slope does not exist:
    # the likelihood increases without bound as the coefficient grows, and any
    # optimiser will simply return whatever large number it stopped at. That is
    # not a calibration measurement, so it is reported as undefined rather than
    # as a spuriously precise value. This is a real occurrence on this dataset,
    # where some models separate the classes perfectly out of fold.
    if separates_perfectly(y.astype(int), lp):
        slope = float("nan")
    else:
        design = np.column_stack([np.ones_like(lp), lp])
        beta = _newton_logistic(design, y, np.array([0.0, 1.0]))
        slope = float(beta[1]) if beta is not None else float("nan")
        # A slope this extreme indicates quasi-separation: the estimate is
        # numerically finite but scientifically meaningless.
        if np.isfinite(slope) and abs(slope) > MAX_IDENTIFIABLE_SLOPE:
            slope = float("nan")

    # --- intercept: logit(P(y=1)) = a + 1 * lp, i.e. lp as a fixed offset ---
    a = 0.0
    ok = False
    for _ in range(60):
        mu = expit(a + lp)
        w = mu * (1.0 - mu)
        denom = float(np.sum(w))
        if denom <= 1e-12:
            break
        step = float(np.sum(y - mu)) / denom
        if not np.isfinite(step):
            break
        if abs(step) > 10.0:
            step = 10.0 * np.sign(step)
        a += step
        if abs(step) < 1e-9:
            ok = True
            break
    else:
        ok = True
    intercept = float(a) if ok and np.isfinite(a) else float("nan")

    return slope, intercept


def expected_calibration_error(
    y_true: Sequence[int], y_prob: Sequence[float], n_bins: int = 10
) -> float:
    """Equal-width binned expected calibration error."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def all_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
    n_calibration_bins: int = 10,
    extra_thresholds: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute the full metric set for one set of predictions."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)

    out: dict[str, float] = {"n": int(len(y)), "n_positive": int(y.sum())}

    if _both_classes_present(y):
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")

    out["brier"] = float(brier_score_loss(y, p))
    # Brier skill score against the observed prevalence, so the number is
    # comparable across resamples with slightly different prevalence.
    prevalence = float(y.mean())
    ref = float(np.mean((y - prevalence) ** 2))
    out["brier_skill_score"] = float(1.0 - out["brier"] / ref) if ref > 0 else float("nan")

    slope, intercept = calibration_slope_intercept(y, p)
    out["calibration_slope"] = slope
    out["calibration_intercept"] = intercept
    out["ece"] = expected_calibration_error(y, p, n_bins=n_calibration_bins)

    tm = threshold_metrics(y, p, threshold)
    out.update({f"{k}": v for k, v in tm.as_dict().items()})

    for name, thr in (extra_thresholds or {}).items():
        if thr is None or not np.isfinite(thr):
            continue
        tm2 = threshold_metrics(y, p, float(thr))
        for k, v in tm2.as_dict().items():
            out[f"{name}_{k}"] = v

    return out


#: Metric columns reported in the headline tables, in presentation order.
HEADLINE_METRICS: tuple[str, ...] = (
    "roc_auc",
    "pr_auc",
    "sensitivity",
    "specificity",
    "precision",
    "npv",
    "f1",
    "balanced_accuracy",
    "brier",
    "calibration_slope",
    "calibration_intercept",
    "ece",
)

#: Human-readable labels for figures and tables.
METRIC_LABELS: dict[str, str] = {
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "sensitivity": "Sensitivity (recall, CKD)",
    "specificity": "Specificity",
    "precision": "Precision (PPV)",
    "npv": "Negative predictive value",
    "f1": "F1 score",
    "balanced_accuracy": "Balanced accuracy",
    "accuracy": "Accuracy",
    "brier": "Brier score",
    "brier_skill_score": "Brier skill score",
    "calibration_slope": "Calibration slope",
    "calibration_intercept": "Calibration intercept",
    "ece": "Expected calibration error",
}

#: Metrics where a lower value is better.
LOWER_IS_BETTER: frozenset[str] = frozenset({"brier", "ece"})
