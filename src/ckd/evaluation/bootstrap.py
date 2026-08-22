"""Aggregation of cross-validated predictions and uncertainty quantification.

Two distinct sources of uncertainty are reported separately, because they
answer different questions and conflating them is a common way to overstate
precision.

1. **Cross-validation partition variability** - the spread of a metric across
   the 25 outer test folds (5 folds x 5 repeats). This says how much the result
   depends on *which* patients happened to fall in the test fold. It does not
   describe uncertainty about the population.

2. **Patient-level bootstrap** - resampling patients with replacement from the
   pooled out-of-fold predictions. This is the closer analogue of a sampling
   confidence interval.

Neither is a substitute for external validation. Both are computed on the same
200 patients, so both are optimistic about how the model would behave in a new
population. The bootstrap interval in particular is narrower than a true
sampling interval would be, because the same patients contributed to fitting
the models whose predictions are being resampled. This is stated in the report
rather than being quietly ignored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import all_metrics

GROUP_KEYS = ["config", "model", "calibration"]


def pooled_predictions(preds: pd.DataFrame) -> pd.DataFrame:
    """Average each patient's out-of-fold probability across repeats.

    In every repeat each patient appears exactly once as a test case, so
    averaging over repeats yields one probability per patient per cell. This is
    the standard "repeated cross-validation" point estimate.
    """
    return (
        preds.groupby(GROUP_KEYS + ["sample_index"], as_index=False)
        .agg(
            y_true=("y_true", "first"),
            y_prob=("y_prob", "mean"),
            screening_threshold=("screening_threshold", "mean"),
            n_repeats=("y_prob", "size"),
        )
        .sort_values(GROUP_KEYS + ["sample_index"])
        .reset_index(drop=True)
    )


def metrics_per_repeat(
    preds: pd.DataFrame, primary_threshold: float = 0.5, n_bins: int = 10
) -> pd.DataFrame:
    """Metrics computed on each repeat's complete set of out-of-fold predictions.

    Within a repeat every patient is predicted exactly once, so this pools the
    5 outer folds into one set of 200 predictions and evaluates it. Repeating
    over the 5 repeats gives the partition-variability distribution.
    """
    rows: list[dict] = []
    for keys, grp in preds.groupby(GROUP_KEYS + ["repeat"]):
        thr = float(grp["screening_threshold"].mean())
        m = all_metrics(
            grp["y_true"].to_numpy(),
            grp["y_prob"].to_numpy(),
            threshold=primary_threshold,
            n_calibration_bins=n_bins,
            extra_thresholds={"screening": thr},
        )
        rows.append(dict(zip(GROUP_KEYS + ["repeat"], keys)) | m)
    return pd.DataFrame(rows)


def metrics_per_fold(
    preds: pd.DataFrame, primary_threshold: float = 0.5, n_bins: int = 10
) -> pd.DataFrame:
    """Metrics computed within each individual outer test fold (n approx. 40).

    Fold-level metrics are noisy at this sample size (a 40-patient fold holds
    roughly 14 non-CKD patients, so specificity moves in steps of about 0.07).
    Reported for completeness and for the fold-level spread figures.
    """
    rows: list[dict] = []
    for keys, grp in preds.groupby(GROUP_KEYS + ["repeat", "fold"]):
        thr = float(grp["screening_threshold"].iloc[0])
        m = all_metrics(
            grp["y_true"].to_numpy(),
            grp["y_prob"].to_numpy(),
            threshold=primary_threshold,
            n_calibration_bins=min(n_bins, 5),
            extra_thresholds={"screening": thr},
        )
        rows.append(dict(zip(GROUP_KEYS + ["repeat", "fold"], keys)) | m)
    return pd.DataFrame(rows)


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    screening_threshold: float,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Stratified patient-level bootstrap of every metric.

    Stratifying by outcome keeps the CKD/non-CKD ratio fixed across resamples,
    which avoids resamples in which a metric is undefined and keeps the
    interval focused on estimation error rather than on prevalence variation.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)

    point = all_metrics(
        y_true, y_prob, threshold=threshold, n_calibration_bins=n_bins,
        extra_thresholds={"screening": screening_threshold},
    )

    draws: list[dict] = []
    for _ in range(n_boot):
        take = np.concatenate(
            [
                rng.choice(pos_idx, size=len(pos_idx), replace=True),
                rng.choice(neg_idx, size=len(neg_idx), replace=True),
            ]
        )
        draws.append(
            all_metrics(
                y_true[take], y_prob[take], threshold=threshold,
                n_calibration_bins=n_bins,
                extra_thresholds={"screening": screening_threshold},
            )
        )

    boot = pd.DataFrame(draws)
    lo_q, hi_q = (1.0 - ci) / 2.0, 1.0 - (1.0 - ci) / 2.0

    rows: list[dict] = []
    for metric, value in point.items():
        if metric in boot.columns and pd.api.types.is_numeric_dtype(boot[metric]):
            col = boot[metric].to_numpy(dtype=float)
            finite = col[np.isfinite(col)]
            lo = float(np.quantile(finite, lo_q)) if finite.size else float("nan")
            hi = float(np.quantile(finite, hi_q)) if finite.size else float("nan")
            se = float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
        else:
            lo = hi = se = float("nan")
        rows.append(
            {
                "metric": metric,
                "estimate": float(value) if isinstance(value, (int, float)) else value,
                "ci_low": lo,
                "ci_high": hi,
                "bootstrap_se": se,
                "n_bootstrap": n_boot,
                "ci_level": ci,
            }
        )
    return pd.DataFrame(rows)


def summarise_across_repeats(per_repeat: pd.DataFrame) -> pd.DataFrame:
    """Mean, SD and min/max of every metric across repeats, per cell."""
    numeric = per_repeat.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [c for c in numeric if c != "repeat"]
    agg = per_repeat.groupby(GROUP_KEYS)[numeric].agg(["mean", "std", "min", "max"])
    agg.columns = [f"{a}__{b}" for a, b in agg.columns]
    return agg.reset_index()


def calibration_curve_points(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10, strategy: str = "quantile"
) -> pd.DataFrame:
    """Binned observed-vs-predicted risk, with per-bin Wilson intervals.

    Quantile binning is the default: with 200 patients, equal-width bins leave
    several bins empty or nearly empty, which produces visually dramatic but
    statistically meaningless calibration plots.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if strategy == "quantile":
        edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 3:
            edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, len(edges) - 2)
        n_eff = len(edges) - 1
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)
        n_eff = n_bins

    rows: list[dict] = []
    for b in range(n_eff):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        obs = float(y_true[mask].mean())
        pred = float(y_prob[mask].mean())
        lo, hi = _wilson(y_true[mask].sum(), n)
        rows.append(
            {
                "bin": b,
                "n": n,
                "mean_predicted": pred,
                "observed_rate": obs,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    return pd.DataFrame(rows)


def _wilson(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval - stable for small n and proportions near 0 or 1."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))
