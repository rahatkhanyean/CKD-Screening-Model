"""Case-mix ("spectrum effect") analysis.

Why this analysis exists
------------------------
The valid configurations on this dataset reach discrimination close to the
ceiling. Before treating that as good news, it has to be explained. The
explanation is visible in the data: 111 of the 128 CKD patients are staged s3
to s5, i.e. moderate-to-severe disease, and haemoglobin alone separates the
groups almost completely (CKD 6.1-14.55 g/dL versus non-CKD 11.95-16.5 g/dL).

That is the profile of a **case-control-like contrast between advanced disease
and comparatively healthy controls**, not of a consecutive screening series in
which most true cases would be early and asymptomatic. Discrimination measured
on such a sample is systematically optimistic for screening use, because the
cases that a screening instrument would actually have to catch - early, mild,
biochemically near-normal - are barely represented.

This module quantifies that directly by re-evaluating the *same* out-of-fold
predictions on clinically defined subgroups.

Important
---------
``stage`` is used **only** to partition patients for evaluation. It never
enters any model as a predictor; the predictions being analysed here were
produced by pipelines from which ``stage`` is programmatically excluded. Using
a post-diagnosis variable to define an evaluation subgroup is standard
case-mix analysis and is not leakage - but it does mean these subgroup results
are exploratory and post hoc, and they are labelled as such.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import all_metrics

#: Subgroup definitions. Each maps to the set of CKD stages retained; non-CKD
#: patients are always retained, because they are the comparison group.
SUBGROUPS: dict[str, tuple[str, ...]] = {
    "all_patients": ("s1", "s2", "s3", "s4", "s5"),
    "early_ckd_only": ("s1", "s2"),
    "moderate_ckd_only": ("s3",),
    "advanced_ckd_only": ("s4", "s5"),
}

SUBGROUP_LABELS = {
    "all_patients": "All patients (as sampled)",
    "early_ckd_only": "Early CKD (s1-s2) vs non-CKD",
    "moderate_ckd_only": "Moderate CKD (s3) vs non-CKD",
    "advanced_ckd_only": "Advanced CKD (s4-s5) vs non-CKD",
}


def spectrum_analysis(
    pooled: pd.DataFrame,
    stage_by_index: pd.Series,
    threshold: float = 0.5,
    group_keys: tuple[str, ...] = ("config", "model", "calibration"),
) -> pd.DataFrame:
    """Re-evaluate pooled out-of-fold predictions within clinical subgroups.

    Parameters
    ----------
    pooled:
        Output of :func:`ckd.evaluation.bootstrap.pooled_predictions`.
    stage_by_index:
        CKD stage label indexed by ``sample_index``.
    """
    rows: list[dict] = []
    stage = stage_by_index.astype(str)

    for keys, grp in pooled.groupby(list(group_keys)):
        g = grp.copy()
        g["stage"] = g["sample_index"].map(stage)
        for name, stages in SUBGROUPS.items():
            keep = (g["y_true"] == 0) | (g["stage"].isin(stages))
            sub = g[keep]
            n_pos = int((sub["y_true"] == 1).sum())
            n_neg = int((sub["y_true"] == 0).sum())
            if n_pos < 3 or n_neg < 3:
                continue
            m = all_metrics(
                sub["y_true"].to_numpy(), sub["y_prob"].to_numpy(), threshold=threshold
            )
            rows.append(
                dict(zip(group_keys, keys))
                | {
                    "subgroup": name,
                    "subgroup_label": SUBGROUP_LABELS[name],
                    "n_ckd": n_pos,
                    "n_non_ckd": n_neg,
                    "prevalence": round(n_pos / (n_pos + n_neg), 4),
                    "roc_auc": m["roc_auc"],
                    "pr_auc": m["pr_auc"],
                    "sensitivity": m["sensitivity"],
                    "specificity": m["specificity"],
                    "npv": m["npv"],
                    "precision": m["precision"],
                    "brier": m["brier"],
                }
            )
    return pd.DataFrame(rows)


def separability_report(X: pd.DataFrame, y: np.ndarray, features: list[str]) -> pd.DataFrame:
    """Univariate discrimination and class-range overlap for each predictor.

    A predictor whose class ranges barely overlap will separate the sample
    almost perfectly on its own. Quantifying this makes it obvious whether a
    high multivariable AUC reflects a genuinely learned combination or simply a
    dataset in which the two groups are already far apart.
    """
    from sklearn.metrics import roc_auc_score

    rows: list[dict] = []
    for col in features:
        v = X[col].to_numpy(dtype=float)
        ok = np.isfinite(v)
        if ok.sum() < 10 or len(np.unique(v[ok])) < 2:
            continue
        auc = roc_auc_score(y[ok], v[ok])
        pos, neg = v[ok & (y == 1)], v[ok & (y == 0)]
        lo = max(pos.min(), neg.min())
        hi = min(pos.max(), neg.max())
        overlap_width = max(0.0, hi - lo)
        total_width = max(pos.max(), neg.max()) - min(pos.min(), neg.min())
        # Two complementary views of overlap. The width fraction says how much
        # of the value scale the two classes share; the patient fraction says
        # how many people actually sit in that shared region. They diverge
        # sharply for binary variables, where a single shared level can hold
        # most of the cohort while the shared *width* is zero.
        n_overlap = int(((v >= lo) & (v <= hi) & ok).sum())
        rows.append(
            {
                "feature": col,
                "univariate_auc": round(float(max(auc, 1 - auc)), 4),
                "ckd_min": float(pos.min()), "ckd_max": float(pos.max()),
                "non_ckd_min": float(neg.min()), "non_ckd_max": float(neg.max()),
                "overlap_fraction_of_range": round(
                    float(overlap_width / total_width) if total_width > 0 else 1.0, 4
                ),
                "n_patients_in_overlap": n_overlap,
                "fraction_patients_in_overlap": round(float(n_overlap / int(ok.sum())), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("univariate_auc", ascending=False).reset_index(drop=True)
