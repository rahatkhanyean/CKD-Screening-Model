"""Feature importance and its stability across resampling.

Three complementary views are produced, because no single importance measure is
trustworthy on 200 patients:

``permutation``
    Model-agnostic. Measures the drop in ROC-AUC when one column is shuffled in
    the held-out outer test fold. Computed on test data and used only for
    *reporting*, never for model or feature selection, so it does not
    contaminate the validation.

``shap``
    Local attributions aggregated to a global mean absolute value. Exact and
    fast for tree ensembles; the linear explainer is used for logistic
    regression. Skipped, with a recorded reason, for models where no exact
    explainer applies.

``coefficient``
    For linear models only, the standardised log-odds coefficient, which is the
    only one of the three with a direct effect-size interpretation.

Stability is then measured across the 25 outer folds: how often a feature
reaches the top-k, how much its rank moves, and how strongly the fold-level
rankings agree (Kendall's W). A feature that is "important" in one fold and
absent in the next is not a finding - it is noise, and it is reported as such.

None of these quantities are causal. They describe how a particular fitted
model uses a column, given the other columns present.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.inspection import permutation_importance


@dataclass
class ImportanceResult:
    per_fold: pd.DataFrame       # long: fold x feature x method x value
    stability: pd.DataFrame      # one row per feature x method
    notes: list[str]


def permutation_importance_fold(
    estimator, X_test: pd.DataFrame, y_test: np.ndarray, seed: int, n_repeats: int = 20
) -> pd.Series:
    """Permutation importance on one outer test fold, scored by ROC-AUC."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = permutation_importance(
            estimator, X_test, y_test,
            scoring="roc_auc", n_repeats=n_repeats, random_state=seed, n_jobs=1,
        )
    return pd.Series(r.importances_mean, index=list(X_test.columns))


def shap_importance_fold(estimator, X_test: pd.DataFrame, model_name: str) -> tuple[pd.Series | None, str]:
    """Mean absolute SHAP value per feature, or ``(None, reason)`` if unavailable."""
    try:
        import shap
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, f"shap not importable: {exc}"

    # Reach through the pipeline: transform the data exactly as the model sees it.
    try:
        pipe = estimator
        steps = dict(pipe.named_steps)
        clf = steps["clf"]
        Xt = X_test
        for name, step in pipe.steps[:-1]:
            Xt = step.transform(Xt)
        Xt = pd.DataFrame(np.asarray(Xt), columns=list(X_test.columns))
    except Exception as exc:
        return None, f"could not reach the estimator through the pipeline: {exc}"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if model_name in ("random_forest", "xgboost"):
                explainer = shap.TreeExplainer(clf)
                vals = explainer.shap_values(Xt, check_additivity=False)
                arr = np.asarray(vals)
                if arr.ndim == 3:
                    # (n, features, classes) or (classes, n, features): take the
                    # positive class in whichever layout is present.
                    arr = arr[..., 1] if arr.shape[-1] == 2 else arr[1]
                return pd.Series(np.abs(arr).mean(axis=0), index=list(X_test.columns)), ""
            if model_name == "logreg":
                explainer = shap.LinearExplainer(clf, Xt)
                arr = np.asarray(explainer.shap_values(Xt))
                return pd.Series(np.abs(arr).mean(axis=0), index=list(X_test.columns)), ""
    except Exception as exc:
        return None, f"explainer failed for {model_name}: {exc}"

    return None, f"no exact SHAP explainer is applicable to {model_name}"


def linear_coefficients(estimator, columns: list[str]) -> pd.Series | None:
    """Standardised logistic-regression coefficients, if the model is linear."""
    clf = estimator.named_steps.get("clf")
    coef = getattr(clf, "coef_", None)
    if coef is None:
        return None
    return pd.Series(np.asarray(coef).ravel(), index=columns)


def kendalls_w(rank_matrix: np.ndarray) -> float:
    """Kendall's coefficient of concordance across raters (folds).

    1.0 = every fold ranks the features identically; 0.0 = no agreement.
    """
    m, n = rank_matrix.shape  # m raters, n items
    if m < 2 or n < 2:
        return float("nan")
    rank_sums = rank_matrix.sum(axis=0)
    mean_rs = rank_sums.mean()
    s = float(((rank_sums - mean_rs) ** 2).sum())
    denom = m**2 * (n**3 - n) / 12.0
    return float(s / denom) if denom > 0 else float("nan")


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


def summarise_stability(per_fold: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Rank stability, top-k selection frequency and concordance, per method."""
    rows: list[dict] = []
    for method, mgrp in per_fold.groupby("method"):
        wide = mgrp.pivot_table(index="fold_id", columns="feature", values="importance")
        wide = wide.dropna(axis=0, how="all")
        if wide.empty:
            continue
        features = list(wide.columns)

        # Rank 1 = most important within each fold (ties averaged).
        ranks = np.vstack([
            rankdata(-np.nan_to_num(wide.loc[f].to_numpy(dtype=float), nan=-np.inf))
            for f in wide.index
        ])
        rank_df = pd.DataFrame(ranks, index=wide.index, columns=features)

        top_sets = [set(rank_df.loc[f].nsmallest(top_k).index) for f in rank_df.index]
        pairwise = [jaccard(a, b) for a, b in combinations(top_sets, 2)]
        w = kendalls_w(ranks)

        for feat in features:
            appearances = sum(1 for s in top_sets if feat in s)
            vals = wide[feat].to_numpy(dtype=float)
            rows.append({
                "method": method,
                "feature": feat,
                "mean_importance": float(np.nanmean(vals)),
                "sd_importance": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else np.nan,
                "median_importance": float(np.nanmedian(vals)),
                "positive_in_folds": int(np.sum(vals > 0)),
                "n_folds": int(len(vals)),
                "mean_rank": float(rank_df[feat].mean()),
                "sd_rank": float(rank_df[feat].std(ddof=1)) if len(rank_df) > 1 else np.nan,
                "best_rank": float(rank_df[feat].min()),
                "worst_rank": float(rank_df[feat].max()),
                f"top{top_k}_frequency": appearances / len(top_sets),
                f"top{top_k}_count": appearances,
                "kendalls_w_all_features": w,
                f"mean_pairwise_jaccard_top{top_k}": float(np.nanmean(pairwise)) if pairwise else np.nan,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["method", "mean_importance"], ascending=[True, False])
    return out.reset_index(drop=True)
