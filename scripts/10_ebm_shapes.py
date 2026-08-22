"""Stage 10: readable low-cost rule — EBM shape functions (N8 / Phase 1d).

The study is titled "explainable", but the model the prespecified rule
selects is an RBF SVM, which is not. This stage supplies the honest
alternative at the same cost: the low-cost **EBM**, a pure GAM
(``interactions=0``), whose per-variable shape functions a clinician can
read and challenge.

Two artefacts:

* ``table_28_ebm_vs_svm.csv`` — discrimination and calibration for the
  low-cost EBM against the low-cost SVM, from the existing pooled
  out-of-fold predictions (no refitting needed for the comparison);
* ``table_28_ebm_shape_functions.csv`` plus ``fig_r13_ebm_shapes`` — the
  fitted shape functions, averaged over the outer training folds so that
  what is plotted is the typical fitted rule rather than one arbitrary
  partition.

Shape functions are extracted per outer fold, using the same fold seeds
and the same pipeline builder as stage 3, so the curves shown correspond
to the models whose performance is reported.

Usage
-----
    python scripts/10_ebm_shapes.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402

from ckd.config import ensure_dirs, load_config  # noqa: E402
from ckd.data.clean import clean_dataset, feature_matrix  # noqa: E402
from ckd.evaluation.bootstrap import pooled_predictions  # noqa: E402
from ckd.evaluation.metrics import calibration_slope_intercept  # noqa: E402
from ckd.evaluation.plots import apply_style, save  # noqa: E402
from ckd.features.configs import get_config  # noqa: E402
from ckd.models.nested_cv import _fold_seed  # noqa: E402
from ckd.models.pipeline import (  # noqa: E402
    assert_pipeline_is_clean,
    build_pipeline,
    pipeline_feature_names,
)
from ckd.models.zoo import get_model  # noqa: E402

CONFIG = "low_cost_model"
COMPARISON_CELLS = [("ebm", "none"), ("ebm", "isotonic"),
                    ("svm", "none"), ("svm", "isotonic")]


def comparison_table(pooled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, calibration in COMPARISON_CELLS:
        cell = pooled[
            (pooled["config"] == CONFIG)
            & (pooled["model"] == model)
            & (pooled["calibration"] == calibration)
        ]
        if cell.empty:
            continue
        y = cell["y_true"].to_numpy()
        prob = cell["y_prob"].to_numpy()
        slope, intercept = calibration_slope_intercept(y, prob)
        pred = prob >= 0.5
        rows.append({
            "config": CONFIG, "model": model, "calibration": calibration,
            "roc_auc": float(roc_auc_score(y, prob)),
            "brier": float(brier_score_loss(y, prob)),
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "sensitivity": float((pred & (y == 1)).sum() / (y == 1).sum()),
            "specificity": float((~pred & (y == 0)).sum() / (y == 0).sum()),
            "readable_shape_functions": model == "ebm",
        })
    return pd.DataFrame(rows)


def extract_shapes(X, y, settings_seed, outer_folds, inner_folds, repeats, scoring):
    """Fit the low-cost EBM on each outer training fold; return its shape
    functions on a common grid per feature (mean and SD across folds)."""
    features = list(get_config(CONFIG).features)
    X_config = X.loc[:, features]
    spec = get_model("ebm")
    per_fold: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {f: [] for f in features}

    for repeat in range(repeats):
        skf = StratifiedKFold(n_splits=outer_folds, shuffle=True,
                              random_state=_fold_seed(settings_seed, repeat, salt=1))
        for fold, (train_idx, _test_idx) in enumerate(skf.split(X_config, y)):
            seed = _fold_seed(settings_seed, repeat, fold, salt=2)
            pipe = build_pipeline("ebm", CONFIG, seed)
            assert_pipeline_is_clean(pipe, CONFIG)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                search = GridSearchCV(
                    pipe, param_grid=spec.param_grid, scoring=scoring,
                    cv=StratifiedKFold(n_splits=inner_folds, shuffle=True,
                                       random_state=seed),
                    n_jobs=1, refit=True,
                )
                search.fit(X_config.iloc[train_idx], y[train_idx])
            best = search.best_estimator_
            ebm = best.named_steps["clf"]
            # The EBM receives a transformed numpy array, so it names terms
            # positionally (feature_0000, ...). The pipeline's own selector
            # is the authority on what those positions mean.
            ordered = pipeline_feature_names(best)
            assert len(ordered) == len(ebm.term_names_), (
                f"term/feature count mismatch: {len(ebm.term_names_)} terms "
                f"vs {len(ordered)} columns"
            )
            explanation = ebm.explain_global()
            for index, feature in enumerate(ordered):
                data = explanation.data(index)
                if data.get("type") != "univariate":
                    continue  # interactions=0, so none expected
                per_fold[feature].append(
                    (np.asarray(data["names"], dtype=float),
                     np.asarray(data["scores"], dtype=float))
                )
        print(f"  repeat {repeat + 1}/{repeats} done")

    rows = []
    for feature, curves in per_fold.items():
        if not curves:
            continue
        lo = min(c[0].min() for c in curves)
        hi = max(c[0].max() for c in curves)
        grid = np.linspace(lo, hi, 60)
        resampled = []
        for edges, scores in curves:
            # EBM returns bin edges (len n+1) with one score per bin.
            centres = (edges[:-1] + edges[1:]) / 2 if len(edges) == len(scores) + 1 else edges
            resampled.append(np.interp(grid, centres, scores[: len(centres)]))
        stacked = np.vstack(resampled)
        for g, mean, sd in zip(grid, stacked.mean(axis=0), stacked.std(axis=0, ddof=1)):
            rows.append({"feature": feature, "value": float(g),
                         "mean_contribution": float(mean),
                         "sd_contribution": float(sd),
                         "n_folds": int(stacked.shape[0])})
    return pd.DataFrame(rows)


def direction_summary(shapes: pd.DataFrame, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Per feature: does the fitted shape point the same way as the raw
    association?

    The published association tables report unsigned strength only
    (Cramer's V in table 03; AUC folded to max(auc, 1-auc) in table 22), so
    the signed univariate AUC is computed here. A GAM term that runs
    against the marginal association is a suppression effect - legitimate,
    but it means the term is NOT readable on its own, which the report must
    say rather than leave to the reader.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    rows = []
    for feature, grp in shapes.groupby("feature"):
        ok = X[feature].notna().to_numpy()
        auc = float(roc_auc_score(y[ok], X[feature].to_numpy()[ok]))
        rho = float(spearmanr(grp["value"], grp["mean_contribution"]).statistic)
        weak = abs(auc - 0.5) < 0.03
        rows.append({
            "feature": feature,
            "shape_spearman_rho": round(rho, 4),
            "signed_univariate_auc": round(auc, 4),
            "univariate_direction_determined": not weak,
            "direction_agrees": bool(weak or ((rho > 0) == (auc > 0.5))),
            "mean_abs_contribution": round(
                float(grp["mean_contribution"].abs().mean()), 4),
        })
    return pd.DataFrame(rows).sort_values("mean_abs_contribution", ascending=False)


def write_shape_notes(summary: pd.DataFrame, path: Path) -> None:
    """Record every direction discrepancy, mirroring table_20's notes file."""
    lines = [
        "EBM shape-function notes (low-cost configuration)",
        "=" * 52,
        "",
        "Each low-cost feature's fitted shape function is compared with its",
        "signed univariate association. Agreement means the term can be read",
        "on its own; disagreement means the term is a suppression effect,",
        "interpretable only jointly with the correlated variables it sits",
        "beside. Both are reported; neither is silently corrected.",
        "",
    ]
    disagreeing = summary[~summary["direction_agrees"]]
    if len(disagreeing):
        lines.append(f"DIRECTION DISCREPANCIES ({len(disagreeing)}):")
        for _, r in disagreeing.iterrows():
            lines.append(
                f"  - {r['feature']}: shape rho {r['shape_spearman_rho']:+.3f} "
                f"but signed univariate AUC {r['signed_univariate_auc']:.3f}. "
                f"The fitted effect runs opposite to the marginal "
                f"association; this term must not be read in isolation."
            )
    else:
        lines.append("DIRECTION DISCREPANCIES: none.")
    lines.append("")
    weak = summary[~summary["univariate_direction_determined"]]
    if len(weak):
        lines.append(
            "Features whose univariate direction is not determined "
            "(|AUC - 0.5| < 0.03), and which are therefore exempt from the "
            f"comparison: {', '.join(weak['feature'])}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def shapes_figure(shapes: pd.DataFrame, importance_order: list[str], fig_dir: Path):
    features = [f for f in importance_order if f in set(shapes["feature"])][:6]
    if not features:
        features = sorted(shapes["feature"].unique())[:6]
    ncols = 3
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows),
                             squeeze=False)
    for ax, feature in zip(axes.flat, features):
        sub = shapes[shapes["feature"] == feature]
        ax.plot(sub["value"], sub["mean_contribution"], lw=2)
        ax.fill_between(sub["value"],
                        sub["mean_contribution"] - sub["sd_contribution"],
                        sub["mean_contribution"] + sub["sd_contribution"],
                        alpha=0.2)
        ax.axhline(0.0, lw=0.8, ls="--", color="0.4")
        ax.set_title(f"`{feature}`")
        ax.set_xlabel("bin representative value")
        ax.set_ylabel("log-odds contribution")
    for ax in axes.flat[len(features):]:
        ax.set_visible(False)
    fig.suptitle("Low-cost EBM shape functions (mean +/- SD across outer folds)",
                 y=1.0)
    fig.tight_layout()
    save(fig, fig_dir, "fig_r13_ebm_shapes")
    plt.close(fig)


def main() -> int:
    cfg = load_config()
    proc_dir, tables_dir = ensure_dirs(cfg["paths"]["processed_dir"],
                                       cfg["paths"]["tables_dir"])
    (fig_dir,) = ensure_dirs(cfg["paths"]["figures_dir"])
    apply_style()

    pooled = pooled_predictions(pd.read_csv(proc_dir / "cv_predictions.csv.gz"))
    comparison = comparison_table(pooled)
    comparison.to_csv(tables_dir / "table_28_ebm_vs_svm.csv", index=False)
    print(f"wrote {tables_dir / 'table_28_ebm_vs_svm.csv'}")
    print(comparison.to_string(index=False))

    result = clean_dataset()
    X = feature_matrix(result)
    y = result.target.to_numpy()
    print("\nextracting EBM shape functions across outer folds ...")
    shapes = extract_shapes(
        X, y,
        settings_seed=int(cfg["seed"]),
        outer_folds=int(cfg["cross_validation"]["outer_folds"]),
        inner_folds=int(cfg["cross_validation"]["inner_folds"]),
        repeats=int(cfg["cross_validation"]["repeats"]),
        scoring=str(cfg["cross_validation"]["inner_scoring"]),
    )
    shapes.to_csv(tables_dir / "table_28_ebm_shape_functions.csv", index=False)
    print(f"wrote {tables_dir / 'table_28_ebm_shape_functions.csv'} ({len(shapes)} rows)")

    summary = direction_summary(shapes, X.loc[:, list(get_config(CONFIG).features)], y)
    summary.to_csv(tables_dir / "table_28_ebm_shape_summary.csv", index=False)
    write_shape_notes(summary, tables_dir / "table_28_ebm_shape_notes.txt")
    n_bad = int((~summary["direction_agrees"]).sum())
    print(f"wrote {tables_dir / 'table_28_ebm_shape_summary.csv'} "
          f"({n_bad} direction discrepancy/ies, documented in "
          f"table_28_ebm_shape_notes.txt)")

    stability = pd.read_csv(tables_dir / "table_20_importance_stability.csv")
    order = (
        stability[(stability["target"] == "low_cost")
                  & (stability["method"] == "permutation")]
        .sort_values("mean_importance", ascending=False)["feature"].tolist()
    )
    shapes_figure(shapes, order, fig_dir)
    print("wrote fig_r13_ebm_shapes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
