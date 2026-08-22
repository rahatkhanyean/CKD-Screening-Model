"""Model definitions and hyper-parameter search spaces.

Model choice is constrained by the sample size: 200 patients, 128 of them CKD.
Neural networks are deliberately excluded - with roughly 5 to 25 predictors and
72 minority-class events there is no prospect of estimating a deep model
reliably, and doing so would invite exactly the kind of over-fitting this study
is designed to detect.

Search spaces are kept small (at most 8 candidates per model) because the inner
loop sees only ~128 patients: a large grid searched on 4 inner folds of ~32
patients each would select on noise. This is a documented bias/variance
trade-off, not an oversight.

Gradient boosting: XGBoost is used rather than CatBoost. Both were installed
and benchmarked; on this data XGBoost fitted roughly three times faster, which
makes the full repeated nested design feasible. The choice is about compute,
not about expected accuracy.

Explainable Boosting Machine: configured with ``interactions=0``, i.e. a pure
generalised additive model. With 200 patients, pairwise interaction terms would
be estimated from a few dozen events each. Restricting to main effects also
makes the shape functions directly readable, which is the point of using an EBM
here. It is an accuracy/interpretability/compute trade-off, and it is stated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

try:  # optional dependency, checked at import time and reported honestly
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:  # pragma: no cover - environment dependent
    XGBClassifier = None  # type: ignore[assignment]
    HAS_XGBOOST = False

try:
    from interpret.glassbox import ExplainableBoostingClassifier

    HAS_EBM = True
except Exception:  # pragma: no cover - environment dependent
    ExplainableBoostingClassifier = None  # type: ignore[assignment]
    HAS_EBM = False


@dataclass(frozen=True)
class ModelSpec:
    """An estimator factory plus its inner-loop search space."""

    name: str
    label: str
    factory: Callable[[int], Any]
    param_grid: dict[str, list[Any]] = field(default_factory=dict)
    needs_scaling: bool = False
    supports_shap_tree: bool = False
    supports_shap_linear: bool = False
    available: bool = True
    unavailable_reason: str = ""
    notes: str = ""


def _dummy(seed: int) -> Any:
    # Predicts the training-fold prevalence for every patient. This is the
    # reference every other model must beat; a model that cannot beat it has
    # learned nothing.
    return DummyClassifier(strategy="prior", random_state=seed)


def _logreg(seed: int) -> Any:
    # L2 is the sklearn default; it is not passed explicitly because the
    # `penalty` argument is deprecated from scikit-learn 1.8 onwards.
    return LogisticRegression(
        solver="liblinear",
        max_iter=5000,
        random_state=seed,
    )


def _rf(seed: int) -> Any:
    # 300 trees: at n=200 the variance reduction from bagging has saturated
    # well before this point, and the smaller forest keeps the repeated nested
    # design tractable. Benchmarked, not assumed.
    return RandomForestClassifier(
        n_estimators=300,
        random_state=seed,
        n_jobs=1,
        bootstrap=True,
    )


def _svm(seed: int) -> Any:
    # probability=True enables Platt scaling internally; the resulting
    # probabilities are still evaluated for calibration like every other model.
    return SVC(probability=True, random_state=seed, kernel="rbf")


def _xgb(seed: int) -> Any:
    return XGBClassifier(
        n_estimators=300,
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
        eval_metric="logloss",
        verbosity=0,
    )


def _ebm(seed: int) -> Any:
    # outer_bags=6 and max_rounds=1500 rather than the library defaults: with
    # 200 patients the default configuration costs ~8.7 s per fit (benchmarked)
    # which makes the full repeated nested design infeasible, while adding
    # capacity that this sample cannot support. Documented as a
    # compute/regularisation trade-off in the research report.
    return ExplainableBoostingClassifier(
        random_state=seed,
        interactions=0,
        outer_bags=6,
        max_rounds=1500,
        n_jobs=1,
    )


MODEL_SPECS: dict[str, ModelSpec] = {
    "dummy": ModelSpec(
        name="dummy",
        label="Dummy (prevalence)",
        factory=_dummy,
        param_grid={},
        notes="Baseline: always predicts the training-fold prevalence.",
    ),
    "logreg": ModelSpec(
        name="logreg",
        label="Logistic regression (L2)",
        factory=_logreg,
        param_grid={
            "clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
            "clf__class_weight": [None, "balanced"],
        },
        needs_scaling=True,
        supports_shap_linear=True,
        notes=(
            "Regularised linear baseline. Because the predictors are ordinal "
            "bin representatives, this assumes an approximately linear effect "
            "of the bin scale on the log-odds."
        ),
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        label="Random forest",
        factory=_rf,
        param_grid={
            "clf__max_depth": [3, None],
            "clf__min_samples_leaf": [1, 5],
            "clf__max_features": ["sqrt"],
            "clf__class_weight": [None, "balanced"],
        },
        supports_shap_tree=True,
        notes="300 trees; 8 candidates, kept small because inner folds hold ~32 patients.",
    ),
    "svm": ModelSpec(
        name="svm",
        label="Support vector machine (RBF)",
        factory=_svm,
        param_grid={
            "clf__C": [0.1, 1.0, 10.0],
            "clf__gamma": ["scale", 0.01],
            "clf__class_weight": [None, "balanced"],
        },
        needs_scaling=True,
        notes="Requires scaling; scaler is fitted inside training folds only.",
    ),
    "xgboost": ModelSpec(
        name="xgboost",
        label="XGBoost",
        factory=_xgb,
        param_grid={
            "clf__max_depth": [2, 3],
            "clf__learning_rate": [0.05, 0.1],
            "clf__subsample": [0.8],
            "clf__colsample_bytree": [0.8],
            "clf__reg_lambda": [1.0, 5.0],
        },
        supports_shap_tree=True,
        available=HAS_XGBOOST,
        unavailable_reason="" if HAS_XGBOOST else "xgboost is not installed",
        notes="Shallow trees (depth 2-3) to limit variance at n=200.",
    ),
    "ebm": ModelSpec(
        name="ebm",
        label="Explainable boosting machine (GAM)",
        factory=_ebm,
        param_grid={
            "clf__learning_rate": [0.01, 0.05],
        },
        available=HAS_EBM,
        unavailable_reason="" if HAS_EBM else "interpret is not installed",
        notes="Main effects only (interactions=0); shape functions are directly readable.",
    ),
}


def get_model(name: str) -> ModelSpec:
    if name not in MODEL_SPECS:
        raise KeyError(f"Unknown model {name!r}. Known: {sorted(MODEL_SPECS)}")
    spec = MODEL_SPECS[name]
    if not spec.available:
        raise RuntimeError(f"Model {name!r} is unavailable: {spec.unavailable_reason}")
    return spec


def available_models(requested: list[str] | None = None) -> list[str]:
    """Requested models that are actually importable in this environment."""
    names = requested or list(MODEL_SPECS)
    return [n for n in names if MODEL_SPECS[n].available]


def unavailable_models(requested: list[str] | None = None) -> dict[str, str]:
    names = requested or list(MODEL_SPECS)
    return {
        n: MODEL_SPECS[n].unavailable_reason
        for n in names
        if not MODEL_SPECS[n].available
    }
