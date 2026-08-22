"""Leakage-safe transformers.

This module is the *enforcement layer* of the study. Every pipeline built for
a clinically valid configuration begins with a :class:`LeakageGuard`, so a
prohibited column cannot silently reach an estimator even if a caller passes
the wrong DataFrame. The guard raises at ``fit`` **and** at ``transform``, so
it also fires if a prohibited column is smuggled in at prediction time.

The design deliberately fails loudly rather than dropping offending columns:
silently dropping a column would hide a bug in the calling code, whereas an
exception makes it impossible to publish a leaked result by accident.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .configs import ALWAYS_FORBIDDEN, FORBIDDEN_IN_VALID


class LeakageError(RuntimeError):
    """Raised when a prohibited column reaches a model pipeline."""


class LeakageGuard(BaseEstimator, TransformerMixin):
    """Reject prohibited columns at fit and at transform time.

    Parameters
    ----------
    forbidden:
        Column names that must not be present. Defaults to
        :data:`ckd.features.configs.FORBIDDEN_IN_VALID`.
    allow:
        Columns exempted from the check. Used *only* by the deliberately
        invalid ``leaky_model`` configuration, which must be able to include
        the prohibited columns in order to quantify their effect. Even then,
        the columns in :data:`ALWAYS_FORBIDDEN` that are not explicitly listed
        remain blocked.
    label:
        Free text included in the error message, normally the configuration
        name, so failures are self-explanatory in test output.
    """

    def __init__(
        self,
        forbidden: Iterable[str] | None = None,
        allow: Iterable[str] = (),
        label: str = "",
    ) -> None:
        self.forbidden = forbidden
        self.allow = allow
        self.label = label

    def _forbidden_set(self) -> frozenset[str]:
        base = frozenset(self.forbidden) if self.forbidden is not None else FORBIDDEN_IN_VALID
        return base - frozenset(self.allow)

    def _check(self, X) -> None:
        if not isinstance(X, pd.DataFrame):
            # Without column names we cannot verify anything, and a silent pass
            # would defeat the purpose of the guard.
            raise LeakageError(
                f"LeakageGuard({self.label!r}) requires a pandas DataFrame with "
                f"column names, received {type(X).__name__}."
            )
        present = sorted(set(X.columns) & self._forbidden_set())
        if present:
            raise LeakageError(
                f"Prohibited column(s) {present} reached pipeline {self.label!r}. "
                "These columns are the outcome, an exact copy of it, or a "
                "post-diagnosis derivative, and must never be used as "
                "predictors in a valid screening model."
            )

    def fit(self, X, y=None):  # noqa: D102 - sklearn API
        self._check(X)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):  # noqa: D102 - sklearn API
        self._check(X)
        return X

    def get_feature_names_out(self, input_features=None):  # noqa: D102
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.asarray(getattr(self, "feature_names_in_", []), dtype=object)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.no_validation = True
        return tags


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select an explicit, ordered list of columns.

    Raises if any requested column is absent, so a typo in a feature
    configuration surfaces immediately instead of silently shrinking the model.
    """

    def __init__(self, columns: Sequence[str]) -> None:
        self.columns = columns

    def _select(self, X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"ColumnSelector requires a pandas DataFrame, got {type(X).__name__}."
            )
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise KeyError(f"Columns absent from the input frame: {missing}")
        return X.loc[:, list(self.columns)]

    def fit(self, X, y=None):  # noqa: D102
        self._select(X)
        self.n_features_in_ = len(self.columns)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):  # noqa: D102
        return self._select(X)

    def get_feature_names_out(self, input_features=None):  # noqa: D102
        return np.asarray(list(self.columns), dtype=object)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.no_validation = True
        return tags


def assert_no_target_correlation_leak(
    X: pd.DataFrame, y: Sequence[int], threshold: float = 0.999
) -> list[str]:
    """Return columns that are (anti-)deterministic functions of the outcome.

    A column whose value determines ``y`` perfectly is an outcome proxy. This
    is a *diagnostic*, used by the data-quality audit, and is deliberately not
    used to filter features automatically: automatic removal based on the
    outcome would itself be a form of using the target during preprocessing.
    """
    y_arr = np.asarray(y)
    suspicious: list[str] = []
    for col in X.columns:
        values = X[col]
        if values.isna().all():
            continue
        # Purity: does knowing the column value determine the outcome?
        grouped = pd.DataFrame({"v": values, "y": y_arr}).dropna()
        if grouped.empty:
            continue
        purity = grouped.groupby("v", observed=True)["y"].apply(
            lambda s: max(s.mean(), 1.0 - s.mean())
        )
        weights = grouped.groupby("v", observed=True)["y"].size()
        weighted_purity = float((purity * weights).sum() / weights.sum())
        if weighted_purity >= threshold:
            suspicious.append(col)
    return suspicious
