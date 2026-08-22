"""Construction of leakage-safe modelling pipelines.

Every pipeline has the same skeleton::

    LeakageGuard -> ColumnSelector -> SimpleImputer -> [StandardScaler] -> estimator

* ``LeakageGuard`` raises if a prohibited column is present, at fit and at
  transform time. For the deliberately invalid ``leaky_model`` configuration
  the guard is constructed with an explicit ``allow`` list, so the leak is an
  opt-in, visible act rather than an omission.
* ``SimpleImputer`` and ``StandardScaler`` are ordinary pipeline steps, so
  their statistics are estimated on the training part of whatever split
  wraps them - inner folds during tuning, outer training folds for the final
  refit. Nothing is ever fitted on complete data.
* Feature selection, where used, is likewise a pipeline step and therefore
  fold-local.
"""

from __future__ import annotations

from typing import Any

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..features.configs import ALWAYS_FORBIDDEN, FORBIDDEN_IN_VALID, get_config
from ..features.encoders import ColumnSelector, LeakageGuard
from .zoo import ModelSpec, get_model


def build_pipeline(
    model_name: str,
    config_name: str,
    seed: int,
    feature_override: list[str] | None = None,
) -> Pipeline:
    """Build the full pipeline for one (model, feature configuration) pair.

    ``feature_override`` replaces the columns the selector passes on, for
    encodings that rename columns (one-hot expands ``sg`` into
    ``sg__<label>``; see :mod:`ckd.data.encodings`). It changes *only* the
    selector: both leakage guards still run first and still test against
    the same prohibited names, so an override cannot be used to smuggle a
    prohibited column past them. Callers must derive the override from a
    configuration's own features - ``assert_pipeline_is_clean`` verifies
    that every overridden name maps back to a declared feature.
    """
    cfg = get_config(config_name)
    spec: ModelSpec = get_model(model_name)

    if cfg.valid_for_clinical_interpretation:
        allow: tuple[str, ...] = ()
        forbidden = FORBIDDEN_IN_VALID
    else:
        # The invalid configuration may include exactly the columns it declares,
        # and nothing else. The outcome column itself is never allowed.
        allow = tuple(c for c in cfg.contains_forbidden if c not in {"class"})
        forbidden = FORBIDDEN_IN_VALID

    # Defence in depth: whatever the configuration claims, the raw outcome
    # column can never pass.
    guard = LeakageGuard(forbidden=forbidden, allow=allow, label=config_name)
    outcome_guard = LeakageGuard(
        forbidden=frozenset({"class"}), allow=(), label=f"{config_name}:outcome"
    )

    steps: list[tuple[str, Any]] = [
        ("outcome_guard", outcome_guard),
        ("leakage_guard", guard),
        ("select", ColumnSelector(
            list(feature_override) if feature_override is not None
            else list(cfg.features)
        )),
        # Median imputation: the only missing cell in this dataset is the
        # invalid ' p ' entry in grf, but the step is kept unconditionally so
        # that the pipeline behaves correctly on any resample or future data.
        ("impute", SimpleImputer(strategy="median")),
    ]
    if spec.needs_scaling:
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", spec.factory(seed)))

    return Pipeline(steps)


def pipeline_feature_names(pipe: Pipeline) -> list[str]:
    """Columns a fitted pipeline actually feeds to its estimator."""
    return list(pipe.named_steps["select"].columns)


def assert_pipeline_is_clean(pipe: Pipeline, config_name: str) -> None:
    """Raise if a valid configuration's pipeline could admit a prohibited column.

    Used by the automated test-suite and by the runner as a belt-and-braces
    check before any result is written to disk.
    """
    cfg = get_config(config_name)
    features = set(pipeline_feature_names(pipe))

    # An encoding may rename a column (one-hot expands 'sg' into
    # 'sg__<label>'). Every selected name must still trace back to a column
    # the configuration declares, so a renaming scheme cannot introduce a
    # column that was never authorised. The base name is taken as the part
    # before the encoding separator.
    declared = set(cfg.features)
    base_names = {name.split("__", 1)[0] for name in features}
    undeclared_bases = base_names - declared
    if undeclared_bases:
        raise AssertionError(
            f"Pipeline for {config_name!r} selects column(s) whose base name "
            f"is not declared by the configuration: {sorted(undeclared_bases)}."
        )
    # Prohibition checks below run against base names too, so that a
    # prohibited column cannot be hidden behind an encoding suffix.
    features = features | base_names

    # The raw outcome column is prohibited unconditionally, in every
    # configuration including the deliberately invalid one.
    if "class" in features:
        raise AssertionError(
            f"Pipeline for {config_name!r} selects the outcome column 'class'."
        )

    if not cfg.valid_for_clinical_interpretation:
        # An invalid configuration may only contain the prohibited columns it
        # explicitly declares, so an accidental leak is still caught.
        undeclared = (features & ALWAYS_FORBIDDEN) - set(cfg.contains_forbidden)
        if undeclared:
            raise AssertionError(
                f"Invalid configuration {config_name!r} selects undeclared "
                f"prohibited column(s) {sorted(undeclared)}."
            )

    if cfg.valid_for_clinical_interpretation:
        illegal = features & FORBIDDEN_IN_VALID
        if illegal:
            raise AssertionError(
                f"Valid configuration {config_name!r} selects prohibited "
                f"column(s) {sorted(illegal)}."
            )
        guard = pipe.named_steps["leakage_guard"]
        if set(guard.allow):
            raise AssertionError(
                f"Valid configuration {config_name!r} has a guard exemption "
                f"list {sorted(guard.allow)}; valid models must have none."
            )
