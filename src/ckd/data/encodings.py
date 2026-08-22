"""Alternative encodings of the published interval labels (Phase 4a / N5).

The reference analysis represents each interval by a single number: the
midpoint of a closed bin, the finite edge of an open-ended one. That is a
modelling choice, and a reviewer is entitled to ask whether it drives any
result. This module supplies three alternatives so the question can be
answered rather than argued.

Every encoding here preserves the property that makes the reference
encoding safe (see ``docs/PROJECT_DOCUMENTATION.md`` §1.5a): each is a
**pure function of one cell's label together with the published bin list
for that column**. None uses the outcome, and none uses any statistic of
the observed rows - so none can transfer information between
cross-validation folds, and all may legitimately be applied before
splitting. ``tests/test_encodings.py`` asserts this directly by re-encoding
arbitrary row subsets.

Encodings
---------
``midpoint``     The reference encoding: bin representative values.
``bin_index``    The bin's ordinal position (0, 1, 2, ...) among that
                 column's published bins, ordered by representative value.
                 Discards the *spacing* of the bins while keeping order.
``rank_normal``  The bin's ordinal position mapped through the inverse
                 normal CDF at evenly spaced quantiles. Keeps order,
                 imposes a Gaussian-like spacing. For tree ensembles this
                 is a monotone relabelling of ``bin_index`` and cannot
                 change the fitted model; it can change linear and
                 kernel models, which is the point of including it.
``onehot``       One indicator column per published bin. Discards order
                 entirely, so a model may fit any pattern across bins at
                 the cost of many more parameters - the least suitable for
                 this sample size, included as the extreme case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

ENCODINGS = ("midpoint", "bin_index", "rank_normal", "onehot")


def _ordered_labels(bin_map: dict[str, list[dict]], column: str) -> list[str]:
    """Published bins for one column, ordered by representative value.

    ``bin_map`` is already sorted by ``(representative, label)`` when the
    cleaning step builds it; this re-sorts defensively so the encoding does
    not silently depend on that.
    """
    entries = sorted(bin_map[column], key=lambda d: (d["representative"], d["label"]))
    return [str(e["label"]) for e in entries]


def encode(
    clean: pd.DataFrame,
    bin_map: dict[str, list[dict]],
    columns: list[str],
    method: str,
) -> pd.DataFrame:
    """Encode label columns of ``clean`` by ``method``.

    Parameters
    ----------
    clean:
        The human-readable frame of interval labels.
    bin_map:
        Published bins per column, from :class:`ckd.data.clean.CleanResult`.
    columns:
        Columns to encode. Others are ignored.
    method:
        One of :data:`ENCODINGS`.
    """
    if method not in ENCODINGS:
        raise ValueError(f"unknown encoding {method!r}; expected one of {ENCODINGS}")

    frames: list[pd.DataFrame] = []
    for column in columns:
        labels = clean[column].astype("string")
        ordered = _ordered_labels(bin_map, column)
        position = {label: i for i, label in enumerate(ordered)}

        if method == "onehot":
            block = pd.DataFrame(
                {
                    f"{column}__{label}": labels.eq(label).astype(float)
                    for label in ordered
                },
                index=clean.index,
            )
            # A missing label must not silently become "all zeros", which a
            # model would read as a real pattern; propagate missingness.
            block.loc[labels.isna(), :] = np.nan
            frames.append(block)
            continue

        index = labels.map(position).astype("Float64").to_numpy(dtype=float, na_value=np.nan)
        if method == "bin_index":
            values = index
        elif method == "rank_normal":
            n = max(len(ordered), 1)
            quantiles = (np.arange(n) + 0.5) / n
            lookup = norm.ppf(quantiles)
            values = np.full(len(index), np.nan)
            finite = np.isfinite(index)
            values[finite] = lookup[index[finite].astype(int)]
        else:  # midpoint
            rep = {
                str(e["label"]): float(e["representative"]) for e in bin_map[column]
            }
            values = labels.map(rep).astype("Float64").to_numpy(
                dtype=float, na_value=np.nan
            )
        frames.append(pd.DataFrame({column: values}, index=clean.index))

    return pd.concat(frames, axis=1)


def encoded_feature_names(
    bin_map: dict[str, list[dict]], columns: list[str], method: str
) -> list[str]:
    """Column names the encoding produces, in order."""
    if method != "onehot":
        return list(columns)
    names: list[str] = []
    for column in columns:
        names.extend(f"{column}__{label}" for label in _ordered_labels(bin_map, column))
    return names
