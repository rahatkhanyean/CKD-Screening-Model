"""Cleaning: missing-token handling, encoding to representative values, target coding.

Design rule enforced here
-------------------------
The *only* transformation applied outside cross-validation folds is the
per-cell, stateless bin -> representative-value mapping documented in
:mod:`ckd.data.bins`. It uses no cross-row statistics, so it cannot transfer
information between folds. Imputation, scaling, feature selection, calibration
and threshold selection are all fold-local and live in :mod:`ckd.models`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import load_config
from .bins import BinParseError, normalise_label, parse_bin
from .load import RawLoadResult, load_raw

# Columns that are categorical labels rather than numeric bins.
NON_NUMERIC_COLUMNS = ("class", "stage")

# CKD stage label -> ordinal code. This is an ordered clinical staging scale
# (G1..G5 in KDIGO terminology); the mapping is stateless and order-preserving.
STAGE_ORDER = {"s1": 1.0, "s2": 2.0, "s3": 3.0, "s4": 4.0, "s5": 5.0}


@dataclass
class CleanResult:
    """Cleaned data plus everything needed to audit the cleaning."""

    clean: pd.DataFrame                    # human-readable, original labels, tokens -> NA
    encoded: pd.DataFrame                  # numeric, model-ready
    target: pd.Series                      # 1 = ckd, 0 = notckd
    bin_map: dict[str, list[dict]] = field(default_factory=dict)
    missing_cells: list[dict] = field(default_factory=list)
    unparsed_cells: list[dict] = field(default_factory=list)
    raw_result: RawLoadResult | None = None


def _is_missing_token(value: object, tokens: set[str]) -> bool:
    return normalise_label(value).strip().lower() in tokens


def clean_dataset(raw_result: RawLoadResult | None = None) -> CleanResult:
    """Produce the cleaned and encoded datasets from the raw file."""
    cfg = load_config()
    raw_result = raw_result or load_raw()
    df = raw_result.patients.copy()

    tokens = {str(t).strip().lower() for t in cfg["data"]["missing_tokens"]}
    target_col = cfg["data"]["target_column"]
    pos, neg = cfg["data"]["positive_label"], cfg["data"]["negative_label"]

    line_col = df.pop("source_csv_line")
    feature_cols = list(df.columns)

    clean = pd.DataFrame(index=df.index)
    encoded = pd.DataFrame(index=df.index)
    bin_map: dict[str, list[dict]] = {}
    missing_cells: list[dict] = []
    unparsed_cells: list[dict] = []

    for col in feature_cols:
        series = df[col]
        cleaned_vals: list[object] = []
        encoded_vals: list[float] = []
        seen: dict[str, dict] = {}

        for idx, raw_val in series.items():
            label = normalise_label(raw_val)

            if _is_missing_token(raw_val, tokens):
                missing_cells.append(
                    {
                        "row_index": int(idx),
                        "source_csv_line": int(line_col.iloc[idx]),
                        "column": col,
                        "raw_value": repr(raw_val),
                        "reason": "matched a declared missing token",
                    }
                )
                cleaned_vals.append(pd.NA)
                encoded_vals.append(np.nan)
                continue

            if col == target_col:
                cleaned_vals.append(label)
                encoded_vals.append(np.nan)  # target handled separately
                continue

            if col == "stage":
                key = label.strip().lower()
                cleaned_vals.append(key)
                code = STAGE_ORDER.get(key, np.nan)
                if np.isnan(code):
                    unparsed_cells.append(
                        {
                            "row_index": int(idx),
                            "source_csv_line": int(line_col.iloc[idx]),
                            "column": col,
                            "raw_value": repr(raw_val),
                            "reason": "not one of s1..s5",
                        }
                    )
                encoded_vals.append(code)
                seen.setdefault(key, {"label": key, "lower": code, "upper": code, "representative": code})
                continue

            try:
                lo, hi, rep = parse_bin(raw_val)
            except BinParseError as exc:
                unparsed_cells.append(
                    {
                        "row_index": int(idx),
                        "source_csv_line": int(line_col.iloc[idx]),
                        "column": col,
                        "raw_value": repr(raw_val),
                        "reason": str(exc),
                    }
                )
                cleaned_vals.append(pd.NA)
                encoded_vals.append(np.nan)
                continue

            cleaned_vals.append(label)
            encoded_vals.append(rep)
            seen.setdefault(label, {"label": label, "lower": lo, "upper": hi, "representative": rep})

        clean[col] = cleaned_vals
        if col != target_col:
            encoded[col] = encoded_vals
        bin_map[col] = sorted(seen.values(), key=lambda d: (d["representative"], d["label"]))

    target = clean[target_col].map({pos: 1, neg: 0}).astype("Int64")
    if target.isna().any():
        bad = clean.loc[target.isna(), target_col].unique().tolist()
        raise ValueError(f"Unmappable outcome labels found: {bad}")
    target = target.astype(int)
    target.name = "target"

    clean.insert(0, "source_csv_line", line_col.to_numpy())
    encoded.insert(0, "source_csv_line", line_col.to_numpy())

    return CleanResult(
        clean=clean,
        encoded=encoded,
        target=target,
        bin_map=bin_map,
        missing_cells=missing_cells,
        unparsed_cells=unparsed_cells,
        raw_result=raw_result,
    )


def feature_matrix(result: CleanResult) -> pd.DataFrame:
    """Model-ready numeric feature matrix (no target, no provenance column)."""
    return result.encoded.drop(columns=["source_csv_line"])
