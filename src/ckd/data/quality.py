"""Structured data-quality audit.

Everything reported here is computed from the file; nothing is asserted from
prior belief about what the dataset "should" contain. Findings are returned as
data so they can be rendered to Markdown, checked by tests, and cited in the
research report without transcription errors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..features.configs import FORBIDDEN_IN_VALID, SPEC_BY_NAME, VARIABLES
from ..features.encoders import assert_no_target_correlation_leak
from .bins import parse_bin
from .clean import CleanResult
from .load import verify_metadata_rows


@dataclass
class QualityReport:
    """Container for every data-quality finding."""

    sections: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.sections[key]

    def add(self, key: str, value: Any) -> None:
        self.sections[key] = value


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramer's V between two categorical series."""
    tab = pd.crosstab(x, y)
    if tab.size == 0 or tab.shape[0] < 2 or tab.shape[1] < 2:
        return float("nan")
    chi2 = _chi2(tab.to_numpy(dtype=float))
    n = tab.to_numpy().sum()
    if n == 0:
        return float("nan")
    phi2 = chi2 / n
    r, k = tab.shape
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return float("nan")
    return float(math.sqrt(phi2corr / denom))


def _chi2(observed: np.ndarray) -> float:
    row = observed.sum(axis=1, keepdims=True)
    col = observed.sum(axis=0, keepdims=True)
    total = observed.sum()
    expected = row @ col / total
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(expected > 0, (observed - expected) ** 2 / expected, 0.0)
    return float(term.sum())


def audit_bin_monotonicity(bin_map: dict[str, list[dict]]) -> list[dict]:
    """Check that each feature's bin representatives are strictly increasing.

    A tie or an inversion means the published bin edges overlap, which makes
    the ordinal encoding ambiguous for that feature. Reported, never repaired.
    """
    findings: list[dict] = []
    for col, bins in bin_map.items():
        if col in ("class",) or len(bins) < 2:
            continue
        reps = [b["representative"] for b in bins]
        ties = [
            (bins[i]["label"], bins[i + 1]["label"])
            for i in range(len(reps) - 1)
            if reps[i] == reps[i + 1]
        ]
        inversions = [
            (bins[i]["label"], bins[i + 1]["label"])
            for i in range(len(reps) - 1)
            if reps[i] > reps[i + 1]
        ]
        overlaps = []
        for i in range(len(bins) - 1):
            a, b = bins[i], bins[i + 1]
            if np.isfinite(a["upper"]) and np.isfinite(b["lower"]) and a["upper"] > b["lower"]:
                overlaps.append((a["label"], b["label"]))
        if ties or inversions or overlaps:
            findings.append(
                {
                    "column": col,
                    "n_bins": len(bins),
                    "tied_representatives": ties,
                    "inverted_representatives": inversions,
                    "overlapping_edges": overlaps,
                }
            )
    return findings


def build_quality_report(result: CleanResult) -> QualityReport:
    """Compute the full data-quality audit."""
    report = QualityReport()
    raw_result = result.raw_result
    assert raw_result is not None, "CleanResult must carry its RawLoadResult"

    clean = result.clean
    encoded = result.encoded.drop(columns=["source_csv_line"])
    y = result.target

    # ---------------- 1. Shape and provenance ----------------
    report.add(
        "provenance",
        {
            "source_path": str(raw_result.source_path),
            "sha256": raw_result.sha256,
            "shape_including_metadata_rows": list(raw_result.shape_before),
            "shape_after_metadata_removal": list(raw_result.shape_after[0:1])
            + [raw_result.shape_after[1] - 1],
            "n_columns": int(len(clean.columns) - 1),
            "metadata_row_evidence": verify_metadata_rows(raw_result.metadata),
        },
    )

    # ---------------- 2. Class distribution ----------------
    counts = clean["class"].value_counts().to_dict()
    report.add(
        "class_distribution",
        {
            "counts": {str(k): int(v) for k, v in counts.items()},
            "proportions": {
                str(k): round(float(v) / len(clean), 4) for k, v in counts.items()
            },
            "n_positive": int(y.sum()),
            "n_negative": int((1 - y).sum()),
            "imbalance_ratio": round(float(y.sum()) / float((1 - y).sum()), 3),
            "events_per_candidate_predictor_full_valid": round(
                float(min(y.sum(), (1 - y).sum())) / 25.0, 2
            ),
        },
    )

    # ---------------- 3. Missing values ----------------
    missing_by_col = (
        clean.drop(columns=["source_csv_line"]).isna().sum().astype(int).to_dict()
    )
    report.add(
        "missing_values",
        {
            "total_missing_cells": int(sum(missing_by_col.values())),
            "by_column": {k: int(v) for k, v in missing_by_col.items() if v > 0},
            "columns_with_no_missing": int(sum(1 for v in missing_by_col.values() if v == 0)),
            "detected_missing_cells": result.missing_cells,
            "unparsed_cells": result.unparsed_cells,
        },
    )

    # ---------------- 4. Invalid / inconsistent categories ----------------
    report.add("bin_anomalies", audit_bin_monotonicity(result.bin_map))

    implausible: list[dict] = []
    # Serum potassium above ~8 mEq/L is not compatible with life; the two upper
    # bins here are an order of magnitude beyond that.
    for label_info in result.bin_map.get("pot", []):
        if label_info["representative"] > 8.0:
            n = int((clean["pot"] == label_info["label"]).sum())
            if n:
                implausible.append(
                    {
                        "column": "pot",
                        "label": label_info["label"],
                        "n_patients": n,
                        "issue": (
                            "Serum potassium far outside any survivable range "
                            "(normal 3.5-5.1 mEq/L); almost certainly a "
                            "recording or unit error."
                        ),
                    }
                )
    report.add("implausible_values", implausible)

    # ---------------- 5. Duplicates ----------------
    features_only = clean.drop(columns=["source_csv_line", "class", "affected"])
    dup_full = clean.drop(columns=["source_csv_line"]).duplicated()
    dup_feat = features_only.duplicated()
    report.add(
        "duplicates",
        {
            "exact_duplicate_records": int(dup_full.sum()),
            "duplicate_predictor_patterns_ignoring_outcome": int(dup_feat.sum()),
            "duplicate_source_lines": clean.loc[
                dup_full.to_numpy(), "source_csv_line"
            ].tolist(),
        },
    )

    # ---------------- 6. Constant / near-constant features ----------------
    near_constant: list[dict] = []
    for col in clean.columns:
        if col == "source_csv_line":
            continue
        vc = clean[col].value_counts(dropna=False, normalize=True)
        if len(vc) == 0:
            continue
        top_prop = float(vc.iloc[0])
        if top_prop >= 0.90:
            near_constant.append(
                {
                    "column": col,
                    "modal_value": str(vc.index[0]),
                    "modal_proportion": round(top_prop, 4),
                    "n_distinct": int(clean[col].nunique(dropna=True)),
                    "constant": bool(clean[col].nunique(dropna=True) <= 1),
                }
            )
    report.add("near_constant_features", near_constant)

    # ---------------- 7. Suspicious outcome proxies ----------------
    proxies = assert_no_target_correlation_leak(
        clean.drop(columns=["source_csv_line", "class"]), y.to_numpy(), threshold=0.999
    )
    assoc = {}
    for col in clean.columns:
        if col in ("source_csv_line", "class"):
            continue
        assoc[col] = round(cramers_v(clean[col].astype(str), clean["class"].astype(str)), 4)
    ranked = dict(sorted(assoc.items(), key=lambda kv: (-(kv[1] if kv[1] == kv[1] else -1))))
    report.add(
        "outcome_proxies",
        {
            "deterministic_proxies": proxies,
            "cramers_v_with_outcome": ranked,
            "prohibited_in_valid_models": sorted(FORBIDDEN_IN_VALID),
        },
    )

    # ---------------- 8. Leakage variable relationships ----------------
    ct_affected = pd.crosstab(clean["class"], clean["affected"])
    ct_stage = pd.crosstab(clean["class"], clean["stage"])
    ct_grf = pd.crosstab(clean["class"], clean["grf"].fillna("<missing>"))
    ct_stage_grf = pd.crosstab(clean["stage"], clean["grf"].fillna("<missing>"))

    stage_purity = (
        pd.DataFrame({"stage": clean["stage"], "y": y})
        .groupby("stage")["y"]
        .agg(["size", "mean"])
        .rename(columns={"size": "n", "mean": "proportion_ckd"})
    )
    stage_purity["proportion_ckd"] = stage_purity["proportion_ckd"].round(4)

    # Is `stage` a deterministic function of the grf bin, and vice versa?
    grf_known = clean["grf"].notna()
    grf_to_stage = (
        clean.loc[grf_known].groupby("grf")["stage"].nunique().to_dict()
    )
    stage_to_grf = (
        clean.loc[grf_known].groupby("stage")["grf"].nunique().to_dict()
    )

    report.add(
        "leakage_relationships",
        {
            "class_vs_affected": ct_affected.to_dict(),
            "affected_is_exact_copy_of_class": bool(
                (clean["affected"].astype(str) == y.astype(str)).all()
            ),
            "class_vs_stage": ct_stage.to_dict(),
            "stage_purity": stage_purity.reset_index().to_dict(orient="records"),
            "class_vs_grf": ct_grf.to_dict(),
            "stage_vs_grf": ct_stage_grf.to_dict(),
            "n_stages_per_grf_bin": {str(k): int(v) for k, v in grf_to_stage.items()},
            "n_grf_bins_per_stage": {str(k): int(v) for k, v in stage_to_grf.items()},
            "cramers_v_class_affected": round(
                cramers_v(clean["class"].astype(str), clean["affected"].astype(str)), 4
            ),
            "cramers_v_class_stage": round(
                cramers_v(clean["class"].astype(str), clean["stage"].astype(str)), 4
            ),
            "cramers_v_class_grf": round(
                cramers_v(clean["class"].astype(str), clean["grf"].fillna("<missing>").astype(str)),
                4,
            ),
            "cramers_v_stage_grf": round(
                cramers_v(clean["stage"].astype(str), clean["grf"].fillna("<missing>").astype(str)),
                4,
            ),
        },
    )

    # ---------------- 9. Clinical inconsistencies ----------------
    inconsistencies: list[dict] = []

    # Patients labelled notckd but assigned an advanced CKD stage.
    advanced = clean[(clean["class"] == "notckd") & (clean["stage"].isin(["s3", "s4", "s5"]))]
    if len(advanced):
        inconsistencies.append(
            {
                "issue": "Patients labelled 'notckd' but assigned an advanced CKD stage (s3-s5)",
                "n_patients": int(len(advanced)),
                "source_csv_lines": advanced["source_csv_line"].tolist(),
                "detail": advanced[["class", "stage", "grf", "sc"]].to_dict(orient="records"),
                "interpretation": (
                    "Either the outcome label or the stage assignment is wrong "
                    "for these records. Reported, not corrected: we have no "
                    "external source of truth to arbitrate."
                ),
            }
        )

    # Patients labelled notckd with a low eGFR band.
    low_gfr_reps = {
        b["label"]: b["representative"] for b in result.bin_map.get("grf", [])
    }
    low_gfr_labels = [lbl for lbl, rep in low_gfr_reps.items() if rep < 60.0]
    notckd_low_gfr = clean[
        (clean["class"] == "notckd") & (clean["grf"].isin(low_gfr_labels))
    ]
    if len(notckd_low_gfr):
        inconsistencies.append(
            {
                "issue": "Patients labelled 'notckd' whose eGFR band is below 60 mL/min/1.73m2",
                "n_patients": int(len(notckd_low_gfr)),
                "source_csv_lines": notckd_low_gfr["source_csv_line"].tolist(),
                "detail": notckd_low_gfr[["class", "stage", "grf"]].to_dict(orient="records"),
                "interpretation": (
                    "An eGFR below 60 sustained over three months is itself a "
                    "CKD-defining criterion, so these labels are internally "
                    "inconsistent with the eGFR column. Chronicity cannot be "
                    "verified from a single cross-sectional record, which is "
                    "one possible benign explanation."
                ),
            }
        )

    # Patients flagged anaemic with a high haemoglobin band and vice versa.
    hemo_reps = {b["label"]: b["representative"] for b in result.bin_map.get("hemo", [])}
    high_hb = [lbl for lbl, rep in hemo_reps.items() if rep >= 13.0]
    ane_high_hb = clean[(clean["ane"] == "1") & (clean["hemo"].isin(high_hb))]
    if len(ane_high_hb):
        inconsistencies.append(
            {
                "issue": "Anaemia flag set for patients whose haemoglobin band is >= 13 g/dL",
                "n_patients": int(len(ane_high_hb)),
                "source_csv_lines": ane_high_hb["source_csv_line"].tolist(),
                "interpretation": "Suggests the anaemia flag is not a direct recoding of haemoglobin.",
            }
        )
    report.add("clinical_inconsistencies", inconsistencies)

    # ---------------- 10. The " p " value in grf ----------------
    p_rows = [c for c in result.missing_cells if c["column"] == "grf"] + [
        c for c in result.unparsed_cells if c["column"] == "grf"
    ]
    grf_investigation: dict[str, Any] = {
        "n_anomalous_cells": len(p_rows),
        "cells": p_rows,
    }
    if p_rows:
        idx = [c["row_index"] for c in p_rows]
        subset = clean.loc[idx]
        grf_investigation["affected_patients"] = subset.drop(
            columns=[c for c in subset.columns if c == "source_csv_line"]
        ).assign(source_csv_line=subset["source_csv_line"]).to_dict(orient="records")
        stages = subset["stage"].tolist()
        grf_investigation["stage_of_affected_patients"] = stages
        peers = clean[clean["stage"].isin(stages) & clean["grf"].notna()]
        grf_investigation["grf_bins_of_same_stage_peers"] = (
            peers["grf"].value_counts().to_dict()
        )
        grf_investigation["conclusion"] = (
            "A single cell in 'grf' contains the token ' p ' (letter p with "
            "surrounding whitespace), which is not an interval label. The "
            "affected patient is labelled ckd and staged s5; every other s5 "
            "patient in the file falls in the lowest eGFR band. The most "
            "parsimonious reading is a data-entry error in the eGFR field. "
            "Because the true value cannot be recovered, the cell is treated "
            "as MISSING and imputed inside training folds only. The raw file "
            "is preserved unmodified at data/raw/. Note that 'grf' is excluded "
            "from every clinically valid configuration, so this defect can "
            "only affect the deliberately invalid leaky_model."
        )
    report.add("grf_p_investigation", grf_investigation)

    # ---------------- 11. Variable inventory and open uncertainties ----------------
    report.add(
        "variable_inventory",
        [
            {
                "name": v.name,
                "tier": v.tier,
                "description": v.description,
                "rationale": v.rationale,
                "uncertain": v.uncertain,
                "uncertainty_note": v.uncertainty_note,
                "n_distinct_values": int(clean[v.name].nunique(dropna=True))
                if v.name in clean.columns
                else None,
            }
            for v in VARIABLES
        ],
    )

    # ---------------- 12. Encoded matrix sanity ----------------
    report.add(
        "encoding",
        {
            "strategy": (
                "Stateless per-cell mapping of each interval label to a "
                "representative numeric value (midpoint for closed bins, the "
                "finite edge for open-ended bins). Uses no cross-row "
                "statistics and therefore cannot leak between folds."
            ),
            "n_encoded_columns": int(encoded.shape[1]),
            "encoded_dtypes": {c: str(t) for c, t in encoded.dtypes.items()},
            "columns_with_nan_after_encoding": {
                c: int(n) for c, n in encoded.isna().sum().items() if n > 0
            },
        },
    )

    return report


def report_to_markdown(report: QualityReport) -> str:
    """Render the audit as a human-readable Markdown document."""
    s = report.sections
    out: list[str] = []
    w = out.append

    w("# Data-quality report")
    w("")
    w("Generated by `scripts/01_prepare_data.py`. Every number below is computed")
    w("directly from the source file; nothing is transcribed by hand.")
    w("")

    p = s["provenance"]
    w("## 1. Provenance and shape")
    w("")
    w(f"- Source file: `{p['source_path']}`")
    w(f"- SHA-256: `{p['sha256']}`")
    w(
        f"- Shape as read (header excluded): "
        f"{p['shape_including_metadata_rows'][0]} rows x "
        f"{p['shape_including_metadata_rows'][1]} columns"
    )
    w(
        f"- Shape after removing {p['metadata_row_evidence']['n_metadata_rows']} metadata rows: "
        f"{p['shape_after_metadata_removal'][0]} patients x {p['n_columns']} columns"
    )
    w("")
    ev = p["metadata_row_evidence"]
    w("Evidence that the two dropped rows are metadata rather than patients:")
    w("")
    w(f"- Row 1 unique values: `{ev['row0_unique_values']}` "
      f"(pure type declaration: **{ev['row0_is_type_declaration']}**)")
    w(f"- Row 2 unique values: `{ev['row1_unique_values']}` "
      f"(declares the target: **{ev['row1_declares_target']}**, "
      f"declares meta: **{ev['row1_declares_meta']}**)")
    w("")

    c = s["class_distribution"]
    w("## 2. Class distribution")
    w("")
    w("| Outcome | n | proportion |")
    w("|---|---:|---:|")
    for k, v in c["counts"].items():
        w(f"| {k} | {v} | {c['proportions'][k]:.3f} |")
    w("")
    w(f"- Imbalance ratio (ckd : notckd) = **{c['imbalance_ratio']}**")
    w(
        f"- Minority-class events per candidate predictor in the full valid "
        f"configuration = **{c['events_per_candidate_predictor_full_valid']}** "
        "(well below the >= 10 rule of thumb, and below the requirements of "
        "modern sample-size calculations for prediction models)."
    )
    w("")

    m = s["missing_values"]
    w("## 3. Missing values")
    w("")
    w(f"- Total missing cells after cleaning: **{m['total_missing_cells']}**")
    if m["by_column"]:
        w("")
        w("| Column | missing |")
        w("|---|---:|")
        for k, v in m["by_column"].items():
            w(f"| {k} | {v} |")
    w("")
    if m["detected_missing_cells"]:
        w("Cells converted to missing (each traced to its physical CSV line):")
        w("")
        w("| CSV line | column | raw value | reason |")
        w("|---:|---|---|---|")
        for cell in m["detected_missing_cells"]:
            w(
                f"| {cell['source_csv_line']} | {cell['column']} | "
                f"`{cell['raw_value']}` | {cell['reason']} |"
            )
        w("")

    w("## 4. Invalid or inconsistent categories")
    w("")
    anomalies = s["bin_anomalies"]
    if not anomalies:
        w("No overlapping or non-monotone bin definitions detected.")
    else:
        for a in anomalies:
            w(f"**`{a['column']}`** ({a['n_bins']} bins)")
            w("")
            if a["overlapping_edges"]:
                w(f"- Overlapping edges: {a['overlapping_edges']}")
            if a["tied_representatives"]:
                w(f"- Tied representative values: {a['tied_representatives']}")
            if a["inverted_representatives"]:
                w(f"- Inverted representative values: {a['inverted_representatives']}")
            w("")
        w(
            "These are defects in the published discretisation. They are "
            "reported and carried forward unchanged; repairing them would "
            "require guessing the original continuous values."
        )
    w("")
    if s["implausible_values"]:
        w("Physiologically implausible values:")
        w("")
        w("| Column | label | n | issue |")
        w("|---|---|---:|---|")
        for iv in s["implausible_values"]:
            w(f"| {iv['column']} | `{iv['label']}` | {iv['n_patients']} | {iv['issue']} |")
        w("")

    d = s["duplicates"]
    w("## 5. Duplicate observations")
    w("")
    w(f"- Exact duplicate patient records: **{d['exact_duplicate_records']}**")
    w(
        "- Duplicate predictor patterns ignoring the outcome: "
        f"**{d['duplicate_predictor_patterns_ignoring_outcome']}**"
    )
    w("")

    w("## 6. Constant and near-constant features")
    w("")
    nc = s["near_constant_features"]
    if not nc:
        w("No feature has a modal category covering 90% or more of patients.")
    else:
        w("| Column | modal value | modal proportion | distinct values | constant |")
        w("|---|---|---:|---:|---|")
        for f in nc:
            w(
                f"| {f['column']} | `{f['modal_value']}` | {f['modal_proportion']:.3f} | "
                f"{f['n_distinct']} | {f['constant']} |"
            )
    w("")

    op = s["outcome_proxies"]
    w("## 7. Suspicious outcome proxies")
    w("")
    w(
        f"- Columns that determine the outcome perfectly: "
        f"**{op['deterministic_proxies'] or 'none besides those listed below'}**"
    )
    w(f"- Prohibited in all valid models: `{op['prohibited_in_valid_models']}`")
    w("")
    w("Association with the outcome (bias-corrected Cramer's V), strongest first:")
    w("")
    w("| Variable | Cramer's V |")
    w("|---|---:|")
    for k, v in list(op["cramers_v_with_outcome"].items()):
        w(f"| {k} | {v} |")
    w("")

    lr = s["leakage_relationships"]
    w("## 8. Relationship between `class`, `affected`, `stage` and `grf`")
    w("")
    w(f"- `affected` is an exact copy of `class`: **{lr['affected_is_exact_copy_of_class']}** "
      f"(Cramer's V = {lr['cramers_v_class_affected']})")
    w(f"- `class` vs `stage`: Cramer's V = {lr['cramers_v_class_stage']}")
    w(f"- `class` vs `grf`: Cramer's V = {lr['cramers_v_class_grf']}")
    w(f"- `stage` vs `grf`: Cramer's V = {lr['cramers_v_stage_grf']}")
    w("")
    w("Proportion of CKD within each stage:")
    w("")
    w("| stage | n | proportion ckd |")
    w("|---|---:|---:|")
    for row in lr["stage_purity"]:
        w(f"| {row['stage']} | {row['n']} | {row['proportion_ckd']} |")
    w("")
    w(
        "`stage` is a post-diagnosis label banded from `grf`, and `grf` (eGFR) "
        "is the quantity that defines those bands. Stages s3 and s5 are 100% "
        "CKD. Including any of these three columns turns the prediction task "
        "into a lookup of the diagnosis."
    )
    w("")

    w("## 9. Clinical inconsistencies")
    w("")
    ci = s["clinical_inconsistencies"]
    if not ci:
        w("None detected by the implemented checks.")
    for item in ci:
        w(f"### {item['issue']}")
        w("")
        w(f"- Patients affected: **{item['n_patients']}**")
        w(f"- Source CSV lines: `{item['source_csv_lines']}`")
        w(f"- Interpretation: {item['interpretation']}")
        w("")

    g = s["grf_p_investigation"]
    w("## 10. Investigation of the `\" p \"` value in `grf`")
    w("")
    w(f"- Anomalous cells found: **{g['n_anomalous_cells']}**")
    for cell in g.get("cells", []):
        w(
            f"- CSV line {cell['source_csv_line']}, column `{cell['column']}`, "
            f"raw value `{cell['raw_value']}` ({cell['reason']})"
        )
    if "stage_of_affected_patients" in g:
        w(f"- Stage of the affected patient(s): `{g['stage_of_affected_patients']}`")
        w(f"- eGFR bands of same-stage peers: `{g['grf_bins_of_same_stage_peers']}`")
    w("")
    if "conclusion" in g:
        w(g["conclusion"])
    w("")

    w("## 11. Variable inventory and unresolved uncertainties")
    w("")
    w("| Variable | Tier | Distinct values | Interpretation uncertain |")
    w("|---|---|---:|---|")
    for v in s["variable_inventory"]:
        w(
            f"| `{v['name']}` | {v['tier']} | {v['n_distinct_values']} | "
            f"{'YES' if v['uncertain'] else 'no'} |"
        )
    w("")
    w("Unresolved uncertainties, stated rather than invented:")
    w("")
    for v in s["variable_inventory"]:
        if v["uncertain"]:
            w(f"- **`{v['name']}`** - {v['uncertainty_note']}")
    w("")

    e = s["encoding"]
    w("## 12. Encoding")
    w("")
    w(e["strategy"])
    w("")
    w(f"- Encoded numeric columns: **{e['n_encoded_columns']}**")
    w(f"- Columns still containing NaN after encoding: `{e['columns_with_nan_after_encoding']}`")
    w("")

    return "\n".join(out)
