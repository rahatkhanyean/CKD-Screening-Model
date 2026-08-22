"""Stage 1: clean the raw file, build the data dictionary, audit data quality.

Outputs
-------
data/processed/ckd_clean.csv          cleaned, human-readable labels, NA for invalid cells
data/processed/ckd_encoded.csv        numeric model-ready matrix + target
data/processed/data_dictionary.csv    one row per variable, verifiable facts only
data/processed/bin_mappings.json      every observed bin label -> bounds + representative
data/processed/feature_configurations.csv
reports/data_quality_report.md
reports/tables/*.csv

The raw file at data/raw/ is never modified.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ckd.config import ensure_dirs, load_config, resolve  # noqa: E402
from ckd.data.clean import clean_dataset  # noqa: E402
from ckd.data.load import load_raw  # noqa: E402
from ckd.data.quality import build_quality_report, report_to_markdown  # noqa: E402
from ckd.features.configs import FEATURE_CONFIGS, VARIABLES  # noqa: E402


def main() -> int:
    cfg = load_config()
    proc_dir, tables_dir = ensure_dirs(
        cfg["paths"]["processed_dir"], cfg["paths"]["tables_dir"]
    )
    reports_dir = resolve("reports")

    print("Loading raw file ...")
    raw_result = load_raw()
    print(f"  source           : {raw_result.source_path}")
    print(f"  sha256           : {raw_result.sha256}")
    print(f"  shape as read    : {raw_result.shape_before}")
    print(f"  metadata rows    : {cfg['data']['n_metadata_rows']} removed")
    print(f"  patient records  : {raw_result.shape_after[0]}")

    print("Cleaning and encoding ...")
    result = clean_dataset(raw_result)

    clean_out = result.clean.copy()
    clean_out["class"] = result.clean["class"]
    clean_out["target"] = result.target.to_numpy()
    clean_path = proc_dir / "ckd_clean.csv"
    clean_out.to_csv(clean_path, index=False, encoding="utf-8")
    print(f"  wrote {clean_path.relative_to(resolve('.'))}  {clean_out.shape}")

    encoded_out = result.encoded.copy()
    encoded_out["target"] = result.target.to_numpy()
    encoded_path = proc_dir / "ckd_encoded.csv"
    encoded_out.to_csv(encoded_path, index=False, encoding="utf-8")
    print(f"  wrote {encoded_path.relative_to(resolve('.'))}  {encoded_out.shape}")

    # ---------------- bin mappings ----------------
    bins_path = proc_dir / "bin_mappings.json"
    with open(bins_path, "w", encoding="utf-8") as fh:
        json.dump(result.bin_map, fh, indent=2, ensure_ascii=False, default=str)
    print(f"  wrote {bins_path.relative_to(resolve('.'))}")

    # ---------------- data dictionary ----------------
    clean = result.clean
    rows = []
    for v in VARIABLES:
        if v.name not in clean.columns:
            continue
        col = clean[v.name]
        bins = result.bin_map.get(v.name, [])
        reps = [b["representative"] for b in bins if b["representative"] == b["representative"]]
        rows.append(
            {
                "variable": v.name,
                "tier": v.tier,
                "description_verifiable_only": v.description,
                "cost_rationale": v.rationale,
                "interpretation_uncertain": v.uncertain,
                "uncertainty_note": v.uncertainty_note,
                "n_distinct_observed": int(col.nunique(dropna=True)),
                "n_missing": int(col.isna().sum()),
                "observed_levels": " | ".join(
                    str(b["label"]) for b in bins if str(b["label"]) != "nan"
                ),
                "encoded_min_representative": min(reps) if reps else None,
                "encoded_max_representative": max(reps) if reps else None,
                "modal_level": str(col.mode(dropna=True).iloc[0]) if col.notna().any() else None,
                "modal_proportion": round(
                    float(col.value_counts(normalize=True, dropna=True).iloc[0]), 4
                )
                if col.notna().any()
                else None,
            }
        )
    dd = pd.DataFrame(rows)
    dd_path = proc_dir / "data_dictionary.csv"
    dd.to_csv(dd_path, index=False, encoding="utf-8")
    dd.to_csv(tables_dir / "table_01_data_dictionary.csv", index=False, encoding="utf-8")
    print(f"  wrote {dd_path.relative_to(resolve('.'))}  ({len(dd)} variables)")

    # ---------------- feature configurations ----------------
    cfg_rows = []
    for name, fc in FEATURE_CONFIGS.items():
        cfg_rows.append(
            {
                "configuration": name,
                "n_features": len(fc.features),
                "valid_for_clinical_interpretation": fc.valid_for_clinical_interpretation,
                "contains_prohibited_columns": ", ".join(fc.contains_forbidden) or "none",
                "features": ", ".join(fc.features),
                "purpose": fc.purpose,
            }
        )
    fc_df = pd.DataFrame(cfg_rows)
    fc_df.to_csv(proc_dir / "feature_configurations.csv", index=False, encoding="utf-8")
    fc_df.to_csv(
        tables_dir / "table_02_feature_configurations.csv", index=False, encoding="utf-8"
    )
    print(f"  wrote feature_configurations.csv ({len(fc_df)} configurations)")

    # ---------------- quality report ----------------
    print("Auditing data quality ...")
    report = build_quality_report(result)
    md = report_to_markdown(report)
    qr_path = reports_dir / "data_quality_report.md"
    qr_path.write_text(md, encoding="utf-8")
    print(f"  wrote {qr_path.relative_to(resolve('.'))}  ({len(md.splitlines())} lines)")

    with open(proc_dir / "data_quality_report.json", "w", encoding="utf-8") as fh:
        json.dump(report.sections, fh, indent=2, ensure_ascii=False, default=str)

    # Machine-readable summary table used by the research report.
    q = report.sections
    summary = pd.DataFrame(
        [
            {"item": "Source SHA-256", "value": q["provenance"]["sha256"]},
            {
                "item": "Rows as read (excluding header)",
                "value": q["provenance"]["shape_including_metadata_rows"][0],
            },
            {"item": "Metadata rows removed", "value": cfg["data"]["n_metadata_rows"]},
            {
                "item": "Patient records analysed",
                "value": q["provenance"]["shape_after_metadata_removal"][0],
            },
            {"item": "Columns", "value": q["provenance"]["n_columns"]},
            {"item": "CKD (positive)", "value": q["class_distribution"]["n_positive"]},
            {"item": "Non-CKD (negative)", "value": q["class_distribution"]["n_negative"]},
            {
                "item": "Missing cells after cleaning",
                "value": q["missing_values"]["total_missing_cells"],
            },
            {
                "item": "Exact duplicate records",
                "value": q["duplicates"]["exact_duplicate_records"],
            },
            {
                "item": "Near-constant features (modal >= 90%)",
                "value": len(q["near_constant_features"]),
            },
            {
                "item": "Bin-definition anomalies",
                "value": len(q["bin_anomalies"]),
            },
            {
                "item": "Clinical inconsistencies flagged",
                "value": len(q["clinical_inconsistencies"]),
            },
            {
                "item": "affected is an exact copy of class",
                "value": q["leakage_relationships"]["affected_is_exact_copy_of_class"],
            },
            {
                "item": "Variables with uncertain interpretation",
                "value": int(dd["interpretation_uncertain"].sum()),
            },
        ]
    )
    summary.to_csv(
        tables_dir / "table_00_dataset_summary.csv", index=False, encoding="utf-8"
    )

    print("\nKey findings")
    for _, r in summary.iterrows():
        print(f"  {r['item']:<45} {r['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
