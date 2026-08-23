"""Stage 18: extract an external cohort from the MIMIC-IV demo.

Produces a harmonised, one-row-per-subject table in this study's schema,
together with the notes the extraction generates about its own limits.

Output:
  data/external/mimic_iv_demo/processed/mimic_cohort.csv
  reports/tables/table_41_mimic_cohort.csv        (coverage and composition)
  reports/tables/table_41_mimic_notes.txt         (extraction caveats)

Usage
-----
    python scripts/18_mimic_etl.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ckd.config import ensure_dirs, load_config, project_root  # noqa: E402
from ckd.data.external import get_dataset, verify_raw_files  # noqa: E402
from ckd.data.mimic import DIPSTICK_ITEMS, LAB_ITEMS, build_cohort  # noqa: E402

ARCHIVE_ROOT = "mimic-iv-clinical-database-demo-2.2"


def ensure_extracted(record) -> Path:
    """Return the extracted archive root, unpacking it if needed."""
    zip_path = project_root() / record.raw_files[0].path
    root = zip_path.parent / ARCHIVE_ROOT
    if not root.is_dir():
        print(f"extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(zip_path.parent)
    return root


def main() -> int:
    cfg = load_config()
    (tables_dir,) = ensure_dirs(cfg["paths"]["tables_dir"])

    record = get_dataset("mimic_iv_demo")
    verify_raw_files(record)
    print(f"raw files verified for {record.dataset_id}")

    root = ensure_extracted(record)
    cohort = build_cohort(root)

    out_dir = project_root() / "data" / "external" / "mimic_iv_demo" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = cohort.frame.copy()
    saved["ckd_label"] = cohort.target
    saved["stage"] = cohort.stage.to_numpy()
    cohort_path = out_dir / "mimic_cohort.csv"
    saved.to_csv(cohort_path, index=False)
    print(f"wrote {cohort_path} ({saved.shape[0]} subjects)")

    variables = list(LAB_ITEMS) + list(DIPSTICK_ITEMS)
    coverage = pd.DataFrame([
        {
            "variable": v,
            "n_observed": int(cohort.frame[v].notna().sum()),
            "coverage": round(float(cohort.frame[v].notna().mean()), 3),
            "observed_in_ckd": int(cohort.frame.loc[cohort.target == 1, v].notna().sum()),
            "observed_in_non_ckd": int(cohort.frame.loc[cohort.target == 0, v].notna().sum()),
        }
        for v in variables
    ]).sort_values("n_observed", ascending=False)
    coverage.to_csv(tables_dir / "table_41_mimic_cohort.csv", index=False)
    print(f"wrote table_41_mimic_cohort.csv")
    print(coverage.to_string(index=False))

    stage_counts = cohort.stage.value_counts(dropna=False)
    lines = ["MIMIC-IV demo extraction notes", "=" * 34, ""]
    lines += [f"  - {n}" for n in cohort.notes]
    lines += ["", "Stage distribution among coded CKD subjects:"]
    for label, count in stage_counts.items():
        shown = "unspecified / not CKD" if pd.isna(label) else str(label)
        lines.append(f"  {shown:24s} {int(count)}")
    lines += [
        "",
        "Interpretation limits, stated because they bound every downstream claim:",
        "  * The label is administrative coding, not chart review or a KDIGO",
        "    determination; no chronicity criterion is verified.",
        "  * The cohort is a critical-care demo extract of 100 subjects, so it is",
        "    neither a screening series nor a general hospital population.",
        "  * Coverage of the urine dipstick tier is partial, which restricts",
        "    external validation to the shared blood panel plus specific gravity.",
    ]
    notes_path = tables_dir / "table_41_mimic_notes.txt"
    notes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {notes_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
