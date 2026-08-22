"""Stage 13: render the TRIPOD+AI checklist as a report appendix (N11 / 5a).

Input ``data/literature/tripod_ai_checklist.csv``: one row per checklist
item with a status and the artefact that satisfies it. Statuses are
deliberately allowed to record failure - ``partly`` and ``not-satisfied``
are visible values, so the appendix is an audit rather than an assertion
of compliance.

Output ``reports/tables/table_36_tripod_ai.csv`` plus a summary table,
both injected into the report by stage 6.

Usage
-----
    python scripts/13_tripod_checklist.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ckd.config import ensure_dirs, load_config, project_root  # noqa: E402

STATUSES = {"satisfied", "partly", "not-satisfied", "not-applicable", "adapted"}
REQUIRED_COLUMNS = ["item", "section", "topic", "requirement", "status", "evidence"]


def main() -> int:
    cfg = load_config()
    (tables_dir,) = ensure_dirs(cfg["paths"]["tables_dir"])
    path = project_root() / "data" / "literature" / "tripod_ai_checklist.csv"
    table = pd.read_csv(path, dtype=str).fillna("")

    missing = [c for c in REQUIRED_COLUMNS if c not in table.columns]
    if missing:
        raise SystemExit(f"checklist missing columns: {missing}")
    bad = set(table["status"]) - STATUSES
    if bad:
        raise SystemExit(f"invalid statuses {bad}; allowed: {sorted(STATUSES)}")
    empty = table[table[REQUIRED_COLUMNS].eq("").any(axis=1)]
    if len(empty):
        raise SystemExit(f"empty cells in items: {list(empty['item'])}")

    table.to_csv(tables_dir / "table_36_tripod_ai.csv", index=False)
    summary = (
        table["status"].value_counts().rename_axis("status")
        .reset_index(name="n_items")
    )
    summary.to_csv(tables_dir / "table_36_tripod_ai_summary.csv", index=False)
    print(f"wrote table_36_tripod_ai.csv ({len(table)} items)")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
