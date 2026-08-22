"""Stage 11: what pre-discretisation cost (N3 / Phase 3d).

Because the provenance check found the analysed cohort inside a
continuous-valued release of the same patients, the information destroyed
by the published binning can be measured directly and within-cohort.

Writes:
  data/processed/ckd_recovered_continuous.csv   recovered values + provenance
  reports/tables/table_30_recovery_coverage.csv
  reports/tables/table_31_binning_cost.csv
  reports/tables/table_32_imputation_audit.csv
  reports/tables/table_33_sc_bin_structure.csv
  reports/tables/table_34_blood_pressure_recovery.csv
  reports/figures/fig_r14_binning_cost

Gated: refuses to run unless the provenance report classifies the source
release SAME-SOURCE. Recovery is only legitimate *because* the cohorts are
the same; if a future run found them independent, recovering one from the
other would be unfounded.

Usage
-----
    python scripts/11_continuous_recovery.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ckd.config import ensure_dirs, load_config  # noqa: E402
from ckd.data.clean import clean_dataset, feature_matrix  # noqa: E402
from ckd.data.external import load_uci2015  # noqa: E402
from ckd.data.provenance import (  # noqa: E402
    CONTAINMENT_VARIABLES,
    compatibility_matrix,
    unique_pins,
)
from ckd.data.recovery import (  # noqa: E402
    RECOVERABLE,
    bin_structure,
    binning_cost,
    blood_pressure_audit,
    imputation_audit,
    recover,
)
from ckd.evaluation.plots import apply_style, save  # noqa: E402

SOURCE_ID = "uci2015"


def check_gate(tables_dir: Path) -> None:
    path = tables_dir / "table_24_provenance.csv"
    if not path.is_file():
        raise SystemExit(
            "table_24_provenance.csv missing - run scripts/00_provenance.py first."
        )
    verdicts = pd.read_csv(path).set_index("dataset_id")["verdict"].to_dict()
    verdict = verdicts.get(SOURCE_ID)
    if verdict != "SAME-SOURCE":
        raise SystemExit(
            f"Recovery requires {SOURCE_ID} to be SAME-SOURCE (found {verdict!r}). "
            "Recovering one cohort's values from another is only legitimate "
            "when they are the same patients."
        )
    print(f"gate ok: {SOURCE_ID} is {verdict}")


def cost_figure(costs: pd.DataFrame, fig_dir: Path) -> None:
    ordered = costs.sort_values("binning_cost")
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    y = np.arange(len(ordered))
    ax.barh(y, ordered["binning_cost"], color="#0072B2")
    ax.set_yticks(y)
    ax.set_yticklabels([f"`{v}`" for v in ordered["variable"]])
    ax.axvline(0.0, color="0.3", lw=0.9)
    ax.set_xlabel("univariate ROC-AUC lost to interval encoding\n"
                  "(continuous - binned, same patients)")
    ax.set_title("What pre-discretisation cost, per variable")
    for yi, (_, r) in zip(y, ordered.iterrows()):
        ax.text(r["binning_cost"] + 0.004, yi, f"{r['binning_cost']:+.3f}",
                va="center", fontsize=9)
    ax.set_xlim(min(-0.02, ordered["binning_cost"].min() - 0.02),
                ordered["binning_cost"].max() + 0.06)
    fig.tight_layout()
    save(fig, fig_dir, "fig_r14_binning_cost")
    plt.close(fig)


def main() -> int:
    cfg = load_config()
    proc_dir, tables_dir = ensure_dirs(cfg["paths"]["processed_dir"],
                                       cfg["paths"]["tables_dir"])
    (fig_dir,) = ensure_dirs(cfg["paths"]["figures_dir"])
    apply_style()
    check_gate(tables_dir)

    internal = clean_dataset()
    X = feature_matrix(internal)
    y = internal.target.to_numpy()
    source = load_uci2015()

    mappings = {
        k: {b["label"]: (float(b["lower"]), float(b["upper"])) for b in v}
        for k, v in internal.bin_map.items()
    }
    compat = compatibility_matrix(
        internal.clean, y, source.frame, source.target.to_numpy(),
        mappings, CONTAINMENT_VARIABLES,
    )
    pins = unique_pins(compat)
    print(f"uniquely pinned patients: {len(pins)} / {len(y)}")

    result = recover(internal.clean, y, source.frame, pins)

    # ---- recovered dataset with per-cell provenance ----
    out = result.frame.copy()
    out.insert(0, "source_csv_line",
               internal.clean.iloc[result.internal_positions]["source_csv_line"].to_numpy())
    out["target"] = result.target
    for variable in RECOVERABLE:
        out[f"{variable}__provenance"] = result.provenance[variable].to_numpy()
    recovered_path = proc_dir / "ckd_recovered_continuous.csv"
    out.to_csv(recovered_path, index=False)
    print(f"wrote {recovered_path} ({out.shape[0]} patients)")

    coverage = pd.DataFrame([
        {
            "variable": v,
            "n_pinned": len(pins),
            "n_recovered_observed": int(np.isfinite(result.frame[v]).sum()),
            "n_unobserved_in_source": int((~np.isfinite(result.frame[v])).sum()),
            "recovery_rate": round(float(np.isfinite(result.frame[v]).mean()), 4),
        }
        for v in RECOVERABLE
    ])
    coverage.to_csv(tables_dir / "table_30_recovery_coverage.csv", index=False)
    print(f"wrote table_30_recovery_coverage.csv")

    costs = binning_cost(X, result)
    costs.to_csv(tables_dir / "table_31_binning_cost.csv", index=False)
    print("wrote table_31_binning_cost.csv")
    print(costs[["variable", "n_observed", "auc_binned_observed_only",
                 "auc_continuous_observed_only", "binning_cost"]].to_string(index=False))

    audit = imputation_audit(internal.clean, result)
    audit.to_csv(tables_dir / "table_32_imputation_audit.csv", index=False)
    print("\nwrote table_32_imputation_audit.csv")
    print(audit[["variable", "n_unobserved_in_source", "n_distinct_labels_assigned",
                 "assigned_label", "missingness_odds_ratio_ckd"]].to_string(index=False))

    worst = costs.iloc[0]["variable"]
    structure = bin_structure(internal.clean, result, worst)
    structure.to_csv(tables_dir / "table_33_sc_bin_structure.csv", index=False)
    print(f"\nwrote table_33_sc_bin_structure.csv (worst variable: {worst})")
    print(structure.to_string(index=False))

    bp = blood_pressure_audit(internal.clean, result, source.frame)
    bp.to_csv(tables_dir / "table_34_blood_pressure_recovery.csv", index=False)
    print("\nwrote table_34_blood_pressure_recovery.csv")
    print(bp.to_string(index=False))

    cost_figure(costs, fig_dir)
    print("\nwrote fig_r14_binning_cost")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
