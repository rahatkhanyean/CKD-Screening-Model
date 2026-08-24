"""Stage 21: sensitivity analyses for the cross-release provenance verdict.

Runs the suite in ``ckd.data.provenance_sensitivity`` and writes the tables
and figure the manuscript uses to defend its principal claim.

Outputs
-------
reports/tables/table_44_provenance_sensitivity.csv   all matching variants
reports/tables/table_45_matched_pair_audit.csv       reproducible pair sample
reports/tables/table_46_provenance_nulls.csv         both null distributions
reports/figures/fig_provenance_sensitivity           forest + null panel

Usage
-----
    python scripts/21_provenance_sensitivity.py [--permutations N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ckd.config import ensure_dirs, load_config  # noqa: E402
from ckd.data.clean import clean_dataset  # noqa: E402
from ckd.data.external import load_uci2015  # noqa: E402
from ckd.data.provenance import (  # noqa: E402
    CONTAINMENT_VARIABLES,
    compatibility_matrix,
    containment_check,
    matching_fraction,
    unique_pins,
)
from ckd.data.provenance_sensitivity import (  # noqa: E402
    leave_one_out,
    match_once,
    copula_null,
    row_permutation_is_degenerate,
    sample_matched_pairs,
    tolerance_sensitivity,
)
from ckd.evaluation.plots import apply_style, save  # noqa: E402

#: Variables whose values in the analysed release are most affected by the
#: undocumented constant imputation documented in stage 11.
IMPUTATION_AFFECTED = ("rbcc", "wbcc", "pcv", "pot", "sod")

#: Cheap, non-laboratory subset: tests whether the match survives on
#: variables a clinician could record without an analyser.
CHEAP_SUBSET = ("age", "sg", "al", "su")

#: Laboratory-only subset.
LAB_SUBSET = ("bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=200)
    args = parser.parse_args()

    cfg = load_config()
    (tables_dir,) = ensure_dirs(cfg["paths"]["tables_dir"])
    (fig_dir,) = ensure_dirs(cfg["paths"]["figures_dir"])
    apply_style()

    internal = clean_dataset()
    y = internal.target.to_numpy()
    source = load_uci2015()
    cand = source.frame
    cand_y = source.target.to_numpy()
    mappings = {
        k: {b["label"]: (float(b["lower"]), float(b["upper"])) for b in v}
        for k, v in internal.bin_map.items()
    }

    rows: list[dict] = []

    def record(summary, analysis: str, **extra) -> None:
        rows.append({
            "analysis": analysis,
            "label": summary.label,
            "n_variables": summary.n_variables,
            "outcome_used": summary.outcome_used,
            "match_fraction": round(summary.match_fraction, 4),
            "n_unique_pins": summary.n_unique_pins,
            "rows_with_any_partner": summary.rows_with_any_partner,
            "mean_partners": round(summary.mean_partners, 3),
            "median_partners": summary.median_partners,
            "max_partners": summary.max_partners,
            **extra,
        })

    print("primary and variant matching configurations ...")
    record(match_once(internal.clean, y, cand, cand_y, mappings,
                      CONTAINMENT_VARIABLES, "primary (all 13, outcome used)"),
           "primary")
    record(match_once(internal.clean, y, cand, cand_y, mappings,
                      CONTAINMENT_VARIABLES, "outcome excluded",
                      use_outcome=False),
           "outcome_excluded")
    record(match_once(internal.clean, y, cand, cand_y, mappings,
                      tuple(v for v in CONTAINMENT_VARIABLES
                            if v not in IMPUTATION_AFFECTED),
                      "imputation-affected variables removed"),
           "imputation_removed")
    record(match_once(internal.clean, y, cand, cand_y, mappings,
                      CHEAP_SUBSET, "cheap subset (age, sg, al, su)"),
           "feature_subset")
    record(match_once(internal.clean, y, cand, cand_y, mappings,
                      LAB_SUBSET, "laboratory subset"),
           "feature_subset")
    for row in rows:
        print(f"  {row['label']:42s} match={row['match_fraction']:.3f} "
              f"unique_pins={row['n_unique_pins']:3d} "
              f"max_partners={row['max_partners']}")

    print("\nleave-one-variable-out ...")
    loo = leave_one_out(internal.clean, y, cand, cand_y, mappings)
    print(f"  match fraction range: {loo['match_fraction'].min():.3f}"
          f"-{loo['match_fraction'].max():.3f}; unique pins "
          f"{loo['n_unique_pins'].min()}-{loo['n_unique_pins'].max()}")

    print("\ntolerance sensitivity ...")
    tol = tolerance_sensitivity(internal.clean, y, cand, cand_y, mappings)
    print(tol[["tolerance", "match_fraction", "n_unique_pins",
               "max_partners"]].to_string(index=False))

    print("\nnegative controls ...")
    # Control 1: shuffle the candidate's row order within outcome strata is
    # handled by the null below. Control 2: match the analysed file against
    # a synthetic cohort with the same marginals but independent records.
    rng = np.random.default_rng(int(cfg["seed"]))
    synthetic = cand.copy()
    for variable in CONTAINMENT_VARIABLES:
        synthetic[variable] = rng.permutation(synthetic[variable].to_numpy())
    record(match_once(internal.clean, y, synthetic, cand_y, mappings,
                      CONTAINMENT_VARIABLES,
                      "negative control: marginal-preserving synthetic"),
           "negative_control")
    print(f"  synthetic control match fraction: {rows[-1]['match_fraction']:.4f}")

    sensitivity = pd.concat(
        [pd.DataFrame(rows), loo, tol], ignore_index=True, sort=False
    )
    sensitivity.to_csv(tables_dir / "table_44_provenance_sensitivity.csv",
                       index=False)
    print(f"\nwrote table_44_provenance_sensitivity.csv ({len(sensitivity)} rows)")

    # ---- nulls -----------------------------------------------------------
    print("\nnull distributions ...")
    primary = containment_check(internal.clean, y, cand, cand_y, mappings,
                                CONTAINMENT_VARIABLES,
                                n_shuffles=args.permutations,
                                seed=int(cfg["seed"]))
    col_null = np.random.default_rng(0)  # placeholder, replaced below
    # Recompute the column-permutation draws explicitly so both nulls are
    # written to disk rather than only summarised.
    from ckd.data.provenance import compatibility_matrix as _cm

    rng2 = np.random.default_rng(int(cfg["seed"]))
    col_draws = np.empty(args.permutations)
    for k in range(args.permutations):
        shuffled = cand.copy()
        for variable in CONTAINMENT_VARIABLES:
            shuffled[variable] = (
                shuffled[variable]
                .sample(frac=1.0, random_state=int(rng2.integers(2**31 - 1)))
                .to_numpy()
            )
        col_draws[k] = matching_fraction(
            _cm(internal.clean, y, shuffled, cand_y, mappings,
                CONTAINMENT_VARIABLES)
        )
    row_draws = copula_null(
        internal.clean, y, cand, cand_y, mappings,
        n_draws=args.permutations, seed=int(cfg["seed"]),
    )
    obs_chk, perm_chk = row_permutation_is_degenerate(
        internal.clean, y, cand, cand_y, mappings, seed=int(cfg["seed"]))
    print(f"  [check] whole-row permutation is degenerate for bipartite "
          f"matching: {obs_chk:.4f} -> {perm_chk:.4f} (identical by construction)")
    nulls = pd.DataFrame({
        "draw": np.arange(args.permutations),
        "column_permutation": col_draws,
        "copula_preserving_correlations": row_draws,
    })
    nulls.to_csv(tables_dir / "table_46_provenance_nulls.csv", index=False)
    observed = float(sensitivity.loc[0, "match_fraction"])
    print(f"  observed              {observed:.4f}")
    print(f"  column permutation    {col_draws.mean():.4f} +/- "
          f"{col_draws.std(ddof=1):.4f} (max {col_draws.max():.4f})")
    print(f"  copula null           {row_draws.mean():.4f} +/- "
          f"{row_draws.std(ddof=1):.4f} (max {row_draws.max():.4f})")

    # ---- matched-pair audit ---------------------------------------------
    compat = compatibility_matrix(internal.clean, y, cand, cand_y, mappings,
                                  CONTAINMENT_VARIABLES)
    pins = unique_pins(compat)
    audit = sample_matched_pairs(internal.clean, cand, pins,
                                 CONTAINMENT_VARIABLES, n=12)
    audit.to_csv(tables_dir / "table_45_matched_pair_audit.csv", index=False)
    print(f"\nwrote table_45_matched_pair_audit.csv ({len(audit)} pairs)")

    # ---- figure ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    forest = sensitivity[sensitivity["analysis"].isin(
        ["primary", "outcome_excluded", "imputation_removed",
         "feature_subset", "negative_control"])].copy()
    forest = forest.iloc[::-1]
    ypos = np.arange(len(forest))
    colours = ["#a0453a" if "negative control" in str(lbl) else "#2c6e8f"
               for lbl in forest["label"]]
    axes[0].barh(ypos, forest["match_fraction"], color=colours)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(
        [str(l).replace("negative control: ", "NEG: ")[:38]
         for l in forest["label"]], fontsize=8)
    axes[0].set_xlabel("maximum bipartite match fraction")
    axes[0].set_xlim(0, 1.05)
    axes[0].axvline(1.0, ls=":", lw=1.0, color="0.5")
    axes[0].set_title("Matching under varied conditions", fontsize=10)
    for yi, v in zip(ypos, forest["match_fraction"]):
        axes[0].text(min(v + 0.02, 0.98), yi, f"{v:.3f}", va="center",
                     fontsize=8)

    axes[1].hist(col_draws, bins=20, alpha=0.75, color="#7d8b9b",
                 label="column permutation")
    axes[1].hist(row_draws, bins=20, alpha=0.75, color="#9c6b15",
                 label="row permutation\n(within outcome)")
    axes[1].axvline(observed, color="#a0453a", lw=2.2,
                    label=f"observed {observed:.3f}")
    axes[1].set_xlabel("match fraction")
    axes[1].set_ylabel("permutations")
    axes[1].set_xlim(-0.03, 1.05)
    axes[1].legend(fontsize=8, frameon=False)
    axes[1].set_title("Null distributions", fontsize=10)

    fig.tight_layout()
    save(fig, fig_dir, "fig_provenance_sensitivity")
    plt.close(fig)
    print("wrote fig_provenance_sensitivity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
