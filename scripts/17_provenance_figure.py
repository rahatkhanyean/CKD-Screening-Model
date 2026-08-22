"""Stage 17: figure for the cohort-overlap evidence (paper figure).

The provenance verdict rests on a comparison between one observed number
and a null distribution. Stage 0 records the summary statistics; this
stage recomputes the permutation draws so the comparison can be shown
rather than asserted, and writes them to a table so the figure is
reproducible from committed data like every other figure here.

Output:
  reports/tables/table_40_provenance_null_draws.csv
  reports/figures/fig_r15_provenance_null

Usage
-----
    python scripts/17_provenance_figure.py
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
from ckd.data.clean import clean_dataset  # noqa: E402
from ckd.data.external import load_uci2015  # noqa: E402
from ckd.data.provenance import (  # noqa: E402
    CONTAINMENT_VARIABLES,
    compatibility_matrix,
    matching_fraction,
)
from ckd.evaluation.plots import apply_style, save  # noqa: E402

N_SHUFFLES = 200


def main() -> int:
    cfg = load_config()
    (tables_dir,) = ensure_dirs(cfg["paths"]["tables_dir"])
    (fig_dir,) = ensure_dirs(cfg["paths"]["figures_dir"])
    apply_style()

    internal = clean_dataset()
    y = internal.target.to_numpy()
    source = load_uci2015()
    mappings = {
        k: {b["label"]: (float(b["lower"]), float(b["upper"])) for b in v}
        for k, v in internal.bin_map.items()
    }

    observed = matching_fraction(
        compatibility_matrix(internal.clean, y, source.frame,
                             source.target.to_numpy(), mappings,
                             CONTAINMENT_VARIABLES)
    )
    print(f"observed match fraction: {observed:.4f}")

    rng = np.random.default_rng(int(cfg["seed"]))
    draws = np.empty(N_SHUFFLES)
    for k in range(N_SHUFFLES):
        shuffled = source.frame.copy()
        for variable in CONTAINMENT_VARIABLES:
            shuffled[variable] = (
                shuffled[variable]
                .sample(frac=1.0, random_state=int(rng.integers(2**31 - 1)))
                .to_numpy()
            )
        draws[k] = matching_fraction(
            compatibility_matrix(internal.clean, y, shuffled,
                                 source.target.to_numpy(), mappings,
                                 CONTAINMENT_VARIABLES)
        )
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{N_SHUFFLES} shuffles")

    table = pd.DataFrame({
        "shuffle": np.arange(N_SHUFFLES),
        "null_match_fraction": draws,
    })
    table.attrs["observed"] = observed
    table.to_csv(tables_dir / "table_40_provenance_null_draws.csv", index=False)
    print(f"wrote table_40_provenance_null_draws.csv "
          f"(null mean {draws.mean():.4f}, max {draws.max():.4f})")

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.hist(draws, bins=24, color="#7d8b9b", edgecolor="white", linewidth=0.6,
            label=f"permutation null ({N_SHUFFLES} shuffles)")
    ax.axvline(observed, color="#a0453a", lw=2.4,
               label=f"observed = {observed:.3f}")
    ax.annotate(
        f"observed\n{observed:.3f}",
        xy=(observed, ax.get_ylim()[1] * 0.62),
        xytext=(observed - 0.16, ax.get_ylim()[1] * 0.80),
        color="#a0453a", fontsize=10, ha="center",
        arrowprops=dict(arrowstyle="->", color="#a0453a", lw=1.3),
    )
    ax.annotate(
        f"null: {draws.mean():.3f} $\\pm$ {draws.std(ddof=1):.3f}\n"
        f"(max {draws.max():.3f})",
        xy=(draws.mean(), ax.get_ylim()[1] * 0.55),
        xytext=(0.16, ax.get_ylim()[1] * 0.72),
        color="#3f4b58", fontsize=9, ha="left",
        arrowprops=dict(arrowstyle="->", color="#6b7787", lw=1.0),
    )
    ax.set_xlim(-0.03, 1.05)
    ax.set_xlabel("fraction of the 200 analysed patients matched into the 2015 release")
    ax.set_ylabel("permutations")
    ax.legend(loc="upper center", frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, fig_dir, "fig_r15_provenance_null")
    plt.close(fig)
    print("wrote fig_r15_provenance_null")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
