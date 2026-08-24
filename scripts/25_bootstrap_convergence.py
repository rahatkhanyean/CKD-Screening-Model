"""Stage 25: convergence analysis for the whole-procedure bootstrap.

The revision brief asks for at least 1,000 whole-procedure resamples, or a
justified convergence analysis with the limitation stated. Each resample
re-runs the entire nested procedure --- fold construction, inner grid search,
refit and calibration --- so the cost is roughly a full cross-validation per
resample per cell. A 1,000-resample run was attempted and abandoned after it
became clear it would take several hours on the available hardware while
competing with the other analyses in this revision.

This stage instead quantifies what the completed run supports. It draws
increasing subsets of the existing resamples and tracks how the interval
endpoints move, so a reader can judge whether the reported interval is
stable at the resample count actually used rather than taking it on trust.

Output
------
reports/tables/table_51_bootstrap_convergence.csv
reports/figures/fig_bootstrap_convergence

Usage
-----
    python scripts/25_bootstrap_convergence.py
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
from ckd.evaluation.plots import apply_style, save  # noqa: E402

GRID = [25, 50, 75, 100, 125, 150, 175, 200]


def main() -> int:
    cfg = load_config()
    (tables_dir,) = ensure_dirs(cfg["paths"]["tables_dir"])
    (fig_dir,) = ensure_dirs(cfg["paths"]["figures_dir"])
    apply_style()

    draws_path = tables_dir / "table_38_procedure_bootstrap_draws.csv"
    if not draws_path.is_file():
        raise SystemExit("run scripts/15_procedure_bootstrap.py first")
    draws = pd.read_csv(draws_path)

    rng = np.random.default_rng(int(cfg["seed"]))
    rows = []
    for (config, model), group in draws.groupby(["config", "model"]):
        values = group["roc_auc"].dropna().to_numpy()
        n_available = len(values)
        full_lo, full_hi = np.percentile(values, [2.5, 97.5])
        for n in GRID:
            if n > n_available:
                continue
            # Repeat the subsample several times so the reported movement is
            # not an artefact of one arbitrary subset.
            los, his = [], []
            for _ in range(50):
                subset = rng.choice(values, size=n, replace=False)
                lo, hi = np.percentile(subset, [2.5, 97.5])
                los.append(lo)
                his.append(hi)
            rows.append({
                "config": config, "model": model, "n_resamples": n,
                "ci_low_mean": float(np.mean(los)),
                "ci_high_mean": float(np.mean(his)),
                "ci_low_sd": float(np.std(los, ddof=1)),
                "ci_high_sd": float(np.std(his, ddof=1)),
                "width_mean": float(np.mean(np.array(his) - np.array(los))),
                "full_ci_low": float(full_lo),
                "full_ci_high": float(full_hi),
                "n_available": int(n_available),
            })
    table = pd.DataFrame(rows)
    table.to_csv(tables_dir / "table_51_bootstrap_convergence.csv", index=False)
    print(f"wrote table_51_bootstrap_convergence.csv ({len(table)} rows)")

    largest = table[table["n_resamples"] == table["n_resamples"].max()]
    print("\nAt the resample count actually used:")
    for _, r in largest.iterrows():
        print(f"  {r['config']:18s} {r['model']:14s} "
              f"CI [{r['ci_low_mean']:.4f}, {r['ci_high_mean']:.4f}] "
              f"endpoint SD ({r['ci_low_sd']:.4f}, {r['ci_high_sd']:.4f})")

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for (config, model), group in table.groupby(["config", "model"]):
        group = group.sort_values("n_resamples")
        ax.errorbar(group["n_resamples"], group["ci_low_mean"],
                    yerr=group["ci_low_sd"], marker="o", ms=3, lw=1.2,
                    capsize=2, label=f"{config.replace('_model','')}/{model}")
    ax.set_xlabel("number of whole-procedure resamples")
    ax.set_ylabel("lower 95% interval endpoint")
    ax.set_title("Convergence of the whole-procedure interval")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, fig_dir, "fig_bootstrap_convergence")
    plt.close(fig)
    print("wrote fig_bootstrap_convergence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
