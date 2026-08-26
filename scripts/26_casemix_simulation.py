"""Stage 26: what measured performance would be under a different case mix.

The manuscript's case-composition argument rests on a counting fact --- 107 of
128 cases carry stage 3--5 disease --- and on the observation that removing
information pathways does not restore headroom. Neither shows what would
happen if the *severity distribution itself* were different. This stage
estimates that directly, and quantifies how much the estimate can be trusted.

What this does
--------------
Holding the fitted models fixed, it re-weights the out-of-fold predictions to
target severity distributions by resampling cases with replacement, keeping
every control, and recomputing the metrics. Monte Carlo uncertainty is
reported from the resampling itself.

What this cannot do, stated plainly
-----------------------------------
1. The models were selected and fitted on the observed mix. This answers
   "what would we have *measured* on a differently composed cohort with this
   model", not "what would a model trained on that cohort achieve".
2. Within-stage distributions are preserved only in the sense that the same
   patients are reused; no new patients are synthesised.
3. The benchmark contains 9 stage-1 and 12 stage-2 cases. An early-heavy
   target therefore resamples a handful of real patients many times over.
   ``n_distinct_source_cases`` and ``max_times_one_patient_used`` are reported
   for every scenario so a reader can see how thin the support is, and no
   scenario whose support is this thin should be read as an estimate of
   performance in an early-stage population.
4. This is a resampling exercise, not prospective evidence.

Outputs
-------
reports/tables/table_54_casemix_simulation.csv   metrics under each target mix
reports/tables/table_55_severity_balanced.csv    severity-balanced resampling
reports/figures/fig_casemix_simulation           metric against early fraction

Usage
-----
    python scripts/26_casemix_simulation.py [--draws N]
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

from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    roc_auc_score,
)

from ckd.config import ensure_dirs, load_config  # noqa: E402
from ckd.data.clean import clean_dataset  # noqa: E402
from ckd.evaluation.bootstrap import pooled_predictions  # noqa: E402
from ckd.evaluation.plots import apply_style, save  # noqa: E402

CONFIGS = ["low_cost_model", "laboratory_model", "full_valid_model"]
STAGES = ("s1", "s2", "s3", "s4", "s5")

#: Target severity distributions over CKD cases. ``None`` means "as sampled".
#: The community series is the published distribution already used for the
#: standardisation analysis; the remainder are deliberate contrasts spanning
#: the range from early-heavy to advanced-heavy.
TARGET_MIXES: dict[str, dict[str, float] | None] = {
    "as sampled (benchmark)": None,
    "community screening series": {"s1": 0.22, "s2": 0.46, "s3": 0.32,
                                   "s4": 0.0, "s5": 0.0},
    "early-heavy": {"s1": 0.40, "s2": 0.40, "s3": 0.20, "s4": 0.0, "s5": 0.0},
    "uniform across stages": {s: 0.2 for s in STAGES},
    "advanced-heavy": {"s1": 0.0, "s2": 0.0, "s3": 0.20, "s4": 0.40,
                       "s5": 0.40},
}


def draw_metrics(
    y: np.ndarray,
    prob: np.ndarray,
    stage: np.ndarray,
    mix: dict[str, float] | None,
    n_draws: int,
    seed: int,
    threshold: float = 0.5,
) -> dict:
    """Metrics under a target severity mix, with Monte Carlo uncertainty."""
    rng = np.random.default_rng(seed)
    case_idx = np.flatnonzero(y == 1)
    control_idx = np.flatnonzero(y == 0)
    n_cases = len(case_idx)

    by_stage = {s: case_idx[stage[case_idx] == s] for s in STAGES}
    available = {s: len(v) for s, v in by_stage.items()}

    if mix is None:
        counts = {s: available[s] for s in STAGES}
    else:
        total = sum(mix.values())
        counts = {s: int(round(n_cases * mix[s] / total)) for s in STAGES}
        # Never request cases from a stratum the benchmark does not contain.
        counts = {s: (c if available[s] > 0 else 0) for s, c in counts.items()}

    requested = {s: c for s, c in counts.items() if c > 0}
    if not requested or any(available[s] == 0 for s in requested):
        return {}

    aucs, praucs, senss, distinct, reuse = [], [], [], [], []
    for _ in range(n_draws):
        picked = np.concatenate([
            rng.choice(by_stage[s], size=c, replace=True)
            for s, c in requested.items()
        ])
        idx = np.concatenate([picked, control_idx])
        yy, pp = y[idx], prob[idx]
        if len(np.unique(yy)) < 2:
            continue
        aucs.append(roc_auc_score(yy, pp))
        praucs.append(average_precision_score(yy, pp))
        pos = yy == 1
        senss.append(float(((pp >= threshold) & pos).sum() / max(pos.sum(), 1)))
        uniq, cts = np.unique(picked, return_counts=True)
        distinct.append(len(uniq))
        reuse.append(int(cts.max()))

    if not aucs:
        return {}

    def summarise(values: list[float], name: str) -> dict:
        arr = np.asarray(values, dtype=float)
        return {
            name: float(arr.mean()),
            f"{name}_mc_low": float(np.percentile(arr, 2.5)),
            f"{name}_mc_high": float(np.percentile(arr, 97.5)),
            f"{name}_mc_se": float(arr.std(ddof=1) / np.sqrt(len(arr))),
        }

    out = {
        "n_cases_drawn": int(sum(requested.values())),
        "n_controls": int(len(control_idx)),
        "early_case_fraction": float(
            sum(requested.get(s, 0) for s in ("s1", "s2"))
            / max(sum(requested.values()), 1)),
        "n_distinct_source_cases": float(np.mean(distinct)),
        "max_times_one_patient_used": int(np.max(reuse)),
        "strata_requested": ";".join(f"{s}:{c}" for s, c in requested.items()),
        "strata_available": ";".join(f"{s}:{available[s]}" for s in STAGES),
        "n_monte_carlo_draws": len(aucs),
    }
    out.update(summarise(aucs, "roc_auc"))
    out.update(summarise(praucs, "pr_auc"))
    out.update(summarise(senss, "sensitivity"))
    return out


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()

    proc_dir, tables_dir = ensure_dirs(cfg["paths"]["processed_dir"],
                                       cfg["paths"]["tables_dir"])
    (fig_dir,) = ensure_dirs(cfg["paths"]["figures_dir"])
    apply_style()

    seed = int(cfg["seed"])
    internal = clean_dataset()
    stage_all = internal.clean["stage"].astype("object").to_numpy()
    pooled = pooled_predictions(pd.read_csv(proc_dir / "cv_predictions.csv.gz"))

    rows: list[dict] = []
    balanced: list[dict] = []
    print("case-mix simulation (models held fixed; predictions re-weighted)")
    for config_name in CONFIGS:
        cell = pooled[(pooled["config"] == config_name)
                      & (pooled["calibration"] == "isotonic")
                      & (pooled["model"] != "dummy")]
        best = (cell.groupby("model")
                .apply(lambda g: roc_auc_score(g["y_true"], g["y_prob"]),
                       include_groups=False).idxmax())
        chosen = cell[cell["model"] == best].sort_values("sample_index")
        idx = chosen["sample_index"].to_numpy()
        y = chosen["y_true"].to_numpy()
        prob = chosen["y_prob"].to_numpy()
        stage = stage_all[idx]

        print(f"\n  {config_name} (model: {best})")
        for label, mix in TARGET_MIXES.items():
            m = draw_metrics(y, prob, stage, mix, args.draws, seed)
            if not m:
                print(f"    {label:30s} not estimable from this benchmark")
                continue
            rows.append({"config": config_name, "model": best,
                         "target_mix": label, **m})
            print(f"    {label:30s} early={m['early_case_fraction']:.2f} "
                  f"AUC {m['roc_auc']:.3f} "
                  f"({m['roc_auc_mc_low']:.3f}-{m['roc_auc_mc_high']:.3f}) "
                  f"sens {m['sensitivity']:.3f} "
                  f"[{int(m['n_distinct_source_cases'])} distinct cases, "
                  f"one used up to {m['max_times_one_patient_used']}x]")

        # Severity-balanced resampling: equal numbers per stage, sized to the
        # smallest stratum so that no patient is reused at all. This is the
        # only variant here that avoids replacement entirely, and it is
        # correspondingly tiny.
        case_idx = np.flatnonzero(y == 1)
        per_stage = min(int((stage[case_idx] == s).sum()) for s in STAGES)
        rng = np.random.default_rng(seed)
        aucs, senss = [], []
        for _ in range(args.draws):
            picked = np.concatenate([
                rng.choice(case_idx[stage[case_idx] == s], size=per_stage,
                           replace=False)
                for s in STAGES
            ])
            sel = np.concatenate([picked, np.flatnonzero(y == 0)])
            if len(np.unique(y[sel])) < 2:
                continue
            aucs.append(roc_auc_score(y[sel], prob[sel]))
            pos = y[sel] == 1
            senss.append(float(((prob[sel] >= 0.5) & pos).sum()
                               / max(pos.sum(), 1)))
        balanced.append({
            "config": config_name, "model": best,
            "cases_per_stage": per_stage,
            "n_cases": per_stage * len(STAGES),
            "sampling": "without replacement",
            "roc_auc": float(np.mean(aucs)),
            "roc_auc_mc_low": float(np.percentile(aucs, 2.5)),
            "roc_auc_mc_high": float(np.percentile(aucs, 97.5)),
            "sensitivity": float(np.mean(senss)),
            "sensitivity_mc_low": float(np.percentile(senss, 2.5)),
            "sensitivity_mc_high": float(np.percentile(senss, 97.5)),
            "n_monte_carlo_draws": len(aucs),
        })
        b = balanced[-1]
        print(f"    severity-balanced ({per_stage}/stage, no replacement): "
              f"AUC {b['roc_auc']:.3f} "
              f"({b['roc_auc_mc_low']:.3f}-{b['roc_auc_mc_high']:.3f})")

    frame = pd.DataFrame(rows)
    frame.to_csv(tables_dir / "table_54_casemix_simulation.csv", index=False)
    print(f"\nwrote table_54_casemix_simulation.csv ({len(frame)} rows)")
    bframe = pd.DataFrame(balanced)
    bframe.to_csv(tables_dir / "table_55_severity_balanced.csv", index=False)
    print(f"wrote table_55_severity_balanced.csv ({len(bframe)} rows)")

    # ---- figure ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for config_name, grp in frame.groupby("config"):
        g = grp.sort_values("early_case_fraction")
        for ax, metric in zip(axes, ("roc_auc", "sensitivity")):
            ax.errorbar(
                g["early_case_fraction"], g[metric],
                yerr=[g[metric] - g[f"{metric}_mc_low"],
                      g[f"{metric}_mc_high"] - g[metric]],
                marker="o", capsize=3, lw=1.2, label=config_name,
            )
    for ax, title in zip(axes, ("ROC-AUC", "Sensitivity at p = 0.50")):
        ax.set_xlabel("fraction of cases at stage 1-2 in the target mix")
        ax.set_ylabel(title)
        ax.set_title(title)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Measured performance under re-weighted case mixes "
                 "(EXPLORATORY; models held fixed)", fontweight="bold")
    fig.text(0.5, -0.06,
             "Resampling of a fixed model's out-of-fold predictions, not "
             "prospective evidence. Early-heavy mixes reuse as few as 9 "
             "stage-1 and 12 stage-2 patients; bars are Monte Carlo "
             "intervals and do not include that support limitation.",
             ha="center", fontsize=8, style="italic", color="#444444")
    save(fig, fig_dir, "fig_casemix_simulation")
    print("wrote fig_casemix_simulation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
