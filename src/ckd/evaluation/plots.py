"""Publication-quality figure helpers.

House style: a single colour-blind-safe qualitative palette (Okabe-Ito), a
consistent typographic scale, no chart junk, and every figure saved at 300 dpi
in both PNG and PDF so it can go straight into a manuscript.

Figures that display a deliberately invalid (leaky) result are drawn in a
distinct warning colour and carry an explicit annotation, so a figure lifted
out of this repository cannot be mistaken for a valid result.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.use("Agg")

#: Okabe-Ito palette: distinguishable under the common forms of colour blindness.
OKABE_ITO = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#F0E442",  # yellow
    "#000000",  # black
]

#: Fixed colour per feature configuration so every figure agrees.
CONFIG_COLOURS = {
    "leaky_model": "#B22222",
    "full_valid_model": "#0072B2",
    "laboratory_model": "#009E73",
    "low_cost_model": "#D55E00",
    "low_cost_plus_urine_micro_model": "#E69F00",
    "clinical_only_model": "#CC79A7",
}

CONFIG_LABELS = {
    "leaky_model": "Leaky (INVALID)",
    "full_valid_model": "Full valid",
    "laboratory_model": "Laboratory",
    "low_cost_model": "Low-cost",
    "low_cost_plus_urine_micro_model": "Low-cost + urine microscopy",
    "clinical_only_model": "Clinical only",
}

MODEL_LABELS = {
    "dummy": "Dummy",
    "logreg": "Logistic regression",
    "random_forest": "Random forest",
    "svm": "SVM (RBF)",
    "xgboost": "XGBoost",
    "ebm": "EBM (GAM)",
}

CALIBRATION_LABELS = {
    "none": "Uncalibrated",
    "sigmoid": "Platt / sigmoid",
    "isotonic": "Isotonic",
}


def apply_style() -> None:
    """Install the house style. Called once by each figure-producing script."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
        }
    )


def save(fig, out_dir: Path | str, stem: str, formats: tuple[str, ...] = ("png", "pdf")) -> list[Path]:
    """Save a figure in every requested format and close it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


def annotate_invalid(ax, text: str = "INVALID - leakage demonstration only") -> None:
    """Stamp a panel that shows leaked results."""
    ax.text(
        0.98,
        0.02,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#B22222",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FDECEC", ec="#B22222", lw=0.7),
    )


def config_label(name: str) -> str:
    return CONFIG_LABELS.get(name, name)


def model_label(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def calibration_label(name: str) -> str:
    return CALIBRATION_LABELS.get(name, name)


def config_colour(name: str) -> str:
    return CONFIG_COLOURS.get(name, "#666666")


def forest_plot(ax, labels, estimates, lows, highs, colours=None, xlabel="", reference=None):
    """Horizontal point-and-interval plot, used for every metric-with-CI figure."""
    y = np.arange(len(labels))[::-1]
    colours = colours or ["#0072B2"] * len(labels)
    for yi, est, lo, hi, col in zip(y, estimates, lows, highs, colours):
        ax.plot([lo, hi], [yi, yi], color=col, lw=2.0, solid_capstyle="round", alpha=0.85)
        ax.plot([est], [yi], "o", color=col, ms=6, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    if reference is not None:
        ax.axvline(reference, color="#555555", ls="--", lw=0.9, zorder=0)
    ax.grid(axis="y", visible=False)
    return ax
