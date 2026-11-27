"""Shared publication-grade academic figure style (blue-white visual system)."""

from __future__ import annotations

import matplotlib.pyplot as plt


# Cohesive academic palette matching the blue-white presentation theme
BLUE_DEEP = "#073DA5"
BLUE_MID = "#2563EB"
TEAL = "#0F766E"
AMBER = "#C2591A"
PURPLE = "#6D28D9"
CRIMSON = "#C81E1E"
SLATE = "#475569"
INK = "#111722"
MUTED = "#5A6A80"
GRID = "#DCE6F2"

COLORS = [BLUE_DEEP, TEAL, AMBER, PURPLE, CRIMSON, "#0369A1", "#3F6212", "#9A3412", SLATE]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#334155",
            "axes.linewidth": 0.9,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linestyle": "--",
            "grid.linewidth": 0.75,
            "grid.alpha": 0.9,
            "xtick.color": "#243447",
            "ytick.color": "#243447",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 10.5,
            "legend.frameon": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.14,
        }
    )
