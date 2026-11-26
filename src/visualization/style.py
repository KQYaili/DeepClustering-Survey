"""Shared figure style. English labels avoid missing CJK fonts in WSL."""

from __future__ import annotations

import matplotlib.pyplot as plt


COLORS = ["#1E3A5F", "#D97706", "#0F766E", "#B91C1C", "#6D28D9", "#0369A1", "#3F6212", "#9A3412", "#475569"]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#0F172A",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E2E8F0",
            "grid.linestyle": "--",
            "font.size": 11,
            "axes.titlesize": 13,
            "legend.frameon": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
