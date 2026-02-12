"""Shared plotting style helpers for notebooks and analysis scripts."""

from __future__ import annotations

import matplotlib as mpl

PUB_COLORS = {
    "l1": "#0E6A7A",  # deep teal
    "corr": "#9A3D6A",  # muted magenta
    "good": "#2B8A3E",
    "warn": "#C62828",
    "neutral": "#C27D2C",  # warm amber
}

PUBLICATION_RCPARAMS = {
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linestyle": "--",
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "font.size": 10,
}


def apply_publication_style() -> None:
    """Apply the shared publication-style matplotlib defaults."""
    mpl.rcParams.update(PUBLICATION_RCPARAMS)


def despine(ax) -> None:
    """Hide top/right spines on an axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
