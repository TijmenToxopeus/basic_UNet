"""Shared plotting style helpers for notebooks and analysis scripts."""

from __future__ import annotations

import matplotlib as mpl

PUB_COLORS = {
    "l1": "#0E6A7A",        # deep teal
    "corr": "#9A3D6A",      # muted magenta
    "good": "#2B8A3E",      # deep green
    "warn": "#C62828",      # red
    "neutral": "#C27D2C",   # warm amber

    # additional colors
    "blue": "#1F4E79",      # dark academic blue
    "light_blue": "#4C8EDA",
    "purple": "#6A4C93",
    "soft_purple": "#8E7DBE",
    "orange": "#D17A22",
    "soft_orange": "#E6A756",
    "olive": "#6B8E23",
    "slate": "#4A5568",     # neutral slate grey
    "light_grey": "#9CA3AF",
    "dark_grey": "#374151",

    # report figure accents (high-contrast)
    "report_ul1": "#1F77B4",      # blue
    "report_upearson": "#FF7F0E", # orange
    "report_rl1": "#2CA02C",      # green
    "report_rpearson": "#D62728", # red
    "report_du": "#9467BD",       # purple
    "report_dr": "#8C564B",       # brown

    # pastel palette
    "pastel_pink": "#F4A7B9",
    "pastel_peach": "#F7C59F",
    "pastel_yellow": "#F6E7A1",
    "pastel_mint": "#A8E6CF",
    "pastel_sky": "#AFCBFF",
    "pastel_lilac": "#CDB4DB",
    "pastel_teal": "#9ADBCF",
    "pastel_gray": "#D8D8D8",


# Generic default palette (10 standard, high-contrast colors)
    "blue": "#1F77B4",  # blue
    "orange": "#FF7F0E",  # orange
    "green": "#2CA02C",  # green
    "red": "#D62728",  # red
    "purple": "#9467BD",  # purple
    "brown": "#8C564B",  # brown
    "pink": "#E377C2",  # pink
    "gray": "#7F7F7F",  # gray
    "olive": "#BCBD22",  # olive
    "cyan": "#17BECF",  # cyan
}

PUBLICATION_RCPARAMS = {
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.45,
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
