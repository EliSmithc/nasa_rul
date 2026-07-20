"""Shared plotting palette and matplotlib defaults for the EDA."""

import matplotlib as mpl

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Categorical slots, assigned in fixed order — never cycled past slot 4 here.
SERIES = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]

# Diverging pair (blue <-> red) with a neutral midpoint, for signed magnitudes.
DIVERGING_LOW = "#2a78d6"
DIVERGING_HIGH = "#d03b3b"
NEUTRAL = "#f0efec"

# Single-hue sequential steps (blue, light -> dark).
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]


def use_style():
    """Apply the shared chart style. Call once per notebook."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.titlecolor": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.labelsize": 10,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
            "legend.frameon": False,
            "legend.labelcolor": INK_SECONDARY,
            "figure.dpi": 110,
        }
    )
