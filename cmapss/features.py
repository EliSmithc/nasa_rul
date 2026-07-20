"""Feature engineering for FD001 RUL models.

Column selection follows the EDA in ``notebooks/01_explore.ipynb``:
drop constant/binary channels, drop the redundant and per-unit-inconsistent
sensors 9/14 (and 8/13), keep the ten consistently monotonic sensors.
"""

import numpy as np
import pandas as pd

# Consistently monotonic within essentially every engine (|rho| ~ 0.63-0.81).
RETAINED_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11",
    "sensor_12", "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]

# Sensors flatten above ~125 cycles remaining; cap the training target there.
RUL_CAP = 125

# Shortest test unit has 31 observed cycles — windows must stay under that,
# and min_periods=1 keeps the first rows of every unit usable.
WINDOW = 20


def _slope_of(y):
    """Per-cycle OLS slope of a window of values against elapsed time."""
    x = np.arange(len(y), dtype=float)
    x -= x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum())


def build_features(df, window=WINDOW):
    """Return ``(X, meta)`` for a trajectory frame from ``load_train``/``load_test``.

    Features per retained sensor: current value, trailing rolling mean, and
    trailing rolling slope (all within-unit, so no leakage across engines).
    ``meta`` carries ``unit`` and ``cycle`` aligned with ``X``'s rows.
    """
    df = df.sort_values(["unit", "cycle"])
    g = df.groupby("unit")

    parts = {"cycle": df["cycle"]}
    for c in RETAINED_SENSORS:
        parts[c] = df[c]
        parts[f"{c}_mean{window}"] = g[c].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        parts[f"{c}_slope{window}"] = g[c].transform(
            lambda s: s.rolling(window, min_periods=2).apply(_slope_of, raw=True)
        ).fillna(0.0)

    X = pd.DataFrame(parts, index=df.index)
    meta = df[["unit", "cycle"]]
    return X, meta


def build_training_set(train, window=WINDOW, cap=RUL_CAP):
    """Return ``(X, y)`` with the piecewise-linear (capped) RUL target."""
    X, _ = build_features(train, window)
    y = train.loc[X.index, "RUL"].clip(upper=cap)
    return X, y


def last_cycle_features(test, window=WINDOW):
    """Features at each test unit's final observed cycle, indexed by unit."""
    X, meta = build_features(test, window)
    last = meta.groupby("unit")["cycle"].idxmax()
    return X.loc[last].set_index(meta.loc[last, "unit"])
