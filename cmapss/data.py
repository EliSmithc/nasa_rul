"""Load the C-MAPSS train / test / truth files into pandas DataFrames."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "CMAPSSData"

INDEX_COLS = ["unit", "cycle"]
OP_SETTING_COLS = [f"op_setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
COLUMNS = INDEX_COLS + OP_SETTING_COLS + SENSOR_COLS


def _read_trajectories(path):
    """Read a space-delimited C-MAPSS trajectory file (no header, trailing spaces)."""
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
        dtype={"unit": "int64", "cycle": "int64"},
    )


def load_train(dataset="FD001", data_dir=DATA_DIR):
    """Training trajectories, each run to failure, with a derived ``RUL`` column."""
    df = _read_trajectories(Path(data_dir) / f"train_{dataset}.txt")
    failure_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = failure_cycle - df["cycle"]
    return df


def load_test(dataset="FD001", data_dir=DATA_DIR):
    """Test trajectories, truncated some time before failure. No ``RUL`` column."""
    return _read_trajectories(Path(data_dir) / f"test_{dataset}.txt")


def load_truth(dataset="FD001", data_dir=DATA_DIR):
    """True RUL at each test unit's final cycle, with a 1-based ``unit`` column."""
    truth = pd.read_csv(
        Path(data_dir) / f"RUL_{dataset}.txt", header=None, names=["RUL"]
    )
    truth.insert(0, "unit", range(1, len(truth) + 1))
    return truth


def load_test_with_rul(dataset="FD001", data_dir=DATA_DIR):
    """Test trajectories with a per-row ``RUL`` column.

    The truth file gives RUL only at each unit's last observed cycle; earlier
    cycles are that value plus the cycles remaining in the record.
    """
    df = load_test(dataset, data_dir)
    truth = load_truth(dataset, data_dir).set_index("unit")["RUL"]
    last_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = (last_cycle - df["cycle"]) + df["unit"].map(truth)
    return df


def load_dataset(dataset="FD001", data_dir=DATA_DIR):
    """Return ``(train, test, truth)`` for one dataset."""
    return (
        load_train(dataset, data_dir),
        load_test(dataset, data_dir),
        load_truth(dataset, data_dir),
    )
