# NASA Turbofan RUL

Predict Remaining Useful Life (RUL) of jet engines from the NASA C-MAPSS turbofan
degradation dataset.

## Scope

**Work only on FD001** — the first dataset — unless explicitly asked otherwise.
FD001: 100 train trajectories, 100 test trajectories, one operating condition
(sea level), one fault mode (HPC degradation). Do not generalize code to
FD002–FD004 or add multi-condition handling preemptively.

## Data

`CMAPSSData/` (gitignored, unzipped from `CMAPSSData.zip`; see README for source URL).

- `train_FD001.txt` — 20,631 rows. Each engine runs to failure; RUL is derived as
  `max(cycle) - cycle` per unit.
- `test_FD001.txt` — 13,096 rows. Trajectories are truncated before failure.
- `RUL_FD001.txt` — 100 rows, true RUL at the last cycle of each test unit, in
  unit-id order.

Format: space-delimited, no header, trailing whitespace on each line. 26 columns:

1. `unit` (engine id)
2. `cycle` (time, in operational cycles)
3–5. `op_setting_1..3`
6–26. `sensor_1..21`

Notes: engines start with unknown initial wear (normal, not a fault); sensors are
noisy; several FD001 sensors are constant or near-constant and carry no signal.

## Layout

- `cmapss/` — importable package. **Pure functions, no side effects**: no prints,
  no file writes, no work at import time. Paths resolve from `__file__`, never cwd.
  - `data.py` — loaders (`load_train`, `load_test`, `load_truth`,
    `load_test_with_rul`, `load_dataset`). Each takes `dataset="FD001"` and an
    overridable `data_dir`.
  - `features.py` — retained-sensor list, capped RUL target (`RUL_CAP = 125`),
    within-unit rolling mean/slope features. Window must stay ≤ 31 cycles (the
    shortest test unit).
  - `models.py` — model zoo (`ridge`, `gbdt`) and metrics, including the official
    asymmetric `nasa_score` (late predictions penalized harder than early).
  - `viz.py` — shared matplotlib palette/style; call `viz.use_style()` once.
- `scripts/` — runnable entry points (the drivers). Anything that executes lives here.
  - `train.py` — trains the zoo, group-aware validation (holds out whole engines),
    scores last-cycle test RUL against the truth file.
- `notebooks/` — EDA; imports from `cmapss`.
  - `01_explore.ipynb` — FD001 EDA. Key findings: 8 dead channels dropped; 10
    monotonic sensors retained (2,3,4,7,11,12,15,17,20,21); sensors 9/14 redundant
    pair; sensors flat above ~125 cycles remaining → capped target; test
    observed-length correlates −0.60 with true RUL (a shortcut models must beat).

Installed editable (`pip install -e .`), so `import cmapss` works from any cwd.
Use `.venv/bin/python` (Python 3.13).
