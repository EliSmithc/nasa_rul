"""Train baseline RUL models on FD001 and score them on the test set.

Usage: .venv/bin/python scripts/train.py
"""

import numpy as np
import pandas as pd

import cmapss
from cmapss import features, models


def main():
    train, test, truth = cmapss.load_dataset("FD001")
    y_true = truth.set_index("unit")["RUL"]

    print("building features...")
    X_train, y_train = features.build_training_set(train)
    X_test = features.last_cycle_features(test).loc[y_true.index]

    # Group-aware validation split: hold out whole engines, never rows.
    rng = np.random.default_rng(0)
    units = train["unit"].unique()
    val_units = set(rng.choice(units, size=20, replace=False))
    val_mask = train.loc[X_train.index, "unit"].isin(val_units)

    # Naive reference: predict the cap-aware mean of the training target.
    baseline_pred = np.full(len(y_true), float(y_train.mean()))
    rows = {"constant-mean": models.evaluate(y_true, baseline_pred)}

    for name, model in models.make_models().items():
        model.fit(X_train[~val_mask], y_train[~val_mask])
        val_pred = model.predict(X_train[val_mask])
        val_rmse = float(np.sqrt(((val_pred - y_train[val_mask]) ** 2).mean()))

        # Refit on all training units before scoring the real test set.
        model.fit(X_train, y_train)
        pred = np.clip(model.predict(X_test), 0, None)
        rows[name] = models.evaluate(y_true, pred) | {"val_rmse": val_rmse}

    report = pd.DataFrame(rows).T
    print()
    print(report.round(2).to_string())
    print("\n(nasa_score: official asymmetric metric, lower is better;")
    print(" val_rmse is against the capped target on 20 held-out engines)")


if __name__ == "__main__":
    main()
