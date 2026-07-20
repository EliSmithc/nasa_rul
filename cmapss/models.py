"""Baseline RUL models and the C-MAPSS evaluation metrics."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_models(random_state=0):
    """The model zoo: name -> unfitted estimator."""
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "gbdt": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=random_state,
        ),
    }


def nasa_score(y_true, y_pred):
    """Official C-MAPSS asymmetric score (lower is better).

    Late predictions (pred > true) are penalized harder than early ones:
    exp(d/10)-1 for d >= 0, exp(-d/13)-1 for d < 0, summed over units.
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13), np.exp(d / 10)) - 1))


def evaluate(y_true, y_pred):
    """RMSE / MAE / bias / NASA score for last-cycle predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    return {
        "rmse": float(np.sqrt((err**2).mean())),
        "mae": float(np.abs(err).mean()),
        "bias": float(err.mean()),
        "nasa_score": nasa_score(y_true, y_pred),
    }
