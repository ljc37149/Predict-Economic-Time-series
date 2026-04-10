"""Lag construction and iterated multi-step forecasts (paper Section 2)."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


def y_to_matrix(y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Supervised samples: target y[s], rows of X are [y[s-p],...,y[s-1]] (time-ordered old→new).
    Valid s from p to len(y)-1.
    """
    n = len(y)
    rows = []
    targets = []
    for s in range(p, n):
        rows.append(y[s - p : s])
        targets.append(y[s])
    return np.asarray(rows, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def select_lag_bic(y_train: np.ndarray, maxlag: int) -> int:
    """BIC lag choice via AutoReg (paper uses AR + information criteria)."""
    from statsmodels.tsa.ar_model import AutoReg

    if len(y_train) <= maxlag + 5:
        return min(maxlag, max(1, len(y_train) // 10))
    best_p, best_bic = 1, float("inf")
    for p in range(1, maxlag + 1):
        try:
            fit = AutoReg(y_train, lags=p, trend="c").fit()
            if fit.bic < best_bic:
                best_bic, best_p = fit.bic, p
        except Exception:
            continue
    return best_p


def random_walk_forecast(lags_recent_to_old: np.ndarray, n_avg: int = 1) -> float:
    """Eq. (4): mean of last n_avg observations (lags_recent_to_old[0] is y_{t-1})."""
    take = min(n_avg, len(lags_recent_to_old))
    return float(np.mean(lags_recent_to_old[:take]))


def inverse_scaled_pred(pred: float, sy: StandardScaler) -> float:
    return float(sy.inverse_transform(np.array([[pred]]))[0, 0])
