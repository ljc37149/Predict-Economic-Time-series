"""Expanding-window real-time forecasts (paper Section 2–3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from statsmodels.tsa.statespace.sarimax import SARIMAX

from lags import inverse_scaled_pred, random_walk_forecast, select_lag_bic, y_to_matrix
from models_tf import build_ar_linear, build_lstm, build_nn, fit_model, predict_one_step, set_seeds


Kind = Literal["ar", "nn", "lstm"]


@dataclass
class RollConfig:
    max_lag: int = 24
    use_bic: bool = False
    """If True, AR/NN use BIC lag choice; LSTM always uses max_lag (paper Table 5)."""
    hidden: int = 50
    lr: float = 0.001
    epochs: int = 500
    rw_n: int = 1
    seed: int = 42
    verbose_fit: int = 0


def _scale_xy_lstm(X_chrono: np.ndarray, y: np.ndarray):
    n, p = X_chrono.shape
    sx = StandardScaler().fit(X_chrono)
    sy = StandardScaler().fit(y.reshape(-1, 1))
    Xs = sx.transform(X_chrono).reshape(n, p, 1)
    ys = sy.transform(y.reshape(-1, 1)).ravel()
    return Xs, ys, sx, sy


def _scale_xy_ff(X_chrono: np.ndarray, y: np.ndarray):
    """Feedforward models: rows [y_{t-1},...,y_{t-p}]."""
    X_ff = np.flip(X_chrono, axis=1)
    sx = StandardScaler().fit(X_ff)
    sy = StandardScaler().fit(y.reshape(-1, 1))
    Xs = sx.transform(X_ff)
    ys = sy.transform(y.reshape(-1, 1)).ravel()
    return Xs, ys, sx, sy


def _predict_multistep_tf(
    model,
    kind: Kind,
    lags_chrono: np.ndarray,
    p: int,
    horizon: int,
    sx: StandardScaler,
    sy: StandardScaler,
) -> np.ndarray:
    out = np.zeros(horizon)
    cur = lags_chrono.astype(np.float64).copy()
    for h in range(horizon):
        if kind == "lstm":
            x2 = sx.transform(cur.reshape(1, -1)).reshape(1, p, 1)
        else:
            x_ff = np.flip(cur).reshape(1, -1)
            x2 = sx.transform(x_ff)
        raw = predict_one_step(model, x2, kind)
        out[h] = inverse_scaled_pred(raw, sy)
        cur = np.roll(cur, -1)
        cur[-1] = out[h]
    return out


def fit_predict_tf(
    y_arr: np.ndarray,
    t: int,
    p: int,
    kind: Kind,
    cfg: RollConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train on y[0:t] (exclusive of y[t]); predict y[t]..y[t+H-1] for horizons 1..H.
    Returns (h-step errors array for one origin, optional) — actually returns forecasts for horizons 1..12 and actuals.
    """
    train_end = t
    if train_end < p + 1:
        raise ValueError("insufficient history")
    y_tr = y_arr[:train_end]
    X, Y = y_to_matrix(y_tr, p)
    if len(Y) < 5:
        raise ValueError("too few training rows")

    set_seeds(cfg.seed)
    if kind == "lstm":
        X_chrono = X
        Xs, Ys, sx, sy = _scale_xy_lstm(X_chrono, Y)
        model = build_lstm(p, cfg.hidden, cfg.lr)
        fit_model(model, Xs, Ys, epochs=cfg.epochs, verbose=cfg.verbose_fit)
    elif kind == "nn":
        Xs, Ys, sx, sy = _scale_xy_ff(X, Y)
        model = build_nn(p, cfg.hidden, cfg.lr)
        fit_model(model, Xs, Ys, epochs=cfg.epochs, verbose=cfg.verbose_fit)
    else:
        Xs, Ys, sx, sy = _scale_xy_ff(X, Y)
        model = build_ar_linear(p, cfg.lr)
        fit_model(model, Xs, Ys, epochs=cfg.epochs, verbose=cfg.verbose_fit)

    lags = y_arr[t - p : t]
    horizon = min(12, len(y_arr) - t)
    fc = _predict_multistep_tf(model, kind, lags, p, horizon, sx, sy)
    actual = y_arr[t : t + horizon]
    return fc, actual


def forecast_rw(y_arr: np.ndarray, t: int, p: int, cfg: RollConfig) -> tuple[np.ndarray, np.ndarray]:
    lags = y_arr[t - p : t]
    horizon = min(12, len(y_arr) - t)
    # Eq. (4): n=1 ⇒ one-step forecast is y_{t-1}; multi-step uses same flat forecast (paper RW).
    rw1 = random_walk_forecast(np.flip(lags), cfg.rw_n)
    out = np.full(horizon, rw1, dtype=float)
    actual = y_arr[t : t + horizon]
    return out, actual


def forecast_sarima(y_arr: np.ndarray, t: int, horizon_max: int = 12):
    """SARIMA(1,1,1)(0,0,1,12) on inflation (paper)."""
    train = y_arr[:t]
    h = min(horizon_max, len(y_arr) - t)
    try:
        mod = SARIMAX(
            train,
            order=(1, 1, 1),
            seasonal_order=(0, 0, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = mod.fit(disp=False, maxiter=200)
        fc = res.forecast(steps=h)
        actual = y_arr[t : t + h]
        return np.asarray(fc, dtype=float), actual
    except Exception:
        return np.full(h, np.nan), y_arr[t : t + h]


def forecast_ms_ar(y_arr: np.ndarray, t: int, order: int = 2, horizon_max: int = 12):
    """Two-state Markov AR, switching variance (paper)."""
    train = y_arr[:t]
    h = min(horizon_max, len(y_arr) - t)
    try:
        mod = MarkovAutoregression(train, k_regimes=2, order=order, switching_variance=True, trend="c")
        res = mod.fit(disp=False, maxiter=200)
        pred = res.predict(start=t, end=t + h - 1)
        actual = y_arr[t : t + h]
        return np.asarray(pred, dtype=float), actual
    except Exception:
        return np.full(h, np.nan), y_arr[t : t + h]


def msfe_matrix(
    y: pd.Series,
    test_start: pd.Timestamp,
    cfg: RollConfig,
    models: list[str],
    max_test_origins: int | None = None,
) -> pd.DataFrame:
    """
    Compute mean squared forecast errors by horizon (1..12) for each model name.
    """
    y_arr = y.values.astype(float)
    dates = y.index
    ts = pd.Timestamp(test_start)
    if ts in dates:
        start_pos = int(dates.get_loc(ts))
    else:
        pos = dates.get_indexer([ts], method="bfill")
        start_pos = int(pos[0]) if pos[0] >= 0 else -1
    if start_pos < 0:
        raise ValueError("test_start not in index")

    p_default = cfg.max_lag
    origins = list(range(start_pos, len(y_arr)))
    if max_test_origins is not None:
        origins = origins[:max_test_origins]

    sq_err = {m: np.zeros(12) for m in models}
    counts = {m: np.zeros(12, dtype=int) for m in models}

    for k, t in enumerate(origins):
        if t < p_default + 1:
            continue
        p_bic = select_lag_bic(y_arr[:t], cfg.max_lag) if cfg.use_bic else cfg.max_lag
        p_bic = min(max(p_bic, 1), cfg.max_lag)
        p_lstm = cfg.max_lag

        for m in models:
            try:
                if m == "RW":
                    fc, act = forecast_rw(y_arr, t, max(p_bic, 1), cfg)
                elif m == "SARIMA":
                    fc, act = forecast_sarima(y_arr, t)
                elif m == "MS":
                    fc, act = forecast_ms_ar(y_arr, t, order=min(2, p_bic))
                elif m == "AR":
                    fc, act = fit_predict_tf(y_arr, t, p_bic, "ar", cfg)
                elif m == "NN":
                    fc, act = fit_predict_tf(y_arr, t, p_bic, "nn", cfg)
                elif m == "LSTM":
                    fc, act = fit_predict_tf(y_arr, t, p_lstm, "lstm", cfg)
                else:
                    continue
            except Exception:
                continue
            H = len(fc)
            for h in range(H):
                e = fc[h] - act[h]
                sq_err[m][h] += e * e
                counts[m][h] += 1

    rows = []
    for m in models:
        msfe = []
        for h in range(12):
            c = counts[m][h]
            msfe.append(sq_err[m][h] / c if c > 0 else np.nan)
        rows.append((m, *msfe))
    cols = ["model"] + [f"h{h+1}" for h in range(12)]
    return pd.DataFrame(rows, columns=cols)
