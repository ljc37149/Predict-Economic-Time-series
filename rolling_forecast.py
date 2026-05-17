"""Expanding-window real-time forecasts (paper Section 2–3)."""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass
from typing import Any, Literal

ROLLING_STATE_VERSION = 1

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
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
    n, p, f = X_chrono.shape
    sx = StandardScaler().fit(X_chrono.reshape(-1, f))
    sy = StandardScaler().fit(y.reshape(-1, 1))
    Xs = sx.transform(X_chrono.reshape(-1, f)).reshape(n, p, f)
    ys = sy.transform(y.reshape(-1, 1)).ravel()
    return Xs, ys, sx, sy


def _scale_xy_ff(X_chrono: np.ndarray, y: np.ndarray):
    """Feedforward models: rows [lagged features at t-1 ... t-p]."""
    n = X_chrono.shape[0]
    X_ff = np.flip(X_chrono, axis=1).reshape(n, -1)
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
    exog_future: np.ndarray | None = None,
) -> np.ndarray:
    out = np.zeros(horizon)
    cur = lags_chrono.astype(np.float64).copy()  # shape (p, n_features)
    n_features = cur.shape[1]
    has_exog = n_features > 1
    for h in range(horizon):
        if kind == "lstm":
            x2 = sx.transform(cur.reshape(-1, n_features)).reshape(1, p, n_features)
        else:
            x_ff = np.flip(cur, axis=0).reshape(1, -1)
            x2 = sx.transform(x_ff)
        raw = predict_one_step(model, x2, kind)
        out[h] = inverse_scaled_pred(raw, sy)
        cur = np.roll(cur, -1, axis=0)
        cur[-1, 0] = out[h]
        if has_exog:
            if exog_future is not None and h < len(exog_future):
                cur[-1, 1:] = exog_future[h]
            else:
                cur[-1, 1:] = cur[-2, 1:]
    return out


def _build_xy_with_exog(y_arr: np.ndarray, exog_arr: np.ndarray | None, p: int) -> tuple[np.ndarray, np.ndarray]:
    X_y, Y = y_to_matrix(y_arr, p)
    if exog_arr is None:
        return X_y[:, :, np.newaxis], Y
    if len(exog_arr) != len(y_arr):
        raise ValueError("exog length must match y length")
    rows_ex = []
    for s in range(p, len(y_arr)):
        rows_ex.append(exog_arr[s - p : s, :])
    X_ex = np.asarray(rows_ex, dtype=np.float64)
    X = np.concatenate([X_y[:, :, np.newaxis], X_ex], axis=2)
    return X, Y


def fit_predict_tf(
    y_arr: np.ndarray,
    t: int,
    p: int,
    kind: Kind,
    cfg: RollConfig,
    exog_arr: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train on y[0:t] (exclusive of y[t]); predict y[t]..y[t+H-1] for horizons 1..H.
    Returns (h-step errors array for one origin, optional) — actually returns forecasts for horizons 1..12 and actuals.
    """
    train_end = t
    if train_end < p + 1:
        raise ValueError("insufficient history")
    y_tr = y_arr[:train_end]
    exog_tr = exog_arr[:train_end] if exog_arr is not None else None
    X, Y = _build_xy_with_exog(y_tr, exog_tr, p)
    if len(Y) < 5:
        raise ValueError("too few training rows")

    set_seeds(cfg.seed)
    if kind == "lstm":
        Xs, Ys, sx, sy = _scale_xy_lstm(X, Y)
        model = build_lstm(p, cfg.hidden, cfg.lr, n_features=X.shape[2])
        fit_model(model, Xs, Ys, epochs=cfg.epochs, verbose=cfg.verbose_fit)
    elif kind == "nn":
        Xs, Ys, sx, sy = _scale_xy_ff(X, Y)
        model = build_nn(Xs.shape[1], cfg.hidden, cfg.lr)
        fit_model(model, Xs, Ys, epochs=cfg.epochs, verbose=cfg.verbose_fit)
    else:
        Xs, Ys, sx, sy = _scale_xy_ff(X, Y)
        model = build_ar_linear(Xs.shape[1], cfg.lr)
        fit_model(model, Xs, Ys, epochs=cfg.epochs, verbose=cfg.verbose_fit)

    lags_y = y_arr[t - p : t].reshape(-1, 1)
    if exog_arr is not None:
        lags_ex = exog_arr[t - p : t, :]
        lags = np.concatenate([lags_y, lags_ex], axis=1)
    else:
        lags = lags_y
    horizon = min(12, len(y_arr) - t)
    exog_future = exog_arr[t : t + horizon, :] if exog_arr is not None else None
    fc = _predict_multistep_tf(model, kind, lags, p, horizon, sx, sy, exog_future=exog_future)
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


def _rolling_state_meta(
    y: pd.Series,
    test_start: pd.Timestamp,
    start_pos: int,
    cfg: RollConfig,
    models: list[str],
    max_test_origins: int | None,
    n_origins_fit: int,
) -> dict[str, Any]:
    idx = y.index
    return {
        "version": ROLLING_STATE_VERSION,
        "len": len(y),
        "index_start": str(idx[0]),
        "index_end": str(idx[-1]),
        "test_start": str(pd.Timestamp(test_start)),
        "start_pos": int(start_pos),
        "max_test_origins": max_test_origins,
        "n_origins_fit": int(n_origins_fit),
        "models": sorted(models),
        "max_lag": cfg.max_lag,
        "use_bic": cfg.use_bic,
        "hidden": cfg.hidden,
        "lr": float(cfg.lr),
        "epochs": cfg.epochs,
        "seed": cfg.seed,
        "rw_n": cfg.rw_n,
    }


def _meta_conflict(a: dict[str, Any], b: dict[str, Any]) -> str | None:
    keys = (
        "version",
        "len",
        "index_start",
        "index_end",
        "test_start",
        "start_pos",
        "max_test_origins",
        "n_origins_fit",
        "models",
        "max_lag",
        "use_bic",
        "hidden",
        "lr",
        "epochs",
        "seed",
        "rw_n",
    )
    for k in keys:
        if a.get(k) != b.get(k):
            return f"{k}: checkpoint={a.get(k)!r} current={b.get(k)!r}"
    return None


def _atomic_pickle_dump(path: str, obj: Any) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _save_rolling_state(
    path: str,
    meta: dict[str, Any],
    sq_err: dict[str, np.ndarray],
    counts: dict[str, np.ndarray],
    fitted_completed: int,
) -> None:
    payload = {
        "meta": meta,
        "fitted_completed": int(fitted_completed),
        "sq_err": {m: sq_err[m].copy() for m in sq_err},
        "counts": {m: counts[m].copy() for m in counts},
    }
    _atomic_pickle_dump(path, payload)


def load_rolling_state(path: str) -> dict[str, Any]:
    """Load a rolling-forecast checkpoint written by :func:`msfe_matrix`."""
    with open(path, "rb") as f:
        return pickle.load(f)


# statsmodels MarkovAutoregression.predict raises NotImplementedError for any out-of-sample
# index; use simulation over the filtered joint + transition matrix instead.
MS_MC_SIMS = 96


def _ms_ar_regime_transition_matrix(res) -> np.ndarray:
    """Row i = P(S_{t+1}=j | S_t=i)."""
    P = np.asarray(res.regime_transition, dtype=float).squeeze(-1)
    k = res.model.k_regimes
    if P.shape != (k, k):
        raise ValueError("unexpected regime_transition shape")
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return P / row_sums


def _ms_ar_parse_mean_params(p: np.ndarray, k: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    """const (k,), ar (k, order) matching MarkovAutoregression param order."""
    const = np.asarray(p[2 : 2 + k], dtype=float)
    o = 2 + 2 * k
    cols = []
    for j in range(order):
        cols.append(np.asarray(p[o + j * k : o + (j + 1) * k], dtype=float))
    ar = np.column_stack(cols) if order else np.zeros((k, 0))
    return const, ar


def _ms_ar_path_forecast_mean(
    res,
    y_arr: np.ndarray,
    t: int,
    h: int,
    order: int,
    *,
    rng: np.random.Generator,
    n_sims: int,
) -> np.ndarray:
    k = res.model.k_regimes
    p = np.asarray(res.params, dtype=float)
    const, ar = _ms_ar_parse_mean_params(p, k, order)
    P = _ms_ar_regime_transition_matrix(res)

    joint = np.asarray(res.filtered_joint_probabilities[..., -1], dtype=float).ravel()
    s = float(joint.sum())
    if s <= 0 or not np.isfinite(s):
        return np.full(h, np.nan)
    joint = joint / s

    lags0 = [float(y_arr[t - 1 - j]) for j in range(order)]
    fc_accum = np.zeros(h, dtype=float)

    shape = (k,) * (order + 1)
    idx_choices = np.arange(joint.size)

    for _ in range(n_sims):
        idx = int(rng.choice(idx_choices, p=joint))
        regs = np.unravel_index(idx, shape)
        regs_a = np.asarray(regs, dtype=np.int64)
        lags = list(lags0)
        for step in range(h):
            st = int(regs_a[0])
            mu = const[st]
            for j in range(order):
                past = int(regs_a[1 + j])
                mu += ar[st, j] * (lags[j] - const[past])
            fc_accum[step] += mu
            # next period: new regime, shift history (Hamilton notation)
            st_cur = int(regs_a[0])
            st_new = int(rng.choice(k, p=P[st_cur, :]))
            regs_a[1:] = regs_a[:-1]
            regs_a[0] = st_new
            # y_{l+1} enters as newest lag
            yhat = mu
            for j in range(order - 1, 0, -1):
                lags[j] = lags[j - 1]
            if order > 0:
                lags[0] = yhat

    return fc_accum / float(n_sims)


def forecast_ms_ar(
    y_arr: np.ndarray,
    t: int,
    order: int = 2,
    horizon_max: int = 12,
    *,
    seed: int | None = None,
    mc_sims: int = MS_MC_SIMS,
):
    """
    Two-state Markov AR, switching variance (paper).

    statsmodels does not implement out-of-sample :meth:`predict` for this class
    (it raises ``NotImplementedError``). Forecasts are Monte Carlo means over
    the last filtered joint distribution and the fitted transition matrix, using
    the same conditionally-Gaussian AR recursion as :meth:`predict_conditional`.
    """
    train = y_arr[:t]
    h = min(horizon_max, len(y_arr) - t)
    if h <= 0 or order < 1 or t < order + 1:
        return np.array([], dtype=float), y_arr[t : t + max(h, 0)]
    try:
        mod = MarkovAutoregression(train, k_regimes=2, order=order, switching_variance=True, trend="c")
        res = mod.fit(disp=False, maxiter=200)
        rng = np.random.default_rng(seed)
        pred = _ms_ar_path_forecast_mean(res, y_arr, t, h, order, rng=rng, n_sims=int(mc_sims))
        actual = y_arr[t : t + h]
        return pred, actual
    except Exception:
        return np.full(h, np.nan), y_arr[t : t + h]


def msfe_matrix(
    y: pd.Series,
    exog: pd.DataFrame | None,
    test_start: pd.Timestamp,
    cfg: RollConfig,
    models: list[str],
    max_test_origins: int | None = None,
    *,
    progress: bool = False,
    progress_every: int = 1,
    state_path: str | None = None,
    resume: bool = False,
    save_every: int = 1,
) -> pd.DataFrame:
    """
    Compute mean squared forecast errors by horizon (1..12) for each model name.
    If progress is True, prints one line per completed origin (optionally every
    progress_every origins) so redirected logs show rolling-window advancement.

    Checkpointing: pass ``state_path`` to write a pickle after each completed
    origin (or every ``save_every`` origins). With ``resume=True``, load that
    file if it exists (meta must match the current run) and continue from the
    next rolling origin.
    """
    y_arr = y.values.astype(float)
    exog_arr = exog.values.astype(float) if exog is not None else None
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

    n_origins_fit = sum(1 for pos in origins if pos >= p_default + 1)
    pe = max(1, int(progress_every))
    save_n = max(1, int(save_every))

    meta = _rolling_state_meta(y, test_start, start_pos, cfg, models, max_test_origins, n_origins_fit)
    sq_err = {m: np.zeros(12) for m in models}
    counts = {m: np.zeros(12, dtype=int) for m in models}
    fitted_completed = 0

    if state_path and resume and os.path.isfile(state_path):
        chk = load_rolling_state(state_path)
        bad = _meta_conflict(chk["meta"], meta)
        if bad:
            raise ValueError(f"checkpoint meta mismatch ({bad}); use the same data and CLI as the saved run")
        need_v = chk["meta"].get("version", 0)
        if need_v != ROLLING_STATE_VERSION:
            raise ValueError(f"unsupported checkpoint version {need_v!r} (expected {ROLLING_STATE_VERSION})")
        fitted_completed = int(chk["fitted_completed"])
        if fitted_completed < 0 or fitted_completed > n_origins_fit:
            raise ValueError(f"invalid fitted_completed={fitted_completed} for n_origins_fit={n_origins_fit}")
        for m in models:
            sq_err[m] = np.asarray(chk["sq_err"][m], dtype=float).copy()
            counts[m] = np.asarray(chk["counts"][m], dtype=int).copy()
    elif state_path and resume and not os.path.isfile(state_path):
        print(f"[state] --resume: no file at {state_path!r}, starting from scratch", flush=True)
    elif state_path and not resume and os.path.isfile(state_path):
        print(
            f"[state] existing {state_path!r} will be overwritten on the next checkpoint save "
            f"(omit file or use --resume to continue)",
            flush=True,
        )

    fitted_idx = -1
    for k, t in enumerate(origins):
        if t < p_default + 1:
            continue
        fitted_idx += 1
        if fitted_idx < fitted_completed:
            continue
        t0_origin = time.perf_counter()
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
                    fc, act = forecast_ms_ar(y_arr, t, order=min(2, p_bic), seed=cfg.seed + t)
                elif m == "AR":
                    fc, act = fit_predict_tf(y_arr, t, p_bic, "ar", cfg, exog_arr=exog_arr)
                elif m == "NN":
                    fc, act = fit_predict_tf(y_arr, t, p_bic, "nn", cfg, exog_arr=exog_arr)
                elif m == "LSTM":
                    fc, act = fit_predict_tf(y_arr, t, p_lstm, "lstm", cfg, exog_arr=exog_arr)
                else:
                    continue
            except Exception:
                continue
            H = len(fc)
            for h in range(H):
                e = fc[h] - act[h]
                sq_err[m][h] += e * e
                counts[m][h] += 1

        processed = fitted_idx + 1
        origin_wall = time.perf_counter() - t0_origin
        if state_path and (processed % save_n == 0 or processed == n_origins_fit):
            _save_rolling_state(state_path, meta, sq_err, counts, processed)
        if progress:
            dt_lbl = pd.Timestamp(dates[t]).strftime("%Y-%m-%d")
            tail = processed == n_origins_fit or processed % pe == 0 or processed == 1
            if tail:
                print(
                    f"[progress] origin {processed}/{n_origins_fit} forecast_date={dt_lbl} "
                    f"wall_s={origin_wall:.1f} models={models}",
                    flush=True,
                )

    rows = []
    for m in models:
        msfe = []
        for h in range(12):
            c = counts[m][h]
            msfe.append(sq_err[m][h] / c if c > 0 else np.nan)
        rows.append((m, *msfe))
    cols = ["model"] + [f"h{h+1}" for h in range(12)]
    return pd.DataFrame(rows, columns=cols)
