"""TensorFlow models: AR (linear), feedforward NN, LSTM (paper setup)."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def set_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_ar_linear(p: int, lr: float) -> keras.Model:
    """AR as single linear layer (identity activation), same as paper."""
    inp = layers.Input(shape=(p,))
    out = layers.Dense(1, activation="linear", use_bias=True)(inp)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model


def build_nn(p: int, hidden: int, lr: float) -> keras.Model:
    """One hidden layer, ReLU (paper)."""
    inp = layers.Input(shape=(p,))
    h = layers.Dense(hidden, activation="relu")(inp)
    out = layers.Dense(1)(h)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model


def build_lstm(p: int, hidden: int, lr: float) -> keras.Model:
    """LSTM over p lags; one scalar input per time step (paper Fig. 5–6)."""
    inp = layers.Input(shape=(p, 1))
    x = layers.LSTM(hidden)(inp)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model


def fit_model(
    model: keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int = 32,
    verbose: int = 0,
) -> None:
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)


def predict_one_step(model: keras.Model, x: np.ndarray, kind: str) -> float:
    if kind == "lstm":
        xb = x if x.ndim == 3 else x[np.newaxis, ...]
        pred = model.predict(xb, verbose=0)
    else:
        xb = x if x.ndim == 2 else x[np.newaxis, :]
        pred = model.predict(xb, verbose=0)
    return float(pred.ravel()[0])
