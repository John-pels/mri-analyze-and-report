"""SCNN model definition, training and metrics shared by the study scripts."""
import math
import os
import random

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score  # noqa: E402

SEED = 20261024
FILTERS = [32, 64, 128, 256]
DENSE_UNITS = 128
MAX_EPOCHS = 25
PATIENCE = 4


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def build_scnn(n_conv: int, n_dense: int, lr: float, n_classes: int = 1, input_size: int = 64) -> tf.keras.Model:
    """Conv(3x3, ReLU) + 2x2 max-pool blocks, hidden dense layers, sigmoid/softmax output."""
    layers = [
        tf.keras.Input(shape=(input_size, input_size, 1)),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(10 / 360),
        tf.keras.layers.RandomZoom(0.10),
    ]
    for i in range(n_conv):
        layers += [
            tf.keras.layers.Conv2D(FILTERS[i], 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(2),
        ]
    layers.append(tf.keras.layers.Flatten())
    for _ in range(n_dense):
        layers += [tf.keras.layers.Dense(DENSE_UNITS, activation="relu"), tf.keras.layers.Dropout(0.5)]
    if n_classes == 1:
        layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"
    else:
        layers.append(tf.keras.layers.Dense(n_classes, activation="softmax"))
        loss = "sparse_categorical_crossentropy"
    model = tf.keras.Sequential(layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss)
    return model


def class_weights(y: np.ndarray) -> dict:
    classes, counts = np.unique(y, return_counts=True)
    return {int(c): len(y) / (len(classes) * n) for c, n in zip(classes, counts)}


def train(model, x_tr, y_tr, x_va, y_va, batch_size: int, verbose: int = 0):
    cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    hist = model.fit(
        x_tr, y_tr, validation_data=(x_va, y_va), epochs=MAX_EPOCHS, batch_size=batch_size,
        class_weight=class_weights(y_tr), callbacks=[cb], verbose=verbose, shuffle=True,
    )
    return hist


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def binary_metrics(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn), "n": int(len(y)),
        "accuracy": (tp + tn) / len(y),
        "sensitivity": tp / (tp + fn) if tp + fn else math.nan,
        "specificity": tn / (tn + fp) if tn + fp else math.nan,
        "precision": tp / (tp + fp) if tp + fp else math.nan,
        "npv": tn / (tn + fn) if tn + fn else math.nan,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        "mcc": matthews_corrcoef(y, pred),
    }
    out["auc"] = roc_auc_score(y, prob) if len(np.unique(y)) == 2 else math.nan
    return out


# ---- DeLong AUC variance (Sun & Xu 2014 fast implementation) ----
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def delong_auc_ci(y: np.ndarray, prob: np.ndarray, alpha: float = 0.95) -> tuple[float, float, float]:
    pos, neg = prob[y == 1], prob[y == 0]
    m, n = len(pos), len(neg)
    tx, ty, tz = _midrank(pos), _midrank(neg), _midrank(np.concatenate([pos, neg]))
    auc = tz[:m].sum() / m / n - (m + 1) / 2 / n
    v01 = (tz[:m] - tx) / n
    v10 = 1 - (tz[m:] - ty) / m
    var = np.var(v01, ddof=1) / m + np.var(v10, ddof=1) / n
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - alpha) / 2)
    se = math.sqrt(var)
    return auc, max(0.0, auc - z * se), min(1.0, auc + z * se)
