"""Train the four-class brain-MRI model used by the web application and export it to TensorFlow.js.

Uses the SCNN configuration selected in evaluate.py (runs/selection.json) with a
four-unit softmax output (no tumour, glioma, meningioma, pituitary tumour).
Evaluates once on the held-out test split and writes a TF.js layers model to
runs/tfjs_mri/ (model.json + weights.bin) for public/models/mri/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, roc_auc_score

import scnn

CLASSES = ["No tumour", "Glioma", "Meningioma", "Pituitary tumour"]


def inference_model(trained: tf.keras.Model, input_size: int, n_classes: int) -> tf.keras.Model:
    """Same network without the augmentation layers, weights copied across."""
    layers = [tf.keras.Input(shape=(input_size, input_size, 1))]
    for layer in trained.layers:
        if layer.__class__.__name__.startswith("Random"):
            continue
        layers.append(layer.__class__.from_config(layer.get_config()))
    m = tf.keras.Sequential(layers)
    src = [l for l in trained.layers if not l.__class__.__name__.startswith("Random")]
    for a, b in zip(m.layers, src):
        a.set_weights(b.get_weights())
    return m


def export_tfjs(model: tf.keras.Model, out_dir: Path, input_size: int) -> None:
    """Write a TF.js 'layers-model' (Keras 2 style topology) for this simple layer set."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counters, layer_cfgs, specs, blobs = {}, [], [], []
    init = {"kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": None}},
            "bias_initializer": {"class_name": "Zeros", "config": {}},
            "kernel_regularizer": None, "bias_regularizer": None, "activity_regularizer": None,
            "kernel_constraint": None, "bias_constraint": None}
    for i, layer in enumerate(model.layers):
        kind = layer.__class__.__name__
        base = {"Conv2D": "conv2d", "MaxPooling2D": "max_pooling2d", "Flatten": "flatten",
                "Dense": "dense", "Dropout": "dropout"}[kind]
        counters[base] = counters.get(base, 0) + 1
        name = f"{base}_{counters[base]}"
        c = layer.get_config()
        cfg = {"name": name, "trainable": True, "dtype": "float32"}
        if i == 0:
            cfg["batch_input_shape"] = [None, input_size, input_size, 1]
        if kind == "Conv2D":
            cfg.update({"filters": c["filters"], "kernel_size": list(c["kernel_size"]), "strides": list(c["strides"]),
                        "padding": c["padding"], "data_format": "channels_last", "dilation_rate": [1, 1],
                        "activation": c["activation"], "use_bias": True, **init})
        elif kind == "MaxPooling2D":
            cfg.update({"pool_size": list(c["pool_size"]), "strides": list(c["strides"]), "padding": c["padding"],
                        "data_format": "channels_last"})
        elif kind == "Flatten":
            cfg["data_format"] = "channels_last"
        elif kind == "Dense":
            cfg.update({"units": c["units"], "activation": c["activation"], "use_bias": True, **init})
        elif kind == "Dropout":
            cfg.update({"rate": c["rate"], "noise_shape": None, "seed": None})
        layer_cfgs.append({"class_name": kind, "config": cfg})
        for wname, w in zip(("kernel", "bias"), layer.get_weights()):
            w = w.astype(np.float32)
            specs.append({"name": f"{name}/{wname}", "shape": list(w.shape), "dtype": "float32"})
            blobs.append(w.tobytes())
    (out_dir / "weights.bin").write_bytes(b"".join(blobs))
    model_json = {
        "format": "layers-model",
        "generatedBy": "keras " + tf.keras.__version__,
        "convertedBy": "research/train_app_model.py",
        "modelTopology": {"class_name": "Sequential", "config": {"name": "scnn_mri_4class", "layers": layer_cfgs},
                          "keras_version": "2.15.0", "backend": "tensorflow"},
        "weightsManifest": [{"paths": ["weights.bin"], "weights": specs}],
    }
    (out_dir / "model.json").write_text(json.dumps(model_json))


def main():
    sel = json.load(open("runs/selection.json"))["best"]
    d = np.load("data/prepared.npz", allow_pickle=True)
    x, y4, split = d["x"], d["y4"], d["split"]
    tr, va, te = split == "train", split == "val", split == "test"
    size = x.shape[1]

    scnn.set_seed()
    model = scnn.build_scnn(int(sel["n_conv"]), int(sel["n_dense"]), float(sel["lr"]), n_classes=4, input_size=size)
    hist = scnn.train(model, x[tr], y4[tr], x[va], y4[va], int(sel["batch_size"]))
    inf = inference_model(model, size, 4)
    prob = inf.predict(x[te], verbose=0)
    pred = prob.argmax(1)
    yt = y4[te]

    cm = confusion_matrix(yt, pred, labels=range(4))
    per_class = []
    for k, name in enumerate(CLASSES):
        tp = int(cm[k, k]); fn = int(cm[k].sum() - tp); fp = int(cm[:, k].sum() - tp); tn = int(cm.sum() - tp - fn - fp)
        per_class.append({"class": name, "n": tp + fn, "sensitivity": tp / (tp + fn),
                          "sens_ci": scnn.wilson(tp, tp + fn), "specificity": tn / (tn + fp),
                          "spec_ci": scnn.wilson(tn, tn + fp), "auc_ovr": roc_auc_score(yt == k, prob[:, k])})
    correct = int((pred == yt).sum())
    yb, pb = (yt != 0).astype(int), 1 - prob[:, 0]
    res = {
        "config": sel, "epochs_trained": len(hist.history["loss"]), "params": int(inf.count_params()),
        "n_test": int(len(yt)), "accuracy": [correct / len(yt), *scnn.wilson(correct, len(yt))],
        "macro_f1": f1_score(yt, pred, average="macro"), "cohen_kappa": cohen_kappa_score(yt, pred),
        "confusion_matrix": cm.tolist(), "per_class": per_class,
        "binary_collapse": scnn.binary_metrics(yb, pb),
    }
    out = Path("runs")
    inf.save(out / "app_mri_4class.keras")
    np.save(out / "app_test_prob.npy", prob)
    export_tfjs(inf, out / "tfjs_mri", size)
    json.dump(res, open(out / "app_model_results.json", "w"), indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
