"""Full-factorial grid search for the SCNN, scored on the validation set only.

4 learning rates x 3 batch sizes x 3 conv depths x 3 dense depths = 108 configs.
Run in chunks so each process stays short:  python grid_search.py --chunk 0 --of 4
Results are appended to runs/grid_results.csv (one row per config).
"""
import argparse
import csv
import itertools
import time
from pathlib import Path

import numpy as np

import scnn

GRID = {
    "lr": [0.1, 0.01, 0.001, 0.0001],
    "batch_size": [10, 20, 30],
    "n_conv": [2, 3, 4],
    "n_dense": [1, 2, 3],
}
CONFIGS = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
FIELDS = ["config_id", "lr", "batch_size", "n_conv", "n_dense", "epochs", "seconds",
          "val_accuracy", "val_sensitivity", "val_specificity", "val_precision", "val_f1", "val_mcc", "val_auc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--out", default="runs/grid_results.csv")
    ap.add_argument("--reverse", action="store_true", help="work from the last config backwards")
    args = ap.parse_args()

    d = np.load("data/prepared.npz", allow_pickle=True)
    x, y, split = d["x"], d["y"], d["split"]
    x_tr, y_tr = x[split == "train"], y[split == "train"]
    x_va, y_va = x[split == "val"], y[split == "val"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with out.open() as f:
            done = {int(r["config_id"]) for r in csv.DictReader(f)}
    new_file = not out.exists()

    order = list(enumerate(CONFIGS))
    if args.reverse:
        order.reverse()
    for cid, cfg in order:
        if cid % args.of != args.chunk or cid in done:
            continue
        # re-check: another worker may have finished this config meanwhile
        with out.open() as f:
            if cid in {int(r["config_id"]) for r in csv.DictReader(f)}:
                continue
        scnn.set_seed()
        t0 = time.time()
        model = scnn.build_scnn(cfg["n_conv"], cfg["n_dense"], cfg["lr"])
        hist = scnn.train(model, x_tr, y_tr, x_va, y_va, cfg["batch_size"])
        prob = model.predict(x_va, verbose=0).ravel()
        prob = np.nan_to_num(prob, nan=0.5)
        m = scnn.binary_metrics(y_va, prob)
        row = {"config_id": cid, **cfg, "epochs": len(hist.history["loss"]), "seconds": round(time.time() - t0, 1),
               **{f"val_{k}": round(float(m[k]), 5) for k in
                  ("accuracy", "sensitivity", "specificity", "precision", "f1", "mcc", "auc")}}
        with out.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                w.writeheader()
                new_file = False
            w.writerow(row)
        print(row, flush=True)


if __name__ == "__main__":
    main()
