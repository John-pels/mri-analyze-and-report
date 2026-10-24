"""Select the SCNN on validation results, then evaluate it once on the test set.

Outputs (runs/): selection.json, test_results.json, test_predictions.csv,
cv_results.csv, roc.png, scnn_selected.keras
"""
import json
import math

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

import scnn

RNG = np.random.default_rng(scnn.SEED)


def fit_config(cfg, x_tr, y_tr, x_va, y_va):
    scnn.set_seed()
    model = scnn.build_scnn(int(cfg["n_conv"]), int(cfg["n_dense"]), float(cfg["lr"]))
    hist = scnn.train(model, x_tr, y_tr, x_va, y_va, int(cfg["batch_size"]))
    return model, hist


def bootstrap_ci(y, prob, fn, n=2000):
    vals = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(fn(y[i], prob[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    d = np.load("data/prepared.npz", allow_pickle=True)
    x, y, split = d["x"], d["y"], d["split"]
    tr, va, te = split == "train", split == "val", split == "test"

    grid = pd.read_csv("runs/grid_results.csv")
    grid = grid.drop_duplicates("config_id", keep="first")
    assert len(grid) == 108, f"grid incomplete: {len(grid)} rows"
    ranked = grid.sort_values(["val_mcc", "val_auc", "val_sensitivity"], ascending=False).reset_index(drop=True)
    best, second = ranked.iloc[0], ranked.iloc[1]
    json.dump({"best": best.to_dict(), "second": second.to_dict(),
               "criterion": "validation MCC, ties broken by validation AUC then sensitivity"},
              open("runs/selection.json", "w"), indent=2, default=float)

    model, hist = fit_config(best, x[tr], y[tr], x[va], y[va])
    model.save("runs/scnn_selected.keras")
    prob = model.predict(x[te], verbose=0).ravel()
    m = scnn.binary_metrics(y[te], prob)

    model2, _ = fit_config(second, x[tr], y[tr], x[va], y[va])
    prob2 = model2.predict(x[te], verbose=0).ravel()
    m2 = scnn.binary_metrics(y[te], prob2)
    c1, c2 = (prob >= 0.5) == y[te], (prob2 >= 0.5) == y[te]
    b, c = int((c1 & ~c2).sum()), int((~c1 & c2).sum())
    mcnemar_p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0

    auc, auc_lo, auc_hi = scnn.delong_auc_ci(y[te], prob)
    tp, fn, fp, tn = m["tp"], m["fn"], m["fp"], m["tn"]
    res = {
        "selected": {k: best[k] for k in ("lr", "batch_size", "n_conv", "n_dense")},
        "epochs_trained": len(hist.history["loss"]),
        "params": int(model.count_params()),
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "n": m["n"]},
        "accuracy": [m["accuracy"], *scnn.wilson(tp + tn, m["n"])],
        "sensitivity": [m["sensitivity"], *scnn.wilson(tp, tp + fn)],
        "specificity": [m["specificity"], *scnn.wilson(tn, tn + fp)],
        "precision": [m["precision"], *scnn.wilson(tp, tp + fp)],
        "npv": [m["npv"], *scnn.wilson(tn, tn + fn)],
        "f1": [m["f1"], *bootstrap_ci(y[te], prob, lambda a, p: scnn.binary_metrics(a, p)["f1"])],
        "mcc": [m["mcc"], *bootstrap_ci(y[te], prob, lambda a, p: scnn.binary_metrics(a, p)["mcc"])],
        "auc_delong": [auc, auc_lo, auc_hi],
        "pauc_fpr_0_0.1_mcclish": roc_auc_score(y[te], prob, max_fpr=0.1),
        "second_best": {k: second[k] for k in ("lr", "batch_size", "n_conv", "n_dense")},
        "second_best_test": {k: m2[k] for k in ("accuracy", "sensitivity", "specificity", "mcc", "auc")},
        "mcnemar": {"selected_only_correct": b, "second_only_correct": c, "exact_p": mcnemar_p},
    }
    pd.DataFrame({"y": y[te], "prob_selected": prob, "prob_second": prob2}).to_csv("runs/test_predictions.csv", index=False)

    # 5-fold stratified CV of the selected configuration on train + validation data.
    pool = tr | va
    xp, yp = x[pool], y[pool]
    rows = []
    for k, (i_tr, i_te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=scnn.SEED).split(xp, yp)):
        # hold out 15% of the training fold for early stopping
        i_tr = RNG.permutation(i_tr)
        cut = int(0.85 * len(i_tr))
        mk, _ = fit_config(best, xp[i_tr[:cut]], yp[i_tr[:cut]], xp[i_tr[cut:]], yp[i_tr[cut:]])
        pk = mk.predict(xp[i_te], verbose=0).ravel()
        mm = scnn.binary_metrics(yp[i_te], pk)
        rows.append({"fold": k + 1, **{q: mm[q] for q in ("accuracy", "sensitivity", "specificity", "mcc", "auc")}})
        print(rows[-1], flush=True)
    cv = pd.DataFrame(rows)
    cv.to_csv("runs/cv_results.csv", index=False)
    res["cv_mean_sd"] = {q: [float(cv[q].mean()), float(cv[q].std(ddof=1))] for q in
                         ("accuracy", "sensitivity", "specificity", "mcc", "auc")}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fpr, tpr, _ = roc_curve(y[te], prob)
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=300)
    ax.plot(fpr, tpr, color="#1f5fa8", lw=1.6, label=f"SCNN, AUC {auc:.3f} (95% CI {auc_lo:.3f}–{auc_hi:.3f})")
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls="--")
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    fig.tight_layout()
    fig.savefig("runs/roc.png")

    json.dump(res, open("runs/test_results.json", "w"), indent=2, default=lambda o: float(o) if not isinstance(o, str) else o)
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
