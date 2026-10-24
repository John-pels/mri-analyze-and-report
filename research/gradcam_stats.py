"""Grad-CAM box size on test tumour images at several thresholds (app model)."""
import json
import numpy as np
import tensorflow as tf

m = tf.keras.models.load_model("runs/app_mri_4class.keras")
d = np.load("data/prepared.npz", allow_pickle=True)
x, y4, s = d["x"], d["y4"], d["split"]
xt, yt = x[s == "test"], y4[s == "test"]
conv_idx = max(i for i, l in enumerate(m.layers) if isinstance(l, tf.keras.layers.Conv2D))
conv_model = tf.keras.Model(m.inputs, m.layers[conv_idx].output)
head = m.layers[conv_idx + 1:]

def cam(img, k):
    with tf.GradientTape() as tape:
        c = conv_model(img[None], training=False)
        tape.watch(c)
        z = c
        for l in head:
            z = l(z, training=False)
        score = z[0, k]
    g = tape.gradient(score, c)[0]
    h = tf.nn.relu(tf.reduce_sum(c[0] * tf.reduce_mean(g, axis=(0, 1)), -1)).numpy()
    return h / (h.max() + 1e-8)

probs = m.predict(xt, verbose=0)
pred = probs.argmax(1)
idx = [i for i in range(len(yt)) if yt[i] != 0 and pred[i] == yt[i]]
out = {}
cams = [cam(xt[i], pred[i]) for i in idx]
for th in (0.5, 0.6, 0.7, 0.8):
    fr = []
    for h in cams:
        r, c = np.nonzero(h >= th)
        fr.append(((r.max() - r.min() + 1) * (c.max() - c.min() + 1)) / h.size)
    out[th] = {"median_box_fraction": float(np.median(fr)), "iqr": [float(np.percentile(fr, 25)), float(np.percentile(fr, 75))]}
out["cam_shape"] = list(cams[0].shape)
out["n"] = len(idx)
np.save("runs/gradcam_examples.npy", np.stack(cams[:8]))
json.dump({"idx": [int(i) for i in idx[:8]]}, open("runs/gradcam_examples_idx.json", "w"))
print(json.dumps(out, indent=1))
