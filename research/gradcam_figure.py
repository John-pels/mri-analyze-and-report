"""Figure: Grad-CAM heatmaps and the 0.7-threshold box for four correctly classified test images."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

d = np.load("data/prepared.npz", allow_pickle=True)
xt, yt = d["x"][d["split"] == "test"], d["y4"][d["split"] == "test"]
import tensorflow as tf
m = tf.keras.models.load_model("runs/app_mri_4class.keras")
ci = max(i for i, l in enumerate(m.layers) if isinstance(l, tf.keras.layers.Conv2D))
conv = tf.keras.Model(m.inputs, m.layers[ci].output)
head = m.layers[ci + 1:]

def cam(img, k):
    with tf.GradientTape() as t:
        c = conv(img[None], training=False); t.watch(c); z = c
        for l in head: z = l(z, training=False)
        s = z[0, k]
    g = t.gradient(s, c)[0]
    h = tf.nn.relu(tf.reduce_sum(c[0] * tf.reduce_mean(g, axis=(0, 1)), -1)).numpy()
    return h / (h.max() + 1e-8)

pred = m.predict(xt, verbose=0).argmax(1)
names = ["No tumour", "Glioma", "Meningioma", "Pituitary tumour"]
rng = np.random.default_rng(20261024)
idx = []
for k in (1, 2, 3):   # a random correctly classified test image per tumour class
    pool = [i for i in range(len(yt)) if yt[i] == k and pred[i] == k]
    idx.append(int(rng.choice(pool)))
idx.append(int(rng.choice([i for i in range(len(yt)) if yt[i] == 1 and pred[i] == 1 and i not in idx])))
cams = np.stack([cam(xt[i], pred[i]) for i in idx])
pick = range(4)

fig, axes = plt.subplots(1, 4, figsize=(8, 2.3), dpi=300)
for ax, k in zip(axes, pick):
    img = xt[idx[k], ..., 0]
    cam = np.asarray(Image.fromarray((cams[k] * 255).astype(np.uint8)).resize((64, 64), Image.BILINEAR)) / 255
    ax.imshow(img, cmap="gray")
    ax.imshow(cam, cmap="jet", alpha=0.35)
    r, c = np.nonzero(cams[k] >= 0.7)
    s = 64 / cams[k].shape[0]
    ax.add_patch(patches.Rectangle((c.min() * s - 0.5, r.min() * s - 0.5), (c.max() - c.min() + 1) * s, (r.max() - r.min() + 1) * s,
                                   fill=False, ec="#00ff66", lw=1.2))
    ax.set_title(names[yt[idx[k]]], fontsize=8)
    ax.axis("off")
fig.tight_layout()
fig.savefig("runs/gradcam_examples.png", bbox_inches="tight")
print("ok", [names[yt[idx[k]]] for k in pick])
