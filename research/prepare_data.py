"""Prepare the brain-MRI dataset for the SCNN study.

Pools the Training and Testing folders of the Sartaj Bhuvaji brain-tumour MRI
dataset, removes exact and near-duplicate images with a perceptual hash, and
makes a stratified 70/15/15 train/validation/test split with a fixed seed.

Output: data/prepared.npz with grayscale 64x64 images scaled to [0, 1],
binary labels (1 = tumour, 0 = healthy), four-class labels and split names,
plus data/manifest.csv and data/dedup_report.json for the paper.
"""
import argparse
import json
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

CLASSES = ["no_tumor", "glioma_tumor", "meningioma_tumor", "pituitary_tumor"]
SIZE = 64
SEED = 20261024
NEAR_DUP_DISTANCE = 4  # Hamming distance on a 64-bit pHash


def load_records(root: Path) -> pd.DataFrame:
    rows = []
    for split_dir in ("Training", "Testing"):
        for cls in CLASSES:
            for f in sorted((root / split_dir / cls).glob("*")):
                rows.append({"path": str(f), "orig_split": split_dir, "class4": cls})
    return pd.DataFrame(rows)


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    hashes = [imagehash.phash(Image.open(p).convert("L")) for p in df["path"]]
    df = df.assign(phash=[str(h) for h in hashes])
    bits = np.array([h.hash.flatten() for h in hashes], dtype=np.uint8)

    # Union-find over pairs within the near-duplicate distance.
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(df)):
        d = (bits[i + 1:] != bits[i]).sum(axis=1)
        for j in np.nonzero(d <= NEAR_DUP_DISTANCE)[0] + i + 1:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[rj] = ri
    df = df.assign(group=[find(i) for i in range(len(df))])

    keep, conflicting, removed = [], 0, 0
    for _, g in df.groupby("group"):
        if g["class4"].nunique() > 1:
            conflicting += len(g)  # same image, different labels: drop all copies
            continue
        keep.append(g.index[0])
        removed += len(g) - 1
    report = {
        "images_before": int(len(df)),
        "duplicate_copies_removed": int(removed),
        "images_dropped_label_conflict": int(conflicting),
        "images_after": int(len(keep)),
        "groups_with_duplicates": int((df.groupby("group").size() > 1).sum()),
        "near_duplicate_hamming_threshold": NEAR_DUP_DISTANCE,
    }
    return df.loc[keep].reset_index(drop=True), report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Brain-Tumor-Classification-DataSet-master")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    out = Path(args.out)

    df = load_records(Path(args.root))
    df, report = deduplicate(df)
    df["label"] = (df["class4"] != "no_tumor").astype(int)

    idx = np.arange(len(df))
    train_idx, rest = train_test_split(idx, test_size=0.30, stratify=df["class4"], random_state=SEED)
    val_idx, test_idx = train_test_split(rest, test_size=0.50, stratify=df["class4"].iloc[rest], random_state=SEED)
    df["split"] = ""
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    images = np.stack([
        np.asarray(Image.open(p).convert("L").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
        for p in df["path"]
    ])[..., None]

    np.savez_compressed(
        out / "prepared.npz",
        x=images,
        y=df["label"].to_numpy(np.int64),
        y4=df["class4"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy(np.int64),
        split=df["split"].to_numpy(),
    )
    df.drop(columns=["group"]).to_csv(out / "manifest.csv", index=False)
    report["split_counts"] = {
        s: {"total": int((df.split == s).sum()), "tumour": int(((df.split == s) & (df.label == 1)).sum()),
            "healthy": int(((df.split == s) & (df.label == 0)).sum())}
        for s in ("train", "val", "test")
    }
    report["class4_counts"] = df["class4"].value_counts().to_dict()
    (out / "dedup_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
