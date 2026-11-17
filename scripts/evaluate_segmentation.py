"""
Quantitative evaluation of leaf segmentation (IoU and Dice) against the
background-removed "segmented" version of PlantVillage, plus a lesion-ratio
analysis (healthy vs. diseased leaves).

Outputs
  results/tables/segmentation_iou.csv        per image / per method scores
  results/tables/segmentation_summary.json   mean / median / std per method
  results/tables/lesion_ratio.csv            lesion area ratio for every test image
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import COLOR_DIR, SEGMENTED_DIR, TABLE_DIR  # noqa: E402
from data import get_split  # noqa: E402
from preprocessing import load_rgb, preprocess  # noqa: E402
from segmentation import LEAF_METHODS, iou_dice, leaf_mask, lesion_mask, reference_leaf_mask  # noqa: E402


def pairs():
    for seg in sorted(glob.glob(os.path.join(glob.escape(SEGMENTED_DIR), "*", "*"))):
        cls = os.path.basename(os.path.dirname(seg))
        stem = os.path.basename(seg).replace("_final_masked.jpg", "")
        hits = glob.glob(os.path.join(glob.escape(COLOR_DIR), glob.escape(cls), glob.escape(stem) + ".*"))
        if hits:
            yield cls, hits[0], seg


def main():
    rows = []
    for cls, colour, seg in pairs():
        ref = reference_leaf_mask(load_rgb(seg))
        raw = load_rgb(colour)
        pre = preprocess(raw)
        for inp_name, img in (("raw", raw), ("preprocessed", pre)):
            for name, fn in LEAF_METHODS.items():
                t = time.time()
                m = fn(img)
                ms = 1000 * (time.time() - t)
                iou, dice = iou_dice(m, ref)
                rows.append({"class": cls, "image": os.path.basename(colour), "input": inp_name,
                             "method": name, "iou": iou, "dice": dice, "ms": ms})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLE_DIR, "segmentation_iou.csv"), index=False)
    summary = (df.groupby(["input", "method"])
                 .agg(iou_mean=("iou", "mean"), iou_median=("iou", "median"), iou_std=("iou", "std"),
                      dice_mean=("dice", "mean"), dice_median=("dice", "median"), ms_mean=("ms", "mean"),
                      n=("iou", "size"))
                 .reset_index())
    summary.to_csv(os.path.join(TABLE_DIR, "segmentation_summary.csv"), index=False)
    print(summary.to_string(index=False))

    # lesion area ratio on the test split (healthy vs. diseased)
    split, _ = get_split(save=False)
    te = split[split.split == "test"]
    lr = []
    for p, c in zip(te.path, te["class"]):
        x = preprocess(load_rgb(p))
        leaf = leaf_mask(x)
        les = lesion_mask(x, leaf)
        lr.append({"class": c, "healthy": "healthy" in c.lower(), "lesion_ratio": les.sum() / max(leaf.sum(), 1)})
    lr = pd.DataFrame(lr)
    lr.to_csv(os.path.join(TABLE_DIR, "lesion_ratio.csv"), index=False)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(~lr.healthy, lr.lesion_ratio)
    stats = {"healthy_median": float(lr[lr.healthy].lesion_ratio.median()),
             "diseased_median": float(lr[~lr.healthy].lesion_ratio.median()),
             "auc_lesion_ratio_healthy_vs_diseased": float(auc), "n": int(len(lr))}
    with open(os.path.join(TABLE_DIR, "lesion_ratio_summary.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(stats)


if __name__ == "__main__":
    main()
