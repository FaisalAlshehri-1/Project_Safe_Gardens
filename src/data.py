"""Dataset indexing and the stratified 70 / 15 / 15 train / validation / test split."""
import glob
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from config import COLOR_DIR, SEED, SPLIT, TABLE_DIR

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def build_index():
    rows = []
    classes = sorted(d for d in os.listdir(COLOR_DIR) if os.path.isdir(os.path.join(COLOR_DIR, d)))
    for label, cls in enumerate(classes):
        for p in sorted(glob.glob(os.path.join(glob.escape(COLOR_DIR), glob.escape(cls), "*"))):
            if p.lower().endswith(IMG_EXT):
                rows.append({"path": p, "class": cls, "label": label})
    return pd.DataFrame(rows), classes


def get_split(save=True):
    """Return (df with a 'split' column, class names). The split is deterministic (seed 42)."""
    df, classes = build_index()
    train_frac, val_frac, test_frac = SPLIT
    tr, rest = train_test_split(df, test_size=val_frac + test_frac, stratify=df["label"], random_state=SEED)
    va, te = train_test_split(rest, test_size=test_frac / (val_frac + test_frac),
                              stratify=rest["label"], random_state=SEED)
    df.loc[tr.index, "split"] = "train"
    df.loc[va.index, "split"] = "val"
    df.loc[te.index, "split"] = "test"
    if save:
        out = df.copy()
        out["path"] = out["path"].map(lambda p: os.path.relpath(p, COLOR_DIR))
        out.to_csv(os.path.join(TABLE_DIR, "data_split.csv"), index=False)
    return df, classes
