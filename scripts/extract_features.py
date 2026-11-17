"""
Extract hand-crafted features for every image and cache them to data/cache/.

Configurations
  full      preprocessing ON,  leaf mask ON   (proposed pipeline)
  no_pre    preprocessing OFF, leaf mask ON   (ablation)
  no_mask   preprocessing ON,  leaf mask OFF  (ablation)
  robust_*  test images corrupted by a degradation, with and without preprocessing

Usage:  python scripts/extract_features.py [--workers 2]
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import CACHE_DIR  # noqa: E402
from data import get_split  # noqa: E402
from features import FEATURE_GROUPS, extract_all  # noqa: E402
from preprocessing import DEGRADATIONS, load_rgb  # noqa: E402


def _job(args):
    path, pre, mask, degr = args
    img = load_rgb(path)
    if degr:
        img = DEGRADATIONS[degr](img)
    d = extract_all(img, apply_preprocessing=pre, use_mask=mask)
    return [d[g] for g in FEATURE_GROUPS]


def run(name, paths, labels, pre, mask, degr, workers):
    out = os.path.join(CACHE_DIR, f"feat_{name}.npz")
    if os.path.exists(out):
        print(f"[skip] {name}")
        return
    t = time.time()
    with Pool(workers) as pool:
        res = pool.map(_job, [(p, pre, mask, degr) for p in paths], chunksize=16)
    arrays = {g: np.stack([r[i] for r in res]) for i, g in enumerate(FEATURE_GROUPS)}
    np.savez_compressed(out, labels=np.asarray(labels), **arrays)
    dt = time.time() - t
    print(f"[done] {name}: {len(paths)} images in {dt:.0f}s ({1000 * dt / len(paths):.1f} ms/img)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    df, _ = get_split()
    paths, labels = df["path"].tolist(), df["label"].tolist()
    run("full", paths, labels, True, True, None, a.workers)
    run("no_pre", paths, labels, False, True, None, a.workers)
    run("no_mask", paths, labels, True, False, None, a.workers)

    test = df[df["split"] == "test"]
    for degr in DEGRADATIONS:
        if degr == "Clean":
            continue
        tag = degr.split(" (")[0].lower().replace(" ", "_")
        run(f"robust_{tag}_pre", test["path"].tolist(), test["label"].tolist(), True, True, degr, a.workers)
        run(f"robust_{tag}_nopre", test["path"].tolist(), test["label"].tolist(), False, True, degr, a.workers)


if __name__ == "__main__":
    main()
