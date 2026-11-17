"""
Download a reproducible, class-balanced subset of the PlantVillage dataset.

The PlantVillage images (Hughes & Salathe, 2015) are mirrored in the public
GitHub repository spMohanty/PlantVillage-Dataset. This script:
  1. lists every image path in that repository (blob-less git clone, no images),
  2. draws a seeded random sample of N images per class (raw/color),
  3. also pulls the matching "segmented" (background-removed) version for a
     subset of the sampled images; these are used as reference leaf masks to
     score our segmentation,
  4. downloads the files in parallel from raw.githubusercontent.com.

Usage:
    python scripts/download_dataset.py --per-class 200 --seg-per-class 10
"""
import argparse
import collections
import os
import random
import subprocess
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = "https://github.com/spMohanty/PlantVillage-Dataset"
RAW = "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/"


def list_repo_files(workdir):
    """Clone the repository without file contents and list all image paths."""
    if not os.path.isdir(os.path.join(workdir, ".git")):
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none",
                        "--no-checkout", "-q", REPO, workdir], check=True)
    out = subprocess.run(["git", "-C", workdir, "ls-tree", "-r", "--name-only",
                          "HEAD", "raw/color", "raw/segmented"],
                         check=True, capture_output=True, text=True).stdout
    return out.splitlines()


def select(paths, per_class, seg_per_class, seed):
    """Seeded, class-balanced sample of colour images + matching masks."""
    rng = random.Random(seed)
    seg = {p for p in paths if p.startswith("raw/segmented/")}
    by_class = collections.defaultdict(list)
    for p in paths:
        if p.startswith("raw/color/"):
            by_class[p.split("/")[2]].append(p)
    colour, masks = [], []
    for cls in sorted(by_class):
        files = sorted(by_class[cls])
        rng.shuffle(files)
        pick = files[:per_class]
        colour += pick
        n = 0
        for p in pick:
            stem = os.path.splitext(os.path.basename(p))[0]
            s = f"raw/segmented/{cls}/{stem}_final_masked.jpg"
            if s in seg and n < seg_per_class:
                masks.append(s)
                n += 1
    return colour, masks


def fetch(rel, out_root):
    dst = os.path.join(out_root, rel.replace("raw/", "", 1))
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url = RAW + urllib.parse.quote(rel)
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            with open(dst, "wb") as f:
                f.write(data)
            return True
        except Exception:
            continue
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--seg-per-class", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "plantvillage"))
    ap.add_argument("--listing-dir", default="/tmp/PlantVillage-listing")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()

    paths = list_repo_files(a.listing_dir)
    colour, masks = select(paths, a.per_class, a.seg_per_class, a.seed)
    todo = colour + masks
    print(f"{len(colour)} colour images, {len(masks)} reference masks")
    ok = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(lambda p: fetch(p, a.out), todo), 1):
            ok += r
            if i % 500 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    print(f"downloaded {ok}/{len(todo)} files into {os.path.abspath(a.out)}")
    sys.exit(0 if ok == len(todo) else 1)


if __name__ == "__main__":
    main()
