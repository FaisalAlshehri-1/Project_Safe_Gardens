"""
Fine-tune MobileNetV2 and evaluate it on the clean and degraded test sets.

Outputs
  models/mobilenetv2_best.pt          best checkpoint (validation macro-F1)
  results/tables/cnn_history.json     per-epoch training log
  data/cache/pred_cnn_<setting>.npz   test-set labels and class probabilities

Usage:  python scripts/train_cnn.py --epochs 12
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cnn import EVAL_TF, _loader, load_trained, predict, train  # noqa: E402
from config import CACHE_DIR, MODEL_DIR, TABLE_DIR  # noqa: E402
from data import get_split  # noqa: E402
from preprocessing import DEGRADATIONS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--skip-train", action="store_true")
    a = ap.parse_args()
    df, classes = get_split()
    ckpt = os.path.join(MODEL_DIR, "mobilenetv2_best.pt")

    if not a.skip_train:
        hist, ckpt = train(df, len(classes), epochs=a.epochs, log=lambda s: print(s, flush=True))
        with open(os.path.join(TABLE_DIR, "cnn_history.json"), "w") as f:
            json.dump(hist, f, indent=1)

    model = load_trained(len(classes), ckpt)
    te = df[df.split == "test"]
    for name, fn in DEGRADATIONS.items():
        tag = "clean" if name == "Clean" else name.split(" (")[0].lower().replace(" ", "_")
        loader = _loader(te.path, te.label, EVAL_TF, False, degrade=None if name == "Clean" else fn)
        t = time.time()
        y, p = predict(model, loader)
        ms = 1000 * (time.time() - t) / len(y)
        np.savez_compressed(os.path.join(CACHE_DIR, f"pred_cnn_{tag}.npz"), y=y, p=p, ms=ms)
        print(f"{name}: acc={np.mean(p.argmax(1) == y):.4f}  ({ms:.1f} ms/img incl. loading)", flush=True)

    # pure model latency (batch of 1, no data loading)
    torch.set_num_threads(1)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        t = time.time()
        for _ in range(50):
            model(x)
    lat = 1000 * (time.time() - t) / 50
    size_mb = os.path.getsize(ckpt) / 1e6
    n_params = sum(p.numel() for p in model.parameters())
    with open(os.path.join(TABLE_DIR, "cnn_efficiency.json"), "w") as f:
        json.dump({"latency_ms_per_image_1thread": lat, "checkpoint_mb": size_mb, "parameters": n_params}, f, indent=1)
    print(f"latency {lat:.1f} ms/img, {size_mb:.1f} MB, {n_params} params")


if __name__ == "__main__":
    main()
