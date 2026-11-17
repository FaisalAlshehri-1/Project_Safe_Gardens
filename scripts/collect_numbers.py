"""Collect every number quoted in the paper and slides into results/tables/summary_numbers.json."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import TABLE_DIR  # noqa: E402

T = lambda f: os.path.join(TABLE_DIR, f)  # noqa: E731


def main():
    split = pd.read_csv(T("data_split.csv"))
    res = json.load(open(T("classical_results.json")))
    seg = pd.read_csv(T("segmentation_summary.csv"))
    main_t = pd.read_csv(T("main_results.csv"))
    hist = json.load(open(T("cnn_history.json")))
    eff = json.load(open(T("cnn_efficiency.json")))
    lr = json.load(open(T("lesion_ratio_summary.json")))
    rob = pd.read_csv(T("robustness_macro_f1.csv"), index_col=0)

    best_ep = max(hist, key=lambda r: r["val_f1"])
    N = {
        "n_images": int(len(split)), "n_classes": int(split["class"].nunique()),
        "n_train": int((split.split == "train").sum()), "n_val": int((split.split == "val").sum()),
        "n_test": int((split.split == "test").sum()),
        "seg": {f"{r.input}|{r.method}": {"iou": r.iou_mean, "iou_median": r.iou_median, "dice": r.dice_mean,
                                          "ms": r.ms_mean, "n": int(r.n)} for r in seg.itertuples()},
        "lesion": lr,
        "feature_groups": res["feature_groups"],
        "best_feature_set": res["best_feature_set"],
        "ablation": res["ablation"],
        "robustness": rob.reset_index().to_dict(orient="records"),
        "main": main_t.to_dict(orient="records"),
        "cnn": {"epochs": len(hist), "best_epoch": best_ep["epoch"], "best_val_f1": best_ep["val_f1"],
                "minutes": sum(r["seconds"] for r in hist) / 60, **eff},
    }
    with open(T("summary_numbers.json"), "w") as f:
        json.dump(N, f, indent=1, default=float)
    print(json.dumps({k: N[k] for k in ("n_images", "n_train", "n_val", "n_test", "best_feature_set")}))
    print(main_t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
