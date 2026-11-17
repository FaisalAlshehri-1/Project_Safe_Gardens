"""
Generate every figure and summary table used in the paper and the slides.

Run after extract_features.py, train_classical.py, train_cnn.py and
evaluate_segmentation.py. Figures go to results/figures/, tables to
results/tables/.
"""
import glob
import json
import os
import sys

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)

matplotlib.use("Agg")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import CACHE_DIR, FIG_DIR, TABLE_DIR, pretty  # noqa: E402
from data import get_split  # noqa: E402
from preprocessing import DEGRADATIONS, load_rgb, preprocess  # noqa: E402
from segmentation import boundary, leaf_mask, leaf_mask_kmeans, leaf_mask_otsu, lesion_mask  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
                     "savefig.bbox": "tight"})
C_CLASSIC, C_RF, C_CNN, C_GREY = "#2a7f62", "#8fbf9f", "#c0504d", "#7f7f7f"


def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name))
    plt.close(fig)
    print("saved", name)


def pick_examples(df):
    want = ["Tomato___Early_blight", "Apple___Apple_scab", "Grape___Black_rot", "Potato___healthy"]
    out = []
    for c in want:
        sub = df[df["class"] == c].sort_values("path")
        out.append(sub.path.iloc[5])
    return out


# ------------------------------------------------------------------ figures
def fig_dataset(df, classes):
    counts = df.groupby(["class", "split"]).size().unstack(fill_value=0)[["train", "val", "test"]]
    counts.index = [pretty(c) for c in counts.index]
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    counts.plot.barh(stacked=True, ax=ax, color=[C_CLASSIC, C_RF, C_CNN], width=0.8)
    ax.set_xlabel("Number of images")
    ax.invert_yaxis()
    ax.legend(["Train (70%)", "Validation (15%)", "Test (15%)"], loc="lower right", frameon=False)
    ax.tick_params(axis="y", labelsize=6.5)
    save(fig, "fig_dataset_distribution.png")

    rng = np.random.default_rng(0)
    sel = rng.choice(len(classes), 12, replace=False)
    fig, axes = plt.subplots(2, 6, figsize=(7.2, 2.8))
    for ax, k in zip(axes.ravel(), sorted(sel)):
        p = df[df.label == k].path.iloc[0]
        ax.imshow(load_rgb(p))
        ax.set_title(pretty(classes[k]).replace(": ", ":\n"), fontsize=6)
        ax.axis("off")
    save(fig, "fig_dataset_samples.png")


def fig_preprocessing(paths):
    img = load_rgb(paths[0])
    _, stages = preprocess(img, return_stages=True)
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.8))
    for ax, (name, im) in zip(axes[0], stages.items()):
        ax.imshow(im)
        ax.set_title(name)
        ax.axis("off")
    for ax, (name, im) in zip(axes[1], stages.items()):
        L = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)[..., 0]
        ax.hist(L.ravel(), 64, color=C_CLASSIC)
        ax.set_yticks([])
        ax.set_xlabel("L* intensity")
    save(fig, "fig_preprocessing_stages.png")


def fig_segmentation(paths):
    cols = ["Pre-processed", "a* channel", "Otsu\n(HSV-S)", "K-means\n(a*b*)",
            "Leaf mask\n(+ morphology)", "Lesion\nmask", "Overlay"]
    fig, axes = plt.subplots(len(paths), len(cols), figsize=(7.2, 1.15 * len(paths) + 0.3))
    for r, p in enumerate(paths):
        x = preprocess(load_rgb(p))
        a = cv2.cvtColor(x, cv2.COLOR_RGB2LAB)[..., 1]
        leaf = leaf_mask(x)
        les = lesion_mask(x, leaf)
        ov = x.copy()
        ov[~leaf] = (ov[~leaf] * 0.3).astype(np.uint8)
        ov[les] = (0.45 * ov[les] + 0.55 * np.array([255, 30, 30])).astype(np.uint8)
        ov[boundary(leaf)] = [255, 255, 0]
        ims = [x, a, leaf_mask_otsu(x), leaf_mask_kmeans(x), leaf, les, ov]
        for c, im in enumerate(ims):
            ax = axes[r, c]
            ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
            ax.axis("off")
            if r == 0:
                ax.set_title(cols[c], fontsize=7)
    save(fig, "fig_segmentation_pipeline.png")


def fig_seg_scores():
    s = pd.read_csv(os.path.join(TABLE_DIR, "segmentation_iou.csv"))
    s = s[s["input"] == "preprocessed"]
    order = s.groupby("method").iou.mean().sort_values().index
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    data = [s[s.method == m].iou for m in order]
    bp = ax.boxplot(data, vert=False, widths=0.6, patch_artist=True, showfliers=False)
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(C_CLASSIC if "proposed" in m else "#d9d9d9")
    for i, d in enumerate(data, 1):
        ax.plot(d.mean(), i, "k^", ms=4)
    ax.set_yticks(range(1, len(order) + 1), [m.replace(" (proposed)", "\n(proposed)") for m in order], fontsize=7)
    ax.set_xlabel("IoU with PlantVillage reference mask")
    save(fig, "fig_segmentation_iou.png")


def fig_lesion_ratio():
    lr = pd.read_csv(os.path.join(TABLE_DIR, "lesion_ratio.csv"))
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    bins = np.linspace(0, 0.6, 40)
    ax.hist(lr[lr.healthy].lesion_ratio.clip(0, 0.6), bins, alpha=0.7, color=C_CLASSIC, density=True, label="Healthy")
    ax.hist(lr[~lr.healthy].lesion_ratio.clip(0, 0.6), bins, alpha=0.6, color=C_CNN, density=True, label="Diseased")
    ax.set_xlabel("Segmented lesion area / leaf area")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    save(fig, "fig_lesion_ratio.png")


def fig_feature_groups(res):
    fg = pd.DataFrame(res["feature_groups"])
    piv = fg.pivot(index="features", columns="classifier", values="macro_f1")
    piv = piv.loc[[k for k in ["Shape", "Texture (GLCM+LBP)", "HOG", "Colour", "Colour+Texture",
                               "Colour+Texture+Shape", "All (C+T+S+HOG)"] if k in piv.index]]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    y = np.arange(len(piv))
    ax.barh(y - 0.2, piv["SVM (RBF)"] * 100, 0.4, color=C_CLASSIC, label="SVM (RBF)")
    ax.barh(y + 0.2, piv["Random Forest"] * 100, 0.4, color=C_RF, label="Random Forest")
    for yy, v in zip(y, piv["SVM (RBF)"]):
        ax.text(v * 100 + 0.5, yy - 0.2, f"{v * 100:.1f}", va="center", fontsize=6.5)
    ax.set_yticks(y, piv.index, fontsize=7)
    ax.set_xlabel("Test macro-F1 (%)")
    ax.set_xlim(0, 105)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    save(fig, "fig_feature_groups.png")


def fig_training():
    h = pd.DataFrame(json.load(open(os.path.join(TABLE_DIR, "cnn_history.json"))))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3))
    axes[0].plot(h.epoch, h.train_loss, "o-", color=C_CLASSIC, ms=3, label="Train")
    axes[0].plot(h.epoch, h.val_loss, "s-", color=C_CNN, ms=3, label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=False)
    axes[1].plot(h.epoch, h.train_acc * 100, "o-", color=C_CLASSIC, ms=3, label="Train accuracy")
    axes[1].plot(h.epoch, h.val_acc * 100, "s-", color=C_CNN, ms=3, label="Validation accuracy")
    axes[1].plot(h.epoch, h.val_f1 * 100, "^--", color=C_GREY, ms=3, label="Validation macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("%")
    axes[1].legend(frameon=False, fontsize=7)
    save(fig, "fig_cnn_training.png")


def fig_confusion(y, pred, classes, name, title):
    cm = confusion_matrix(y, pred, labels=range(len(classes)), normalize="true")
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    im = ax.imshow(cm, cmap="Greens", vmin=0, vmax=1)
    labels = [pretty(c) for c in classes]
    ax.set_xticks(range(len(classes)), labels, rotation=90, fontsize=5.5)
    ax.set_yticks(range(len(classes)), labels, fontsize=5.5)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] >= 0.05 and i != j:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=4.5, color="#c0504d")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    save(fig, name)


def fig_per_class(y, p_svm, p_cnn, classes, compact=False):
    f_svm = f1_score(y, p_svm, average=None, labels=range(len(classes)))
    f_cnn = f1_score(y, p_cnn, average=None, labels=range(len(classes)))
    order = np.argsort(f_svm)
    fig, ax = plt.subplots(figsize=(3.5, 6.2) if compact else (7.2, 6.0))
    yy = np.arange(len(classes))
    ax.barh(yy - 0.2, f_svm[order] * 100, 0.4, color=C_CLASSIC, label="Classical DIP + SVM")
    ax.barh(yy + 0.2, f_cnn[order] * 100, 0.4, color=C_CNN, label="MobileNetV2")
    ax.set_yticks(yy, [pretty(classes[i]).replace("Two-spotted spider mite", "").replace(" Gray leaf spot", "")
                       .replace("(Isariopsis Leaf Spot)", "").replace("(Citrus greening)", "").replace("(Black Measles)", "")
                       for i in order], fontsize=5.5 if compact else 6)
    ax.set_xlabel("Per-class F1 (%)")
    ax.set_ylim(-0.7, len(classes) - 0.3)
    ax.set_xlim(min(50, (f_svm.min() * 100) // 10 * 10), 100.5)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, fontsize=7 if compact else 9)
    save(fig, "fig_per_class_f1_col.png" if compact else "fig_per_class_f1.png")


def fig_robustness(res, compact=False):
    rob = pd.DataFrame(res["robustness"])
    cnn = []
    for tag in ["clean", "gaussian_noise", "gaussian_blur", "low_light", "uneven_illumination"]:
        z = np.load(os.path.join(CACHE_DIR, f"pred_cnn_{tag}.npz"))
        cnn.append({"degradation": tag, "pipeline": "MobileNetV2",
                    "macro_f1": f1_score(z["y"], z["p"].argmax(1), average="macro")})
    rob = pd.concat([rob, pd.DataFrame(cnn)])
    names = {"clean": "Clean", "gaussian_noise": "Gaussian\nnoise", "gaussian_blur": "Gaussian\nblur",
             "low_light": "Low\nlight", "uneven_illumination": "Uneven\nillumination"}
    piv = rob.pivot(index="degradation", columns="pipeline", values="macro_f1").loc[list(names)]
    fig, ax = plt.subplots(figsize=(3.5, 2.7) if compact else (7.2, 2.5))
    x = np.arange(len(piv))
    series = [("without pre-processing", "#d9d9d9", "Classical, no pre-processing"),
              ("with pre-processing", C_CLASSIC, "Classical, with pre-processing"),
              ("MobileNetV2", C_CNN, "MobileNetV2")]
    for i, (col, color, lab) in enumerate(series):
        ax.bar(x + (i - 1) * 0.27, piv[col] * 100, 0.27, color=color, label=lab, edgecolor="white")
        for xx, v in zip(x, piv[col]):
            ax.text(xx + (i - 1) * 0.27, v * 100 + 1, f"{v * 100:.0f}", ha="center", fontsize=5 if compact else 6)
    ax.set_xticks(x, [names[k] for k in piv.index], fontsize=6.5 if compact else 9)
    ax.set_ylabel("Test macro-F1 (%)")
    ax.set_ylim(0, 110)
    ax.legend(frameon=False, ncol=1 if compact else 3, fontsize=6.5 if compact else 7, loc="upper center",
              bbox_to_anchor=(0.5, 1.32 if compact else 1.18))
    if compact:
        save(fig, "fig_robustness_col.png")
        return
    save(fig, "fig_robustness.png")
    piv.to_csv(os.path.join(TABLE_DIR, "robustness_macro_f1.csv"))


def fig_degradations(path):
    img = load_rgb(path)
    fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.1))
    for c, (name, fn) in enumerate(DEGRADATIONS.items()):
        d = fn(img)
        axes[0, c].imshow(d)
        axes[0, c].set_title(name.replace(" (", "\n("), fontsize=7)
        axes[1, c].imshow(preprocess(d))
        for r in range(2):
            axes[r, c].axis("off")
    axes[0, 0].text(-30, 128, "Degraded", rotation=90, va="center", fontsize=7)
    axes[1, 0].text(-30, 128, "Pre-processed", rotation=90, va="center", fontsize=7)
    save(fig, "fig_degradations.png")


def summary_table(classes, res):
    """Overall test metrics for the main models."""
    rows = []
    best = res["best_feature_set"]
    models = [(f"SVM – {best}", f"pred_svm_{best}.npz", "s"),
              (f"Random Forest – {best}", f"pred_random_{best}.npz", "s"),
              ("SVM – HOG only", "pred_svm_HOG.npz", "s"),
              ("MobileNetV2 (fine-tuned)", "pred_cnn_clean.npz", "p")]
    preds = {}
    for name, fn, key in models:
        z = np.load(os.path.join(CACHE_DIR, fn))
        y, s = z["y"], z[key]
        pred = s.argmax(1)
        prob = np.exp(s - s.max(1, keepdims=True))
        prob /= prob.sum(1, keepdims=True)
        p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
        rows.append({"Model": name, "Accuracy": np.mean(pred == y), "Precision (macro)": p,
                     "Recall (macro)": r, "F1 (macro)": f,
                     "ROC-AUC (macro OvR)": roc_auc_score(y, prob, multi_class="ovr", average="macro")})
        preds[name] = (y, pred)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TABLE_DIR, "main_results.csv"), index=False)
    print(t.to_string(index=False))
    for name, (y, pred) in preds.items():
        rep = classification_report(y, pred, target_names=[pretty(c) for c in classes], digits=4, zero_division=0)
        slug = "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")
        with open(os.path.join(TABLE_DIR, f"classification_report_{slug}.txt"), "w") as f:
            f.write(rep)
    return preds


def main():
    df, classes = get_split(save=False)
    res = json.load(open(os.path.join(TABLE_DIR, "classical_results.json")))
    ex = pick_examples(df)
    fig_dataset(df, classes)
    fig_preprocessing(ex)
    fig_segmentation(ex)
    fig_degradations(ex[0])
    fig_seg_scores()
    fig_lesion_ratio()
    fig_feature_groups(res)
    fig_training()
    preds = summary_table(classes, res)
    best = res["best_feature_set"]
    y, p_svm = preds[f"SVM – {best}"]
    _, p_cnn = preds["MobileNetV2 (fine-tuned)"]
    fig_confusion(y, p_svm, classes, "fig_confusion_svm.png", f"Classical pipeline (SVM, {best})")
    fig_confusion(y, p_cnn, classes, "fig_confusion_cnn.png", "MobileNetV2 (fine-tuned)")
    fig_per_class(y, p_svm, p_cnn, classes)
    fig_per_class(y, p_svm, p_cnn, classes, compact=True)
    fig_robustness(res, compact=True)
    fig_robustness(res)


if __name__ == "__main__":
    main()
