"""
Train and evaluate the classical (hand-crafted feature) classifiers.

Experiments
  E1  feature-group study    SVM and Random Forest on each feature group and
                             their combinations (preprocessing + leaf mask ON)
  E2  pipeline ablation      best feature set with preprocessing OFF / mask OFF
  E3  robustness             clean-trained models evaluated on degraded test
                             images, with vs. without the preprocessing stage

SVM hyper-parameters (C, gamma) are selected on the validation split; the
final model is refit on train only and scored once on the held-out test split.

Usage:  python scripts/train_classical.py
"""
import json
import os
import sys
import time

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import CACHE_DIR, MODEL_DIR, SEED, TABLE_DIR  # noqa: E402
from data import get_split  # noqa: E402

FEATURE_SETS = {
    "Colour": ["colour"],
    "Texture (GLCM+LBP)": ["texture"],
    "Shape": ["shape"],
    "HOG": ["hog"],
    "Colour+Texture": ["colour", "texture"],
    "Colour+Texture+Shape": ["colour", "texture", "shape"],
    "All (C+T+S+HOG)": ["colour", "texture", "shape", "hog"],
}
DEGR_TAGS = ["gaussian_noise", "gaussian_blur", "low_light", "uneven_illumination"]


def load(name):
    return np.load(os.path.join(CACHE_DIR, f"feat_{name}.npz"))


def stack(z, groups):
    return np.hstack([z[g] for g in groups]).astype(np.float32)


class _PCASVC(SVC):
    """RBF-SVM whose gamma is set relative to the PCA output dimension."""

    def __init__(self, C=1.0, gamma_factor=1.0):
        super().__init__(C=C, kernel="rbf", decision_function_shape="ovr", random_state=SEED)
        self.gamma_factor = gamma_factor

    def fit(self, X, y, sample_weight=None):
        self.gamma = self.gamma_factor / (X.shape[1] * X.var())
        return super().fit(X, y, sample_weight)


def svm_pipeline(C, gamma, n_feat):
    steps = [StandardScaler()]
    if n_feat > 400:                       # PCA keeps 99% variance of the large HOG block
        steps.append(PCA(n_components=0.99, svd_solver="full", random_state=SEED))
    if isinstance(gamma, str) and gamma.startswith("pca:"):
        steps.append(_PCASVC(C=C, gamma_factor=float(gamma[4:])))
    else:
        steps.append(SVC(C=C, gamma=gamma, kernel="rbf", decision_function_shape="ovr", random_state=SEED))
    return make_pipeline(*steps)


def fit_svm(Xtr, ytr, Xva, yva):
    """Grid search on the validation split: C in {10, 30, 100}, gamma = f / d with
    f in {0.25, 0.5, 1} (d = number of input features after scaling/PCA;
    f = 1 corresponds to sklearn's gamma='scale' for standardised data)."""
    best = (-1, None)
    d = Xtr.shape[1] if Xtr.shape[1] <= 400 else None
    for C in (10, 30, 100):
        for f in (0.25, 0.5, 1.0):
            gamma = f / d if d else f"pca:{f}"
            m = svm_pipeline(C, gamma, Xtr.shape[1]).fit(Xtr, ytr)
            f = f1_score(yva, m.predict(Xva), average="macro")
            if f > best[0]:
                best = (f, (C, gamma), m)
    return best[2], {"C": best[1][0], "gamma": best[1][1], "val_macro_f1": best[0]}


def fit_rf(Xtr, ytr, Xva, yva):
    m = RandomForestClassifier(n_estimators=500, max_features="sqrt", n_jobs=-1, random_state=SEED).fit(Xtr, ytr)
    return m, {"n_estimators": 500, "val_macro_f1": f1_score(yva, m.predict(Xva), average="macro")}


def scores(model, X):
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)


def evaluate(model, X, y):
    t = time.time()
    s = scores(model, X)
    ms = 1000 * (time.time() - t) / len(y)
    pred = s.argmax(1)
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
            "weighted_f1": f1_score(y, pred, average="weighted"), "classify_ms_per_img": ms}, s


def main():
    df, classes = get_split()
    split = df["split"].to_numpy()
    tr, va, te = split == "train", split == "val", split == "test"
    full = load("full")
    y = full["labels"]
    results = {"feature_groups": [], "ablation": [], "robustness": []}

    # ---------------- E1: feature-group study ---------------------------------
    best_set, best_f1, best_svm = None, -1, None
    for name, groups in FEATURE_SETS.items():
        X = stack(full, groups)
        for clf_name, fitter in (("SVM (RBF)", fit_svm), ("Random Forest", fit_rf)):
            t = time.time()
            m, hp = fitter(X[tr], y[tr], X[va], y[va])
            fit_s = time.time() - t
            met, s = evaluate(m, X[te], y[te])
            row = {"features": name, "dims": X.shape[1], "classifier": clf_name, **met, **hp, "fit_seconds": fit_s}
            results["feature_groups"].append(row)
            print(json.dumps(row), flush=True)
            np.savez_compressed(os.path.join(CACHE_DIR, f"pred_{clf_name.split()[0].lower()}_{name}.npz"),
                                y=y[te], s=s)
            if clf_name.startswith("SVM") and met["macro_f1"] > best_f1:
                best_set, best_f1, best_svm = name, met["macro_f1"], m
    joblib.dump(best_svm, os.path.join(MODEL_DIR, "svm_best.joblib"), compress=3)

    groups = FEATURE_SETS[best_set]
    best_hp = next(r for r in results["feature_groups"] if r["features"] == best_set and r["classifier"] == "SVM (RBF)")
    C, G = best_hp["C"], best_hp["gamma"]

    # ---------------- E2: ablation of preprocessing and leaf mask ------------
    trained = {"full": best_svm}
    for cfg, label in (("full", "Full pipeline (pre-processing + leaf mask)"),
                       ("no_pre", "Without pre-processing"),
                       ("no_mask", "Without leaf segmentation mask")):
        z = load(cfg)
        X = stack(z, groups)
        m = trained.get(cfg) or svm_pipeline(C, G, X.shape[1]).fit(X[tr], y[tr])
        trained[cfg] = m
        met, _ = evaluate(m, X[te], y[te])
        row = {"setting": label, "features": best_set, **met}
        results["ablation"].append(row)
        print(json.dumps(row), flush=True)

    # ---------------- E3: robustness to image degradations -------------------
    Xte_full = stack(full, groups)[te]
    Xte_nopre = stack(load("no_pre"), groups)[te]
    for tag, Xp, Xn in [("clean", Xte_full, Xte_nopre)] + [
            (t, stack(load(f"robust_{t}_pre"), groups), stack(load(f"robust_{t}_nopre"), groups)) for t in DEGR_TAGS]:
        for prep, X, m in (("with pre-processing", Xp, trained["full"]), ("without pre-processing", Xn, trained["no_pre"])):
            met, _ = evaluate(m, X, y[te])
            row = {"degradation": tag, "pipeline": prep, **met}
            results["robustness"].append(row)
            print(json.dumps(row), flush=True)

    results["best_feature_set"] = best_set
    with open(os.path.join(TABLE_DIR, "classical_results.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
