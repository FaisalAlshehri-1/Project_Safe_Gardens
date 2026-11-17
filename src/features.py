"""
Stage 3 - Hand-crafted feature extraction.

For each preprocessed leaf image we compute four feature groups:

  COLOUR  (78)  colour moments (mean, std, skewness) of the leaf region in RGB,
                HSV and CIELAB (27) + the same moments of the lesion region (27)
                + a 16-bin hue histogram and 8-bin a* histogram of the leaf (24)
  TEXTURE (64)  GLCM on the grey leaf region: contrast, dissimilarity,
                homogeneity, energy, correlation, ASM at distances {1, 2, 4}
                (mean and range over 0/45/90/135 degrees) = 36, plus a uniform
                LBP (P=8, R=1) histogram (10) and LBP (P=16, R=2) histogram (18)
  SHAPE   (8)   lesion area ratio, number of lesion blobs, mean / max blob area,
                leaf solidity, Canny edge density inside the leaf, lesion
                boundary length ratio, leaf area ratio
  HOG     (1764) Histogram of Oriented Gradients on a 128x128 grey image
                (9 orientations, 16x16 cells, 2x2 blocks)
"""
import warnings

import cv2
import numpy as np
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern

from config import HOG_SIZE
from preprocessing import preprocess
from segmentation import boundary, leaf_mask, lesion_mask

warnings.filterwarnings("ignore", category=RuntimeWarning)

GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")
GLCM_DIST = (1, 2, 4)
GLCM_ANGLES = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)


def _moments(pixels):
    """Mean, standard deviation and skewness of each channel (N x C -> 3C)."""
    if len(pixels) < 3:
        return np.zeros(pixels.shape[1] * 3, np.float32)
    p = pixels.astype(np.float32)
    sk = np.nan_to_num(skew(p, axis=0))
    return np.concatenate([p.mean(0), p.std(0), sk])


def colour_features(rgb, leaf, lesion):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    stack = np.concatenate([rgb, hsv, lab], axis=2)              # H x W x 9
    f_leaf = _moments(stack[leaf])
    f_les = _moments(stack[lesion]) if lesion.sum() >= 3 else np.zeros(27, np.float32)
    hue_hist = np.histogram(hsv[..., 0][leaf], bins=16, range=(0, 180))[0]
    a_hist = np.histogram(lab[..., 1][leaf], bins=8, range=(96, 176))[0]
    n = max(leaf.sum(), 1)
    return np.concatenate([f_leaf, f_les, hue_hist / n, a_hist / n])


def texture_features(grey, leaf):
    # GLCM on a 64-level quantised grey image; background set to level 0 and
    # the (0, *) / (*, 0) co-occurrences removed so only leaf texture counts.
    q = (grey.astype(np.float32) / 256 * 63).astype(np.uint8) + 1
    q[~leaf] = 0
    glcm = graycomatrix(q, GLCM_DIST, GLCM_ANGLES, levels=64, symmetric=True, normed=False).astype(np.float64)
    glcm[0, :, :, :] = 0
    glcm[:, 0, :, :] = 0
    glcm /= np.maximum(glcm.sum(axis=(0, 1), keepdims=True), 1)
    glcm_f = []
    for prop in GLCM_PROPS:
        v = np.nan_to_num(graycoprops(glcm, prop))               # dist x angle
        glcm_f += [v.mean(1), v.max(1) - v.min(1)]
    glcm_f = np.concatenate(glcm_f)                              # 6 props x 3 dist x 2 = 36

    lbp_f = []
    for P, R in ((8, 1), (16, 2)):
        lbp = local_binary_pattern(grey, P, R, "uniform")
        h = np.histogram(lbp[leaf], bins=P + 2, range=(0, P + 2))[0].astype(np.float32)
        lbp_f.append(h / max(h.sum(), 1))
    return np.concatenate([glcm_f] + lbp_f)


def shape_features(grey, leaf, lesion):
    n_leaf = max(leaf.sum(), 1)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(lesion.astype(np.uint8), 8)
    areas = stats[1:, cv2.CC_STAT_AREA] / n_leaf if n > 1 else np.zeros(1)
    cnts, _ = cv2.findContours(leaf.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solidity = 0.0
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        hull = cv2.contourArea(cv2.convexHull(c))
        solidity = cv2.contourArea(c) / hull if hull > 0 else 0.0
    edges = cv2.Canny(grey, 50, 150) > 0
    return np.array([
        lesion.sum() / n_leaf,                 # lesion area ratio
        n - 1,                                 # number of lesion blobs
        areas.mean(), areas.max(),
        solidity,
        edges[leaf].mean() if leaf.any() else 0.0,
        boundary(lesion).sum() / n_leaf,
        leaf.mean(),
    ], np.float32)


def hog_features(grey):
    g = cv2.resize(grey, (HOG_SIZE, HOG_SIZE), interpolation=cv2.INTER_AREA)
    return hog(g, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
               block_norm="L2-Hys", feature_vector=True).astype(np.float32)


def extract_all(rgb, apply_preprocessing=True, use_mask=True):
    """Return a dict {group_name: vector} for one RGB image.

    apply_preprocessing / use_mask switch the corresponding stage off for the
    ablation study (without a mask the whole image is treated as leaf).
    """
    x = preprocess(rgb) if apply_preprocessing else rgb
    if use_mask:
        leaf = leaf_mask(x)
        if leaf.sum() < 0.02 * leaf.size:      # degenerate mask -> use whole frame
            leaf = np.ones(x.shape[:2], bool)
    else:
        leaf = np.ones(x.shape[:2], bool)
    lesion = lesion_mask(x, leaf)
    grey = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    return {
        "colour": colour_features(x, leaf, lesion).astype(np.float32),
        "texture": texture_features(grey, leaf).astype(np.float32),
        "shape": shape_features(grey, leaf, lesion),
        "hog": hog_features(grey),
    }


FEATURE_GROUPS = ("colour", "texture", "shape", "hog")
