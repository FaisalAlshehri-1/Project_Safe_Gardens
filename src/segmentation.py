"""
Stage 2 - Colour-space segmentation and morphology.

Two masks are produced for every leaf image:

1. LEAF mask (leaf vs. background)
   - Otsu thresholding of the HSV saturation channel (background paper is grey,
     i.e. low saturation; leaf tissue - healthy or diseased - is chromatic), OR
   - K-means (k=3) clustering of the CIELAB chromaticity plane (a*, b*), where
     the cluster with the lowest chroma sqrt(a*^2 + b*^2) is background.
   - Proposed: K-means on (a*, b*) followed by morphological closing/opening,
     largest-connected-component selection and hole filling. (Ablations: Otsu,
     Otsu/K-means fusion and GrabCut refinement were also evaluated.)

2. LESION mask (diseased vs. healthy tissue, inside the leaf)
   - K-means (k=3) on (a*, b*) of leaf pixels only; the cluster whose centre is
     least green (largest a*) is the lesion candidate.
   - It is accepted only where the pixel is genuinely non-green: a* above an
     Otsu threshold computed on the leaf's a* values AND hue outside the
     healthy-green band, or very dark necrotic tissue.
   - Morphological opening removes speckle; closing fills gaps inside spots.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi

from config import KMEANS_K, MORPH_KERNEL, SEED

_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)


def _kmeans(samples, k):
    cv2.setRNGSeed(SEED)
    _, labels, centres = cv2.kmeans(samples.astype(np.float32), k, None, _CRITERIA, 3, cv2.KMEANS_PP_CENTERS)
    return labels.ravel(), centres


def _largest_component(mask):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask.astype(bool)
    keep = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return lab == keep


def clean_mask(mask, largest=True, fill=True):
    """Morphological closing then opening, keep the largest blob, fill holes."""
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _KERNEL, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _KERNEL, iterations=1)
    m = m.astype(bool)
    if largest:
        m = _largest_component(m)
    if fill:
        m = ndi.binary_fill_holes(m)
    return m


# ------------------------------------------------------------------ leaf mask
def leaf_mask_otsu(rgb):
    """Otsu threshold on HSV saturation (raw, no post-processing)."""
    s = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1]
    _, m = cv2.threshold(s, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m.astype(bool)


def _is_leaf_colour(centres, min_b=4.0, min_chroma=6.0):
    """Leaf tissue (green, yellow or brown) lies in the b* > 0 half-plane of
    CIELAB and is chromatic; grey, blue or purple backgrounds and cast shadows
    are either achromatic or have b* <= 0."""
    a, b = centres[:, 0], centres[:, 1]
    return (b > min_b) & (np.hypot(a, b) > min_chroma)


def leaf_mask_kmeans(rgb, k=KMEANS_K):
    """K-means on (a*, b*); clusters with a leaf-like chromaticity are kept (raw)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    ab = lab[:, 1:] - 128.0
    labels, centres = _kmeans(ab, k)
    keep = _is_leaf_colour(centres)
    if not keep.any():                      # fallback: drop the least chromatic cluster
        keep = np.ones(k, bool)
        keep[np.argmin(np.hypot(centres[:, 0], centres[:, 1]))] = False
    return keep[labels].reshape(rgb.shape[:2])


def leaf_mask(rgb):
    """Proposed leaf segmentation: K-means on Lab chromaticity + morphology."""
    return clean_mask(leaf_mask_kmeans(rgb))


def leaf_mask_fusion(rgb):
    """Ablation: union of the Otsu-S and K-means cues, then morphology."""
    return clean_mask(leaf_mask_otsu(rgb) | leaf_mask_kmeans(rgb))


def leaf_mask_grabcut(rgb, band=15):
    """Ablation: GrabCut refinement initialised from the proposed mask."""
    m = leaf_mask(rgb)
    k = np.ones((band, band), np.uint8)
    inner = cv2.erode(m.astype(np.uint8), k) > 0
    outer = cv2.dilate(m.astype(np.uint8), k) > 0
    gc = np.full(m.shape, cv2.GC_BGD, np.uint8)
    gc[outer] = cv2.GC_PR_BGD
    gc[m] = cv2.GC_PR_FGD
    gc[inner] = cv2.GC_FGD
    try:
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), gc, None,
                    np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:          # leaf fills the whole frame -> nothing to refine
        return m
    return clean_mask((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD))


LEAF_METHODS = {
    "Otsu (HSV-S)": leaf_mask_otsu,
    "Otsu + morphology": lambda x: clean_mask(leaf_mask_otsu(x)),
    "K-means (Lab a*b*)": leaf_mask_kmeans,
    "Otsu/K-means fusion + morph.": leaf_mask_fusion,
    "K-means + GrabCut": leaf_mask_grabcut,
    "K-means + morphology (proposed)": leaf_mask,
}


# ---------------------------------------------------------------- lesion mask
def lesion_mask(rgb, leaf):
    """Segment diseased tissue inside the leaf mask."""
    if leaf.sum() < 50:
        return np.zeros_like(leaf)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    a, b, L = lab[..., 1].astype(np.float32), lab[..., 2].astype(np.float32), lab[..., 0]
    h, v = hsv[..., 0], hsv[..., 2]

    idx = np.flatnonzero(leaf.ravel())
    ab = np.stack([a.ravel()[idx], b.ravel()[idx]], 1) - 128.0
    labels, centres = _kmeans(ab, KMEANS_K)
    lesion_cluster = np.argmax(centres[:, 0])            # least green centre
    cand = np.zeros(leaf.size, bool)
    cand[idx[labels == lesion_cluster]] = True
    cand = cand.reshape(leaf.shape)

    # Otsu on a* of the leaf pixels decides how "non-green" a pixel must be
    a_leaf = a[leaf].astype(np.uint8).reshape(-1, 1)
    t_a, _ = cv2.threshold(a_leaf, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    non_green_hue = (h < 25) | (h > 95)                   # OpenCV hue: green ~ 35-85
    chroma_ok = a > max(t_a, 118)                         # 128 = neutral, lower = greener
    inner = cv2.erode(leaf.astype(np.uint8), _KERNEL, iterations=3).astype(bool)
    necrotic = (v < 60) & (L < 70) & inner                # dark dead tissue, away from edge shadows
    lesion = leaf & ((cand & chroma_ok & non_green_hue) | necrotic)

    lesion = cv2.morphologyEx(lesion.astype(np.uint8), cv2.MORPH_OPEN, _KERNEL)
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_CLOSE, _KERNEL)
    return lesion.astype(bool) & leaf


def boundary(mask):
    """Morphological gradient = dilation - erosion (object contour)."""
    m = mask.astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_GRADIENT, _KERNEL).astype(bool)


# ---------------------------------------------------------------- metrics
def iou_dice(pred, ref):
    pred, ref = pred.astype(bool), ref.astype(bool)
    inter = np.logical_and(pred, ref).sum()
    union = np.logical_or(pred, ref).sum()
    s = pred.sum() + ref.sum()
    return (inter / union if union else 1.0), (2 * inter / s if s else 1.0)


def reference_leaf_mask(segmented_rgb, thr=20):
    """PlantVillage 'segmented' images have a black background -> binary mask."""
    m = segmented_rgb.max(axis=2) > thr
    return ndi.binary_fill_holes(clean_mask(m, largest=True, fill=False))
