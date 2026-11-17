"""
Stage 1 - Image preprocessing.

    resize (256x256) -> Gaussian smoothing -> bilateral filtering -> CLAHE on L*

* Gaussian low-pass filtering suppresses sensor (Gaussian-like) noise.
* The bilateral filter is an edge-preserving smoother: it averages pixels that
  are close in BOTH space and intensity, so lesion borders stay sharp.
* CLAHE (Contrast Limited Adaptive Histogram Equalisation) is applied only to
  the lightness channel L* of CIELAB, so illumination is normalised without
  shifting the chromatic channels a*/b* that carry the disease colour cues.
"""
import cv2
import numpy as np

from config import (BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE,
                    CLAHE_CLIP, CLAHE_TILE, GAUSS_KSIZE, GAUSS_SIGMA, IMG_SIZE)


def load_rgb(path, size=IMG_SIZE):
    """Read an image from disk as an RGB uint8 array of shape (size, size, 3)."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"cannot read {path}")
    if bgr.shape[:2] != (size, size):
        bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def gaussian(rgb):
    return cv2.GaussianBlur(rgb, (GAUSS_KSIZE, GAUSS_KSIZE), GAUSS_SIGMA)


def bilateral(rgb):
    return cv2.bilateralFilter(rgb, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)


def clahe_lab(rgb):
    """CLAHE on the L* channel of CIELAB only."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess(rgb, return_stages=False):
    """Full preprocessing chain. Returns the enhanced RGB image (and stages if asked)."""
    g = gaussian(rgb)
    b = bilateral(g)
    c = clahe_lab(b)
    if return_stages:
        return c, {"Original": rgb, "Gaussian": g, "Bilateral": b, "CLAHE (L*)": c}
    return c


# --------------------------------------------------------------------------
# Synthetic degradations used for the robustness experiment
# --------------------------------------------------------------------------
def add_gaussian_noise(rgb, sigma=20, seed=0):
    rng = np.random.default_rng(seed)
    noisy = rgb.astype(np.float32) + rng.normal(0, sigma, rgb.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def darken(rgb, gamma=2.2, gain=0.6):
    """Under-exposure: gamma > 1 compresses mid-tones, gain scales intensity down."""
    x = (rgb.astype(np.float32) / 255.0) ** gamma * gain
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def blur(rgb, k=7):
    return cv2.GaussianBlur(rgb, (k, k), 0)


def uneven_illumination(rgb, strength=0.6):
    """Linear left-to-right illumination gradient (shadow on one side)."""
    h, w = rgb.shape[:2]
    ramp = np.linspace(1.0 - strength, 1.0, w, dtype=np.float32)[None, :, None]
    return np.clip(rgb.astype(np.float32) * ramp, 0, 255).astype(np.uint8)


DEGRADATIONS = {
    "Clean": lambda x: x,
    "Gaussian noise (sigma=20)": add_gaussian_noise,
    "Gaussian blur (7x7)": blur,
    "Low light (gamma=2.2)": darken,
    "Uneven illumination": uneven_illumination,
}
