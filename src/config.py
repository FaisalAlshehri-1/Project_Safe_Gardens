"""Central configuration for the Safe Gardens pipeline."""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.environ.get("SAFE_GARDENS_DATA", os.path.join(ROOT, "data", "plantvillage"))
COLOR_DIR = os.path.join(DATA_DIR, "color")          # RGB leaf images, one folder per class
SEGMENTED_DIR = os.path.join(DATA_DIR, "segmented")  # PlantVillage background-removed references
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
TABLE_DIR = os.path.join(RESULTS_DIR, "tables")
MODEL_DIR = os.path.join(ROOT, "models")
CACHE_DIR = os.path.join(ROOT, "data", "cache")
PRETRAINED_MNV2 = os.environ.get(
    "MNV2_WEIGHTS", os.path.join(ROOT, "data", "weights", "mobilenetv2_1.0-0c6065bc.pth"))

SEED = 42
IMG_SIZE = 256          # all images are resized to 256 x 256 before processing
HOG_SIZE = 128          # HOG is computed on a 128 x 128 grey image
CNN_SIZE = 224          # MobileNetV2 input resolution
SPLIT = (0.70, 0.15, 0.15)  # stratified train / validation / test

# Preprocessing parameters
GAUSS_KSIZE, GAUSS_SIGMA = 5, 1.0
BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE = 9, 50, 50
CLAHE_CLIP, CLAHE_TILE = 2.0, (8, 8)

# Segmentation parameters
KMEANS_K = 3
MORPH_KERNEL = 5

for d in (RESULTS_DIR, FIG_DIR, TABLE_DIR, MODEL_DIR, CACHE_DIR):
    os.makedirs(d, exist_ok=True)


def pretty(cls_name):
    """'Tomato___Early_blight' -> 'Tomato: Early blight'."""
    crop, _, disease = cls_name.partition("___")
    crop = crop.replace("_", " ").replace(",", "").replace("(including sour)", "").replace("(maize)", "").strip()
    return f"{crop}: {disease.replace('_', ' ').strip()}"
