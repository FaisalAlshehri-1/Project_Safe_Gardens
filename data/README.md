# Data

Images are not stored in the repository. In `notebooks/Safe_Gardens_Pipeline.ipynb`, set
`DOWNLOAD_DATA = True` in the setup cell and run section 1.1. This downloads the class-balanced
PlantVillage subset used in the paper (seed 42, up to 300 images per class) into:

- `data/plantvillage/color/<class>/*.jpg` – 11,227 images, 38 classes
- `data/plantvillage/segmented/<class>/*_final_masked.jpg` – 555 reference leaf masks

The exact file list and train/validation/test assignment is in `results/tables/data_split.csv`.

The ImageNet weights for MobileNetV2 (`mobilenetv2_1.0-0c6065bc.pth`, Apache-2.0, from
https://github.com/d-li14/mobilenetv2.pytorch) go in `data/weights/`.
