# Multi-Label Deep Learning — Pneumonia & Pleural Effusion Detection

A multi-label chest X-ray classification model built with **DenseNet121** and trained on the **CheXpert** dataset to simultaneously detect **Pneumonia** and **Pleural Effusion** from frontal chest radiographs.

---

## Project Overview

This project tackles a **multi-label binary classification** problem where each chest X-ray can be positive for one or both conditions at the same time:

| Label              | Description                         |
| ------------------ | ----------------------------------- |
| `Pneumonia`        | Lung infection causing inflammation |
| `Pleural Effusion` | Fluid buildup around the lungs      |

---

## Dataset

- **Source:** [CheXpert (Stanford)](https://stanfordmlgroup.github.io/competitions/chexpert/) — `CheXpert-v1.0-small`
- **Images used:** Frontal views only
- **Split:** 95% training / 5% validation (stratified by label combination)
- **Uncertain labels (`-1`) handling:**
  - Pneumonia `-1` → `1` (treated as positive)
  - Pleural Effusion `-1` → `0` (treated as negative)

---

## Model Architecture

- **Backbone:** `DenseNet121` (pretrained on ImageNet)
- **Output:** 2 neurons with `Sigmoid` activation (one per label)
- **Loss:** `BCEWithLogitsLoss` with label smoothing (`0.9 / 0.05`)
- **Optimizer:** Adam
- **Scheduler:** `ReduceLROnPlateau`
- **Mixed Precision:** `torch.cuda.amp` (faster training on GPU)

---

## Training Pipeline

```
CheXpert CSV
    ↓
Frontal-only filter
    ↓
Label cleaning (NaN → 0, uncertain → policy)
    ↓
Stratified Train/Val Split (95/5)
    ↓
CheXpertDataset (custom PyTorch Dataset)
    ↓
DenseNet121 fine-tuning
    ↓
Epoch checkpoints saved (.pth)
    ↓
Evaluation on official validation set (AUROC, F1)
    ↓
GradCAM visualization
```

---

## Evaluation Metrics

| Metric       | Description                                                  |
| ------------ | ------------------------------------------------------------ |
| **AUROC**    | Area Under the ROC Curve (per label)                         |
| **F1 Score** | With tuned thresholds: `0.25` (Pneumonia), `0.45` (Effusion) |

---

## Explainability — GradCAM

The project includes a custom **GradCAM** implementation targeting `DenseNet121`'s `denseblock4` layer, generating heatmaps that highlight the regions of the X-ray that influenced each prediction.

---

## Project Structure

```
├── multi-label-dl-pneumonia-effusion.ipynb   # Main notebook
├── README.md
└── assets/
    └── demo.mp4                               # demo video
├── .gitignore                                # Excludes .pth files from Git
└── checkpoints/
    └── *.pth                                 # Saved model weights (not tracked by Git)

```

> **Note:** Model weights (`.pth` files) are **not included** in this repository due to file size. You can train the model using the notebook or download the weights separately.

---

## Requirements

```bash
pip install torch torchvision
pip install pandas scikit-learn matplotlib seaborn
pip install tqdm opencv-python Pillow
```

Or run directly on **Kaggle** with GPU enabled (recommended).

---

## How to Run

1. Download the [CheXpert-v1.0-small dataset](https://stanfordmlgroup.github.io/competitions/chexpert/) and place it under `/kaggle/input/datasets/ashery/chexpert/`
2. Open `multi-label-dl-pneumonia-effusion.ipynb`
3. Run all cells in order
4. Model checkpoints will be saved to `/kaggle/working/checkpoints/`

---

## Author

**Manel Teffahi** — [GitHub](https://github.com/Manel-Tfh)
**Sara Meftah** — [GitHub](https://github.com/saraMeftah)
**Gaçemi Raida**— [GitHub](https://github.com/Rayda1121)

---

## Supervisor

**Abderrahmane Khiat**

---

## License

Copyright © 2026 **Manel Teffahi & Sara Meftah & Raida Gaçemi ** — All rights reserved.

This project is licensed under the **MIT License**.
Anyone who uses or builds on this project **must keep our names** (Manel Teffahi & Sara Meftah & Raida Gaçemi ) in all copies and derivatives. Removing the copyright notice is a violation of the license.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
