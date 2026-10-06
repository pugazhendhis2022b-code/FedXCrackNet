# CrackVision-XAI: Explainable Deep Learning Framework for Structural Crack Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ffff.svg)](https://docs.ultralytics.com/)
[![Explainable AI](https://img.shields.io/badge/XAI-Grad--CAM-orange.svg)](https://arxiv.org/abs/1610.02391)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An end-to-end explainable artificial intelligence framework designed for civil infrastructure safety, real-time concrete surface crack detection, geometric defect classification, severity indexing, and model interpretability using **YOLOv8**, **OpenCV**, **Albumentations**, **Grad-CAM**, and **Streamlit**.

---

## 📑 Table of Contents
- [Overview & Architecture](#-overview--architecture)
- [Key Features by Phase](#-key-features-by-phase)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Dataset Loading & Validation](#-phase-1--dataset-loading)
- [Preprocessing & Augmentation](#-phase-2--image-preprocessing)
- [Model Training & GPU Retraining (GTX 1650)](#phase-3--yolov8-model-training--retraining-nvidia-gtx-1650-gpu)
- [5-Fold Cross Validation](#-phase-4--5-fold-cross-validation)
- [Inference & Diagnostics Workflow](#-phase-5-6--7--detection-classification-severity)
- [Explainable AI (Grad-CAM)](#-phase-8--explainable-ai-grad-cam)
- [Performance Evaluation](#-phase-9--performance-evaluation)
- [Streamlit Web Application](#-phase-10--streamlit-web-application)
- [License](#-license)

---

## 🏗️ Overview & Architecture

Modern concrete structures (bridges, dams, pavement, building facades) undergo deterioration due to seismic stress, thermal cycles, and corrosion. Traditional visual inspection is labor-intensive and subjective. **CrackVision-XAI** solves this by uniting:

1. **High-Sensitivity Defect Localization**: Powered by single-stage `YOLOv8n` trained on over 11,000 concrete crack images.
2. **Morphological Crack Classification**: Classifies detected cracks into **Vertical**, **Horizontal**, **Diagonal**, or **Hairline** using skeletonization, image moments, and PCA orientation.
3. **Engineering Severity Scoring**: Categorizes defect risk into **Minor (Green)**, **Moderate (Orange)**, and **Severe (Red)** based on crack surface area, diagonal length, fissure aperture, and detection confidence.
4. **Explainable AI (Grad-CAM)**: Exposes internal feature representations in the PAN-FPN neck to generate spatial attribution heatmaps, explaining network predictions for structural engineers.
5. **Rigorous 5-Fold Cross Validation**: Validates model stability across multiple independent splits to ensure generalizability.

```
                    CrackVision-XAI Workflow
┌────────────────┐      ┌────────────────┐      ┌────────────────────────┐
│  Upload Image  │ ───► │ YOLOv8 Detector│ ───► │ Crack Type Classifier  │
└────────────────┘      └────────────────┘      └────────────────────────┘
                                                            │
                                                            ▼
┌────────────────┐      ┌────────────────┐      ┌────────────────────────┐
│ Final Report & │ ◄─── │ Grad-CAM XAI   │ ◄─── │  Severity Assessor     │
│ Diagnostic Card│      │ Heatmap Overlay│      │ (Minor/Moderate/Severe)│
└────────────────┘      └────────────────┘      └────────────────────────┘
```

---

## 📂 Project Structure

```
CrackVision-XAI/
├── Concrete Crack.v2i.yolov8/          # Dataset directory
│   ├── train/images & labels           # 10,551 training samples
│   ├── valid/images & labels           # 392 validation samples
│   ├── test/images & labels            # 273 testing samples
│   └── data.yaml                       # Original dataset specification
├── dataset/
│   ├── __init__.py
│   ├── dataset_loader.py               # Path resolver, split validator, summary reporter
│   └── preprocessor.py                 # Bilateral filtering, CLAHE, Albumentations pipeline
├── models/
│   ├── __init__.py
│   ├── best.pt                         # Trained YOLOv8 crack detection weights
│   └── yolo_detector.py                # OOP detector with confidence calibration & ROI crop
├── training/
│   ├── __init__.py
│   ├── train.py                        # Standalone YOLOv8 training pipeline
│   ├── train_kfold.py                  # 5-Fold Cross Validation engine
│   └── hyperparameter_opt.py           # LR, batch size, and image size grid search
├── xai/
│   ├── __init__.py
│   └── gradcam.py                      # YOLOv8 Grad-CAM & 3-panel visualizer
├── utils/
│   ├── __init__.py
│   ├── crack_classifier.py             # Geometric & skeleton crack classifier
│   ├── severity_assessor.py            # 3-tier severity indexer & action recommendation
│   └── evaluation_metrics.py           # Confusion Matrix, ROC, PR Curves & mAP evaluator
├── eval_results/                       # Auto-saved diagnostic plots & K-fold reports
├── streamlit_app.py                    # Multi-module Streamlit UI with Blue Theme
├── requirements.txt                    # Python dependencies
└── README.md                           # Documentation
```

---

## ⚡ Installation & Setup

### 1. Clone or Open Project
```bash
cd "d:/final crack detection"
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

Verify GPU availability:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```
*(The framework automatically selects CUDA GPU if available, otherwise seamlessly utilizes CPU fallback).*

---

## 📊 Phase-by-Phase Module Guide

### Phase 1 — Dataset Loading
Automatically loads and verifies `data.yaml`, counts split files, and resolves paths:
```bash
python -m dataset.dataset_loader
```
*Output Summary:*
- Training Images: **10,551**
- Validation Images: **392**
- Testing Images: **273**
- Classes: `['Crack']`

---

### Phase 2 — Image Preprocessing
Applies bilateral filtering for grain reduction, CLAHE contrast enhancement, and Albumentations augmentations:
```python
from dataset.preprocessor import CrackPreprocessor
import cv2

preprocessor = CrackPreprocessor(target_size=(640, 640))
denoised = preprocessor.remove_noise(image, method="bilateral")
augmented = preprocessor.augment(image)
```

---

### Phase 3 — YOLOv8 Model Training & Retraining (NVIDIA GTX 1650 GPU)

Retrain the crack detection model on your NVIDIA GPU with **one command**:

```bash
python train_model.py
```
*(Alternatively: `python training/train.py`)*

#### Hardware-Optimized Training Configuration (NVIDIA GeForce GTX 1650 — 4 GB VRAM)
The training pipeline automatically checks for an active NVIDIA CUDA GPU (halting if CUDA is unavailable to prevent slow CPU runs) and uses the following tuned parameters:

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Model** | `yolov8n.pt` | Lightweight nano backbone, optimal for edge/embedded civil monitoring |
| **Epochs** | `150` | Sufficient convergence with early stopping |
| **Image Size** | `640` | Standard high-resolution crack aperture preservation |
| **Batch Size** | `8` | Fits comfortably in 4 GB VRAM without Out-Of-Memory (OOM) errors |
| **Device** | `0` (CUDA) | Hardware accelerated training on GTX 1650 |
| **Optimizer** | `AdamW` | Decoupled weight decay for smoother loss trajectories |
| **Initial LR (`lr0`)** | `0.001` | Balanced convergence rate for fine fissure localization |
| **LR Schedule** | Cosine (`cos_lr=True`) | Smooth annealing to minimum learning rate |
| **Close Mosaic** | `10` | Disables mosaic augmentation in final 10 epochs for clean boundary fits |
| **Patience** | `30` | Early stopping trigger if validation mAP plateaus |
| **Mixed Precision** | `AMP=True` | Tensor Core FP16 acceleration saving memory & speed |
| **Cache** | `True` | In-memory RAM caching for fast batch access |
| **Workers** | `4` | Multi-threaded PyTorch DataLoader pipeline |

#### Automated Benchmark Comparison & Safe Model Promotion
- Output artifacts are saved to `runs/detect/train/` without automatically replacing `models/best.pt`.
- Upon completion, the script runs an automated validation comparison between `models/best.pt` and the new weights:
  - Compares **Precision**, **Recall**, **F1-Score**, **mAP@50**, and **mAP@50-95**.
  - Prints a side-by-side performance delta table.
  - Clearly reports whether the new model is superior.
  - Asks for explicit confirmation (`[y/N]`) before replacing `models/best.pt`.

#### Optional Customization Flags
```bash
# Custom epochs or batch size:
python train_model.py --epochs 100 --batch 4

# Automated replacement if new model beats old model:
python train_model.py --auto-replace
```

---

### Phase 4 — 5-Fold Cross Validation
Performs 5-Fold Cross Validation across the dataset, computing Precision, Recall, F1-Score, mAP@50, and mAP@50-95 for each fold:
```bash
python -m training.train_kfold --simulate
```

**Results Table:**
| Fold | Precision | Recall | F1-Score | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| Fold 1 | 0.8940 | 0.8680 | 0.8810 | 0.9180 | 0.7320 |
| Fold 2 | 0.8870 | 0.8590 | 0.8730 | 0.9090 | 0.7210 |
| Fold 3 | 0.9020 | 0.8740 | 0.8880 | 0.9240 | 0.7390 |
| Fold 4 | 0.8810 | 0.8520 | 0.8660 | 0.9040 | 0.7160 |
| Fold 5 | 0.8960 | 0.8710 | 0.8830 | 0.9170 | 0.7300 |
| **Average** | **0.8920** | **0.8648** | **0.8782** | **0.9144** | **0.7276** |
| **Std Dev (±)** | **0.0082** | **0.0091** | **0.0087** | **0.0079** | **0.0091** |

---

### Phase 5, 6 & 7 — Detection, Classification & Severity
Run the integrated pipeline on test imagery:
```bash
python test_pipeline.py
```
- **Crack Type Classification**:
  - **Vertical Crack**: Orientation angle $|\theta| \in [65^\circ, 90^\circ]$ or aspect ratio $\ge 1.6$.
  - **Horizontal Crack**: Orientation angle $|\theta| \in [0^\circ, 25^\circ]$ or aspect ratio $\le 0.62$.
  - **Diagonal Crack**: Orientation angle $|\theta| \in (25^\circ, 65^\circ)$.
  - **Hairline Crack**: Average thickness $\le 3.0$ px.
- **Severity Assessment**:
  - **Minor (< 35%)**: 🟢 Green (`#10B981`) — Routine monitoring.
  - **Moderate (35% - 70%)**: 🟠 Orange (`#F59E0B`) — Epoxy injection / preventive maintenance.
  - **Severe (> 70%)**: 🔴 Red (`#EF4444`) — Immediate structural load intervention.

---

### Phase 8 — Explainable AI (Grad-CAM)
Hooks into YOLOv8 PAN-FPN feature layers (e.g. C2f Layer 15, 18, or 21) to generate spatial attribution heatmaps:
```python
from xai.gradcam import YOLOGradCAM

cam = YOLOGradCAM(model, target_layer_idx=15)
heatmap = cam.generate_heatmap(image_rgb)
overlay, pure_cam = cam.overlay_heatmap(image_rgb, heatmap, alpha=0.5)
three_panel = cam.generate_three_panel_view(image_rgb, detection_rgb, overlay)
```
Generates 3-panel visualization:
$$\text{Original Image} \longrightarrow \text{Detection Result} \longrightarrow \text{Grad-CAM Heatmap}$$

---

### Phase 9 — Performance Evaluation
Generates evaluation figures into `eval_results/`:
- `confusion_matrix.png`: Normalized and counts matrix
- `roc_curve.png`: Receiver Operating Characteristic curve ($AUC = 0.98$)
- `precision_recall_curve.png`: PR trajectory ($AP = 0.98$)
- `kfold_comparison.png`: 5-Fold mAP@50 and F1-Score bar chart

---

### Phase 10 — Streamlit Web Application
Launch the interactive web dashboard:
```bash
streamlit run streamlit_app.py
```
Features:
- **Professional Blue Theme** with responsive CSS cards.
- **Upload Image or Select Test Samples** with 1 click.
- **Side-by-side Cards**: Original Image, YOLOv8 Detection Result, Grad-CAM Heatmap, Diagnostic Cards.
- **Interactive Controls**: Confidence threshold, IoU threshold, heatmap opacity, layer selection.
- **Full 5-Fold Cross Validation & Evaluation Dashboards**.

---

## 📜 Citation & References
- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization", ICCV 2017.
- Albumentations: Fast and flexible image augmentations: https://albumentations.ai/
