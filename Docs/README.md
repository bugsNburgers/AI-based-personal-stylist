<div align="center">

# 👗 AI-Powered Personal Stylist

### Visual Garment Understanding & Representation Pipeline

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org/)
[![CLIP](https://img.shields.io/badge/Model-CLIP-green.svg)](https://github.com/openai/CLIP)
[![License](https://img.shields.io/badge/Status-Private-orange.svg)]()

</div>

---

## 📋 Overview

This project is part of a **B. Tech capstone** titled: 

> **AI-Powered Personal Stylist:  Integrating Visual Compatibility Analysis with Social-Temporal Trend Prediction**

The long-term goal of the system is to analyze a real user outfit image, understand individual garments, and later evaluate outfit compatibility and trend relevance.

**Current implementation** focuses on the foundational visual understanding stage, which is the most critical and error-prone part of such systems.

AI-Powered Personal Stylist is a fashion analysis and recommendation system that evaluates a person’s outfit based on **visual compatibility and current fashion trends**.

The system analyzes the garments in an outfit, uses a **Graph Neural Network (GNN)** to understand how the garments work together, and uses **time-based Pinterest fashion data** to determine whether the styles are currently trending.

Based on the analysis, the system identifies potential weak points in the outfit and recommends suitable replacements that are both **compatible with the existing outfit and aligned with current trends**.

## Key Features

- Outfit compatibility analysis using GNN
- Visual feature extraction using CLIP
- Temporal fashion trend analysis using Pinterest
- Weak-garment identification
- Trend-aware garment recommendations
- Explainable outfit analysis

---

## 🎯 What is Implemented So Far (Current Scope)

The project currently implements a robust garment extraction and representation pipeline using the **DeepFashion2** dataset and **CLIP** embeddings.

### 1️⃣ Garment Segmentation & Extraction

- ✅ Uses DeepFashion2 annotations (bounding boxes + polygon segmentations)
- ✅ Extracts individual garments from full-body images
- ✅ Applies pixel-accurate masks
- ✅ Saves garments as transparent PNGs (RGBA) to remove background bias
- ✅ Stores metadata for each extracted garment

### 2️⃣ Visual Representation (Embeddings)

- ✅ Uses **CLIP (ViT-B/32)** image encoder
- ✅ Generates 512-dimensional normalized embeddings per garment
- ✅ Runs entirely on **CPU** (no GPU required)
- ✅ Embeddings are saved as `.npy` files alongside garment images

These representations will later be used for: 
- Visual similarity
- Compatibility analysis
- Trend-aware recommendation

---

## 📁 Project Structure

```
.
├── run_visualize.py                  # Entry point for garment extraction
├── deepfashion2_parser.py             # Core DeepFashion2 parsing + extraction logic
├── clip_extract_embeddings.py         # CLIP embedding extraction script
├── outputs/
│   └── <image_id>/
│       ├── top_0.png                 # Extracted garment (transparent PNG)
│       ├── top_0.npy                 # CLIP embedding (512-D)
│       ├── metadata.json             # Garment metadata
│       └── <image_id>_vis.jpg         # Visualization with boxes & masks
└── README.md
```

> **Note:** Dataset files (`data/`) and virtual environment (`.venv/`) are intentionally excluded from the repository due to size. 

---

## 🚀 How to Run (Current Pipeline)

### Prerequisites

- **Python 3.9+**
- **Windows OS**
- DeepFashion2 images + annotations available locally

### 1️⃣ Set up virtual environment (once)

```bash
python -m venv .venv
. venv\Scripts\activate
```

### 2️⃣ Install dependencies

```bash
pip install numpy opencv-python matplotlib pillow
pip install torch torchvision ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

### 3️⃣ Garment extraction & visualization

Edit `run_visualize.py` if needed:

```python
from deepfashion2_parser import visualize
visualize("000002")
```

Run:

```bash
python run_visualize.py
```

**This will:**
- Visualize bounding boxes & segmentation
- Extract garments as transparent PNGs
- Save metadata and outputs in `outputs/`

### 4️⃣ CLIP embedding extraction

Run:

```bash
python clip_extract_embeddings.py
```

**This will:**
- Load each extracted garment PNG
- Compute a CLIP image embedding
- Save it as `.npy` next to the image

---

## 📦 Example Output

For a single image ID:

```
outputs/
 └── 010931/
      ├── top_0.png        # Transparent garment crop
      ├── top_0.npy        # CLIP embedding (512-D)
      ├── metadata.json
      └── 010931_vis.jpg   # Visualization output
```

---

## 🎨 Design Philosophy

| Principle | Description |
|-----------|-------------|
| **Accuracy over shortcuts** | Pixel-level segmentation is used instead of bounding-box crops to avoid background leakage |
| **Modularity** | Visual understanding is isolated from downstream logic (compatibility, trends) |
| **Explainability** | Intermediate outputs (masks, crops, embeddings) are explicitly saved and inspectable |
| **Reproducibility** | Deterministic preprocessing and frozen pretrained models |

---

## 🔮 What is NOT Implemented Yet

The following components are planned for later stages:

- [ ] Outfit compatibility scoring
- [ ] Graph-based garment interaction modeling
- [ ] Trend analysis using time-aware social data
- [ ] Weak-link detection
- [ ] Recommendation generation
- [ ] Web / UI integration (optional)

> These are intentionally deferred until the core visual pipeline is complete and stable. 

---

## 🔒 Why the Repository is Private

- Dataset files are large and proprietary
- Code is under active development
- Prevents premature exposure of incomplete modules

---

## 📊 Status

**Current Stage:**

✅ Garment segmentation & extraction  
✅ Visual embeddings (CLIP)

---

<div align="center">

**Built with 💙 for Fashion AI Research**

</div>
