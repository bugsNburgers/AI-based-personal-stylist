# 👗 Real-Image Understanding Pipeline (YOLOS → Transparent Crop → CLIP)

A production-ready, autonomous real-photo garment understanding and visual representation pipeline for the **AI-Powered Personal Stylist**.

This replaces the previous ground-truth DeepFashion2 annotation parser (`deepfashion2_parser.py`) with an autonomous detection pipeline using **YOLOS-Fashionpedia** and **OpenAI CLIP**, preserving fine-grained specific garment identities (e.g. `jacket`, `blouse`, `sweater`, `skirt`, `pants`, `shorts`, `dress`, `shoe`, `bag`) while maintaining 100% downstream compatibility with the `outputs/<image_id>/` contract.

---

## 🚀 Pipeline Workflow

```
real_image.jpg  (any real outfit photo)
      ↓
YOLOS-Fashionpedia Detection (valentinafeve/yolos-fashionpedia)
      ↓
Specific Garment Classification (jacket, blouse, sweater, cardigan, pants, shorts, skirt, dress, shoe, bag, etc.)
      ↓
NMS Deduplication (Filters redundant overlapping detections)
      ↓
Garment Cropping & Transparent Alpha Isolation (removes background bias)
      ↓
Outputs Contract: outputs/<image_id>/
      ├── jacket_0.png       (Transparent RGBA crop)
      ├── jacket_0.npy       (512-D normalized CLIP embedding)
      ├── skirt_0.png
      ├── skirt_0.npy
      ├── metadata.json      (BBoxes, specific category, broad category, raw label, confidence)
      └── <image_id>_vis.jpg (Annotated visual inspection overlay)
      ↓
Ready for GNN Outfit Compatibility & Recommendation
```

---

## 📂 File Overview

- `yolo_detect_and_crop.py`: Specific garment detection, NMS filtering, transparent RGBA crop extraction, metadata generation, and visual inspection overlay.
- `category_mapping.py`: Direct mapping of specific garment types (`jacket`, `blouse`, `sweater`, `cardigan`, `coat`, `vest`, `pants`, `shorts`, `skirt`, `dress`, `shoe`, `bag`, `hat`, etc.) while pruning part-level details (zippers, buckles, buttons).
- `bg_removal.py`: Transparent garment foreground isolation using `rembg` (U2Net) with OpenCV GrabCut fallback.
- `clip_extract_embeddings.py`: OpenAI CLIP (ViT-B/32) feature extraction with neutral white composite for transparent crops.
- `run_pipeline.py`: Unified CLI runner for single image or batch processing.
- `test_pipeline.py`: Automated unit test suite.

---

## ⚡ Quickstart

### 1. Process a Single Real Photo
```bash
python real_image_pipeline/run_pipeline.py --image data/input_images/010932.jpg
```

### 2. Batch Process a Folder of Real Images
```bash
python real_image_pipeline/run_pipeline.py --input_dir data/input_images/ --batch_limit 5
```

### 3. Custom Output Directory and Confidence Threshold
```bash
python real_image_pipeline/run_pipeline.py --image path/to/photo.jpg --output_dir outputs --threshold 0.30
```

### 4. Raw Bounding-Box Mode (Disable Background Removal)
```bash
python real_image_pipeline/run_pipeline.py --image path/to/photo.jpg --no_bg_removal
```

---

## 📊 Output Schema

Each processed image generates a directory `outputs/<image_id>/`:

```
outputs/
 └── 010932/
      ├── dress_0.png        # Transparent garment crop (RGBA)
      ├── dress_0.npy        # 512-D L2-normalized CLIP embedding vector
      ├── skirt_0.png        # Transparent garment crop (RGBA)
      ├── skirt_0.npy        # 512-D L2-normalized CLIP embedding vector
      ├── t_shirt_0.png      # Transparent garment crop (RGBA)
      ├── t_shirt_0.npy      # 512-D L2-normalized CLIP embedding vector
      ├── metadata.json      # Structured metadata with specific & broad classes
      └── 010932_vis.jpg     # Annotated detection overlay
```

### `metadata.json` Format:
```json
[
  {
    "file": "dress_0.png",
    "category": "dress",
    "specific_category": "dress",
    "broad_category": "dress",
    "raw_category": "dress",
    "confidence": 0.881,
    "bbox": [2, 14, 634, 956],
    "crop_path": "outputs/010932/dress_0.png"
  },
  {
    "file": "skirt_0.png",
    "category": "skirt",
    "specific_category": "skirt",
    "broad_category": "bottom",
    "raw_category": "skirt",
    "confidence": 0.8501,
    "bbox": [0, 475, 585, 960],
    "crop_path": "outputs/010932/skirt_0.png"
  }
]
```
