"""
yolo_detect_and_crop.py
=======================
Autonomous real-photo garment detection and extraction stage using
YOLOS-Fashionpedia (valentinafeve/yolos-fashionpedia).

Features:
  1. Real-image input (no ground-truth annotations needed).
  2. Multi-garment localization with specific clothing classifications (jacket, blouse, pants, skirt, etc.).
  3. Preserves fine-grained garment identity without over-generalization.
  4. IoU-based Non-Maximum Suppression (NMS) deduplication.
  5. Transparent RGBA garment isolation (background removal) to avoid background bias.
  6. Outputs contract matching outputs/<image_id>/ for downstream compatibility.
  7. Annotated visual overlay image (<image_id>_vis.jpg) for explainability.
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import YolosImageProcessor, YolosForObjectDetection

# Local imports
try:
    from category_mapping import map_category
    from bg_removal import remove_full_image_background, isolate_garment_background, is_rembg_available
except ImportError:
    from real_image_pipeline.category_mapping import map_category
    from real_image_pipeline.bg_removal import remove_full_image_background, isolate_garment_background, is_rembg_available

# Model configuration
DEFAULT_MODEL_NAME = "valentinafeve/yolos-fashionpedia"
DEFAULT_CONF_THRESHOLD = 0.25

# Global cached detector
_PROCESSOR = None
_MODEL = None
_DEVICE = None


def get_detector(
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None
) -> Tuple[YolosImageProcessor, YolosForObjectDetection, str]:
    """Loads and caches the YOLOS-Fashionpedia processor and model."""
    global _PROCESSOR, _MODEL, _DEVICE

    if device is None:
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        target_device = device

    if _MODEL is None or _DEVICE != target_device:
        print(f"[INFO] Loading YOLOS-Fashionpedia ({model_name}) on {target_device}...")
        _PROCESSOR = YolosImageProcessor.from_pretrained(model_name)
        _MODEL = YolosForObjectDetection.from_pretrained(model_name).to(target_device)
        _MODEL.eval()
        _DEVICE = target_device
        print("[INFO] YOLOS-Fashionpedia ready.")

    return _PROCESSOR, _MODEL, _DEVICE


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """Calculates Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def apply_nms(
    detections: List[Dict],
    iou_threshold: float = 0.65
) -> List[Dict]:
    """Performs Non-Maximum Suppression to filter duplicate overlapping boxes of same category."""
    if not detections:
        return []

    # Sort by confidence descending
    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept = []

    for det in sorted_dets:
        discard = False
        for k in kept:
            # If same category and high overlap, suppress lower confidence detection
            if det["category"] == k["category"]:
                iou = calculate_iou(det["bbox"], k["bbox"])
                if iou > iou_threshold:
                    discard = True
                    break
        if not discard:
            kept.append(det)

    return kept


def detect_garments(
    image_path: str,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    specific: bool = True,
    allowed_categories: Optional[Set[str]] = None,
    device: Optional[str] = None
) -> Tuple[Image.Image, List[Dict]]:
    """
    Detects specific garments in an input image using YOLOS-Fashionpedia.

    Args:
        image_path: Path to the real input image.
        conf_threshold: Minimum confidence score to accept.
        specific: If True (default), outputs specific garment names (jacket, blouse, pants, etc.).
        allowed_categories: Set of categories to keep.
        device: 'cuda' or 'cpu'.

    Returns:
        (PIL.Image, list of detection dicts)
    """
    processor, model, dev = get_detector(device=device)

    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[orig_h, orig_w]], device=dev)
    results = processor.post_process_object_detection(
        outputs, threshold=conf_threshold, target_sizes=target_sizes
    )[0]

    detections = []
    for score, label_idx, box in zip(results["scores"], results["labels"], results["boxes"]):
        raw_label = model.config.id2label[label_idx.item()]
        spec_cat = map_category(raw_label, specific=True, allowed_categories=allowed_categories)
        broad_cat = map_category(raw_label, specific=False)

        if spec_cat is None:
            continue

        category = spec_cat if specific else broad_cat

        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        # Clamp to image bounds
        x1 = max(0, min(orig_w - 1, x1))
        y1 = max(0, min(orig_h - 1, y1))
        x2 = max(x1 + 1, min(orig_w, x2))
        y2 = max(y1 + 1, min(orig_h, y2))

        # Filter out tiny degenerate boxes
        if (x2 - x1) < 15 or (y2 - y1) < 15:
            continue

        detections.append({
            "category": category,
            "specific_category": spec_cat,
            "broad_category": broad_cat,
            "raw_category": raw_label,
            "confidence": round(float(score), 4),
            "bbox": [x1, y1, x2, y2],
        })

    # Apply NMS
    filtered_detections = apply_nms(detections, iou_threshold=0.65)
    return image, filtered_detections


def draw_visual_overlay(
    image: Image.Image,
    detections: List[Dict]
) -> Image.Image:
    """Draws visual inspection bounding boxes and specific garment labels onto the image."""
    vis_img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(vis_img)

    # Color palette for distinct garment items
    color_palette = [
        (46, 204, 113),   # Emerald Green
        (52, 152, 219),   # Peter River Blue
        (230, 126, 34),   # Carrot Orange
        (155, 89, 182),   # Amethyst Purple
        (241, 196, 15),   # Sun Yellow
        (231, 76, 60),    # Alizarin Red
        (26, 188, 156),   # Turquoise
        (243, 156, 18),   # Orange
        (211, 84, 0),     # Pumpkin
        (142, 68, 173),   # Wisteria
    ]

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        color = color_palette[i % len(color_palette)]
        cat_name = det["category"]
        label_text = f"{cat_name} ({det['confidence']:.2f})"

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label header
        text_bbox = draw.textbbox((x1, max(0, y1 - 18)), label_text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1 + 2, max(0, y1 - 18)), label_text, fill=(0, 0, 0), font=font)

    return vis_img


def process_image(
    image_path: str,
    image_id: Optional[str] = None,
    output_root: str = "outputs",
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    specific: bool = True,
    remove_bg: bool = True,
    allowed_categories: Optional[Set[str]] = None,
    save_vis: bool = True,
    device: Optional[str] = None
) -> Dict:
    """
    Complete Stage 1 pipeline: Detect specific garments -> Crop & Background Removal -> Save & Metadata.
    """
    if image_id is None:
        image_id = os.path.splitext(os.path.basename(image_path))[0]

    out_dir = os.path.join(output_root, image_id)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[PROCESS] Processing '{image_path}' (ID: {image_id})...")
    image, detections = detect_garments(
        image_path,
        conf_threshold=conf_threshold,
        specific=specific,
        allowed_categories=allowed_categories,
        device=device
    )

    garments_meta = []
    category_counters = {}

    # Full image transparent foreground isolation (sharp, contextual, non-blurry)
    if remove_bg:
        print("  [BG] Isolating sharp foreground transparency with U2Net...")
        transparent_full = remove_full_image_background(image)
    else:
        transparent_full = image.convert("RGBA")

    for det in detections:
        cat = det["category"]
        category_counters[cat] = category_counters.get(cat, 0) + 1
        idx = category_counters[cat] - 1
        filename = f"{cat}_{idx}.png"
        out_file_path = os.path.join(out_dir, filename)

        x1, y1, x2, y2 = det["bbox"]
        # Crop directly from transparent image to preserve crisp garment edges
        crop_rgba = transparent_full.crop((x1, y1, x2, y2))
        crop_rgba.save(out_file_path, format="PNG")

        meta_entry = {
            "file": filename,
            "category": cat,
            "specific_category": det["specific_category"],
            "broad_category": det["broad_category"],
            "raw_category": det["raw_category"],
            "confidence": det["confidence"],
            "bbox": det["bbox"],
            "crop_path": out_file_path
        }
        garments_meta.append(meta_entry)
        print(f"  [SAVED] {filename} (Garment: '{cat}', conf={det['confidence']:.2f})")

    # Save metadata.json matching repository contract
    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(garments_meta, f, indent=2)

    # Save visual overlay
    if save_vis:
        vis_image = draw_visual_overlay(image, detections)
        vis_path = os.path.join(out_dir, f"{image_id}_vis.jpg")
        vis_image.save(vis_path, quality=95)
        root_vis_path = os.path.join(output_root, f"{image_id}_vis.jpg")
        vis_image.save(root_vis_path, quality=95)
        print(f"  [SAVED] Visual overlay saved to {vis_path}")

    print(f"[DONE] Extracted {len(garments_meta)} specific garments for {image_id}.")
    return {
        "image_id": image_id,
        "output_dir": out_dir,
        "num_garments": len(garments_meta),
        "garments": garments_meta
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOS Real-Image Garment Detection and Extraction")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--image_id", type=str, default=None, help="Identifier for image")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONF_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--broad", action="store_true", help="Use broad categories instead of specific garment names")
    parser.add_argument("--no_bg_removal", action="store_true", help="Disable transparent background removal")
    parser.add_argument("--device", type=str, default=None, help="'cpu' or 'cuda'")

    args = parser.parse_args()
    process_image(
        image_path=args.image,
        image_id=args.image_id,
        output_root=args.output_dir,
        conf_threshold=args.threshold,
        specific=not args.broad,
        remove_bg=not args.no_bg_removal,
        device=args.device
    )
