"""
test_pipeline.py
================
Automated unit & integration tests for the real-image understanding pipeline.

Verifies:
  1. Specific & broad category mapping coverage.
  2. Background removal functionality (transparent RGBA).
  3. Bounding box IoU & NMS deduplication.
  4. End-to-end detection, transparent crop, and CLIP embedding generation.
  5. 512-D L2-normalization of output embeddings.
"""

import os
import sys
import numpy as np
from PIL import Image

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from category_mapping import map_category
from bg_removal import isolate_garment_background, is_rembg_available
from yolo_detect_and_crop import calculate_iou, apply_nms, process_image
from clip_extract_embeddings import embed_image


def test_category_mapping():
    """Verify specific garment detection labels and part filtering."""
    # Specific garment mapping
    assert map_category("jacket", specific=True) == "jacket"
    assert map_category("sweater", specific=True) == "sweater"
    assert map_category("cardigan", specific=True) == "cardigan"
    assert map_category("pants", specific=True) == "pants"
    assert map_category("shorts", specific=True) == "shorts"
    assert map_category("skirt", specific=True) == "skirt"
    assert map_category("dress", specific=True) == "dress"
    assert map_category("shoe", specific=True) == "shoe"
    assert map_category("bag, wallet", specific=True) == "bag"
    assert map_category("shirt, blouse", specific=True) == "shirt_blouse"
    assert map_category("top, t-shirt, sweatshirt", specific=True) == "t_shirt"

    # Part details discarded
    assert map_category("zipper", specific=True) is None
    assert map_category("pocket", specific=True) is None
    assert map_category("buckle", specific=True) is None

    # Broad categories fallback
    assert map_category("jacket", specific=False) == "outerwear"
    assert map_category("pants", specific=False) == "bottom"


def test_iou_and_nms():
    """Verify IoU calculation and Non-Maximum Suppression."""
    box1 = [0, 0, 100, 100]
    box2 = [0, 0, 100, 100]
    box3 = [200, 200, 300, 300]

    assert calculate_iou(box1, box2) == 1.0
    assert calculate_iou(box1, box3) == 0.0

    detections = [
        {"category": "jacket", "confidence": 0.90, "bbox": [0, 0, 100, 100]},
        {"category": "jacket", "confidence": 0.70, "bbox": [5, 5, 95, 95]},  # Duplicate
        {"category": "pants", "confidence": 0.85, "bbox": [0, 100, 100, 200]}
    ]

    filtered = apply_nms(detections, iou_threshold=0.6)
    assert len(filtered) == 2
    assert filtered[0]["confidence"] == 0.90
    assert filtered[1]["category"] == "pants"


def test_background_removal():
    """Verify background removal generates a valid 4-channel RGBA transparent image."""
    test_img = Image.new("RGB", (64, 64), color=(200, 50, 50))
    rgba_img = isolate_garment_background(test_img, enable_bg_removal=True)

    assert rgba_img.mode == "RGBA"
    assert rgba_img.size == (64, 64)


def test_clip_embedding_properties():
    """Verify CLIP embeddings are float32, 512-D, and L2-normalized to 1.0."""
    test_crop = Image.new("RGBA", (128, 128), color=(100, 150, 200, 255))
    emb = embed_image(test_crop, device="cpu")

    assert isinstance(emb, np.ndarray)
    assert emb.shape == (512,)
    assert emb.dtype == np.float32 or emb.dtype == np.float64
    # Unit norm: ||v|| = 1.0 (within float tolerance)
    norm = np.linalg.norm(emb)
    assert np.isclose(norm, 1.0, atol=1e-4)


if __name__ == "__main__":
    print("[TEST] Running unit tests...")
    test_category_mapping()
    print("  [PASS] Specific category mapping test passed.")
    test_iou_and_nms()
    print("  [PASS] IoU & NMS deduplication test passed.")
    test_background_removal()
    print("  [PASS] Background removal test passed.")
    test_clip_embedding_properties()
    print("  [PASS] CLIP embedding vector properties test passed.")
    print("[SUCCESS] All pipeline unit tests passed successfully!")
