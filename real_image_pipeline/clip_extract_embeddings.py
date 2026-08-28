"""
clip_extract_embeddings.py
==========================
Extracts visual feature representations (512-D normalized embeddings)
from transparent garment crops using OpenAI CLIP (ViT-B/32).

Saves .npy files alongside the garment PNGs, preserving the output contract:
  outputs/<image_id>/{category}_{index}.png -> outputs/<image_id>/{category}_{index}.npy

Features:
  - Transparent RGBA -> Neutral White Alpha Composite (eliminates dark boundary artifacts).
  - L2-normalized 512-D embedding output.
  - Supports single image folder, single image path, or full outputs/ tree scan.
  - CPU/GPU execution support.
"""

import os
import sys
import argparse
from typing import Optional, Union, List
import numpy as np
import torch
from PIL import Image
import clip

# Configuration
DEFAULT_CLIP_MODEL = "ViT-B/32"
DEFAULT_OUTPUTS_DIR = "outputs"

_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_DEVICE = None


def get_clip_model(
    model_name: str = DEFAULT_CLIP_MODEL,
    device: Optional[str] = None
):
    """Loads and caches OpenAI CLIP model."""
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE

    if device is None:
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        target_device = device

    if _CLIP_MODEL is None or _CLIP_DEVICE != target_device:
        print(f"[INFO] Loading CLIP ({model_name}) on {target_device}...")
        _CLIP_MODEL, _CLIP_PREPROCESS = clip.load(model_name, device=target_device)
        _CLIP_MODEL.eval()
        _CLIP_DEVICE = target_device
        print("[INFO] CLIP model ready.")

    return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE


def embed_image(
    image_input: Union[str, Image.Image],
    device: Optional[str] = None
) -> np.ndarray:
    """
    Computes a 512-D L2-normalized CLIP embedding for a garment image.

    Garment crops saved as transparent PNGs (RGBA) are composited onto a
    neutral white background so transparency does not become black artifacts.

    Args:
        image_input: Filepath string or PIL Image object.
        device: 'cuda' or 'cpu'.

    Returns:
        np.ndarray of shape (512,) with float32 dtype and unit norm.
    """
    model, preprocess, dev = get_clip_model(device=device)

    if isinstance(image_input, str):
        img_pil = Image.open(image_input)
    else:
        img_pil = image_input

    # Neutral background composite for RGBA
    if img_pil.mode == "RGBA":
        white_bg = Image.new("RGBA", img_pil.size, (255, 255, 255, 255))
        img_rgb = Image.alpha_composite(white_bg, img_pil).convert("RGB")
    else:
        img_rgb = img_pil.convert("RGB")

    tensor = preprocess(img_rgb).unsqueeze(0).to(dev)

    with torch.no_grad():
        embedding = model.encode_image(tensor)
        # Normalize to unit sphere
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    return embedding.cpu().numpy()[0]


def extract_folder_embeddings(
    image_folder: str,
    force_recompute: bool = False,
    device: Optional[str] = None
) -> List[str]:
    """
    Extracts embeddings for all garment crops (*.png) in a specific folder.

    Args:
        image_folder: Path to outputs/<image_id>/ directory.
        force_recompute: If True, overwrites existing .npy files.
        device: 'cuda' or 'cpu'.

    Returns:
        List of generated .npy file paths.
    """
    if not os.path.isdir(image_folder):
        print(f"[WARN] Folder not found: {image_folder}")
        return []

    saved_embs = []
    png_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])

    for file in png_files:
        img_path = os.path.join(image_folder, file)
        emb_path = os.path.splitext(img_path)[0] + ".npy"

        if os.path.exists(emb_path) and not force_recompute:
            continue

        print(f"  [EMBED] Generating CLIP embedding for {file}...")
        emb = embed_image(img_path, device=device)
        np.save(emb_path, emb)
        saved_embs.append(emb_path)

    return saved_embs


def extract_all_outputs(
    outputs_root: str = DEFAULT_OUTPUTS_DIR,
    force_recompute: bool = False,
    device: Optional[str] = None
):
    """Walks through all image folders in outputs_root and extracts embeddings."""
    if not os.path.exists(outputs_root):
        print(f"[WARN] Outputs directory '{outputs_root}' does not exist.")
        return

    print(f"[INFO] Scanning '{outputs_root}' for garment crops...")
    total_embedded = 0

    for item in os.listdir(outputs_root):
        folder_path = os.path.join(outputs_root, item)
        if os.path.isdir(folder_path):
            embs = extract_folder_embeddings(folder_path, force_recompute=force_recompute, device=device)
            total_embedded += len(embs)

    print(f"[DONE] Extracted {total_embedded} new embeddings across outputs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CLIP embeddings for garment crops")
    parser.add_argument("--target_dir", type=str, default=None, help="Target image folder (e.g. outputs/010931)")
    parser.add_argument("--outputs_dir", type=str, default=DEFAULT_OUTPUTS_DIR, help="Root outputs directory")
    parser.add_argument("--device", type=str, default=None, help="'cpu' or 'cuda'")
    parser.add_argument("--force", action="store_true", help="Force recomputation of existing embeddings")

    args = parser.parse_args()

    if args.target_dir:
        extract_folder_embeddings(args.target_dir, force_recompute=args.force, device=args.device)
    else:
        extract_all_outputs(args.outputs_dir, force_recompute=args.force, device=args.device)
