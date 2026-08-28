"""
bg_removal.py
=============
Provides high-resolution garment foreground isolation / background removal.
Generates crisp transparent RGBA PNGs (alpha channel) to remove background noise
without blurriness, feathering, or loss of garment texture.

Primary engine: `rembg` with full high-resolution `u2net` architecture.
Fallback engine: OpenCV GrabCut if rembg is unavailable.
"""

import numpy as np
from PIL import Image
import cv2
from typing import Optional

_REMBG_SESSION = None
_REMBG_AVAILABLE = None


def is_rembg_available() -> bool:
    """Checks if rembg library is available in the current environment."""
    global _REMBG_AVAILABLE
    if _REMBG_AVAILABLE is None:
        try:
            import rembg
            _REMBG_AVAILABLE = True
        except ImportError:
            _REMBG_AVAILABLE = False
    return _REMBG_AVAILABLE


def get_rembg_session(model_name: str = "u2net"):
    """Initializes and caches the high-resolution rembg session."""
    global _REMBG_SESSION
    if _REMBG_SESSION is None and is_rembg_available():
        import rembg
        try:
            # Full high-res U2Net model for sharp, crisp edges without blur
            _REMBG_SESSION = rembg.new_session(model_name=model_name)
        except Exception:
            _REMBG_SESSION = rembg.new_session(model_name="u2net")
    return _REMBG_SESSION


def remove_full_image_background(
    image: Image.Image,
    model_name: str = "u2net"
) -> Image.Image:
    """
    Removes background from the full image using full-context U2Net.
    Processing full images provides complete human anatomy context,
    preventing edge blurriness and false transparency on garment folds.
    """
    if is_rembg_available():
        try:
            import rembg
            session = get_rembg_session(model_name=model_name)
            # Use alpha matting post-processing for crisp edges
            return rembg.remove(
                image.convert("RGB"),
                session=session,
                alpha_matting=False,
                post_process_mask=True
            )
        except Exception as e:
            print(f"[WARN] Full-image rembg failed ({e}), falling back to GrabCut...")
            return remove_background_grabcut(image)
    else:
        return remove_background_grabcut(image)


def remove_background_grabcut(image: Image.Image) -> Image.Image:
    """
    Fallback background estimation using OpenCV GrabCut algorithm.
    Returns a transparent RGBA PIL Image.
    """
    img_np = np.array(image.convert("RGB"))
    h, w = img_np.shape[:2]

    if h < 10 or w < 10:
        return image.convert("RGBA")

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    margin_x = max(1, int(w * 0.05))
    margin_y = max(1, int(h * 0.05))
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    try:
        cv2.grabCut(img_np, mask, rect, bgd_model, fgd_model, iterCount=3, mode=cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
    except Exception:
        fg_mask = np.ones((h, w), dtype=np.uint8) * 255

    rgba = cv2.cvtColor(img_np, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = fg_mask
    return Image.fromarray(rgba)


def isolate_garment_background(
    crop_image: Image.Image,
    enable_bg_removal: bool = True
) -> Image.Image:
    """
    Isolates single garment crop foreground.
    """
    if not enable_bg_removal:
        return crop_image.convert("RGBA")

    if is_rembg_available():
        try:
            import rembg
            session = get_rembg_session()
            return rembg.remove(crop_image.convert("RGB"), session=session, post_process_mask=True)
        except Exception:
            return remove_background_grabcut(crop_image)
    else:
        return remove_background_grabcut(crop_image)
