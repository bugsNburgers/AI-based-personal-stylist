"""
category_mapping.py
===================
Maps fine-grained Fashionpedia labels (from YOLOS-Fashionpedia) to 
specific, clean clothing categories (e.g. 'jacket', 'blouse', 'sweater', 'pants', 'skirt', 'dress'):
  - Specific garment types are preserved (no over-generalization).
  - Part-level labels (e.g. collar, sleeve, zipper, pocket) and fine embellishments
    are mapped to None (discarded) so that downstream stages only receive whole garments.
"""

from typing import Optional, Dict, Set

# Specific clean garment mapping (Default Mode)
FASHIONPEDIA_TO_SPECIFIC_CATEGORY: Dict[str, Optional[str]] = {
    # ── Tops & Shirts ──
    "shirt, blouse": "shirt_blouse",
    "top, t-shirt, sweatshirt": "t_shirt",
    "sweater": "sweater",
    "cardigan": "cardigan",

    # ── Outerwear ──
    "jacket": "jacket",
    "vest": "vest",
    "coat": "coat",
    "cape": "cape",

    # ── Bottoms ──
    "pants": "pants",
    "shorts": "shorts",
    "skirt": "skirt",
    "tights, stockings": "stockings",

    # ── Full-body / Dresses ──
    "dress": "dress",
    "jumpsuit": "jumpsuit",

    # ── Footwear ──
    "shoe": "shoe",

    # ── Bags ──
    "bag, wallet": "bag",

    # ── Accessories ──
    "glasses": "glasses",
    "hat": "hat",
    "headband, head covering, hair accessory": "headband",
    "tie": "tie",
    "glove": "glove",
    "watch": "watch",
    "belt": "belt",
    "leg warmer": "leg_warmer",
    "sock": "sock",
    "scarf": "scarf",
    "umbrella": "umbrella",
    "hood": "hood",

    # ── Part-level details & hardware (Discarded) ──
    "collar": None,
    "lapel": None,
    "epaulette": None,
    "sleeve": None,
    "pocket": None,
    "neckline": None,
    "buckle": None,
    "zipper": None,
    "applique": None,
    "bead": None,
    "bow": None,
    "flower": None,
    "fringe": None,
    "ribbon": None,
    "rivet": None,
    "ruffle": None,
    "sequin": None,
    "tassel": None,
}

# Broad grouped mapping (Optional Mode)
FASHIONPEDIA_TO_BROAD_CATEGORY: Dict[str, Optional[str]] = {
    "shirt, blouse": "top",
    "top, t-shirt, sweatshirt": "top",
    "sweater": "top",
    "cardigan": "top",
    "jacket": "outerwear",
    "vest": "outerwear",
    "coat": "outerwear",
    "cape": "outerwear",
    "pants": "bottom",
    "shorts": "bottom",
    "skirt": "bottom",
    "tights, stockings": "bottom",
    "dress": "dress",
    "jumpsuit": "dress",
    "shoe": "shoe",
    "bag, wallet": "bag",
    "glasses": "accessory",
    "hat": "accessory",
    "headband, head covering, hair accessory": "accessory",
    "tie": "accessory",
    "glove": "accessory",
    "watch": "accessory",
    "belt": "accessory",
    "leg warmer": "accessory",
    "sock": "accessory",
    "scarf": "accessory",
    "umbrella": "accessory",
    "hood": "accessory",
}


def map_category(
    fashionpedia_label: str,
    specific: bool = True,
    allowed_categories: Optional[Set[str]] = None
) -> Optional[str]:
    """
    Maps a Fashionpedia label to a clean clothing category.

    Args:
        fashionpedia_label: Raw string label from YOLOS model.
        specific: If True (default), returns specific garment label (e.g. 'jacket', 'sweater', 'skirt', 'pants').
                  If False, returns broad category (e.g. 'top', 'bottom', 'outerwear').
        allowed_categories: Optional set of allowed categories to filter against.

    Returns:
        Clean category string or None if discarded.
    """
    normalized_label = fashionpedia_label.strip().lower()
    table = FASHIONPEDIA_TO_SPECIFIC_CATEGORY if specific else FASHIONPEDIA_TO_BROAD_CATEGORY

    # Exact match first
    category = table.get(normalized_label)

    # Substring fallback for variations
    if category is None and normalized_label:
        for k, v in table.items():
            if v is not None and (k in normalized_label or normalized_label in k):
                category = v
                break

    if category is None:
        return None

    if allowed_categories is not None and category not in allowed_categories:
        return None

    return category
