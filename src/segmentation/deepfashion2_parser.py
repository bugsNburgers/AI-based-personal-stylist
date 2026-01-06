import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm

# =========================
# CATEGORY MAP
# =========================
CATEGORY_MAP = {
    1: "short sleeve top",
    2: "long sleeve top",
    3: "short sleeve outwear",
    4: "long sleeve outwear",
    5: "vest",
    6: "sling",
    7: "shorts",
    8: "trousers",
    9: "skirt",
    10: "short sleeve dress",
    11: "long sleeve dress",
    12: "vest dress",
    13: "sling dress"
}

# =========================
# NORMALIZED CATEGORY
# =========================
def normalize_category(name):
    name = name.lower()
    if "dress" in name:
        return "dress"
    if "trouser" in name or "shorts" in name or "skirt" in name:
        return "bottom"
    if "outwear" in name:
        return "outerwear"
    return "top"

# =========================
# COLORS FOR VIS
# =========================
COLORS = (cm.tab20(np.linspace(0, 1, 20))[:, :3] * 255).astype(int)

# =========================
# POLYGON → MASK
# =========================
def polygon_to_mask(seg, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(seg).reshape(-1, 2).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask

# =========================
# MASK + BBOX → RGBA
# =========================
def extract_garment(img, mask, bbox):
    x1, y1, x2, y2 = bbox
    crop_img = img[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]

    alpha = np.where(crop_mask > 0, 255, 0).astype(np.uint8)
    rgba = cv2.cvtColor(crop_img, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = alpha
    return rgba

# =========================
# MAIN FUNCTION (ENTRYPOINT)
# =========================
def visualize(
    image_id,
    base_path="C:/Users/Suprateek Yawagal/Downloads/Capstone/Train/train",
    save=True
):
    img_path = f"{base_path}/image/{image_id}.jpg"
    ann_path = f"{base_path}/annos/{image_id}.json"

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    with open(ann_path, "r") as f:
        anno = json.load(f)

    os.makedirs(f"outputs/{image_id}", exist_ok=True)

    vis_img = img.copy()
    metadata = []
    instance_index = 0

    for key in anno:
        if "item" not in key:
            continue

        item = anno[key]
        bbox = item.get("bounding_box")
        seg = item.get("segmentation")

        if bbox is None or seg is None:
            continue

        raw_category = item.get(
            "category_name",
            CATEGORY_MAP.get(item.get("category_id"), "unknown")
        )
        norm_category = normalize_category(raw_category)

        color = tuple(map(int, COLORS[instance_index % len(COLORS)]))

        # ---- VISUALIZATION ----
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis_img,
            raw_category,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        overlay = vis_img.copy()
        pts = np.array(seg[0]).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(overlay, [pts], color)
        vis_img = cv2.addWeighted(overlay, 0.35, vis_img, 0.65, 0)

        # ---- EXTRACTION ----
        mask = polygon_to_mask(seg[0], h, w)
        garment_rgba = extract_garment(img, mask, bbox)

        out_name = f"{norm_category}_{instance_index}.png"
        out_path = f"outputs/{image_id}/{out_name}"

        cv2.imwrite(out_path, cv2.cvtColor(garment_rgba, cv2.COLOR_RGBA2BGRA))

        metadata.append({
            "file": out_name,
            "raw_category": raw_category,
            "normalized_category": norm_category,
            "bbox": bbox
        })

        instance_index += 1

    with open(f"outputs/{image_id}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if save:
        cv2.imwrite(
            f"outputs/{image_id}_vis.jpg",
            cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        )

    plt.figure(figsize=(12, 12))
    plt.imshow(vis_img)
    plt.axis("off")
    plt.show()
