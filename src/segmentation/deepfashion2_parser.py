import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm

# Category map
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

# Colors (tab20 gives 20 distinct colors)
COLORS = (cm.tab20(np.linspace(0, 1, 20))[:, :3] * 255).astype(int)

# DeepFashion2 skeleton pairs (official)
SKELETON = [
    (1, 2), (2, 3), (3, 4), (4, 5),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (12, 13), (13, 14), (14, 15)
]


def draw_segmentation(img, seg, color):
    pts = np.array(seg).reshape(-1, 2).astype(np.int32)

    overlay = img.copy()
    # color already is tuple(int,int,int)
    cv2.fillPoly(overlay, [pts], color)
    img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    cv2.polylines(img, [pts], True, color, 2)

    return img


def draw_keypoints(img, kpts, color):
    for i in range(0, len(kpts), 3):
        x, y, v = kpts[i:i+3]
        if v > 0:
            cv2.circle(img, (int(x), int(y)), 3, color, -1)

    for a, b in SKELETON:
        xa, ya, va = kpts[(a - 1) * 3: (a - 1) * 3 + 3]
        xb, yb, vb = kpts[(b - 1) * 3: (b - 1) * 3 + 3]
        if va > 0 and vb > 0:
            cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), color, 2)

    return img


def visualize(image_id, base_path="C:/Users/Suprateek Yawagal/Downloads/Capstone/Train/train", save=True):
    img_path = f"{base_path}/image/{image_id}.jpg"
    ann_path = f"{base_path}/annos/{image_id}.json"

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with open(ann_path, "r") as f:
        anno = json.load(f)

    instance_index = 0

    for key in anno.keys():
        if "item" not in key:
            continue

        item = anno[key]

        # FIX: Proper color conversion
        raw_color = COLORS[instance_index % len(COLORS)]
        color = tuple(map(int, raw_color.tolist()))

        # =========================
        # 1. BOUNDING BOX
        # =========================
        bbox = item.get("bounding_box", None)
        if bbox is not None:
            x1, y1, x2, y2 = bbox

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

            label = item.get(
                "category_name",
                CATEGORY_MAP.get(item.get("category_id"), "unknown")
            )

            cv2.putText(
                img, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

        # =========================
        # 2. SEGMENTATION
        # =========================
        seg = item.get("segmentation", None)
        if seg and len(seg) > 0:
            try:
                img = draw_segmentation(img, seg[0], color)
            except Exception as e:
                print("Segmentation error:", e)

        # =========================
        # 3. LANDMARKS / KEYPOINTS
        # =========================
        landmarks = item.get("landmarks", None)
        if landmarks and len(landmarks) > 0:
            try:
                img = draw_keypoints(img, landmarks, color)
            except Exception as e:
                print("Landmark error:", e)

        instance_index += 1

    # save
    os.makedirs("output", exist_ok=True)
    out_path = f"output/{image_id}_vis.jpg"

    if save:
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Saved → {out_path}")

    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.axis("off")
    plt.show()
