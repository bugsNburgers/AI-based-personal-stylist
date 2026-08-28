"""
demo.py — Phase A: Outfit Compatibility Demo
=============================================
Takes an outfit photo and produces:
  - Detected garments with bounding boxes
  - HSV colour analysis per garment
  - Harmoniousness Score (0–100) from trained GNN
  - Weak-link identification
  - Annotated output image

Usage:
    python demo.py --image outfit.jpg
                   --gnn_model path/to/outfit_gnn.pt
                   --data_dir  path/to/Fashion144k_v1
                   --out_dir   ./output

Requirements:
    pip install torch transformers Pillow scipy numpy matplotlib
"""

import os
import sys
import argparse
import random
import colorsys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from scipy.io import loadmat
from scipy.sparse import csr_matrix


# ──────────────────────────────────────────────────────────────
# 1. GNN MODEL  (matches outfit_gnn.pt exactly)
# ──────────────────────────────────────────────────────────────

class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm   = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index, num_nodes):
        if edge_index.shape[1] == 0:
            return F.relu(self.norm(self.linear(x)))
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        deg = torch.zeros(num_nodes, device=x.device)
        deg.index_add_(0, dst, torch.ones(len(dst), device=x.device))
        agg = agg / deg.clamp(min=1).unsqueeze(1)
        return F.relu(self.norm(self.linear(x + agg)))


class OutfitGNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding   = nn.Embedding(vocab_size, embed_dim)
        self.gnn1        = GNNLayer(embed_dim, hidden_dim)
        self.gnn2        = GNNLayer(hidden_dim, hidden_dim)
        self.item_score  = nn.Linear(hidden_dim, 1)
        self.outfit_score = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, node_ids, edge_index):
        N  = len(node_ids)
        x  = self.embedding(node_ids)
        x  = self.gnn1(x, edge_index, N)
        x  = self.gnn2(x, edge_index, N)
        per_item    = self.item_score(x).squeeze(-1)
        outfit_repr = x.mean(dim=0, keepdim=True)
        score       = self.outfit_score(outfit_repr).squeeze()
        return score, per_item


def load_gnn(model_path, vocab_size, device,
             embed_dim=64, hidden_dim=128):
    model = OutfitGNN(vocab_size, embed_dim, hidden_dim).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"[GNN] Loaded from {model_path}  "
          f"(vocab={vocab_size}, embed={embed_dim}, hidden={hidden_dim})")
    return model


def build_edge_index(n, device):
    if n <= 1:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    src = [i for i in range(n) for j in range(n) if i != j]
    dst = [j for i in range(n) for j in range(n) if i != j]
    return torch.tensor([src, dst], dtype=torch.long, device=device)


# ──────────────────────────────────────────────────────────────
# 2. LABEL MAPPER  (YOLOS label → garment vocab index)
#
# Fashion144K's garflat_cco vocabulary (1352 indices) has no
# string labels shipped with it. We build the mapper by finding
# which vocab indices are MOST FREQUENTLY used overall — these
# correspond to common garment types that YOLOS also detects.
# We assign YOLOS labels to the top-N indices by frequency,
# grouped by broad garment category.
# ──────────────────────────────────────────────────────────────

def build_label_mapper(data_dir, top_k=30):
    """
    Build a mapping: YOLOS label string → garment vocab index.

    Strategy:
      1. Load garflat_cco.mat
      2. Count how often each vocab index appears across all outfits
      3. Sort by frequency → top indices = most common garment types
      4. Assign YOLOS labels to vocab index buckets by category rank

    Returns:
        mapper     : dict {yolos_label: vocab_index}
        vocab_size : int (1352)
        freq_order : sorted vocab indices by frequency (for reference)
    """
    path = os.path.join(data_dir, "feat", "garflat_cco.mat")
    data = loadmat(path)
    mat  = csr_matrix((
        data['data'].flatten(),
        (data['rows'].flatten().astype(int),
         data['cols'].flatten().astype(int))
    ))
    vocab_size = mat.shape[1]

    # Count frequency of each vocab index across all outfits
    freq = np.array(mat.sum(axis=0)).flatten()  # (vocab_size,)
    freq_order = np.argsort(freq)[::-1]          # sorted by descending freq

    # YOLOS Fashionpedia labels grouped by rough category
    # We map each to one of the top-frequency vocab indices
    # Top indices in Fashion144K tend to be: tops, pants, shoes, dresses
    # (since Chictopia is a women's fashion blog)
    # Category groups — garments in the same group share a vocab index
    # because Fashion144K co-occurrence treats them as interchangeable
    # within the same category. This makes the GNN score reflect
    # whether the *combination of categories* is compatible,
    # which is what it actually learned.
    category_groups = {
        # Group 0: tops (most frequent in Fashion144K)
        "top, t-shirt":  int(freq_order[0]),
        "shirt, blouse": int(freq_order[0]),
        "sweater":       int(freq_order[0]),
        "hoodie":        int(freq_order[0]),
        "vest":          int(freq_order[0]),
        "cardigan":      int(freq_order[0]),
        # Group 1: outerwear
        "jacket":        int(freq_order[1]),
        "coat":          int(freq_order[1]),
        "outwear":       int(freq_order[1]),
        "cape":          int(freq_order[1]),
        # Group 2: bottoms
        "pants":         int(freq_order[2]),
        "skirt":         int(freq_order[2]),
        "shorts":        int(freq_order[2]),
        "jumpsuit":      int(freq_order[3]),
        # Group 3: dresses (own group — different co-occurrence pattern)
        "dress":         int(freq_order[4]),
        # Group 4: footwear
        "shoe":          int(freq_order[5]),
        "boot":          int(freq_order[5]),
        "sandal":        int(freq_order[5]),
        # Group 5: accessories
        "bag, wallet":   int(freq_order[6]),
        "hat":           int(freq_order[7]),
        "headband":      int(freq_order[7]),
        "belt":          int(freq_order[8]),
        "glasses":       int(freq_order[8]),
    }

    mapper = category_groups
    print(f"[Mapper] Built label mapper: {len(mapper)} YOLOS labels "
          f"→ vocab indices (vocab_size={vocab_size})")
    return mapper, vocab_size, freq_order


# ──────────────────────────────────────────────────────────────
# 3. YOLOS DETECTOR
# ──────────────────────────────────────────────────────────────

KEEP_LABELS = [
    "top, t-shirt", "shirt, blouse", "skirt", "pants", "shorts",
    "jacket", "coat", "dress", "outwear", "jumpsuit", "cape",
    "vest", "cardigan", "hoodie", "sweater", "shoe", "boot",
    "sandal", "bag, wallet", "hat", "headband", "glasses", "belt"
]

_yolos_proc  = None
_yolos_model = None
_yolos_dev   = None

def load_detector(device):
    global _yolos_proc, _yolos_model, _yolos_dev
    if _yolos_model is None or _yolos_dev != device:
        from transformers import (YolosImageProcessor,
                                  YolosForObjectDetection)
        print(f"[Detector] Loading YOLOS-Fashionpedia on {device}...")
        _yolos_proc  = YolosImageProcessor.from_pretrained(
            "valentinafeve/yolos-fashionpedia")
        _yolos_model = YolosForObjectDetection.from_pretrained(
            "valentinafeve/yolos-fashionpedia").to(device)
        _yolos_model.eval()
        _yolos_dev = device
        print("[Detector] Ready.")
    return _yolos_proc, _yolos_model


def detect_garments(image_path, device, threshold=0.3):
    proc, model = load_detector(device)
    image  = Image.open(image_path).convert("RGB")
    inputs = {k: v.to(device)
              for k, v in proc(images=image,
                               return_tensors="pt").items()}
    with torch.no_grad():
        outputs = model(**inputs)

    results = proc.post_process_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=torch.tensor([image.size[::-1]])
    )[0]

    garments, seen = [], set()
    for score, label, box in zip(results["scores"],
                                  results["labels"],
                                  results["boxes"]):
        name = model.config.id2label[label.item()]
        if not any(k in name for k in KEEP_LABELS):
            continue
        if name in seen:
            continue
        seen.add(name)

        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        x1 = max(0, x1);            y1 = max(0, y1)
        x2 = min(image.width, x2);  y2 = min(image.height, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        garments.append({
            "label":      name,
            "confidence": float(score),
            "box":        (x1, y1, x2, y2),
            "crop":       image.crop((x1, y1, x2, y2)),
        })

    return garments, image


# ──────────────────────────────────────────────────────────────
# 4. HSV COLOUR ANALYSIS
# ──────────────────────────────────────────────────────────────

def analyse_hsv(crop):
    """
    Compute dominant HSV values for a garment crop.
    Uses the most-saturated pixels (ignores washed-out background)
    to avoid the crop background bleeding into the colour reading.

    Returns dict with:
        hue        : 0-360 degrees
        saturation : 0-100 %
        brightness : 0-100 %
        color_name : human-readable colour label
    """
    img_rgb = np.array(crop.resize((64, 64)).convert("RGB")) / 255.0
    h_vals, s_vals, v_vals = [], [], []
    for row in img_rgb:
        for px in row:
            hh, ss, vv = colorsys.rgb_to_hsv(*px)
            h_vals.append(hh * 360)
            s_vals.append(ss * 100)
            v_vals.append(vv * 100)

    h_arr = np.array(h_vals)
    s_arr = np.array(s_vals)
    v_arr = np.array(v_vals)

    # Use pixels with above-median saturation as the "garment" pixels
    # This filters out desaturated background/shadow regions
    sat_threshold = max(np.median(s_arr), 10.0)
    mask = s_arr >= sat_threshold

    if mask.sum() < 10:
        # Fallback to full mean if not enough saturated pixels
        mean_h = float(np.mean(h_arr))
        mean_s = float(np.mean(s_arr))
        mean_v = float(np.mean(v_arr))
    else:
        mean_h = float(np.mean(h_arr[mask]))
        mean_s = float(np.mean(s_arr[mask]))
        mean_v = float(np.mean(v_arr[mask]))

    color_name = _hue_to_name(mean_h, mean_s, mean_v)
    return {
        "hue":        round(mean_h, 1),
        "saturation": round(mean_s, 1),
        "brightness": round(mean_v, 1),
        "color_name": color_name,
    }


def _hue_to_name(h, s, v):
    """Map HSV to a readable colour name."""
    if v < 15:
        return "black"
    if v > 85 and s < 15:
        return "white"
    if s < 15:
        return "grey"
    if h < 15 or h >= 345:
        return "red"
    if h < 45:
        return "orange"
    if h < 70:
        return "yellow"
    if h < 150:
        return "green"
    if h < 195:
        return "cyan"
    if h < 255:
        return "blue"
    if h < 285:
        return "purple"
    if h < 345:
        return "pink"
    return "unknown"


def compute_hue_harmony(garments):
    """
    Compute a hue harmony penalty (0 = clashing, 1 = harmonious).
    Fashion theory: analogous colours (close hues) or
    complementary (180° apart) are harmonious.
    """
    hues = [g["hsv"]["hue"] for g in garments if "hsv" in g]
    if len(hues) < 2:
        return 1.0

    penalties = []
    for i in range(len(hues)):
        for j in range(i + 1, len(hues)):
            diff = abs(hues[i] - hues[j])
            diff = min(diff, 360 - diff)   # circular distance

            # Analogous: diff < 45° → good
            # Complementary: diff ≈ 180° → good
            # Triadic: diff ≈ 120° → decent
            # Everything else → penalty
            if diff < 45:
                penalties.append(0.0)    # analogous ✓
            elif abs(diff - 180) < 30:
                penalties.append(0.1)    # complementary ✓
            elif abs(diff - 120) < 20:
                penalties.append(0.2)    # triadic, ok
            else:
                penalties.append(0.5)    # clashing ✗

    return float(1.0 - np.mean(penalties))


# ──────────────────────────────────────────────────────────────
# 5. SCORING
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def score_outfit(model, node_ids, device):
    """
    Run GNN forward pass.
    Returns (compatibility_0_to_100, per_item_scores_numpy)
    """
    ids        = torch.tensor(node_ids, dtype=torch.long).to(device)
    edge_index = build_edge_index(len(node_ids), device)
    raw, per   = model(ids, edge_index)
    score_norm = float(torch.sigmoid(raw).item() * 100)
    return score_norm, per.cpu().numpy()


def combined_score(gnn_score, hue_harmony):
    """
    Harmoniousness Score = 70% GNN + 30% colour harmony.
    Both inputs already in [0, 100] / [0, 1] range.
    """
    return round(0.70 * gnn_score + 0.30 * hue_harmony * 100, 1)


# ──────────────────────────────────────────────────────────────
# 6. VISUALISATION
# ──────────────────────────────────────────────────────────────

def visualise(orig_image, garments, per_item_scores,
              weak_idx, harmoniousness_score, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # ── Left: annotated outfit photo ──
    ax = axes[0]
    ax.imshow(orig_image)
    for i, g in enumerate(garments):
        x1, y1, x2, y2 = g["box"]
        is_weak = (i == weak_idx)
        color   = "red" if is_weak else "limegreen"
        rect    = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=3, edgecolor=color, facecolor="none")
        ax.add_patch(rect)

        hsv  = g.get("hsv", {})
        tag  = (f"{g['label']}\n"
                f"{hsv.get('color_name','?')} | "
                f"score: {per_item_scores[i]:.3f}")
        if is_weak:
            tag += "\n← WEAK LINK"
        ax.text(x1, max(y1-4, 0), tag, color=color, fontsize=8,
                va="bottom",
                bbox=dict(facecolor="white", alpha=0.75,
                          edgecolor="none", pad=2))

    ax.set_title(f"Harmoniousness Score: {harmoniousness_score} / 100",
                 fontsize=13, fontweight="bold")
    ax.axis("off")

    # ── Right: per-garment breakdown bar chart ──
    ax2 = axes[1]
    labels = [g["label"].split(",")[0] for g in garments]
    colors = ["red" if i == weak_idx else "steelblue"
              for i in range(len(garments))]
    bars   = ax2.barh(labels, per_item_scores, color=colors)
    ax2.set_xlabel("Per-Garment Contribution Score")
    ax2.set_title("Garment Contribution Breakdown")
    ax2.axvline(x=np.mean(per_item_scores), color="orange",
                linestyle="--", label="mean")
    ax2.legend()

    # Add colour swatches anchored to x=0 edge
    x_max = max(per_item_scores) if max(per_item_scores) > 0 else 0.02
    swatch_w = (abs(min(per_item_scores)) + x_max) * 0.04
    for i, g in enumerate(garments):
        hsv = g.get("hsv", {})
        h   = hsv.get("hue", 0) / 360
        s   = hsv.get("saturation", 50) / 100
        v   = hsv.get("brightness", 70) / 100
        rgb = colorsys.hsv_to_rgb(h, s, v)
        ax2.barh(labels[i], swatch_w,
                 left=0.002,
                 color=rgb, height=0.4, zorder=5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[Demo] Visualisation saved → {save_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────
# 7. MAIN DEMO
# ──────────────────────────────────────────────────────────────

def run_demo(args, device):
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Step 1: Build label mapper from Fashion144K vocab ──
    print("\n[Step 1/4] Building garment label mapper...")
    mapper, vocab_size, _ = build_label_mapper(args.data_dir)

    # ── Step 2: Load trained GNN ──
    print("\n[Step 2/4] Loading trained GNN...")
    model = load_gnn(args.gnn_model, vocab_size, device)

    # ── Step 3: Detect garments with YOLOS ──
    print(f"\n[Step 3/4] Detecting garments in: {args.image}")
    garments, orig_image = detect_garments(
        args.image, device, threshold=args.confidence)

    if not garments:
        sys.exit("[Error] No garments detected. "
                 "Try --confidence 0.15 to lower the threshold.")

    print(f"\n  Detected {len(garments)} garment(s):")

    # ── Step 4: Analyse each garment ──
    print("\n[Step 4/4] Analysing garments...")
    node_ids = []
    for g in garments:
        # HSV colour analysis
        g["hsv"] = analyse_hsv(g["crop"])

        # Map YOLOS label → vocab index
        vocab_id = mapper.get(g["label"], 0)
        g["vocab_id"] = vocab_id
        node_ids.append(vocab_id)

        print(f"  • {g['label']:30s} "
              f"conf:{g['confidence']:.0%}  "
              f"colour:{g['hsv']['color_name']:8s}  "
              f"H:{g['hsv']['hue']:5.1f}°  "
              f"vocab_id:{vocab_id}")

    # ── GNN compatibility score ──
    gnn_score, per_item_scores = score_outfit(model, node_ids, device)

    # ── Colour harmony score ──
    hue_harmony = compute_hue_harmony(garments)

    # ── Combined harmoniousness score ──
    harmoniousness = combined_score(gnn_score, hue_harmony)

    # ── Weak link ──
    weak_idx  = int(np.argmin(per_item_scores))
    weak_item = garments[weak_idx]["label"]

    # ── Print results ──
    print(f"\n{'═'*55}")
    print(f"  GNN Compatibility Score : {gnn_score:.1f} / 100")
    print(f"  Colour Harmony Score    : {hue_harmony*100:.1f} / 100")
    print(f"  ─────────────────────────────────────────")
    print(f"  Harmoniousness Score    : {harmoniousness} / 100")
    print(f"\n  Per-Garment Breakdown:")
    for i, g in enumerate(garments):
        weak_marker = "  ← WEAK LINK 🔴" if i == weak_idx else ""
        print(f"    {g['label']:35s} "
              f"{per_item_scores[i]:+.4f}"
              f"  [{g['hsv']['color_name']}]{weak_marker}")
    print(f"\n  Weak Link  : {weak_item}")
    print(f"  Suggestion : Consider replacing the "
          f"'{weak_item}' to improve outfit harmony.")
    print(f"{'═'*55}\n")

    # ── Visualise ──
    save_path = os.path.join(args.out_dir, "outfit_analysis.png")
    visualise(orig_image, garments, per_item_scores,
              weak_idx, harmoniousness, save_path)

    return {
        "harmoniousness_score": harmoniousness,
        "gnn_score":            gnn_score,
        "hue_harmony":          round(hue_harmony * 100, 1),
        "weak_link":            weak_item,
        "garments":             garments,
    }


# ──────────────────────────────────────────────────────────────
# 8. ARGS
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase A Demo — Outfit Compatibility Analysis")
    parser.add_argument("--image",      required=True,
                        help="Path to outfit photo")
    parser.add_argument("--gnn_model",  required=True,
                        help="Path to outfit_gnn.pt")
    parser.add_argument("--data_dir",   required=True,
                        help="Path to Fashion144K root folder")
    parser.add_argument("--out_dir",    default="./output")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="YOLOS detection threshold (default 0.3)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    run_demo(args, device)