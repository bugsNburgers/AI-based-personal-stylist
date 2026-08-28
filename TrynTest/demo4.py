# FINAL DEMO (CLEANED YOLOS + CORRECT GNN)

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import YolosImageProcessor, YolosForObjectDetection
from scipy.io import loadmat
from scipy.sparse import csr_matrix
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------------- MODEL ----------------
class WeightedGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index, edge_weights, num_nodes):
        if edge_index.shape[1] == 0:
            return F.relu(self.norm(self.linear(x)))

        src, dst = edge_index
        weighted_msgs = x[src] * edge_weights.unsqueeze(1)

        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, weighted_msgs)

        weight_sum = torch.zeros(num_nodes, device=x.device)
        weight_sum.index_add_(0, dst, edge_weights)
        weight_sum = weight_sum.clamp(min=1e-6).unsqueeze(1)

        agg = agg / weight_sum
        return F.relu(self.norm(self.linear(x + agg)))


class OutfitGNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, colour_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim + colour_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.gnn1 = WeightedGNNLayer(hidden_dim, hidden_dim)
        self.gnn2 = WeightedGNNLayer(hidden_dim, hidden_dim)

        self.item_score = nn.Linear(hidden_dim, 1)
        self.outfit_score = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, node_ids, node_feats, edge_index, edge_weights):
        N = len(node_ids)
        emb = self.embedding(node_ids)
        x = torch.cat([emb, node_feats], dim=1)
        x = self.input_proj(x)

        x = self.gnn1(x, edge_index, edge_weights, N)
        x = self.gnn2(x, edge_index, edge_weights, N)

        per_item = self.item_score(x).squeeze(-1)
        outfit_repr = x.mean(dim=0, keepdim=True)
        score = self.outfit_score(outfit_repr).squeeze()

        return score, per_item

# ---------------- DATA ----------------
def load_sparse_cco(path):
    data = loadmat(path)
    return csr_matrix((data['data'].flatten(),
                       (data['rows'].flatten().astype(int),
                        data['cols'].flatten().astype(int))))


def load_freq(data_dir):
    mat = load_sparse_cco(os.path.join(data_dir, "feat", "garflat_cco.mat"))
    freq = np.array(mat.sum(axis=0)).flatten()
    return np.argsort(freq)[::-1], mat.shape[1]

# ---------------- CATEGORY ----------------
def build_category_mapper(freq_order):
    return {
        "top": freq_order[:12],
        "outerwear": freq_order[12:18],
        "bottom": freq_order[18:28],
        "dress": freq_order[28:33],
        "footwear": freq_order[33:41],
        "accessory": freq_order[41:60]
    }

YOLOS_TO_CATEGORY = {
    "shirt": "top", "t-shirt": "top", "blouse": "top",
    "jacket": "outerwear", "coat": "outerwear",
    "pants": "bottom", "skirt": "bottom",
    "dress": "dress",
    "shoe": "footwear", "boot": "footwear",
    "bag": "accessory", "belt": "accessory"
}

KEEP_LABELS = [
    "top", "shirt", "dress", "pants", "skirt",
    "shoe", "coat", "jacket", "bag"
]

# ---------------- YOLOS ----------------
def load_detector(device):
    print(f"[Detector] Loading YOLOS-Fashionpedia on {device}...")
    proc = YolosImageProcessor.from_pretrained("valentinafeve/yolos-fashionpedia")
    model = YolosForObjectDetection.from_pretrained("valentinafeve/yolos-fashionpedia").to(device)
    model.eval()
    print("[Detector] Ready.")
    return proc, model


def detect(image_path, device, threshold=0.5):
    proc, model = load_detector(device)
    image = Image.open(image_path).convert("RGB")

    inputs = proc(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = proc.post_process_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=torch.tensor([image.size[::-1]])
    )[0]

    garments = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        name = model.config.id2label[label.item()].lower()

        if not any(k in name for k in KEEP_LABELS):
            continue

        x1, y1, x2, y2 = [int(v) for v in box.tolist()]

        garments.append({
            "label": name,
            "confidence": float(score),
            "box": (x1, y1, x2, y2)
        })

    # 🔧 remove duplicates (keep highest confidence)
    unique = {}
    for g in garments:
        key = g['label'].split(",")[0]
        if key not in unique or g['confidence'] > unique[key]['confidence']:
            unique[key] = g

    garments = list(unique.values())

    # 🔧 keep top 5 only
    garments = sorted(garments, key=lambda x: x["confidence"], reverse=True)[:5]

    return garments, image

# ---------------- GRAPH ----------------
def build_graph(node_ids):
    N = len(node_ids)
    node_feats = torch.zeros((N, 32))

    src, dst, weights = [], [], []
    for i in range(N):
        for j in range(N):
            if i != j:
                src.append(i)
                dst.append(j)
                weights.append(0.5)

    return node_feats, torch.tensor([src, dst]), torch.tensor(weights)

# ---------------- VISUAL ----------------
def visualise(image, garments, scores, total_score, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    ax = axes[0]
    ax.imshow(image)

    for i, g in enumerate(garments):
        x1, y1, x2, y2 = g["box"]

        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=3, edgecolor="limegreen", facecolor="none")
        ax.add_patch(rect)

        ax.text(x1, y1,
                f"{g['label']}\n{scores[i]:.3f}",
                color="limegreen",
                bbox=dict(facecolor="white", alpha=0.7))

    ax.set_title(f"Outfit Analysis")
    ax.axis("off")

    ax2 = axes[1]
    labels = [g['label'] for g in garments]
    ax2.barh(labels, scores)
    ax2.axvline(x=float(scores.mean()), linestyle='--', label='mean')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[Demo] Visualisation saved → {save_path}")
    plt.show()

# ---------------- MAIN ----------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print("[Step 1/4] Building garment label mapper...")
    freq_order, vocab_size = load_freq(args.data_dir)
    category_buckets = build_category_mapper(freq_order)
    print(f"[Mapper] Built category mapper (vocab_size={vocab_size})")

    print("[Step 2/4] Loading trained GNN...")
    model = OutfitGNN(vocab_size).to(device)
    model.load_state_dict(torch.load(args.gnn_model, map_location=device))
    model.eval()
    print(f"[GNN] Loaded from {args.gnn_model}")

    print(f"[Step 3/4] Detecting garments in: {args.image}")
    garments, image = detect(args.image, device)

    print(f"  Detected {len(garments)} garment(s):")

    print("[Step 4/4] Processing garments...")

    node_ids = []
    final_garments = []

    for g in garments:
        for key in YOLOS_TO_CATEGORY:
            if key in g['label']:
                cat = YOLOS_TO_CATEGORY[key]
                idx = int(random.choice(category_buckets[cat]))
                node_ids.append(idx)
                final_garments.append(g)

                print(f"  • {g['label']:25s} conf:{g['confidence']*100:.0f}% vocab_id:{idx}")
                break

    if len(node_ids) == 0:
        print("No valid garments detected")
        return

    node_ids = torch.tensor(node_ids).to(device)
    node_feats, edge_index, edge_weights = build_graph(node_ids)

    node_feats = node_feats.to(device)
    edge_index = edge_index.to(device)
    edge_weights = edge_weights.to(device)

    with torch.no_grad():
        score, per_item = model(node_ids, node_feats, edge_index, edge_weights)

    score = torch.sigmoid(score).item() * 100
    per_item = per_item.cpu().numpy()

    print("\n" + "═"*55)
    
    print("\n  Per-Garment Breakdown:")
    for i, g in enumerate(final_garments):
        print(f"    {g['label']:30s} {per_item[i]:+.4f}")
    print("═"*55 + "\n")

    os.makedirs("output", exist_ok=True)
    save_path = os.path.join("output", "outfit_analysis.png")

    visualise(image, final_garments, per_item, score, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--gnn_model", required=True)
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    main(args)