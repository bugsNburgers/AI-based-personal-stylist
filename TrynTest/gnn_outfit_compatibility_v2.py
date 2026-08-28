"""
gnn_outfit_compatibility_v2.py
===============================
Improved GNN for outfit compatibility — Fashion144K.

Fixes applied vs v1:
  1. Richer node features: garment embedding + colour features + fashionability
     weight per outfit (from col_cco.mat + relvotes.mat)
  2. Weighted edges: category-pair importance (top↔bottom > shoe↔hat)
  3. Harder negative sampling: corrupt with a garment from a SIMILAR
     category, not a completely random one — forces the model to learn
     fine-grained compatibility, not just "valid vs garbage"
  4. Fashionability-weighted BPR loss: outfits with higher relvotes scores
     contribute more to the loss signal
  5. Gradient accumulation in train_epoch — no more laptop crashes
  6. weights_only=True on torch.load

Usage:
    python gnn_outfit_compatibility_v2.py \
        --data_dir "C:/path/to/Fashion144k_v1" \
        --out_dir  "C:/path/to/gnn_output"

Outputs:
    outfit_gnn_v2.pt       — trained model weights
    results_v2.txt         — per-epoch metrics
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────
# 1. CATEGORY STRUCTURE
# Category pairs with high fashion importance get higher edge weights.
# Based on standard fashion rules: top↔bottom is the most critical
# pairing, outerwear↔anything matters, accessories↔everything is weak.
# ──────────────────────────────────────────────────────────────

# Broad category groups — used for edge weighting and hard negatives
CATEGORY_GROUPS = {
    "top":        [0],          # placeholder — filled from freq at runtime
    "outerwear":  [1],
    "bottom":     [2, 3],
    "dress":      [4],
    "footwear":   [5],
    "accessory":  [6, 7, 8],
}

# Edge weight matrix between category groups (symmetric)
# Higher = more important relationship for compatibility
EDGE_WEIGHT_MATRIX = {
    ("top",      "bottom"):    1.0,   # most important pairing
    ("top",      "outerwear"): 0.9,
    ("top",      "dress"):     0.3,   # unusual combination
    ("top",      "footwear"):  0.5,
    ("top",      "accessory"): 0.4,
    ("outerwear","bottom"):    0.8,
    ("outerwear","dress"):     0.7,
    ("outerwear","footwear"):  0.5,
    ("outerwear","accessory"): 0.4,
    ("bottom",   "footwear"):  0.8,
    ("bottom",   "accessory"): 0.4,
    ("dress",    "footwear"):  0.9,   # dress + shoes is a key pairing
    ("dress",    "accessory"): 0.6,
    ("footwear", "accessory"): 0.3,   # least important
}

def get_edge_weight(cat_a, cat_b):
    """Look up edge weight for two category names."""
    if cat_a == cat_b:
        return 0.5   # same-category edges (e.g. layering) get mid weight
    key = tuple(sorted([cat_a, cat_b]))
    return EDGE_WEIGHT_MATRIX.get(key, 0.3)


# ──────────────────────────────────────────────────────────────
# 2. DATA LOADING
# ──────────────────────────────────────────────────────────────

def load_sparse_cco(path):
    """Load a _cco.mat sparse file → scipy csr_matrix."""
    data = loadmat(path)
    rows = data['rows'].flatten().astype(int)
    cols = data['cols'].flatten().astype(int)
    vals = data['data'].flatten().astype(np.float32)
    return csr_matrix((vals, (rows, cols)))


def load_fashion144k(data_dir):
    """
    Load Fashion144K data including:
      - garment co-occurrence (garflat_cco.mat)
      - colour features      (col_cco.mat)
      - fashionability votes (relvotes.mat)

    Returns:
        garment_mat   : (N, 1352) sparse — garment presence per outfit
        colour_mat    : (N, 604)  sparse — colour presence per outfit
        votes         : (N,)      float  — normalised fashionability score
        train/val/test_ids: filtered index arrays
        freq_order    : garment vocab sorted by frequency (for mapper)
    """
    print("Loading Fashion144K...")

    garment_mat = load_sparse_cco(
        os.path.join(data_dir, "feat", "garflat_cco.mat"))
    colour_mat = load_sparse_cco(
        os.path.join(data_dir, "feat", "col_cco.mat"))

    # Fashionability votes: normalise to [0, 1]
    votes_data = loadmat(os.path.join(data_dir, "feat", "relvotes.mat"))
    votes_raw  = votes_data['X'].flatten().astype(np.float32)
    votes_min, votes_max = votes_raw.min(), votes_raw.max()
    votes = (votes_raw - votes_min) / (votes_max - votes_min + 1e-8)
    print(f"  Fashionability votes: min={votes_raw.min():.1f} "
          f"max={votes_raw.max():.1f} mean={votes_raw.mean():.1f}")

    # Garment frequency order (most → least common)
    freq = np.array(garment_mat.sum(axis=0)).flatten()
    freq_order = np.argsort(freq)[::-1]

    splits    = loadmat(os.path.join(data_dir, "split.mat"))
    train_ids = splits['trainids'].flatten().astype(int) + 1
    val_ids   = splits['validids'].flatten().astype(int) + 1
    test_ids  = splits['testids'].flatten().astype(int)  + 1

    N = garment_mat.shape[0]
    train_ids = train_ids[train_ids < N]
    val_ids   = val_ids[val_ids < N]
    test_ids  = test_ids[test_ids < N]

    # Keep only outfits with ≥2 garments
    items_per_outfit = np.array(garment_mat.sum(axis=1)).flatten()
    valid_mask = items_per_outfit >= 2
    train_ids = train_ids[valid_mask[train_ids]]
    val_ids   = val_ids[valid_mask[val_ids]]
    test_ids  = test_ids[valid_mask[test_ids]]

    print(f"  Garment vocab : {garment_mat.shape[1]}")
    print(f"  Colour vocab  : {colour_mat.shape[1]}")
    print(f"  Train outfits : {len(train_ids)}")
    print(f"  Val outfits   : {len(val_ids)}")
    print(f"  Test outfits  : {len(test_ids)}")

    return garment_mat, colour_mat, votes, train_ids, val_ids, test_ids, freq_order


# ──────────────────────────────────────────────────────────────
# 3. CATEGORY MAPPER
# Maps each garment vocab index to a broad category name.
# Used for edge weighting and hard negative sampling.
# ──────────────────────────────────────────────────────────────

def build_category_mapper(freq_order):
    """
    Assign each of the top-frequency vocab indices to a category.
    We use frequency rank as a proxy for garment type since
    Fashion144K has no string labels for vocab indices.

    Returns:
        vocab_to_cat: dict {vocab_idx: category_name}
        cat_to_vocab: dict {category_name: [vocab_idx, ...]}
    """
    # Category bucket sizes based on typical Fashion144K distribution
    # Top items → tops; next → outerwear; etc.
    buckets = [
        ("top",       12),   # indices 0-11  → tops/shirts
        ("outerwear",  6),   # indices 12-17 → jackets/coats
        ("bottom",    10),   # indices 18-27 → pants/skirts
        ("dress",      5),   # indices 28-32 → dresses
        ("footwear",   8),   # indices 33-40 → shoes/boots
        ("accessory", 10),   # indices 41-50 → bags/hats/belts
    ]

    vocab_to_cat = {}
    cat_to_vocab = {cat: [] for cat, _ in buckets}

    pos = 0
    for cat_name, count in buckets:
        for _ in range(count):
            if pos < len(freq_order):
                idx = int(freq_order[pos])
                vocab_to_cat[idx] = cat_name
                cat_to_vocab[cat_name].append(idx)
                pos += 1

    # All remaining vocab indices → "accessory" (catch-all)
    for i in range(pos, len(freq_order)):
        idx = int(freq_order[i])
        vocab_to_cat[idx] = "accessory"
        cat_to_vocab["accessory"].append(idx)

    return vocab_to_cat, cat_to_vocab


# ──────────────────────────────────────────────────────────────
# 4. GRAPH CONSTRUCTION
# ──────────────────────────────────────────────────────────────

def outfit_to_graph(outfit_idx, garment_mat, colour_mat,
                    vocab_to_cat, colour_dim=32):
    """
    Build a graph for one outfit with:
      - node IDs      : garment vocab indices
      - node features : colour bag-of-words for this outfit
                        (same for all nodes — outfit-level colour context)
      - edge index    : fully connected
      - edge weights  : category-pair importance scores

    Returns:
        node_ids    : (N,) LongTensor
        node_feats  : (N, colour_dim) FloatTensor — colour context
        edge_index  : (2, E) LongTensor
        edge_weights: (E,)   FloatTensor
    """
    gar_row = garment_mat.getrow(outfit_idx)
    garment_indices = gar_row.indices
    N = len(garment_indices)
    node_ids = torch.tensor(garment_indices, dtype=torch.long)

    # Colour features: take top colour_dim entries from col_cco
    col_row  = colour_mat.getrow(outfit_idx)
    col_feat = np.zeros(colour_dim, dtype=np.float32)
    for ci, cv in zip(col_row.indices, col_row.data):
        if ci < colour_dim:
            col_feat[ci] = cv
    # Normalise
    norm = col_feat.sum()
    if norm > 0:
        col_feat /= norm
    # Same colour context for every node in this outfit
    node_feats = torch.tensor(
        np.tile(col_feat, (N, 1)), dtype=torch.float32)   # (N, colour_dim)

    # Weighted fully-connected edges
    if N > 1:
        src, dst, weights = [], [], []
        for i in range(N):
            for j in range(N):
                if i != j:
                    cat_i = vocab_to_cat.get(int(garment_indices[i]), "accessory")
                    cat_j = vocab_to_cat.get(int(garment_indices[j]), "accessory")
                    w = get_edge_weight(cat_i, cat_j)
                    src.append(i)
                    dst.append(j)
                    weights.append(w)
        edge_index   = torch.tensor([src, dst], dtype=torch.long)
        edge_weights = torch.tensor(weights, dtype=torch.float32)
    else:
        edge_index   = torch.zeros((2, 0), dtype=torch.long)
        edge_weights = torch.zeros((0,),   dtype=torch.float32)

    return node_ids, node_feats, edge_index, edge_weights


def corrupt_outfit_hard(garment_indices, vocab_to_cat, cat_to_vocab,
                        garment_vocab_size):
    """
    Hard negative sampling: replace one garment with another from
    the SAME broad category.

    Example: replace "blazer" with "hoodie" (both tops) rather than
    with a random shoe. This forces the model to learn fine-grained
    compatibility within categories, not just category validity.

    Falls back to random replacement if the category has only one item.
    """
    corrupted = list(garment_indices.numpy() if hasattr(
        garment_indices, 'numpy') else garment_indices)

    replace_pos  = random.randint(0, len(corrupted) - 1)
    original_id  = corrupted[replace_pos]
    original_cat = vocab_to_cat.get(int(original_id), "accessory")

    # Get all vocab indices in the same category, excluding current outfit
    same_cat_candidates = [
        v for v in cat_to_vocab.get(original_cat, [])
        if v not in corrupted
    ]

    if len(same_cat_candidates) >= 1:
        new_garment = random.choice(same_cat_candidates)
    else:
        # Fallback: fully random
        new_garment = random.randint(0, garment_vocab_size - 1)
        while new_garment in corrupted:
            new_garment = random.randint(0, garment_vocab_size - 1)

    corrupted[replace_pos] = new_garment
    return torch.tensor(corrupted, dtype=torch.long)


# ──────────────────────────────────────────────────────────────
# 5. DATASET
# ──────────────────────────────────────────────────────────────

class OutfitDataset:
    def __init__(self, outfit_ids, garment_mat, colour_mat,
                 votes, vocab_to_cat, cat_to_vocab,
                 colour_dim=32):
        self.outfit_ids    = outfit_ids
        self.garment_mat   = garment_mat
        self.colour_mat    = colour_mat
        self.votes         = votes
        self.vocab_to_cat  = vocab_to_cat
        self.cat_to_vocab  = cat_to_vocab
        self.vocab_size    = garment_mat.shape[1]
        self.colour_dim    = colour_dim

    def __len__(self):
        return len(self.outfit_ids)

    def __getitem__(self, idx):
        outfit_idx = self.outfit_ids[idx]

        node_ids, node_feats, edge_index, edge_weights = outfit_to_graph(
            outfit_idx, self.garment_mat, self.colour_mat,
            self.vocab_to_cat, self.colour_dim)

        corrupted_node_ids = corrupt_outfit_hard(
            node_ids, self.vocab_to_cat, self.cat_to_vocab, self.vocab_size)

        # Fashionability weight for this outfit (used in weighted BPR loss)
        vote_weight = float(self.votes[outfit_idx]) \
            if outfit_idx < len(self.votes) else 0.5

        return {
            'node_ids':           node_ids,
            'node_feats':         node_feats,
            'edge_index':         edge_index,
            'edge_weights':       edge_weights,
            'corrupted_node_ids': corrupted_node_ids,
            'vote_weight':        vote_weight,
        }


# ──────────────────────────────────────────────────────────────
# 6. GNN MODEL
# ──────────────────────────────────────────────────────────────

class WeightedGNNLayer(nn.Module):
    """
    GNN layer with edge-weight-aware aggregation.
    Neighbours with higher edge weights contribute more to the update.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm   = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index, edge_weights, num_nodes):
        if edge_index.shape[1] == 0:
            return F.relu(self.norm(self.linear(x)))

        src, dst = edge_index[0], edge_index[1]

        # Weighted aggregation: weight each neighbour message by edge weight
        weighted_msgs = x[src] * edge_weights.unsqueeze(1)   # (E, dim)
        agg  = torch.zeros_like(x)
        agg.index_add_(0, dst, weighted_msgs)

        # Normalise by sum of weights (not count)
        weight_sum = torch.zeros(num_nodes, device=x.device)
        weight_sum.index_add_(0, dst, edge_weights)
        weight_sum = weight_sum.clamp(min=1e-6).unsqueeze(1)
        agg = agg / weight_sum

        return F.relu(self.norm(self.linear(x + agg)))


class OutfitGNN(nn.Module):
    """
    Improved 2-layer GNN for outfit compatibility scoring.

    Node input = garment embedding (64-dim) + colour context (colour_dim)
    Edge weights = category-pair importance
    Output = compatibility score + per-garment contribution scores
    """
    def __init__(self, vocab_size, embed_dim=64,
                 hidden_dim=128, colour_dim=32):
        super().__init__()
        self.colour_dim = colour_dim

        # Garment type embedding (learned from co-occurrence)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Project concatenated [embedding + colour_feat] → hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim + colour_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.gnn1 = WeightedGNNLayer(hidden_dim, hidden_dim)
        self.gnn2 = WeightedGNNLayer(hidden_dim, hidden_dim)

        # Per-garment contribution score (weak-link detection)
        self.item_score = nn.Linear(hidden_dim, 1)

        # Outfit-level compatibility score
        self.outfit_score = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, node_ids, node_feats, edge_index, edge_weights):
        """
        node_ids     : (N,)        garment vocab indices
        node_feats   : (N, C)      colour context features
        edge_index   : (2, E)
        edge_weights : (E,)        category-pair importance
        """
        N = len(node_ids)

        # Combine garment embedding + colour context
        emb = self.embedding(node_ids)                # (N, embed_dim)
        x   = torch.cat([emb, node_feats], dim=1)    # (N, embed+colour)
        x   = self.input_proj(x)                     # (N, hidden_dim)

        # Weighted message passing
        x = self.gnn1(x, edge_index, edge_weights, N)
        x = self.gnn2(x, edge_index, edge_weights, N)

        per_item    = self.item_score(x).squeeze(-1)
        outfit_repr = x.mean(dim=0, keepdim=True)
        score       = self.outfit_score(outfit_repr).squeeze()

        return score, per_item


# ──────────────────────────────────────────────────────────────
# 7. TRAINING
# ──────────────────────────────────────────────────────────────

def weighted_bpr_loss(pos_scores, neg_scores, weights):
    """
    Fashionability-weighted BPR loss.
    Outfits with higher votes contribute more to the gradient signal.
    This helps the model learn from high-quality outfits more than
    from average/low-quality ones.
    """
    raw_loss = -F.logsigmoid(pos_scores - neg_scores)
    return (raw_loss * weights).mean()


def train_epoch(model, dataset, optimizer, device,
                accumulation_steps=8):
    """
    Training with gradient accumulation.
    backward() is called immediately per sample — no large tensor lists.
    """
    model.train()
    total_loss = 0.0
    n_steps    = 0
    indices    = list(range(len(dataset)))
    random.shuffle(indices)

    optimizer.zero_grad()

    for i, idx in enumerate(tqdm(indices, desc="  Training", leave=False)):
        sample = dataset[idx]
        node_ids      = sample['node_ids'].to(device)
        node_feats    = sample['node_feats'].to(device)
        edge_index    = sample['edge_index'].to(device)
        edge_weights  = sample['edge_weights'].to(device)
        corrupt_ids   = sample['corrupted_node_ids'].to(device)
        vote_w        = torch.tensor(
            [sample['vote_weight']], dtype=torch.float32, device=device)

        pos_score, _ = model(node_ids, node_feats, edge_index, edge_weights)
        neg_score, _ = model(corrupt_ids, node_feats, edge_index, edge_weights)

        loss = weighted_bpr_loss(
            pos_score.unsqueeze(0),
            neg_score.unsqueeze(0),
            vote_w
        ) / accumulation_steps
        loss.backward()

        total_loss += loss.item() * accumulation_steps

        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            n_steps += 1

    if len(indices) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        n_steps += 1

    return total_loss / max(1, n_steps)


@torch.no_grad()
def evaluate(model, dataset, device, max_samples=2000):
    """AUC: can model separate real outfits from hard negatives?"""
    model.eval()
    labels, scores = [], []

    indices = random.sample(range(len(dataset)),
                            min(max_samples, len(dataset)))

    for idx in indices:
        sample = dataset[idx]
        node_ids     = sample['node_ids'].to(device)
        node_feats   = sample['node_feats'].to(device)
        edge_index   = sample['edge_index'].to(device)
        edge_weights = sample['edge_weights'].to(device)
        corrupt_ids  = sample['corrupted_node_ids'].to(device)

        ps, _ = model(node_ids, node_feats, edge_index, edge_weights)
        ns, _ = model(corrupt_ids, node_feats, edge_index, edge_weights)

        scores.append(ps.item())
        scores.append(ns.item())
        labels.append(1)
        labels.append(0)

    return roc_auc_score(labels, scores)


# ──────────────────────────────────────────────────────────────
# 8. MAIN
# ──────────────────────────────────────────────────────────────

def main(args):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    garment_mat, colour_mat, votes, train_ids, val_ids, test_ids, freq_order = \
        load_fashion144k(args.data_dir)

    GARMENT_VOCAB = garment_mat.shape[1]
    COLOUR_DIM    = args.colour_dim

    # Build category mapper
    vocab_to_cat, cat_to_vocab = build_category_mapper(freq_order)
    print(f"  Category distribution:")
    for cat, idxs in cat_to_vocab.items():
        print(f"    {cat:12s}: {len(idxs)} vocab indices")

    # Datasets
    train_ds = OutfitDataset(train_ids, garment_mat, colour_mat,
                             votes, vocab_to_cat, cat_to_vocab, COLOUR_DIM)
    val_ds   = OutfitDataset(val_ids,   garment_mat, colour_mat,
                             votes, vocab_to_cat, cat_to_vocab, COLOUR_DIM)
    test_ds  = OutfitDataset(test_ids,  garment_mat, colour_mat,
                             votes, vocab_to_cat, cat_to_vocab, COLOUR_DIM)

    # Model
    model = OutfitGNN(
        vocab_size  = GARMENT_VOCAB,
        embed_dim   = args.embed_dim,
        hidden_dim  = args.hidden_dim,
        colour_dim  = COLOUR_DIM,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.5)

    best_val_auc = 0.0
    results      = []
    model_path   = os.path.join(args.out_dir, "outfit_gnn_v2.pt")

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        loss    = train_epoch(model, train_ds, optimizer, device)
        val_auc = evaluate(model, val_ds, device)
        scheduler.step()

        line = f"Epoch {epoch:3d} | Loss {loss:.4f} | Val AUC {val_auc:.4f}"
        results.append(line)
        print(line)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Saved best model (AUC {best_val_auc:.4f})")

    # Final test evaluation
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True))
    test_auc = evaluate(model, test_ds, device, max_samples=2000)
    print(f"\nTest AUC: {test_auc:.4f}")

    # Save results
    results_path = os.path.join(args.out_dir, "results_v2.txt")
    with open(results_path, "w") as f:
        f.write("\n".join(results))
        f.write(f"\n\nTest AUC: {test_auc:.4f}\n")
    print(f"Results saved → {results_path}")
    print(f"Model saved   → {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GNN Outfit Compatibility v2 — Fashion144K")
    parser.add_argument("--data_dir",   type=str,   required=True)
    parser.add_argument("--out_dir",    type=str,   default=".")
    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--embed_dim",  type=int,   default=64)
    parser.add_argument("--hidden_dim", type=int,   default=128)
    parser.add_argument("--colour_dim", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
