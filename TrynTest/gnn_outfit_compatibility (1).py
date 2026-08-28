"""
GNN Outfit Compatibility Pipeline — Fashion144K
================================================
Trains a 2-layer Graph Neural Network to score outfit compatibility.
Each outfit = a graph (nodes = garment types, edges = co-occurrence).
Uses BPR loss: real outfit score > corrupted outfit score.

Usage:
    pip install torch torch-geometric scipy numpy scikit-learn tqdm

    python gnn_outfit_compatibility.py --data_dir "C:/path/to/Fashion144k_v1"

Outputs:
    outfit_gnn.pt          — trained model weights
    garment_embeddings.npy — learned per-garment embeddings
    results.txt            — train/val AUC scores
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────────────────────

def load_sparse_cco(path):
    """Load a sparse _cco.mat file → scipy csr_matrix."""
    data = loadmat(path)
    rows = data['rows'].flatten().astype(int)
    cols = data['cols'].flatten().astype(int)
    vals = data['data'].flatten().astype(np.float32)
    # +1 because MATLAB uses 0-indexed but we need to check
    # Actually Fashion144K rows/cols are already 0-indexed per README
    mat = csr_matrix((vals, (rows, cols)))
    return mat


def load_fashion144k(data_dir):
    """
    Load Fashion144K and return outfit data.
    Returns:
        garment_mat: (N_outfits, 1352) sparse matrix — garment presence
        colour_mat:  (N_outfits, 604)  sparse matrix — colour presence
        train_ids:   1D array of training outfit indices
        val_ids:     1D array of validation outfit indices
        test_ids:    1D array of test outfit indices
    """
    print("Loading Fashion144K...")

    garment_mat = load_sparse_cco(
        os.path.join(data_dir, "feat", "garflat_cco.mat"))
    colour_mat = load_sparse_cco(
        os.path.join(data_dir, "feat", "col_cco.mat"))

    splits = loadmat(os.path.join(data_dir, "split.mat"))
    # README: ids are 0-indexed in MATLAB, so +1 to convert to Python
    train_ids = splits['trainids'].flatten().astype(int) + 1
    val_ids   = splits['validids'].flatten().astype(int) + 1
    test_ids  = splits['testids'].flatten().astype(int)  + 1

    N = garment_mat.shape[0]

    # Clamp to valid range (guard against any off-by-one at boundaries)
    train_ids = train_ids[train_ids < N]
    val_ids   = val_ids[val_ids < N]
    test_ids  = test_ids[test_ids < N]

    # Filter to only outfits with ≥2 garments (needed for a graph)
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

    return garment_mat, colour_mat, train_ids, val_ids, test_ids


# ──────────────────────────────────────────────────────────────
# 2. OUTFIT GRAPH CONSTRUCTION
# ──────────────────────────────────────────────────────────────

def outfit_to_graph(outfit_idx, garment_mat, colour_mat):
    """
    Convert a single outfit into a graph for GNN processing.

    Returns:
        node_ids   : (num_garments,)       — garment vocab indices
        node_feats : (num_garments, F)     — node feature vectors
        edge_index : (2, num_edges)        — fully connected edges
    """
    # Get which garments are in this outfit
    row = garment_mat.getrow(outfit_idx)
    garment_indices = row.indices  # indices into garment vocab
    garment_weights = row.data     # presence/count weights

    num_nodes = len(garment_indices)

    # Node features: garment one-hot index + colour info
    # Use the garment vocab index as the node ID (for embedding lookup)
    node_ids = torch.tensor(garment_indices, dtype=torch.long)

    # Build fully-connected edge index (every garment connects to every other)
    if num_nodes > 1:
        src, dst = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    src.append(i)
                    dst.append(j)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return node_ids, edge_index


def corrupt_outfit(garment_indices, garment_vocab_size, num_corruptions=1):
    """
    Create a negative sample by replacing one garment with a random one.
    Returns corrupted garment indices.
    """
    corrupted = list(garment_indices.numpy())
    for _ in range(num_corruptions):
        replace_pos = random.randint(0, len(corrupted) - 1)
        new_garment = random.randint(0, garment_vocab_size - 1)
        while new_garment in corrupted:
            new_garment = random.randint(0, garment_vocab_size - 1)
        corrupted[replace_pos] = new_garment
    return torch.tensor(corrupted, dtype=torch.long)


# ──────────────────────────────────────────────────────────────
# 3. DATASET
# ──────────────────────────────────────────────────────────────

class OutfitDataset(Dataset):
    def __init__(self, outfit_ids, garment_mat, colour_mat):
        self.outfit_ids   = outfit_ids
        self.garment_mat  = garment_mat
        self.colour_mat   = colour_mat
        self.vocab_size   = garment_mat.shape[1]

    def __len__(self):
        return len(self.outfit_ids)

    def __getitem__(self, idx):
        outfit_idx = self.outfit_ids[idx]
        node_ids, edge_index = outfit_to_graph(
            outfit_idx, self.garment_mat, self.colour_mat)
        corrupted_node_ids = corrupt_outfit(node_ids, self.vocab_size)
        return {
            'node_ids':           node_ids,
            'edge_index':         edge_index,
            'corrupted_node_ids': corrupted_node_ids,
            'num_nodes':          len(node_ids),
        }


def collate_fn(batch):
    """Custom collate: pack variable-size graphs into a batch."""
    return batch  # Return list of dicts; GNN processes each graph separately


# ──────────────────────────────────────────────────────────────
# 4. GNN MODEL
# ──────────────────────────────────────────────────────────────

class GNNLayer(nn.Module):
    """Simple mean-aggregation GNN layer."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm   = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index, num_nodes):
        """
        x          : (N, in_dim) node features
        edge_index : (2, E) edges
        """
        if edge_index.shape[1] == 0:
            # No edges — just transform
            return F.relu(self.norm(self.linear(x)))

        src, dst = edge_index[0], edge_index[1]

        # Aggregate: mean of neighbour features
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])

        # Count neighbours for mean
        deg = torch.zeros(num_nodes, device=x.device)
        deg.index_add_(0, dst, torch.ones(len(dst), device=x.device))
        deg = deg.clamp(min=1).unsqueeze(1)

        agg = agg / deg

        # Combine self + aggregated
        out = self.linear(x + agg)
        return F.relu(self.norm(out))


class OutfitGNN(nn.Module):
    """
    2-layer GNN for outfit compatibility scoring.

    Architecture:
        Embedding lookup → GNN layer 1 → GNN layer 2 → mean pool → MLP → score
    """
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.gnn1 = GNNLayer(embed_dim, hidden_dim)
        self.gnn2 = GNNLayer(hidden_dim, hidden_dim)

        # Per-garment contribution head (for weak-link detection)
        self.item_score = nn.Linear(hidden_dim, 1)

        # Outfit-level compatibility score
        self.outfit_score = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, node_ids, edge_index):
        """
        node_ids   : (N,) garment vocab indices
        edge_index : (2, E)

        Returns:
            outfit_score     : scalar compatibility score
            per_item_scores  : (N,) per-garment contribution scores
        """
        N = len(node_ids)

        # Node features from embedding
        x = self.embedding(node_ids)           # (N, embed_dim)

        # GNN message passing
        x = self.gnn1(x, edge_index, N)        # (N, hidden_dim)
        x = self.gnn2(x, edge_index, N)        # (N, hidden_dim)

        # Per-item scores (for weak-link detection)
        per_item = self.item_score(x).squeeze(-1)   # (N,)

        # Outfit-level score: mean pool → MLP
        outfit_repr = x.mean(dim=0, keepdim=True)   # (1, hidden_dim)
        score = self.outfit_score(outfit_repr).squeeze()

        return score, per_item


# ──────────────────────────────────────────────────────────────
# 5. TRAINING
# ──────────────────────────────────────────────────────────────

def bpr_loss(pos_score, neg_score):
    """Bayesian Personalised Ranking loss: pos should score higher than neg."""
    return -F.logsigmoid(pos_score - neg_score).mean()


def train_epoch(model, dataset, optimizer, device, batch_size=256):
    model.train()
    total_loss = 0
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    for start in tqdm(range(0, len(indices), batch_size),
                      desc="Training", leave=False):
        batch_idx = indices[start:start + batch_size]
        pos_scores, neg_scores = [], []

        for idx in batch_idx:
            sample = dataset[idx]
            node_ids    = sample['node_ids'].to(device)
            edge_index  = sample['edge_index'].to(device)
            corrupt_ids = sample['corrupted_node_ids'].to(device)

            pos_score, _ = model(node_ids, edge_index)
            neg_score, _ = model(corrupt_ids, edge_index)

            pos_scores.append(pos_score)
            neg_scores.append(neg_score)

        pos_scores = torch.stack(pos_scores)
        neg_scores = torch.stack(neg_scores)

        loss = bpr_loss(pos_scores, neg_scores)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(indices) // batch_size)


@torch.no_grad()
def evaluate(model, dataset, device, max_samples=2000):
    """Compute AUC: can model distinguish real outfits from corrupted ones?"""
    model.eval()
    labels, scores = [], []

    indices = random.sample(range(len(dataset)),
                            min(max_samples, len(dataset)))

    for idx in indices:
        sample = dataset[idx]
        node_ids    = sample['node_ids'].to(device)
        edge_index  = sample['edge_index'].to(device)
        corrupt_ids = sample['corrupted_node_ids'].to(device)

        pos_score, _ = model(node_ids, edge_index)
        neg_score, _ = model(corrupt_ids, edge_index)

        scores.extend([pos_score.item(), neg_score.item()])
        labels.extend([1, 0])

    return roc_auc_score(labels, scores)


# ──────────────────────────────────────────────────────────────
# 6. WEAK-LINK DETECTION (inference utility)
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def find_weak_link(model, garment_indices, garment_vocab, device):
    """
    Given a list of garment vocab indices, find the weakest item.

    Returns:
        weak_idx      : position in garment_indices of the weak link
        per_item_scores: scores for each garment
        outfit_score  : overall compatibility score
    """
    model.eval()
    node_ids = torch.tensor(garment_indices, dtype=torch.long).to(device)

    # Build fully connected edge index
    N = len(garment_indices)
    src, dst = [], []
    for i in range(N):
        for j in range(N):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long).to(device)

    score, per_item = model(node_ids, edge_index)

    weak_idx = per_item.argmin().item()
    print(f"\nOutfit compatibility score: {score.item():.4f}")
    print(f"Per-garment scores:")
    for i, (gid, s) in enumerate(zip(garment_indices, per_item.tolist())):
        marker = " ← WEAK LINK" if i == weak_idx else ""
        print(f"  Garment {gid:4d}: {s:.4f}{marker}")

    return weak_idx, per_item.cpu().numpy(), score.item()


# ──────────────────────────────────────────────────────────────
# 7. MAIN
# ──────────────────────────────────────────────────────────────

def main(args):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    garment_mat, colour_mat, train_ids, val_ids, test_ids = \
        load_fashion144k(args.data_dir)

    GARMENT_VOCAB = garment_mat.shape[1]

    # Datasets
    train_ds = OutfitDataset(train_ids, garment_mat, colour_mat)
    val_ds   = OutfitDataset(val_ids,   garment_mat, colour_mat)
    test_ds  = OutfitDataset(test_ids,  garment_mat, colour_mat)

    # Model
    model = OutfitGNN(
        vocab_size  = GARMENT_VOCAB,
        embed_dim   = args.embed_dim,
        hidden_dim  = args.hidden_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.5)

    best_val_auc = 0.0
    results = []

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            model, train_ds, optimizer, device, args.batch_size)
        val_auc = evaluate(model, val_ds, device)
        scheduler.step()

        results.append(
            f"Epoch {epoch:3d} | Loss {loss:.4f} | Val AUC {val_auc:.4f}")
        print(results[-1])

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(),
                       os.path.join(args.out_dir, "outfit_gnn.pt"))
            print(f"  ✓ Saved best model (AUC {best_val_auc:.4f})")

    # ── Final test evaluation ──
    model.load_state_dict(
        torch.load(os.path.join(args.out_dir, "outfit_gnn.pt"),
                   map_location=device))
    test_auc = evaluate(model, test_ds, device, max_samples=5000)
    print(f"\nTest AUC: {test_auc:.4f}")

    # ── Save garment embeddings ──
    embeddings = model.embedding.weight.detach().cpu().numpy()
    np.save(os.path.join(args.out_dir, "garment_embeddings.npy"), embeddings)
    print(f"Saved garment embeddings: {embeddings.shape}")

    # ── Save results log ──
    results_path = os.path.join(args.out_dir, "results.txt")
    with open(results_path, "w") as f:
        f.write("\n".join(results))
        f.write(f"\n\nTest AUC: {test_auc:.4f}\n")
    print(f"Results saved to {results_path}")

    # ── Demo: weak-link detection on a random outfit ──
    print("\n--- Demo: Weak-Link Detection ---")
    sample_idx = random.choice(test_ids)
    row = garment_mat.getrow(sample_idx)
    garment_indices = row.indices.tolist()
    if len(garment_indices) >= 2:
        find_weak_link(model, garment_indices, GARMENT_VOCAB, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GNN Outfit Compatibility — Fashion144K")
    parser.add_argument("--data_dir",   type=str,   required=True)
    parser.add_argument("--out_dir",    type=str,   default=".")
    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--batch_size", type=int,   default=256)
    parser.add_argument("--embed_dim",  type=int,   default=64)
    parser.add_argument("--hidden_dim", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    main(args)