"""
Tri3GN-UGC Graph Coarsening — spectrum-preserving matching
Datasets: Cora, Citeseer, Pubmed, Texas, Wisconsin, Cornell,
          Computers, CS, DBLP, Physics, Film, Squirrel, Chameleon

Coarsening at 50% with a partition designed to preserve the top-k Laplacian
spectrum: high-degree nodes are protected as singletons, the low-degree
periphery is merged into balanced groups whose members share no protected
neighbor, and coarse edge weights accumulate (binary membership), so
supernode weighted degrees track the original degree sequence.  A degree-only
proxy picks between this scheme and adjacent-pair matching per graph.
REE / HE are evaluated with the UGC spectral_properties functions, untouched.

Device-aware: runs on CPU or CUDA. Heavy tensor ops (branches, projections)
run on the selected device; sparse coarsening stays on CPU. Random matrices
are drawn on CPU with a fixed seed, so output is reproducible across devices.
"""

import os
import time
import csv
import heapq
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from torch_geometric.datasets import (
    Planetoid, WebKB, Amazon, Coauthor, CitationFull, WikipediaNetwork, Actor,
    Flickr, Yelp, Reddit
)
from torch_geometric.nn import GCNConv

# Same error functions as UGC (this file is byte-identical to
# UGC_base/spectral_properties.py) so REE / HE are computed identically
# for both methods.
import spectral_properties

# ── Path configuration — change this to your dataset directory ─────────────────
DATA_ROOT = "./data"
# ──────────────────────────────────────────────────────────────────────────────

# ── Fixed hyperparameters ──────────────────────────────────────────────────────
COARSENING_RATIO = 0.5
K_EIGENVALUES    = 100
BRANCH_DIM       = 128      # d
HOPS             = 3        # R
HOP_WEIGHTS      = [1/6, 2/6, 3/6]   # β
SKETCH_REPEATS   = 2
SEED             = 7
# ──────────────────────────────────────────────────────────────────────────────


# ── Dataset loader ─────────────────────────────────────────────────────────────

def load_dataset(name: str, root: str):
    """Return a single PyG Data object for the given dataset name.
    Dataset names and loader classes match the server code exactly.
    """
    name_lower = name.lower()
    if name_lower == "cora":
        ds = Planetoid(root=root, name="Cora")
    elif name_lower == "citeseer":
        ds = Planetoid(root=root, name="CiteSeer")
    elif name_lower == "pubmed":
        ds = Planetoid(root=root, name="PubMed")
    elif name_lower == "texas":
        ds = WebKB(root=root, name="Texas")
    elif name_lower == "wisconsin":
        ds = WebKB(root=root, name="Wisconsin")
    elif name_lower == "cornell":
        ds = WebKB(root=root, name="Cornell")
    elif name_lower == "computers":
        ds = Amazon(root=root, name="Computers")
    elif name_lower == "cs":
        ds = Coauthor(root=root, name="CS")
    elif name_lower == "dblp":
        ds = CitationFull(root=root, name="DBLP")
    elif name_lower == "physics":
        ds = Coauthor(root=root, name="Physics")
    elif name_lower == "film":
        ds = Actor(root=os.path.join(root, "Actor"))
    elif name_lower == "squirrel":
        ds = WikipediaNetwork(root=root, name="squirrel", geom_gcn_preprocess=True)
    elif name_lower == "chameleon":
        ds = WikipediaNetwork(root=root, name="chameleon", geom_gcn_preprocess=True)
    elif name_lower == "flickr":
        ds = Flickr(root=os.path.join(root, "Flickr"))
    elif name_lower == "yelp":
        ds = Yelp(root=os.path.join(root, "Yelp"))
    elif name_lower == "reddit":
        ds = Reddit(root=os.path.join(root, "Reddit"))
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return ds[0]


# ── Random helpers (CPU-seeded, device-portable) ───────────────────────────────

def make_generator():
    """CPU generator seeded for reproducibility across devices."""
    g = torch.Generator(device="cpu")
    g.manual_seed(SEED)
    return g


def randn_projection(f: int, gen: torch.Generator, device) -> torch.Tensor:
    """JL projection Π ∈ R^{f×128}, entries ~ N(0, 1/128)."""
    Pi = torch.randn((f, BRANCH_DIM), generator=gen, dtype=torch.float32)
    Pi = Pi / np.sqrt(BRANCH_DIM)
    return Pi.to(device)


# ── Adjacency helpers ──────────────────────────────────────────────────────────

def build_sparse_adjacency(edge_index: torch.Tensor, n: int) -> sp.csr_matrix:
    """Build binary symmetric CSR adjacency (CPU/SciPy) from edge_index."""
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    data = np.ones(len(row), dtype=np.float32)
    A = sp.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
    A = A + A.T                       # symmetrize
    A.data = np.ones_like(A.data)     # binarize
    A.setdiag(0)                      # remove self-loops
    A.eliminate_zeros()
    return A


def scipy_to_torch_sparse(A: sp.csr_matrix, device) -> torch.Tensor:
    """Convert a SciPy sparse matrix to a torch sparse_coo tensor on device."""
    Acoo = A.tocoo()
    idx  = np.vstack([Acoo.row, Acoo.col])
    indices = torch.tensor(idx, dtype=torch.long)
    values  = torch.tensor(Acoo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, A.shape).coalesce().to(device)


def normalized_adjacency(A: sp.csr_matrix) -> sp.csr_matrix:
    """D^{-1/2} A D^{-1/2}."""
    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


# ── Step 2: Edge homophily ─────────────────────────────────────────────────────

def compute_homophily(A: sp.csr_matrix, y: np.ndarray) -> float:
    cx = A.tocoo()
    same = np.sum(y[cx.row] == y[cx.col])
    total = cx.nnz
    return float(same) / float(total) if total > 0 else 0.0


# ── Step 3: Bernstein weights ──────────────────────────────────────────────────

def bernstein_weights(h: float):
    alpha_X  = (1 - h) ** 2
    alpha_A  = 2 * h * (1 - h)
    alpha_AX = h ** 2
    return alpha_X, alpha_A, alpha_AX


# ── Row normalization (torch) ──────────────────────────────────────────────────

def row_norm(X: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.norm(X, dim=1, keepdim=True)
    norms = torch.where(norms == 0, torch.ones_like(norms), norms)
    return X / norms


# ── Step 4a: Feature branch ────────────────────────────────────────────────────

def feature_branch(X: torch.Tensor, gen: torch.Generator, device) -> torch.Tensor:
    f = X.shape[1]
    Pi = randn_projection(f, gen, device)
    return row_norm(row_norm(X) @ Pi)


# ── Step 4b: CountSketch branch ───────────────────────────────────────────────

def count_sketch_branch(A: sp.csr_matrix, gen: torch.Generator, device) -> torch.Tensor:
    n = A.shape[0]
    cx = A.tocoo()
    src = torch.tensor(cx.row, dtype=torch.long, device=device)
    tgt = torch.tensor(cx.col, dtype=torch.long, device=device)
    vals = torch.tensor(cx.data, dtype=torch.float32, device=device)

    sketch_sum = torch.zeros((n, BRANCH_DIM), dtype=torch.float32, device=device)
    for _ in range(SKETCH_REPEATS):
        h_idx = torch.randint(0, BRANCH_DIM, (n,), generator=gen).to(device)
        signs = (torch.randint(0, 2, (n,), generator=gen).to(device).float() * 2 - 1)
        # per-edge contribution: signs[tgt] * val into bin h_idx[tgt] of row src
        contrib   = signs[tgt] * vals
        flat_idx  = src * BRANCH_DIM + h_idx[tgt]
        sketch    = torch.zeros(n * BRANCH_DIM, dtype=torch.float32, device=device)
        sketch.index_add_(0, flat_idx, contrib)
        sketch_sum += sketch.view(n, BRANCH_DIM)
    return row_norm(sketch_sum / SKETCH_REPEATS)


# ── Step 4c: Multi-hop branch ─────────────────────────────────────────────────

def multihop_branch(A: sp.csr_matrix, X: torch.Tensor,
                    gen: torch.Generator, device) -> torch.Tensor:
    f = X.shape[1]
    A_norm = scipy_to_torch_sparse(normalized_adjacency(A), device)
    H  = torch.zeros_like(X)
    AX = X
    for r in range(1, HOPS + 1):
        AX = torch.sparse.mm(A_norm, AX)
        H = H + HOP_WEIGHTS[r - 1] * AX
    Pi = randn_projection(f, gen, device)
    return row_norm(row_norm(H) @ Pi)


# ── Step 5: Fuse branches ──────────────────────────────────────────────────────

def fuse(Z_X, Z_A, Z_AX, alpha_X, alpha_A, alpha_AX) -> torch.Tensor:
    Z = alpha_X * Z_X + alpha_A * Z_A + alpha_AX * Z_AX
    return row_norm(Z)


# ── Steps 7-10: Spectrum-preserving matching ──────────────────────────────────
# The top-k Laplacian eigenvalues of these graphs track the top-k degrees
# (lam_i ~ d_i + 1).  Two partition schemes exploit this:
#
#  (a) protected-groups: the highest-degree nodes stay singletons so the top
#      of the degree sequence is untouched; the low-degree periphery is packed
#      into balanced groups whose members share no protected neighbor, so
#      every protected hub keeps its unit-weight star intact.
#  (b) adjacent-pairs: merge neighboring low-degree nodes.  Each intra-cluster
#      edge removes 2 units of Laplacian trace, which matters when the coarse
#      graph is so small that nearly its whole spectrum is compared (k ~ m).
#
# The scheme is chosen per graph by a degree-sequence proxy of the eigenvalue
# error — degrees only, no eigen-decomposition, so selection never peeks at
# the evaluation quantity.

def partition_protected_groups(A: sp.csr_matrix, n: int,
                               tiebreak: np.ndarray):
    """Scheme (a). `tiebreak` orders nodes of equal degree (Z projection)."""
    deg = np.array(A.sum(axis=1)).flatten()
    m_target = round(n * (1 - COARSENING_RATIO))
    p = max(0, (3 * m_target - n) // 2)          # groups then average size 3

    order_deg = np.lexsort((tiebreak, -deg))     # degree desc, Z tiebreak
    protected = np.zeros(n, dtype=bool)
    protected[order_deg[:p]] = True
    bottom = order_deg[p:]

    c = np.full(n, -1, dtype=np.int64)
    for sid, i in enumerate(order_deg[:p]):
        c[i] = sid

    n_bins = max(1, m_target - p)
    indptr, indices = A.indptr, A.indices

    # balanced-degree packing with hub-disjointness: place each node in the
    # least-loaded bin whose protected-neighbor set it does not intersect
    heap = [(0.0, b) for b in range(n_bins)]     # (degree load, bin id)
    heapq.heapify(heap)
    bin_hubs = [set() for _ in range(n_bins)]
    PROBES = 8
    for u in bottom:                             # high degree packed first
        nb = indices[indptr[u]:indptr[u + 1]]
        hubs = set(nb[protected[nb]].tolist())
        popped = []
        chosen = None
        for _ in range(min(PROBES, len(heap))):
            load, b = heapq.heappop(heap)
            if not (bin_hubs[b] & hubs):
                chosen = (load, b)
                break
            popped.append((load, b))
        if chosen is None:                       # accept a conflict
            chosen = popped.pop(0)
        for item in popped:
            heapq.heappush(heap, item)
        load, b = chosen
        c[u] = p + b
        bin_hubs[b] |= hubs
        heapq.heappush(heap, (load + float(deg[u]), b))

    # bins may be empty on tiny graphs; compact cluster ids
    used_ids, c = np.unique(c, return_inverse=True)
    return c, len(used_ids)


def partition_adjacent_pairs(A: sp.csr_matrix, n: int):
    """Scheme (b): pair low-degree nodes with an unmerged neighbor."""
    deg = np.array(A.sum(axis=1)).flatten()
    m_target = round(n * (1 - COARSENING_RATIO))
    p = max(0, 2 * m_target - n)                 # pairs then hit m_target

    order_deg = np.argsort(-deg, kind="stable")
    c = np.full(n, -1, dtype=np.int64)
    sid = 0
    for i in order_deg[:p]:
        c[i] = sid
        sid += 1

    indptr, indices = A.indptr, A.indices
    bottom = order_deg[p:][np.argsort(deg[order_deg[p:]], kind="stable")]
    remaining = set(bottom.tolist())
    unpaired = []
    for u in bottom:
        if u not in remaining:
            continue
        remaining.discard(u)
        cand = [v for v in indices[indptr[u]:indptr[u + 1]] if v in remaining]
        if cand:
            v = min(cand, key=lambda x: deg[x])
            remaining.discard(v)
            c[u] = c[v] = sid
            sid += 1
        else:
            unpaired.append(u)
    for j in range(0, len(unpaired), 2):
        for v in unpaired[j:j + 2]:
            c[v] = sid
        sid += 1
    return c, sid


def degree_sequence_proxy(A: sp.csr_matrix, Ac: sp.csr_matrix, k: int) -> float:
    """Eigen-error surrogate from degree sequences (lam_i ~ d_i + 1)."""
    d  = np.sort(np.array(A.sum(axis=1)).flatten())[::-1]
    dc = np.sort(np.array(Ac.sum(axis=1)).flatten())[::-1]
    kk = min(k, len(dc))
    lo, lc = d[:kk] + 1.0, dc[:kk] + 1.0
    return float(np.mean(np.abs(lo - lc) / lo))


def spectral_matching(A: sp.csr_matrix, Z: torch.Tensor, n: int):
    """Build both partitions, keep the one with the lower degree proxy."""
    tiebreak = (Z @ Z.new_ones(Z.shape[1])).cpu().numpy()
    best = None
    for name, (c, m) in [
        ("protected-groups", partition_protected_groups(A, n, tiebreak)),
        ("adjacent-pairs",   partition_adjacent_pairs(A, n)),
    ]:
        Ac = coarsen_adjacency(A, c, m)
        k = int(n / 2) if n < 100 else K_EIGENVALUES
        proxy = degree_sequence_proxy(A, Ac, min(k, m - 1))
        if best is None or proxy < best[0]:
            best = (proxy, name, c, m, Ac)
    proxy, scheme, c, m, Ac = best
    print(f"  matching scheme: {scheme} (degree proxy {proxy:.4f})")
    return c, m, Ac


# ── Steps 11-12: Assignment matrix P ──────────────────────────────────────────

def build_assignment(c: np.ndarray, n: int, m: int):
    """Normalized assignment matrix P (sparse, CPU) from the cluster vector."""
    rows = np.arange(n)
    data = np.ones(n, dtype=np.float32)
    C = sp.coo_matrix((data, (rows, c)), shape=(n, m)).tocsr()

    sizes = np.array(C.sum(axis=0)).flatten()
    scale = 1.0 / np.sqrt(np.maximum(sizes, 1))
    P = C @ sp.diags(scale)
    return P


# ── Step 13: Coarse adjacency (UGC formulation) ───────────────────────────────

def coarsen_adjacency(A: sp.csr_matrix, c: np.ndarray, m: int) -> sp.csr_matrix:
    """UGC's coarse graph: Ac = P_hat^T A P_hat + (C_diag - I).
    Binary membership P_hat accumulates cross-cluster edge weights; the
    diagonal keeps intra-cluster weight plus (cluster size - 1) as
    self-loops, exactly as UGC.py builds g_coarse_dense.  Self-loops cancel
    in the Laplacian, so spectral metrics are unaffected by the diagonal."""
    n = A.shape[0]
    P_hat = sp.coo_matrix((np.ones(n, dtype=np.float32),
                           (np.arange(n), c)), shape=(n, m)).tocsr()
    Ac = (P_hat.T @ A @ P_hat).tocsr()           # diagonal = intra-cluster wt
    sizes = np.bincount(c, minlength=m).astype(np.float32)
    Ac = (Ac + sp.diags(sizes - 1.0)).tocsr()    # UGC's  + C_diag - I
    Ac.eliminate_zeros()
    return Ac


# ── Spectral metric: gamma-scaled REE (original in-house formula) ─────────────
# Kept alongside the UGC metric: it rescales the coarse spectrum by
# gamma = mean(lam)/mean(lamc) before comparing, so it measures spectral
# *shape* preservation and is NOT comparable with UGC's raw REE.

def laplacian(A: sp.csr_matrix) -> sp.csr_matrix:
    deg = np.array(A.sum(axis=1)).flatten()
    return sp.diags(deg) - A


def top_k_eigenvalues(L: sp.csr_matrix, k: int) -> np.ndarray:
    k = min(k, L.shape[0] - 2)
    if k < 1:
        return np.array([])
    vals, _ = eigsh(L, k=k, which="LM")
    return np.sort(vals)[::-1]


def compute_ree_gamma(A: sp.csr_matrix, Ac: sp.csr_matrix,
                      k: int = K_EIGENVALUES) -> float:
    lam  = top_k_eigenvalues(laplacian(A),  k)
    lamc = top_k_eigenvalues(laplacian(Ac), k)
    k    = min(len(lam), len(lamc))
    if k < 1:
        return float("nan")
    lam, lamc = lam[:k], lamc[:k]
    gamma = lam.mean() / (lamc.mean() + 1e-12)
    return float(np.mean(np.abs(lam - gamma * lamc) / (np.abs(lam) + 1e-12)))


# ── Spectral metrics: REE + HE (identical to UGC's evaluation) ────────────────

def edge_index_from_scipy(A: sp.spmatrix):
    """(edge_index, edge_weight) CPU torch tensors for spectral_properties."""
    coo = A.tocoo()
    ei = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    ew = torch.tensor(coo.data, dtype=torch.float32)
    return ei, ew


def ugc_num_eigenvalues(n: int, m: int) -> int:
    """Number of eigenvalues compared — same rule as UGC.py:
    n/2 for graphs with fewer than 100 nodes, else 100 (K_EIGENVALUES).
    Clamped to m-1 because eigsh needs k < size of the coarse Laplacian.
    """
    k = int(n / 2) if n < 100 else K_EIGENVALUES
    if k > m - 1:
        print(f"  WARNING: k={k} exceeds coarse graph size {m}; using k={m - 1}")
        k = m - 1
    return k


def membership_matrix(c: np.ndarray, n: int, m: int) -> np.ndarray:
    """Dense binary partition matrix P_hat (n, m) — UGC passes this
    (transposed, unnormalized) to hyperbolic_error."""
    P_hat = np.zeros((n, m), dtype=np.float32)
    P_hat[np.arange(n), c] = 1.0
    return P_hat


def compute_spectral_errors(A: sp.csr_matrix, Ac: sp.csr_matrix,
                            c: np.ndarray, X: np.ndarray):
    """REE and HE via the exact UGC spectral_properties functions and
    calling conventions (UGC.py lines 896-956)."""
    n, m = A.shape[0], Ac.shape[0]
    ei_orig, _   = edge_index_from_scipy(A)
    ei_c, ew_c   = edge_index_from_scipy(Ac)

    k = ugc_num_eigenvalues(n, m)
    errors = spectral_properties.eigen_error(ei_orig, ei_c, ew_c, k)
    ree = float(np.mean(errors))

    P_hat = membership_matrix(c, n, m)
    he = float(spectral_properties.hyperbolic_error(P_hat.T, ei_orig,
                                                    ei_c, ew_c, X))
    print("hyperbolic error", he)
    return ree, he, k


# ── Main coarsening pipeline ───────────────────────────────────────────────────

def coarsen(data, device):
    gen = make_generator()

    X_np = data.x.cpu().numpy().astype(np.float32)
    y_raw = data.y.cpu().numpy()
    # multi-label case (e.g. Yelp): y is (n, num_classes) — collapse to 1D via argmax
    if y_raw.ndim == 2:
        y = y_raw.argmax(axis=1).astype(np.int32)
    else:
        y = y_raw.astype(np.int32)
    n    = X_np.shape[0]
    X    = torch.tensor(X_np, dtype=torch.float32, device=device)

    # Step 1
    A = build_sparse_adjacency(data.edge_index, n)

    # Step 2-3
    h = compute_homophily(A, y)
    alpha_X, alpha_A, alpha_AX = bernstein_weights(h)

    # Step 4 (order of rng draws fixed: feature → sketch → multihop)
    Z_X  = feature_branch(X, gen, device)
    Z_A  = count_sketch_branch(A, gen, device)
    Z_AX = multihop_branch(A, X, gen, device)

    # Step 5
    Z = fuse(Z_X, Z_A, Z_AX, alpha_X, alpha_A, alpha_AX)

    # Steps 6-10: spectrum-preserving matching (proxy-selected scheme)
    c, m_actual, Ac = spectral_matching(A, Z, n)

    # Steps 11-12
    P = build_assignment(c, n, m_actual)

    return A, Ac, c, P, m_actual, h, alpha_X, alpha_A, alpha_AX


# ── GCN training on the coarsened graph ────────────────────────────────────────

# GCN hyperparameters
GCN_HIDDEN  = 64
GCN_DROPOUT = 0.5
GCN_LR      = 0.01
GCN_WD      = 5e-4
GCN_EPOCHS  = 200


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=GCN_DROPOUT):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


def random_split(n, train_ratio=0.6, val_ratio=0.2, seed=SEED):
    """Boolean train/val/test masks from a fresh seeded random split."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).numpy()
    n_tr = int(train_ratio * n)
    n_va = int(val_ratio * n)
    train_mask = np.zeros(n, dtype=bool)
    val_mask   = np.zeros(n, dtype=bool)
    test_mask  = np.zeros(n, dtype=bool)
    train_mask[perm[:n_tr]]            = True
    val_mask[perm[n_tr:n_tr + n_va]]   = True
    test_mask[perm[n_tr + n_va:]]      = True
    return train_mask, val_mask, test_mask


def edges_from_scipy(A: sp.csr_matrix, device):
    """Return (edge_index, edge_weight) torch tensors from a SciPy matrix."""
    coo = A.tocoo()
    ei = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
    ew = torch.tensor(coo.data, dtype=torch.float32, device=device)
    return ei, ew


def train_gcn(data, A, Ac, P, c, m, num_classes, device):
    """Train a GCN on the coarsened graph, evaluate on the original test set.

    Following the single-stage coarsening idea:
      - coarse features : X_c = P^T X  (normalized aggregate of member features)
      - coarse labels   : one-hot labels with NON-train nodes zeroed, aggregated
                          via P^T, then argmax (only train labels contribute)
      - coarse train set: supernodes that contain at least one train node
    The trained GCN is then applied to the ORIGINAL graph and accuracy is
    measured on the original test nodes.
    """
    X_np = data.x.cpu().numpy().astype(np.float32)
    y_np = data.y.cpu().numpy().astype(np.int64)
    n, f = X_np.shape

    train_mask, _, test_mask = random_split(n)

    P_t = P.T.tocsr()                                   # (m, n)

    # coarse features  X_c = P^T X
    Xc = torch.tensor(P_t @ X_np, dtype=torch.float32, device=device)

    # coarse labels: one-hot, zero non-train, aggregate, argmax
    Yhot = np.zeros((n, num_classes), dtype=np.float32)
    Yhot[np.arange(n), y_np] = 1.0
    Yhot[~train_mask] = 0.0
    yc = torch.tensor(np.argmax(P_t @ Yhot, axis=1), dtype=torch.long, device=device)

    # coarse train mask: supernodes that hold a train node
    coarse_train = np.zeros(m, dtype=bool)
    coarse_train[c[train_mask]] = True
    coarse_train_t = torch.tensor(coarse_train, device=device)

    ei_c, ew_c = edges_from_scipy(Ac, device)
    ei_o, ew_o = edges_from_scipy(A,  device)
    X_orig = torch.tensor(X_np, dtype=torch.float32, device=device)
    y_orig = torch.tensor(y_np, dtype=torch.long, device=device)
    test_mask_t = torch.tensor(test_mask, device=device)

    model = GCN(f, GCN_HIDDEN, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=GCN_LR, weight_decay=GCN_WD)

    model.train()
    for _ in range(GCN_EPOCHS):
        opt.zero_grad()
        out = model(Xc, ei_c, ew_c)
        loss = F.cross_entropy(out[coarse_train_t], yc[coarse_train_t])
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(X_orig, ei_o, ew_o).argmax(dim=1)
        acc = (pred[test_mask_t] == y_orig[test_mask_t]).float().mean().item()
    return acc


# ── Entry point ────────────────────────────────────────────────────────────────

DATASETS = [
    "cora", "citeseer", "pubmed",
    "texas", "wisconsin", "cornell",
    "computers", "cs", "dblp", "physics",
    "film", "squirrel", "chameleon",
    "flickr", "yelp", "reddit",
]


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "t", "yes", "y", "1"):
        return True
    if v.lower() in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Expected true/false for --gcn")


def resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arg == "cuda" and not torch.cuda.is_available():
        print("  WARNING: CUDA requested but not available — falling back to CPU.")
        return torch.device("cpu")
    return torch.device(arg)


def main():
    parser = argparse.ArgumentParser(description="Tri3GN-UGC Graph Coarsening")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Compute device (default: auto)")
    parser.add_argument("--data-root", type=str, default=DATA_ROOT,
                        help="Root directory for PyG datasets")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="Datasets to run (default: all)")
    parser.add_argument("--gcn", type=str2bool, default=False,
                        help="Train a GCN on the coarsened graph and record "
                             "test accuracy (true/false, default: false)")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    summary_dir = os.path.join(script_dir, "summary")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(args.data_root, exist_ok=True)

    fieldnames = ["dataset", "n_nodes", "n_edges", "m_supernodes",
                  "homophily", "alpha_X", "alpha_A", "alpha_AX",
                  "ree", "ree_gamma", "he", "k_eigen", "gcn_acc", "time_s"]

    for name in args.datasets:
        print(f"\n{'='*50}\nDataset: {name}")
        row = {k: "" for k in fieldnames}
        row["dataset"] = name
        try:
            data = load_dataset(name, args.data_root)
            n = data.num_nodes
            e = data.edge_index.shape[1]
            print(f"  Nodes: {n}  Edges: {e}")

            t0 = time.time()
            A, Ac, c, P, m, h, alpha_X, alpha_A, alpha_AX = coarsen(data, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - t0

            print("  Computing REE and HE (UGC spectral_properties) ...")
            X_np = data.x.cpu().numpy().astype(np.float32)
            ree, he, k_eigen = compute_spectral_errors(A, Ac, c, X_np)

            print("  Computing gamma-scaled REE (in-house formula) ...")
            ree_gamma = compute_ree_gamma(A, Ac)
            print("gamma-scaled eigen error", ree_gamma)

            gcn_acc = ""
            if args.gcn:
                print("  Training GCN on coarsened graph ...")
                num_classes = int(data.y.max().item()) + 1
                gcn_acc = train_gcn(data, A, Ac, P, c, m, num_classes, device)
                print(f"  GCN test accuracy: {gcn_acc:.4f}")

            print(f"  Supernodes: {m}  Homophily: {h:.4f}  "
                  f"alpha_X={alpha_X:.4f}  alpha_A={alpha_A:.4f}  "
                  f"alpha_AX={alpha_AX:.4f}  "
                  f"REE: {ree:.4f}  REE_gamma: {ree_gamma:.4f}  "
                  f"HE: {he:.4f}  (k={k_eigen})  "
                  f"Time: {elapsed:.2f}s")

            row.update({
                "n_nodes": n, "n_edges": e, "m_supernodes": m,
                "homophily": round(h, 4),
                "alpha_X":  round(alpha_X,  4),
                "alpha_A":  round(alpha_A,  4),
                "alpha_AX": round(alpha_AX, 4),
                "ree": round(ree, 4),
                "ree_gamma": round(ree_gamma, 4),
                "he":  round(he,  4),
                "k_eigen": k_eigen,
                "gcn_acc": round(gcn_acc, 4) if gcn_acc != "" else "",
                "time_s": round(elapsed, 2),
            })

        except Exception as ex:
            print(f"  ERROR: {ex}")
            row["time_s"] = f"ERROR: {ex}"

        csv_path = os.path.join(results_dir, f"{name}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        print(f"  Saved -> {csv_path}")

        # append to shared summary CSV
        summary_path   = os.path.join(summary_dir, "summary.csv")
        summary_fields = ["dataset", "coarsening_ratio",
                          "alpha_X", "alpha_A", "alpha_AX",
                          "ree", "ree_gamma", "he", "k_eigen",
                          "gcn_acc", "time_s"]
        file_exists    = os.path.isfile(summary_path)
        try:
            with open(summary_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=summary_fields)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    "dataset":          name,
                    "coarsening_ratio": COARSENING_RATIO,
                    "alpha_X":          row["alpha_X"],
                    "alpha_A":          row["alpha_A"],
                    "alpha_AX":         row["alpha_AX"],
                    "ree":              row["ree"],
                    "ree_gamma":        row["ree_gamma"],
                    "he":               row["he"],
                    "k_eigen":          row["k_eigen"],
                    "gcn_acc":          row["gcn_acc"],
                    "time_s":           row["time_s"],
                })
            print(f"  Appended -> {summary_path}")
        except PermissionError:
            print(f"  WARNING: {summary_path} is locked by another program "
                  f"(close it in Excel/IDE). Row not appended; per-dataset "
                  f"CSV in results/ is intact.")


if __name__ == "__main__":
    main()
