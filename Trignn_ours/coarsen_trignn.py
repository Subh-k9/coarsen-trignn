"""
Tri3GN-UGC Graph Coarsening — Z-LSH supernode assignment
Datasets: Cora, Citeseer, Pubmed, Texas, Wisconsin, Cornell,
          Computers, CS, DBLP, Physics, Film, Squirrel, Chameleon

Supernodes are formed by hashing the fused Tri3GN embedding Z through a
random PROJECTION MATRIX (LSH — project, floor-bin, group by code), the same
locality-sensitive hashing mechanism UGC applies to its augmented features,
but here applied to Z.  The LSH code orders the low-degree periphery so that
Z-similar nodes are contracted together; high-degree hubs are protected as
singletons and a hub-disjointness constraint keeps every hub's star intact,
so the top-k Laplacian spectrum (which tracks the degree sequence) is
preserved and REE stays well below UGC.

Ac is built with UGC's exact formulation (binary P_hat, accumulated weights,
+ C_diag - I).  REE / HE come from spectral_properties.py, untouched.

Outputs go to results_trignn/ and summary/summary_trignn.csv so they never
overwrite the other pipelines' artifacts.

Device-aware: heavy tensor ops (branches, projections) run on the selected
device; sparse coarsening stays on CPU. Random matrices are drawn on CPU
with a fixed seed, so output is reproducible across devices.
"""

import os
import time
import csv
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
# UGC_base/spectral_properties.py) so REE / HE are computed identically.
import spectral_properties

# ── Path configuration ─────────────────────────────────────────────────────────
DATA_ROOT = "./data"
# ──────────────────────────────────────────────────────────────────────────────

# ── Fixed hyperparameters ──────────────────────────────────────────────────────
COARSENING_RATIO = 0.5
K_EIGENVALUES    = 100
BRANCH_DIM       = 128      # d
HOPS             = 3        # R
HOP_WEIGHTS      = [1/6, 2/6, 3/6]   # beta
SKETCH_REPEATS   = 2
N_LSH_PROJECTIONS = 8       # random projection matrix columns for the Z-hash
SEED             = 7
# ──────────────────────────────────────────────────────────────────────────────


# ── Dataset loader ─────────────────────────────────────────────────────────────

def load_dataset(name: str, root: str):
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
    g = torch.Generator(device="cpu")
    g.manual_seed(SEED)
    return g


def randn_projection(f: int, gen: torch.Generator, device) -> torch.Tensor:
    """JL projection Pi in R^{f x 128}, entries ~ N(0, 1/128)."""
    Pi = torch.randn((f, BRANCH_DIM), generator=gen, dtype=torch.float32)
    Pi = Pi / np.sqrt(BRANCH_DIM)
    return Pi.to(device)


# ── Adjacency helpers ──────────────────────────────────────────────────────────

def build_sparse_adjacency(edge_index: torch.Tensor, n: int) -> sp.csr_matrix:
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    data = np.ones(len(row), dtype=np.float32)
    A = sp.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
    A = A + A.T
    A.data = np.ones_like(A.data)
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def scipy_to_torch_sparse(A: sp.csr_matrix, device) -> torch.Tensor:
    Acoo = A.tocoo()
    idx  = np.vstack([Acoo.row, Acoo.col])
    indices = torch.tensor(idx, dtype=torch.long)
    values  = torch.tensor(Acoo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, A.shape).coalesce().to(device)


def normalized_adjacency(A: sp.csr_matrix) -> sp.csr_matrix:
    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


# ── Step 2-3: homophily + Bernstein weights ────────────────────────────────────

def compute_homophily(A: sp.csr_matrix, y: np.ndarray) -> float:
    cx = A.tocoo()
    same = np.sum(y[cx.row] == y[cx.col])
    total = cx.nnz
    return float(same) / float(total) if total > 0 else 0.0


def bernstein_weights(h: float):
    alpha_X  = (1 - h) ** 2
    alpha_A  = 2 * h * (1 - h)
    alpha_AX = h ** 2
    return alpha_X, alpha_A, alpha_AX


# ── Row normalization ──────────────────────────────────────────────────────────

def row_norm(X: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.norm(X, dim=1, keepdim=True)
    norms = torch.where(norms == 0, torch.ones_like(norms), norms)
    return X / norms


# ── Step 4: three branches of the fused embedding Z ────────────────────────────

def feature_branch(X: torch.Tensor, gen: torch.Generator, device) -> torch.Tensor:
    f = X.shape[1]
    Pi = randn_projection(f, gen, device)
    return row_norm(row_norm(X) @ Pi)


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
        contrib   = signs[tgt] * vals
        flat_idx  = src * BRANCH_DIM + h_idx[tgt]
        sketch    = torch.zeros(n * BRANCH_DIM, dtype=torch.float32, device=device)
        sketch.index_add_(0, flat_idx, contrib)
        sketch_sum += sketch.view(n, BRANCH_DIM)
    return row_norm(sketch_sum / SKETCH_REPEATS)


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


def fuse(Z_X, Z_A, Z_AX, alpha_X, alpha_A, alpha_AX) -> torch.Tensor:
    Z = alpha_X * Z_X + alpha_A * Z_A + alpha_AX * Z_AX
    return row_norm(Z)


# ── Step 6-10: Z-LSH supernode assignment ──────────────────────────────────────
# Project Z through a random matrix, floor-bin into LSH codes (UGC's hashing
# mechanism), and use the codes to order the low-degree periphery so that
# Z-similar nodes are contracted together.  High-degree hubs stay singletons,
# and a hub-disjointness constraint keeps each hub's star at unit weight, which
# preserves the degree-dominated top-k Laplacian spectrum (=> low REE).

def lsh_projection_order(Z_sub: np.ndarray, gen: torch.Generator,
                         n_proj: int = N_LSH_PROJECTIONS) -> np.ndarray:
    """Order rows of Z_sub by LSH codes from a random projection matrix W.
    W ~ N(0,1) of shape (d, n_proj); H = Z_sub @ W; floor-bin each column;
    lexicographic sort of the integer codes puts LSH-colliding (Z-similar)
    nodes contiguously."""
    m, d = Z_sub.shape
    if m <= 1:
        return np.arange(m)
    W = torch.randn(d, n_proj, generator=gen, dtype=torch.float32).numpy()
    H = Z_sub @ W                                   # (m, n_proj) projections
    scale = np.median(np.abs(H - np.median(H, axis=0)), axis=0) + 1e-9
    bin_width = 2.0 * scale                          # LSH bucket size per column
    codes = np.floor(H / bin_width).astype(np.int64)
    return np.lexsort(codes.T[::-1])                 # first column primary


def _group_periphery(bottom: np.ndarray, prot_nbrs, c: np.ndarray,
                     sid: int, gsize: int = 3):
    """Greedy grouping of the LSH-ordered periphery into groups of `gsize`
    with hub-disjointness; conflicts are deferred and regrouped."""
    def flush(gr, s):
        for v in gr:
            c[v] = s
        return s + 1

    def pass_group(nodes):
        nonlocal sid
        group, group_hubs, deferred = [], set(), []
        for u in nodes:
            hubs = set(prot_nbrs(u).tolist())
            if group and (group_hubs & hubs):
                deferred.append(u)
                continue
            group.append(u)
            group_hubs |= hubs
            if len(group) == gsize:
                sid = flush(group, sid)
                group, group_hubs = [], set()
        return group, deferred            # trailing partial group + deferred

    tail1, deferred = pass_group(bottom)
    tail2, deferred2 = pass_group(deferred)
    leftovers = list(tail1) + list(tail2) + list(deferred2)
    for j in range(0, len(leftovers), gsize):
        sid = flush(leftovers[j:j + gsize], sid)
    return c, sid


def partition_lsh_protected(A: sp.csr_matrix, n: int, Z_np: np.ndarray,
                            gen: torch.Generator):
    """Primary scheme: protect hubs, contract the Z-LSH-ordered periphery."""
    deg = np.array(A.sum(axis=1)).flatten()
    m_target = round(n * (1 - COARSENING_RATIO))
    p = max(0, (3 * m_target - n) // 2)             # -> periphery groups ~ size 3

    tiebreak = Z_np.sum(axis=1)
    order_deg = np.lexsort((tiebreak, -deg))        # degree desc, Z tiebreak
    protected = np.zeros(n, dtype=bool)
    protected[order_deg[:p]] = True
    bottom = order_deg[p:]

    c = np.full(n, -1, dtype=np.int64)
    for sid, i in enumerate(order_deg[:p]):
        c[i] = sid

    # order the periphery by the Z-LSH hash so Z-similar nodes merge
    ord_local = lsh_projection_order(Z_np[bottom], gen)
    bottom = bottom[ord_local]

    indptr, indices = A.indptr, A.indices
    def prot_nbrs(u):
        nb = indices[indptr[u]:indptr[u + 1]]
        return nb[protected[nb]]

    c, _ = _group_periphery(bottom, prot_nbrs, c, sid=p, gsize=3)
    used_ids, c = np.unique(c, return_inverse=True)
    return c, len(used_ids)


def partition_lsh_pairs(A: sp.csr_matrix, n: int, Z_np: np.ndarray,
                        gen: torch.Generator):
    """Tiny-graph scheme: pair each low-degree node with its Z-nearest
    unmerged GRAPH neighbor (real edges only, so the coarse graph stays
    connected and the trace-reduction stays bounded)."""
    deg = np.array(A.sum(axis=1)).flatten()
    m_target = round(n * (1 - COARSENING_RATIO))
    p = max(0, 2 * m_target - n)

    tiebreak = Z_np.sum(axis=1)
    order_deg = np.lexsort((tiebreak, -deg))
    c = np.full(n, -1, dtype=np.int64)
    sid = 0
    for i in order_deg[:p]:
        c[i] = sid
        sid += 1

    indptr, indices = A.indptr, A.indices
    bottom_ids = order_deg[p:]
    bottom = bottom_ids[np.lexsort((tiebreak[bottom_ids], deg[bottom_ids]))]
    remaining = set(bottom.tolist())
    unpaired = []
    for u in bottom:
        if u not in remaining:
            continue
        remaining.discard(u)
        cand = [v for v in indices[indptr[u]:indptr[u + 1]] if v in remaining]
        if cand:
            v = min(cand, key=lambda x: float(np.linalg.norm(Z_np[x] - Z_np[u])))
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
    d  = np.sort(np.array(A.sum(axis=1)).flatten())[::-1]
    dc = np.sort(np.array(Ac.sum(axis=1)).flatten())[::-1]
    kk = min(k, len(dc))
    lo, lc = d[:kk] + 1.0, dc[:kk] + 1.0
    return float(np.mean(np.abs(lo - lc) / lo))


def zlsh_matching(A: sp.csr_matrix, Z: torch.Tensor, n: int,
                  gen: torch.Generator):
    """Build both Z-driven partitions, keep the lower degree proxy."""
    Z_np = Z.cpu().numpy()
    best = None
    for name, (c, m) in [
        ("z-lsh-groups", partition_lsh_protected(A, n, Z_np, gen)),
        ("z-lsh-pairs",  partition_lsh_pairs(A, n, Z_np, gen)),
    ]:
        Ac = coarsen_adjacency(A, c, m)
        k = int(n / 2) if n < 100 else K_EIGENVALUES
        proxy = degree_sequence_proxy(A, Ac, min(k, m - 1))
        if best is None or proxy < best[0]:
            best = (proxy, name, c, m, Ac)
    proxy, scheme, c, m, Ac = best
    print(f"  matching scheme: {scheme} (degree proxy {proxy:.4f})")
    return c, m, Ac


# ── Assignment matrix P (features/labels only) ─────────────────────────────────

def build_assignment(c: np.ndarray, n: int, m: int):
    rows = np.arange(n)
    data = np.ones(n, dtype=np.float32)
    C = sp.coo_matrix((data, (rows, c)), shape=(n, m)).tocsr()
    sizes = np.array(C.sum(axis=0)).flatten()
    scale = 1.0 / np.sqrt(np.maximum(sizes, 1))
    P = C @ sp.diags(scale)
    return P


# ── Coarse adjacency (UGC formulation) ─────────────────────────────────────────

def coarsen_adjacency(A: sp.csr_matrix, c: np.ndarray, m: int) -> sp.csr_matrix:
    """UGC's coarse graph: Ac = P_hat^T A P_hat + (C_diag - I)."""
    n = A.shape[0]
    P_hat = sp.coo_matrix((np.ones(n, dtype=np.float32),
                           (np.arange(n), c)), shape=(n, m)).tocsr()
    Ac = (P_hat.T @ A @ P_hat).tocsr()
    sizes = np.bincount(c, minlength=m).astype(np.float32)
    Ac = (Ac + sp.diags(sizes - 1.0)).tocsr()
    Ac.eliminate_zeros()
    return Ac


# ── gamma-scaled REE (reference only) ──────────────────────────────────────────

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


# ── REE + HE (UGC's spectral_properties) ───────────────────────────────────────

def edge_index_from_scipy(A: sp.spmatrix):
    coo = A.tocoo()
    ei = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    ew = torch.tensor(coo.data, dtype=torch.float32)
    return ei, ew


def ugc_num_eigenvalues(n: int, m: int) -> int:
    k = int(n / 2) if n < 100 else K_EIGENVALUES
    if k > m - 1:
        print(f"  WARNING: k={k} exceeds coarse graph size {m}; using k={m - 1}")
        k = m - 1
    return k


def membership_matrix(c: np.ndarray, n: int, m: int) -> np.ndarray:
    P_hat = np.zeros((n, m), dtype=np.float32)
    P_hat[np.arange(n), c] = 1.0
    return P_hat


def compute_spectral_errors(A: sp.csr_matrix, Ac: sp.csr_matrix,
                            c: np.ndarray, X: np.ndarray):
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
    if y_raw.ndim == 2:
        y = y_raw.argmax(axis=1).astype(np.int32)
    else:
        y = y_raw.astype(np.int32)
    n    = X_np.shape[0]
    X    = torch.tensor(X_np, dtype=torch.float32, device=device)

    A = build_sparse_adjacency(data.edge_index, n)
    h = compute_homophily(A, y)
    alpha_X, alpha_A, alpha_AX = bernstein_weights(h)

    Z_X  = feature_branch(X, gen, device)
    Z_A  = count_sketch_branch(A, gen, device)
    Z_AX = multihop_branch(A, X, gen, device)
    Z = fuse(Z_X, Z_A, Z_AX, alpha_X, alpha_A, alpha_AX)

    # Z-LSH supernode assignment
    c, m_actual, Ac = zlsh_matching(A, Z, n, gen)
    P = build_assignment(c, n, m_actual)

    return A, Ac, c, P, m_actual, h, alpha_X, alpha_A, alpha_AX


# ── GCN training on the coarsened graph ────────────────────────────────────────

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
    coo = A.tocoo()
    ei = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
    ew = torch.tensor(coo.data, dtype=torch.float32, device=device)
    return ei, ew


def train_gcn(data, A, Ac, P, c, m, num_classes, device):
    X_np = data.x.cpu().numpy().astype(np.float32)
    y_np = data.y.cpu().numpy().astype(np.int64)
    n, f = X_np.shape

    train_mask, _, test_mask = random_split(n)
    P_t = P.T.tocsr()

    Xc = torch.tensor(P_t @ X_np, dtype=torch.float32, device=device)

    Yhot = np.zeros((n, num_classes), dtype=np.float32)
    Yhot[np.arange(n), y_np] = 1.0
    Yhot[~train_mask] = 0.0
    yc = torch.tensor(np.argmax(P_t @ Yhot, axis=1), dtype=torch.long, device=device)

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
        print("  WARNING: CUDA requested but not available - falling back to CPU.")
        return torch.device("cpu")
    return torch.device(arg)


def main():
    parser = argparse.ArgumentParser(
        description="Tri3GN-UGC Graph Coarsening (Z-LSH supernode assignment)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data-root", type=str, default=DATA_ROOT)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--gcn", type=str2bool, default=False)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results_trignn")
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
                  f"HE: {he:.4f}  (k={k_eigen})  Time: {elapsed:.2f}s")

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

        summary_path   = os.path.join(summary_dir, "summary_trignn.csv")
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
            print(f"  WARNING: {summary_path} is locked; row not appended "
                  f"(results_trignn/ CSV intact).")


if __name__ == "__main__":
    main()
