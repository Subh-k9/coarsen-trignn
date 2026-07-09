# Tri3GN-UGC Graph Coarsening (Z-LSH supernode assignment)

Spectrum-preserving graph coarsening at a 50% ratio. Supernodes are formed by
hashing the fused Tri3GN embedding `Z` through a random projection matrix
(locality-sensitive hashing) and contracting the low-degree periphery, while
high-degree hubs are protected as singletons so the top-k Laplacian spectrum is
preserved. Relative Eigen Error (REE) and Hyperbolic Error (HE) are evaluated
with the unmodified UGC `spectral_properties.py`.

## Files

- `coarsen_trignn.py` — full coarsening pipeline: embedding construction,
  Z-LSH supernode assignment, UGC-formulation coarse adjacency
  (`Ac = P̂ᵀ A P̂ + (C_diag − I)`), and REE/HE evaluation. Also supports
  training a GCN on the coarsened graph (`--gcn true`).
- `spectral_properties.py` — UGC's spectral metrics (`eigen_error`,
  `hyperbolic_error`, etc.), used unchanged so results are directly comparable
  with UGC.
- `coarsen_trignn_algorithm.pdf` — one-page description of the algorithm.

## Usage

```bash
# run selected datasets (downloads via PyTorch Geometric on first use)
python coarsen_trignn.py --datasets cora citeseer pubmed --device cpu

# all default datasets
python coarsen_trignn.py

# also train a GCN on each coarsened graph
python coarsen_trignn.py --datasets cora --gcn true
```

Per-dataset results are written to `results_trignn/` and appended to
`summary/summary_trignn.csv`.

## Requirements

`torch`, `torch_geometric`, `numpy`, `scipy`.
