# Tri3GN-UGC Graph Coarsening (Z-LSH supernode assignment)

Spectrum-preserving graph coarsening at a 50% ratio. Supernodes are formed by
hashing the fused Tri3GN embedding `Z` through a random projection matrix
(locality-sensitive hashing) and contracting the low-degree periphery, while
high-degree hubs are protected as singletons so the top-k Laplacian spectrum is
preserved. Relative Eigen Error (REE) and Hyperbolic Error (HE) are evaluated
with the unmodified UGC `spectral_properties.py`.

## Structure

```
UGC_base/
  spectral_properties.py   # UGC's spectral metrics (eigen_error, hyperbolic_error, ...)
Trignn_ours/
  coarsen_trignn.py         # coarsening pipeline (embedding, Z-LSH assignment,
                             # UGC-formulation coarse adjacency, REE/HE, optional GCN)
  spectral_properties.py    # same file as UGC_base/, kept alongside so
                             # coarsen_trignn.py's `import spectral_properties`
                             # resolves when run from this folder
```

## Usage

```bash
cd Trignn_ours

# run selected datasets (downloads via PyTorch Geometric on first use)
python coarsen_trignn.py --datasets cora citeseer pubmed --device cpu

# all default datasets
python coarsen_trignn.py

# also train a GCN on each coarsened graph
python coarsen_trignn.py --datasets cora --gcn true
```

Per-dataset results are written to `Trignn_ours/results_trignn/` and appended
to `Trignn_ours/summary/summary_trignn.csv` (both git-ignored, generated at
runtime — not checked in).

## Requirements

`torch`, `torch_geometric`, `numpy`, `scipy`.
