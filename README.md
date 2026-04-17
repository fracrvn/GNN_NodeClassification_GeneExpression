# GNN Node Classification on Gene Expression (Luminal A vs Luminal B)

This project focuses on tumor subtype classification (Luminal A / Luminal B) from gene expression profiles, comparing an MLP baseline with graph-based models (GCN/GAT), including a contrastive variant with **Cosine Center Loss**.

It was developed for the **laboratory component of a university exam**.

## Repository Contents

```text
.
├── code/
│   ├── main.ipynb              # full pipeline: EDA → graph construction → training → statistical analysis
│   ├── models.py               # MLP, GNN (GCN/GAT), linear/mlp heads, combined loss
│   ├── utils.py                # training loop, early stopping, statistics, plotting, corrected t-test
│   └── cosine_center_loss.py   # Cosine Center Loss implementation
├── dataset/
│   └── dataset_LUMINAL_A_B.csv # dataset used in experiments
├── docs/
│   └── presentation.pdf        # supporting material
├── requirements.txt
└── LICENSE
```

## Dataset

- File: `dataset/dataset_LUMINAL_A_B.csv`
- Samples: **100**
- Classes: **Luminal A (50)** and **Luminal B (50)** (column `l`)
- Features: gene expression values (ENSG columns)

## Experimental Pipeline (`code/main.ipynb`)

1. **Environment setup and imports**
2. **Dataset loading and label cleaning**
3. **EDA**:
   - data quality and structure checks
   - distribution, variance, sparsity, class balance
   - PCA and correlation analysis
4. **Graph construction**:
   - log2 transform + low-expression gene filtering
   - gene-wise standardization
   - KNN graph (`k=5`, correlation metric), converted to `torch_geometric.data.Data`
   - global and local homophily analysis
5. **Training and model comparison**:
   - MLP baseline
   - GCN/GAT with linear or MLP classifier heads
   - contrastive variants with Cosine Center Loss
   - projection-based variant (`ProjectedGNN`)
6. **Statistical evaluation**:
   - Repeated Stratified K-Fold (`5 x 20` folds)
   - internal train/validation split
   - metrics: Accuracy and cosine Silhouette
   - model comparison with corrected t-test (Nadeau & Bengio)

## Implemented Models

- `MLP` (baseline on non-graph features)
- `GNN` with:
  - `GCN` or `GAT` layers
  - `linear` or `mlp` classifier head
  - optional `contrastive=True` to combine Cross Entropy + Cosine Center Loss
- `ProjectedGNN` (defined in the notebook): feature compression block before the GNN backbone

## Requirements

Dependencies are listed in `requirements.txt` (PyTorch, PyG, scikit-learn, pandas, matplotlib, etc.).

> Note: the file includes the CPU PyG wheel source for Torch 2.8:
> `--find-links https://data.pyg.org/whl/torch-2.8.0+cpu.html`

## Quick Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## How to Run

The main workflow is the notebook:

```bash
jupyter notebook code/main.ipynb
```

Run cells in order to reproduce preprocessing, graph construction, training, and final analysis.

## Notes

- The code is primarily configured for **CPU** execution (default notebook setup).
- Training utilities in `code/utils.py` use early stopping (`max_epochs=500`, `patience=20` by default).
- There is no dedicated automated test suite; validation is centered on notebook experiments.

## License

This project is released under the **MIT** License. See `LICENSE`.
