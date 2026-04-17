# GNN Node Classification on Gene Expression (Luminal A vs Luminal B)

Progetto di classificazione di sottotipi tumorali (Luminal A / Luminal B) a partire da profili di espressione genica, con confronto tra baseline MLP e modelli su grafo (GCN/GAT), inclusa una variante contrastiva con **Cosine Center Loss**.

## Contenuto della repository

```text
.
├── code/
│   ├── main.ipynb              # pipeline completa: EDA → grafo → training → analisi statistica
│   ├── models.py               # MLP, GNN (GCN/GAT), classificatori linear/mlp, loss combinata
│   ├── utils.py                # training loop, early stopping, statistiche, plotting, t-test corretto
│   └── cosine_center_loss.py   # implementazione Cosine Center Loss
├── dataset/
│   └── dataset_LUMINAL_A_B.csv # dataset usato negli esperimenti
├── docs/
│   └── presentation.pdf        # materiale di supporto
├── requirements.txt
└── LICENSE
```

## Dataset

- File: `dataset/dataset_LUMINAL_A_B.csv`
- Campioni: **100**
- Classi: **Luminal A (50)** e **Luminal B (50)** (colonna `l`)
- Feature: espressione genica (colonne ENSG)

## Pipeline sperimentale (notebook `code/main.ipynb`)

1. **Import e configurazione ambiente**
2. **Caricamento dataset e pulizia label**
3. **EDA**:
   - controllo struttura e qualità dati
   - distribuzioni, varianza, sparsity, bilanciamento classi
   - PCA e analisi correlazioni
4. **Costruzione grafo**:
   - trasformazione log2 + filtro geni poco espressi
   - standardizzazione per gene
   - grafo KNN (`k=5`, metrica correlation), conversione in `torch_geometric.data.Data`
   - analisi omofilia globale e locale
5. **Training e confronto modelli**:
   - MLP baseline
   - GCN/GAT con classifier lineare o MLP
   - versioni contrastive con Cosine Center Loss
   - variante con blocco di proiezione (`ProjectedGNN`)
6. **Valutazione statistica**:
   - Repeated Stratified K-Fold (`5 x 20` fold)
   - split interno train/validation
   - metriche: Accuracy e Silhouette (cosine)
   - confronto tra modelli con t-test corretto (Nadeau & Bengio)

## Modelli implementati

- `MLP` (baseline su feature non strutturate)
- `GNN` con:
  - layer `GCN` o `GAT`
  - classificatore `linear` o `mlp`
  - opzione `contrastive=True` per combinare Cross Entropy + Cosine Center Loss
- `ProjectedGNN` (nel notebook): compressione iniziale delle feature prima del backbone GNN

## Requisiti

Dipendenze in `requirements.txt` (PyTorch, PyG, sklearn, pandas, matplotlib, ecc.).

> Nota: il file include il link wheel PyG CPU per Torch 2.8:
> `--find-links https://data.pyg.org/whl/torch-2.8.0+cpu.html`

## Setup rapido

Da root repository:

```bash
python -m venv .venv
source .venv/bin/activate  # su Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Come eseguire

Il workflow principale è il notebook:

```bash
jupyter notebook code/main.ipynb
```

Eseguire le celle in ordine per riprodurre preprocessing, costruzione del grafo, training e analisi finale.

## Note utili

- Il codice è pensato principalmente per esecuzione su **CPU** (configurazione di default nel notebook).
- Le utility di training (`code/utils.py`) usano early stopping (`max_epochs=500`, `patience=20` per default).
- Non è presente una suite di test automatizzati dedicata: la validazione è centrata sugli esperimenti nel notebook.

## Licenza

Progetto rilasciato sotto licenza **MIT**. Vedi `LICENSE`.
