# GNN Drug-Target Interaction Predictor & Drug Repurposing Engine

A from-scratch implementation of a graph neural network for drug-target binding affinity prediction, rigorously benchmarked against a classical ECFP+XGBoost baseline, with a live interactive demo and a downstream drug repurposing screening pipeline.

This project was built to deeply understand — not just apply — every component of a molecular GNN pipeline, and to practice honest, rigorous ML evaluation: multi-seed statistics, generalization testing via strict holdouts, and reporting negative/null results alongside positive ones.

**[Live demo →](https://gnn-drug-target-interaction-by-jimit.streamlit.app/)** · [Key findings](#findings--honest-results) · [Setup](#setup)

## Key results

| Model | Pearson R | Concordance Index | Real-binder R |
|---|---|---|---|
| **ECFP + XGBoost** (+ mutation features) | **0.714** | **0.837** | **0.579** |
| GNN (5-seed mean, LR-scheduled) | 0.546 ± 0.013 | 0.781 ± 0.014 | — |
| GNN (5-seed mean, no LR scheduling) | 0.472 ± 0.079 | 0.737 ± 0.054 | 0.306 ± 0.038 |

**The classical ECFP+XGBoost baseline outperforms the from-scratch GNN on this dataset.** This is reported as the honest, primary finding of this project — see [Findings](#findings--honest-results) for why, and what it means. The live demo runs the winning model (XGBoost).

## What this project does

1. **Drug encoder** — SMILES → RDKit → molecular graph → 3-layer GCN with learned attention pooling → fixed-size drug embedding
2. **Protein encoder** — amino acid sequence → embedding → 1D CNN → max pooling → fixed-size protein embedding
3. **Affinity predictor** — concatenated embeddings → MLP → predicted binding affinity (pKd)
4. **Baseline** — ECFP molecular fingerprints + amino-acid-composition protein features, extended with explicit mutation-site features → XGBoost regressor (this is the model served in the live demo)
5. **Repurposing screen** — the trained model scores 1,497 real FDA-approved drugs against a curated panel of 83 oncology-relevant kinases, ranked by predicted affinity
6. **Interactive demo** — a deployed Streamlit app to run live predictions, inspect the model comparison, and browse the repurposing screen results

## Dataset

[Davis kinase binding affinity dataset](https://github.com/thinng/GraphDTA) — 68 drugs × 442 kinase targets (including known resistance-mutation variants like `ABL1(T315I)`, `EGFR(T790M)`), ~30,056 measured pairs, sourced via the GraphDTA repository (same provenance as ChEMBL/BindingDB-derived literature benchmarks; this project's sandbox could not reach ChEMBL/BindingDB's live APIs directly).

**Known dataset property, addressed explicitly in evaluation**: 69.6% of labels sit at a single censoring value (Kd = 10,000 nM, i.e., "no detectable binding"), not a precise measurement. All evaluation in this project reports performance separately for this "ceiling" subset vs. genuine measured binders, since pooled metrics can be misleadingly inflated by the dominant ceiling class.

## Findings & honest results

This section reports what was found, including results that did not support the original hypothesis.

### 1. XGBoost beats the GNN, consistently, across every check performed

Verified across: standard pair-split, strict drug-level holdout (14 unseen drugs), strict protein-level holdout (88 unseen kinases), and 5 independent random seeds. The gap is real and stable, not a fluke of one run.

**Likely explanation**: ECFP fingerprints are a mature, information-dense, hand-engineered representation; the from-scratch GNN had to learn its own representation from ~25K examples with a CPU-constrained architecture (64-dim hidden layers, 8 epochs) — a small-data regime where gradient-boosted trees are well known to be highly competitive with deep learning on tabular/fixed-feature biological data.

### 2. A real, fixable training instability was found and resolved

Initial 5-seed GNN training showed high variance (Pearson R standard deviation = 0.079, with one seed dropping to R=0.32 vs. ~0.50 for the others). Adding step-decay learning rate scheduling (reduce LR by 70% every 4 epochs) resolved this: variance dropped 6× (σ = 0.013) and mean performance improved (R: 0.472 → 0.546). The previously worst-performing seed became the best-performing seed under the new schedule.

### 3. Generalization was explicitly tested, not assumed

Performance was statistically consistent across pair-split, drug-holdout, and protein-holdout evaluation, indicating the model learns transferable structure–affinity patterns rather than memorizing the training grid.

### 4. Explicit mutation features partially fixed a real model blind spot

The baseline's amino-acid-composition protein features could not distinguish point-mutant variants (e.g., `FLT3(D835H)` vs `FLT3(D835Y)` received identical predictions). Adding explicit (position, original residue, mutant residue) features fixed this for point mutations (small but real gain: R 0.709 → 0.714) but did **not** improve results for insertion-type mutations (e.g., `FLT3-ITD`) — reported as a null result, not papered over.

### 5. The repurposing screen found no validated novel candidates

1,497 FDA-approved drugs were screened against 83 curated oncology kinases. Top hits were dominated by drugs already approved as kinase inhibitors for those exact targets (dasatinib→ABL1, sunitinib→KIT) — a useful sanity check (the model recovers known biology) but not a novel finding. After excluding known kinase inhibitors, four candidates were flagged (Josamycin, Eribulin, Tacrolimus, Nevirapine); a literature check found no corroborating evidence of kinase activity for any of them. Given the baseline's known false-positive rate (~34% even at its best precision threshold), these are reported as **unvalidated** — not a discovery.

## Repository structure

```
.
├── app.py                                 # Streamlit interactive demo (live link above)
├── packages.txt                           # System-level (apt) dependencies for cloud deployment
├── requirements.txt                       # Python dependencies for cloud deployment
├── app_data/                               # Minimal Davis data slice the deployed app needs
│   ├── ligands_can.txt
│   └── proteins.txt
├── notebooks/
│   ├── 01_molecule_to_graph.ipynb         # SMILES → RDKit → graph featurization, built step by step
│   ├── 02_protein_to_vector.ipynb         # Protein sequence encoder, full model, first training run
│   └── 03_multi_seed_evaluation.ipynb     # Multi-seed rigor, precision/recall, mutation features,
│                                            generalization tests, LR scheduling fix, repurposing screen
├── models/
│   ├── xgb_model_v2.json                  # Trained baseline model (loaded by app.py)
│   ├── *_predictions.json                 # Saved GNN predictions per seed (reproducible without retraining)
│   └── multi_seed_results.json / lr_scheduled_results.json
└── README.md
```

## Setup (local)

```bash
git clone https://github.com/JP-Bro/gnn-drug-target-interaction.git
cd gnn-drug-target-interaction

python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

pip install torch torch_geometric rdkit scikit-learn xgboost mlflow jupyter pandas scipy streamlit

git clone --depth 1 https://github.com/thinng/GraphDTA.git data/GraphDTA
```

**Important — Python version**: use Python 3.12, not 3.14. At time of writing, Python 3.14 is too new for several dependencies to have pre-built wheels (notably `pyarrow`, an MLflow dependency), forcing a from-source build that requires `cmake` and a C++ toolchain most machines don't have installed. Python 3.12.10 (the last version with an official Windows binary installer) avoids this entirely.

**Important — virtual environment discipline**: every command (`pip install`, `jupyter notebook`, `streamlit run`) must be run from a terminal with the `venv` activated (prompt shows `(venv)`). Running Jupyter or Streamlit without activating venv first will silently use your global Python installation instead, causing `ModuleNotFoundError`s for packages that are actually installed — just in the wrong place. Check which interpreter is actually running with `import sys; print(sys.executable)` if anything seems off.

```bash
streamlit run app.py
```

## Deployment

Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud). Two non-obvious dependency layers were required beyond a standard `requirements.txt`:

- **System libraries** (`packages.txt`, apt-installed) — RDKit's drawing module (`rdMolDraw2D`) requires X11 rendering libraries (`libxrender1`, `libxext6`, `libsm6`, `libice6`) that aren't present in the base cloud container by default, even though the app only generates SVGs and never displays on-screen graphics.
- **Transitive Python dependency** — `xgboost.XGBRegressor` is a scikit-learn–compatible wrapper and requires `scikit-learn` to be importable even when only `.predict()` is called; this isn't obvious from `app.py`'s own import statements and must be listed explicitly in `requirements.txt`.

## Tooling

PyTorch · PyTorch Geometric · RDKit · scikit-learn · XGBoost · MLflow · Streamlit (deployed on Streamlit Community Cloud)

## Limitations (explicit)

- CPU-only training; GNN architecture and epoch budget were constrained accordingly
- No pretrained protein language model (e.g., ESM) was used — amino-acid composition and CNN features are comparatively coarse, and this is the main reason mutation-site features had to be added explicitly rather than learned
- Repurposing candidate validation relied on general web search, not a systematic bioactivity database query (e.g., ChEMBL/PubChem BioAssay)
- Single train/test fold per split strategy (no nested cross-validation)
- Repurposing drug list (1,497 compounds) sourced from a 2013 DrugBank-derived export; does not include drugs approved after that date
- The live demo serves the XGBoost baseline only; the GNN was not wired into the deployed app since it underperforms the baseline and depends on the heavier PyTorch/PyTorch Geometric stack

## License

MIT
