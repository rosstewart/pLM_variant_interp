# pLM Variant Interpretation via Sparse Autoencoders

Interpretability pipeline for protein missense variants using sparse autoencoders (SAEs)
trained over ProtT5-XL residue embeddings. Variants are projected into a sparse
high-dimensional feature space; individual SAE features are then tested for enrichment
in phenotypically distinct variant bins (stability, functional activity) and ranked by
Spearman correlation and logistic regression weights.

---

## Environment

```bash
conda activate ppi   # /home/rcstewart/miniconda3/envs/ppi/
# Python 3.10, torch 2.5.1+cu121
# GPU: cuda:2
```

Convert any `.py` jupytext source to a notebook:
```bash
/home/rcstewart/miniconda3/bin/jupytext --to ipynb <file>.py
```

---

## Notebooks

### 1. `clinvar_sparse_bottleneck_v2.py` / `.ipynb`

**Purpose:** Train SAE models on ClinVar pathogenicity features; encode MegaScale
stability variants; run enrichment, Spearman, and permutation tests across models.

**Key steps:**
- Load ClinVar ΔΔG-labeled variants from H5 (ProtT5 final-layer embeddings)
- Train 9 model designs (D0–D8, see table below)
- Load MegaScale (~271k variants, 298 proteins) from `preprocessed.pkl`
- Encode MegaScale variants through each trained model
- Test for enrichment in ΔΔG bins; compute Spearman ρ with ΔΔG

**Data:**
- ClinVar H5: `/data/ross/ppi_lossgain/interaction_loss/home/clinvar/...`
- MegaScale: `/data/ross/ppi_lossgain/interaction_loss/megascale_preprocessed/preprocessed.pkl`

---

### 2. `clinvar_collab_sae.py` / `.ipynb`

**Purpose:** Apply a pre-trained collaborator residue-level SAE (ProtT5 layer-20,
trained on UniRef) to MegaScale stability variants. Compute ΔZ = z_VT − z_WT and
test enrichment of gained/lost features in stability bins.

**Key steps:**
- Extract ProtT5 layer-20 hidden states at the mutated residue (WT and VT)
  — one forward pass per protein for WT; batched VT passes
- Encode through collaborator SAE → ΔZ_pos = max(ΔZ, 0), ΔZ_neg = max(−ΔZ, 0)
- Run sparse GPU permutation test + Spearman correlation

**Data / cache:**
- SAE weights: `/data/karna/model_weights/sae_weights/t5/trainer_0/t5_layer20_topk256_ef16.pt`
- Cache: `/data/ross/interp/collab_sae_cache/`

---

### 3. `clinvar_activity_sae.py` / `.ipynb`

**Purpose:** Apply D5 and the collaborator SAE to DMS functional activity variants
(17 human genes). Bins: LoF (<0.75), wt-like (0.80–1.20), GoF (>1.25).

**Key steps:**
- Load `/data/ross/assay_calibration/labelseq_dataframe_processed.csv.gz`
- Filter to `assay == 'activity'`, missense only; average scores across treatments
- Fetch canonical UniProt sequences; verify WT residue at variant position
- Extract ProtT5 layer-20 and final-layer embeddings in a single forward pass per batch
- Encode through D5 (WT+VT final-layer concat → 8192-dim) and collab SAE (ΔZ at layer-20)
- Enrichment (GoF vs wt-like, LoF vs wt-like) + Spearman with activity score

**Data / cache:**
- Activity CSV: `/data/ross/assay_calibration/labelseq_dataframe_processed.csv.gz`
- Cache: `/data/ross/interp/activity_sae_cache/`

---

### 4. `sae_probing_analysis.py` / `.ipynb`

**Purpose:** L1-regularized logistic regression (probing classifiers) using SAE sparse
latents as features. Evaluates how well individual SAE features predict phenotype class,
using leave-one-protein-out (LOPO) cross-validation. Ranks latents by regression weight
for biological interpretation.

**Probing tasks (per model):**
- Stability (collab SAE ΔZ, D5): 3-class (stab/neutral/destab) + binary + ΔΔG regression
- Activity (collab SAE ΔZ, D5): 3-class (LoF/wt-like/GoF) + binary + score regression
- Baseline comparison: L2 logistic regression on raw ProtT5 features (1024-dim)

**L1 C values:** 1, 4, 16, 64, 256 (lower C = stronger sparsity)

---

## Model Designs

| Name | Class | in_dim | dict_size | k | Input |
|------|-------|--------|-----------|---|-------|
| D0 | SparseBNClassifier | 1024 | 256 | — | VT−WT diff (final layer) |
| D1 | SparseBNClassifier | 2048 | 256 | — | WT+VT concat (final layer) |
| D2 | SupervisedSAE | 2048 | 256 | — | WT+VT concat |
| D3 | UnsupervisedSAE | 2048 | 256 | — | WT+VT concat |
| D4 | UnsupervisedSAE | 1024 | 256 | — | VT only (final layer) |
| **D5** | **TopKSAE** | **2048** | **8192** | **128** | **WT+VT concat (final layer)** |
| D6 | SupervisedTopKSAE | 2048 | 8192 | 128 | WT+VT concat |
| D7 | TopKSAE | 1024 | 4096 | 64 | VT−WT diff |
| D8 | SupervisedTopKSAE | 1024 | 4096 | 64 | VT−WT diff |
| **Collab SAE** | **AutoEncoderTopK** | **1024** | **16384** | **256** | **Layer-20 residue embedding** |

D5 is the primary unsupervised model used for cross-dataset inference.

---

## Variant Bins

**Stability (ΔΔG, kcal/mol):**
| Bin | Threshold |
|-----|-----------|
| Highly stabilizing | ΔΔG < −1.0 |
| Mildly stabilizing | −1.0 ≤ ΔΔG < −0.5 |
| Near neutral | \|ΔΔG\| < 0.5 |
| Mildly destabilizing | 0.5 ≤ ΔΔG < 1.5 |
| Highly destabilizing | ΔΔG ≥ 1.5 |

**Functional activity (normalized score):**
| Bin | Threshold |
|-----|-----------|
| LoF | score < 0.75 |
| *(gap excluded)* | 0.75–0.80 |
| wt-like | 0.80 ≤ score ≤ 1.20 |
| *(gap excluded)* | 1.20–1.25 |
| GoF | score > 1.25 |

---

## ProtT5 Extraction

- Model: `Rostlab/prot_t5_xl_half_uniref50-enc` (T5-XL, 24 encoder blocks)
- Layer indexing: `hidden_states[L+1]` = output of block L (0-based)
  - Layer 20 → `hidden_states[21]` (collab SAE input)
  - Final layer (23) → `hidden_states[24]` (D5 input)
- WT: **one forward pass per protein** (all positions from single pass)
- VT: batched by position in groups of 16

---

## Cache Locations

| Path | Contents |
|------|----------|
| `/data/ross/interp/collab_sae_cache/` | layer20_wt/vt.npy, dz_pos/neg.npy, valid_mask.npy, protein_ids_valid.npy, ddg_valid.npy |
| `/data/ross/interp/activity_sae_cache/` | layer20_wt/vt.npy, final_layer_wt/vt.npy, z_d5.npy, dz_pos/neg.npy, valid_idx.npy, protein_ids.npy |
| `/data/ross/interp/ms_z_d5_sparse.npz` | MegaScale D5 encodings (N_ms, 8192) — scipy sparse CSR |
| `/data/ross/interp/ms_x_diff.npy` | MegaScale final-layer mutation diff (N_ms, 1024) |
| `sparse_bottleneck/ms_ddg.npy` | MegaScale ΔΔG labels (N_ms,) |
| `sparse_bottleneck/ms_protein_ids.npy` | MegaScale protein IDs per variant (N_ms,) |
| `sparse_bottleneck/v2_model_d5_topk.pt` | D5 TopKSAE weights |
| `sparse_bottleneck/v2_model_d6_sup_topk.pt` | D6 weights |
| `sparse_bottleneck/v2_model_d7_topk_diff.pt` | D7 weights |
| `sparse_bottleneck/v2_model_d8_sup_topk_diff.pt` | D8 weights |
