# MutPred-PPI: Sparse Bottleneck / SAE Interpretability

Interpretability analyses for the MutPred-PPI model using Sparse Autoencoders (SAEs).
SAE latents are used to discover and validate disease mechanisms in an unsupervised way.

## Setup

```bash
conda activate ppi
# All outputs → /data/ross/interp/  (never /data/ross/ppi_lossgain/interaction_loss/)
```

## Repository layout

```
shared_infrastructure.py       ← common loaders, TopKSAE, probe training (imported by discovery/)
training/
  train_combined_topksae.py    ← SAE training
caching/
  clinvar_sparse_bottleneck_v2.py   ← encode ClinVar / gnomAD / HGMD → sparse Z matrices
  save_ms_d5_cache.py               ← encode MegaScale stability variants
  patch_activity_cache.py           ← reconstruct missing protein_ids in activity cache
probing/
  sae_probing_analysis.py      ← L1 logistic probes + LOPO cross-validation
patching/
  sae_activation_patching_v2.py    ← per-latent patch / inject experiments
discovery/
  unsupervised_latent_analysis.py  ← 5a: per-latent disease enrichment (Path A)
  disease_mechanism_analysis.py    ← 5b: disease-only k-means clustering (Path B)
  validate_disease_clusters.py     ← 5c: gene names + Enrichr + condition counts
  cluster_validation_suite.py      ← 5d: full cluster validation (Sections 0B–5)
  within_family_analysis.py        ← within-family pathogenic vs. benign analysis
archive/                       ← superseded v1 scripts and notebooks
```

---

## Output path convention

```
/data/ross/interp/
  latent_analysis/                  ← clustering, enrichment, UMAP outputs
    validation/                     ← cluster_validation_suite outputs
      within_family/                ← within_family_analysis outputs
  combined_sae_cache/               ← z_stab / z_act npz (labeled phenotype variants)
  collab_sae_cache/                 ← ddg_valid.npy, valid_mask.npy, protein_ids_valid.npy
  activity_sae_cache/               ← valid_idx.npy, activity variant cache
  patching_results/                 ← activation_patching_results_v2.csv
  probing_results/                  ← probe output CSVs

/data/ross/ppi_lossgain/interaction_loss/
  sae_weights/combined/             ← trained SAE checkpoints (canonical named files)
```

---

## Shared infrastructure (`shared_infrastructure.py`)

Import from here instead of duplicating logic. All discovery scripts add their parent
directory to `sys.path` and `from shared_infrastructure import ...`.

| Symbol | What it returns |
|--------|----------------|
| `TopKSAE` | model class — load checkpoints with `TopKSAE.load(path)` |
| `load_decoder(name)` | `(W_dec_diff, b_dec_diff, W_dec, b_dec)` |
| `load_clinvar_data(name)` | `(Z_cv, labels, prot_ids)` — all ClinVar rows (pathogenic + benign) |
| `load_hgmd_gnomad(name)` | `(Z_hg, Z_gn)` |
| `load_phenotype_data(name)` | `(Z_stab, y_stab, Z_act, y_act, stab_mask, act_mask)` |
| `train_recon_probes(...)` | `dict[task → LogisticRegression]` with pickle caching |
| `reconstruct_clinvar_variant_keys(cache_path)` | `(complex_ids, variant_1b)` aligned to Z_cv rows |
| `run_disease_kmeans(Z, n_clusters, random_state)` | `(km, cluster_ids, Z_norm)` cosine-normalised MiniBatchKMeans |
| `disease_enriched_subspace(Z, enrichment_csv, threshold)` | `(Z_sub, enr_lats)` |
| `enrichr_query(gene_list, ...)` | `dict[gene_set → top terms]` REST API |

Key constant: `W_dec_diff = W_dec[1024:] - W_dec[:1024]`
(VT − WT decoder component; projects SAE latents into mutation-effect reconstruction space).

---

## 1. SAE Training (`training/`)

### `train_combined_topksae.py`

Trains 6 TopK SAEs on concatenated ClinVar + gnomAD + HGMD (901k variants).

**Prerequisites:**
- `combined_wt.npy` and `combined_vt.npy` in `/data/ross/ppi_lossgain/interaction_loss/sae_weights/combined/`
  — ProtT5 embeddings for all 901k variants, stacked in order (ClinVar → gnomAD → HGMD)

**Outputs:** `/data/ross/ppi_lossgain/interaction_loss/sae_weights/combined/combined_{name}.pt`

```bash
python -u training/train_combined_topksae.py > training/train.log 2>&1 &
```

Focus model: `concat_ef1_k128` (EF=1, K=128, dict_size=2048, in_dim=2048).

---

## 2. Feature Caching (`caching/`)

Run once after training. All three scripts are independent of each other.

### `clinvar_sparse_bottleneck_v2.py`

Encodes ClinVar + gnomAD + HGMD variants through the SAE encoder; saves sparse Z matrices.

**Prerequisites:**
- Trained SAE checkpoint: `sae_weights/combined/combined_{name}.pt`
- ClinVar variant H5 file: built by the PPI graph pipeline (path inside script)
- gnomAD and HGMD H5 files: same pipeline

**Outputs (per model name `{name}`):**
- `/data/ross/interp/latent_analysis/z_cv_{name}.npz` — ClinVar (pathogenic + benign), sparse (227189, 2048)
- `/data/ross/interp/latent_analysis/z_gn_{name}.npz` — gnomAD, sparse (653465, 2048)
- `/data/ross/interp/latent_analysis/z_hg_{name}.npz` — HGMD, sparse (~7k, 2048)
- `clinvar_labels.npy` — 0=benign, 1=pathogenic, aligned to z_cv rows
- `clinvar_protein_ids.npy` — UniProt ID per row

```bash
python -u caching/clinvar_sparse_bottleneck_v2.py > caching/cache.log 2>&1 &
```

### `save_ms_d5_cache.py`

Encodes MegaScale stability variants (298 proteins, ~230k variants) through the SAE.

**Prerequisites:**
- Trained SAE checkpoint: `sae_weights/combined/combined_{name}.pt`
- MegaScale ΔΔG data: `/data/ross/interp/` (DDG values + sequences)

**Outputs:** `/data/ross/interp/combined_sae_cache/z_stab_{name}.npz`

```bash
python -u caching/save_ms_d5_cache.py > caching/cache_stab.log 2>&1 &
```

### `patch_activity_cache.py`

Reconstructs missing `protein_ids.npy` in the activity cache (one-time fix).

**Prerequisites:**
- Activity cache directory: `/data/ross/interp/activity_sae_cache/`
- DMS variant TSV with gene annotations

**Outputs:** `/data/ross/interp/activity_sae_cache/protein_ids.npy`

```bash
python -u caching/patch_activity_cache.py
```

---

## 3. Probing (`probing/`)

### `sae_probing_analysis.py`

L1-regularized logistic probes on SAE latents and reconstruction space. Compares sparse
latent space vs. recon space vs. raw ProtT5 baseline. Includes LOPO cross-validation.

**Prerequisites:**
- `z_cv_{name}.npz` from `clinvar_sparse_bottleneck_v2.py`
- `z_stab_{name}.npz` from `save_ms_d5_cache.py`
- Activity cache: `/data/ross/interp/activity_sae_cache/`
- `clinvar_labels.npy`, `clinvar_protein_ids.npy`

**Outputs:** `/data/ross/interp/probing_results/`

```bash
python -u probing/sae_probing_analysis.py > probing/probing.log 2>&1 &
```

Published AUC (concat_ef1_k128, reconstruction space):

| Task | AUC |
|------|-----|
| destab_vs_neutral | 0.951 |
| stab_vs_neutral | 0.904 |
| GoF_vs_wt | 0.673 |
| LoF_vs_wt | 0.692 |

---

## 4. Activation Patching (`patching/`)

### `sae_activation_patching_v2.py`

Per-latent patch (zero) and inject (fill to mean positive-class value) for all 6 combined
TopK SAEs. Measures effect on recon probe scores and sparse probe scores.

**Prerequisites:**
- All 6 trained SAE checkpoints in `sae_weights/combined/`
- `z_cv_{name}.npz` and `z_stab_{name}.npz` from caching scripts
- Trained probes (generated on first run of `sae_probing_analysis.py`, or cached pkl)
- `clinvar_labels.npy`, `clinvar_protein_ids.npy`

**Outputs:** `/data/ross/interp/patching_results/activation_patching_results_v2.csv`

```bash
python -u patching/sae_activation_patching_v2.py > patching/patching.log 2>&1 &
```

---

## 5. Unsupervised Disease Mechanism Discovery (`discovery/`)

Run in order: **5a → 5b → 5c → 5d**, then within-family. All outputs go to
`/data/ross/interp/latent_analysis/` and subfolders.

### 5a. Per-latent disease enrichment — `unsupervised_latent_analysis.py`

Computes log2 fire-rate ratio (disease vs. benign) for each of the 2048 SAE latents.
Also runs UMAP on the full ClinVar variant set.

**Prerequisites:**
- `z_cv_{name}.npz` from `clinvar_sparse_bottleneck_v2.py`
- `clinvar_labels.npy`

**Outputs:**
- `latent_enrichment_{name}.csv` — per-latent log2 enrichment + fire rates (**required by 5b–5d**)
- `umap_clinvar_{name}.png`

```bash
python -u discovery/unsupervised_latent_analysis.py > discovery/run_enrichment.log 2>&1 &
```

### 5b. Disease-only clustering — `disease_mechanism_analysis.py`

k-means k=50 on 186k pathogenic variants (ClinVar pathogenic + HGMD), cosine-normalised.
Annotates clusters with top latents, probe deltas, and decoder reconstruction effects.

**Prerequisites:**
- `z_cv_{name}.npz` and `z_hg_{name}.npz` from `clinvar_sparse_bottleneck_v2.py`
- `clinvar_labels.npy`
- Trained recon probes (cached pkl, or re-trains automatically via `shared_infrastructure`)

**Outputs:**
- `disease_clusters.csv` — cluster ID, n_variants, top latents, probe deltas (**required by 5c, 5d**)
- `module_annotations_{name}.csv`
- `umap_disease_clusters_*.png`

```bash
python -u discovery/disease_mechanism_analysis.py > discovery/run_disease.log 2>&1 &
```

Design choices:
- Disease-only clustering: prevents neutral majority dominating centroids
- Cosine normalisation: variants with few firing latents don't cluster by magnitude
- k=50: coarse on purpose — each cluster = a distinct mechanism, not fine subtypes

### 5c. Cluster annotation — `validate_disease_clusters.py`

Annotates each cluster with UniProt gene names (batch API), Enrichr pathway enrichment
(KEGG / GO / Reactome), and raw ClinVar condition counts.

**Prerequisites:**
- `disease_clusters.csv` from `disease_mechanism_analysis.py`
- `z_cv_{name}.npz` and `clinvar_labels.npy`
- Network access for UniProt + Enrichr REST APIs

**Outputs:**
- `cluster_validation_report.csv`
- `cluster_enrichr_results.json`

Note: ClinVar condition counts are gene-abundance-biased. Use Section 5 of
`cluster_validation_suite.py` (Fisher's exact) for corrected condition enrichment.

```bash
python -u discovery/validate_disease_clusters.py > discovery/run_validate.log 2>&1 &
```

### 5d. Comprehensive validation — `cluster_validation_suite.py`

Full validation infrastructure. Imports from `shared_infrastructure.py`.

**Prerequisites:**
- `z_cv_{name}.npz`, `z_gn_{name}.npz`, `z_hg_{name}.npz` from `clinvar_sparse_bottleneck_v2.py`
- `z_stab_{name}.npz` from `save_ms_d5_cache.py`
- `clinvar_labels.npy`, `clinvar_protein_ids.npy`
- `latent_enrichment_{name}.csv` from `unsupervised_latent_analysis.py`
- `disease_clusters.csv` from `disease_mechanism_analysis.py`
- ClinVar variant_summary.txt.gz (for Section 5 condition enrichment)
- Network access for Enrichr REST API (Sections 0B, 3)

**Outputs:** `/data/ross/interp/latent_analysis/validation/`

```bash
# All sections (~30–60 min with Enrichr calls)
python -u discovery/cluster_validation_suite.py \
  > /data/ross/interp/latent_analysis/validation/run_validation.log 2>&1 &

# Specific sections (faster)
python -u discovery/cluster_validation_suite.py --sections 1,2
```

| Section | What it does | Key output |
|---------|-------------|------------|
| 0B | Approach A (full-space) vs B (disease-enriched 309-latent subspace) gating | `approach_comparison_summary.txt` |
| 1 | KS test + Mann-Whitney for probe scores per cluster | `probe_distribution_stats.csv` |
| 2 | fire_in/fire_out latent specificity + causal probe shift | `latent_specificity.csv` |
| 3 | Leave-one-gene-out centroid stability + Enrichr re-query | `leave_one_gene_out.csv` |
| 4 | Residualized (protein-agnostic) clustering + Jaccard overlap | `residualized_vs_original_summary.csv` |
| 5 | Fisher's exact test for ClinVar condition enrichment | `condition_enrichment_fisher.csv` |

**Decision (Section 0B):** Approach B adopted — 309-latent disease-enriched subspace gives
higher fire_in/out specificity (p=0.005) and higher mean disease enrichment per cluster (p≈0).

### Within-family analysis — `within_family_analysis.py`

Validates that cluster-defining latents fire specifically on pathogenic variants *within*
the same protein family, not on ClinVar Benign variants of the same protein.

**Prerequisites:**
- `z_cv_{name}.npz` and `z_gn_{name}.npz` from `clinvar_sparse_bottleneck_v2.py`
- `clinvar_labels.npy`, `clinvar_protein_ids.npy`
- `latent_enrichment_{name}.csv` from `unsupervised_latent_analysis.py`

**Outputs:** `/data/ross/interp/latent_analysis/validation/within_family/`

```bash
mkdir -p /data/ross/interp/latent_analysis/validation/within_family
python -u discovery/within_family_analysis.py \
  > /data/ross/interp/latent_analysis/validation/within_family/run.log 2>&1 &
```

Key result: all major cluster-defining latents fire on 0% of ClinVar Benign variants of
the same protein (∞ within-family enrichment, Fisher p=0), confirming they encode
pathomechanisms rather than protein identity.

---

## Key design decisions

**`W_dec_diff`** — Reconstruction space is `xh = Z @ W_dec_diff.T + b_dec_diff` where
`W_dec_diff = W_dec[1024:] - W_dec[:1024]`. The two SAE decoder halves correspond to WT
and VT ProtT5 embeddings concatenated as input; their difference isolates mutation effect.

**OOD contamination replaced by fire_in/out** — Early validation used fraction of benign
variants assigned to a pathogenic centroid. Replaced in Section 2 with the fire_in/fire_out
ratio, which is fully in-distribution.

**H5 position convention** — ClinVar H5 keys use 0-based variant positions; label TSV
files use 1-based. Always apply `pos_1b = pos_0b + 1` when matching. See
`reconstruct_clinvar_variant_keys()` in `shared_infrastructure.py`.

**Residualized clustering** — Subtracting per-protein mean Z before clustering disentangles
ProtT5 sequence-context similarity from mechanism signal. Clusters dissolve (Jaccard < 0.025)
after residualization, confirming protein identity is the primary clustering axis — but
within-family analysis (above) confirms mechanism-specific latents exist independently.
