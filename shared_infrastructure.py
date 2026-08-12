"""
shared_infrastructure.py — Common data loaders, model definitions, and utilities
for the ProtT5 SAE interpretability pipeline.

All scripts in sparse_bottleneck/ should import from here instead of duplicating
class definitions and path constants.

Output convention: ALL analysis outputs go under /data/ross/interp/ (never under
/data/ross/ppi_lossgain/interaction_loss/ which is for PPI model outputs).
"""
import re
import pickle
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ── Canonical paths ───────────────────────────────────────────────────────────
SB      = Path("/home/rcstewart/ppi_lossgain/sparse_bottleneck")
LA      = Path("/data/ross/interp/latent_analysis")          # all clustering outputs
INTD    = Path("/data/ross/interp")
COMBINED     = Path("/data/ross/ppi_lossgain/interaction_loss/sae_weights/combined")
COMBINED_CACHE = Path("/data/ross/interp/combined_sae_cache")  # z_stab / z_act npz files
STAB_CACHE   = Path("/data/ross/interp/collab_sae_cache")
ACT_CACHE    = Path("/data/ross/interp/activity_sae_cache")
CV_H5    = Path("/data/ross/ppi_lossgain/interaction_loss/clinvar/prott5_subgraphs.h5")
HGMD_H5  = Path("/data/ross/ppi_lossgain/interaction_loss/hgmd/prott5_embeddings.h5")
LABEL_DIR = Path("/data/ross/ppi_lossgain/interaction_loss/home/data_interaction_loss")
ACT_CSV   = Path("/data/ross/assay_calibration/labelseq_dataframe_processed.csv.gz")

# Legacy read-only path (existing cached Z and CSV outputs live here; do not write new files here)
LA_LEGACY = Path("/data/ross/ppi_lossgain/interaction_loss/latent_analysis")

# ── Default model config ──────────────────────────────────────────────────────
DEFAULT_NAME  = "concat_ef1_k128"
DEFAULT_IN_DIM, DEFAULT_EF, DEFAULT_K = 2048, 1, 128
RANDOM_SEED   = 42
N_CLUSTERS    = 50

# ── Variant regex (same as preprocessing) ─────────────────────────────────────
_VAR_RE = re.compile(r'^([A-Z])(\d+)([A-Z])$')
_AA3 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}
_RE_HGVS = re.compile(r'^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$')


# ═════════════════════════════════════════════════════════════════════════════
# 1. Model definition
# ═════════════════════════════════════════════════════════════════════════════

class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder — encode-only version for inference.

    Weights loadable from checkpoints saved by train_combined_topksae.py.
    """
    def __init__(self, in_dim: int, ef: int, k: int):
        super().__init__()
        d = ef * in_dim
        self.k = k
        self.d = d
        self.encoder = nn.Linear(in_dim, d)
        self.decoder = nn.Linear(d, in_dim, bias=False)
        self.register_buffer("b_dec", torch.zeros(in_dim))

    def encode(self, x: torch.Tensor):
        pre = torch.relu(self.encoder(x - self.b_dec))
        topk_vals, topk_idx = pre.topk(self.k, dim=-1, sorted=False)
        z = torch.zeros_like(pre).scatter_(-1, topk_idx, topk_vals)
        return z

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return z, self.decoder(z) + self.b_dec


# ═════════════════════════════════════════════════════════════════════════════
# 2. Decoder / weight loading
# ═════════════════════════════════════════════════════════════════════════════

def load_decoder(name: str = DEFAULT_NAME,
                 in_dim: int = DEFAULT_IN_DIM,
                 ef: int = DEFAULT_EF,
                 k: int = DEFAULT_K):
    """Load SAE checkpoint and return decoder components.

    Returns
    -------
    W_dec_diff : np.ndarray (1024, dict_size) float32
        VT-minus-WT decoder weight matrix: W_dec[1024:] - W_dec[:1024].
        Projects SAE latents into the mutation-effect (diff) reconstruction space.
    b_dec_diff : np.ndarray (1024,) float32
        Corresponding bias difference: b_dec[1024:] - b_dec[:1024].
    W_dec : np.ndarray (in_dim, dict_size) float32
        Full decoder weight matrix.
    b_dec : np.ndarray (in_dim,) float32
        Full decoder bias.
    """
    ckpt_path = COMBINED / f"combined_{name}.pt"
    model = TopKSAE(in_dim, ef, k)
    model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
    model.eval()

    W_dec = model.decoder.weight.detach().numpy().astype(np.float32)  # (in_dim, dict_size)
    b_dec = model.b_dec.detach().numpy().astype(np.float32)           # (in_dim,)
    del model

    half = in_dim // 2
    W_dec_diff = (W_dec[half:] - W_dec[:half]).astype(np.float32)    # (half, dict_size)
    b_dec_diff = (b_dec[half:] - b_dec[:half]).astype(np.float32)    # (half,)
    return W_dec_diff, b_dec_diff, W_dec, b_dec


# ═════════════════════════════════════════════════════════════════════════════
# 3. ClinVar data loading
# ═════════════════════════════════════════════════════════════════════════════

def load_clinvar_data(name: str = DEFAULT_NAME):
    """Load pre-encoded ClinVar Z matrix, labels, and protein IDs.

    Returns
    -------
    Z_cv : scipy.sparse.csr_matrix (227189, dict_size)
    labels : np.ndarray (227189,) int64   0=benign, 1=pathogenic
    prot_ids : np.ndarray (227189,) object  UniProt IDs (protA of each complex)
    """
    # Z matrices: check new LA first, fall back to legacy
    z_path = LA / f"z_cv_{name}.npz"
    if not z_path.exists():
        z_path = LA_LEGACY / f"z_cv_{name}.npz"
    Z_cv     = sp.load_npz(str(z_path))
    labels   = np.load(SB / "clinvar_labels.npy")
    prot_ids = np.load(SB / "clinvar_protein_ids.npy", allow_pickle=True)
    return Z_cv, labels, prot_ids


def load_hgmd_gnomad(name: str = DEFAULT_NAME):
    """Load pre-encoded HGMD and gnomAD Z matrices.

    Returns
    -------
    Z_hg : scipy.sparse.csr_matrix  HGMD variants (all disease-labeled)
    Z_gn : scipy.sparse.csr_matrix  gnomAD variants (all benign-labeled)
    """
    def _load(prefix):
        p = LA / f"{prefix}_{name}.npz"
        if not p.exists():
            p = LA_LEGACY / f"{prefix}_{name}.npz"
        return sp.load_npz(str(p))

    return _load("z_hg"), _load("z_gn")


# ═════════════════════════════════════════════════════════════════════════════
# 4. Phenotype data loading (stability + activity)
# ═════════════════════════════════════════════════════════════════════════════

def load_phenotype_data(name: str = DEFAULT_NAME):
    """Load labeled stability (MegaScale) and activity (DMS) Z matrices with labels.

    Stability labels from MegaScale DDG:
        0 = stabilising  (ddg < -1.0 kcal/mol)
        1 = neutral      (|ddg| < 0.5 kcal/mol)
        2 = destabilising (ddg >= 1.5 kcal/mol)
        -1 = intermediate (excluded from binary probes)

    Activity labels from DMS fitness scores:
        0 = LoF       (score < 0.75)
        1 = wt_like   (0.80 <= score <= 1.20)
        2 = GoF       (score > 1.25)
        -1 = ambiguous (excluded)

    Returns
    -------
    Z_stab   : sparse (N_stab, dict_size)
    y_stab   : np.ndarray (N_stab,) int8   — values in {-1, 0, 1, 2}
    Z_act    : sparse (N_act, dict_size)
    y_act    : np.ndarray (N_act,) int8
    stab_mask: boolean array — rows with valid (non -1) label
    act_mask : boolean array
    """
    # ── Stability ──────────────────────────────────────────────────────────────
    ddg_stab  = np.load(STAB_CACHE / "ddg_valid.npy")
    y_stab    = np.full(len(ddg_stab), -1, dtype=np.int8)
    y_stab[ddg_stab < -1.0]         = 0   # stabilising
    y_stab[np.abs(ddg_stab) < 0.5]  = 1   # neutral
    y_stab[ddg_stab >= 1.5]         = 2   # destabilising
    stab_mask = y_stab >= 0

    Z_stab = sp.load_npz(str(COMBINED_CACHE / f"z_stab_{name}.npz"))

    # ── Activity ───────────────────────────────────────────────────────────────
    valid_idx = np.load(ACT_CACHE / "valid_idx.npy")

    df_act = pd.read_csv(ACT_CSV, compression="gzip")
    df_act = df_act[df_act["assay"] == "activity"].copy()
    df_act = df_act[df_act["variant"].str.match(
        r'^p\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2}$', na=False)]

    def _parse_hgvs(v):
        m = _RE_HGVS.match(v)
        if m is None:
            return None, None, None
        return _AA3.get(m.group(1)), int(m.group(2)), _AA3.get(m.group(3))

    parsed = [_parse_hgvs(v) for v in df_act["variant"]]
    df_act["aa_ref"] = [p[0] for p in parsed]
    df_act["aa_pos"] = [p[1] for p in parsed]
    df_act["aa_alt"] = [p[2] for p in parsed]
    df_act = df_act.dropna(subset=["aa_ref", "aa_pos", "aa_alt"])

    dv = (df_act.groupby(["uniprot_accession", "Gene", "aa_ref", "aa_pos", "aa_alt"])
               ["average score"].mean().reset_index())
    dv.rename(columns={"average score": "score"}, inplace=True)

    def _abin(s):
        if s < 0.75:              return "LoF"
        if 0.80 <= s <= 1.20:    return "wt_like"
        if s > 1.25:             return "GoF"
        return None

    dv["bin"] = dv["score"].map(_abin)
    dv = dv[dv["bin"].notna()].reset_index(drop=True)

    bins = [dv["bin"].tolist()[i] for i in valid_idx]
    y_act = np.full(len(bins), -1, dtype=np.int8)
    y_act[[i for i, b in enumerate(bins) if b == "LoF"]]     = 0
    y_act[[i for i, b in enumerate(bins) if b == "wt_like"]] = 1
    y_act[[i for i, b in enumerate(bins) if b == "GoF"]]     = 2
    act_mask = y_act >= 0

    Z_act = sp.load_npz(str(COMBINED_CACHE / f"z_act_{name}.npz"))

    return Z_stab, y_stab, Z_act, y_act, stab_mask, act_mask


# ═════════════════════════════════════════════════════════════════════════════
# 5. Recon probe training
# ═════════════════════════════════════════════════════════════════════════════

PROBE_TASKS = {
    "destab_vs_neut": (2, 1, "stab"),   # pos_cls, neg_cls, dataset
    "stab_vs_neut":   (0, 1, "stab"),
    "gof_vs_wt":      (2, 1, "act"),
    "lof_vs_wt":      (0, 1, "act"),
}


def train_recon_probes(Z_stab, y_stab, Z_act, y_act,
                       W_dec_diff: np.ndarray, b_dec_diff: np.ndarray,
                       cache_path: Path = None,
                       verbose: bool = True) -> dict:
    """Train four L1-penalised logistic regression probes in reconstruction space.

    Parameters
    ----------
    Z_stab, y_stab, Z_act, y_act : matching Z matrices and labels from load_phenotype_data()
    W_dec_diff : (half_dim, dict_size) — diff decoder weights
    b_dec_diff : (half_dim,)
    cache_path : if given, load from pkl if it exists, otherwise train and save
    verbose    : print AUC per task

    Returns
    -------
    probes : dict  task -> sklearn LogisticRegression (fitted)
    """
    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path, "rb") as f:
            probes = pickle.load(f)
        if verbose:
            print(f"  Probes loaded from {cache_path}")
        return probes

    probes = {}
    Z_by_dataset = {"stab": Z_stab, "act": Z_act}
    y_by_dataset = {"stab": y_stab, "act": y_act}

    for task, (pos_cls, neg_cls, ds) in PROBE_TASKS.items():
        Z_t = Z_by_dataset[ds]
        y_t = y_by_dataset[ds]
        mask_t  = (y_t == pos_cls) | (y_t == neg_cls)
        Z_bin   = Z_t[mask_t]
        y_bin   = (y_t[mask_t] == pos_cls).astype(int)
        xh      = np.asarray(Z_bin.dot(W_dec_diff.T), dtype=np.float32) + b_dec_diff
        clf = LogisticRegression(
            penalty="l1", C=0.1, solver="liblinear",
            class_weight="balanced", max_iter=1000, tol=1e-4)
        clf.fit(xh, y_bin)
        probes[task] = clf
        if verbose:
            auc = roc_auc_score(y_bin, clf.predict_proba(xh)[:, 1])
            print(f"  {task}: AUC={auc:.4f}")

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(probes, f)
        if verbose:
            print(f"  Probes saved → {cache_path}")

    return probes


# ═════════════════════════════════════════════════════════════════════════════
# 6. Reconstruct ClinVar variant keys (for per-variant cluster assignment)
# ═════════════════════════════════════════════════════════════════════════════

def reconstruct_clinvar_variant_keys(cache_path: Path = None):
    """Reconstruct (complex_id, variant_1b) for each row in clinvar_feats.npy.

    Replicates the exact H5 filter from clinvar_sparse_bottleneck_v2.py:
    - Only variants matching [A-Z][0-9]+[A-Z] regex
    - Only variants in pathogenic_set OR benign_set (excluding conflicts)
    - H5 variant positions are 0-based; label sets use 1-based
    - Returns arrays aligned with clinvar_labels.npy row order

    Returns
    -------
    complex_ids : np.ndarray (227189,) — e.g. "P12345_Q67890"
    variant_1b  : np.ndarray (227189,) — e.g. "A123V" (1-based)
    """
    import h5py

    if cache_path is not None and Path(cache_path).exists():
        data = np.load(cache_path, allow_pickle=True)
        return data["complex_ids"], data["variant_1b"]

    def _load_label_set(tsv_path):
        s = set()
        with open(tsv_path) as fh:
            for line in fh:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    s.add((parts[0], parts[1]))
        return s

    pathogenic_set = _load_label_set(LABEL_DIR / "clinvar_pathogenic_dirbind_variants.tsv")
    benign_set     = _load_label_set(LABEL_DIR / "clinvar_benign_dirbind_variants.tsv")
    conflicts      = pathogenic_set & benign_set

    complex_ids = []
    variant_1b  = []
    with h5py.File(CV_H5, "r") as f:
        for complex_id in f.keys():
            interactor_id = complex_id.split('_')[0]
            cgrp = f[complex_id]
            for var_0b in cgrp.keys():
                m = _VAR_RE.match(var_0b)
                if m is None:
                    continue
                ref, pos_0b, alt = m.group(1), int(m.group(2)), m.group(3)
                var_1b_str = f"{ref}{pos_0b + 1}{alt}"
                key = (interactor_id, var_1b_str)
                if key in conflicts:
                    continue
                if key not in pathogenic_set and key not in benign_set:
                    continue
                complex_ids.append(complex_id)
                variant_1b.append(var_1b_str)

    complex_ids = np.array(complex_ids)
    variant_1b  = np.array(variant_1b)

    labels = np.load(SB / "clinvar_labels.npy")
    assert len(complex_ids) == len(labels), (
        f"H5 filter gave {len(complex_ids)} rows but labels has {len(labels)}")

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, complex_ids=complex_ids, variant_1b=variant_1b)

    return complex_ids, variant_1b


def reconstruct_hgmd_variant_keys():
    """Return (prot_ids, variant_strs) for HGMD in the same order as z_hg.

    HGMD H5 key format: '{prot_id} {variant}' for VT entries.
    Only entries where WT key '{prot_id}' also exists are included.
    """
    import h5py

    prot_ids  = []
    var_strs  = []
    with h5py.File(HGMD_H5, "r") as f:
        vt_keys = [k for k in f.keys() if " " in k]
        for vt_key in vt_keys:
            prot_id, var_str = vt_key.split(" ", 1)
            if prot_id in f:
                prot_ids.append(prot_id)
                var_strs.append(var_str)

    return np.array(prot_ids), np.array(var_strs)


# ═════════════════════════════════════════════════════════════════════════════
# 7. Clustering
# ═════════════════════════════════════════════════════════════════════════════

def run_disease_kmeans(Z_path: sp.csr_matrix,
                       n_clusters: int = N_CLUSTERS,
                       random_state: int = RANDOM_SEED,
                       verbose: bool = True):
    """Cosine-normalised MiniBatchKMeans on disease variants.

    Parameters
    ----------
    Z_path : sparse (N_disease, dict_size) — disease variant SAE activations
    n_clusters : k for k-means

    Returns
    -------
    km           : fitted MiniBatchKMeans
    cluster_ids  : np.ndarray (N_disease,) int
    Z_path_norm  : np.ndarray (N_disease, dict_size) float32 — L2-normalised
    """
    if verbose:
        print(f"  Normalising {Z_path.shape[0]:,} variants …")
    Z_norm = normalize(Z_path, norm="l2")

    if verbose:
        print(f"  MiniBatchKMeans k={n_clusters} …")
    km = MiniBatchKMeans(
        n_clusters=n_clusters, batch_size=8192,
        max_iter=300, n_init=5, random_state=random_state)
    cluster_ids = km.fit_predict(Z_norm)

    if verbose:
        counts = np.unique(cluster_ids, return_counts=True)[1]
        print(f"  Cluster sizes: min={counts.min()} median={int(np.median(counts))} max={counts.max()}")

    return km, cluster_ids, Z_norm


def disease_enriched_subspace(Z_path: sp.csr_matrix,
                               enrichment_csv: Path,
                               threshold: float = 0.5):
    """Slice Z to disease-enriched latents only (Path A enrichment > threshold).

    Parameters
    ----------
    Z_path          : sparse (N, dict_size)
    enrichment_csv  : path to latent_enrichment_{name}.csv from Path A
    threshold       : log2 enrichment cutoff (0.5 → 1.4× enriched in disease)

    Returns
    -------
    Z_sub    : sparse (N, K) where K = # enriched latents
    enr_lats : np.ndarray (K,) int — original latent indices
    """
    df_enr    = pd.read_csv(enrichment_csv)
    enr_lats  = df_enr.index[df_enr["enrichment"] > threshold].to_numpy()
    Z_sub     = Z_path[:, enr_lats]
    return Z_sub, enr_lats


# ═════════════════════════════════════════════════════════════════════════════
# 8. Enrichr helper (stateless REST calls)
# ═════════════════════════════════════════════════════════════════════════════

import time, requests

ENRICHR_BASE = "https://maayanlab.cloud/Enrichr"
ENRICHR_GENE_SETS = ["KEGG_2021_Human", "GO_Biological_Process_2023", "Reactome_2022"]


def enrichr_query(gene_list: list, description: str = "query",
                  gene_sets: list = None, adj_p_threshold: float = 0.05,
                  n_top: int = 10) -> dict:
    """Submit gene list to Enrichr and return top enriched terms per gene set.

    Returns
    -------
    dict : gene_set_name -> list of (term, pval, adj_pval, gene_list)
    """
    if len(gene_list) < 3:
        return {}
    if gene_sets is None:
        gene_sets = ENRICHR_GENE_SETS

    try:
        r = requests.post(
            f"{ENRICHR_BASE}/addList",
            files={"list": (None, "\n".join(gene_list)),
                   "description": (None, description)},
            timeout=30)
        if r.status_code != 200:
            return {}
        user_list_id = r.json()["userListId"]
        time.sleep(0.5)
    except Exception:
        return {}

    results = {}
    for gs in gene_sets:
        try:
            r2 = requests.get(
                f"{ENRICHR_BASE}/enrich?userListId={user_list_id}&backgroundType={gs}",
                timeout=30)
            if r2.status_code != 200:
                continue
            data = r2.json().get(gs, [])
            top  = [(d[1], float(d[2]), float(d[6]), d[5])
                    for d in data[:n_top] if d[6] < adj_p_threshold]
            results[gs] = top
        except Exception:
            pass
        time.sleep(0.3)

    return results
