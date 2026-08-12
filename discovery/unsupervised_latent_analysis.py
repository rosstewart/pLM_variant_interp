"""
unsupervised_latent_analysis.py

Path A: Per-latent disease enrichment score
  - ClinVar (pathogenic/benign), gnomAD (benign), HGMD (disease)
  - log2 enrichment per latent across all 6 combined SAE models
  - Cross-reference with activation_patching_results_v2.csv injection ΔAUCs
  - Output: latent_enrichment_{model}.csv, enrichment_scatter_{model}.png

Path B: Sparse clustering of phenotype-labeled variants
  - Mini-batch k-means (cosine) on existing z_stab / z_act caches
  - Per-cluster: phenotype composition + dominant latents
  - Cross-reference dominant latents with Path A disease enrichment
  - UMAP coloured by cluster / phenotype
  - Output: cluster_summary_{model}.csv, umap_{model}_{task}.png

All large outputs → /data/ross/interp/latent_analysis/
"""

import warnings; warnings.filterwarnings("ignore")
import re, sys, pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score

# ── Paths ──────────────────────────────────────────────────────────────────────
SB            = Path("/home/rcstewart/ppi_lossgain/sparse_bottleneck")
COMBINED      = Path("/data/ross/ppi_lossgain/interaction_loss/sae_weights/combined")
COMBINED_CACHE= Path("/data/ross/interp/combined_sae_cache")
STAB_CACHE    = Path("/data/ross/interp/collab_sae_cache")
OUT_DIR       = Path("/data/ross/interp/latent_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLINVAR_H5  = Path("/data/ross/ppi_lossgain/interaction_loss/clinvar/prott5_subgraphs.h5")
GNOMAD_H5   = Path("/data/ross/ppi_lossgain/interaction_loss/gnomad/prott5_subgraphs.h5")
HGMD_H5     = Path("/data/ross/ppi_lossgain/interaction_loss/hgmd/prott5_embeddings.h5")

GNOMAD_WT_CACHE = OUT_DIR / "gnomad_wt.npy"
GNOMAD_VT_CACHE = OUT_DIR / "gnomad_vt.npy"
HGMD_WT_CACHE   = OUT_DIR / "hgmd_wt.npy"
HGMD_VT_CACHE   = OUT_DIR / "hgmd_vt.npy"

PATCH_CSV   = INTD / "patching_results/activation_patching_results_v2.csv"
MEGA_PKL    = "/data/ross/ppi_lossgain/interaction_loss/megascale_preprocessed/preprocessed.pkl"
ACT_CACHE   = Path("/data/ross/interp/activity_sae_cache")
ACT_CSV     = Path("/data/ross/assay_calibration/labelseq_dataframe_processed.csv.gz")

DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
BATCH  = 4096

MODEL_CONFIGS = [
    ("concat_ef1_k128", "concat", 2048, 1, 128),
    ("diff_ef4_k256",   "diff",   1024, 4, 256),
]

# ── TopKSAE ────────────────────────────────────────────────────────────────────
class TopKSAE(nn.Module):
    def __init__(self, in_dim, ef, k):
        super().__init__()
        d = ef * in_dim
        self.k = k; self.d = d
        self.encoder = nn.Linear(in_dim, d)
        self.decoder = nn.Linear(d, in_dim, bias=False)
        self.register_buffer("b_dec", torch.zeros(in_dim))
    def encode(self, x):
        pre = torch.relu(self.encoder(x - self.b_dec))
        tv, ti = pre.topk(self.k, dim=-1, sorted=False)
        return torch.zeros_like(pre).scatter_(-1, ti, tv)
    def forward(self, x):
        z = self.encode(x)
        return z, self.decoder(z) + self.b_dec


def load_model(name, in_dim, ef, k):
    m = TopKSAE(in_dim, ef, k)
    m.load_state_dict(torch.load(str(COMBINED / f"combined_{name}.pt"), map_location="cpu"))
    m.eval()
    return m.to(DEVICE)


def encode_to_sparse(model, X_np, desc="encode"):
    """Encode (N, in_dim) float32 array → scipy CSR sparse (N, dict_size)."""
    rows, cols, vals = [], [], []
    n = len(X_np)
    with torch.no_grad():
        for i in tqdm(range(0, n, BATCH), desc=desc, leave=False):
            xb = torch.from_numpy(X_np[i:i+BATCH]).to(DEVICE)
            z  = model.encode(xb).cpu().numpy()
            ri, ci = np.nonzero(z)
            rows.extend(ri + i)
            cols.extend(ci)
            vals.extend(z[ri, ci])
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, model.d), dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# §1  Load source embeddings (with caching)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("§1  Loading source embeddings")
print("=" * 70)

# ── ClinVar (already cached as concat features) ──────────────────────────────
print("ClinVar: loading from precomputed cache …")
cv_feats  = np.load(INTD / "clinvar_feats.npy")        # (N_cv, 2048) concat(WT, VT)
cv_labels = np.load(INTD / "clinvar_labels.npy")       # 0=benign, 1=pathogenic
print(f"  ClinVar: {len(cv_feats):,} variants  (path={cv_labels.sum():,}  ben={(cv_labels==0).sum():,})")

# Derive WT / VT from concat features
cv_wt = cv_feats[:, :1024]
cv_vt = cv_feats[:, 1024:]


# ── gnomAD (load from H5 or cache) ────────────────────────────────────────────
def _load_subgraph_h5(h5_path):
    import h5py
    wt_list, vt_list = [], []
    seen = set()
    with h5py.File(str(h5_path), "r") as f:
        for complex_key in tqdm(f.keys(), desc=h5_path.stem, leave=False):
            prot_id = complex_key.split("_")[0]
            cgrp = f[complex_key]
            for var_key in cgrp.keys():
                uid = (prot_id, var_key)
                if uid in seen: continue
                seen.add(uid)
                try:
                    vgrp = cgrp[var_key]
                    emb  = vgrp["node_emb"][:]
                    diff = vgrp["mut_diff"][:]
                    idx  = int(vgrp.attrs["mut_local_idx"])
                    vt   = emb[idx]; wt = vt - diff
                    wt_list.append(wt.astype(np.float32))
                    vt_list.append(vt.astype(np.float32))
                except Exception:
                    pass
    return np.stack(wt_list), np.stack(vt_list)


def _load_hgmd_h5(h5_path):
    import h5py
    _re = re.compile(r'^[A-Za-z*](\d+)[A-Za-z*]$')
    wt_list, vt_list = [], []
    with h5py.File(str(h5_path), "r") as f:
        all_keys = list(f.keys())
        vt_keys  = [k for k in all_keys if ' ' in k]
        wt_set   = set(all_keys) - set(vt_keys)
        for vt_key in tqdm(vt_keys, desc="HGMD", leave=False):
            prot_id, variant = vt_key.split(' ', 1)
            if prot_id not in wt_set: continue
            m = _re.match(variant)
            if not m: continue
            pos = int(m.group(1))
            try:
                wt_full = f[prot_id][:]
                vt_full = f[vt_key][:]
                if pos >= wt_full.shape[0] or pos >= vt_full.shape[0]: continue
                wt_list.append(wt_full[pos].astype(np.float32))
                vt_list.append(vt_full[pos].astype(np.float32))
            except Exception:
                pass
    return np.stack(wt_list), np.stack(vt_list)


if GNOMAD_WT_CACHE.exists() and GNOMAD_VT_CACHE.exists():
    print("gnomAD: loading from cache …")
    gn_wt = np.load(GNOMAD_WT_CACHE); gn_vt = np.load(GNOMAD_VT_CACHE)
else:
    print("gnomAD: loading from H5 (this takes ~20 min) …")
    gn_wt, gn_vt = _load_subgraph_h5(GNOMAD_H5)
    np.save(GNOMAD_WT_CACHE, gn_wt); np.save(GNOMAD_VT_CACHE, gn_vt)
print(f"  gnomAD:  {len(gn_wt):,} variants")

if HGMD_WT_CACHE.exists() and HGMD_VT_CACHE.exists():
    print("HGMD: loading from cache …")
    hg_wt = np.load(HGMD_WT_CACHE); hg_vt = np.load(HGMD_VT_CACHE)
else:
    print("HGMD: loading from H5 …")
    hg_wt, hg_vt = _load_hgmd_h5(HGMD_H5)
    np.save(HGMD_WT_CACHE, hg_wt); np.save(HGMD_VT_CACHE, hg_vt)
print(f"  HGMD:    {len(hg_wt):,} variants")

n_cv = len(cv_wt); n_gn = len(gn_wt); n_hg = len(hg_wt)


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Encode each source through each model → sparse Z  (cached per model)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("§2  Encoding sources through SAE models")
print("=" * 70)

def _source_features(input_type, wt, vt):
    if input_type == "concat":
        return np.concatenate([wt, vt], axis=1)
    return (vt - wt)


all_enrichment = {}   # name → (enrichment array, fire_dis, fire_ben)

for name, itype, in_dim, ef, k in MODEL_CONFIGS:
    z_cv_path = OUT_DIR / f"z_cv_{name}.npz"
    z_gn_path = OUT_DIR / f"z_gn_{name}.npz"
    z_hg_path = OUT_DIR / f"z_hg_{name}.npz"

    if z_cv_path.exists() and z_gn_path.exists() and z_hg_path.exists():
        print(f"[{name}] loading cached Z …")
        Z_cv = sp.load_npz(str(z_cv_path))
        Z_gn = sp.load_npz(str(z_gn_path))
        Z_hg = sp.load_npz(str(z_hg_path))
    else:
        print(f"[{name}] encoding (in_dim={in_dim}, ef={ef}, k={k}) …")
        model = load_model(name, in_dim, ef, k)

        X_cv = _source_features(itype, cv_wt, cv_vt)
        X_gn = _source_features(itype, gn_wt, gn_vt)
        X_hg = _source_features(itype, hg_wt, hg_vt)

        Z_cv = encode_to_sparse(model, X_cv, f"  ClinVar/{name}")
        Z_gn = encode_to_sparse(model, X_gn, f"  gnomAD/{name}")
        Z_hg = encode_to_sparse(model, X_hg, f"  HGMD/{name}")

        sp.save_npz(str(z_cv_path), Z_cv)
        sp.save_npz(str(z_gn_path), Z_gn)
        sp.save_npz(str(z_hg_path), Z_hg)
        del model
        torch.cuda.empty_cache()

    print(f"  Z_cv={Z_cv.shape}  Z_gn={Z_gn.shape}  Z_hg={Z_hg.shape}")

    # ── Path A: per-latent disease enrichment ──────────────────────────────
    # disease = ClinVar pathogenic + HGMD
    # benign  = ClinVar benign     + gnomAD
    dis_cv = (cv_labels == 1)
    ben_cv = (cv_labels == 0)

    Z_dis = sp.vstack([Z_cv[dis_cv], Z_hg])   # (n_path + n_hg, D)
    Z_ben = sp.vstack([Z_cv[ben_cv], Z_gn])   # (n_ben  + n_gn, D)

    n_dis = Z_dis.shape[0]; n_ben = Z_ben.shape[0]
    fire_dis = np.asarray((Z_dis > 0).sum(axis=0)).ravel() / n_dis
    fire_ben = np.asarray((Z_ben > 0).sum(axis=0)).ravel() / n_ben

    enrichment = np.log2((fire_dis + 1e-6) / (fire_ben + 1e-6))
    all_enrichment[name] = enrichment

    df_enr = pd.DataFrame({
        "latent_idx":  np.arange(len(enrichment)),
        "enrichment":  enrichment,
        "fire_dis":    fire_dis,
        "fire_ben":    fire_ben,
    })
    df_enr.to_csv(OUT_DIR / f"latent_enrichment_{name}.csv", index=False)
    print(f"  enrichment: min={enrichment.min():.3f}  max={enrichment.max():.3f}"
          f"  median={np.median(enrichment):.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# §3  Cross-reference with injection ΔAUCs → scatter plot  (Path A)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("§3  Cross-referencing with injection ΔAUCs (Path A scatter)")
print("=" * 70)

df_patch = pd.read_csv(PATCH_CSV)
# keep only recon inject (strongest signal per earlier analysis)
df_inj = df_patch[(df_patch["probe_type"] == "recon") & (df_patch["intervention"] == "inject")].copy()

TASK_COLORS = {
    "destab_vs_neut": "#c0392b",
    "stab_vs_neut":   "#2980b9",
    "gof_vs_wt":      "#27ae60",
    "lof_vs_wt":      "#e67e22",
}

for name, itype, in_dim, ef, k in MODEL_CONFIGS:
    enr = all_enrichment[name]
    df_m = df_inj[df_inj["model"] == name].copy()
    if df_m.empty:
        print(f"  [{name}] no injection rows, skipping scatter")
        continue

    # max ΔAUC across tasks per latent
    df_max = df_m.groupby(["latent_idx", "task"])["delta_auc"].max().reset_index()
    df_max["enrichment"] = enr[df_max["latent_idx"].values]

    fig, ax = plt.subplots(figsize=(7, 5))
    for task, color in TASK_COLORS.items():
        sub = df_max[df_max["task"] == task]
        ax.scatter(sub["enrichment"], sub["delta_auc"],
                   c=color, s=12, alpha=0.55, label=task, linewidths=0)

    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Disease enrichment (log2 path+HGMD / ben+gnomAD)", fontsize=10)
    ax.set_ylabel("Injection ΔAUC (recon probe)", fontsize=10)
    ax.set_title(f"{name} — Disease enrichment vs phenotype causal effect", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, markerscale=2, framealpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / f"enrichment_scatter_{name}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [{name}] scatter saved")

    # top 20 disease-enriched AND phenotype-causal latents
    df_max["score"] = df_max["enrichment"] * df_max["delta_auc"].clip(lower=0)
    top = df_max.nlargest(20, "score")[["latent_idx", "task", "enrichment", "delta_auc", "score"]]
    print(f"  [{name}] top 10 (enrichment × ΔAUC):")
    print(top.head(10).to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# §4  Load phenotype labels for stability and activity  (shared for Path B)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("§4  Loading phenotype labels for clustering")
print("=" * 70)

# ── Stability ──────────────────────────────────────────────────────────────────
valid_mask = np.load(STAB_CACHE / "valid_mask.npy").astype(bool)
ddg_stab   = np.load(STAB_CACHE / "ddg_valid.npy")
y_stab = np.full(len(ddg_stab), -1, dtype=np.int8)
y_stab[ddg_stab < -1.0]             = 0  # stabilising
y_stab[np.abs(ddg_stab) < 0.5]      = 1  # neutral
y_stab[ddg_stab >= 1.5]             = 2  # destabilising
stab_mask = y_stab >= 0
print(f"  Stability: {stab_mask.sum():,} labeled  "
      f"(stab={( y_stab==0).sum():,} neut={(y_stab==1).sum():,} destab={(y_stab==2).sum():,})")

# ── Activity ───────────────────────────────────────────────────────────────────
pid_act    = np.load(ACT_CACHE / "protein_ids.npy", allow_pickle=True)
_valid_idx = np.load(ACT_CACHE / "valid_idx.npy")
_AA3 = {"Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E",
        "Gly":"G","His":"H","Ile":"I","Leu":"L","Lys":"K","Met":"M","Phe":"F",
        "Pro":"P","Ser":"S","Thr":"T","Trp":"W","Tyr":"Y","Val":"V"}
_re2 = re.compile(r'^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$')
def _abin(s):
    if s < 0.75:  return "LoF"
    if 0.80 <= s <= 1.20: return "wt_like"
    if s > 1.25:  return "GoF"
_df = pd.read_csv(ACT_CSV, compression="gzip")
_df = _df[_df["assay"] == "activity"].copy()
_df = _df[_df["variant"].str.match(r'^p\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2}$', na=False)]
def _pv(v):
    m = _re2.match(v)
    return (None, None, None) if m is None else (_AA3.get(m.group(1)), int(m.group(2)), _AA3.get(m.group(3)))
_p = [_pv(v) for v in _df["variant"]]
_df["aa_ref"] = [x[0] for x in _p]; _df["aa_pos"] = [x[1] for x in _p]; _df["aa_alt"] = [x[2] for x in _p]
_df = _df.dropna(subset=["aa_ref", "aa_pos", "aa_alt"])
_dv = (_df.groupby(["uniprot_accession", "Gene", "aa_ref", "aa_pos", "aa_alt"])["average score"]
       .mean().reset_index())
_dv.rename(columns={"average score": "score"}, inplace=True)
_dv["bin"] = _dv["score"].map(_abin); _dv = _dv[_dv["bin"].notna()].reset_index(drop=True)
bins = [_dv["bin"].tolist()[i] for i in _valid_idx]
y_act = np.full(len(bins), -1, dtype=np.int8)
y_act[[i for i, b in enumerate(bins) if b == "LoF"]]     = 0
y_act[[i for i, b in enumerate(bins) if b == "wt_like"]] = 1
y_act[[i for i, b in enumerate(bins) if b == "GoF"]]     = 2
act_mask = y_act >= 0
print(f"  Activity:  {act_mask.sum():,} labeled  "
      f"(LoF={(y_act==0).sum():,} WT={(y_act==1).sum():,} GoF={(y_act==2).sum():,})")


# ═══════════════════════════════════════════════════════════════════════════════
# §5  Sparse clustering of phenotype-labeled variants  (Path B)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("§5  Sparse clustering (Path B)")
print("=" * 70)

N_CLUSTERS = 100
UMAP_SUBSAMPLE = 30_000

try:
    import umap
    HAS_UMAP = True
except ImportError:
    from sklearn.decomposition import PCA
    HAS_UMAP = False
    print("  umap-learn not found; will use PCA for 2D embedding")

STAB_LABELS = {0: "Stabilising", 1: "Neutral", 2: "Destabilising"}
ACT_LABELS  = {0: "LoF", 1: "WT-like", 2: "GoF"}
STAB_COLORS = {0: "#2980b9", 1: "#95a5a6", 2: "#c0392b"}
ACT_COLORS  = {0: "#e67e22", 1: "#95a5a6", 2: "#27ae60"}

cluster_rows = []  # for combined summary

for name, itype, in_dim, ef, k in MODEL_CONFIGS:
    enr = all_enrichment[name]
    clust_path = OUT_DIR / f"cluster_assignments_{name}.npz"

    for task_tag, z_path, y_full, mask, lbl_map, col_map in [
        ("stability", COMBINED_CACHE / f"z_stab_{name}.npz", y_stab, stab_mask, STAB_LABELS, STAB_COLORS),
        ("activity",  COMBINED_CACHE / f"z_act_{name}.npz",  y_act,  act_mask,  ACT_LABELS,  ACT_COLORS),
    ]:
        if not z_path.exists():
            print(f"  [{name}/{task_tag}] Z cache not found, skipping")
            continue

        print(f"  [{name}/{task_tag}] loading Z …")
        Z = sp.load_npz(str(z_path))

        # keep only labeled variants
        y_lab  = y_full[mask]
        Z_lab  = Z[mask]
        n_lab  = Z_lab.shape[0]

        # row-normalise for cosine k-means
        print(f"  [{name}/{task_tag}] normalising {n_lab:,} rows …")
        Z_norm = normalize(Z_lab, norm="l2")

        # Mini-batch k-means
        print(f"  [{name}/{task_tag}] k-means k={N_CLUSTERS} …")
        km = MiniBatchKMeans(n_clusters=N_CLUSTERS, batch_size=8192,
                             max_iter=200, n_init=3, random_state=42, verbose=0)
        cluster_ids = km.fit_predict(Z_norm)

        # per-cluster summary
        dict_size = Z_lab.shape[1]
        Z_dense_mean = np.asarray(Z_lab.mean(axis=0)).ravel()  # global mean per latent

        for c in range(N_CLUSTERS):
            cidx = np.where(cluster_ids == c)[0]
            if len(cidx) == 0: continue
            y_c = y_lab[cidx]
            label_counts = {v: int((y_c == v).sum()) for v in lbl_map}
            size = len(cidx)
            purity = max(label_counts.values()) / size
            dominant_label = max(label_counts, key=label_counts.get)

            # mean activation per latent within cluster
            Z_c_mean = np.asarray(Z_lab[cidx].mean(axis=0)).ravel()
            top_lats = np.argsort(Z_c_mean)[::-1][:10]
            top_enr  = enr[top_lats]
            mean_top_enr = float(top_enr.mean())

            cluster_rows.append({
                "model":           name,
                "task":            task_tag,
                "cluster":         c,
                "size":            size,
                "purity":          round(purity, 4),
                "dominant_label":  lbl_map[dominant_label],
                "top_latents":     ",".join(map(str, top_lats)),
                "top_enr_mean":    round(mean_top_enr, 4),
                **{f"n_{lbl_map[v].lower().replace('-','_')}": label_counts.get(v, 0)
                   for v in lbl_map},
            })

        # ── UMAP / PCA plot ────────────────────────────────────────────────────
        print(f"  [{name}/{task_tag}] 2D embedding …")
        rng = np.random.default_rng(42)
        sub_idx = rng.choice(n_lab, size=min(UMAP_SUBSAMPLE, n_lab), replace=False)
        Z_sub   = Z_norm[sub_idx].toarray() if sp.issparse(Z_norm) else Z_norm[sub_idx]
        y_sub   = y_lab[sub_idx]
        c_sub   = cluster_ids[sub_idx]

        if HAS_UMAP:
            reducer = umap.UMAP(n_components=2, metric="cosine", n_neighbors=20,
                                min_dist=0.1, random_state=42, verbose=False)
            emb = reducer.fit_transform(Z_sub)
        else:
            pca = PCA(n_components=50, random_state=42)
            emb = pca.fit_transform(Z_sub)
            from sklearn.manifold import TSNE
            emb = TSNE(n_components=2, metric="cosine", random_state=42,
                       init="pca", perplexity=50).fit_transform(emb)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        # left: coloured by phenotype
        for lv, lname in lbl_map.items():
            idx = y_sub == lv
            axes[0].scatter(emb[idx, 0], emb[idx, 1], c=col_map[lv],
                            s=3, alpha=0.4, label=lname, linewidths=0)
        axes[0].set_title(f"Phenotype", fontsize=10, fontweight="bold")
        axes[0].legend(fontsize=8, markerscale=3)

        # right: coloured by cluster (use tab20 cycling)
        cmap = plt.cm.get_cmap("tab20", N_CLUSTERS)
        axes[1].scatter(emb[:, 0], emb[:, 1], c=c_sub, cmap="tab20",
                        s=3, alpha=0.4, linewidths=0)
        axes[1].set_title(f"Cluster (k={N_CLUSTERS})", fontsize=10, fontweight="bold")

        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7)

        fig.suptitle(f"{name} — {task_tag}  (n={n_lab:,}, subsample={len(sub_idx):,})",
                     fontsize=10, fontweight="bold")
        plt.tight_layout()
        plt.savefig(str(OUT_DIR / f"umap_{name}_{task_tag}.png"), dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  [{name}/{task_tag}] UMAP saved")

# ── Cluster summary CSV ────────────────────────────────────────────────────────
df_clust = pd.DataFrame(cluster_rows)
df_clust.to_csv(OUT_DIR / "cluster_summary.csv", index=False)
print(f"\nCluster summary: {len(df_clust)} rows → {OUT_DIR / 'cluster_summary.csv'}")

# top clusters by purity × disease enrichment
top_clusters = (df_clust[df_clust["purity"] > 0.6]
                .sort_values("top_enr_mean", ascending=False)
                .head(20))
print("\nTop phenotype-pure + disease-enriched clusters:")
print(top_clusters[["model", "task", "cluster", "size", "purity",
                     "dominant_label", "top_enr_mean"]].to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# §6  Latent co-activation network  (Path C)
#     Run on one representative model per input type to keep compute tractable.
#     Build latent-latent Jaccard co-occurrence → sparse kNN graph → Leiden /
#     Louvain community detection → annotate modules with disease enrichment.
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("§6  Latent co-activation network (Path C)")
print("=" * 70)

# Try leiden; fall back to networkx louvain
try:
    import igraph as ig
    import leidenalg
    USE_LEIDEN = True
    print("  Using Leiden algorithm")
except ImportError:
    try:
        import networkx as nx
        from networkx.algorithms.community import louvain_communities
        USE_LEIDEN = False
        print("  leidenalg not found — using networkx Louvain")
    except ImportError:
        USE_LEIDEN = None
        print("  WARNING: neither leidenalg nor networkx found — skipping Path C")

PATH_C_MODELS = ["concat_ef1_k128"]

if USE_LEIDEN is not None:
    for name, itype, in_dim, ef, k in MODEL_CONFIGS:
        if name not in PATH_C_MODELS:
            continue
        enr = all_enrichment[name]
        D   = ef * in_dim   # dict_size

        coact_path = OUT_DIR / f"coact_{name}.npz"
        if coact_path.exists():
            print(f"  [{name}] loading cached co-activation matrix …")
            C_jac = sp.load_npz(str(coact_path))
        else:
            print(f"  [{name}] building co-activation matrix (D={D}) …")
            # Stack all disease-source Z (ClinVar + HGMD) for richer signal
            z_cv_path = OUT_DIR / f"z_cv_{name}.npz"
            z_hg_path = OUT_DIR / f"z_hg_{name}.npz"
            z_gn_path = OUT_DIR / f"z_gn_{name}.npz"
            Z_all = sp.vstack([
                sp.load_npz(str(z_cv_path)),
                sp.load_npz(str(z_hg_path)),
                sp.load_npz(str(z_gn_path)),
            ])
            Z_bin = (Z_all > 0).astype(np.float32)
            del Z_all

            # co-occurrence counts: (D, D) — each entry = # variants both fired
            print(f"    co-occurrence multiply ({Z_bin.shape}) …")
            C_count = (Z_bin.T @ Z_bin)    # sparse (D, D)
            # marginals = diagonal = per-latent fire count
            marg = np.asarray(C_count.diagonal()).ravel()  # (D,)
            # Jaccard: C[i,j] / (marg[i] + marg[j] - C[i,j])
            # Compute as sparse — iterate nonzero entries
            C_coo = C_count.tocoo()
            ii, jj, vv = C_coo.row, C_coo.col, C_coo.data
            denom = marg[ii] + marg[jj] - vv
            jac   = np.where(denom > 0, vv / denom, 0.0).astype(np.float32)
            # zero the diagonal (self-similarity not useful)
            diag_mask = ii != jj
            C_jac = sp.csr_matrix((jac[diag_mask], (ii[diag_mask], jj[diag_mask])),
                                  shape=(D, D))
            sp.save_npz(str(coact_path), C_jac)
            print(f"    co-activation matrix saved ({C_jac.nnz:,} entries)")

        # ── kNN graph: top-15 neighbours per latent by Jaccard ────────────────
        KNN = 15
        print(f"  [{name}] building kNN graph (k={KNN}) …")
        C_csr = C_jac.tocsr()
        edge_src, edge_dst, edge_w = [], [], []
        for i in range(D):
            row = np.asarray(C_csr[i].todense()).ravel()
            top_j = np.argpartition(row, -KNN)[-KNN:]
            top_j = top_j[row[top_j] > 0]
            for j in top_j:
                edge_src.append(i); edge_dst.append(int(j)); edge_w.append(float(row[j]))

        # ── Community detection ────────────────────────────────────────────────
        print(f"  [{name}] community detection …")
        if USE_LEIDEN:
            G = ig.Graph(n=D, edges=list(zip(edge_src, edge_dst)),
                         edge_attrs={"weight": edge_w}, directed=False)
            partition = leidenalg.find_partition(
                G, leidenalg.ModularityVertexPartition,
                weights="weight", n_iterations=5, seed=42)
            memberships = np.array(partition.membership)
        else:
            G = nx.Graph()
            G.add_nodes_from(range(D))
            for s, d, w in zip(edge_src, edge_dst, edge_w):
                G.add_edge(s, d, weight=w)
            communities = louvain_communities(G, weight="weight", seed=42)
            memberships = np.full(D, -1, dtype=int)
            for cid, community in enumerate(communities):
                for node in community:
                    memberships[node] = cid

        n_modules = memberships.max() + 1
        print(f"  [{name}] found {n_modules} modules")

        # ── Annotate modules ──────────────────────────────────────────────────
        df_patch_model = df_inj[df_inj["model"] == name]
        # max ΔAUC per latent across all tasks
        lat_delta = (df_patch_model.groupby("latent_idx")["delta_auc"]
                     .max().rename("max_delta_auc"))

        module_rows = []
        for mid in range(n_modules):
            lats = np.where(memberships == mid)[0]
            if len(lats) < 2: continue
            m_enr  = float(enr[lats].mean())
            m_fire = float(((C_jac[lats].sum(axis=1)) > 0).mean())  # connectivity
            # phenotype causal: mean max ΔAUC for module latents in the injection results
            lats_in_patch = [l for l in lats if l in lat_delta.index]
            m_delta = float(lat_delta.loc[lats_in_patch].mean()) if lats_in_patch else 0.0
            top_lats_by_enr = lats[np.argsort(enr[lats])[::-1][:5]]
            module_rows.append({
                "model":          name,
                "module":         mid,
                "size":           len(lats),
                "mean_enr":       round(m_enr, 4),
                "mean_max_delta": round(m_delta, 4),
                "top_latents_by_enr": ",".join(map(str, top_lats_by_enr)),
            })

        df_mod = pd.DataFrame(module_rows).sort_values("mean_enr", ascending=False)
        df_mod.to_csv(OUT_DIR / f"latent_modules_{name}.csv", index=False)

        print(f"\n  [{name}] top 10 modules by mean disease enrichment:")
        print(df_mod.head(10)[["module", "size", "mean_enr", "mean_max_delta",
                                "top_latents_by_enr"]].to_string(index=False))

        # ── Module map heatmap (top 30 modules by size, top 50 latents each) ──
        top_mods = df_mod.nlargest(min(30, len(df_mod)), "size")["module"].tolist()
        hm_rows, hm_labels, hm_enr = [], [], []
        for mid in top_mods:
            lats = np.where(memberships == mid)[0][:50]
            hm_rows.append(enr[lats])
            hm_labels.append(f"M{mid}(n={len(np.where(memberships==mid)[0])})")
        max_len = max(len(r) for r in hm_rows)
        hm_mat  = np.full((len(hm_rows), max_len), np.nan)
        for i, r in enumerate(hm_rows):
            hm_mat[i, :len(r)] = r

        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(hm_mat, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
        ax.set_yticks(range(len(hm_labels))); ax.set_yticklabels(hm_labels, fontsize=7)
        ax.set_xlabel("Latent rank within module", fontsize=9)
        ax.set_title(f"{name} — Disease enrichment per latent, top modules",
                     fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, label="log2 enrichment")
        plt.tight_layout()
        plt.savefig(str(OUT_DIR / f"module_heatmap_{name}.png"), dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  [{name}] module heatmap saved")

print("\n\nDone — all outputs in:", OUT_DIR)
