"""
disease_mechanism_analysis.py

Revised Path B: cluster only pathogenic variants → disease mechanism clusters
  - Input: Z_cv[pathogenic] + Z_hg (186k disease variants)
  - k=50 clusters, each = a candidate disease mechanism
  - Per-cluster: top latents, benign contamination, injection ΔAUC validation
  - UMAP: disease clusters + benign overlay

Revised Path C: annotate Louvain latent modules as pathomechanism signatures
  - Disease specificity: module activation rate in disease vs benign variants
  - Phenotype linkage: which phenotype class (destab/GoF/LoF) each module predicts
  - Aggregate decoder effect: net recon probe effect across module latents
  - Protein diversity: how many unique proteins activate each module

Outputs → /data/ross/interp/latent_analysis/
"""

import warnings; warnings.filterwarnings("ignore")
import argparse, re, pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ── Paths ──────────────────────────────────────────────────────────────────────
SB         = Path("/home/rcstewart/ppi_lossgain/sparse_bottleneck")
LA         = Path("/data/ross/interp/latent_analysis")
COMBINED   = Path("/data/ross/ppi_lossgain/interaction_loss/sae_weights/combined")
STAB_CACHE = Path("/data/ross/interp/collab_sae_cache")
ACT_CACHE  = Path("/data/ross/interp/activity_sae_cache")
COMBINED_CACHE = Path("/data/ross/interp/combined_sae_cache")
ACT_CSV    = Path("/data/ross/assay_calibration/labelseq_dataframe_processed.csv.gz")
MEGA_PKL   = "/data/ross/ppi_lossgain/interaction_loss/megascale_preprocessed/preprocessed.pkl"
PATCH_CSV  = SB / "activation_patching_results_v2.csv"

_MODEL_REGISTRY = {
    "concat_ef1_k128": (2048, 1, 128),
    "concat_ef4_k128": (2048, 4, 128),
    "concat_ef4_k64":  (2048, 4,  64),
    "diff_ef4_k256":   (1024, 4, 256),
    "diff_ef4_k64":    (1024, 4,  64),
    "diff_ef4_k32":    (1024, 4,  32),
    "diff_ef1_k64":    (1024, 1,  64),
}

_ap = argparse.ArgumentParser(description="Disease mechanism clustering")
_ap.add_argument("--name", default="concat_ef1_k128",
                 help="SAE model name (default: concat_ef1_k128)")
_args, _ = _ap.parse_known_args()

NAME = _args.name
IN_DIM, EF, K = _MODEL_REGISTRY.get(NAME, (2048, 1, 128))
DICT_SIZE = EF * IN_DIM

N_CLUSTERS   = 50
MOD_THRESH   = 0.25     # fraction of module latents that must be active for "module fires"
UMAP_N       = 30_000

# ── TopKSAE (encode-only) ──────────────────────────────────────────────────────
class TopKSAE(nn.Module):
    def __init__(self, in_dim, ef, k):
        super().__init__()
        d = ef * in_dim
        self.k = k; self.d = d
        self.encoder = nn.Linear(in_dim, d)
        self.decoder = nn.Linear(d, in_dim, bias=False)
        self.register_buffer("b_dec", torch.zeros(in_dim))
    def forward(self, x):
        pre = torch.relu(self.encoder(x - self.b_dec))
        tv, ti = pre.topk(self.k, dim=-1, sorted=False)
        z = torch.zeros_like(pre).scatter_(-1, ti, tv)
        return z, self.decoder(z) + self.b_dec


# ── Load decoder weights ───────────────────────────────────────────────────────
print("Loading SAE decoder …")
model = TopKSAE(IN_DIM, EF, K)
model.load_state_dict(torch.load(str(COMBINED / f"combined_{NAME}.pt"), map_location="cpu"))
model.eval()
W_dec = model.decoder.weight.detach().numpy().astype(np.float32)   # (2048, 2048)
b_dec = model.b_dec.detach().numpy().astype(np.float32)
# diff decoder: effect on VT-WT recon space
W_dec_diff = (W_dec[1024:] - W_dec[:1024]).astype(np.float32)      # (1024, 2048)
b_dec_diff = (b_dec[1024:] - b_dec[:1024]).astype(np.float32)
del model

# ── Load source Z and labels ───────────────────────────────────────────────────
print("Loading Z matrices …")
Z_cv = sp.load_npz(str(LA / f"z_cv_{NAME}.npz"))   # (227189, 2048)
Z_gn = sp.load_npz(str(LA / f"z_gn_{NAME}.npz"))   # (599100, 2048)
Z_hg = sp.load_npz(str(LA / f"z_hg_{NAME}.npz"))   # (13390,  2048)

cv_labels  = np.load(SB / "clinvar_labels.npy")      # 0=benign, 1=pathogenic
cv_prot_ids = np.load(SB / "clinvar_protein_ids.npy", allow_pickle=True)

path_mask = cv_labels == 1
ben_mask  = cv_labels == 0

Z_path = sp.vstack([Z_cv[path_mask], Z_hg])          # (186214, 2048) disease
Z_ben  = sp.vstack([Z_cv[ben_mask],  Z_gn])          # (653465, 2048) benign

n_path = Z_path.shape[0]; n_ben = Z_ben.shape[0]
print(f"  Disease: {n_path:,}  Benign: {n_ben:,}")

# Protein IDs for disease variants (ClinVar pathogenic only; HGMD not cached)
path_prot_ids = cv_prot_ids[path_mask]   # (172824,) — first n_cv_path rows of Z_path

# ── Load enrichment and injection results ─────────────────────────────────────
df_enr   = pd.read_csv(LA / f"latent_enrichment_{NAME}.csv")
enr      = df_enr["enrichment"].values   # (2048,)

df_patch = pd.read_csv(PATCH_CSV)
df_inj   = df_patch[(df_patch["probe_type"]=="recon") &
                     (df_patch["intervention"]=="inject") &
                     (df_patch["model"]==NAME)].copy()
lat_delta_by_task = df_inj.groupby(["latent_idx","task"])["delta_auc"].max().unstack(fill_value=0)

# ── Load phenotype labels (stability + activity) ───────────────────────────────
print("Loading phenotype labels …")
valid_mask = np.load(STAB_CACHE / "valid_mask.npy").astype(bool)
ddg_stab   = np.load(STAB_CACHE / "ddg_valid.npy")
y_stab = np.full(len(ddg_stab), -1, dtype=np.int8)
y_stab[ddg_stab < -1.0]        = 0   # stabilising
y_stab[np.abs(ddg_stab) < 0.5] = 1   # neutral
y_stab[ddg_stab >= 1.5]        = 2   # destabilising
stab_mask = y_stab >= 0

pid_act    = np.load(ACT_CACHE / "protein_ids.npy", allow_pickle=True)
_valid_idx = np.load(ACT_CACHE / "valid_idx.npy")
_AA3 = {"Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E",
        "Gly":"G","His":"H","Ile":"I","Leu":"L","Lys":"K","Met":"M","Phe":"F",
        "Pro":"P","Ser":"S","Thr":"T","Trp":"W","Tyr":"Y","Val":"V"}
_re2 = re.compile(r'^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$')
def _abin(s):
    if s < 0.75: return "LoF"
    if 0.80 <= s <= 1.20: return "wt_like"
    if s > 1.25: return "GoF"
_df = pd.read_csv(ACT_CSV, compression="gzip")
_df = _df[_df["assay"]=="activity"].copy()
_df = _df[_df["variant"].str.match(r'^p\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2}$', na=False)]
def _pv(v):
    m=_re2.match(v); return (None,None,None) if m is None else (_AA3.get(m.group(1)),int(m.group(2)),_AA3.get(m.group(3)))
_p = [_pv(v) for v in _df["variant"]]
_df["aa_ref"]=[x[0] for x in _p]; _df["aa_pos"]=[x[1] for x in _p]; _df["aa_alt"]=[x[2] for x in _p]
_df = _df.dropna(subset=["aa_ref","aa_pos","aa_alt"])
_dv = (_df.groupby(["uniprot_accession","Gene","aa_ref","aa_pos","aa_alt"])["average score"]
       .mean().reset_index())
_dv.rename(columns={"average score":"score"},inplace=True)
_dv["bin"] = _dv["score"].map(_abin); _dv = _dv[_dv["bin"].notna()].reset_index(drop=True)
bins = [_dv["bin"].tolist()[i] for i in _valid_idx]
y_act = np.full(len(bins),-1,dtype=np.int8)
y_act[[i for i,b in enumerate(bins) if b=="LoF"]]     = 0
y_act[[i for i,b in enumerate(bins) if b=="wt_like"]] = 1
y_act[[i for i,b in enumerate(bins) if b=="GoF"]]     = 2
act_mask = y_act >= 0

Z_stab = sp.load_npz(str(COMBINED_CACHE / f"z_stab_{NAME}.npz"))
Z_act  = sp.load_npz(str(COMBINED_CACHE / f"z_act_{NAME}.npz"))


# ═══════════════════════════════════════════════════════════════════════════════
# Path B (revised): cluster only disease variants
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("Path B (revised): Disease-only clustering")
print("="*70)

# Normalise disease Z for cosine k-means
print(f"Normalising {n_path:,} disease variants …")
Z_path_norm = normalize(Z_path, norm="l2")

print(f"k-means k={N_CLUSTERS} …")
km = MiniBatchKMeans(n_clusters=N_CLUSTERS, batch_size=8192,
                     max_iter=300, n_init=5, random_state=42, verbose=0)
dis_cluster_ids = km.fit_predict(Z_path_norm)

# Assign benign variants to nearest cluster centroid
print(f"Assigning {n_ben:,} benign variants to clusters for contamination …")
Z_ben_norm = normalize(Z_ben, norm="l2")
# Cosine similarity to centroids: (n_ben, k) = Z_ben_norm @ centroids.T
ben_cluster_ids = km.predict(Z_ben_norm)

# ── Per-cluster summary ────────────────────────────────────────────────────────
# Build binary activation matrices for speed
Z_path_bin = (Z_path > 0).astype(np.int8)
Z_stab_bin = (Z_stab[stab_mask] > 0).astype(np.int8)
Z_act_bin  = (Z_act[act_mask]   > 0).astype(np.int8)
y_stab_lab = y_stab[stab_mask]
y_act_lab  = y_act[act_mask]

# Assign stability/activity variants to nearest centroid
print("Assigning stability/activity variants to clusters …")
Z_stab_norm = normalize(Z_stab[stab_mask], norm="l2")
Z_act_norm  = normalize(Z_act[act_mask],   norm="l2")
stab_cluster_ids = km.predict(Z_stab_norm)
act_cluster_ids  = km.predict(Z_act_norm)

cluster_rows = []
for c in range(N_CLUSTERS):
    dis_idx = np.where(dis_cluster_ids == c)[0]
    ben_idx = np.where(ben_cluster_ids == c)[0]
    n_dis_c = len(dis_idx)
    n_ben_c = len(ben_idx)
    contamination = n_ben_c / (n_dis_c + n_ben_c) if (n_dis_c + n_ben_c) > 0 else 1.0

    # Top latents by mean activation in this cluster
    Z_c = Z_path[dis_idx]
    lat_mean = np.asarray(Z_c.mean(axis=0)).ravel()
    top_lats = np.argsort(lat_mean)[::-1][:10]
    top_enr  = float(enr[top_lats].mean())

    # Injection ΔAUC for top latents
    inj_delta = {}
    for task in ["destab_vs_neut","stab_vs_neut","gof_vs_wt","lof_vs_wt"]:
        if task in lat_delta_by_task.columns:
            vals = lat_delta_by_task.loc[
                lat_delta_by_task.index.isin(top_lats), task].values
            inj_delta[f"mean_delta_{task.split('_')[0]}"] = float(vals.mean()) if len(vals) else 0.0

    # Phenotype fractions from stability variants nearest this cluster
    si = np.where(stab_cluster_ids == c)[0]
    y_sc = y_stab_lab[si]
    stab_frac = {f"stab_{l}": int((y_sc==v).sum()) for v,l in [(0,"stab"),(1,"neut"),(2,"destab")]}

    ai = np.where(act_cluster_ids == c)[0]
    y_ac = y_act_lab[ai]
    act_frac = {f"act_{l}": int((y_ac==v).sum()) for v,l in [(0,"lof"),(1,"wt"),(2,"gof")]}

    # Protein diversity (ClinVar pathogenic only; first 172824 rows of Z_path)
    cv_path_n = path_mask.sum()
    dis_cv_idx = dis_idx[dis_idx < cv_path_n]
    n_unique_prots = len(np.unique(path_prot_ids[dis_cv_idx])) if len(dis_cv_idx) else 0
    top_prots = (pd.Series(path_prot_ids[dis_cv_idx]).value_counts().head(5).index.tolist()
                 if len(dis_cv_idx) else [])

    cluster_rows.append({
        "cluster":        c,
        "n_disease":      n_dis_c,
        "n_benign_nn":    n_ben_c,
        "contamination":  round(contamination, 4),
        "top_latents":    ",".join(map(str, top_lats)),
        "top_enr_mean":   round(top_enr, 4),
        "n_unique_prots": n_unique_prots,
        "top_prots":      ",".join(top_prots),
        **stab_frac, **act_frac, **inj_delta,
    })

df_dis = pd.DataFrame(cluster_rows)
df_dis["destab_purity"] = df_dis["stab_destab"] / (df_dis["stab_stab"] + df_dis["stab_neut"] + df_dis["stab_destab"]).clip(lower=1)
df_dis["gof_purity"]    = df_dis["act_gof"]   / (df_dis["act_lof"] + df_dis["act_wt"] + df_dis["act_gof"]).clip(lower=1)
df_dis["lof_purity"]    = df_dis["act_lof"]   / (df_dis["act_lof"] + df_dis["act_wt"] + df_dis["act_gof"]).clip(lower=1)
df_dis.to_csv(LA / "disease_clusters.csv", index=False)

print(f"\nDisease cluster summary ({N_CLUSTERS} clusters):")
print(f"  Contamination: min={df_dis['contamination'].min():.3f}  "
      f"median={df_dis['contamination'].median():.3f}  "
      f"max={df_dis['contamination'].max():.3f}")

# Top clusters by lowest contamination AND highest destab/GoF/LoF signal
print("\nTop disease-specific clusters (contamination < 0.60, sorted by top_enr):")
clean = df_dis[df_dis["contamination"] < 0.60].sort_values("top_enr_mean", ascending=False)
cols = ["cluster","n_disease","contamination","top_enr_mean","n_unique_prots","top_latents"]
delta_cols = [c for c in df_dis.columns if "mean_delta" in c]
print(clean[cols + delta_cols].head(15).to_string(index=False))

# ── UMAP ──────────────────────────────────────────────────────────────────────
print("\nUMAP …")
try:
    import umap as _umap
    HAS_UMAP = True
except ImportError:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    HAS_UMAP = False

rng = np.random.default_rng(42)
dis_sub = rng.choice(n_path, size=min(UMAP_N, n_path), replace=False)
ben_sub = rng.choice(n_ben,  size=min(10_000, n_ben),  replace=False)

Z_sub = np.vstack([
    Z_path_norm[dis_sub].toarray() if sp.issparse(Z_path_norm) else Z_path_norm[dis_sub],
    Z_ben_norm[ben_sub].toarray()  if sp.issparse(Z_ben_norm)  else Z_ben_norm[ben_sub],
])
labels_sub = np.array(["disease"] * len(dis_sub) + ["benign"] * len(ben_sub))
cluster_sub = np.concatenate([dis_cluster_ids[dis_sub], np.full(len(ben_sub), -1)])

if HAS_UMAP:
    reducer = _umap.UMAP(n_components=2, metric="cosine", n_neighbors=20,
                         min_dist=0.1, random_state=42, verbose=False)
    emb = reducer.fit_transform(Z_sub)
else:
    pca = PCA(n_components=50, random_state=42)
    emb = TSNE(n_components=2, metric="cosine", random_state=42,
               init="pca", perplexity=50).fit_transform(pca.fit_transform(Z_sub))

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
# Left: disease vs benign
is_dis = labels_sub == "disease"
axes[0].scatter(emb[~is_dis,0], emb[~is_dis,1], c="#95a5a6", s=2, alpha=0.3,
                label="Benign/gnomAD", linewidths=0)
axes[0].scatter(emb[is_dis,0],  emb[is_dis,1],  c="#c0392b", s=2, alpha=0.4,
                label="Disease (path+HGMD)", linewidths=0)
axes[0].legend(fontsize=9, markerscale=4); axes[0].set_title("Disease vs Benign", fontweight="bold")

# Right: disease clusters
cmap = plt.cm.get_cmap("tab20", N_CLUSTERS)
axes[1].scatter(emb[~is_dis,0], emb[~is_dis,1], c="#e0e0e0", s=2, alpha=0.2, linewidths=0)
sc = axes[1].scatter(emb[is_dis,0], emb[is_dis,1], c=cluster_sub[is_dis],
                      cmap="tab20", s=2, alpha=0.5, linewidths=0)
axes[1].set_title(f"Disease mechanism clusters (k={N_CLUSTERS})", fontweight="bold")

for ax in axes:
    ax.spines[["top","right"]].set_visible(False); ax.tick_params(labelsize=7)
fig.suptitle(f"{NAME} — Disease-only clustering", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(str(LA / f"umap_disease_clusters_{NAME}.png"), dpi=130, bbox_inches="tight")
plt.close()
print("  UMAP saved")


# ═══════════════════════════════════════════════════════════════════════════════
# Path C (revised): annotate Louvain modules as pathomechanism signatures
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("Path C (revised): Pathomechanism module annotation")
print("="*70)

df_mod_raw = pd.read_csv(LA / f"latent_modules_{NAME}.csv")

# ── Train recon probes for each task (needed for aggregate decoder effect) ────
print("Training recon probes …")
recon_coefs = {}
with_intercept = {}

PROBE_TASKS = {
    "destab_vs_neut": (Z_stab[stab_mask], y_stab_lab, 2, 1),
    "stab_vs_neut":   (Z_stab[stab_mask], y_stab_lab, 0, 1),
    "gof_vs_wt":      (Z_act[act_mask],   y_act_lab,  2, 1),
    "lof_vs_wt":      (Z_act[act_mask],   y_act_lab,  0, 1),
}
for task, (Z_t, y_t, pos_cls, neg_cls) in PROBE_TASKS.items():
    # recon space: Z @ W_dec_diff.T + b_dec_diff
    mask_t  = (y_t == pos_cls) | (y_t == neg_cls)
    Z_bin_t = Z_t[mask_t]
    y_bin   = (y_t[mask_t] == pos_cls).astype(int)
    xh      = np.asarray(Z_bin_t.dot(W_dec_diff.T), dtype=np.float32) + b_dec_diff
    classes = np.unique(y_bin)
    sign    = 1.0 if pos_cls > neg_cls else -1.0
    clf = LogisticRegression(penalty="l1", C=0.1, solver="liblinear",
                             class_weight="balanced", max_iter=1000, tol=1e-4)
    clf.fit(xh, y_bin)
    recon_coefs[task]   = clf.coef_[0].astype(np.float64)
    with_intercept[task] = float(clf.intercept_[0])
    pred = clf.predict_proba(xh)[:,1]
    auc  = roc_auc_score(y_bin, pred)
    print(f"  {task}: AUC={auc:.4f}")


# ── Per-module annotation ──────────────────────────────────────────────────────
print("\nAnnotating modules …")
Z_path_csc = sp.csc_matrix(Z_path)
Z_ben_csc  = sp.csc_matrix(Z_ben)
Z_stab_csc = sp.csc_matrix(Z_stab[stab_mask])
Z_act_csc  = sp.csc_matrix(Z_act[act_mask])

module_rows = []
for _, row in df_mod_raw.iterrows():
    mid       = int(row["module"])
    mod_size  = int(row["size"])
    top_lats_enr = [int(x) for x in str(row["top_latents_by_enr"]).split(",")]

    # Parse all latents in this module — not stored in CSV, reconstruct
    # from latent_modules CSV (only top-5 by enrichment stored).
    # Re-derive by loading module membership from the co-activation npz.
    # Skip for now — approximate using stored top_latents_by_enr as representatives.
    mod_lats = top_lats_enr   # best-5 representatives

    n_mod = len(mod_lats)
    if n_mod == 0: continue
    thresh = max(1, int(np.ceil(n_mod * MOD_THRESH)))   # ≥25% of module latents active

    # Disease activation rate
    Z_path_mod = np.asarray(Z_path_csc[:, mod_lats].todense())
    fires_path = (Z_path_mod > 0).sum(axis=1).A1 if hasattr((Z_path_mod > 0).sum(axis=1), 'A1') else (Z_path_mod > 0).sum(axis=1)
    rate_dis = float((fires_path >= thresh).mean())

    Z_ben_mod = np.asarray(Z_ben_csc[:, mod_lats].todense())
    fires_ben = (Z_ben_mod > 0).sum(axis=1)
    rate_ben = float((fires_ben >= thresh).mean())

    disease_specificity = np.log2((rate_dis + 1e-6) / (rate_ben + 1e-6))

    # Phenotype linkage — mean fires per phenotype class
    pheno_link = {}
    for task, (Z_t_csc, y_t) in [
        ("destab", (Z_stab_csc, y_stab_lab)),
        ("stab",   (Z_stab_csc, y_stab_lab)),
        ("gof",    (Z_act_csc,  y_act_lab)),
        ("lof",    (Z_act_csc,  y_act_lab)),
    ]:
        cls = {"destab":2,"stab":0,"gof":2,"lof":0}[task]
        neg = {"destab":1,"stab":1,"gof":1,"lof":1}[task]
        idx_pos = np.where(y_t == cls)[0]
        idx_neg = np.where(y_t == neg)[0]
        Z_t_mod = np.asarray(Z_t_csc[:, mod_lats].todense())
        rate_pos = float((Z_t_mod[idx_pos] > 0).mean()) if len(idx_pos) else 0.0
        rate_neg = float((Z_t_mod[idx_neg] > 0).mean()) if len(idx_neg) else 0.0
        pheno_link[f"pheno_{task}_vs_ctrl"] = round(rate_pos - rate_neg, 5)

    # Aggregate decoder effect on each recon probe
    decoder_effects = {}
    for task, coef in recon_coefs.items():
        # Decoder direction for each latent in module, weighted by its disease mean activation
        mean_act = np.asarray(Z_path_csc[:, mod_lats].mean(axis=0)).ravel()  # (n_mod,)
        effect = sum(float(W_dec_diff[:, lat].astype(np.float64) @ coef) * float(mean_act[i])
                     for i, lat in enumerate(mod_lats))
        decoder_effects[f"decoder_{task.split('_')[0]}"] = round(effect, 6)

    # Protein diversity (ClinVar pathogenic only)
    cv_path_n = int(path_mask.sum())
    Z_path_mod_cv = Z_path_mod[:cv_path_n]
    fires_cv = (Z_path_mod_cv > 0).sum(axis=1)
    heavy_cv = fires_cv >= thresh
    prots_in_module = path_prot_ids[heavy_cv]
    n_unique = len(np.unique(prots_in_module))
    top_prots = (pd.Series(prots_in_module).value_counts().head(5).index.tolist()
                 if len(prots_in_module) else [])

    module_rows.append({
        "module":              mid,
        "n_module_latents":    mod_size,
        "mean_enr":            float(row["mean_enr"]),
        "disease_specificity": round(disease_specificity, 4),
        "rate_disease":        round(rate_dis, 5),
        "rate_benign":         round(rate_ben, 5),
        "n_unique_prots":      n_unique,
        "top_prots":           ",".join(map(str, top_prots)),
        "top_latents_by_enr":  row["top_latents_by_enr"],
        **pheno_link,
        **decoder_effects,
    })

df_ann = pd.DataFrame(module_rows)
# composite score: disease specificity × max absolute decoder effect
dec_cols = [c for c in df_ann.columns if c.startswith("decoder_")]
df_ann["max_abs_decoder"] = df_ann[dec_cols].abs().max(axis=1)
df_ann["score"] = df_ann["disease_specificity"] * df_ann["max_abs_decoder"]
df_ann = df_ann.sort_values("score", ascending=False)
df_ann.to_csv(LA / f"module_annotations_{NAME}.csv", index=False)

print(f"\nTop 15 modules by disease_specificity × |decoder_effect|:")
view_cols = ["module","n_module_latents","disease_specificity","max_abs_decoder",
             "score","n_unique_prots","top_latents_by_enr"] + dec_cols
print(df_ann[view_cols].head(15).to_string(index=False))

# ── Module annotation heatmap ─────────────────────────────────────────────────
print("\nSaving module annotation heatmap …")
top20 = df_ann.head(20)
fig, ax = plt.subplots(figsize=(9, 7))
heat_cols = ["disease_specificity"] + [c for c in df_ann.columns if "pheno_" in c] + dec_cols
mat = top20[heat_cols].values.astype(float)
vmax = np.nanpercentile(np.abs(mat), 95)
im = ax.imshow(mat.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax.set_xticks(range(len(top20)))
ax.set_xticklabels([f"M{int(r['module'])}(n={int(r['n_module_latents'])})"
                    for _,r in top20.iterrows()], rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(len(heat_cols)))
ax.set_yticklabels(heat_cols, fontsize=8)
ax.set_title(f"{NAME} — Module pathomechanism annotation (top 20 by score)",
             fontsize=10, fontweight="bold")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(str(LA / f"module_annotation_heatmap_{NAME}.png"), dpi=130, bbox_inches="tight")
plt.close()

print(f"\nDone. Outputs in {LA}")
