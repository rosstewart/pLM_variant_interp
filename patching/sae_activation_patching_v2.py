# %% [markdown]
# # SAE Activation Patching v2 — Combined Multi-Source TopK SAEs
#
# Same patching + injection analysis as sae_activation_patching.py, but run on the
# six TopK SAEs trained on ClinVar + gnomAD + HGMD combined data.
#
# Models (from /data/ross/ppi_lossgain/interaction_loss/sae_weights/combined/):
#   concat_ef4_k128  — concat(WT,VT) 2048-dim, EF=4, K=128  (matches D5 architecture)
#   concat_ef4_k64   — concat(WT,VT) 2048-dim, EF=4, K=64
#   concat_ef1_k128  — concat(WT,VT) 2048-dim, EF=1, K=128
#   diff_ef4_k64     — diff(VT-WT)   1024-dim, EF=4, K=64   (matches D7 architecture)
#   diff_ef4_k32     — diff(VT-WT)   1024-dim, EF=4, K=32
#   diff_ef1_k64     — diff(VT-WT)   1024-dim, EF=1, K=64
#
# For concat models: W_dec_diff = W_dec[1024:] - W_dec[:1024]  (VT - WT decoder component)
# For diff models:   W_dec_diff = W_dec                         (decoder already operates on diffs)

# %%
import os, sys, warnings, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import gaussian_kde
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import re as _re

# %% [markdown]
# ## 1. Config

# %%
STAB_CACHE    = Path("/data/ross/interp/collab_sae_cache")
ACT_CACHE     = Path("/data/ross/interp/activity_sae_cache")
MS_CACHE      = Path("/data/ross/interp")
V2_DIR        = Path("/home/rcstewart/ppi_lossgain/sparse_bottleneck")
OUT_DIR       = Path("/home/rcstewart/ppi_lossgain/sparse_bottleneck")
MEGASCALE_PKL = "/data/ross/ppi_lossgain/interaction_loss/megascale_preprocessed/preprocessed.pkl"
COMBINED_DIR  = Path("/data/ross/ppi_lossgain/interaction_loss/sae_weights/combined")
ENCODE_CACHE  = Path("/data/ross/interp/combined_sae_cache")
ENCODE_CACHE.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

FIRING_THRESH = 0.05
N_TOP_PLOT    = 30
N_SCORE_PLOT  = 3
C_L1_SPARSE   = 0.01
C_L1_RECON    = 0.1

STAB_DDG_THRESH = -1.0
NEUT_DDG_MAX    =  0.5
DEST_DDG_THRESH =  1.5

_AA3_TO_1 = {
    "Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E",
    "Gly":"G","His":"H","Ile":"I","Leu":"L","Lys":"K","Met":"M","Phe":"F",
    "Pro":"P","Ser":"S","Thr":"T","Trp":"W","Tyr":"Y","Val":"V",
}
_ACT_CSV = Path("/data/ross/assay_calibration/labelseq_dataframe_processed.csv.gz")

# Model configurations: (name, input_type, in_dim, ef, k)
MODEL_CONFIGS = [
    ("concat_ef4_k128", "concat", 2048, 4, 128),
    ("concat_ef4_k64",  "concat", 2048, 4,  64),
    ("concat_ef1_k128", "concat", 2048, 1, 128),
    ("diff_ef4_k64",    "diff",   1024, 4,  64),
    ("diff_ef4_k32",    "diff",   1024, 4,  32),
    ("diff_ef1_k64",    "diff",   1024, 1,  64),
]

print(f"Device: {DEVICE}")
print(f"Models: {[c[0] for c in MODEL_CONFIGS]}")

# %% [markdown]
# ## 2. TopKSAE Definition

# %%
class TopKSAE(nn.Module):
    def __init__(self, in_dim: int, ef: int, k: int):
        super().__init__()
        d = ef * in_dim
        self.k = k; self.d = d; self.in_dim = in_dim; self.ef = ef
        self.encoder = nn.Linear(in_dim, d)
        self.decoder = nn.Linear(d, in_dim, bias=False)
        self.register_buffer("b_dec", torch.zeros(in_dim))

    def encode(self, x):
        pre_act = torch.relu(self.encoder(x - self.b_dec))
        topk_vals, topk_idx = pre_act.topk(self.k, dim=-1, sorted=False)
        z = torch.zeros_like(pre_act).scatter_(-1, topk_idx, topk_vals)
        return z, topk_idx

    def forward(self, x):
        z, _ = self.encode(x)
        return z, self.decoder(z) + self.b_dec

# %% [markdown]
# ## 3. Load Stability Data

# %%
print("Loading stability data …")

valid_mask = np.load(STAB_CACHE / "valid_mask.npy").astype(bool)
ddg_stab   = np.load(STAB_CACHE / "ddg_valid.npy")
pid_stab   = np.load(STAB_CACHE / "protein_ids_valid.npy", allow_pickle=True)

mask_s_stab = ddg_stab <  STAB_DDG_THRESH
mask_s_neut = np.abs(ddg_stab) < NEUT_DDG_MAX
mask_s_dest = ddg_stab >= DEST_DDG_THRESH

y_stab = np.full(len(ddg_stab), -1, dtype=np.int8)
y_stab[mask_s_stab] = 0
y_stab[mask_s_neut] = 1
y_stab[mask_s_dest] = 2

print(f"  N_valid={len(ddg_stab):,}  stab={mask_s_stab.sum():,}  "
      f"neut={mask_s_neut.sum():,}  dest={mask_s_dest.sum():,}")

print("  Loading pkl train/test split …")
with open(MEGASCALE_PKL, "rb") as f:
    _ms = pickle.load(f)
_splits = _ms["splits"]
del _ms

_valid_pos   = np.where(valid_mask)[0]
_ms_to_valid = {int(ms_i): v_i for v_i, ms_i in enumerate(_valid_pos)}

def _map_split(ms_idx):
    return np.array([_ms_to_valid[i] for i in ms_idx if i in _ms_to_valid])

stab_train_idx = _map_split(_splits["train"])
stab_test_idx  = _map_split(_splits["test"])
print(f"  Stability train={len(stab_train_idx):,}  test={len(stab_test_idx):,}")

# Load raw MegaScale embeddings for encoding through new models
print("  Loading MegaScale raw embeddings …")
ms_feats_concat = np.load(V2_DIR / "megascale_feats.npy")   # (N_ms, 2048) concat(WT,VT)
ms_feats_diff   = np.load(MS_CACHE / "ms_x_diff.npy")       # (N_ms, 1024) VT-WT
# Subset to valid_mask variants
ms_feats_concat = ms_feats_concat[valid_mask]
ms_feats_diff   = ms_feats_diff[valid_mask]
print(f"  ms_feats_concat: {ms_feats_concat.shape}  ms_feats_diff: {ms_feats_diff.shape}")

# %% [markdown]
# ## 4. Load Activity Data

# %%
print("Loading activity data …")

pid_act    = np.load(ACT_CACHE / "protein_ids.npy", allow_pickle=True)
_valid_idx = np.load(ACT_CACHE / "valid_idx.npy")

_varre = _re.compile(r'^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$')
def _pv(v):
    m = _varre.match(v)
    if m is None: return None, None, None
    r3, p, a3 = m.groups()
    return _AA3_TO_1.get(r3), int(p), _AA3_TO_1.get(a3)
def _abin(s):
    if s < 0.75:          return "LoF"
    if 0.80 <= s <= 1.20: return "wt_like"
    if s > 1.25:          return "GoF"
    return None

_df = pd.read_csv(_ACT_CSV, compression="gzip")
_df_act = _df[_df["assay"] == "activity"].copy()
_df_act = _df_act[_df_act["variant"].str.match(r'^p\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2}$', na=False)]
_parsed = [_pv(v) for v in _df_act["variant"]]
_df_act["aa_ref"] = [p[0] for p in _parsed]
_df_act["aa_pos"] = [p[1] for p in _parsed]
_df_act["aa_alt"] = [p[2] for p in _parsed]
_df_act = _df_act.dropna(subset=["aa_ref", "aa_pos", "aa_alt"])
_df_var = (_df_act.groupby(["uniprot_accession", "Gene", "aa_ref", "aa_pos", "aa_alt"])["average score"]
           .mean().reset_index())
_df_var.rename(columns={"average score": "score"}, inplace=True)
_df_var["bin"] = _df_var["score"].map(_abin)
_df_var = _df_var[_df_var["bin"].notna()].copy().reset_index(drop=True)
bins_act_full = _df_var["bin"].tolist()
bins_act      = [bins_act_full[i] for i in _valid_idx]
del _df, _df_act, _df_var, bins_act_full

mask_gof = np.array([b == "GoF"     for b in bins_act])
mask_lof = np.array([b == "LoF"     for b in bins_act])
mask_wt  = np.array([b == "wt_like" for b in bins_act])

y_act = np.full(len(bins_act), -1, dtype=np.int8)
y_act[mask_lof] = 0
y_act[mask_wt]  = 1
y_act[mask_gof] = 2

print(f"  N_act={len(bins_act):,}  LoF={mask_lof.sum()}  wt-like={mask_wt.sum()}  GoF={mask_gof.sum()}")

_rng_act = np.random.default_rng(42)
_train_idx, _test_idx = [], []
for _prot in np.unique(pid_act):
    _pidx = np.where(pid_act == _prot)[0]
    for _lbl in np.unique(y_act[_pidx]):
        _lidx = _pidx[y_act[_pidx] == _lbl]
        if len(_lidx) < 2:
            _train_idx.extend(_lidx)
            continue
        _n_test = max(1, int(len(_lidx) * 0.20))
        _perm   = _rng_act.permutation(len(_lidx))
        _test_idx.extend(_lidx[_perm[:_n_test]])
        _train_idx.extend(_lidx[_perm[_n_test:]])

act_train_idx = np.array(sorted(_train_idx))
act_test_idx  = np.array(sorted(_test_idx))
print(f"  Activity train={len(act_train_idx):,}  test={len(act_test_idx):,}")

# Load raw activity embeddings for encoding through new models
print("  Loading activity raw embeddings …")
act_wt = np.load(ACT_CACHE / "final_layer_wt.npy").astype(np.float32)  # (N_act, 1024)
act_vt = np.load(ACT_CACHE / "final_layer_vt.npy").astype(np.float32)  # (N_act, 1024)
act_feats_concat = np.concatenate([act_wt, act_vt], axis=1)            # (N_act, 2048)
act_feats_diff   = act_vt - act_wt                                      # (N_act, 1024)
del act_wt, act_vt
print(f"  act_feats_concat: {act_feats_concat.shape}  act_feats_diff: {act_feats_diff.shape}")

# %% [markdown]
# ## 5. Encode Through Combined SAE Models

# %%
def encode_with_model(model: TopKSAE, X: np.ndarray, batch_size: int = 1024) -> sp.csr_matrix:
    """Encode X through model on GPU, return sparse CSR Z matrix."""
    model.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(DEVICE)
            z, _ = model(xb)
            parts.append(z.cpu().numpy())
    Z = np.concatenate(parts, axis=0)
    return sp.csr_matrix(Z.astype(np.float32))


# For each model: load weights, extract decoder, encode both datasets, cache to disk
model_data = {}   # name → {"Z_stab", "Z_act", "W_dec_diff", "b_dec_diff", "x_hat_stab", "x_hat_act"}

for name, input_type, in_dim, ef, k in MODEL_CONFIGS:
    print(f"\n{'='*60}")
    print(f"Loading [{name}]  input={input_type}  in_dim={in_dim}  EF={ef}  K={k}")

    weights_path = COMBINED_DIR / f"combined_{name}.pt"
    if not weights_path.exists():
        print(f"  WARNING: {weights_path} not found — skipping.")
        continue

    # Encoder cache paths
    z_stab_path = ENCODE_CACHE / f"z_stab_{name}.npz"
    z_act_path  = ENCODE_CACHE / f"z_act_{name}.npz"

    model = TopKSAE(in_dim=in_dim, ef=ef, k=k).to(DEVICE)
    state = torch.load(str(weights_path), map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    # Decoder weights (always on CPU for numpy ops)
    W_dec = model.decoder.weight.detach().cpu().numpy().astype(np.float32)  # (in_dim, dict_size)
    b_dec = model.b_dec.detach().cpu().numpy().astype(np.float32)           # (in_dim,)

    if input_type == "concat":
        # Decoder is trained on concat(WT,VT); diff component = VT decoder - WT decoder
        W_dec_diff = (W_dec[1024:] - W_dec[:1024]).astype(np.float32)       # (1024, dict_size)
        b_dec_diff = (b_dec[1024:] - b_dec[:1024]).astype(np.float32)       # (1024,)
        X_stab = ms_feats_concat
        X_act  = act_feats_concat
    else:
        # Decoder operates directly on diffs
        W_dec_diff = W_dec.astype(np.float32)                                # (1024, dict_size)
        b_dec_diff = b_dec.astype(np.float32)                                # (1024,)
        X_stab = ms_feats_diff
        X_act  = act_feats_diff

    # Encode stability (MegaScale)
    if z_stab_path.exists():
        print(f"  Loading cached Z_stab from {z_stab_path.name} …")
        Z_stab = sp.load_npz(str(z_stab_path))
    else:
        print(f"  Encoding {len(X_stab):,} stability variants …")
        Z_stab = encode_with_model(model, X_stab)
        sp.save_npz(str(z_stab_path), Z_stab)
        print(f"  Saved → {z_stab_path.name}  nnz={Z_stab.nnz:,}")

    # Encode activity
    if z_act_path.exists():
        print(f"  Loading cached Z_act from {z_act_path.name} …")
        Z_act = sp.load_npz(str(z_act_path))
    else:
        print(f"  Encoding {len(X_act):,} activity variants …")
        Z_act = encode_with_model(model, X_act)
        sp.save_npz(str(z_act_path), Z_act)
        print(f"  Saved → {z_act_path.name}  nnz={Z_act.nnz:,}")

    del model, state

    # Reconstructed diffs: Z @ W_dec_diff.T + b_dec_diff
    print("  Computing reconstructed diffs …")
    Z_stab_dense = Z_stab.toarray().astype(np.float32)
    Z_act_dense  = Z_act.toarray().astype(np.float32)
    x_hat_stab   = Z_stab_dense @ W_dec_diff.T + b_dec_diff   # (N_stab, 1024)
    x_hat_act    = Z_act_dense  @ W_dec_diff.T + b_dec_diff   # (N_act,  1024)

    sparsity_stab = 1 - Z_stab.nnz / (Z_stab.shape[0] * Z_stab.shape[1])
    sparsity_act  = 1 - Z_act.nnz  / (Z_act.shape[0]  * Z_act.shape[1])
    print(f"  stab sparsity={sparsity_stab:.3%}  act sparsity={sparsity_act:.3%}")

    model_data[name] = {
        "input_type": input_type,
        "Z_stab":      Z_stab,
        "Z_act":       Z_act,
        "W_dec_diff":  W_dec_diff,
        "b_dec_diff":  b_dec_diff,
        "x_hat_stab":  x_hat_stab,
        "x_hat_act":   x_hat_act,
    }

print(f"\nLoaded {len(model_data)} models: {list(model_data.keys())}")

# %% [markdown]
# ## 6. Utilities (identical to v1)

# %%
def _sigmoid(x):
    return (1.0 / (1.0 + np.exp(-np.clip(x.astype(np.float64), -30, 30)))).astype(np.float32)


def train_l1_probe(X, y, train_idx, C):
    classes = np.unique(y[train_idx])
    binary  = len(classes) == 2
    clf = LogisticRegression(
        penalty="l1", C=C,
        solver="liblinear" if binary else "saga",
        multi_class="ovr",
        class_weight="balanced",
        max_iter=1000, tol=1e-4)
    clf.fit(X[train_idx], y[train_idx])
    return clf, classes


def get_candidate_latents(dz, train_idx, bin_mask, thresh=FIRING_THRESH):
    target_tr = np.array([i for i in train_idx if bin_mask[i]])
    if len(target_tr) == 0:
        return np.array([], dtype=int)
    dz_sub = dz[target_tr]
    fire_rate = (np.asarray((np.abs(dz_sub) > 0).mean(0)).ravel()
                 if sp.issparse(dz_sub) else (np.abs(dz_sub) > 0).mean(0))
    return np.where(fire_rate > thresh)[0]


def _binary_subset(global_tr, global_te, mask):
    s = set(np.where(mask)[0])
    return (np.array([i for i in global_tr if i in s]),
            np.array([i for i in global_te  if i in s]))


def run_patch_and_inject(dz_full, x_hat_full, W_dec_diff,
                          clf_sparse, clf_recon,
                          y_full, train_idx, test_idx,
                          classes, pos_class, neg_class,
                          candidates):
    is_sparse = sp.issparse(dz_full)
    sign_s = 1.0 if (pos_class == classes[1]) else -1.0
    sign_r = 1.0 if (pos_class == classes[1]) else -1.0
    coef_s = clf_sparse.coef_[0].astype(np.float64) * sign_s
    coef_r = clf_recon.coef_[0].astype(np.float64)

    if is_sparse:
        dz_te_csc = sp.csc_matrix(dz_full[test_idx])
        dz_tr_csc = sp.csc_matrix(dz_full[train_idx])
    else:
        dz_te_arr = np.asarray(dz_full[test_idx], dtype=np.float64)
        dz_tr_arr = np.asarray(dz_full[train_idx], dtype=np.float64)

    x_hat_te = x_hat_full[test_idx].astype(np.float64)
    y_te = y_full[test_idx]; y_tr = y_full[train_idx]
    pos_te = (y_te == pos_class); neg_te = (y_te == neg_class); pos_tr = (y_tr == pos_class)
    y_te_bin = pos_te.astype(np.int8)

    if is_sparse:
        logits_s_base = (np.asarray(dz_te_csc @ clf_sparse.coef_[0]).ravel()
                         + clf_sparse.intercept_[0]) * sign_s
    else:
        logits_s_base = (dz_te_arr @ clf_sparse.coef_[0].astype(np.float64)
                         + clf_sparse.intercept_[0]) * sign_s

    logits_r_base = (x_hat_te @ coef_r * sign_r + clf_recon.intercept_[0] * sign_r)

    def _auc(logits):
        try:    return float(roc_auc_score(y_te_bin, logits))
        except: return float("nan")

    base_auc_s = _auc(logits_s_base)
    base_auc_r = _auc(logits_r_base)
    decoder_effects = W_dec_diff.T.astype(np.float64) @ (coef_r * sign_r)

    all_recs = []
    keys = [("sparse","patch"), ("recon","patch"), ("sparse","inject"), ("recon","inject")]
    after_scores = {k: {} for k in keys}

    print("  Extracting candidate columns …")
    if is_sparse:
        col_mat_te = np.asarray(dz_te_csc[:, candidates].todense(), dtype=np.float64)
        col_mat_tr = np.asarray(dz_tr_csc[:, candidates].todense(), dtype=np.float64)
    else:
        col_mat_te = dz_te_arr[:, candidates].astype(np.float64)
        col_mat_tr = dz_tr_arr[:, candidates].astype(np.float64)

    pos_tr_mat = col_mat_tr[pos_tr]
    fires      = pos_tr_mat != 0
    mean_vals  = np.where(fires.any(0),
                          np.where(fires, pos_tr_mat, 0).sum(0) / np.maximum(fires.sum(0), 1),
                          0.0)

    delta_mat = np.zeros_like(col_mat_te)
    delta_mat[neg_te] = mean_vals[None, :] - col_mat_te[neg_te]

    cs_cands = coef_s[candidates]
    de_cands = decoder_effects[candidates]

    logits_s_pat_mat = logits_s_base[:, None] - col_mat_te * cs_cands
    logits_r_pat_mat = logits_r_base[:, None] - col_mat_te * de_cands
    logits_s_inj_mat = logits_s_base[:, None] + delta_mat  * cs_cands
    logits_r_inj_mat = logits_r_base[:, None] + delta_mat  * de_cands

    print("  Computing AUCs …")
    for c_idx, i in enumerate(tqdm(candidates, desc="  AUC loop", leave=False)):
        for key, base_auc, logits_col in [
            (("sparse","patch"),  base_auc_s, logits_s_pat_mat[:, c_idx]),
            (("recon", "patch"),  base_auc_r, logits_r_pat_mat[:, c_idx]),
            (("sparse","inject"), base_auc_s, logits_s_inj_mat[:, c_idx]),
            (("recon", "inject"), base_auc_r, logits_r_inj_mat[:, c_idx]),
        ]:
            probe_type, interv = key
            mod_auc = _auc(logits_col)
            all_recs.append(dict(
                latent_idx   = int(i),
                probe_type   = probe_type,
                intervention = interv,
                baseline_auc = round(base_auc, 5),
                modified_auc = round(mod_auc, 5),
                delta_auc    = round(base_auc - mod_auc, 5),
            ))
            sigs = _sigmoid(logits_col)
            after_scores[key][int(i)] = (sigs[pos_te], sigs[neg_te])

    df = (pd.DataFrame(all_recs)
          .sort_values(["probe_type", "intervention", "delta_auc"],
                       ascending=[True, True, False])
          .reset_index(drop=True))

    baseline_scores = {
        "sparse": (_sigmoid(logits_s_base[pos_te]), _sigmoid(logits_s_base[neg_te])),
        "recon":  (_sigmoid(logits_r_base[pos_te]), _sigmoid(logits_r_base[neg_te])),
    }
    return df, baseline_scores, after_scores


def plot_score_dists(baseline_pos, baseline_neg, after_dict,
                     df_sub, n_top, title, out_path):
    top_lats = (df_sub.nlargest(n_top, "delta_auc")["latent_idx"].tolist()
                if len(df_sub) else [])
    top_lats = [i for i in top_lats if i in after_dict][:n_top]
    if not top_lats:
        return

    fig, axes = plt.subplots(1, len(top_lats), figsize=(5 * len(top_lats), 4), squeeze=False)
    for col, lat_i in enumerate(top_lats):
        ax = axes[0, col]
        after_pos, after_neg = after_dict[lat_i]
        row = df_sub[df_sub["latent_idx"] == lat_i]
        da  = row["delta_auc"].values[0] if len(row) else float("nan")

        for sb, sa, color, label in [
            (baseline_pos, after_pos, "tab:red",  "pos"),
            (baseline_neg, after_neg, "tab:blue", "neg"),
        ]:
            if len(sb) < 3:
                continue
            lo = min(sb.min(), sa.min()); hi = max(sb.max(), sa.max())
            xs = np.linspace(lo, hi, 300)
            try:
                ax.fill_between(xs, gaussian_kde(sb)(xs), alpha=0.07, color=color)
                ax.plot(xs, gaussian_kde(sb)(xs), color=color, lw=1.5, ls="--",
                        label=f"{label} before")
                ax.plot(xs, gaussian_kde(sa)(xs), color=color, lw=1.5, ls="-",
                        label=f"{label} after")
            except Exception:
                pass

        ax.set_title(f"Latent {lat_i}  Δ AUC={da:.4f}", fontsize=9)
        ax.set_xlabel("P(pos class)", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        if col == 0:
            ax.legend(fontsize=7)

    fig.suptitle(title, fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")

# %% [markdown]
# ## 7. Unified Task Loop

# %%
mask_dn = mask_s_dest | mask_s_neut
mask_sn = mask_s_stab | mask_s_neut
mask_gw = mask_gof | mask_wt
mask_lw = mask_lof | mask_wt

tr_dn, te_dn = _binary_subset(stab_train_idx, stab_test_idx, mask_dn)
tr_sn, te_sn = _binary_subset(stab_train_idx, stab_test_idx, mask_sn)
tr_gw, te_gw = _binary_subset(act_train_idx,  act_test_idx,  mask_gw)
tr_lw, te_lw = _binary_subset(act_train_idx,  act_test_idx,  mask_lw)

# Build TASKS from all loaded models
TASKS = []
for name, md in model_data.items():
    Z_s  = md["Z_stab"];    xh_s = md["x_hat_stab"]
    Z_a  = md["Z_act"];     xh_a = md["x_hat_act"]
    Wdiff = md["W_dec_diff"]
    TASKS += [
        ("destab_vs_neut", name, "stability", 2, 1, y_stab, tr_dn, te_dn, Z_s, xh_s, Wdiff),
        ("stab_vs_neut",   name, "stability", 0, 1, y_stab, tr_sn, te_sn, Z_s, xh_s, Wdiff),
        ("gof_vs_wt",      name, "activity",  2, 1, y_act,  tr_gw, te_gw, Z_a, xh_a, Wdiff),
        ("lof_vs_wt",      name, "activity",  0, 1, y_act,  tr_lw, te_lw, Z_a, xh_a, Wdiff),
    ]

print(f"Total tasks: {len(TASKS)}")

all_results = []
all_plots   = []

for (task_name, model_name, dataset, pos_class, neg_class,
     y_full, train_idx, test_idx, dz_full, x_hat_full, W_dec_diff) in TASKS:

    print(f"\n=== {model_name}  {dataset}: {task_name}  "
          f"(pos={pos_class}, neg={neg_class}) ===")

    candidates = get_candidate_latents(dz_full, train_idx,
                                       (y_full == pos_class), FIRING_THRESH)
    print(f"  Candidate latents (fire>{FIRING_THRESH:.0%} in pos-class train): {len(candidates)}")

    if len(candidates) == 0:
        print("  No candidates — skipping.")
        continue

    print("  Training sparse L1 probe …")
    clf_sparse, classes = train_l1_probe(dz_full,    y_full, train_idx, C_L1_SPARSE)
    print(f"    nonzero coef: {int((clf_sparse.coef_[0] != 0).sum())}")

    print("  Training recon L1 probe …")
    clf_recon, _ = train_l1_probe(x_hat_full, y_full, train_idx, C_L1_RECON)
    print(f"    nonzero coef: {int((clf_recon.coef_[0] != 0).sum())}")

    df, baseline_scores, after_scores = run_patch_and_inject(
        dz_full, x_hat_full, W_dec_diff,
        clf_sparse, clf_recon,
        y_full, train_idx, test_idx,
        classes, pos_class, neg_class,
        candidates)

    df["model"]   = model_name
    df["dataset"] = dataset
    df["task"]    = task_name
    all_results.append(df)

    for (pt, iv), grp in df.groupby(["probe_type", "intervention"]):
        top3 = grp.nlargest(3, "delta_auc")
        base = grp["baseline_auc"].iloc[0]
        print(f"  [{pt} {iv}] baseline AUC={base:.4f}  top-3 Δ AUC: "
              + "  ".join(f"L{r.latent_idx}:{r.delta_auc:+.4f}" for _, r in top3.iterrows()))

    safe = f"{model_name}_{dataset}_{task_name}"
    for pt in ["sparse", "recon"]:
        bp, bn = baseline_scores[pt]
        for iv in ["patch", "inject"]:
            key = (pt, iv)
            df_sub = df[(df["probe_type"] == pt) & (df["intervention"] == iv)]
            all_plots.append((bp, bn, after_scores[key], df_sub.copy(),
                              f"{model_name}  {dataset}: {task_name}  [{pt} {iv}]",
                              OUT_DIR / f"v2_scoredist_{safe}_{pt}_{iv}.png",
                              OUT_DIR / f"v2_barchart_{safe}_{pt}_{iv}.png"))

# %% [markdown]
# ## 8. Save CSV + Plots

# %%
if all_results:
    df_all = pd.concat(all_results, ignore_index=True)
    out_csv = OUT_DIR / "activation_patching_results_v2.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df_all)} rows → {out_csv}")
else:
    df_all = pd.DataFrame()
    print("No results to save.")

for (bp, bn, after_dict, df_sub, title, score_path, bar_path) in all_plots:
    plot_score_dists(bp, bn, after_dict, df_sub, N_SCORE_PLOT, title, score_path)

    top = df_sub.nlargest(N_TOP_PLOT, "delta_auc")
    if len(top) == 0:
        continue
    fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.28)))
    colors = ["tab:red" if d > 0 else "tab:blue" for d in top["delta_auc"]]
    y_pos  = np.arange(len(top))[::-1]
    ax.barh(y_pos, top["delta_auc"].values, color=colors, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"L{i}" for i in top["latent_idx"]], fontsize=7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Δ AUC (baseline − modified)  red=helps, blue=hurts", fontsize=9)
    ax.set_title(f"{title}\nbaseline={top['baseline_auc'].iloc[0]:.4f}  "
                 f"top-{len(top)} of {len(df_sub)} candidates",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {bar_path.name}")

print("Done.")
