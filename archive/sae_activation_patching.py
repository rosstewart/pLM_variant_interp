# %% [markdown]
# # SAE Activation Patching + Injection
#
# For each binary task (destab/stab vs neutral; GoF/LoF vs wt-like) and each model
# (Collab SAE, D5), trains two probes and runs two interventions:
#
# **Probes:**
# - `sparse_l1` — L1 logistic on raw sparse ΔZ (16384 or 8192-dim), C=0.01
# - `recon_l1`  — L1 logistic on reconstructed diff x_hat = ΔZ @ W_dec.T (1024-dim), C=0.1
#
# **Interventions (per candidate latent i):**
# - `patch`  — zero latent i's contribution; re-evaluate frozen probe
# - `inject` — set latent i to its mean positive-class training value for all negative
#              test examples; re-evaluate frozen probe
#
# Uses a linear shortcut: logit after intervention = logit_base ± col_i × effect_i.
# No per-latent matrix operations needed.
#
# **Score distribution plots**: for the top-N_SCORE_PLOT latents by |Δ AUC|, KDE of
# logistic scores for pos/neg class before (dashed) and after (solid) each intervention.

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

COLLAB_SAE_PATH = Path("/data/karna/model_weights/sae_weights/t5/trainer_0/t5_layer20_topk256_ef16.pt")
D5_WEIGHTS      = V2_DIR / "v2_model_d5_topk.pt"

FIRING_THRESH = 0.05    # latents active in >5% of target-bin training variants
N_TOP_PLOT    = 30      # latents in bar charts
N_SCORE_PLOT  = 3       # top latents shown in score distribution plots
C_L1_SPARSE   = 0.01   # heavy L1 on 16384/8192-dim sparse features (λ=100)
C_L1_RECON    = 0.1    # heavy L1 on 1024-dim reconstructed features (λ=10)

STAB_DDG_THRESH = -1.0
NEUT_DDG_MAX    =  0.5
DEST_DDG_THRESH =  1.5

_AA3_TO_1 = {
    "Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E",
    "Gly":"G","His":"H","Ile":"I","Leu":"L","Lys":"K","Met":"M","Phe":"F",
    "Pro":"P","Ser":"S","Thr":"T","Trp":"W","Tyr":"Y","Val":"V",
}
_ACT_CSV = Path("/data/ross/assay_calibration/labelseq_dataframe_processed.csv.gz")

print("Config loaded.")

# %% [markdown]
# ## 2. Load SAE Decoder Weights

# %%
print("Loading Collab SAE decoder weights …")
_DL_ROOT = V2_DIR / "pLMinterp"
sys.path.insert(0, str(_DL_ROOT))
from dictionary_learning.dictionary_learning.trainers.top_k import AutoEncoderTopK

collab_sae = AutoEncoderTopK.from_pretrained(str(COLLAB_SAE_PATH), device="cpu")
collab_sae.eval()
W_dec_collab = collab_sae.decoder.weight.detach().numpy().astype(np.float32)  # (1024, 16384)
print(f"  W_dec_collab: {W_dec_collab.shape}")
del collab_sae

print("Loading D5 TopKSAE decoder weights …")
EF_TOPK = 4
K_TOPK  = 128

class TopKSAE(nn.Module):
    def __init__(self, in_dim=2048, ef=EF_TOPK, k=K_TOPK):
        super().__init__()
        self.k = k
        self.d = ef * in_dim
        self.encoder = nn.Linear(in_dim, self.d, bias=True)
        self.decoder = nn.Linear(self.d, in_dim, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(in_dim))

    def encode(self, x):
        pre_act = self.encoder(x - self.b_dec)
        topk_vals, topk_idx = pre_act.topk(self.k, dim=-1, sorted=False)
        topk_vals = torch.clamp(topk_vals, min=0)
        z = torch.zeros_like(pre_act).scatter_(-1, topk_idx, topk_vals)
        return z, topk_vals

    def forward(self, x):
        return self.encode(x)

model_d5 = TopKSAE(in_dim=2048)
state = torch.load(D5_WEIGHTS, map_location="cpu")
model_d5.load_state_dict(state)
model_d5.eval()

W_dec_d5      = model_d5.decoder.weight.detach().numpy().astype(np.float32)   # (2048, 8192)
b_dec_d5      = model_d5.b_dec.detach().numpy().astype(np.float32)
W_dec_d5_diff = (W_dec_d5[1024:] - W_dec_d5[:1024]).astype(np.float32)       # (1024, 8192)
b_dec_d5_diff = (b_dec_d5[1024:] - b_dec_d5[:1024]).astype(np.float32)
print(f"  W_dec_d5: {W_dec_d5.shape}  W_dec_d5_diff: {W_dec_d5_diff.shape}")
del model_d5, state

# %% [markdown]
# ## 3. Load Stability Data + Reconstructed Diffs

# %%
print("Loading stability data …")

valid_mask    = np.load(STAB_CACHE / "valid_mask.npy").astype(bool)
dz_pos_stab   = np.load(STAB_CACHE / "dz_pos.npy")
dz_neg_stab   = np.load(STAB_CACHE / "dz_neg.npy")
ddg_stab      = np.load(STAB_CACHE / "ddg_valid.npy")
pid_stab      = np.load(STAB_CACHE / "protein_ids_valid.npy", allow_pickle=True)

dz_stab_dense = (dz_pos_stab - dz_neg_stab).astype(np.float32)
dz_stab       = sp.csr_matrix(dz_stab_dense)
del dz_pos_stab, dz_neg_stab

print("  Computing collab SAE reconstructed stab diffs …")
x_hat_collab_stab = dz_stab_dense @ W_dec_collab.T    # (N_valid, 1024)

print("  Loading D5 MegaScale sparse encodings …")
z_d5_stab       = sp.load_npz(str(MS_CACHE / "ms_z_d5_sparse.npz"))[valid_mask]
z_d5_stab_dense = z_d5_stab.toarray().astype(np.float32)
print("  Computing D5 reconstructed stab diffs …")
x_hat_d5_stab   = z_d5_stab_dense @ W_dec_d5_diff.T + b_dec_d5_diff   # (N_valid, 1024)

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

# %% [markdown]
# ## 4. Load Activity Data + Reconstructed Diffs

# %%
print("Loading activity data …")

Z_d5_act   = np.load(ACT_CACHE / "z_d5.npy").astype(np.float32)
dz_pos_act = np.load(ACT_CACHE / "dz_pos.npy")
dz_neg_act = np.load(ACT_CACHE / "dz_neg.npy")
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

dz_act_dense = (dz_pos_act - dz_neg_act).astype(np.float32)
dz_act       = sp.csr_matrix(dz_act_dense)
del dz_pos_act, dz_neg_act

print("  Computing collab SAE reconstructed activity diffs …")
x_hat_collab_act = dz_act_dense @ W_dec_collab.T    # (N_act, 1024)

print("  Computing D5 reconstructed activity diffs …")
x_hat_d5_act = Z_d5_act @ W_dec_d5_diff.T + b_dec_d5_diff  # (N_act, 1024)

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
print(f"  Activity within-protein split: train={len(act_train_idx):,}  test={len(act_test_idx):,}")

# %% [markdown]
# ## 5. Utilities

# %%
def _sigmoid(x):
    return (1.0 / (1.0 + np.exp(-np.clip(x.astype(np.float64), -30, 30)))).astype(np.float32)


def train_l1_probe(X, y, train_idx, C):
    """L1 logistic regression; accepts sparse CSR or dense X.
    Uses liblinear for binary (10-50x faster than saga) and saga for multiclass."""
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
    """Latents active in >thresh fraction of training examples in bin_mask."""
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
    """
    Run patching and injection for both sparse and recon probes.

    Linear shortcut: logit after intervention = logit_base ± col_i × effect_i.
    coef_ in OvR points toward classes[1]; we flip sign when pos_class is classes[0].

    Returns (df_results, baseline_scores, after_scores)
    - baseline_scores: {"sparse": (pos_arr, neg_arr), "recon": (pos_arr, neg_arr)}
    - after_scores: {("sparse"|"recon", "patch"|"inject"): {latent_idx: (pos_arr, neg_arr)}}
    """
    is_sparse = sp.issparse(dz_full)

    # Signed coefficients oriented toward pos_class
    sign_s = 1.0 if (pos_class == classes[1]) else -1.0
    sign_r = 1.0 if (pos_class == classes[1]) else -1.0

    coef_s = clf_sparse.coef_[0].astype(np.float64) * sign_s   # (D_sparse,)
    coef_r = clf_recon.coef_[0].astype(np.float64)             # (1024,)

    # CSC for fast column extraction
    if is_sparse:
        dz_te_csc = sp.csc_matrix(dz_full[test_idx])
        dz_tr_csc = sp.csc_matrix(dz_full[train_idx])
    else:
        dz_te_arr = np.asarray(dz_full[test_idx], dtype=np.float64)
        dz_tr_arr = np.asarray(dz_full[train_idx], dtype=np.float64)

    x_hat_te = x_hat_full[test_idx].astype(np.float64)
    y_te      = y_full[test_idx]
    y_tr      = y_full[train_idx]

    pos_te = (y_te == pos_class)
    neg_te = (y_te == neg_class)
    pos_tr = (y_tr == pos_class)

    # Binary y for roc_auc_score (1=pos_class, 0=neg_class)
    y_te_bin = pos_te.astype(np.int8)

    # Baseline logits (once per task)
    if is_sparse:
        logits_s_base = (np.asarray(dz_te_csc @ clf_sparse.coef_[0]).ravel()
                         + clf_sparse.intercept_[0]) * sign_s
    else:
        logits_s_base = (dz_te_arr @ clf_sparse.coef_[0].astype(np.float64)
                         + clf_sparse.intercept_[0]) * sign_s

    logits_r_base = (x_hat_te @ coef_r * sign_r
                     + clf_recon.intercept_[0] * sign_r)

    def _auc(logits):
        try:    return float(roc_auc_score(y_te_bin, logits))
        except: return float("nan")

    base_auc_s = _auc(logits_s_base)
    base_auc_r = _auc(logits_r_base)

    # Decoder effect per latent: W_dec_diff[:, i] @ coef_r (oriented toward pos_class)
    decoder_effects = W_dec_diff.T.astype(np.float64) @ (coef_r * sign_r)  # (D,)

    all_recs = []
    keys = [("sparse","patch"), ("recon","patch"), ("sparse","inject"), ("recon","inject")]
    after_scores = {k: {} for k in keys}

    # ── Vectorized extraction of all candidate columns ─────────────────────────
    # col_mat_te: (N_test, N_cands)  col_mat_tr: (N_train, N_cands)
    print("  Extracting candidate columns …")
    if is_sparse:
        col_mat_te = np.asarray(dz_te_csc[:, candidates].todense(), dtype=np.float64)
        col_mat_tr = np.asarray(dz_tr_csc[:, candidates].todense(), dtype=np.float64)
    else:
        col_mat_te = dz_te_arr[:, candidates].astype(np.float64)
        col_mat_tr = dz_tr_arr[:, candidates].astype(np.float64)

    # Injection mean per candidate: mean of positive-class training fires (from train)
    pos_tr_mat    = col_mat_tr[pos_tr]                            # (N_pos_tr, N_cands)
    fires         = pos_tr_mat != 0                               # (N_pos_tr, N_cands)
    mean_vals     = np.where(fires.any(0),
                             np.where(fires, pos_tr_mat, 0).sum(0) / np.maximum(fires.sum(0), 1),
                             0.0)                                  # (N_cands,)

    # Injection deltas: (N_test, N_cands) — only neg_te rows differ from 0
    delta_mat = np.zeros_like(col_mat_te)
    delta_mat[neg_te] = mean_vals[None, :] - col_mat_te[neg_te]  # broadcast

    # Signed effects per candidate
    cs_cands = coef_s[candidates]           # (N_cands,)
    de_cands = decoder_effects[candidates]  # (N_cands,)

    # All four logit matrices: (N_test, N_cands)
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
    """KDE of logistic scores: pos (red) vs neg (blue), dashed=before, solid=after."""
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
# ## 6. Unified Task Loop

# %%
# Pre-compute binary train/test subsets for each task
mask_dn = mask_s_dest | mask_s_neut
mask_sn = mask_s_stab | mask_s_neut
mask_gw = mask_gof | mask_wt
mask_lw = mask_lof | mask_wt

tr_dn, te_dn = _binary_subset(stab_train_idx, stab_test_idx, mask_dn)
tr_sn, te_sn = _binary_subset(stab_train_idx, stab_test_idx, mask_sn)
tr_gw, te_gw = _binary_subset(act_train_idx,  act_test_idx,  mask_gw)
tr_lw, te_lw = _binary_subset(act_train_idx,  act_test_idx,  mask_lw)

# TASKS: (task_name, model, dataset, pos_class, neg_class, y, tr, te,
#          dz_for_firing+sparse_probe, x_hat, W_dec_diff)
TASKS = [
    ("destab_vs_neut", "Collab_SAE", "stability", 2, 1, y_stab, tr_dn, te_dn,
     dz_stab, x_hat_collab_stab, W_dec_collab),
    ("stab_vs_neut",   "Collab_SAE", "stability", 0, 1, y_stab, tr_sn, te_sn,
     dz_stab, x_hat_collab_stab, W_dec_collab),
    ("destab_vs_neut", "D5",         "stability", 2, 1, y_stab, tr_dn, te_dn,
     z_d5_stab, x_hat_d5_stab, W_dec_d5_diff),
    ("stab_vs_neut",   "D5",         "stability", 0, 1, y_stab, tr_sn, te_sn,
     z_d5_stab, x_hat_d5_stab, W_dec_d5_diff),
    ("gof_vs_wt",  "Collab_SAE", "activity", 2, 1, y_act, tr_gw, te_gw,
     dz_act, x_hat_collab_act, W_dec_collab),
    ("lof_vs_wt",  "Collab_SAE", "activity", 0, 1, y_act, tr_lw, te_lw,
     dz_act, x_hat_collab_act, W_dec_collab),
    ("gof_vs_wt",  "D5",         "activity", 2, 1, y_act, tr_gw, te_gw,
     sp.csr_matrix(Z_d5_act), x_hat_d5_act, W_dec_d5_diff),
    ("lof_vs_wt",  "D5",         "activity", 0, 1, y_act, tr_lw, te_lw,
     sp.csr_matrix(Z_d5_act), x_hat_d5_act, W_dec_d5_diff),
]

all_results  = []
all_plots    = []   # (probe_type, interv, baseline_pos, baseline_neg, after_dict, df_sub, tag)

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
    nnz_s = int((clf_sparse.coef_[0] != 0).sum())
    print(f"    nonzero coef: {nnz_s}")

    print("  Training recon L1 probe …")
    clf_recon,  _       = train_l1_probe(x_hat_full, y_full, train_idx, C_L1_RECON)
    nnz_r = int((clf_recon.coef_[0] != 0).sum())
    print(f"    nonzero coef: {nnz_r}")

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

    # Print top-3 by Δ AUC for each analysis
    for (pt, iv), grp in df.groupby(["probe_type", "intervention"]):
        top = grp.nlargest(3, "delta_auc")
        base = grp["baseline_auc"].iloc[0]
        print(f"  [{pt} {iv}] baseline AUC={base:.4f}  top-3 Δ AUC: "
              + "  ".join(f"L{r.latent_idx}:{r.delta_auc:+.4f}" for _, r in top.iterrows()))

    # Store plot metadata
    safe = f"{model_name}_{dataset}_{task_name}"
    for pt in ["sparse", "recon"]:
        bp, bn = baseline_scores[pt]
        for iv in ["patch", "inject"]:
            key = (pt, iv)
            df_sub = df[(df["probe_type"] == pt) & (df["intervention"] == iv)]
            all_plots.append((bp, bn, after_scores[key], df_sub.copy(),
                              f"{model_name}  {dataset}: {task_name}  [{pt} {iv}]",
                              OUT_DIR / f"scoredist_{safe}_{pt}_{iv}.png",
                              OUT_DIR / f"barchart_{safe}_{pt}_{iv}.png"))

# %% [markdown]
# ## 7. Save CSV + Plots

# %%
if all_results:
    df_all = pd.concat(all_results, ignore_index=True)
    out_csv = OUT_DIR / "activation_patching_results.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df_all)} rows → {out_csv}")
else:
    df_all = pd.DataFrame()
    print("No results to save.")

for (bp, bn, after_dict, df_sub, title, score_path, bar_path) in all_plots:
    # ── Score distribution plot ───────────────────────────────────────────────
    plot_score_dists(bp, bn, after_dict, df_sub, N_SCORE_PLOT, title, score_path)

    # ── Bar chart ─────────────────────────────────────────────────────────────
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
