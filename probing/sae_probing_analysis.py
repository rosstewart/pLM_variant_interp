# %% [markdown]
# # SAE Probing Analysis — L1 Logistic Regression over Sparse Latents
#
# Trains L1-regularized probing classifiers (and Lasso for regression) using SAE sparse
# latent representations as features. Evaluates generalization via leave-one-protein-out
# (LOPO) cross-validation. Compares SAE features against a raw ProtT5 baseline.
#
# **Datasets:**
# - **Stability**: MegaScale ΔΔG (~230k variants, 298 proteins). Bins: stab/neutral/destab.
# - **Activity**: DMS assay (17 genes). Bins: LoF/wt-like/GoF.
#
# **Models probed:**
# - **Collab SAE ΔZ** (16384-dim signed): z_VT − z_WT at ProtT5 layer-20.
# - **D5 TopKSAE** (8192-dim): encoding of concat(WT_final, VT_final).
#
# **Baselines** (no L1, raw ProtT5 representations):
# - Collab baseline: layer-20 VT−WT diff (1024-dim)
# - D5 baseline: final-layer VT−WT diff (1024-dim)

# %%
import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import scipy.sparse as sp

from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score, roc_auc_score,
    r2_score, mean_squared_error,
)

# %% [markdown]
# ## 1. Config

# %%
STAB_CACHE   = Path("/data/ross/interp/collab_sae_cache")
ACT_CACHE    = Path("/data/ross/interp/activity_sae_cache")
MS_CACHE     = Path("/data/ross/interp")
V2_DIR       = Path("/home/rcstewart/ppi_lossgain/sparse_bottleneck")
OUT_DIR      = Path("/home/rcstewart/ppi_lossgain/sparse_bottleneck")
MEGASCALE_PKL = "/data/ross/ppi_lossgain/interaction_loss/megascale_preprocessed/preprocessed.pkl"

# L1 hyperparameter grid — sklearn C = 1/λ (inverse regularization strength).
# C=0.01 → λ=100 (very sparse: 0-2 nonzero features)
# C=10   → λ=0.1  (~3-5 nonzero features)
# C=100  → λ=0.01 (hundreds of nonzero features, nearly dense)
C_VALUES = [0.01, 0.1, 1, 10, 100]

# Stability 3-class bins (extreme bins only — mildly stab/destab excluded)
STAB_DDG_THRESH  = -1.0   # ΔΔG < this → class 0 (stabilizing)
NEUT_DDG_MAX     =  0.5   # |ΔΔG| < this → class 1 (neutral)
DEST_DDG_THRESH  =  1.5   # ΔΔG ≥ this → class 2 (destabilizing)

N_TOP_FEATURES   = 50     # features to plot in importance bar charts

# Downsample stability training set for fast exploratory runs.
# Set to None to use the full ~215k training samples.
STAB_TRAIN_SUBSAMPLE = 20_000

print("Config loaded.")

# %% [markdown]
# ## 2. Load Stability Data
#
# Collab SAE: signed ΔZ = dz_pos − dz_neg (16384-dim).
# D5 on MegaScale: loaded as sparse npz, restricted to collab valid_mask.
# Baselines: layer-20 and final-layer VT−WT mutation diffs (1024-dim each).
#
# Train/test split: uses the pre-computed protein-grouped split from preprocessed.pkl
# (same split used for SAE training in v2.py), mapped into valid-mask space.
# This avoids LOPO over 298 proteins which would be extremely slow.

# %%
import pickle
print("Loading stability data …")

valid_mask       = np.load(STAB_CACHE / "valid_mask.npy").astype(bool)
dz_pos_stab      = np.load(STAB_CACHE / "dz_pos.npy")
dz_neg_stab      = np.load(STAB_CACHE / "dz_neg.npy")
h_wt_stab        = np.load(STAB_CACHE / "layer20_wt.npy")
h_vt_stab        = np.load(STAB_CACHE / "layer20_vt.npy")
ddg_stab         = np.load(STAB_CACHE / "ddg_valid.npy")
pid_stab         = np.load(STAB_CACHE / "protein_ids_valid.npy", allow_pickle=True)

# Signed ΔZ — convert immediately to sparse CSR (ΔZ has k=256/16384 ≈ 3% density per
# sample; sparse saves ~15 GB dense → ~500 MB and makes batch extraction fast)
dz_stab = sp.csr_matrix((dz_pos_stab - dz_neg_stab).astype(np.float32))
del dz_pos_stab, dz_neg_stab
print(f"  dz_stab sparse: nnz={dz_stab.nnz:,}  density={dz_stab.nnz/np.prod(dz_stab.shape):.2%}")

# Baseline: layer-20 VT−WT mutation diff
baseline_stab_collab = (h_vt_stab - h_wt_stab).astype(np.float32)
del h_wt_stab, h_vt_stab

# D5 MegaScale encodings (sparse) — restrict to same valid_mask rows
print("  Loading D5 MegaScale sparse encodings …")
Z_ms_d5_full = sp.load_npz(str(MS_CACHE / "ms_z_d5_sparse.npz"))  # (N_ms, 8192)
Z_d5_stab    = Z_ms_d5_full[valid_mask]                             # (N_valid, 8192)
del Z_ms_d5_full

# D5 baseline: final-layer VT−WT diff (dense; restrict by valid_mask)
print("  Loading ms_x_diff for D5 baseline …")
x_diff_full       = np.load(MS_CACHE / "ms_x_diff.npy")            # (N_ms, 1024)
baseline_stab_d5  = x_diff_full[valid_mask]
del x_diff_full

print(f"  N_valid={len(ddg_stab):,}  N_proteins={len(np.unique(pid_stab))}")
print(f"  dz_stab={dz_stab.shape}  Z_d5_stab={Z_d5_stab.shape}")

# ── Load pre-computed protein-grouped train/test split from pkl ───────────────
# Split indices are in N_ms space; map them into N_valid (valid_mask) space.
print("  Loading pkl train/test splits …")
with open(MEGASCALE_PKL, "rb") as f:
    _ms = pickle.load(f)
_splits = _ms["splits"]
del _ms

# Build mapping: ms_index → valid_index (only for rows that passed valid_mask)
_valid_positions = np.where(valid_mask)[0]          # shape (N_valid,), values in [0, N_ms)
_ms_to_valid     = {int(ms_i): v_i for v_i, ms_i in enumerate(_valid_positions)}

def _map_split(ms_indices):
    """Map pkl split indices (N_ms space) → valid space, dropping rows not in valid_mask."""
    return np.array([_ms_to_valid[i] for i in ms_indices if i in _ms_to_valid])

stab_train_idx = _map_split(_splits["train"])
stab_test_idx  = _map_split(_splits["test"])
print(f"  Stability train={len(stab_train_idx):,}  test={len(stab_test_idx):,} (after valid_mask)")

if STAB_TRAIN_SUBSAMPLE and len(stab_train_idx) > STAB_TRAIN_SUBSAMPLE:
    _rng = np.random.default_rng(42)
    stab_train_idx = np.sort(_rng.choice(stab_train_idx, STAB_TRAIN_SUBSAMPLE, replace=False))
    print(f"  Downsampled stability train → {len(stab_train_idx):,}")

# Build stability labels
mask_s_stab = ddg_stab <  STAB_DDG_THRESH               # highly stabilizing
mask_s_neut = np.abs(ddg_stab) < NEUT_DDG_MAX           # near neutral
mask_s_dest = ddg_stab >= DEST_DDG_THRESH               # highly destabilizing
mask_s_3cls = mask_s_stab | mask_s_neut | mask_s_dest   # 3-class subset

y_stab_3    = np.full(len(ddg_stab), -1, dtype=np.int8)
y_stab_3[mask_s_stab] = 0
y_stab_3[mask_s_neut] = 1
y_stab_3[mask_s_dest] = 2
y_stab_cont = ddg_stab.copy()

print(f"  3-class: stab={mask_s_stab.sum():,}  neut={mask_s_neut.sum():,}  "
      f"dest={mask_s_dest.sum():,}  (excluded={1 - mask_s_3cls.mean():.1%})")

# %% [markdown]
# ## 3. Load Activity Data

# %%
print("Loading activity data …")

Z_d5_act    = np.load(ACT_CACHE / "z_d5.npy")
dz_pos_act  = np.load(ACT_CACHE / "dz_pos.npy")
dz_neg_act  = np.load(ACT_CACHE / "dz_neg.npy")
h_l20_wt    = np.load(ACT_CACHE / "layer20_wt.npy")
h_l20_vt    = np.load(ACT_CACHE / "layer20_vt.npy")
h_lfn_wt    = np.load(ACT_CACHE / "final_layer_wt.npy")
h_lfn_vt    = np.load(ACT_CACHE / "final_layer_vt.npy")
pid_act     = np.load(ACT_CACHE / "protein_ids.npy", allow_pickle=True)

# Retrieve scores and bins by reloading the activity CSV (small overhead for 17 genes)
_ACT_CSV = Path("/data/ross/assay_calibration/labelseq_dataframe_processed.csv.gz")
_valid_idx = np.load(ACT_CACHE / "valid_idx.npy")
_AA3_TO_1 = {
    "Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E",
    "Gly":"G","His":"H","Ile":"I","Leu":"L","Lys":"K","Met":"M","Phe":"F",
    "Pro":"P","Ser":"S","Thr":"T","Trp":"W","Tyr":"Y","Val":"V",
}
import re as _re
_varre = _re.compile(r'^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$')
def _pv(v):
    m = _varre.match(v)
    if m is None: return None, None, None
    r3,p,a3=m.groups(); return _AA3_TO_1.get(r3), int(p), _AA3_TO_1.get(a3)
def _abin(s):
    if s < 0.75: return "LoF"
    if 0.80 <= s <= 1.20: return "wt_like"
    if s > 1.25: return "GoF"
    return None

_df = pd.read_csv(_ACT_CSV, compression="gzip")
_df_act = _df[_df["assay"]=="activity"].copy()
_df_act = _df_act[_df_act["variant"].str.match(r'^p\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2}$', na=False)]
_parsed = [_pv(v) for v in _df_act["variant"]]
_df_act["aa_ref"] = [p[0] for p in _parsed]
_df_act["aa_pos"] = [p[1] for p in _parsed]
_df_act["aa_alt"] = [p[2] for p in _parsed]
_df_act = _df_act.dropna(subset=["aa_ref","aa_pos","aa_alt"])
_df_var = (_df_act.groupby(["uniprot_accession","Gene","aa_ref","aa_pos","aa_alt"])["average score"]
           .mean().reset_index())
_df_var.rename(columns={"average score":"score"}, inplace=True)
_df_var["bin"] = _df_var["score"].map(_abin)
_df_var = _df_var[_df_var["bin"].notna()].copy().reset_index(drop=True)
scores_act_full = _df_var["score"].to_numpy(np.float32)
bins_act_full   = _df_var["bin"].tolist()
scores_act  = scores_act_full[_valid_idx]
bins_act    = [bins_act_full[i] for i in _valid_idx]
del _df, _df_act, _df_var, scores_act_full, bins_act_full

dz_act = sp.csr_matrix((dz_pos_act - dz_neg_act).astype(np.float32))
del dz_pos_act, dz_neg_act
baseline_act_collab = (h_l20_vt - h_l20_wt).astype(np.float32)
baseline_act_d5     = (h_lfn_vt - h_lfn_wt).astype(np.float32)
del h_l20_wt, h_l20_vt, h_lfn_wt, h_lfn_vt

mask_gof = np.array([b == "GoF"     for b in bins_act])
mask_lof = np.array([b == "LoF"     for b in bins_act])
mask_wt  = np.array([b == "wt_like" for b in bins_act])

y_act_3    = np.full(len(bins_act), -1, dtype=np.int8)
y_act_3[mask_lof] = 0
y_act_3[mask_wt]  = 1
y_act_3[mask_gof] = 2
y_act_cont = scores_act.copy()

print(f"  N_act={len(bins_act):,}  N_proteins={len(np.unique(pid_act))}")
print(f"  LoF={mask_lof.sum()}  wt-like={mask_wt.sum()}  GoF={mask_gof.sum()}")
print(f"  dz_act={dz_act.shape}  Z_d5_act={Z_d5_act.shape}")

# Within-protein 80/20 stratified split: hold out 20% of variants per protein
# per label bin. Tests whether SAE features rank variants within a protein —
# the cross-protein split had near-chance results because GoF/LoF signals are
# protein-family-specific and don't transfer across gene families.
_rng_act   = np.random.default_rng(42)
_train_idx, _test_idx = [], []
for _prot in np.unique(pid_act):
    _pidx = np.where(pid_act == _prot)[0]
    # stratify by bin label so each split keeps all classes represented
    for _lbl in np.unique(y_act_3[_pidx]):
        _lidx = _pidx[y_act_3[_pidx] == _lbl]
        if len(_lidx) < 2:
            _train_idx.extend(_lidx)   # too few to split — keep in train
            continue
        _n_test = max(1, int(len(_lidx) * 0.20))
        _perm   = _rng_act.permutation(len(_lidx))
        _test_idx.extend(_lidx[_perm[:_n_test]])
        _train_idx.extend(_lidx[_perm[_n_test:]])

act_train_idx = np.array(sorted(_train_idx))
act_test_idx  = np.array(sorted(_test_idx))
print(f"  Activity within-protein split: train={len(act_train_idx):,}  test={len(act_test_idx):,}")

# %% [markdown]
# ## 4. Training Utilities — sklearn L1 (saga / Lasso)
#
# L1 classifiers use **LogisticRegression(penalty='l1', solver='saga')**, which applies
# exact ISTA-style soft-thresholding in Cython — provably produces exact zeros.
# Adam + proximal thresholding was dropped: Adam's adaptive step (~lr/sqrt(eps) ≈ 100)
# overwhelms the threshold, preventing true sparsity even at strong regularization.
#
# saga accepts scipy sparse CSR directly; no densification or GPU needed.
# C = 1/λ; small C = strong regularization = few nonzero features.

# %%

def _fit_classify(X, y, train_idx, test_idx, C):
    classes = np.unique(y[train_idx])
    binary  = len(classes) == 2
    multi   = "ovr" if binary else "multinomial"
    clf = LogisticRegression(
        penalty="l1", C=C, solver="saga", multi_class=multi,
        class_weight="balanced", max_iter=5000, tol=1e-4)
    clf.fit(X[train_idx], y[train_idx])
    probs = clf.predict_proba(X[test_idx])
    preds = clf.predict(X[test_idx])
    coef  = clf.coef_   # (1, D) binary or (K, D) multiclass
    return preds, probs, coef, classes


def _fit_regress(X, y, train_idx, test_idx, alpha):
    reg = Lasso(alpha=alpha, max_iter=10000, tol=1e-4, selection="cyclic")
    reg.fit(X[train_idx], y[train_idx])
    preds = reg.predict(X[test_idx])
    return preds, reg.coef_


def _classify_metrics(y_te, preds, probs, coef, classes):
    binary    = len(classes) == 2
    coef_mean = coef[0] if binary else coef.mean(0)
    result = dict(
        test_acc          = accuracy_score(y_te, preds),
        test_balanced_acc = balanced_accuracy_score(y_te, preds),
        coef_mean         = coef_mean,
        n_nonzero_coef    = int((coef_mean != 0).sum()),
    )
    try:
        result["test_auc"] = (roc_auc_score(y_te, probs[:, 1]) if binary else
                              roc_auc_score(y_te, probs, multi_class="ovr",
                                            average="macro", labels=classes))
    except Exception:
        result["test_auc"] = float("nan")
    return result


def _regress_metrics(y_te, preds, coef):
    return dict(
        test_r2        = r2_score(y_te, preds),
        test_mse       = mean_squared_error(y_te, preds),
        coef_mean      = coef,
        n_nonzero_coef = int((coef != 0).sum()),
    )


# ── Train/test split wrappers ──────────────────────────────────────────────────

def tt_classify(X, y, train_idx, test_idx, C, multiclass=True):
    preds, probs, coef, classes = _fit_classify(X, y, train_idx, test_idx, C)
    return _classify_metrics(y[test_idx], preds, probs, coef, classes)


def tt_regress(X, y, train_idx, test_idx, alpha):
    preds, coef = _fit_regress(X, y, train_idx, test_idx, alpha)
    return _regress_metrics(y[test_idx], preds, coef)


# ── sklearn L2 baselines (no sparsity penalty — upper-bound reference) ────────

def _dense(X, idx):
    rows = X[idx]
    return rows.toarray().astype(np.float32) if sp.issparse(rows) else np.asarray(rows, np.float32)

def tt_baseline_classify(X, y, train_idx, test_idx, multiclass=True):
    X_tr, X_te = _dense(X, train_idx), _dense(X, test_idx)
    y_tr, y_te = y[train_idx], y[test_idx]
    classes    = np.unique(y_tr)
    binary     = len(classes) == 2
    scaler     = StandardScaler(); X_tr = scaler.fit_transform(X_tr); X_te = scaler.transform(X_te)
    solver     = "lbfgs" if (multiclass and not binary) else "liblinear"
    multi      = "multinomial" if (multiclass and not binary) else "ovr"
    clf = LogisticRegression(penalty="l2", C=1.0, solver=solver, multi_class=multi,
                             max_iter=2000, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    y_pred, y_prob = clf.predict(X_te), clf.predict_proba(X_te)
    result = dict(test_acc=accuracy_score(y_te, y_pred),
                  test_balanced_acc=balanced_accuracy_score(y_te, y_pred))
    try:
        result["test_auc"] = (roc_auc_score(y_te, y_prob[:, 1]) if binary else
                              roc_auc_score(y_te, y_prob, multi_class="ovr",
                                            average="macro", labels=classes))
    except Exception:
        result["test_auc"] = float("nan")
    return result

def tt_baseline_regress(X, y, train_idx, test_idx):
    X_tr, X_te = _dense(X, train_idx), _dense(X, test_idx)
    scaler = StandardScaler(); X_tr = scaler.fit_transform(X_tr); X_te = scaler.transform(X_te)
    reg = Ridge(alpha=1.0); reg.fit(X_tr, y[train_idx])
    y_pred = reg.predict(X_te)
    return dict(test_r2=r2_score(y[test_idx], y_pred),
                test_mse=mean_squared_error(y[test_idx], y_pred))

def run_tt_grid(X, y, train_idx, test_idx, c_values, task="classify",
                multiclass=True, label=""):
    """Train/test grid sweep using sklearn L1 saga (classify) or Lasso (regress)."""
    results = {}
    fn = tt_classify if task == "classify" else tt_regress
    for C in tqdm(c_values, desc=label):
        kw = dict(C=C, multiclass=multiclass) if task == "classify" else dict(alpha=1.0/C)
        results[C] = fn(X, y, train_idx, test_idx, **kw)
    return results

# %% [markdown]
# ## 5. Stability — Collab SAE Probing
#
# Uses the pre-computed protein-grouped train/test split from preprocessed.pkl.
# Avoids LOPO over 298 proteins (which would take hours); one train/test fit per C.

# %%
print("=== Stability: Collab SAE (16384-dim ΔZ) ===")

# Build per-task train/test index arrays restricted to each bin subset.
# All indices are in N_valid space (rows of dz_stab, y_stab_3, etc.)
idx3      = np.where(mask_s_3cls)[0]
idx3_set  = set(idx3.tolist())
mask_sn   = mask_s_stab | mask_s_neut
mask_dn   = mask_s_dest | mask_s_neut

def _split_subset(global_train, global_test, subset_mask):
    """Restrict pre-computed train/test arrays to a bin subset."""
    sub_idx  = np.where(subset_mask)[0]
    sub_set  = set(sub_idx.tolist())
    tr = np.array([i for i in global_train if i in sub_set])
    te = np.array([i for i in global_test  if i in sub_set])
    return tr, te

tr3, te3       = _split_subset(stab_train_idx, stab_test_idx, mask_s_3cls)
tr_sn, te_sn   = _split_subset(stab_train_idx, stab_test_idx, mask_sn)
tr_dn, te_dn   = _split_subset(stab_train_idx, stab_test_idx, mask_dn)

STAB_COLLAB = {}

STAB_COLLAB["3class"] = run_tt_grid(
    dz_stab, y_stab_3, tr3, te3, C_VALUES,
    task="classify", multiclass=True, label="Collab 3-class")

STAB_COLLAB["stab_vs_neut"] = run_tt_grid(
    dz_stab, y_stab_3, tr_sn, te_sn, C_VALUES,
    task="classify", multiclass=False, label="Collab stab/neut")

STAB_COLLAB["dest_vs_neut"] = run_tt_grid(
    dz_stab, y_stab_3, tr_dn, te_dn, C_VALUES,
    task="classify", multiclass=False, label="Collab dest/neut")

STAB_COLLAB["regression"] = run_tt_grid(
    dz_stab, y_stab_cont, stab_train_idx, stab_test_idx, C_VALUES,
    task="regress", label="Collab regression")

print("  Baseline (L2 logistic / Ridge) …")
STAB_COLLAB["baseline_3class"]     = tt_baseline_classify(
    baseline_stab_collab, y_stab_3, tr3, te3, multiclass=True)
STAB_COLLAB["baseline_stab_neut"]  = tt_baseline_classify(
    baseline_stab_collab, y_stab_3, tr_sn, te_sn, multiclass=False)
STAB_COLLAB["baseline_dest_neut"]  = tt_baseline_classify(
    baseline_stab_collab, y_stab_3, tr_dn, te_dn, multiclass=False)
STAB_COLLAB["baseline_regression"] = tt_baseline_regress(
    baseline_stab_collab, y_stab_cont, stab_train_idx, stab_test_idx)

print("Done.")

# %% [markdown]
# ## 6. Stability — D5 Probing

# %%
print("=== Stability: D5 TopKSAE (8192-dim Z) ===")

STAB_D5 = {}

STAB_D5["3class"] = run_tt_grid(
    Z_d5_stab, y_stab_3, tr3, te3, C_VALUES,
    task="classify", multiclass=True, label="D5 3-class")

STAB_D5["stab_vs_neut"] = run_tt_grid(
    Z_d5_stab, y_stab_3, tr_sn, te_sn, C_VALUES,
    task="classify", multiclass=False, label="D5 stab/neut")

STAB_D5["dest_vs_neut"] = run_tt_grid(
    Z_d5_stab, y_stab_3, tr_dn, te_dn, C_VALUES,
    task="classify", multiclass=False, label="D5 dest/neut")

STAB_D5["regression"] = run_tt_grid(
    Z_d5_stab, y_stab_cont, stab_train_idx, stab_test_idx, C_VALUES,
    task="regress", label="D5 regression")

print("  Baseline (L2 logistic / Ridge) …")
STAB_D5["baseline_3class"]     = tt_baseline_classify(
    baseline_stab_d5, y_stab_3, tr3, te3, multiclass=True)
STAB_D5["baseline_stab_neut"]  = tt_baseline_classify(
    baseline_stab_d5, y_stab_3, tr_sn, te_sn, multiclass=False)
STAB_D5["baseline_dest_neut"]  = tt_baseline_classify(
    baseline_stab_d5, y_stab_3, tr_dn, te_dn, multiclass=False)
STAB_D5["baseline_regression"] = tt_baseline_regress(
    baseline_stab_d5, y_stab_cont, stab_train_idx, stab_test_idx)

print("Done.")

# %% [markdown]
# ## 7. Activity — Collab SAE Probing

# %%
print("=== Activity: Collab SAE (16384-dim ΔZ) ===")
mask_lw   = mask_lof | mask_wt
mask_gw   = mask_gof | mask_wt
mask_3cls = mask_lof | mask_wt | mask_gof

def _act_split(subset_mask):
    """Restrict activity protein-grouped train/test to a bin subset."""
    sub = np.where(subset_mask)[0]
    sub_set = set(sub.tolist())
    tr = np.array([i for i in act_train_idx if i in sub_set])
    te = np.array([i for i in act_test_idx  if i in sub_set])
    return tr, te

tr_act3, te_act3 = _act_split(mask_3cls)
tr_lw,   te_lw   = _act_split(mask_lw)
tr_gw,   te_gw   = _act_split(mask_gw)

ACT_COLLAB = {}

ACT_COLLAB["3class"] = run_tt_grid(
    dz_act, y_act_3, tr_act3, te_act3, C_VALUES,
    task="classify", multiclass=True, label="Collab act 3-class")

ACT_COLLAB["lof_vs_wt"] = run_tt_grid(
    dz_act, y_act_3, tr_lw, te_lw, C_VALUES,
    task="classify", multiclass=False, label="Collab LoF/wt")

ACT_COLLAB["gof_vs_wt"] = run_tt_grid(
    dz_act, y_act_3, tr_gw, te_gw, C_VALUES,
    task="classify", multiclass=False, label="Collab GoF/wt")

ACT_COLLAB["regression"] = run_tt_grid(
    dz_act, y_act_cont, act_train_idx, act_test_idx, C_VALUES,
    task="regress", label="Collab act regression")

print("  Baseline …")
ACT_COLLAB["baseline_3class"]     = tt_baseline_classify(
    baseline_act_collab, y_act_3, tr_act3, te_act3, multiclass=True)
ACT_COLLAB["baseline_lof_wt"]     = tt_baseline_classify(
    baseline_act_collab, y_act_3, tr_lw, te_lw, multiclass=False)
ACT_COLLAB["baseline_gof_wt"]     = tt_baseline_classify(
    baseline_act_collab, y_act_3, tr_gw, te_gw, multiclass=False)
ACT_COLLAB["baseline_regression"] = tt_baseline_regress(
    baseline_act_collab, y_act_cont, act_train_idx, act_test_idx)
print("Done.")

# %% [markdown]
# ## 8. Activity — D5 Probing

# %%
print("=== Activity: D5 TopKSAE (8192-dim Z) ===")

ACT_D5 = {}

ACT_D5["3class"] = run_tt_grid(
    Z_d5_act, y_act_3, tr_act3, te_act3, C_VALUES,
    task="classify", multiclass=True, label="D5 act 3-class")

ACT_D5["lof_vs_wt"] = run_tt_grid(
    Z_d5_act, y_act_3, tr_lw, te_lw, C_VALUES,
    task="classify", multiclass=False, label="D5 LoF/wt")

ACT_D5["gof_vs_wt"] = run_tt_grid(
    Z_d5_act, y_act_3, tr_gw, te_gw, C_VALUES,
    task="classify", multiclass=False, label="D5 GoF/wt")

ACT_D5["regression"] = run_tt_grid(
    Z_d5_act, y_act_cont, act_train_idx, act_test_idx, C_VALUES,
    task="regress", label="D5 act regression")

print("  Baseline …")
ACT_D5["baseline_3class"]     = tt_baseline_classify(
    baseline_act_d5, y_act_3, tr_act3, te_act3, multiclass=True)
ACT_D5["baseline_lof_wt"]     = tt_baseline_classify(
    baseline_act_d5, y_act_3, tr_lw, te_lw, multiclass=False)
ACT_D5["baseline_gof_wt"]     = tt_baseline_classify(
    baseline_act_d5, y_act_3, tr_gw, te_gw, multiclass=False)
ACT_D5["baseline_regression"] = tt_baseline_regress(
    baseline_act_d5, y_act_cont, act_train_idx, act_test_idx)
print("Done.")

# %% [markdown]
# ## 9. Results Summary Tables

# %%
def _fmt(d, key, fmt=".3f"):
    v = d.get(key, float("nan"))
    if isinstance(v, np.ndarray): return "—"
    return format(v, fmt) if not np.isnan(float(v)) else "—"

def print_classify_table(grid, baseline, task_names, c_values):
    print(f"\n{'Task':<18} {'C':>6} {'BalAcc':>8} {'AUC':>8} {'Nonzero':>8} │ {'BL BalAcc':>10} {'BL AUC':>8}")
    print("─" * 85)
    for task, bl_key in task_names:
        bl = baseline.get(bl_key, {})
        for i, C in enumerate(c_values):
            r = grid.get(task, {}).get(C, {})
            pfx = f"{task:<18}" if i == 0 else " " * 18
            nnz = r.get("n_nonzero_coef", "—")
            nnz_fmt = f"{nnz:>8}" if isinstance(nnz, int) else f"{'—':>8}"
            blfmt = (f"{_fmt(bl,'test_balanced_acc'):>10} {_fmt(bl,'test_auc'):>8}"
                     if i == 0 else " " * 19)
            print(f"{pfx} {C:>6} {_fmt(r,'test_balanced_acc'):>8} {_fmt(r,'test_auc'):>8}"
                  f"{nnz_fmt} │ {blfmt}")

def print_regress_table(grid, baseline, task_names, c_values):
    print(f"\n{'Task':<18} {'C':>6} {'Test R²':>9} {'MSE':>8} {'Nonzero':>8} │ {'BL R²':>7} {'BL MSE':>8}")
    print("─" * 85)
    for task, bl_key in task_names:
        bl = baseline.get(bl_key, {})
        for i, C in enumerate(c_values):
            r = grid.get(task, {}).get(C, {})
            pfx = f"{task:<18}" if i == 0 else " " * 18
            nnz = r.get("n_nonzero_coef", "—")
            nnz_fmt = f"{nnz:>8}" if isinstance(nnz, int) else f"{'—':>8}"
            blfmt = (f"{_fmt(bl,'test_r2'):>7} {_fmt(bl,'test_mse'):>8}"
                     if i == 0 else " " * 16)
            print(f"{pfx} {C:>6} {_fmt(r,'test_r2'):>9} {_fmt(r,'test_mse'):>8}"
                  f"{nnz_fmt} │ {blfmt}")

# ── Stability ─────────────────────────────────────────────────────────────────
print("=" * 80)
print("STABILITY  —  Collab SAE ΔZ (16384-dim)")
print("=" * 80)
print_classify_table(
    STAB_COLLAB, STAB_COLLAB,
    [("3class","baseline_3class"), ("stab_vs_neut","baseline_stab_neut"),
     ("dest_vs_neut","baseline_dest_neut")],
    C_VALUES)
print_regress_table(
    STAB_COLLAB, STAB_COLLAB,
    [("regression","baseline_regression")],
    C_VALUES)

print("\n" + "=" * 80)
print("STABILITY  —  D5 TopKSAE (8192-dim)")
print("=" * 80)
print_classify_table(
    STAB_D5, STAB_D5,
    [("3class","baseline_3class"), ("stab_vs_neut","baseline_stab_neut"),
     ("dest_vs_neut","baseline_dest_neut")],
    C_VALUES)
print_regress_table(
    STAB_D5, STAB_D5,
    [("regression","baseline_regression")],
    C_VALUES)

print("\n" + "=" * 80)
print("ACTIVITY  —  Collab SAE ΔZ (16384-dim)")
print("=" * 80)
print_classify_table(
    ACT_COLLAB, ACT_COLLAB,
    [("3class","baseline_3class"), ("lof_vs_wt","baseline_lof_wt"),
     ("gof_vs_wt","baseline_gof_wt")],
    C_VALUES)
print_regress_table(
    ACT_COLLAB, ACT_COLLAB,
    [("regression","baseline_regression")],
    C_VALUES)

print("\n" + "=" * 80)
print("ACTIVITY  —  D5 TopKSAE (8192-dim)")
print("=" * 80)
print_classify_table(
    ACT_D5, ACT_D5,
    [("3class","baseline_3class"), ("lof_vs_wt","baseline_lof_wt"),
     ("gof_vs_wt","baseline_gof_wt")],
    C_VALUES)
print_regress_table(
    ACT_D5, ACT_D5,
    [("regression","baseline_regression")],
    C_VALUES)

# %% [markdown]
# ## 10. Top Feature Weights (Interpretation)
#
# For each model × dataset, pick the best C value (highest AUC on 3-class)
# and plot the top features by |mean coefficient|.
# Positive coefficient: feature activation → destab/GoF class.
# Negative coefficient: feature activation → stab/LoF class.

# %%
def plot_top_features(coef, n_top, title, out_path=None):
    """Bar chart of top-N features by |coefficient|, colored by sign."""
    top_idx = np.argsort(np.abs(coef))[::-1][:n_top]
    top_coef = coef[top_idx]
    colors = ["tab:red" if c > 0 else "tab:blue" for c in top_coef]
    fig, ax = plt.subplots(figsize=(10, max(4, n_top * 0.22)))
    y_pos = np.arange(n_top)[::-1]
    ax.barh(y_pos, top_coef, color=colors, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"F{i}" for i in top_idx], fontsize=7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Mean LOPO coefficient (red=+, blue=−)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


# ─ Stability: Collab SAE ─────────────────────────────────────────────────────
best_C_stab_collab = max(C_VALUES,
    key=lambda C: STAB_COLLAB["3class"].get(C, {}).get("test_auc", -1))
coef_stab_collab = STAB_COLLAB["3class"][best_C_stab_collab].get("coef_mean")
if coef_stab_collab is not None:
    plot_top_features(coef_stab_collab, N_TOP_FEATURES,
        f"Stability — Collab SAE ΔZ  (C={best_C_stab_collab}, 3-class)\n"
        f"red=destab  blue=stab",
        OUT_DIR / "probe_stab_collab_top_features.png")

# ─ Stability: D5 ─────────────────────────────────────────────────────────────
best_C_stab_d5 = max(C_VALUES,
    key=lambda C: STAB_D5["3class"].get(C, {}).get("test_auc", -1))
coef_stab_d5 = STAB_D5["3class"][best_C_stab_d5].get("coef_mean")
if coef_stab_d5 is not None:
    plot_top_features(coef_stab_d5, N_TOP_FEATURES,
        f"Stability — D5 TopKSAE  (C={best_C_stab_d5}, 3-class)\n"
        f"red=destab  blue=stab",
        OUT_DIR / "probe_stab_d5_top_features.png")

# ─ Activity: Collab SAE ───────────────────────────────────────────────────────
best_C_act_collab = max(C_VALUES,
    key=lambda C: ACT_COLLAB["3class"].get(C, {}).get("test_auc", -1))
coef_act_collab = ACT_COLLAB["3class"][best_C_act_collab].get("coef_mean")
if coef_act_collab is not None:
    plot_top_features(coef_act_collab, N_TOP_FEATURES,
        f"Activity — Collab SAE ΔZ  (C={best_C_act_collab}, 3-class)\n"
        f"red=GoF  blue=LoF",
        OUT_DIR / "probe_act_collab_top_features.png")

# ─ Activity: D5 ──────────────────────────────────────────────────────────────
best_C_act_d5 = max(C_VALUES,
    key=lambda C: ACT_D5["3class"].get(C, {}).get("test_auc", -1))
coef_act_d5 = ACT_D5["3class"][best_C_act_d5].get("coef_mean")
if coef_act_d5 is not None:
    plot_top_features(coef_act_d5, N_TOP_FEATURES,
        f"Activity — D5 TopKSAE  (C={best_C_act_d5}, 3-class)\n"
        f"red=GoF  blue=LoF",
        OUT_DIR / "probe_act_d5_top_features.png")

# %% [markdown]
# ## 11. AUC vs C Curve

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("AUC vs L1 Regularization (C)", fontsize=12, fontweight="bold")

configs = [
    (STAB_COLLAB, "Stability — Collab SAE ΔZ",  axes[0, 0],
     [("3class","baseline_3class"),("stab_vs_neut","baseline_stab_neut"),
      ("dest_vs_neut","baseline_dest_neut")]),
    (STAB_D5,     "Stability — D5 TopKSAE",      axes[0, 1],
     [("3class","baseline_3class"),("stab_vs_neut","baseline_stab_neut"),
      ("dest_vs_neut","baseline_dest_neut")]),
    (ACT_COLLAB,  "Activity — Collab SAE ΔZ",    axes[1, 0],
     [("3class","baseline_3class"),("lof_vs_wt","baseline_lof_wt"),
      ("gof_vs_wt","baseline_gof_wt")]),
    (ACT_D5,      "Activity — D5 TopKSAE",        axes[1, 1],
     [("3class","baseline_3class"),("lof_vs_wt","baseline_lof_wt"),
      ("gof_vs_wt","baseline_gof_wt")]),
]
colors_tasks = ["tab:blue", "tab:green", "tab:orange"]
task_labels  = ["3-class", "stab/neut (or LoF/wt)", "dest/neut (or GoF/wt)"]

for grid, title, ax, tasks in configs:
    for (task, bl_key), color, tlabel in zip(tasks, colors_tasks, task_labels):
        vals = [grid.get(task, {}).get(C, {}).get("test_auc", np.nan)
                for C in C_VALUES]
        ax.plot(C_VALUES, vals, marker="o", color=color, label=tlabel)
        bl = grid.get(bl_key, {}).get("test_auc", np.nan)
        if not np.isnan(float(bl)):
            ax.axhline(bl, linestyle="--", color=color, alpha=0.5, lw=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("C (1/λ)  — higher = less regularization", fontsize=9)
    ax.set_ylabel("AUC", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(OUT_DIR / "probe_auc_vs_C.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 12. Save Summary CSVs

# %%
rows = []
for model_name, grid, tasks in [
    ("Collab_SAE_stab",  STAB_COLLAB, ["3class","stab_vs_neut","dest_vs_neut","regression"]),
    ("D5_stab",          STAB_D5,     ["3class","stab_vs_neut","dest_vs_neut","regression"]),
    ("Collab_SAE_act",   ACT_COLLAB,  ["3class","lof_vs_wt","gof_vs_wt","regression"]),
    ("D5_act",           ACT_D5,      ["3class","lof_vs_wt","gof_vs_wt","regression"]),
]:
    for task in tasks:
        bl_key = ("baseline_" + ("3class" if task == "3class"
                  else "stab_neut" if task == "stab_vs_neut"
                  else "dest_neut" if task == "dest_vs_neut"
                  else "lof_wt"    if task == "lof_vs_wt"
                  else "gof_wt"    if task == "gof_vs_wt"
                  else "regression"))
        bl = grid.get(bl_key, {})
        for C in C_VALUES:
            r = grid.get(task, {}).get(C, {})
            rows.append(dict(
                model=model_name, task=task, C=C,
                test_balanced_acc = r.get("test_balanced_acc", np.nan),
                test_auc          = r.get("test_auc", np.nan),
                test_r2           = r.get("test_r2", np.nan),
                test_mse          = r.get("test_mse", np.nan),
                n_nonzero_coef    = r.get("n_nonzero_coef", np.nan),
                baseline_bal_acc  = bl.get("test_balanced_acc", np.nan),
                baseline_auc      = bl.get("test_auc", np.nan),
                baseline_r2       = bl.get("test_r2", np.nan),
            ))

df_results = pd.DataFrame(rows)
df_results.to_csv(OUT_DIR / "probe_results.csv", index=False)
print(f"Saved probe_results.csv  ({len(df_results)} rows)")
print(df_results.to_string(max_rows=20))
