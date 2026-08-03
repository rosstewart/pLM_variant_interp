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
STAB_CACHE = Path("/data/ross/interp/collab_sae_cache")
ACT_CACHE  = Path("/data/ross/interp/activity_sae_cache")
MS_CACHE   = Path("/data/ross/interp")
V2_DIR     = Path("/home/rcstewart/ppi_lossgain/sparse_bottleneck")
OUT_DIR    = Path("/home/rcstewart/ppi_lossgain/sparse_bottleneck")

# L1 hyperparameter grid — sklearn C = 1/λ
# C=1: strong regularization (very sparse); C=256: weak (many features used)
C_VALUES = [1, 4, 16, 64, 256]

# Lasso alpha grid for regression (alpha = 1/C: 1.0 strong → 0.004 weak)
LASSO_ALPHAS = [1.0 / c for c in C_VALUES]

# Stability 3-class bins (extreme bins only — mildly stab/destab excluded)
STAB_DDG_THRESH  = -1.0   # ΔΔG < this → class 0 (stabilizing)
NEUT_DDG_MAX     =  0.5   # |ΔΔG| < this → class 1 (neutral)
DEST_DDG_THRESH  =  1.5   # ΔΔG ≥ this → class 2 (destabilizing)

LOPO_MAX_ITER    = 2000   # LogisticRegression max_iter
N_TOP_FEATURES   = 50     # features to plot in importance bar charts

print("Config loaded.")

# %% [markdown]
# ## 2. Load Stability Data
#
# Collab SAE: signed ΔZ = dz_pos − dz_neg (16384-dim).
# D5 on MegaScale: loaded as sparse npz, restricted to collab valid_mask.
# Baselines: layer-20 and final-layer VT−WT mutation diffs (1024-dim each).

# %%
print("Loading stability data …")

valid_mask       = np.load(STAB_CACHE / "valid_mask.npy").astype(bool)
dz_pos_stab      = np.load(STAB_CACHE / "dz_pos.npy")
dz_neg_stab      = np.load(STAB_CACHE / "dz_neg.npy")
h_wt_stab        = np.load(STAB_CACHE / "layer20_wt.npy")
h_vt_stab        = np.load(STAB_CACHE / "layer20_vt.npy")
ddg_stab         = np.load(STAB_CACHE / "ddg_valid.npy")
pid_stab         = np.load(STAB_CACHE / "protein_ids_valid.npy", allow_pickle=True)

# Signed ΔZ: (N_valid, 16384); already filtered by valid_mask inside collab notebook
dz_stab = (dz_pos_stab - dz_neg_stab).astype(np.float32)
del dz_pos_stab, dz_neg_stab

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

dz_act = (dz_pos_act - dz_neg_act).astype(np.float32)
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

# %% [markdown]
# ## 4. LOPO Utilities

# %%
def _to_sparse(X):
    """Convert dense ndarray to csr_matrix if not already sparse."""
    if sp.issparse(X):
        return X.tocsr()
    return sp.csr_matrix(X)


def lopo_classify(X, y, protein_ids, C, multiclass=True, max_iter=LOPO_MAX_ITER):
    """
    Leave-one-protein-out L1 logistic regression.
    X: (N, D) array or sparse matrix.
    y: (N,) int labels — must have no -1 (filter before calling).
    Returns: dict with test metrics and mean coefficient vector (D,).
    """
    X_sp      = _to_sparse(X)
    proteins  = np.unique(protein_ids)
    solver    = "saga" if multiclass else "liblinear"
    classes   = np.unique(y)
    n_classes = len(classes)
    binary    = n_classes == 2

    y_true_all, y_pred_all, y_prob_all = [], [], []
    coef_sum = None

    for prot in proteins:
        test_mask  = protein_ids == prot
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        X_tr = X_sp[train_mask]
        X_te = X_sp[test_mask]
        y_tr = y[train_mask]
        y_te = y[test_mask]

        if len(np.unique(y_tr)) < n_classes:
            continue   # skip if train lacks a class

        # Scale: fit on train sparse rows (StandardScaler with_mean=False for sparse)
        scaler = StandardScaler(with_mean=False)
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)

        multi = "multinomial" if (multiclass and not binary) else "ovr"
        clf = LogisticRegression(
            penalty="l1", C=C, solver=solver,
            multi_class=multi, max_iter=max_iter,
            class_weight="balanced", warm_start=False,
        )
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)

        y_true_all.append(y_te)
        y_pred_all.append(y_pred)
        y_prob_all.append(y_prob)

        # Accumulate coefficients
        c = clf.coef_[0] if binary else clf.coef_.mean(0)  # (D,) signed average
        coef_sum = c if coef_sum is None else coef_sum + c

    if not y_true_all:
        return {}

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    y_prob = np.concatenate(y_prob_all, axis=0)
    coef_mean = coef_sum / len(proteins)

    result = dict(
        test_acc          = accuracy_score(y_true, y_pred),
        test_balanced_acc = balanced_accuracy_score(y_true, y_pred),
        coef_mean         = coef_mean,
        n_folds           = len(proteins),
    )
    try:
        if binary:
            result["test_auc"] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            result["test_auc"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro",
                labels=classes)
    except Exception:
        result["test_auc"] = float("nan")

    return result


def lopo_regress(X, y, protein_ids, alpha):
    """
    Leave-one-protein-out Lasso regression.
    Returns dict with test metrics and mean coefficient vector.
    """
    X_sp     = _to_sparse(X)
    proteins = np.unique(protein_ids)

    y_true_all, y_pred_all = [], []
    coef_sum = None

    for prot in proteins:
        test_mask  = protein_ids == prot
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        X_tr = X_sp[train_mask].toarray()
        X_te = X_sp[test_mask].toarray()
        y_tr = y[train_mask]
        y_te = y[test_mask]

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)

        reg = Lasso(alpha=alpha, max_iter=5000, warm_start=False)
        reg.fit(X_tr, y_tr)

        y_pred = reg.predict(X_te)
        y_true_all.append(y_te)
        y_pred_all.append(y_pred)

        coef_sum = reg.coef_.copy() if coef_sum is None else coef_sum + reg.coef_

    if not y_true_all:
        return {}

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    return dict(
        test_r2   = r2_score(y_true, y_pred),
        test_mse  = mean_squared_error(y_true, y_pred),
        coef_mean = coef_sum / len(proteins),
        n_folds   = len(proteins),
    )


def lopo_baseline_classify(X, y, protein_ids, multiclass=True):
    """L2 logistic regression (no sparsity) on raw ProtT5 features."""
    X_sp      = _to_sparse(X)
    proteins  = np.unique(protein_ids)
    binary    = len(np.unique(y)) == 2
    multi     = "multinomial" if (multiclass and not binary) else "ovr"
    solver    = "lbfgs" if multiclass else "liblinear"
    classes   = np.unique(y)

    y_true_all, y_pred_all, y_prob_all = [], [], []

    for prot in proteins:
        test_mask  = protein_ids == prot
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        X_tr = X_sp[train_mask].toarray()
        X_te = X_sp[test_mask].toarray()
        y_tr = y[train_mask]
        y_te = y[test_mask]
        if len(np.unique(y_tr)) < len(classes):
            continue

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)

        clf = LogisticRegression(
            penalty="l2", C=1.0, solver=solver,
            multi_class=multi, max_iter=LOPO_MAX_ITER,
            class_weight="balanced",
        )
        clf.fit(X_tr, y_tr)
        y_true_all.append(y_te)
        y_pred_all.append(clf.predict(X_te))
        y_prob_all.append(clf.predict_proba(X_te))

    if not y_true_all:
        return {}

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    y_prob = np.concatenate(y_prob_all, axis=0)

    result = dict(
        test_acc          = accuracy_score(y_true, y_pred),
        test_balanced_acc = balanced_accuracy_score(y_true, y_pred),
        n_folds           = len(proteins),
    )
    try:
        if binary:
            result["test_auc"] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            result["test_auc"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro", labels=classes)
    except Exception:
        result["test_auc"] = float("nan")
    return result


def lopo_baseline_regress(X, y, protein_ids):
    """Ridge regression (no sparsity) on raw ProtT5 features."""
    X_sp     = _to_sparse(X)
    proteins = np.unique(protein_ids)
    y_true_all, y_pred_all = [], []

    for prot in proteins:
        test_mask  = protein_ids == prot
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        X_tr = X_sp[train_mask].toarray()
        X_te = X_sp[test_mask].toarray()
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)
        reg = Ridge(alpha=1.0)
        reg.fit(X_tr, y[train_mask])
        y_true_all.append(y[test_mask])
        y_pred_all.append(reg.predict(X_te))

    if not y_true_all:
        return {}
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    return dict(test_r2=r2_score(y_true, y_pred),
                test_mse=mean_squared_error(y_true, y_pred),
                n_folds=len(proteins))


def run_grid(X, y, protein_ids, c_values, task="classify", multiclass=True, label=""):
    """Run LOPO across all C values for a single (X, y, protein_ids) combo."""
    results = {}
    fn = lopo_classify if task == "classify" else lopo_regress
    for C in tqdm(c_values, desc=f"{label}  {'classify' if task=='classify' else 'regress'}"):
        kw = dict(C=C, multiclass=multiclass) if task == "classify" else dict(alpha=1.0/C)
        results[C] = fn(X, y, protein_ids, **kw)
    return results

# %% [markdown]
# ## 5. Stability — Collab SAE Probing

# %%
print("=== Stability: Collab SAE (16384-dim ΔZ) ===")
idx3  = np.where(mask_s_3cls)[0]
pidx3 = pid_stab[idx3]

STAB_COLLAB = {}

# 3-class (stab / neutral / destab)
STAB_COLLAB["3class"] = run_grid(
    dz_stab[idx3], y_stab_3[idx3], pidx3, C_VALUES,
    task="classify", multiclass=True, label="Collab 3-class")

# Binary: stab vs neutral
mask_sn = mask_s_stab | mask_s_neut
STAB_COLLAB["stab_vs_neut"] = run_grid(
    dz_stab[mask_sn], y_stab_3[mask_sn], pid_stab[mask_sn], C_VALUES,
    task="classify", multiclass=False, label="Collab stab/neut")

# Binary: destab vs neutral
mask_dn = mask_s_dest | mask_s_neut
STAB_COLLAB["dest_vs_neut"] = run_grid(
    dz_stab[mask_dn], y_stab_3[mask_dn], pid_stab[mask_dn], C_VALUES,
    task="classify", multiclass=False, label="Collab dest/neut")

# Regression on ΔΔG
STAB_COLLAB["regression"] = run_grid(
    dz_stab, y_stab_cont, pid_stab, C_VALUES,
    task="regress", label="Collab regression")

# Baseline (raw layer-20 diff)
print("  Baseline (L2 logistic / Ridge) …")
STAB_COLLAB["baseline_3class"]     = lopo_baseline_classify(
    baseline_stab_collab[idx3], y_stab_3[idx3], pidx3, multiclass=True)
STAB_COLLAB["baseline_stab_neut"]  = lopo_baseline_classify(
    baseline_stab_collab[mask_sn], y_stab_3[mask_sn], pid_stab[mask_sn], multiclass=False)
STAB_COLLAB["baseline_dest_neut"]  = lopo_baseline_classify(
    baseline_stab_collab[mask_dn], y_stab_3[mask_dn], pid_stab[mask_dn], multiclass=False)
STAB_COLLAB["baseline_regression"] = lopo_baseline_regress(
    baseline_stab_collab, y_stab_cont, pid_stab)

print("Done.")

# %% [markdown]
# ## 6. Stability — D5 Probing

# %%
print("=== Stability: D5 TopKSAE (8192-dim Z) ===")

STAB_D5 = {}

STAB_D5["3class"] = run_grid(
    Z_d5_stab[idx3], y_stab_3[idx3], pidx3, C_VALUES,
    task="classify", multiclass=True, label="D5 3-class")

STAB_D5["stab_vs_neut"] = run_grid(
    Z_d5_stab[mask_sn], y_stab_3[mask_sn], pid_stab[mask_sn], C_VALUES,
    task="classify", multiclass=False, label="D5 stab/neut")

STAB_D5["dest_vs_neut"] = run_grid(
    Z_d5_stab[mask_dn], y_stab_3[mask_dn], pid_stab[mask_dn], C_VALUES,
    task="classify", multiclass=False, label="D5 dest/neut")

STAB_D5["regression"] = run_grid(
    Z_d5_stab, y_stab_cont, pid_stab, C_VALUES,
    task="regress", label="D5 regression")

print("  Baseline (L2 logistic / Ridge) …")
STAB_D5["baseline_3class"]     = lopo_baseline_classify(
    baseline_stab_d5[idx3], y_stab_3[idx3], pidx3, multiclass=True)
STAB_D5["baseline_stab_neut"]  = lopo_baseline_classify(
    baseline_stab_d5[mask_sn], y_stab_3[mask_sn], pid_stab[mask_sn], multiclass=False)
STAB_D5["baseline_dest_neut"]  = lopo_baseline_classify(
    baseline_stab_d5[mask_dn], y_stab_3[mask_dn], pid_stab[mask_dn], multiclass=False)
STAB_D5["baseline_regression"] = lopo_baseline_regress(
    baseline_stab_d5, y_stab_cont, pid_stab)

print("Done.")

# %% [markdown]
# ## 7. Activity — Collab SAE Probing

# %%
print("=== Activity: Collab SAE (16384-dim ΔZ) ===")
idx_act3  = np.where(mask_gof | mask_lof | mask_wt)[0]
pid_act3  = pid_act[idx_act3]

ACT_COLLAB = {}

ACT_COLLAB["3class"] = run_grid(
    dz_act[idx_act3], y_act_3[idx_act3], pid_act3, C_VALUES,
    task="classify", multiclass=True, label="Collab act 3-class")

mask_lw = mask_lof | mask_wt
ACT_COLLAB["lof_vs_wt"] = run_grid(
    dz_act[mask_lw], y_act_3[mask_lw], pid_act[mask_lw], C_VALUES,
    task="classify", multiclass=False, label="Collab LoF/wt")

mask_gw = mask_gof | mask_wt
ACT_COLLAB["gof_vs_wt"] = run_grid(
    dz_act[mask_gw], y_act_3[mask_gw], pid_act[mask_gw], C_VALUES,
    task="classify", multiclass=False, label="Collab GoF/wt")

ACT_COLLAB["regression"] = run_grid(
    dz_act, y_act_cont, pid_act, C_VALUES,
    task="regress", label="Collab act regression")

print("  Baseline …")
ACT_COLLAB["baseline_3class"]  = lopo_baseline_classify(
    baseline_act_collab[idx_act3], y_act_3[idx_act3], pid_act3, multiclass=True)
ACT_COLLAB["baseline_lof_wt"]  = lopo_baseline_classify(
    baseline_act_collab[mask_lw], y_act_3[mask_lw], pid_act[mask_lw], multiclass=False)
ACT_COLLAB["baseline_gof_wt"]  = lopo_baseline_classify(
    baseline_act_collab[mask_gw], y_act_3[mask_gw], pid_act[mask_gw], multiclass=False)
ACT_COLLAB["baseline_regression"] = lopo_baseline_regress(
    baseline_act_collab, y_act_cont, pid_act)
print("Done.")

# %% [markdown]
# ## 8. Activity — D5 Probing

# %%
print("=== Activity: D5 TopKSAE (8192-dim Z) ===")

ACT_D5 = {}

ACT_D5["3class"] = run_grid(
    Z_d5_act[idx_act3], y_act_3[idx_act3], pid_act3, C_VALUES,
    task="classify", multiclass=True, label="D5 act 3-class")

ACT_D5["lof_vs_wt"] = run_grid(
    Z_d5_act[mask_lw], y_act_3[mask_lw], pid_act[mask_lw], C_VALUES,
    task="classify", multiclass=False, label="D5 LoF/wt")

ACT_D5["gof_vs_wt"] = run_grid(
    Z_d5_act[mask_gw], y_act_3[mask_gw], pid_act[mask_gw], C_VALUES,
    task="classify", multiclass=False, label="D5 GoF/wt")

ACT_D5["regression"] = run_grid(
    Z_d5_act, y_act_cont, pid_act, C_VALUES,
    task="regress", label="D5 act regression")

print("  Baseline …")
ACT_D5["baseline_3class"]     = lopo_baseline_classify(
    baseline_act_d5[idx_act3], y_act_3[idx_act3], pid_act3, multiclass=True)
ACT_D5["baseline_lof_wt"]     = lopo_baseline_classify(
    baseline_act_d5[mask_lw], y_act_3[mask_lw], pid_act[mask_lw], multiclass=False)
ACT_D5["baseline_gof_wt"]     = lopo_baseline_classify(
    baseline_act_d5[mask_gw], y_act_3[mask_gw], pid_act[mask_gw], multiclass=False)
ACT_D5["baseline_regression"] = lopo_baseline_regress(
    baseline_act_d5, y_act_cont, pid_act)
print("Done.")

# %% [markdown]
# ## 9. Results Summary Tables

# %%
def _fmt(d, key, fmt=".3f"):
    v = d.get(key, float("nan"))
    if isinstance(v, np.ndarray): return "—"
    return format(v, fmt) if not np.isnan(float(v)) else "—"

def print_classify_table(grid, baseline, task_names, c_values):
    print(f"\n{'Task':<18} {'C':>6} {'BalAcc':>8} {'AUC':>8} │ {'Baseline BalAcc':>16} {'Baseline AUC':>12}")
    print("─" * 80)
    for task, bl_key in task_names:
        bl = baseline.get(bl_key, {})
        for i, C in enumerate(c_values):
            r = grid.get(task, {}).get(C, {})
            pfx = f"{task:<18}" if i == 0 else " " * 18
            blfmt = (f"{_fmt(bl,'test_balanced_acc'):>16} {_fmt(bl,'test_auc'):>12}"
                     if i == 0 else " " * 29)
            print(f"{pfx} {C:>6} {_fmt(r,'test_balanced_acc'):>8} {_fmt(r,'test_auc'):>8} │ {blfmt}")

def print_regress_table(grid, baseline, task_names, c_values):
    print(f"\n{'Task':<18} {'C':>6} {'Test R²':>9} {'Test MSE':>10} │ {'Baseline R²':>12} {'Baseline MSE':>13}")
    print("─" * 85)
    for task, bl_key in task_names:
        bl = baseline.get(bl_key, {})
        for i, C in enumerate(c_values):
            r = grid.get(task, {}).get(C, {})
            pfx = f"{task:<18}" if i == 0 else " " * 18
            blfmt = (f"{_fmt(bl,'test_r2'):>12} {_fmt(bl,'test_mse'):>13}"
                     if i == 0 else " " * 26)
            print(f"{pfx} {C:>6} {_fmt(r,'test_r2'):>9} {_fmt(r,'test_mse'):>10} │ {blfmt}")

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
# For each model × dataset, pick the best C value (highest balanced accuracy on 3-class)
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
    key=lambda C: STAB_COLLAB["3class"].get(C, {}).get("test_balanced_acc", -1))
coef_stab_collab = STAB_COLLAB["3class"][best_C_stab_collab].get("coef_mean")
if coef_stab_collab is not None:
    plot_top_features(coef_stab_collab, N_TOP_FEATURES,
        f"Stability — Collab SAE ΔZ  (C={best_C_stab_collab}, 3-class)\n"
        f"red=destab  blue=stab",
        OUT_DIR / "probe_stab_collab_top_features.png")

# ─ Stability: D5 ─────────────────────────────────────────────────────────────
best_C_stab_d5 = max(C_VALUES,
    key=lambda C: STAB_D5["3class"].get(C, {}).get("test_balanced_acc", -1))
coef_stab_d5 = STAB_D5["3class"][best_C_stab_d5].get("coef_mean")
if coef_stab_d5 is not None:
    plot_top_features(coef_stab_d5, N_TOP_FEATURES,
        f"Stability — D5 TopKSAE  (C={best_C_stab_d5}, 3-class)\n"
        f"red=destab  blue=stab",
        OUT_DIR / "probe_stab_d5_top_features.png")

# ─ Activity: Collab SAE ───────────────────────────────────────────────────────
best_C_act_collab = max(C_VALUES,
    key=lambda C: ACT_COLLAB["3class"].get(C, {}).get("test_balanced_acc", -1))
coef_act_collab = ACT_COLLAB["3class"][best_C_act_collab].get("coef_mean")
if coef_act_collab is not None:
    plot_top_features(coef_act_collab, N_TOP_FEATURES,
        f"Activity — Collab SAE ΔZ  (C={best_C_act_collab}, 3-class)\n"
        f"red=GoF  blue=LoF",
        OUT_DIR / "probe_act_collab_top_features.png")

# ─ Activity: D5 ──────────────────────────────────────────────────────────────
best_C_act_d5 = max(C_VALUES,
    key=lambda C: ACT_D5["3class"].get(C, {}).get("test_balanced_acc", -1))
coef_act_d5 = ACT_D5["3class"][best_C_act_d5].get("coef_mean")
if coef_act_d5 is not None:
    plot_top_features(coef_act_d5, N_TOP_FEATURES,
        f"Activity — D5 TopKSAE  (C={best_C_act_d5}, 3-class)\n"
        f"red=GoF  blue=LoF",
        OUT_DIR / "probe_act_d5_top_features.png")

# %% [markdown]
# ## 11. Balanced Accuracy vs C Curve

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("LOPO Balanced Accuracy vs L1 Regularization (C)", fontsize=12, fontweight="bold")

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
        vals = [grid.get(task, {}).get(C, {}).get("test_balanced_acc", np.nan)
                for C in C_VALUES]
        ax.plot(C_VALUES, vals, marker="o", color=color, label=tlabel)
        bl = grid.get(bl_key, {}).get("test_balanced_acc", np.nan)
        if not np.isnan(float(bl)):
            ax.axhline(bl, linestyle="--", color=color, alpha=0.5, lw=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("C (1/λ)  — higher = less regularization", fontsize=9)
    ax.set_ylabel("Balanced Accuracy (LOPO)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(OUT_DIR / "probe_balanced_acc_vs_C.png", dpi=150, bbox_inches="tight")
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
                baseline_bal_acc  = bl.get("test_balanced_acc", np.nan),
                baseline_auc      = bl.get("test_auc", np.nan),
                baseline_r2       = bl.get("test_r2", np.nan),
            ))

df_results = pd.DataFrame(rows)
df_results.to_csv(OUT_DIR / "probe_results.csv", index=False)
print(f"Saved probe_results.csv  ({len(df_results)} rows)")
print(df_results.to_string(max_rows=20))
