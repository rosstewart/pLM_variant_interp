# %% [markdown]
# # ClinVar Sparse Bottleneck
#
# Concatenate WT + VT ProtT5 embeddings at the mutation site → 2048-dim feature.
# Train model variants and probe sparse neuron activations on megascale ∆∆G variants.

# %%
import os, warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

import sys, pickle, re
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import f_oneway
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from pathlib import Path

DEVICE        = torch.device("cuda:2")
BATCH_SIZE    = 512
D             = 256
LR            = 1e-3
LAMBDA_SPARSE = 0.01
ES_PATIENCE   = 5
MAX_EPOCHS    = 100
N_PERM        = 10000
SEED          = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"device: {DEVICE}")

# %% [markdown]
# ## 1. Load ClinVar Labels

# %%
LABEL_DIR = Path("/data/ross/ppi_lossgain/interaction_loss/home/data_interaction_loss")
H5_PATH   = "/data/ross/ppi_lossgain/interaction_loss/clinvar/prott5_subgraphs.h5"
OUT_DIR   = Path("/home/rcstewart/ppi_lossgain/sparse_bottleneck")

def load_label_set(tsv_path):
    """Returns set of (uniprot_interactor, variant_1based_str) tuples."""
    s = set()
    with open(tsv_path) as fh:
        for line in fh:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            s.add((parts[0], parts[1]))
    return s

pathogenic_set = load_label_set(LABEL_DIR / "clinvar_pathogenic_dirbind_variants.tsv")
benign_set     = load_label_set(LABEL_DIR / "clinvar_benign_dirbind_variants.tsv")
conflicts      = pathogenic_set & benign_set

print(f"Pathogenic: {len(pathogenic_set):,}")
print(f"Benign:     {len(benign_set):,}")
print(f"Conflicts:  {len(conflicts):,}  (will be skipped)")

# %% [markdown]
# ## 2. Extract Features from H5
#
# For each labeled variant:
# - `vt_emb = node_emb[mut_local_idx]`          (VT embedding at mutation site)
# - `wt_emb = vt_emb - mut_diff`                 (WT: vt − (vt − wt) = wt)
# - `feat   = concat([wt_emb, vt_emb])`          (2048-dim)
#
# H5 variant keys are 0-based; TSV labels are 1-based — add 1 before lookup.

# %%
FEAT_CACHE   = OUT_DIR / "clinvar_feats.npy"
LABEL_CACHE  = OUT_DIR / "clinvar_labels.npy"
PROTID_CACHE = OUT_DIR / "clinvar_protein_ids.npy"

_var_re = re.compile(r'^([A-Z])(\d+)([A-Z])$')

def extract_features():
    feats, labels, prot_ids = [], [], []
    with h5py.File(H5_PATH, 'r') as f:
        n_complexes = len(f)
        for ci, complex_id in enumerate(f.keys()):
            interactor_id = complex_id.split('_')[0]
            cgrp = f[complex_id]
            for var_0b in cgrp.keys():
                m = _var_re.match(var_0b)
                if m is None:
                    continue
                ref, pos_0b, alt = m.group(1), int(m.group(2)), m.group(3)
                var_1b = f"{ref}{pos_0b + 1}{alt}"
                key = (interactor_id, var_1b)
                if key in conflicts:
                    continue
                if key in pathogenic_set:
                    label = 1
                elif key in benign_set:
                    label = 0
                else:
                    continue
                vgrp          = cgrp[var_0b]
                node_emb      = vgrp["node_emb"][:]
                mut_diff      = vgrp["mut_diff"][:]
                mut_local_idx = int(vgrp.attrs["mut_local_idx"])
                vt_emb = node_emb[mut_local_idx]
                wt_emb = vt_emb - mut_diff
                feats.append(np.concatenate([wt_emb, vt_emb]).astype(np.float32))
                labels.append(label)
                prot_ids.append(interactor_id)
            if (ci + 1) % 1000 == 0:
                print(f"  {ci+1:,}/{n_complexes:,} complexes  ({len(feats):,} labeled)",
                      flush=True)
    return np.stack(feats), np.array(labels, dtype=np.int64), np.array(prot_ids)

all_cached = FEAT_CACHE.exists() and LABEL_CACHE.exists() and PROTID_CACHE.exists()
if all_cached:
    X           = np.load(FEAT_CACHE)
    y           = np.load(LABEL_CACHE)
    protein_ids = np.load(PROTID_CACHE, allow_pickle=True)
    print(f"Loaded from cache: X={X.shape}, y={y.shape}")
else:
    print("Building feature matrix from H5 (~15-30 min)...")
    X, y, protein_ids = extract_features()
    np.save(FEAT_CACHE,   X)
    np.save(LABEL_CACHE,  y)
    np.save(PROTID_CACHE, protein_ids)
    print(f"Saved to {OUT_DIR}")

X_diff = X[:, 1024:] - X[:, :1024]   # mut_diff baseline: vt − wt

print(f"X:      {X.shape}  Pathogenic: {y.sum():,}  Benign: {(y==0).sum():,}")
print(f"X_diff: {X_diff.shape}")
print(f"Unique proteins: {len(set(protein_ids)):,}")

# %% [markdown]
# ## 3. Verification

# %%
with h5py.File(H5_PATH, 'r') as f:
    cid = list(f.keys())[0]
    vid = list(f[cid].keys())[0]
    vgrp = f[cid][vid]
    node_emb = vgrp["node_emb"][:]
    mut_diff = vgrp["mut_diff"][:]
    mid = int(vgrp.attrs["mut_local_idx"])
    vt = node_emb[mid]
    wt = vt - mut_diff

print(f"Sample: {cid} / {vid}")
print(f"  mut_diff norm : {np.linalg.norm(mut_diff):.4f}")
print(f"  vt_emb norm   : {np.linalg.norm(vt):.4f}")
print(f"  wt_emb norm   : {np.linalg.norm(wt):.4f}")
print(f"  wt != vt      : {not np.allclose(wt, vt)}")
print(f"Feature stats — X: mean={X.mean():.4f} std={X.std():.4f} | "
      f"X_diff: mean={X_diff.mean():.4f} std={X_diff.std():.4f}")

# %% [markdown]
# ## 4. Group-Based Split (by protein ID)

# %%
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
idx_all = np.arange(len(y))
idx_trainval, idx_test = next(gss_test.split(idx_all, y, groups=protein_ids))

gss_val = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=SEED)
local_train, local_val = next(
    gss_val.split(idx_trainval, y[idx_trainval], groups=protein_ids[idx_trainval]))
idx_train = idx_trainval[local_train]
idx_val   = idx_trainval[local_val]

assert len(set(protein_ids[idx_train]) & set(protein_ids[idx_test])) == 0, "Train/test leak"
assert len(set(protein_ids[idx_val])   & set(protein_ids[idx_test])) == 0, "Val/test leak"

y_train = y[idx_train]; y_val = y[idx_val]; y_test = y[idx_test]
X_train  = X[idx_train];      X_val  = X[idx_val];      X_test  = X[idx_test]
Xd_train = X_diff[idx_train]; Xd_val = X_diff[idx_val]; Xd_test = X_diff[idx_test]

print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
print(f"Pos rate — train: {y_train.mean():.3f}  val: {y_val.mean():.3f}  test: {y_test.mean():.3f}")
print(f"Proteins — train: {len(set(protein_ids[idx_train])):,}  "
      f"val: {len(set(protein_ids[idx_val])):,}  "
      f"test: {len(set(protein_ids[idx_test])):,}")

# %% [markdown]
# ## 5. Datasets & DataLoaders

# %%
pos_weight = torch.tensor(
    [(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float
).to(DEVICE)
print(f"pos_weight (benign:pathogenic ratio): {pos_weight.item():.3f}")

class FeatDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):        return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

class ReconDataset(Dataset):
    """Unsupervised dataset — no labels."""
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)
    def __len__(self):        return len(self.X)
    def __getitem__(self, i): return self.X[i]

def make_loader(X, y, shuffle=True):
    return DataLoader(FeatDataset(X, y), batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader    = make_loader(X_train,  y_train, shuffle=True)
val_loader      = make_loader(X_val,    y_val,   shuffle=False)
test_loader     = make_loader(X_test,   y_test,  shuffle=False)

d0_train_loader = make_loader(Xd_train, y_train, shuffle=True)
d0_val_loader   = make_loader(Xd_val,   y_val,   shuffle=False)
d0_test_loader  = make_loader(Xd_test,  y_test,  shuffle=False)

# %% [markdown]
# ## 6. Model Definitions
#
# **Design 0** — mut_diff baseline: `SparseBNClassifier(in_dim=1024)`
#
# **Design 1** — WT+VT concat: `SparseBNClassifier(in_dim=2048)`
# ```
# Linear(in_dim→D) → ReLU → z → Linear(D→1)
# Loss: BCE + λ‖z‖₁
# ```
#
# **Design 2** — Supervised SAE:
# ```
# Encoder: Linear(2048→D) → ReLU → z
# Decoder: Linear(D→2048) → x̂
# Cls A: Linear(D→1) on z  |  Cls B: Linear(2048→1) on x̂
# Loss: MSE(x̂,x) + λ‖z‖₁ + BCE_A + BCE_B
# ```
#
# **Design 3 (D3_MegaSAE)** — Unsupervised SAE trained on megascale variants only:
# ```
# Encoder: Linear(2048→D) → ReLU → z
# Decoder: Linear(D→2048) → x̂
# Loss: MSE(x̂,x) + λ‖z‖₁   (no pathogenicity supervision)
# ```

# %%
class SparseBNClassifier(nn.Module):
    def __init__(self, in_dim=2048, d=D):
        super().__init__()
        self.encoder    = nn.Linear(in_dim, d)
        self.classifier = nn.Linear(d, 1)

    def forward(self, x):
        z     = torch.relu(self.encoder(x))
        logit = self.classifier(z)
        return logit, z


class SparseSAE(nn.Module):
    def __init__(self, in_dim=2048, d=D):
        super().__init__()
        self.encoder    = nn.Linear(in_dim, d)
        self.decoder    = nn.Linear(d, in_dim)
        self.cls_sparse = nn.Linear(d, 1)
        self.cls_recon  = nn.Linear(in_dim, 1)

    def forward(self, x):
        z       = torch.relu(self.encoder(x))
        x_hat   = self.decoder(z)
        logit_a = self.cls_sparse(z)
        logit_b = self.cls_recon(x_hat)
        return logit_a, logit_b, z, x_hat


class MegascaleSAE(nn.Module):
    """Unsupervised SAE trained purely on stability data — no pathogenicity labels."""
    def __init__(self, in_dim=2048, d=D):
        super().__init__()
        self.encoder = nn.Linear(in_dim, d)
        self.decoder = nn.Linear(d, in_dim)

    def forward(self, x):
        z     = torch.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return z, x_hat


print(f"D0 params (1024): {sum(p.numel() for p in SparseBNClassifier(1024).parameters()):,}")
print(f"D1 params (2048): {sum(p.numel() for p in SparseBNClassifier(2048).parameters()):,}")
print(f"D2 params:        {sum(p.numel() for p in SparseSAE().parameters()):,}")
print(f"D3 params:        {sum(p.numel() for p in MegascaleSAE().parameters()):,}")

# %% [markdown]
# ## 7. Training Utilities

# %%
bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

def compute_loss(model, xb, yb, design):
    if design in (0, 1):
        logit, z = model(xb)
        loss = bce_fn(logit.squeeze(1), yb) + LAMBDA_SPARSE * z.abs().mean()
        return loss, logit, logit
    else:
        logit_a, logit_b, z, x_hat = model(xb)
        recon  = F.mse_loss(x_hat, xb)
        sparse = LAMBDA_SPARSE * z.abs().mean()
        cls    = bce_fn(logit_a.squeeze(1), yb) + bce_fn(logit_b.squeeze(1), yb)
        return recon + sparse + cls, logit_a, logit_b


@torch.no_grad()
def eval_loader(model, loader, design):
    model.eval()
    total_loss = 0.0
    la_buf, lb_buf, lbl_buf = [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        loss, la, lb = compute_loss(model, xb, yb, design)
        total_loss += loss.item()
        la_buf.append(la.squeeze(1).cpu())
        lb_buf.append(lb.squeeze(1).cpu())
        lbl_buf.append(yb.cpu())
    labels   = torch.cat(lbl_buf).numpy()
    logits_a = torch.cat(la_buf).numpy()
    logits_b = torch.cat(lb_buf).numpy()
    auc_a    = roc_auc_score(labels, 1 / (1 + np.exp(-logits_a)))
    auc_b    = roc_auc_score(labels, 1 / (1 + np.exp(-logits_b)))
    return total_loss / len(loader), auc_a, auc_b


def train_model(model, design, tr_loader, vl_loader, tag=""):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_auc = 0.0
    best_state   = None
    patience_ctr = 0
    history = {"train_loss": [], "val_loss": [], "val_auc_a": [], "val_auc_b": []}

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss, _, _ = compute_loss(model, xb, yb, design)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(tr_loader)

        val_loss, auc_a, auc_b = eval_loader(model, vl_loader, design)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc_a"].append(auc_a)
        history["val_auc_b"].append(auc_b)

        if design == 2:
            print(f"{tag} Epoch {epoch+1:3d} | train={train_loss:.4f} val={val_loss:.4f} "
                  f"auc_a={auc_a:.4f} auc_b={auc_b:.4f}", flush=True)
        else:
            print(f"{tag} Epoch {epoch+1:3d} | train={train_loss:.4f} val={val_loss:.4f} "
                  f"val_auc={auc_a:.4f}", flush=True)

        if auc_a > best_val_auc:
            best_val_auc = auc_a
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= ES_PATIENCE:
                print(f"{tag} Early stop at epoch {epoch+1}  (best val_auc={best_val_auc:.4f})")
                break

    model.load_state_dict(best_state)
    return history


def train_megascale_sae(model, tr_loader, vl_loader):
    """Unsupervised reconstruction training for MegascaleSAE."""
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_loss = float('inf')
    best_state    = None
    patience_ctr  = 0
    history       = {"train_loss": [], "val_loss": []}

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        for xb in tr_loader:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            z, x_hat = model(xb)
            loss = F.mse_loss(x_hat, xb) + LAMBDA_SPARSE * z.abs().mean()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(tr_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb in vl_loader:
                xb = xb.to(DEVICE)
                z, x_hat = model(xb)
                val_loss += (F.mse_loss(x_hat, xb) + LAMBDA_SPARSE * z.abs().mean()).item()
        val_loss /= len(vl_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"[D3] Epoch {epoch+1:3d} | train={train_loss:.4f} val={val_loss:.4f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= ES_PATIENCE:
                print(f"[D3] Early stop at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return history

# %% [markdown]
# ## 8. Train Design 0 — mut_diff Baseline (1024-dim)

# %%
model_d0   = SparseBNClassifier(in_dim=1024, d=D).to(DEVICE)
history_d0 = train_model(model_d0, design=0,
                         tr_loader=d0_train_loader, vl_loader=d0_val_loader, tag="[D0]")

# %% [markdown]
# ## 9. Train Design 1 — Discriminative Sparse Bottleneck (2048-dim)

# %%
model_d1   = SparseBNClassifier(in_dim=2048, d=D).to(DEVICE)
history_d1 = train_model(model_d1, design=1,
                         tr_loader=train_loader, vl_loader=val_loader, tag="[D1]")

# %% [markdown]
# ## 10. Train Design 2 — Supervised SAE Sparse Bottleneck

# %%
model_d2   = SparseSAE(in_dim=2048, d=D).to(DEVICE)
history_d2 = train_model(model_d2, design=2,
                         tr_loader=train_loader, vl_loader=val_loader, tag="[D2]")

# %% [markdown]
# ## 11. Test Set Evaluation (D0–D2)

# %%
@torch.no_grad()
def get_test_outputs(model, loader, design):
    model.eval()
    la_buf, lb_buf, z_buf, lbl_buf = [], [], [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        if design in (0, 1):
            logit, z = model(xb)
            la_buf.append(logit.squeeze(1).cpu())
            lb_buf.append(logit.squeeze(1).cpu())
        else:
            la, lb, z, _ = model(xb)
            la_buf.append(la.squeeze(1).cpu())
            lb_buf.append(lb.squeeze(1).cpu())
        z_buf.append(z.cpu())
        lbl_buf.append(yb)
    return (torch.cat(la_buf).numpy(), torch.cat(lb_buf).numpy(),
            torch.cat(z_buf).numpy(),  torch.cat(lbl_buf).numpy())

la_d0, _,     Z_test_d0, yt = get_test_outputs(model_d0, d0_test_loader, design=0)
la_d1, _,     Z_test_d1, _  = get_test_outputs(model_d1, test_loader,    design=1)
la_d2, lb_d2, Z_test_d2, _  = get_test_outputs(model_d2, test_loader,    design=2)

def report(name, logits, labels):
    probs = 1 / (1 + np.exp(-logits))
    auc   = roc_auc_score(labels, probs)
    apr   = average_precision_score(labels, probs)
    print(f"  {name:<42s}  ROC-AUC={auc:.4f}  PR-AUC={apr:.4f}")

print("=== Test Set ===")
report("D0  mut_diff baseline (1024-dim)",        la_d0, yt)
report("D1  sparse BN (WT+VT 2048-dim)",          la_d1, yt)
report("D2A SAE — classifier on z (sparse)",      la_d2, yt)
report("D2B SAE — classifier on x̂ (recon)",      lb_d2, yt)

# %%
def sparsity_report(Z, name):
    frac = (Z > 0).mean(axis=1)
    pn   = (Z > 0).mean(axis=0)
    print(f"{name}: firing frac/sample={frac.mean():.3f}±{frac.std():.3f}  "
          f"dead={( pn==0).sum()}  always-on={(pn==1).sum()}")

sparsity_report(Z_test_d0, "D0")
sparsity_report(Z_test_d1, "D1")
sparsity_report(Z_test_d2, "D2")

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for ax, hist, title in zip(axes,
        [history_d0, history_d1, history_d2],
        ["D0 mut_diff", "D1 WT+VT concat", "D2 SAE"]):
    ax.plot(hist["train_loss"], label="train loss")
    ax.plot(hist["val_loss"],   label="val loss")
    ax2 = ax.twinx()
    ax2.plot(hist["val_auc_a"], color="green",  linestyle="--", label="val AUC (a)")
    if title.startswith("D2"):
        ax2.plot(hist["val_auc_b"], color="orange", linestyle=":",  label="val AUC (b)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax2.set_ylabel("AUC")
    ax.set_title(title)
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
plt.tight_layout()
plt.savefig(OUT_DIR / "learning_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 12. Load Megascale Variants

# %%
sys.path.insert(0, "/home/rcstewart/ppi_lossgain/mutpred_ppi_scripts")
from preprocess_stability_data import expand_emb

MEGASCALE_PKL = "/data/ross/ppi_lossgain/interaction_loss/megascale_preprocessed/preprocessed.pkl"
print("Loading megascale data...", flush=True)
with open(MEGASCALE_PKL, "rb") as f:
    ms = pickle.load(f)

ms_ddg    = ms["ddg_labels"]
ms_mutidx = ms["mutation_indices"]
ms_diffs  = np.array(ms["mutation_site_diffs"])
ms_embs   = ms["prott5_embeddings"]
N_ms      = len(ms_ddg)
print(f"Megascale: {N_ms:,} samples  ddg∈[{ms_ddg.min():.2f}, {ms_ddg.max():.2f}]")

# %%
MS_CACHE = OUT_DIR / "megascale_feats.npy"

if MS_CACHE.exists():
    X_ms = np.load(MS_CACHE)
    print(f"Loaded megascale feats: {X_ms.shape}")
else:
    print("Extracting megascale features...", flush=True)
    X_ms = np.empty((N_ms, 2048), dtype=np.float32)
    for i in range(N_ms):
        full_emb = expand_emb(ms_embs[i])
        mi       = int(ms_mutidx[i])
        vt_emb   = full_emb[mi]
        wt_emb   = vt_emb - ms_diffs[i]
        X_ms[i]  = np.concatenate([wt_emb, vt_emb])
        if (i + 1) % 10000 == 0:
            print(f"  {i+1:,}/{N_ms:,}", flush=True)
    np.save(MS_CACHE, X_ms)
    print(f"Saved to {MS_CACHE}")

# %% [markdown]
# ## 13. Train Design 3 — Unsupervised Megascale SAE
#
# Trained only on megascale stability data with reconstruction loss + L1 sparsity.
# No pathogenicity supervision — tests whether stability disruption is captured
# purely from the structural embedding distribution.

# %%
ms_idx_tr, ms_idx_val = train_test_split(
    np.arange(N_ms), test_size=0.1, random_state=SEED)

ms_train_loader = DataLoader(
    ReconDataset(X_ms[ms_idx_tr]),  batch_size=BATCH_SIZE, shuffle=True)
ms_val_loader   = DataLoader(
    ReconDataset(X_ms[ms_idx_val]), batch_size=BATCH_SIZE, shuffle=False)

model_mega   = MegascaleSAE(in_dim=2048, d=D).to(DEVICE)
history_mega = train_megascale_sae(model_mega, ms_train_loader, ms_val_loader)

# %%
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(history_mega["train_loss"], label="train loss")
ax.plot(history_mega["val_loss"],   label="val loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("Recon + L1 loss")
ax.set_title("D3 MegaSAE — unsupervised training"); ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "learning_curves_d3.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 14. Encode Megascale with All Models

# %%
X_ms_diff = X_ms[:, 1024:] - X_ms[:, :1024]   # 1024-dim input for D0

@torch.no_grad()
def encode_all(model, X, design, batch=2048):
    """design: 0/1=SparseBNClassifier, 2=SparseSAE, 3=MegascaleSAE"""
    model.eval()
    parts = []
    for i in range(0, len(X), batch):
        xb = torch.tensor(X[i:i+batch], dtype=torch.float32).to(DEVICE)
        if design in (0, 1):
            _, z = model(xb)
        elif design == 2:
            _, _, z, _ = model(xb)
        else:  # design == 3
            z, _ = model(xb)
        parts.append(z.cpu().numpy())
    return np.concatenate(parts)

ENCODERS = {
    "D0_mutdiff": (model_d0,   X_ms_diff, 0),
    "D1_concat":  (model_d1,   X_ms,      1),
    "D2_SAE":     (model_d2,   X_ms,      2),
    "D3_MegaSAE": (model_mega, X_ms,      3),
}

Z_by_model = {}
for name, (model, X_in, design) in ENCODERS.items():
    Z_by_model[name] = encode_all(model, X_in, design=design)
    print(f"{name}: {Z_by_model[name].shape}")

# %% [markdown]
# ## 15. ∆∆G Bins

# %%
BINS = {
    "highly stabilizing":   ms_ddg < -1.0,
    "mildly stabilizing":  (ms_ddg >= -1.0) & (ms_ddg < -0.5),
    "near neutral":        (np.abs(ms_ddg) < 0.5),
    "mildly destabilizing":(ms_ddg >= 0.5)  & (ms_ddg < 1.5),
    "highly destabilizing": ms_ddg >= 1.5,
}
bin_names = list(BINS.keys())
bin_masks = list(BINS.values())
n_bins    = len(bin_names)

for name, mask in BINS.items():
    print(f"  {name:<25s}: {mask.sum():,}")

# %% [markdown]
# ## 16. Per-Model Analysis: Permutation Test + Bonferroni (GPU-accelerated)
#
# Each permutation: sample n_dest rows from combined → compute mean and firing rate,
# derive wt complement from total sum (avoids materialising the wt slice).
# Runs entirely on GPU — no CPU thread overhead, no fancy-indexing cache misses.

# %%
EPS        = 1e-6
bonf_alpha = 0.05 / D
lower_pct  = 100 * bonf_alpha / 2
upper_pct  = 100 * (1 - bonf_alpha / 2)
# GPU memory per batch ≈ GPU_BATCH × n_dest × D × 4 bytes
# (n_dest≈53K, D=256) → GPU_BATCH=200 ≈ 2.7 GB; reduce if OOM
GPU_BATCH  = 200

print(f"Bonferroni alpha={bonf_alpha:.5f}  lower_pct={lower_pct:.5f}  upper_pct={upper_pct:.5f}")
print(f"N_PERM={N_PERM}  GPU_BATCH={GPU_BATCH}  device={DEVICE}")


@torch.no_grad()
def perm_test_gpu(Z_arr, wt_mask, dest_mask, n_perm, device, eps=EPS, batch=GPU_BATCH):
    Z_wt   = Z_arr[wt_mask]
    Z_dest = Z_arr[dest_mask]
    n_wt, n_dest, D_ = len(Z_wt), len(Z_dest), Z_arr.shape[1]
    N = n_wt + n_dest

    combined   = torch.tensor(np.concatenate([Z_wt, Z_dest]), dtype=torch.float32, device=device)
    total_act  = combined.sum(0)                       # (D,) — precomputed once
    total_fire = (combined > 0).float().sum(0)         # (D,)

    null_act  = np.empty((n_perm, D_), dtype=np.float32)
    null_fire = np.empty((n_perm, D_), dtype=np.float32)

    for start in range(0, n_perm, batch):
        bs = min(batch, n_perm - start)

        # topk of uniform noise = random partial permutation without full sort
        rand_vals = torch.rand(bs, N, device=device)
        dest_idx  = rand_vals.topk(n_dest, dim=1, largest=False).indices   # (bs, n_dest)
        del rand_vals

        pd = combined[dest_idx]                         # (bs, n_dest, D)
        sum_dest_act  = pd.sum(1)                       # (bs, D)
        sum_dest_fire = (pd > 0).sum(1).float()         # (bs, D) — int sum → float, avoids double float copy
        del pd, dest_idx

        mean_dest_act  = sum_dest_act  / n_dest
        mean_wt_act    = (total_act  - sum_dest_act)  / n_wt
        mean_dest_fire = sum_dest_fire / n_dest
        mean_wt_fire   = (total_fire - sum_dest_fire) / n_wt

        null_act [start:start+bs] = (mean_dest_act  / (mean_wt_act  + eps)).cpu().numpy()
        null_fire[start:start+bs] = (mean_dest_fire / (mean_wt_fire + eps)).cpu().numpy()

    del combined
    torch.cuda.empty_cache()
    return null_act, null_fire


RESULTS = {}

for model_name, Z_ms_m in Z_by_model.items():
    print(f"\n=== {model_name} ===", flush=True)
    Z_wt_m   = Z_ms_m[BINS["near neutral"]]
    Z_dest_m = Z_ms_m[BINS["highly destabilizing"]]

    obs_ratio_act  = Z_dest_m.mean(0)       / (Z_wt_m.mean(0)       + EPS)
    obs_ratio_fire = (Z_dest_m > 0).mean(0) / ((Z_wt_m > 0).mean(0) + EPS)

    null_ratio_act, null_ratio_fire = perm_test_gpu(
        Z_ms_m, BINS["near neutral"], BINS["highly destabilizing"], N_PERM, DEVICE
    )

    # two-sided p-values
    p_act  = np.minimum(
        (null_ratio_act  >= obs_ratio_act[None, :]).mean(0),
        (null_ratio_act  <= obs_ratio_act[None, :]).mean(0)
    ) * 2
    p_fire = np.minimum(
        (null_ratio_fire >= obs_ratio_fire[None, :]).mean(0),
        (null_ratio_fire <= obs_ratio_fire[None, :]).mean(0)
    ) * 2

    lo_act,  hi_act  = (np.percentile(null_ratio_act,  lower_pct, axis=0),
                        np.percentile(null_ratio_act,  upper_pct, axis=0))
    lo_fire, hi_fire = (np.percentile(null_ratio_fire, lower_pct, axis=0),
                        np.percentile(null_ratio_fire, upper_pct, axis=0))

    sig_enrich_act   = obs_ratio_act  > hi_act
    sig_deplete_act  = obs_ratio_act  < lo_act
    sig_enrich_fire  = obs_ratio_fire > hi_fire
    sig_deplete_fire = obs_ratio_fire < lo_fire

    print(f"  Activation  — enriched: {sig_enrich_act.sum():3d}  depleted: {sig_deplete_act.sum():3d}")
    print(f"  Firing rate — enriched: {sig_enrich_fire.sum():3d}  depleted: {sig_deplete_fire.sum():3d}")

    f_stats_m = np.array([
        f_oneway(*[Z_ms_m[mask, i] for mask in bin_masks]).statistic
        for i in range(D)
    ])
    f_stats_m = np.nan_to_num(f_stats_m, nan=0.0)

    RESULTS[model_name] = dict(
        Z_ms=Z_ms_m, Z_wt=Z_wt_m, Z_dest=Z_dest_m,
        obs_ratio_act=obs_ratio_act,   obs_ratio_fire=obs_ratio_fire,
        null_ratio_act=null_ratio_act, null_ratio_fire=null_ratio_fire,
        p_act=p_act, p_fire=p_fire,
        sig_enrich_act=sig_enrich_act,   sig_deplete_act=sig_deplete_act,
        sig_enrich_fire=sig_enrich_fire, sig_deplete_fire=sig_deplete_fire,
        f_stats=f_stats_m,
        mean_act=np.stack([Z_ms_m[mask].mean(0)      for mask in bin_masks], axis=1),
        fire_rate=np.stack([(Z_ms_m[mask]>0).mean(0) for mask in bin_masks], axis=1),
    )

# %% [markdown]
# ## 17. Plots — Per Model

# %%
import os as _os
for model_name, R in RESULTS.items():
    top_idx = np.argsort(R["f_stats"])[::-1][:30]
    top20   = np.argsort(np.abs(np.log2(R["obs_ratio_act"] + EPS)))[::-1][:20]

    fig, axes = plt.subplots(1, 4, figsize=(24, 8))
    fig.suptitle(model_name, fontsize=13, fontweight="bold")

    # firing rate heatmap
    ax = axes[0]
    im = ax.imshow(R["fire_rate"][top_idx], aspect="auto", cmap="magma")
    ax.set_xticks(range(n_bins)); ax.set_xticklabels(bin_names, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(top_idx))); ax.set_yticklabels([f"N{n}" for n in top_idx], fontsize=6)
    ax.set_title("Firing rate (top 30 F-stat)"); plt.colorbar(im, ax=ax, fraction=0.03)

    # mean activation heatmap
    ax = axes[1]
    im = ax.imshow(R["mean_act"][top_idx], aspect="auto", cmap="viridis")
    ax.set_xticks(range(n_bins)); ax.set_xticklabels(bin_names, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(top_idx))); ax.set_yticklabels([f"N{n}" for n in top_idx], fontsize=6)
    ax.set_title("Mean activation (top 30 F-stat)"); plt.colorbar(im, ax=ax, fraction=0.03)

    # binary bar chart
    ax = axes[2]
    log2_ratio = np.log2(R["obs_ratio_fire"][top20] + EPS)
    colors = ["tab:red" if R["sig_enrich_fire"][n] else
              "tab:blue" if R["sig_deplete_fire"][n] else "lightgray"
              for n in top20]
    ax.barh(range(len(top20)), log2_ratio[::-1], color=colors[::-1])
    ax.axvline(0, color='k', lw=0.8)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels([f"N{n}" for n in top20[::-1]], fontsize=7)
    ax.set_xlabel("log₂(fire_rate destab / fire_rate neutral)")
    ax.set_title("Firing ratio: destab vs neutral\n(red=enriched  blue=depleted  Bonferroni)")

    # scatter: fire_dest vs log2 ratio, colored by enrichment/depletion
    ax = axes[3]
    fire_dest_m    = (R["Z_dest"] > 0).mean(0)
    log2_ratio_all = np.log2(R["obs_ratio_fire"] + EPS)
    for cat, mask, color, size, zo in [
            ("enriched",  R["sig_enrich_fire"],                            "tab:red",  40, 4),
            ("depleted",  R["sig_deplete_fire"],                           "tab:blue", 40, 4),
            ("n.s.",     ~(R["sig_enrich_fire"] | R["sig_deplete_fire"]), "lightgray", 10, 2),
    ]:
        ax.scatter(fire_dest_m[mask], log2_ratio_all[mask],
                   c=color, s=size, alpha=0.8, label=cat, zorder=zo)
    ax.axhline(0, color='k', lw=0.8, linestyle='--')
    ax.set_xlabel("Firing rate (highly destabilizing)", fontsize=9)
    ax.set_ylabel("log₂(fire_rate destab / fire_rate neutral)", fontsize=9)
    ax.set_title("Enrichment scatter (Bonferroni)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"ddg_analysis_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 18. Save Models

# %%
torch.save(model_d0.state_dict(),   OUT_DIR / "model_d0.pt")
torch.save(model_d1.state_dict(),   OUT_DIR / "model_d1.pt")
torch.save(model_d2.state_dict(),   OUT_DIR / "model_d2.pt")
torch.save(model_mega.state_dict(), OUT_DIR / "model_d3_megasae.pt")
print("Models saved.")
