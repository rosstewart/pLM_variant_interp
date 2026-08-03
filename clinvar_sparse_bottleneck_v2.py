# %% [markdown]
# # ClinVar Sparse Bottleneck — v2
#
# Changes vs v1:
# - Literature-standard SAE training: pre-encoder bias (b_pre/b_dec) + decoder column normalisation
# - D3 UnsupervisedSAE now reconstructs ClinVar variants (not megascale) to eliminate
#   small-protein-count / easy-reconstruction bias
# - Stability perm test runs for both highly destabilising AND highly stabilising vs neutral
# - Pathogenicity sanity check: do stability-enriched neurons also fire more in pathogenic ClinVar?

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
from scipy.stats import f_oneway, rankdata
from scipy.special import ndtr
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

# D5 TopK SAE hyperparameters (matching collaborator's sparsity ratio k/d ≈ 1.56%)
EF_TOPK     = 4           # expansion factor: dict_size = EF_TOPK * in_dim = 8192
K_TOPK      = 128         # exactly k features fire per sample
AUXK_ALPHA  = 1 / 32     # dead-neuron auxiliary loss weight (TopKTrainer default)
DEAD_THRESH = 10_000_000  # samples since last fire before neuron counted as dead
K_TOPK_DIFF = K_TOPK // 2  # 64: maintains ~1.56% sparsity for 4096-feature (1024-dim input) dict

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

X_diff = X[:, 1024:] - X[:, :1024]
print(f"X: {X.shape}  Pathogenic: {y.sum():,}  Benign: {(y==0).sum():,}")
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
print(f"  wt != vt      : {not np.allclose(wt, vt)}")
print(f"Feature stats — X: mean={X.mean():.4f} std={X.std():.4f}")

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
print(f"Proteins — train: {len(set(protein_ids[idx_train])):,}  "
      f"val: {len(set(protein_ids[idx_val])):,}  "
      f"test: {len(set(protein_ids[idx_test])):,}")

# %% [markdown]
# ## 5. Datasets & DataLoaders

# %%
pos_weight = torch.tensor(
    [(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float
).to(DEVICE)
print(f"pos_weight: {pos_weight.item():.3f}")

class FeatDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):        return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

class ReconDataset(Dataset):
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
# Literature-standard SAE training additions (Bricken et al. 2023, Templeton et al. 2024):
# - **Pre-encoder bias (b_pre / b_dec)**: initialized to data mean, subtracted before encoding
#   and added back after decoding. Centers input so encoder doesn't waste capacity on the mean.
# - **Decoder column normalisation**: after each gradient step, decoder weight columns are
#   projected to unit norm, preventing amplitude trade-off between encoder/decoder that would
#   make L1 scale meaningless across neurons.

# %%
class SparseBNClassifier(nn.Module):
    def __init__(self, in_dim=2048, d=D):
        super().__init__()
        self.encoder    = nn.Linear(in_dim, d)
        self.classifier = nn.Linear(d, 1)
        self.register_buffer('b_pre', torch.zeros(in_dim))

    def forward(self, x):
        z     = torch.relu(self.encoder(x - self.b_pre))
        logit = self.classifier(z)
        return logit, z


class SparseSAE(nn.Module):
    def __init__(self, in_dim=2048, d=D):
        super().__init__()
        self.encoder    = nn.Linear(in_dim, d)
        self.decoder    = nn.Linear(d, in_dim)
        self.cls_sparse = nn.Linear(d, 1)
        self.register_buffer('b_dec', torch.zeros(in_dim))

    def forward(self, x):
        x_c   = x - self.b_dec
        z     = torch.relu(self.encoder(x_c))
        x_hat = self.decoder(z) + self.b_dec
        logit = self.cls_sparse(z)
        return logit, z, x_hat


class UnsupervisedSAE(nn.Module):
    """Pure reconstruction SAE — no pathogenicity supervision.
    Trained on ClinVar WT+VT features to learn a general protein-variant manifold."""
    def __init__(self, in_dim=2048, d=D):
        super().__init__()
        self.encoder = nn.Linear(in_dim, d)
        self.decoder = nn.Linear(d, in_dim)
        self.register_buffer('b_dec', torch.zeros(in_dim))

    def forward(self, x):
        x_c   = x - self.b_dec
        z     = torch.relu(self.encoder(x_c))
        x_hat = self.decoder(z) + self.b_dec
        return z, x_hat


def _norm_decoder(model):
    """Project decoder columns to unit norm (called after every opt.step for SAEs)."""
    with torch.no_grad():
        model.decoder.weight.data = F.normalize(model.decoder.weight.data, dim=0)


class TopKSAE(nn.Module):
    """Over-complete SAE with hard TopK sparsity — exact port of AutoEncoderTopK.

    Differences from UnsupervisedSAE (D3/D4):
    - dict_size = EF_TOPK * in_dim  (expansion, not bottleneck)
    - Exactly k features fire per sample; no L1 penalty in the loss
    - decoder has no bias (b_dec handles centering on both sides)
    """
    def __init__(self, in_dim=2048, expansion_factor=EF_TOPK, k=K_TOPK):
        super().__init__()
        d = expansion_factor * in_dim
        self.k = k
        self.d = d
        self.encoder = nn.Linear(in_dim, d)
        self.decoder = nn.Linear(d, in_dim, bias=False)
        self.register_buffer('b_dec', torch.zeros(in_dim))

    def encode(self, x):
        pre_act = torch.relu(self.encoder(x - self.b_dec))
        topk_vals, topk_idx = pre_act.topk(self.k, dim=-1, sorted=False)
        z = torch.zeros_like(pre_act).scatter_(-1, topk_idx, topk_vals)
        return z, topk_vals, topk_idx, pre_act

    def decode(self, z):
        return self.decoder(z) + self.b_dec

    def forward(self, x):
        z, _, _, _ = self.encode(x)
        return z, self.decode(z)


def _geometric_median(X, n_iter=50):
    """Weiszfeld's algorithm for geometric median of rows of X (tensor)."""
    m = X.mean(0)
    for _ in range(n_iter):
        dists = torch.norm(X - m, dim=1, keepdim=True).clamp(min=1e-8)
        w = 1.0 / dists
        m = (w * X).sum(0) / w.sum()
    return m


def _remove_parallel_grad(model):
    """Project out decoder gradient components parallel to each column direction.
    Prevents the optimizer from undoing the unit-norm constraint on decoder columns."""
    W = model.decoder.weight      # [in_dim, d]
    g = model.decoder.weight.grad
    if g is None:
        return
    # columns are unit-norm so ||W_j||=1; parallel component = (g·W) * W
    parallel = (g * W).sum(0, keepdim=True) * W
    model.decoder.weight.grad = g - parallel


def _auxk_loss(model, x, x_hat, pre_act, tokens_since_fired):
    """Dead-neuron auxiliary loss (TopKTrainer.get_auxiliary_loss).

    Uses top-k dead features to reconstruct the reconstruction residual,
    normalized by residual variance. Returns 0 when no dead neurons exist."""
    dead = tokens_since_fired >= DEAD_THRESH
    if not dead.any():
        return torch.tensor(0.0, device=x.device)
    k_aux = min(model.encoder.in_features // 2, int(dead.sum()))
    masked = torch.where(dead, pre_act, torch.full_like(pre_act, float('-inf')))
    aux_vals, aux_idx = masked.topk(k_aux, dim=-1, sorted=False)
    aux_z = torch.zeros_like(pre_act).scatter_(-1, aux_idx, aux_vals)
    residual = (x - x_hat).detach()
    x_aux  = model.decoder(aux_z)  # raw decoder (no b_dec — target is already a residual)
    l2_aux = (residual - x_aux).pow(2).sum(-1).mean()
    denom  = (residual - residual.mean(0)).pow(2).sum(-1).mean()
    return (l2_aux / denom).nan_to_num(0.0)


print(f"D0 params (1024): {sum(p.numel() for p in SparseBNClassifier(1024).parameters()):,}")
print(f"D1 params (2048): {sum(p.numel() for p in SparseBNClassifier(2048).parameters()):,}")
print(f"D2 params:        {sum(p.numel() for p in SparseSAE().parameters()):,}")
print(f"D3 params:        {sum(p.numel() for p in UnsupervisedSAE().parameters()):,}")
print(f"D5 params (TopK): {sum(p.numel() for p in TopKSAE().parameters()):,}"
      f"  dict_size={EF_TOPK * 2048}")


class SupervisedTopKSAE(nn.Module):
    """TopK SAE with pathogenicity classifier on z.

    Combines hard TopK sparsity (monosemanticity) with ClinVar supervision so the
    active features are pulled toward disease-relevant molecular effects.
    Same architecture as TopKSAE but adds cls_sparse head; same signature as SparseSAE
    so design=2 branch of encode_all / get_test_outputs works without change.
    """
    def __init__(self, in_dim=2048, expansion_factor=EF_TOPK, k=K_TOPK):
        super().__init__()
        d = expansion_factor * in_dim
        self.k = k
        self.d = d
        self.encoder    = nn.Linear(in_dim, d)
        self.decoder    = nn.Linear(d, in_dim, bias=False)
        self.classifier = nn.Linear(d, 1)
        self.register_buffer('b_dec', torch.zeros(in_dim))

    def encode(self, x):
        pre_act = torch.relu(self.encoder(x - self.b_dec))
        topk_vals, topk_idx = pre_act.topk(self.k, dim=-1, sorted=False)
        z = torch.zeros_like(pre_act).scatter_(-1, topk_idx, topk_vals)
        return z, topk_vals, topk_idx, pre_act

    def decode(self, z):
        return self.decoder(z) + self.b_dec

    def forward(self, x):
        z, _, _, _ = self.encode(x)
        return self.classifier(z), z, self.decode(z)


print(f"D6 params (SupTopK): {sum(p.numel() for p in SupervisedTopKSAE().parameters()):,}"
      f"  dict_size={EF_TOPK * 2048}")

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
        logit, z, x_hat = model(xb)
        recon  = F.mse_loss(x_hat, xb)
        sparse = LAMBDA_SPARSE * z.abs().mean()
        cls    = bce_fn(logit.squeeze(1), yb)
        return recon + sparse + cls, logit, logit


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
            if design == 2:
                _norm_decoder(model)
            train_loss += loss.item()
        train_loss /= len(tr_loader)

        val_loss, auc_a, auc_b = eval_loader(model, vl_loader, design)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc_a"].append(auc_a)
        history["val_auc_b"].append(auc_b)

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


def train_unsupervised_sae(model, tr_loader, vl_loader, tag="[D3]"):
    """Unsupervised reconstruction training with decoder column normalisation."""
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
            _norm_decoder(model)
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
        print(f"{tag} Epoch {epoch+1:3d} | train={train_loss:.4f} val={val_loss:.4f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= ES_PATIENCE:
                print(f"{tag} Early stop at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return history


def train_topk_sae(model, tr_loader, vl_loader, tag="[D5]"):
    """TopK SAE training matching AutoEncoderTopK / TopKTrainer workflow.

    Sparsity is enforced by hard TopK masking (no L1). Extras vs train_unsupervised_sae:
    - b_dec initialized to geometric median of first batch
    - Dead-neuron auxiliary loss (auxk) after DEAD_THRESH samples of silence
    - Gradient parallel to decoder columns projected out before opt.step
    - Decoder columns renormalized after every step
    """
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    tokens_since_fired = torch.zeros(model.d, device=DEVICE)
    best_val_loss = float('inf')
    best_state    = None
    patience_ctr  = 0
    history       = {"train_loss": [], "val_loss": [], "dead_neurons": []}
    initialized   = False

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        for xb in tr_loader:
            xb = xb.to(DEVICE)

            if not initialized:
                with torch.no_grad():
                    model.b_dec.data = _geometric_median(xb)
                initialized = True

            z, topk_vals, topk_idx, pre_act = model.encode(xb)
            x_hat = model.decode(z)
            e     = xb - x_hat
            l2    = e.pow(2).sum(-1).mean()
            auxk  = _auxk_loss(model, xb, x_hat, pre_act, tokens_since_fired)
            loss  = l2 + AUXK_ALPHA * auxk

            fired = torch.zeros(model.d, dtype=torch.bool, device=DEVICE)
            fired[topk_idx.flatten()] = True
            tokens_since_fired += xb.size(0)
            tokens_since_fired[fired] = 0

            opt.zero_grad()
            loss.backward()
            _remove_parallel_grad(model)
            opt.step()
            _norm_decoder(model)
            train_loss += loss.item()

        train_loss /= len(tr_loader)
        dead_count = int((tokens_since_fired >= DEAD_THRESH).sum())

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb in vl_loader:
                xb = xb.to(DEVICE)
                z, x_hat = model(xb)
                val_loss += (xb - x_hat).pow(2).sum(-1).mean().item()
        val_loss /= len(vl_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["dead_neurons"].append(dead_count)
        print(f"{tag} Epoch {epoch+1:3d} | train={train_loss:.4f} val={val_loss:.4f} "
              f"dead={dead_count}/{model.d}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= ES_PATIENCE:
                print(f"{tag} Early stop at epoch {epoch+1}  (best val_loss={best_val_loss:.4f})")
                break

    model.load_state_dict(best_state)
    return history


def train_supervised_topk_sae(model, tr_loader, vl_loader, tag="[D6]"):
    """Supervised TopK SAE: L2 reconstruction + BCE pathogenicity + auxk dead-neuron loss.

    Early stopping on val AUC (like train_model for D0–D2).
    All TopK mechanics (gradient projection, decoder normalization, dead-neuron tracking,
    geometric-median b_dec init) are identical to train_topk_sae.
    """
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    tokens_since_fired = torch.zeros(model.d, device=DEVICE)
    best_val_auc = 0.0
    best_state   = None
    patience_ctr = 0
    history      = {"train_loss": [], "val_loss": [], "val_auc": [], "dead_neurons": []}
    initialized  = False

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            if not initialized:
                with torch.no_grad():
                    model.b_dec.data = _geometric_median(xb)
                initialized = True

            z, topk_vals, topk_idx, pre_act = model.encode(xb)
            x_hat = model.decode(z)
            logit = model.classifier(z)
            l2   = (xb - x_hat).pow(2).sum(-1).mean()
            cls  = bce_fn(logit.squeeze(1), yb)
            auxk = _auxk_loss(model, xb, x_hat, pre_act, tokens_since_fired)
            loss = l2 + cls + AUXK_ALPHA * auxk

            fired = torch.zeros(model.d, dtype=torch.bool, device=DEVICE)
            fired[topk_idx.flatten()] = True
            tokens_since_fired += xb.size(0)
            tokens_since_fired[fired] = 0

            opt.zero_grad()
            loss.backward()
            _remove_parallel_grad(model)
            opt.step()
            _norm_decoder(model)
            train_loss += loss.item()

        train_loss /= len(tr_loader)
        dead_count  = int((tokens_since_fired >= DEAD_THRESH).sum())

        model.eval()
        val_loss = 0.0
        logit_buf, lbl_buf = [], []
        with torch.no_grad():
            for xb, yb in vl_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logit, z, x_hat = model(xb)
                val_loss += ((xb - x_hat).pow(2).sum(-1).mean() +
                             bce_fn(logit.squeeze(1), yb)).item()
                logit_buf.append(logit.squeeze(1).cpu())
                lbl_buf.append(yb.cpu())
        val_loss /= len(vl_loader)
        val_auc = roc_auc_score(torch.cat(lbl_buf).numpy(),
                                torch.sigmoid(torch.cat(logit_buf)).numpy())

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["dead_neurons"].append(dead_count)
        print(f"{tag} Epoch {epoch+1:3d} | train={train_loss:.4f} val={val_loss:.4f} "
              f"val_auc={val_auc:.4f} dead={dead_count}/{model.d}", flush=True)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= ES_PATIENCE:
                print(f"{tag} Early stop  (best val_auc={best_val_auc:.4f})")
                break

    model.load_state_dict(best_state)
    return history

# %% [markdown]
# ## 8. Train Design 0 — mut_diff Baseline (1024-dim)

# %%
Xd_mean = torch.tensor(Xd_train.mean(0), dtype=torch.float32)
model_d0 = SparseBNClassifier(in_dim=1024, d=D).to(DEVICE)
model_d0.b_pre.copy_(Xd_mean)
history_d0 = train_model(model_d0, design=0,
                         tr_loader=d0_train_loader, vl_loader=d0_val_loader, tag="[D0]")

# %% [markdown]
# ## 9. Train Design 1 — Discriminative Sparse Bottleneck (2048-dim)

# %%
X_mean = torch.tensor(X_train.mean(0), dtype=torch.float32)
model_d1 = SparseBNClassifier(in_dim=2048, d=D).to(DEVICE)
model_d1.b_pre.copy_(X_mean)
history_d1 = train_model(model_d1, design=1,
                         tr_loader=train_loader, vl_loader=val_loader, tag="[D1]")

# %% [markdown]
# ## 10. Train Design 2 — Supervised SAE Sparse Bottleneck

# %%
model_d2 = SparseSAE(in_dim=2048, d=D).to(DEVICE)
model_d2.b_dec.copy_(X_mean)
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
            la, z, _ = model(xb)
            la_buf.append(la.squeeze(1).cpu())
            lb_buf.append(la.squeeze(1).cpu())
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

print("=== Test Set (D0–D2) ===")
report("D0  mut_diff baseline (1024-dim)",        la_d0, yt)
report("D1  sparse BN (WT+VT 2048-dim)",          la_d1, yt)
report("D2  SAE — classifier on z",                la_d2, yt)

# %%
def sparsity_report(Z, name):
    frac = (Z > 0).mean(axis=1)
    pn   = (Z > 0).mean(axis=0)
    print(f"{name}: firing frac/sample={frac.mean():.3f}±{frac.std():.3f}  "
          f"dead={(pn==0).sum()}  always-on={(pn==1).sum()}")

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
    ax2.plot(hist["val_auc_a"], color="green", linestyle="--", label="val AUC")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax2.set_ylabel("AUC")
    ax.set_title(title)
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
plt.tight_layout()
plt.savefig(OUT_DIR / "v2_learning_curves.png", dpi=150, bbox_inches="tight")
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
# ## 13. Train Design 3 — Unsupervised SAE on ClinVar
#
# Trained on ClinVar WT+VT features with reconstruction loss + L1 sparsity only —
# no pathogenicity labels. Using ClinVar (vs megascale in v1) gives broader protein
# coverage and harder reconstruction, preventing trivial solutions from narrow protein
# distributions. Then applied to megascale for stability analysis.

# %%
unsup_train_loader = DataLoader(ReconDataset(X_train), batch_size=BATCH_SIZE, shuffle=True)
unsup_val_loader   = DataLoader(ReconDataset(X_val),   batch_size=BATCH_SIZE, shuffle=False)

model_unsup = UnsupervisedSAE(in_dim=2048, d=D).to(DEVICE)
model_unsup.b_dec.copy_(X_mean)
history_unsup = train_unsupervised_sae(model_unsup, unsup_train_loader, unsup_val_loader)

# %% [markdown]
# ## 13b. Train Design 4 — Unsupervised SAE on ClinVar VT-only (1024-dim)
#
# Ablation of D3: same architecture and training procedure but input is the mutant
# residue embedding only (1024-dim), dropping WT context entirely.
# Comparing D3 vs D4 shows whether WT–MT contrast is needed to capture stability signal.

# %%
X_vt_train = X_train[:, 1024:]   # VT slice
X_vt_val   = X_val[:,   1024:]

vt_mean = torch.tensor(X_vt_train.mean(0), dtype=torch.float32)
unsup_vt_train_loader = DataLoader(ReconDataset(X_vt_train), batch_size=BATCH_SIZE, shuffle=True)
unsup_vt_val_loader   = DataLoader(ReconDataset(X_vt_val),   batch_size=BATCH_SIZE, shuffle=False)

model_unsup_vt = UnsupervisedSAE(in_dim=1024, d=D).to(DEVICE)
model_unsup_vt.b_dec.copy_(vt_mean)
history_unsup_vt = train_unsupervised_sae(model_unsup_vt, unsup_vt_train_loader,
                                           unsup_vt_val_loader, tag="[D4]")

# %% [markdown]
# ## 13c. Train Design 5 — TopK SAE on ClinVar (over-complete, hard sparsity)
#
# Exact port of the collaborator's AutoEncoderTopK / TopKTrainer architecture adapted to
# variant-level WT+VT 2048-dim embeddings. Key differences from D3 (UnsupervisedSAE):
# - Over-complete dictionary: dict_size = EF_TOPK × in_dim = 8192 (vs bottleneck D=256)
# - Hard TopK sparsity: exactly K_TOPK=128 features fire per sample (no L1 penalty)
# - Dead-neuron auxiliary loss revives features silent for >= DEAD_THRESH samples
# - b_dec initialized to geometric median of first batch (Weiszfeld algorithm)
# - Gradient components parallel to decoder columns are projected out before opt.step

# %%
unsup_topk_train_loader = DataLoader(ReconDataset(X_train), batch_size=BATCH_SIZE, shuffle=True)
unsup_topk_val_loader   = DataLoader(ReconDataset(X_val),   batch_size=BATCH_SIZE, shuffle=False)

model_d5 = TopKSAE(in_dim=2048).to(DEVICE)
history_d5 = train_topk_sae(model_d5, unsup_topk_train_loader, unsup_topk_val_loader)

assert model_d5.d == EF_TOPK * 2048, "D5 dict_size mismatch"
sparsity_report(model_d5.encode(
    torch.tensor(X_test[:512], dtype=torch.float32).to(DEVICE))[0].detach().cpu().numpy(), "D5")

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for ax, hist, title in zip(axes,
        [history_unsup, history_unsup_vt, history_d5],
        ["D3 UnsupSAE — ClinVar WT+VT (2048-dim, L1)",
         "D4 UnsupSAE — ClinVar VT-only (1024-dim, L1)",
         f"D5 TopKSAE — ClinVar WT+VT (dict={EF_TOPK*2048}, k={K_TOPK})"]):
    ax.plot(hist["train_loss"], label="train loss")
    ax.plot(hist["val_loss"],   label="val loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Recon loss")
    ax.set_title(title); ax.legend()
    if "dead_neurons" in hist:
        ax2 = ax.twinx()
        ax2.plot(hist["dead_neurons"], color="orange", linestyle=":", alpha=0.7, label="dead neurons")
        ax2.set_ylabel("dead neurons"); ax2.legend(loc="upper right")
plt.tight_layout()
plt.savefig(OUT_DIR / "v2_learning_curves_d3d4d5.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 13d. Train Design 6 — Supervised TopK SAE
#
# Combines the hard TopK sparsity of D5 (monosemanticity, over-complete dictionary) with
# the ClinVar pathogenicity supervision of D2. Classification gradient pushes active
# features toward disease-relevant molecular effects; L2 reconstruction keeps features
# grounded in the embedding geometry; auxk loss prevents dead neurons.
#
# Key differences vs D2 (supervised L1 bottleneck):
# - Over-complete dictionary (8192 features vs 256) — can represent superposed concepts
# - Hard TopK: exactly K_TOPK=128 features fire per sample, not soft L1
# - No L1 penalty: sparsity is enforced architecturally, not via regularization weight

# %%
model_d6 = SupervisedTopKSAE(in_dim=2048).to(DEVICE)
history_d6 = train_supervised_topk_sae(model_d6, train_loader, val_loader)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
ax.plot(history_d6["train_loss"], label="train loss")
ax.plot(history_d6["val_loss"],   label="val loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("Recon + BCE loss")
ax.set_title(f"D6 SupTopKSAE (dict={EF_TOPK*2048}, k={K_TOPK})"); ax.legend()
ax2 = ax.twinx()
ax2.plot(history_d6["val_auc"],  color="green",  linestyle="--", label="val AUC")
ax2.set_ylabel("AUC"); ax2.legend(loc="upper right")
ax = axes[1]
ax.plot(history_d6["dead_neurons"], color="orange")
ax.set_xlabel("Epoch"); ax.set_ylabel("Dead neurons")
ax.set_title("D6 dead neuron count over training")
plt.tight_layout()
plt.savefig(OUT_DIR / "v2_learning_curves_d6.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
la_d6, _, Z_test_d6, _ = get_test_outputs(model_d6, test_loader, design=2)
print("=== Test Set (D6) ===")
report("D6  Supervised TopK SAE", la_d6, yt)
sparsity_report(Z_test_d6, "D6")

# %% [markdown]
# ## 13e. Train Design 7 — TopK SAE on mut_diff (unsupervised, 1024-dim)
#
# Same architecture and training as D5, but input is VT−WT mutation diff (1024-dim)
# rather than WT+VT concat (2048-dim). Dict size = 4 × 1024 = 4096; k=K_TOPK_DIFF=64
# preserves the same ~1.56% sparsity ratio as D5.

# %%
X_diff_train = X_train[:, 1024:] - X_train[:, :1024]
X_diff_val   = X_val[:, 1024:]   - X_val[:, :1024]
X_diff_test  = X_test[:, 1024:]  - X_test[:, :1024]

diff_topk_train_loader = DataLoader(ReconDataset(X_diff_train), batch_size=BATCH_SIZE, shuffle=True)
diff_topk_val_loader   = DataLoader(ReconDataset(X_diff_val),   batch_size=BATCH_SIZE, shuffle=False)

model_d7 = TopKSAE(in_dim=1024, k=K_TOPK_DIFF).to(DEVICE)
history_d7 = train_topk_sae(model_d7, diff_topk_train_loader, diff_topk_val_loader, tag="[D7]")

assert model_d7.d == EF_TOPK * 1024, "D7 dict_size mismatch"
sparsity_report(model_d7.encode(
    torch.tensor(X_diff_test[:512], dtype=torch.float32).to(DEVICE))[0].detach().cpu().numpy(), "D7")

# %% [markdown]
# ## 13f. Train Design 8 — Supervised TopK SAE on mut_diff (1024-dim)
#
# Same as D6 but with mut_diff input. Pathogenicity supervision on 4096-feature
# TopK SAE (k=K_TOPK_DIFF=64) of the mutation-only signal. Analogous to D0
# (discriminative mut_diff baseline) but with TopK monosemanticity.

# %%
diff_sup_train_loader = DataLoader(
    FeatDataset(X_diff_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
diff_sup_val_loader   = DataLoader(
    FeatDataset(X_diff_val,   y_val),   batch_size=BATCH_SIZE, shuffle=False)
diff_sup_test_loader  = DataLoader(
    FeatDataset(X_diff_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False)

model_d8 = SupervisedTopKSAE(in_dim=1024, k=K_TOPK_DIFF).to(DEVICE)
history_d8 = train_supervised_topk_sae(
    model_d8, diff_sup_train_loader, diff_sup_val_loader, tag="[D8]")

# %%
la_d8, _, Z_test_d8, _ = get_test_outputs(model_d8, diff_sup_test_loader, design=2)
print("=== Test Set (D8) ===")
report("D8  Supervised TopK SAE mut_diff", la_d8, yt)
sparsity_report(Z_test_d8, "D8")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, hist, title in zip(axes,
        [history_d7, history_d8],
        [f"D7 TopKSAE mut_diff (dict={EF_TOPK*1024}, k={K_TOPK_DIFF})",
         f"D8 SupTopKSAE mut_diff (dict={EF_TOPK*1024}, k={K_TOPK_DIFF})"]):
    ax.plot(hist["train_loss"], label="train loss")
    ax.plot(hist["val_loss"],   label="val loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(title); ax.legend()
    if "dead_neurons" in hist:
        ax2 = ax.twinx()
        ax2.plot(hist["dead_neurons"], color="orange", linestyle=":", alpha=0.7)
        ax2.set_ylabel("dead neurons")
plt.tight_layout()
plt.savefig(OUT_DIR / "v2_learning_curves_d7d8.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 14. Encode Megascale with All Models

# %%
X_ms_diff = X_ms[:, 1024:] - X_ms[:, :1024]

@torch.no_grad()
def encode_all(model, X, design, batch=2048):
    """design: 0/1=SparseBNClassifier, 2=SparseSAE, 3=UnsupervisedSAE"""
    model.eval()
    parts = []
    for i in range(0, len(X), batch):
        xb = torch.tensor(X[i:i+batch], dtype=torch.float32).to(DEVICE)
        if design in (0, 1):
            _, z = model(xb)
        elif design == 2:
            _, z, _ = model(xb)
        else:
            z, _ = model(xb)
        parts.append(z.cpu().numpy())
    return np.concatenate(parts)

ENCODERS = {
    "D0_mutdiff":        (model_d0,       X_ms_diff,        0),
    "D1_concat":         (model_d1,       X_ms,             1),
    "D2_SAE":            (model_d2,       X_ms,             2),
    "D3_UnsupSAE":       (model_unsup,    X_ms,             3),
    "D4_UnsupSAE_VT":    (model_unsup_vt, X_ms[:, 1024:],   3),
    "D5_TopKSAE":        (model_d5,       X_ms,             3),
    "D6_SupTopKSAE":     (model_d6,       X_ms,             2),
    "D7_TopKSAE_diff":   (model_d7,       X_ms_diff,        3),  # design=3: z, _ = model(xb)
    "D8_SupTopKSAE_diff":(model_d8,       X_ms_diff,        2),  # design=2: _, z, _ = model(xb)
}

# ClinVar inputs per model (keyed by name — D4 uses VT slice, D0/D7/D8 use mut_diff)
CV_INPUTS = {
    "D0_mutdiff":        X_diff,
    "D1_concat":         X,
    "D2_SAE":            X,
    "D3_UnsupSAE":       X,
    "D4_UnsupSAE_VT":    X[:, 1024:],
    "D5_TopKSAE":        X,
    "D6_SupTopKSAE":     X,
    "D7_TopKSAE_diff":   X_diff,
    "D8_SupTopKSAE_diff":X_diff,
}

Z_by_model = {}
for name, (model, X_in, design) in ENCODERS.items():
    Z_by_model[name] = encode_all(model, X_in, design=design)
    print(f"{name}: {Z_by_model[name].shape}")

# %% [markdown]
# ## 15. Encode ClinVar for Pathogenicity Sanity Check

# %%
# Encode all ClinVar variants — used later to ask whether stability-enriched neurons
# are also enriched in pathogenic vs benign variants.
Z_clinvar = {}
for name, (model, _, design) in ENCODERS.items():
    Z_clinvar[name] = encode_all(model, CV_INPUTS[name], design=design)
    print(f"{name} ClinVar: {Z_clinvar[name].shape}")

# Per-neuron firing rates split by pathogenicity label
cv_stats = {}
for name, Z_cv in Z_clinvar.items():
    fp = (Z_cv[y == 1] > 0).mean(0)   # firing rate — pathogenic
    fb = (Z_cv[y == 0] > 0).mean(0)   # firing rate — benign
    ap = Z_cv[y == 1].mean(0)          # mean activation — pathogenic
    ab = Z_cv[y == 0].mean(0)          # mean activation — benign
    cv_stats[name] = dict(
        fire_path=fp, fire_beni=fb,
        act_path=ap,  act_beni=ab,
        fire_ratio=fp / (fb + 1e-6),
        act_ratio =ap / (ab + 1e-6),
    )
    print(f"{name}  path_fire={fp.mean():.3f}  beni_fire={fb.mean():.3f}")

# %% [markdown]
# ## 16. ∆∆G Bins

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
# ## 17. Per-Model Analysis: Permutation Test + Bonferroni (GPU-accelerated)

# %%
EPS              = 1e-6
GPU_BATCH        = 50   # dense perm test; mem ≈ GPU_BATCH × n_treat × D × 4 bytes
GPU_BATCH_SPARSE = 50   # sparse perm test; mem ≈ GPU_BATCH × n_treat × k × 8 bytes ≈ 1.3 GB

# Bonferroni alpha is computed per-model inside _sig_from_null (n_feats varies: 256 for D0–D4,
# 8192 for D5/D6, 4096 for D7/D8). Displayed here for the D=256 bottleneck models only.
_bonf_d256 = 0.05 / D
print(f"Bonferroni alpha (D={D}): {_bonf_d256:.5f}  "
      f"percentile window: [{100*_bonf_d256/2:.5f}, {100*(1-_bonf_d256/2):.5f}]")
print(f"N_PERM={N_PERM}  GPU_BATCH={GPU_BATCH}  GPU_BATCH_SPARSE={GPU_BATCH_SPARSE}  device={DEVICE}")


@torch.no_grad()
def perm_test_gpu(Z_arr, wt_mask, treat_mask, n_perm, device, eps=EPS, batch=GPU_BATCH):
    Z_wt    = Z_arr[wt_mask]
    Z_treat = Z_arr[treat_mask]
    n_wt, n_treat, D_ = len(Z_wt), len(Z_treat), Z_arr.shape[1]
    N = n_wt + n_treat

    combined   = torch.tensor(np.concatenate([Z_wt, Z_treat]), dtype=torch.float32, device=device)
    total_act  = combined.sum(0)
    total_fire = (combined > 0).float().sum(0)

    null_act  = np.empty((n_perm, D_), dtype=np.float32)
    null_fire = np.empty((n_perm, D_), dtype=np.float32)

    for start in range(0, n_perm, batch):
        bs        = min(batch, n_perm - start)
        rand_vals = torch.rand(bs, N, device=device)
        treat_idx = rand_vals.topk(n_treat, dim=1, largest=False).indices
        del rand_vals
        pd = combined[treat_idx]
        sum_treat_act  = pd.sum(1)
        sum_treat_fire = (pd > 0).sum(1).float()
        del pd, treat_idx
        mean_treat_act  = sum_treat_act  / n_treat
        mean_wt_act     = (total_act  - sum_treat_act)  / n_wt
        mean_treat_fire = sum_treat_fire / n_treat
        mean_wt_fire    = (total_fire - sum_treat_fire) / n_wt
        null_act [start:start+bs] = (mean_treat_act  / (mean_wt_act  + eps)).cpu().numpy()
        null_fire[start:start+bs] = (mean_treat_fire / (mean_wt_fire + eps)).cpu().numpy()

    del combined
    torch.cuda.empty_cache()
    return null_act, null_fire


@torch.no_grad()
def perm_test_gpu_sparse(Z_arr, wt_mask, treat_mask, n_perm, k, device, eps=EPS,
                         batch=GPU_BATCH_SPARSE):
    """Permutation test for TopK SAE where each sample has exactly k non-zero activations.

    The dense approach OOMs for large D (e.g. D5's 8192) because combined = (N, D) is ~5 GB
    on GPU. This version keeps combined on CPU, extracts the (N, k) sparse representation
    (indices + values), copies only that to GPU (~150 MB), then uses scatter_add to accumulate
    per-permutation sums. Peak GPU memory ≈ batch × n_treat × k × 8 bytes ≈ 1.3 GB at batch=50.
    """
    Z_wt    = Z_arr[wt_mask]
    Z_treat = Z_arr[treat_mask]
    n_wt, n_treat, D_ = len(Z_wt), len(Z_treat), Z_arr.shape[1]
    N = n_wt + n_treat

    combined = np.concatenate([Z_wt, Z_treat])   # stays on CPU
    del Z_wt, Z_treat

    # Precompute totals (complement trick; stays on GPU as 1-D vectors)
    total_act  = torch.tensor(combined.sum(0),               dtype=torch.float32, device=device)
    total_fire = torch.tensor((combined > 0).sum(0).astype(np.float32), device=device)

    # Sparse representation: indices of k largest values per sample
    # argpartition is O(D_) per row — fast even for D_=8192
    fire_idx_np = np.argpartition(combined, D_ - k, axis=1)[:, D_ - k:]   # (N, k)
    fire_val_np = combined[np.arange(N)[:, None], fire_idx_np]             # (N, k)
    del combined

    fire_idx_t = torch.tensor(fire_idx_np, dtype=torch.long,    device=device)  # (N, k)
    fire_val_t = torch.tensor(fire_val_np, dtype=torch.float32, device=device)  # (N, k)
    del fire_idx_np, fire_val_np

    null_act  = np.empty((n_perm, D_), dtype=np.float32)
    null_fire = np.empty((n_perm, D_), dtype=np.float32)

    for start in range(0, n_perm, batch):
        bs = min(batch, n_perm - start)

        rand_vals = torch.rand(bs, N, device=device)
        treat_idx = rand_vals.topk(n_treat, dim=1, largest=False).indices   # (bs, n_treat)
        del rand_vals

        # Gather sparse descriptors: (bs, n_treat, k)
        t_fidx = fire_idx_t[treat_idx]   # (bs, n_treat, k)
        t_fval = fire_val_t[treat_idx]   # (bs, n_treat, k)
        del treat_idx

        flat_idx = t_fidx.view(bs, -1)   # (bs, n_treat*k)
        flat_val = t_fval.view(bs, -1)
        del t_fidx, t_fval

        sum_treat_act  = torch.zeros(bs, D_, device=device)
        sum_treat_fire = torch.zeros(bs, D_, device=device)
        sum_treat_act.scatter_add_(1, flat_idx, flat_val)
        # Count only features that actually fired (value > 0); zeros from relu+topk don't count
        sum_treat_fire.scatter_add_(1, flat_idx, (flat_val > 0).float())
        del flat_idx, flat_val

        mean_treat_act  = sum_treat_act  / n_treat
        mean_wt_act     = (total_act  - sum_treat_act)  / n_wt
        mean_treat_fire = sum_treat_fire / n_treat
        mean_wt_fire    = (total_fire - sum_treat_fire) / n_wt
        del sum_treat_act, sum_treat_fire

        null_act [start:start+bs] = (mean_treat_act  / (mean_wt_act  + eps)).cpu().numpy()
        null_fire[start:start+bs] = (mean_treat_fire / (mean_wt_fire + eps)).cpu().numpy()

    del fire_idx_t, fire_val_t, total_act, total_fire
    torch.cuda.empty_cache()
    return null_act, null_fire


def _sig_from_null(obs_act, obs_fire, null_act, null_fire):
    # Bonferroni correction uses actual feature count (supports both D=256 and D5's 8192)
    n_feats = null_act.shape[1]
    ba = 0.05 / n_feats
    lp, up = 100 * ba / 2, 100 * (1 - ba / 2)
    p_act  = np.minimum(
        (null_act  >= obs_act [None, :]).mean(0),
        (null_act  <= obs_act [None, :]).mean(0)) * 2
    p_fire = np.minimum(
        (null_fire >= obs_fire[None, :]).mean(0),
        (null_fire <= obs_fire[None, :]).mean(0)) * 2
    lo_act,  hi_act  = np.percentile(null_act,  [lp, up], axis=0)
    lo_fire, hi_fire = np.percentile(null_fire, [lp, up], axis=0)
    return dict(
        p_act=p_act, p_fire=p_fire,
        sig_enrich_act =(obs_act  > hi_act),  sig_deplete_act =(obs_act  < lo_act),
        sig_enrich_fire=(obs_fire > hi_fire),  sig_deplete_fire=(obs_fire < lo_fire),
    )


RESULTS = {}

for model_name, Z_ms_m in Z_by_model.items():
    print(f"\n=== {model_name} ===", flush=True)
    Z_wt_m   = Z_ms_m[BINS["near neutral"]]
    Z_dest_m = Z_ms_m[BINS["highly destabilizing"]]
    Z_stab_m = Z_ms_m[BINS["highly stabilizing"]]

    # Dispatch: any model with a .k attribute (TopKSAE or SupervisedTopKSAE) uses sparse perm test
    _model_obj = ENCODERS[model_name][0]
    if hasattr(_model_obj, 'k'):
        def _perm(wt_m, treat_m):
            return perm_test_gpu_sparse(Z_ms_m, wt_m, treat_m, N_PERM,
                                        k=_model_obj.k, device=DEVICE)
    else:
        def _perm(wt_m, treat_m):
            return perm_test_gpu(Z_ms_m, wt_m, treat_m, N_PERM, DEVICE)

    # destabilizing vs neutral
    obs_ratio_act_dest  = Z_dest_m.mean(0)       / (Z_wt_m.mean(0)       + EPS)
    obs_ratio_fire_dest = (Z_dest_m > 0).mean(0) / ((Z_wt_m > 0).mean(0) + EPS)
    null_act_dest, null_fire_dest = _perm(BINS["near neutral"], BINS["highly destabilizing"])
    sig_dest = _sig_from_null(obs_ratio_act_dest, obs_ratio_fire_dest,
                               null_act_dest, null_fire_dest)
    print(f"  [dest] Act  enriched={sig_dest['sig_enrich_act'].sum():3d}  "
          f"depleted={sig_dest['sig_deplete_act'].sum():3d}")
    print(f"  [dest] Fire enriched={sig_dest['sig_enrich_fire'].sum():3d}  "
          f"depleted={sig_dest['sig_deplete_fire'].sum():3d}")

    # stabilizing vs neutral
    obs_ratio_act_stab  = Z_stab_m.mean(0)       / (Z_wt_m.mean(0)       + EPS)
    obs_ratio_fire_stab = (Z_stab_m > 0).mean(0) / ((Z_wt_m > 0).mean(0) + EPS)
    null_act_stab, null_fire_stab = _perm(BINS["near neutral"], BINS["highly stabilizing"])
    sig_stab = _sig_from_null(obs_ratio_act_stab, obs_ratio_fire_stab,
                               null_act_stab, null_fire_stab)
    print(f"  [stab] Act  enriched={sig_stab['sig_enrich_act'].sum():3d}  "
          f"depleted={sig_stab['sig_deplete_act'].sum():3d}")
    print(f"  [stab] Fire enriched={sig_stab['sig_enrich_fire'].sum():3d}  "
          f"depleted={sig_stab['sig_deplete_fire'].sum():3d}")

    f_stats_m = np.array([
        f_oneway(*[Z_ms_m[mask, i] for mask in bin_masks]).statistic
        for i in range(Z_ms_m.shape[1])
    ])
    f_stats_m = np.nan_to_num(f_stats_m, nan=0.0)

    RESULTS[model_name] = dict(
        Z_ms=Z_ms_m, Z_wt=Z_wt_m, Z_dest=Z_dest_m, Z_stab=Z_stab_m,
        # destabilizing vs neutral
        obs_ratio_act =obs_ratio_act_dest,  obs_ratio_fire =obs_ratio_fire_dest,
        null_ratio_act=null_act_dest,        null_ratio_fire=null_fire_dest,
        **{k: v for k, v in sig_dest.items()},
        # stabilizing vs neutral (stab suffix)
        obs_ratio_act_stab =obs_ratio_act_stab,  obs_ratio_fire_stab =obs_ratio_fire_stab,
        null_ratio_act_stab=null_act_stab,        null_ratio_fire_stab=null_fire_stab,
        **{f"{k}_stab": v for k, v in sig_stab.items()},
        # shared
        f_stats   =f_stats_m,
        mean_act  =np.stack([Z_ms_m[mask].mean(0)       for mask in bin_masks], axis=1),
        fire_rate =np.stack([(Z_ms_m[mask] > 0).mean(0) for mask in bin_masks], axis=1),
    )

# %% [markdown]
# ## 17b. Spearman Correlation with ΔΔG
#
# Per-neuron Spearman correlation between activation magnitude and ΔΔG across all megascale
# variants. Positive ρ → neuron fires more for destabilizing mutations; negative ρ → fires
# more for stabilizing. Significance via Fisher z-transform (exact for n≈271k). Bonferroni
# correction per model (threshold = 0.05 / n_features). Chunked ranking avoids materializing
# the full ranked matrix for large-dictionary models (D5/D6 with 8192 features).

# %%
def spearman_vs_ddg(Z_ms_m, ddg, tag=""):
    n, D_ = Z_ms_m.shape
    r_ddg = rankdata(ddg).astype(np.float32)
    r_ddg -= r_ddg.mean()
    r_ddg /= np.linalg.norm(r_ddg)

    rho   = np.empty(D_, dtype=np.float32)
    chunk = 256
    for start in range(0, D_, chunk):
        end = min(start + chunk, D_)
        rZ  = np.apply_along_axis(rankdata, 0, Z_ms_m[:, start:end]).astype(np.float32)
        rZ -= rZ.mean(0)
        norms = np.linalg.norm(rZ, axis=0)
        norms[norms == 0] = 1.0
        rZ /= norms
        rho[start:end] = rZ.T @ r_ddg
        if tag and end % 1024 == 0:
            print(f"  {tag} spearman {end}/{D_}", end="\r", flush=True)

    z_stat = np.arctanh(np.clip(rho, -0.9999, 0.9999)) * np.sqrt(n - 3)
    p_vals = 2 * ndtr(-np.abs(z_stat))
    sig    = p_vals < (0.05 / D_)
    return rho, p_vals, sig

SPEARMAN = {}
for model_name, Z_ms_m in Z_by_model.items():
    rho, p_vals, sig = spearman_vs_ddg(Z_ms_m, ms_ddg, tag=model_name)
    sig_destab = sig & (rho > 0)
    sig_stab   = sig & (rho < 0)
    max_rho    = np.abs(rho[sig]).max() if sig.any() else 0.0
    print(f"\n{model_name}: sig_destab={sig_destab.sum()}  sig_stab={sig_stab.sum()}  "
          f"max|ρ|={max_rho:.3f}")
    top10 = np.argsort(np.abs(rho))[::-1][:10]
    print(f"  top-10 |ρ|: {[(f'N{n}', f'ρ={rho[n]:.3f}') for n in top10]}")
    SPEARMAN[model_name] = dict(rho=rho, p_vals=p_vals, sig=sig,
                                sig_destab=sig_destab, sig_stab=sig_stab)
    # Store in RESULTS for use in plots
    if model_name in RESULTS:
        RESULTS[model_name]["spearman_rho"]        = rho
        RESULTS[model_name]["spearman_sig_destab"] = sig_destab
        RESULTS[model_name]["spearman_sig_stab"]   = sig_stab

# %%
fig, axes = plt.subplots(len(SPEARMAN), 2, figsize=(14, 3 * len(SPEARMAN)))
if len(SPEARMAN) == 1:
    axes = axes[np.newaxis, :]
for i, (model_name, S) in enumerate(SPEARMAN.items()):
    rho  = S["rho"]
    sig  = S["sig"]
    fire_dest = RESULTS[model_name]["fire_rate"][:, bin_names.index("highly destabilizing")]

    ax = axes[i, 0]
    ax.hist(rho[~sig],         bins=80, color="lightgray", label="n.s.")
    ax.hist(rho[S["sig_destab"]], bins=40, color="tab:red",  alpha=0.8, label="destab (sig)")
    ax.hist(rho[S["sig_stab"]],   bins=40, color="tab:blue", alpha=0.8, label="stab (sig)")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Spearman ρ (activation vs ΔΔG)")
    ax.set_ylabel("# neurons")
    ax.set_title(f"{model_name}  destab={S['sig_destab'].sum()}  stab={S['sig_stab'].sum()}")
    ax.legend(fontsize=7)

    ax = axes[i, 1]
    for mask, color, label in [(~sig,           "lightgray", "n.s."),
                                (S["sig_destab"], "tab:red",  "destab (sig)"),
                                (S["sig_stab"],   "tab:blue", "stab (sig)")]:
        ax.scatter(fire_dest[mask], np.abs(rho[mask]),
                   c=color, s=8, alpha=0.6, label=label, rasterized=True)
    ax.set_xlabel("Firing rate (highly destabilizing)")
    ax.set_ylabel("|Spearman ρ|")
    ax.set_title(f"{model_name} — |ρ| vs firing rate")
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(OUT_DIR / "v2_spearman_ddg.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 18. Stability Plots — Per Model (4 × 2 layout)

# %%
USE_LOG2 = True

def _tr(ratio):
    return np.log2(ratio + EPS) if USE_LOG2 else ratio

def _center(ratio):
    return _tr(ratio) - (0 if USE_LOG2 else 1)

ref_val      = 0 if USE_LOG2 else 1
ratio_ylabel = ("log₂(fire_rate: treat / neutral)" if USE_LOG2
                else "fire_rate: treat / neutral")
center_label = "log₂(ratio)" if USE_LOG2 else "ratio − 1"

for model_name, R in RESULTS.items():
    top_idx = np.argsort(R["f_stats"])[::-1][:30]

    fig, axes = plt.subplots(4, 2, figsize=(16, 24))
    fig.suptitle(model_name, fontsize=13, fontweight="bold")

    # row 0: shared heatmaps
    ax = axes[0, 0]
    im = ax.imshow(R["fire_rate"][top_idx], aspect="auto", cmap="magma")
    ax.set_xticks(range(n_bins)); ax.set_xticklabels(bin_names, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(top_idx))); ax.set_yticklabels([f"N{n}" for n in top_idx], fontsize=6)
    ax.set_title("Firing rate (top 30 F-stat)"); plt.colorbar(im, ax=ax, fraction=0.03)

    ax = axes[0, 1]
    im = ax.imshow(R["mean_act"][top_idx], aspect="auto", cmap="viridis")
    ax.set_xticks(range(n_bins)); ax.set_xticklabels(bin_names, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(top_idx))); ax.set_yticklabels([f"N{n}" for n in top_idx], fontsize=6)
    ax.set_title("Mean activation (top 30 F-stat)"); plt.colorbar(im, ax=ax, fraction=0.03)

    # rows 1–2: individual comparisons
    comparisons = [
        (1, "highly destabilizing",
         R["obs_ratio_fire"],      R["sig_enrich_fire"],      R["sig_deplete_fire"],
         (R["Z_dest"] > 0).mean(0)),
        (2, "highly stabilizing",
         R["obs_ratio_fire_stab"], R["sig_enrich_fire_stab"], R["sig_deplete_fire_stab"],
         (R["Z_stab"] > 0).mean(0)),
    ]
    for row, label, obs_fire, sig_enrich, sig_deplete, fire_treat in comparisons:
        ratio_vals = _tr(obs_fire)
        # Restrict to neurons that fire for >= 5% of the "positive" (treat) set
        eligible   = np.where(fire_treat >= 0.05)[0]
        top20      = eligible[np.argsort(np.abs(ratio_vals[eligible] - ref_val))[::-1][:20]]

        ax = axes[row, 0]
        colors = ["tab:red"  if sig_enrich[n] else
                  "tab:blue" if sig_deplete[n] else "lightgray" for n in top20]
        ax.barh(range(len(top20)), ratio_vals[top20[::-1]], color=colors[::-1])
        ax.axvline(ref_val, color="k", lw=0.8)
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels([f"N{n}" for n in top20[::-1]], fontsize=7)
        ax.set_xlabel(ratio_ylabel, fontsize=9)
        ax.set_title(f"Firing ratio: {label} vs neutral\n"
                     f"(≥5% treat firing; red=enriched  blue=depleted  Bonferroni)")

        ax = axes[row, 1]
        eligible_mask = fire_treat >= 0.05
        for cat, base_mask, color, size, zo in [
                ("enriched",  sig_enrich,                "tab:red",   40, 4),
                ("depleted",  sig_deplete,                "tab:blue",  40, 4),
                ("n.s.",     ~(sig_enrich | sig_deplete), "lightgray", 10, 2),
        ]:
            mask = base_mask & eligible_mask
            ax.scatter(fire_treat[mask], ratio_vals[mask],
                       c=color, s=size, alpha=0.8, label=cat, zorder=zo)
        ax.axhline(ref_val, color="k", lw=0.8, linestyle="--")
        ax.set_xlabel(f"Firing rate ({label}), ≥5% only", fontsize=9)
        ax.set_ylabel(ratio_ylabel, fontsize=9)
        ax.set_title(f"Enrichment scatter: {label} vs neutral (Bonferroni, ≥5% firing)")
        ax.legend(fontsize=8)

    # row 3: combo analysis
    dest_c = _center(R["obs_ratio_fire"])
    stab_c = _center(R["obs_ratio_fire_stab"])
    combo  = dest_c - stab_c

    sig_dest_any = R["sig_enrich_fire"]      | R["sig_deplete_fire"]
    sig_stab_any = R["sig_enrich_fire_stab"] | R["sig_deplete_fire_stab"]

    dest_anti = (dest_c > 0) & (stab_c < 0) & (sig_dest_any | sig_stab_any)
    stab_anti = (dest_c < 0) & (stab_c > 0) & (sig_dest_any | sig_stab_any)
    other_sig = (sig_dest_any | sig_stab_any) & ~dest_anti & ~stab_anti
    no_sig    = ~(sig_dest_any | sig_stab_any)

    ax = axes[3, 0]
    for cat, mask, color, size, zo in [
            ("n.s.",               no_sig,    "lightgray", 10, 1),
            ("sig, co-directional",other_sig, "#aaaaaa",   25, 2),
            ("dest↑ stab↓",       dest_anti, "tab:red",   55, 4),
            ("dest↓ stab↑",       stab_anti, "tab:blue",  55, 4),
    ]:
        ax.scatter(dest_c[mask], stab_c[mask],
                   c=color, s=size, alpha=0.8, label=f"{cat} (n={mask.sum()})", zorder=zo)
    ax.axhline(0, color="k", lw=0.8, linestyle="--")
    ax.axvline(0, color="k", lw=0.8, linestyle="--")
    lim = max(np.abs(dest_c).max(), np.abs(stab_c).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color="gray", lw=0.6, linestyle=":", label="y=x (equal)")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"destabilizing {center_label}", fontsize=9)
    ax.set_ylabel(f"stabilizing {center_label}", fontsize=9)
    ax.set_title("Combo: destabilizing vs stabilizing\n(neutral=origin; anti-diagonal=directional neurons)")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[3, 1]
    top_combo    = np.argsort(np.abs(combo))[::-1][:20]
    sorted_order = top_combo[np.argsort(combo[top_combo])]
    y_pos  = np.arange(len(sorted_order))
    bar_h  = 0.38
    dest_bar_c = ["tab:red"   if (R["sig_enrich_fire"][n]      or R["sig_deplete_fire"][n])
                  else "#ffbbbb" for n in sorted_order]
    stab_bar_c = ["tab:blue"  if (R["sig_enrich_fire_stab"][n] or R["sig_deplete_fire_stab"][n])
                  else "#bbbbff" for n in sorted_order]
    ax.barh(y_pos + bar_h / 2, dest_c[sorted_order], height=bar_h,
            color=dest_bar_c, label="destabilizing")
    ax.barh(y_pos - bar_h / 2, stab_c[sorted_order], height=bar_h,
            color=stab_bar_c, label="stabilizing", alpha=0.85)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"N{n}" for n in sorted_order], fontsize=7)
    ax.set_xlabel(center_label, fontsize=9)
    ax.set_title("Top 20 neurons by |combo| score\n"
                 "(bright=Bonferroni sig; sorted: stab-specific→dest-specific)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"v2_ddg_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 19. Pathogenicity Sanity Check
#
# For each model: do neurons that are enriched/depleted in destabilising or stabilising
# variants (relative to neutral) also show enrichment/depletion in pathogenic vs benign
# ClinVar variants? This cross-validates whether the stability-sensitive neurons are
# biologically meaningful.

# %%
for model_name, R in RESULTS.items():
    cv = cv_stats[model_name]
    path_ratio = cv["fire_ratio"]    # per-neuron: fire_pathogenic / fire_benign

    dest_c = _center(R["obs_ratio_fire"])
    stab_c = _center(R["obs_ratio_fire_stab"])
    combo  = dest_c - stab_c

    sig_dest_any = R["sig_enrich_fire"]      | R["sig_deplete_fire"]
    sig_stab_any = R["sig_enrich_fire_stab"] | R["sig_deplete_fire_stab"]
    dest_anti    = (dest_c > 0) & (stab_c < 0) & (sig_dest_any | sig_stab_any)
    stab_anti    = (dest_c < 0) & (stab_c > 0) & (sig_dest_any | sig_stab_any)
    other_sig    = (sig_dest_any | sig_stab_any) & ~dest_anti & ~stab_anti
    no_sig       = ~(sig_dest_any | sig_stab_any)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{model_name} — pathogenicity sanity check", fontsize=12, fontweight="bold")

    # panel 1: dest ratio vs path ratio
    ax = axes[0]
    for cat, mask, color, size, zo in [
            ("enriched in dest (sig)",  R["sig_enrich_fire"],  "tab:red",   50, 4),
            ("depleted in dest (sig)",  R["sig_deplete_fire"], "tab:blue",  50, 4),
            ("n.s.",                   ~sig_dest_any,          "lightgray", 10, 2),
    ]:
        ax.scatter(dest_c[mask], path_ratio[mask],
                   c=color, s=size, alpha=0.8, label=cat, zorder=zo)
    ax.axvline(0, color="k", lw=0.8, linestyle="--")
    ax.axhline(1, color="k", lw=0.8, linestyle="--", label="path=beni")
    ax.set_xlabel(f"destabilizing {center_label}", fontsize=9)
    ax.set_ylabel("fire_rate (pathogenic) / fire_rate (benign)", fontsize=9)
    ax.set_title("Destabilising enrichment vs pathogenicity enrichment")
    ax.legend(fontsize=8)

    # panel 2: stab ratio vs path ratio
    ax = axes[1]
    for cat, mask, color, size, zo in [
            ("enriched in stab (sig)",  R["sig_enrich_fire_stab"],  "tab:green",  50, 4),
            ("depleted in stab (sig)",  R["sig_deplete_fire_stab"], "tab:purple", 50, 4),
            ("n.s.",                   ~sig_stab_any,               "lightgray",  10, 2),
    ]:
        ax.scatter(stab_c[mask], path_ratio[mask],
                   c=color, s=size, alpha=0.8, label=cat, zorder=zo)
    ax.axvline(0, color="k", lw=0.8, linestyle="--")
    ax.axhline(1, color="k", lw=0.8, linestyle="--")
    ax.set_xlabel(f"stabilizing {center_label}", fontsize=9)
    ax.set_ylabel("fire_rate (pathogenic) / fire_rate (benign)", fontsize=9)
    ax.set_title("Stabilising enrichment vs pathogenicity enrichment")
    ax.legend(fontsize=8)

    # panel 3: combo score vs path ratio, highlighting anti-diagonal neurons
    ax = axes[2]
    for cat, mask, color, size, zo in [
            ("n.s.",           no_sig,    "lightgray", 10, 1),
            ("co-directional", other_sig, "#aaaaaa",   25, 2),
            ("dest↑ stab↓",   dest_anti, "tab:red",   60, 4),
            ("dest↓ stab↑",   stab_anti, "tab:blue",  60, 4),
    ]:
        ax.scatter(combo[mask], path_ratio[mask],
                   c=color, s=size, alpha=0.8, label=f"{cat} (n={mask.sum()})", zorder=zo)
    ax.axvline(0, color="k", lw=0.8, linestyle="--")
    ax.axhline(1, color="k", lw=0.8, linestyle="--", label="path=beni")
    ax.set_xlabel(f"combo score (dest − stab, {center_label})", fontsize=9)
    ax.set_ylabel("fire_rate (pathogenic) / fire_rate (benign)", fontsize=9)
    ax.set_title("Combo score vs pathogenicity enrichment\n"
                 "(dest↑stab↓ = destab-specific neurons; expect path enrichment?)")
    ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"v2_path_sanity_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 19b. Significance method comparison: per-feature null vs. collaborator pooled null
#
# Our method: per-feature Bonferroni, N_PERM=10,000 two-sided.
# Collaborator method: pool null enrichments from N_COLLAB shuffles × n_active_features
# into a single distribution; one-sided p; Bonferroni/n_active.
# For large-dict models (D5–D8) the pooled threshold is extreme — the global null requires
# exceeding the max enrichment across all active features in all shuffles.

# %%
N_COLLAB = 100
_rng_collab = np.random.default_rng(SEED)

print(f"{'Model':<25} | {'Ours_dest':>9} | {'Collab_dest':>11} | "
      f"{'Ours_stab':>9} | {'Collab_stab':>11} | {'Thr_dest':>10} | {'Thr_stab':>10}")
print("-" * 105)

for model_name, R in RESULTS.items():
    null_fire_dest = R["null_ratio_fire"]        # (N_PERM, D_)
    null_fire_stab = R["null_ratio_fire_stab"]   # (N_PERM, D_)
    obs_fire_dest  = R["obs_ratio_fire"]         # (D_,)
    obs_fire_stab  = R["obs_ratio_fire_stab"]    # (D_,)

    n_ours_dest = (R["sig_enrich_fire"] | R["sig_deplete_fire"]).sum()
    n_ours_stab = (R["sig_enrich_fire_stab"] | R["sig_deplete_fire_stab"]).sum()

    n_perm_r = null_fire_dest.shape[0]
    shuf = _rng_collab.choice(n_perm_r, size=min(N_COLLAB, n_perm_r), replace=False)

    def _collab_n(null_fire, obs_fire):
        active   = obs_fire > 0
        n_active = max(int(active.sum()), 1)
        sub_null = null_fire[shuf][:, active]           # (N_COLLAB, n_active)
        pooled   = np.sort(sub_null.ravel())
        idx      = int((1.0 - 0.05 / n_active) * len(pooled))
        thresh   = float(pooled[min(idx, len(pooled) - 1)])
        n_sig    = int((obs_fire[active] > thresh).sum())
        return n_sig, thresh

    n_col_dest, thr_dest = _collab_n(null_fire_dest, obs_fire_dest)
    n_col_stab, thr_stab = _collab_n(null_fire_stab, obs_fire_stab)

    print(f"{model_name:<25} | {n_ours_dest:>9d} | {n_col_dest:>11d} | "
          f"{n_ours_stab:>9d} | {n_col_stab:>11d} | {thr_dest:>10.3f} | {thr_stab:>10.3f}")

# %% [markdown]
# ## 19c. Model comparison: aggregate metrics and top neuron candidates

# %%
print(f"{'Model':<25} {'D':>6} {'De↑':>6} {'De↓':>6} {'St↑':>6} {'St↓':>6} "
      f"{'Joint':>6} {'maxEnr':>7} {'Sp_de':>7} {'Sp_st':>7}")
print("-" * 95)

for model_name, R in RESULTS.items():
    D_ = len(R["obs_ratio_fire"])
    n_de = int(R["sig_enrich_fire"].sum())
    n_dd = int(R["sig_deplete_fire"].sum())
    n_se = int(R["sig_enrich_fire_stab"].sum())
    n_sd = int(R["sig_deplete_fire_stab"].sum())
    joint = int(((R["sig_enrich_fire"] & R["sig_deplete_fire_stab"]) |
                 (R["sig_deplete_fire"] & R["sig_enrich_fire_stab"])).sum())
    enr = R["obs_ratio_fire"]
    max_enr = float(enr[R["sig_enrich_fire"]].max()) if R["sig_enrich_fire"].any() else 0.0
    sp = SPEARMAN.get(model_name, {})
    sp_dest = int(sp.get("sig_destab", np.zeros(1, dtype=bool)).sum())
    sp_stab = int(sp.get("sig_stab",   np.zeros(1, dtype=bool)).sum())
    print(f"{model_name:<25} {D_:>6d} {n_de:>6d} {n_dd:>6d} {n_se:>6d} {n_sd:>6d} "
          f"{joint:>6d} {max_enr:>7.2f} {sp_dest:>7d} {sp_stab:>7d}")

# %%
# Scatter: log₂ fire-ratio enrichment (destabilizing/neutral) vs Spearman ρ, per model
n_models = len(RESULTS)
ncols = min(3, n_models)
nrows = (n_models + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                         constrained_layout=True)
axes_flat = np.array(axes).ravel()

for ax_i, (model_name, R) in enumerate(RESULTS.items()):
    ax = axes_flat[ax_i]
    enr       = _tr(R["obs_ratio_fire"])
    sp        = SPEARMAN.get(model_name, {})
    rho       = sp.get("rho", np.zeros(len(enr)))
    fire_dest = (R["Z_dest"] > 0).mean(0)
    eligible  = fire_dest >= 0.05

    sig_dest  = R["sig_enrich_fire"] | R["sig_deplete_fire"]
    sig_stab  = R["sig_enrich_fire_stab"] | R["sig_deplete_fire_stab"]
    anti_dest = R["sig_enrich_fire"] & R["sig_deplete_fire_stab"]
    anti_stab = R["sig_deplete_fire"] & R["sig_enrich_fire_stab"]
    other_sig = (sig_dest | sig_stab) & ~anti_dest & ~anti_stab
    neither   = ~(sig_dest | sig_stab)

    for mask, color, label, size, zo in [
            (neither   & eligible, "lightgray", "n.s.",            5,  1),
            (other_sig & eligible, "#ddaaaa",  "sig (not joint)", 15,  2),
            (anti_stab & eligible, "tab:blue", "stab-specific",   40,  4),
            (anti_dest & eligible, "tab:red",  "dest-specific",   40,  4),
    ]:
        ax.scatter(enr[mask], rho[mask], c=color, s=size, alpha=0.8,
                   label=f"{label} (n={mask.sum()})", zorder=zo, rasterized=True)

    ax.axhline(0, color="k", lw=0.8, linestyle="--")
    ax.axvline(0, color="k", lw=0.8, linestyle="--")
    ax.set_title(f"{model_name}  D={len(enr)}  (≥5% dest firing)\n"
                 f"dest-spec={anti_dest.sum()}  stab-spec={anti_stab.sum()}", fontsize=9)
    ax.set_xlabel("log₂ fire ratio (destabilizing/neutral)", fontsize=8)
    ax.set_ylabel("Spearman ρ (activation vs ΔΔG)", fontsize=8)
    ax.legend(fontsize=6, ncol=2)

for ax in axes_flat[n_models:]:
    ax.set_visible(False)

plt.suptitle("Neuron candidate landscape (perm-test enrichment vs Spearman ρ)",
             fontsize=12, fontweight="bold")
plt.savefig(OUT_DIR / "v2_neuron_candidates.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Top-10 dest-specific candidates per model
print("\n=== Top-10 dest-specific candidates (sig_enrich_dest & sig_deplete_stab) ===\n")
for model_name, R in RESULTS.items():
    anti_dest = R["sig_enrich_fire"] & R["sig_deplete_fire_stab"]
    if not anti_dest.any():
        print(f"{model_name}: no dest-specific candidates\n")
        continue
    rho = SPEARMAN.get(model_name, {}).get("rho", np.zeros(len(R["obs_ratio_fire"])))
    enr = R["obs_ratio_fire"]
    scores = (enr * np.abs(rho))[anti_dest]
    cand_i = np.where(anti_dest)[0][np.argsort(scores)[::-1][:10]]
    print(f"{model_name} ({anti_dest.sum()} dest-specific):")
    print(f"  {'Neuron':<10} {'fire_ratio':>10} {'Spearman_ρ':>12} {'score':>10}")
    for n in cand_i:
        print(f"  N{n:<9} {enr[n]:>10.3f} {rho[n]:>12.4f} {enr[n]*abs(rho[n]):>10.4f}")
    print()

# %% [markdown]
# ## 20. Save Models

# %%
torch.save(model_d0.state_dict(),       OUT_DIR / "v2_model_d0.pt")
torch.save(model_d1.state_dict(),       OUT_DIR / "v2_model_d1.pt")
torch.save(model_d2.state_dict(),       OUT_DIR / "v2_model_d2.pt")
torch.save(model_unsup.state_dict(),    OUT_DIR / "v2_model_d3_unsup.pt")
torch.save(model_unsup_vt.state_dict(), OUT_DIR / "v2_model_d4_unsup_vt.pt")
torch.save(model_d5.state_dict(),       OUT_DIR / "v2_model_d5_topk.pt")
torch.save(model_d6.state_dict(),       OUT_DIR / "v2_model_d6_sup_topk.pt")
torch.save(model_d7.state_dict(),       OUT_DIR / "v2_model_d7_topk_diff.pt")
torch.save(model_d8.state_dict(),       OUT_DIR / "v2_model_d8_sup_topk_diff.pt")
print("Models saved.")
