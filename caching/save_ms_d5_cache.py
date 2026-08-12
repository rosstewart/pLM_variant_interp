"""
Temp script: load D5 TopKSAE, encode MegaScale variants, save missing cache files.

Saves to:
  /data/ross/interp/ms_z_d5_sparse.npz   — D5 encodings (sparse CSR)
  /data/ross/interp/ms_x_diff.npy        — final-layer VT-WT diff (dense)
  <OUT_DIR>/ms_ddg.npy                   — ΔΔG labels
  <OUT_DIR>/ms_protein_ids.npy           — protein ID per variant
"""

import pickle, sys
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
from pathlib import Path
from tqdm import tqdm

DEVICE    = torch.device("cuda:2")
OUT_DIR   = Path("/home/rcstewart/ppi_lossgain/sparse_bottleneck")
MS_CACHE  = Path("/data/ross/interp")
MS_CACHE.mkdir(parents=True, exist_ok=True)

MEGASCALE_PKL = "/data/ross/ppi_lossgain/interaction_loss/megascale_preprocessed/preprocessed.pkl"
D5_WEIGHTS    = OUT_DIR / "v2_model_d5_topk.pt"
FEATS_CACHE   = OUT_DIR / "megascale_feats.npy"

EF_TOPK = 4
K_TOPK  = 128

# ── D5 model definition (copied from v2.py) ───────────────────────────────────
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
        z, topk_vals = self.encode(x)
        return z, topk_vals


# ── Load MegaScale pkl ────────────────────────────────────────────────────────
print("Loading MegaScale pkl …")
with open(MEGASCALE_PKL, "rb") as f:
    ms = pickle.load(f)

vt_ids    = ms["vt_ids"]
ms_ddg    = ms["ddg_labels"]
ms_mutidx = ms["mutation_indices"]
ms_diffs  = np.array(ms["mutation_site_diffs"])
ms_embs   = ms["prott5_embeddings"]
N_ms      = len(ms_ddg)
print(f"  N={N_ms:,}  ΔΔG∈[{ms_ddg.min():.2f}, {ms_ddg.max():.2f}]")


# ── Build or load X_ms (WT+VT concat, 2048-dim) ───────────────────────────────
def expand_emb(emb):
    """Expand stored embedding to full (seq_len, 1024) array."""
    if isinstance(emb, np.ndarray) and emb.ndim == 2:
        return emb
    # sparse or compressed format — try scipy sparse
    if sp.issparse(emb):
        return emb.toarray()
    return np.array(emb)

if FEATS_CACHE.exists():
    print(f"Loading cached X_ms from {FEATS_CACHE} …")
    X_ms = np.load(FEATS_CACHE)
else:
    print("Building X_ms (WT+VT concat) …")
    X_ms = np.empty((N_ms, 2048), dtype=np.float32)
    for i in tqdm(range(N_ms)):
        full_emb = expand_emb(ms_embs[i])
        mi       = int(ms_mutidx[i])
        vt_emb   = full_emb[mi]
        wt_emb   = vt_emb - ms_diffs[i]
        X_ms[i]  = np.concatenate([wt_emb, vt_emb])
    np.save(FEATS_CACHE, X_ms)
    print(f"  Saved to {FEATS_CACHE}")

print(f"X_ms: {X_ms.shape}")


# ── Save small metadata (always overwrite to be safe) ─────────────────────────
print("Saving ms_ddg.npy and ms_protein_ids.npy …")
np.save(OUT_DIR / "ms_ddg.npy", ms_ddg.astype(np.float32))
np.save(OUT_DIR / "ms_protein_ids.npy",
        np.array([v.rsplit(" ", 1)[0] for v in vt_ids], dtype=object))


# ── Save ms_x_diff (final-layer VT-WT diff) ───────────────────────────────────
x_diff_path = MS_CACHE / "ms_x_diff.npy"
if not x_diff_path.exists():
    print("Saving ms_x_diff.npy …")
    np.save(x_diff_path, (X_ms[:, 1024:] - X_ms[:, :1024]).astype(np.float32))
else:
    print(f"  {x_diff_path} already exists, skipping.")


# ── Encode with D5 and save sparse ────────────────────────────────────────────
sparse_path = MS_CACHE / "ms_z_d5_sparse.npz"
if sparse_path.exists():
    print(f"  {sparse_path} already exists, skipping D5 encoding.")
    sys.exit(0)

print("Loading D5 weights …")
model_d5 = TopKSAE(in_dim=2048).to(DEVICE)
state = torch.load(D5_WEIGHTS, map_location=DEVICE)
model_d5.load_state_dict(state)
model_d5.eval()
print(f"  D5 loaded: dict={model_d5.d}  k={model_d5.k}")

print("Encoding MegaScale with D5 …")
ENCODE_BATCH = 2048
parts = []
with torch.no_grad():
    for i in tqdm(range(0, N_ms, ENCODE_BATCH)):
        xb = torch.tensor(X_ms[i:i+ENCODE_BATCH], dtype=torch.float32).to(DEVICE)
        z, _ = model_d5(xb)
        parts.append(z.cpu().numpy())

Z_d5 = np.concatenate(parts, axis=0)
print(f"  Z_d5: {Z_d5.shape}  sparsity={1 - np.count_nonzero(Z_d5) / Z_d5.size:.3%} nonzero")

print(f"Saving sparse NPZ to {sparse_path} …")
Z_sparse = sp.csr_matrix(Z_d5.astype(np.float32))
sp.save_npz(str(sparse_path), Z_sparse)
print(f"  Done. nnz={Z_sparse.nnz:,}  size≈{sparse_path.stat().st_size/1e6:.0f} MB")
