# %% [markdown]
# # train_combined_topksae.py
#
# Trains TopK SAEs on ClinVar + gnomAD + HGMD variants combined (unsupervised).
# Two input types: concat(WT, VT) (2048-dim) and diff VT−WT (1024-dim).
# Multiple EF/K hyperparameter variants.
#
# Data sources:
#   ClinVar   /data/ross/ppi_lossgain/interaction_loss/clinvar/prott5_subgraphs.h5
#   gnomAD    /data/ross/ppi_lossgain/interaction_loss/gnomad/prott5_subgraphs.h5
#   HGMD      /data/ross/ppi_lossgain/interaction_loss/hgmd/prott5_embeddings.h5
#
# Models saved to: /data/ross/ppi_lossgain/interaction_loss/sae_weights/combined/
# Feature cache:   /data/ross/ppi_lossgain/interaction_loss/sae_weights/combined/

# %% [markdown]
# ## 1. Imports and Config

# %%
import re
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
CLINVAR_H5  = Path("/data/ross/ppi_lossgain/interaction_loss/clinvar/prott5_subgraphs.h5")
GNOMAD_H5   = Path("/data/ross/ppi_lossgain/interaction_loss/gnomad/prott5_subgraphs.h5")
HGMD_H5     = Path("/data/ross/ppi_lossgain/interaction_loss/hgmd/prott5_embeddings.h5")
SAVE_DIR    = Path("/data/ross/ppi_lossgain/interaction_loss/sae_weights/combined")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── Training hyperparams ───────────────────────────────────────────────────────
DEVICE      = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 512
LR          = 1e-3
MAX_EPOCHS  = 100
ES_PATIENCE = 5
VAL_FRAC    = 0.15

# TopK training constants (match clinvar_sparse_bottleneck_v2.py)
AUXK_ALPHA  = 1 / 32        # dead-neuron auxiliary loss weight
DEAD_THRESH = 10_000_000    # samples since last fire before neuron counts as dead

# ── Model configurations ───────────────────────────────────────────────────────
# Each entry: (name_suffix, input_type, in_dim, ef, k)
# input_type: "concat" (2048-dim WT+VT) or "diff" (1024-dim VT-WT)
#
# Sparsity note:
#   EF=4, K=128, dim=8192 → 1.56% active  (concat baseline, matches D5)
#   EF=4, K=64,  dim=4096 → 1.56% active  (diff baseline, matches D7)
#   EF=4, K=64,  dim=8192 → 0.78% active  (sparser concat)
#   EF=1, K=128, dim=2048 → 6.25% active  (concat, smaller dict)
#   EF=1, K=64,  dim=1024 → 6.25% active  (diff, smaller dict)
MODEL_CONFIGS = [
    ("concat_ef4_k128", "concat", 2048, 4, 128),  # matches D5 architecture
    ("concat_ef4_k64",  "concat", 2048, 4,  64),  # sparser
    ("concat_ef1_k128", "concat", 2048, 1, 128),  # smaller dict
    ("diff_ef4_k256",   "diff",   1024, 4, 256),  # diff, EF=4, K=256 (6.25% active)
    ("diff_ef4_k64",    "diff",   1024, 4,  64),  # matches D7 architecture
    ("diff_ef4_k32",    "diff",   1024, 4,  32),  # sparser
    ("diff_ef1_k64",    "diff",   1024, 1,  64),  # smaller dict
]

# ── CLI: optionally train a subset ────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train combined multi-source TopK SAEs")
parser.add_argument("--models", nargs="*", metavar="NAME",
                    help="Subset of model name suffixes to train (default: all)")
parser.add_argument("--rebuild-cache", action="store_true",
                    help="Re-read HDF5 files even if cache exists")
args, _ = parser.parse_known_args()

if args.models:
    MODEL_CONFIGS = [c for c in MODEL_CONFIGS if c[0] in args.models]
    print(f"Training subset: {[c[0] for c in MODEL_CONFIGS]}")


# %% [markdown]
# ## 2. Data Loading

# %%
def load_subgraph_h5(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load WT and VT embeddings from prott5_subgraphs.h5 (ClinVar / gnomAD format).

    Structure: top-level groups {UniProtA}_{UniProtB} → variant subgroups (e.g. A823V)
      node_emb (N_nodes, 1024): VT ProtT5 embeddings for subgraph residues
      mut_diff (1024,): VT − WT diff at mutation site
      attr mut_local_idx: int, index of mutated residue in subgraph
    """
    try:
        import h5py
    except ImportError:
        sys.exit("h5py not found. Try: pip install h5py  or activate the correct env.")

    wt_list, vt_list = [], []
    seen = set()   # (protein_id, variant_key) — deduplicate across complexes
    skipped = 0
    with h5py.File(str(h5_path), "r") as f:
        for complex_key in tqdm(f.keys(), desc=h5_path.stem, leave=False):
            prot_id = complex_key.split("_")[0]   # mutated protein is first in pair
            cgrp = f[complex_key]
            for var_key in cgrp.keys():
                uid = (prot_id, var_key)
                if uid in seen:
                    continue
                seen.add(uid)
                vgrp = cgrp[var_key]
                try:
                    node_emb      = vgrp["node_emb"][:]        # (N_nodes, 1024)
                    mut_diff      = vgrp["mut_diff"][:]         # (1024,)
                    mut_local_idx = int(vgrp.attrs["mut_local_idx"])
                    vt_emb = node_emb[mut_local_idx]            # (1024,) VT at mut site
                    wt_emb = vt_emb - mut_diff                  # (1024,) WT = VT − (VT−WT)
                    wt_list.append(wt_emb)
                    vt_list.append(vt_emb)
                except Exception:
                    skipped += 1
    if skipped:
        print(f"  [{h5_path.stem}] skipped {skipped} malformed entries")
    return np.stack(wt_list).astype(np.float32), np.stack(vt_list).astype(np.float32)


def _parse_position(variant_str: str) -> int | None:
    """Parse 0-indexed mutation position from variant string like G371V or A12T.

    Returns None if the string cannot be parsed (insertions, deletions, etc.).
    """
    m = re.match(r'^[A-Za-z*](\d+)[A-Za-z*]$', variant_str)
    if m:
        return int(m.group(1))   # HGMD uses 0-indexed positions directly
    return None


def load_hgmd_h5(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load WT and VT embeddings from prott5_embeddings.h5 (HGMD format).

    Structure (flat keys):
      '{protein_id}'          → (seq_len, 1024) full WT sequence embedding
      '{protein_id} {var}'   → (seq_len, 1024) full VT sequence embedding

    Mutation-site extraction: position parsed from variant string (0-indexed — verified against FASTA).
    """
    try:
        import h5py
    except ImportError:
        sys.exit("h5py not found.")

    wt_list, vt_list = [], []
    skipped = 0
    with h5py.File(str(h5_path), "r") as f:
        all_keys = list(f.keys())
        vt_keys  = [k for k in all_keys if ' ' in k]
        wt_set   = set(all_keys) - set(vt_keys)

        for vt_key in tqdm(vt_keys, desc=h5_path.stem, leave=False):
            parts    = vt_key.split(' ', 1)
            prot_id  = parts[0]
            variant  = parts[1]

            if prot_id not in wt_set:
                skipped += 1
                continue

            pos = _parse_position(variant)
            if pos is None:
                skipped += 1
                continue

            try:
                wt_full = f[prot_id][:]      # (seq_len, 1024)
                vt_full = f[vt_key][:]       # (seq_len, 1024)
                if pos >= wt_full.shape[0] or pos >= vt_full.shape[0]:
                    skipped += 1
                    continue
                wt_list.append(wt_full[pos].astype(np.float32))
                vt_list.append(vt_full[pos].astype(np.float32))
            except Exception:
                skipped += 1

    if skipped:
        print(f"  [HGMD] skipped {skipped} unparseable / out-of-bounds entries")
    return np.stack(wt_list).astype(np.float32), np.stack(vt_list).astype(np.float32)


# ── Build or load combined WT/VT cache ────────────────────────────────────────
WT_CACHE = SAVE_DIR / "combined_wt.npy"
VT_CACHE = SAVE_DIR / "combined_vt.npy"

if WT_CACHE.exists() and VT_CACHE.exists() and not args.rebuild_cache:
    print(f"Loading cached embeddings from {SAVE_DIR} …")
    wt_all = np.load(WT_CACHE)
    vt_all = np.load(VT_CACHE)
    print(f"  combined WT: {wt_all.shape}  VT: {vt_all.shape}")
else:
    print("Building combined WT/VT embedding cache …")
    sources = []

    print("Loading ClinVar …")
    wt_cv, vt_cv = load_subgraph_h5(CLINVAR_H5)
    print(f"  ClinVar: {len(wt_cv):,} variants")
    sources.append((wt_cv, vt_cv))

    print("Loading gnomAD …")
    wt_gn, vt_gn = load_subgraph_h5(GNOMAD_H5)
    print(f"  gnomAD:  {len(wt_gn):,} variants")
    sources.append((wt_gn, vt_gn))

    print("Loading HGMD …")
    wt_hg, vt_hg = load_hgmd_h5(HGMD_H5)
    print(f"  HGMD:    {len(wt_hg):,} variants")
    sources.append((wt_hg, vt_hg))

    wt_all = np.concatenate([s[0] for s in sources], axis=0)
    vt_all = np.concatenate([s[1] for s in sources], axis=0)
    print(f"Combined: {len(wt_all):,} variants total")

    np.save(WT_CACHE, wt_all)
    np.save(VT_CACHE, vt_all)
    print(f"Saved cache → {SAVE_DIR}")


# %% [markdown]
# ## 3. TopKSAE Model

# %%
class TopKSAE(nn.Module):
    """Over-complete SAE with hard TopK sparsity (unsupervised reconstruction).

    Ported from clinvar_sparse_bottleneck_v2.py. Key properties:
    - dict_size = ef * in_dim (expansion)
    - Exactly k features fire per sample (no L1 penalty)
    - b_dec handles input centering (subtracted before encoder, added after decoder)
    - decoder has no bias (b_dec provides that offset)
    """
    def __init__(self, in_dim: int, ef: int, k: int):
        super().__init__()
        d = ef * in_dim
        self.k = k
        self.d = d
        self.in_dim  = in_dim
        self.ef      = ef
        self.encoder = nn.Linear(in_dim, d)
        self.decoder = nn.Linear(d, in_dim, bias=False)
        self.register_buffer("b_dec", torch.zeros(in_dim))

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


# %% [markdown]
# ## 4. Training Utilities

# %%
def _norm_decoder(model: TopKSAE):
    """Renormalize decoder columns to unit L2 norm after each optimizer step."""
    with torch.no_grad():
        W = model.decoder.weight.data          # (in_dim, d)
        model.decoder.weight.data = F.normalize(W, dim=0)


def _remove_parallel_grad(model: TopKSAE):
    """Project out gradient components parallel to each decoder column.

    Prevents the optimizer from undoing the unit-norm constraint.
    """
    W = model.decoder.weight        # (in_dim, d)
    g = model.decoder.weight.grad
    if g is None:
        return
    parallel = (g * W).sum(0, keepdim=True) * W
    model.decoder.weight.grad = g - parallel


def _geometric_median(X: torch.Tensor, n_iter: int = 50) -> torch.Tensor:
    """Weiszfeld's algorithm for geometric median of rows of X."""
    m = X.mean(0)
    for _ in range(n_iter):
        dists = torch.norm(X - m, dim=1, keepdim=True).clamp(min=1e-8)
        w = 1.0 / dists
        m = (w * X).sum(0) / w.sum()
    return m


def _auxk_loss(model: TopKSAE, x: torch.Tensor, x_hat: torch.Tensor,
               pre_act: torch.Tensor, tokens_since_fired: torch.Tensor) -> torch.Tensor:
    """Dead-neuron auxiliary reconstruction loss.

    Latents silent for >= DEAD_THRESH samples reconstruct the residual using their
    top-AUXK pre-activations. Loss is residual-variance-normalized.
    """
    dead = tokens_since_fired >= DEAD_THRESH
    if not dead.any():
        return torch.tensor(0.0, device=x.device)
    k_aux   = min(model.in_dim // 2, int(dead.sum()))
    masked  = torch.where(dead, pre_act, torch.full_like(pre_act, float("-inf")))
    aux_v, aux_i = masked.topk(k_aux, dim=-1, sorted=False)
    aux_z   = torch.zeros_like(pre_act).scatter_(-1, aux_i, aux_v)
    residual = (x - x_hat).detach()
    x_aux   = model.decoder(aux_z)   # no b_dec: target is already a residual
    l2_aux  = (residual - x_aux).pow(2).sum(-1).mean()
    denom   = (residual - residual.mean(0)).pow(2).sum(-1).mean()
    return (l2_aux / denom).nan_to_num(0.0)


def train_topksae(model: TopKSAE, X: np.ndarray, device: torch.device,
                  save_path: Path, tag: str = "") -> dict:
    """Train one TopKSAE model on X (numpy array, shape (N, in_dim)).

    Returns history dict with train_loss, val_loss, dead_neurons per epoch.
    Saves best checkpoint to save_path.
    """
    N = len(X)
    n_val  = max(1, int(N * VAL_FRAC))
    n_tr   = N - n_val
    perm   = np.random.permutation(N)
    tr_idx = perm[:n_tr]
    vl_idx = perm[n_tr:]

    X_tr = torch.tensor(X[tr_idx], dtype=torch.float32)
    X_vl = torch.tensor(X[vl_idx], dtype=torch.float32)

    tr_loader = DataLoader(TensorDataset(X_tr), batch_size=BATCH_SIZE, shuffle=True)
    vl_loader = DataLoader(TensorDataset(X_vl), batch_size=BATCH_SIZE * 4, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    tokens_since_fired = torch.zeros(model.d, device=device)

    best_val_loss = float("inf")
    best_state    = None
    patience_ctr  = 0
    history       = {"train_loss": [], "val_loss": [], "dead_neurons": []}
    initialized   = False

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0.0

        for (xb,) in tr_loader:
            xb = xb.to(device)

            if not initialized:
                with torch.no_grad():
                    model.b_dec.data = _geometric_median(xb)
                initialized = True

            z, topk_vals, topk_idx, pre_act = model.encode(xb)
            x_hat = model.decode(z)
            l2    = (xb - x_hat).pow(2).sum(-1).mean()
            auxk  = _auxk_loss(model, xb, x_hat, pre_act, tokens_since_fired)
            loss  = l2 + AUXK_ALPHA * auxk

            fired = torch.zeros(model.d, dtype=torch.bool, device=device)
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
        with torch.no_grad():
            for (xb,) in vl_loader:
                xb = xb.to(device)
                _, x_hat = model(xb)
                val_loss += (xb - x_hat).pow(2).sum(-1).mean().item()
        val_loss /= len(vl_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["dead_neurons"].append(dead_count)
        print(f"{tag} Epoch {epoch+1:3d} | train={train_loss:.5f} val={val_loss:.5f} "
              f"dead={dead_count}/{model.d}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= ES_PATIENCE:
                print(f"{tag} Early stop at epoch {epoch+1}  "
                      f"(best val_loss={best_val_loss:.5f})")
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), str(save_path))
    print(f"{tag} Saved → {save_path}")
    return history


# %% [markdown]
# ## 5. Main Training Loop

# %%
print(f"\nDevice: {DEVICE}")
print(f"Combined dataset: {len(wt_all):,} variants")
print(f"Training {len(MODEL_CONFIGS)} model(s): {[c[0] for c in MODEL_CONFIGS]}\n")

all_histories = {}

for name, input_type, in_dim, ef, k in MODEL_CONFIGS:
    save_path = SAVE_DIR / f"combined_{name}.pt"
    if save_path.exists():
        print(f"[{name}] Already trained — {save_path} exists. Skipping.")
        print("  (Delete the file or pass --models to retrain.)")
        continue

    print(f"\n{'='*70}")
    print(f"Training [{name}]  input={input_type}  in_dim={in_dim}  EF={ef}  K={k}")
    print(f"  dict_size={ef*in_dim}  sparsity={k/(ef*in_dim)*100:.2f}%")
    print(f"  save → {save_path}")
    print(f"{'='*70}")

    if input_type == "concat":
        X = np.concatenate([wt_all, vt_all], axis=1)   # (N, 2048)
    else:
        X = (vt_all - wt_all)                           # (N, 1024)

    model = TopKSAE(in_dim=in_dim, ef=ef, k=k).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    history = train_topksae(model, X, DEVICE, save_path, tag=f"[{name}]")
    all_histories[name] = history

    # save training history alongside weights
    hist_path = SAVE_DIR / f"history_{name}.npz"
    np.savez(str(hist_path),
             train_loss=np.array(history["train_loss"]),
             val_loss=np.array(history["val_loss"]),
             dead_neurons=np.array(history["dead_neurons"]))
    print(f"  History → {hist_path}")


# %% [markdown]
# ## 6. Summary

# %%
print("\n" + "="*70)
print("Training complete.")
for name, hist in all_histories.items():
    best_val = min(hist["val_loss"])
    n_epochs = len(hist["val_loss"])
    final_dead = hist["dead_neurons"][-1] if hist["dead_neurons"] else "n/a"
    print(f"  {name:<22} epochs={n_epochs:3d}  best_val={best_val:.5f}  "
          f"dead@end={final_dead}")

print(f"\nModels saved in: {SAVE_DIR}")
