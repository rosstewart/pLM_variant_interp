# %% [markdown]
# # Collaborator SAE Analysis — ΔZ at ProtT5 Layer 20
#
# Compares collaborator's pre-trained sparse autoencoder (trained on UniRef residue-level
# ProtT5 representations) with MegaScale stability data.
#
# For each megascale variant:
#   1. Extract layer-20 ProtT5 hidden state at the mutated residue, for both WT and VT
#      sequences.
#   2. Encode both through the shared SAE: z_wt, z_vt (each k=256-sparse, dim=16384).
#   3. ΔZ = z_vt − z_wt.
#   4. ΔZ_pos = max(ΔZ, 0) — features *gained* by the mutation.
#      ΔZ_neg = max(−ΔZ, 0) — features *lost* by the mutation.
#   5. Run the same permutation-test and Spearman analysis as v2.

# %%
import os, sys, re, pickle, warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from scipy.io import loadmat
from scipy.stats import rankdata
from scipy.special import ndtr
import matplotlib.pyplot as plt

# ─ Collaborator's dictionary_learning package ─────────────────────────────────
_DL_ROOT = Path("/home/rcstewart/ppi_lossgain/sparse_bottleneck/pLMinterp")
if str(_DL_ROOT) not in sys.path:
    sys.path.insert(0, str(_DL_ROOT))
from dictionary_learning.dictionary_learning.trainers.top_k import AutoEncoderTopK

# ─ transformers ───────────────────────────────────────────────────────────────
from transformers import T5Tokenizer, T5EncoderModel

# %% [markdown]
# ## 1. Config

# %%
DEVICE    = torch.device("cuda:2")
SEED      = 42
N_PERM    = 10_000
EPS       = 1e-6
GPU_BATCH_SPARSE = 50

SAE_PATH  = Path("/data/karna/model_weights/sae_weights/t5/trainer_0/t5_layer20_topk256_ef16.pt")
T5_NAME   = "Rostlab/prot_t5_xl_half_uniref50-enc"
T5_LAYER  = 20          # layer index to hook (0-based); SAE trained on layer 20

GRAPH_DIR = Path("/data/ross/ppi_lossgain/interaction_loss/megascale_preprocessed/graphs")
MS_PKL    = Path("/data/ross/ppi_lossgain/interaction_loss/megascale_preprocessed/preprocessed.pkl")
OUT_DIR   = Path("/home/rcstewart/ppi_lossgain/sparse_bottleneck")
CACHE_DIR = Path("/data/ross/interp/collab_sae_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

EXTRACT_BATCH = 16   # VT sequences per ProtT5 forward pass (same-length seqs per protein)

_var_re = re.compile(r'^([A-Z])(\d+)([A-Z])$')

np.random.seed(SEED)
torch.manual_seed(SEED)
print(f"device: {DEVICE}  SAE: {SAE_PATH.name}")

# %% [markdown]
# ## 2. Load MegaScale Preprocessed Data

# %%
print("Loading preprocessed.pkl …")
with open(MS_PKL, "rb") as f:
    ms = pickle.load(f)

vt_ids       = ms["vt_ids"]                        # list of "pid variant_str"
ms_ddg       = ms["ddg_labels"].astype(np.float32) # (N,)
mut_indices  = ms["mutation_indices"]              # (N,) 0-based mutated position
N_ms         = len(vt_ids)

print(f"  N_variants={N_ms:,}  N_proteins={len(set(v.rsplit(' ',1)[0] for v in vt_ids)):,}")
print(f"  ΔΔG range: [{ms_ddg.min():.2f}, {ms_ddg.max():.2f}]")

# %% [markdown]
# ## 3. Build Protein Sequence Lookup from Graph .mat Files

# %%
def _read_mat_seq(mat_path: Path) -> str:
    mat = loadmat(str(mat_path))
    raw = mat["L"]
    flat = raw.flatten() if hasattr(raw, "flatten") else raw
    return "".join(str(c) for c in flat)

print("Reading graph .mat files …")
pid_to_seq: dict[str, str] = {}
for mat_path in sorted(GRAPH_DIR.glob("*.mat")):
    pid = mat_path.stem
    try:
        pid_to_seq[pid] = _read_mat_seq(mat_path)
    except Exception as e:
        print(f"  warning: could not read {mat_path.name}: {e}")

print(f"  Loaded {len(pid_to_seq)} sequences.  "
      f"Coverage: {sum(v.rsplit(' ',1)[0] in pid_to_seq for v in vt_ids):,}/{N_ms:,} variants")

# %% [markdown]
# ## 4. ProtT5 Loader and Batched Layer-N Extraction
#
# Uses `output_hidden_states=True` (native HuggingFace) — no nnsight needed.
# HF T5Stack collects hidden states BEFORE each block, so indexing is:
#   hidden_states[0]          = embedding output (input to block 0)
#   hidden_states[i]          = output of block i−1
#   hidden_states[T5_LAYER+1] = output of block T5_LAYER  ← what we want
# T5 has no leading CLS token; residue pos maps directly to token index pos.

# %%
def load_prott5(device):
    print(f"Loading ProtT5 ({T5_NAME}) …")
    tokenizer = T5Tokenizer.from_pretrained(T5_NAME, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(T5_NAME).to(device)
    model.eval()
    print(f"  ProtT5 loaded.")
    return model, tokenizer


@torch.no_grad()
def get_hidden_at_positions(model, tokenizer, seqs: list[str],
                             positions: list[int], layer: int, device) -> np.ndarray:
    """
    Batch forward pass; returns the hidden state at block `layer` output for each
    sequence at the specified (per-sequence) residue position.

    Returns: (len(seqs), 1024) float32 array.
    """
    t5_inputs = [" ".join(list(s)) for s in seqs]
    inputs = tokenizer(t5_inputs, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs, output_hidden_states=True)
    # hidden_states[layer+1] = output of block `layer`
    hidden = outputs.hidden_states[layer + 1]   # (batch, max_seq_len, 1024)
    result = np.empty((len(seqs), 1024), dtype=np.float32)
    for b, pos in enumerate(positions):
        result[b] = hidden[b, pos, :].detach().cpu().float().numpy()
    return result


@torch.no_grad()
def get_all_positions_single_seq(model, tokenizer, seq: str,
                                  layer: int, device) -> np.ndarray:
    """
    ONE forward pass for a single sequence; returns all residue positions.
    Returns: np.ndarray(len(seq), 1024) — index directly by position.
    Use for WT sequences so the cost is one pass per protein, not one per
    unique mutation position.
    """
    t5_input = " ".join(list(seq))
    inputs = tokenizer([t5_input], return_tensors="pt", padding=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer + 1]   # (1, seq_len, 1024)
    return hidden[0, :len(seq), :].detach().cpu().float().numpy()

# %% [markdown]
# ## 5. Extract Layer-20 Representations (Cached)
#
# For each variant: extract hidden state at the mutated residue for WT and VT sequences.
# WT sequences are deduplicated per protein (one ProtT5 pass per protein × k VT-batch passes).
# Cached to `collab_sae_cache/layer20_wt.npy` and `collab_sae_cache/layer20_vt.npy`.

# %%
CACHE_WT  = CACHE_DIR / "layer20_wt.npy"
CACHE_VT  = CACHE_DIR / "layer20_vt.npy"
CACHE_MASK = CACHE_DIR / "valid_mask.npy"

if CACHE_WT.exists() and CACHE_VT.exists() and CACHE_MASK.exists():
    print("Loading cached layer-20 representations …")
    h_wt       = np.load(CACHE_WT)
    h_vt       = np.load(CACHE_VT)
    valid_mask = np.load(CACHE_MASK).astype(bool)
    print(f"  h_wt={h_wt.shape}  h_vt={h_vt.shape}  valid={valid_mask.sum():,}/{N_ms:,}")
    _pid_cache = CACHE_DIR / "protein_ids_valid.npy"
    if _pid_cache.exists():
        protein_ids_valid = np.load(_pid_cache, allow_pickle=True)
    else:
        protein_ids_valid = np.array(
            [vt_ids[i].rsplit(" ", 1)[0] for i in np.where(valid_mask)[0]], dtype=object)

else:
    print("Extracting layer-20 representations (first run — will cache) …")
    t5_model, tokenizer = load_prott5(DEVICE)

    h_wt       = np.zeros((N_ms, 1024), dtype=np.float32)
    h_vt       = np.zeros((N_ms, 1024), dtype=np.float32)
    valid_mask = np.zeros(N_ms, dtype=bool)

    # Group variant indices by protein
    pid_to_indices: dict[str, list[int]] = {}
    for i, vid in enumerate(vt_ids):
        pid, variant_str = vid.rsplit(" ", 1)
        if pid not in pid_to_seq:
            continue
        pid_to_indices.setdefault(pid, []).append(i)

    for pid, indices in tqdm(pid_to_indices.items(), desc="proteins"):
        wt_seq = pid_to_seq[pid]

        # ─── WT: ONE forward pass for the whole protein; read positions directly ──
        wt_all = get_all_positions_single_seq(t5_model, tokenizer, wt_seq, T5_LAYER, DEVICE)
        # wt_all: (len(wt_seq), 1024)
        for i in indices:
            pos = int(mut_indices[i])
            if pos < len(wt_seq):
                h_wt[i] = wt_all[pos]

        # ─── VT: batch by position (same-length seqs within one protein) ─────
        pos_to_variant_indices: dict[int, list[int]] = {}
        for i in indices:
            pos_to_variant_indices.setdefault(int(mut_indices[i]), []).append(i)

        for pos, var_indices in pos_to_variant_indices.items():
            if pos >= len(wt_seq):
                continue
            vt_seqs = []
            for i in var_indices:
                _, var_str_i = vt_ids[i].rsplit(" ", 1)
                m = _var_re.match(var_str_i)
                vt_seqs.append(wt_seq[:pos] + m.group(3) + wt_seq[pos + 1:]
                               if m else None)

            for batch_start in range(0, len(var_indices), EXTRACT_BATCH):
                batch_seqs = vt_seqs[batch_start:batch_start + EXTRACT_BATCH]
                batch_idx  = var_indices[batch_start:batch_start + EXTRACT_BATCH]
                valid_b = [(j, s) for j, s in zip(batch_idx, batch_seqs) if s is not None]
                if not valid_b:
                    continue
                b_idx, b_seqs = zip(*valid_b)
                reps = get_hidden_at_positions(
                    t5_model, tokenizer,
                    list(b_seqs), [pos] * len(b_seqs), T5_LAYER, DEVICE
                )
                for k_b, i in enumerate(b_idx):
                    h_vt[i] = reps[k_b]
                    valid_mask[i] = True

    del t5_model
    torch.cuda.empty_cache()

    np.save(CACHE_WT,   h_wt)
    np.save(CACHE_VT,   h_vt)
    np.save(CACHE_MASK, valid_mask)
    # Save per-variant protein IDs and ΔΔG for the valid subset (used by probing notebook)
    protein_ids_valid = np.array(
        [vt_ids[i].rsplit(" ", 1)[0] for i in np.where(valid_mask)[0]], dtype=object)
    np.save(CACHE_DIR / "protein_ids_valid.npy", protein_ids_valid)
    np.save(CACHE_DIR / "ddg_valid.npy",         ms_ddg[valid_mask].astype(np.float32))
    print(f"  Saved.  valid={valid_mask.sum():,}/{N_ms:,} variants")

# Apply valid mask
h_wt      = h_wt[valid_mask]
h_vt      = h_vt[valid_mask]
ms_ddg_v  = ms_ddg[valid_mask]
print(f"After masking: {h_wt.shape[0]:,} variants")

# %% [markdown]
# ## 6. Load Collaborator SAE and Compute ΔZ

# %%
print(f"Loading SAE from {SAE_PATH.name} …")
sae = AutoEncoderTopK.from_pretrained(str(SAE_PATH), device=str(DEVICE))
sae.eval()
K_SAE    = int(sae.k.item())
DICT_SAE = sae.dict_size
print(f"  SAE: activation_dim={sae.activation_dim}  dict_size={DICT_SAE}  k={K_SAE}")

# %%
CACHE_DZ_POS = CACHE_DIR / "dz_pos.npy"
CACHE_DZ_NEG = CACHE_DIR / "dz_neg.npy"
N_valid = h_wt.shape[0]

if CACHE_DZ_POS.exists() and CACHE_DZ_NEG.exists():
    print("Loading cached ΔZ …")
    dz_pos = np.load(CACHE_DZ_POS)
    dz_neg = np.load(CACHE_DZ_NEG)
    print(f"  dz_pos={dz_pos.shape}  dz_neg={dz_neg.shape}")
else:
    print("Encoding through SAE (batched) …")
    ENCODE_BATCH = 512
    dz_pos = np.empty((N_valid, DICT_SAE), dtype=np.float32)
    dz_neg = np.empty((N_valid, DICT_SAE), dtype=np.float32)

    for start in tqdm(range(0, N_valid, ENCODE_BATCH), desc="SAE encode"):
        end = min(start + ENCODE_BATCH, N_valid)
        xw  = torch.tensor(h_wt[start:end], dtype=torch.float32, device=DEVICE)
        xv  = torch.tensor(h_vt[start:end], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            zw = sae.encode(xw).cpu().numpy()   # (batch, DICT_SAE) sparse TopK
            zv = sae.encode(xv).cpu().numpy()
        dz        = zv - zw
        dz_pos[start:end] = np.maximum(dz, 0)  # features gained
        dz_neg[start:end] = np.maximum(-dz, 0) # features lost

    np.save(CACHE_DZ_POS, dz_pos)
    np.save(CACHE_DZ_NEG, dz_neg)
    print("  ΔZ saved.")

del sae  # free GPU memory
torch.cuda.empty_cache()

# Quick sparsity check
nnz_pos = (dz_pos > 0).sum(1)
nnz_neg = (dz_neg > 0).sum(1)
print(f"ΔZ_pos sparsity: mean={nnz_pos.mean():.1f}  max={nnz_pos.max()}")
print(f"ΔZ_neg sparsity: mean={nnz_neg.mean():.1f}  max={nnz_neg.max()}")

# %% [markdown]
# ## 7. ΔΔG Bins

# %%
BINS = {
    "highly stabilizing":    ms_ddg_v < -1.0,
    "mildly stabilizing":   (ms_ddg_v >= -1.0) & (ms_ddg_v < -0.5),
    "near neutral":          np.abs(ms_ddg_v) < 0.5,
    "mildly destabilizing": (ms_ddg_v >= 0.5)  & (ms_ddg_v < 1.5),
    "highly destabilizing":  ms_ddg_v >= 1.5,
}
bin_names = list(BINS.keys())
bin_masks = list(BINS.values())
n_bins    = len(bin_names)
for name, mask in BINS.items():
    print(f"  {name:<25s}: {mask.sum():,}")

# %% [markdown]
# ## 8. Permutation Test (Sparse, GPU-accelerated)
#
# ΔZ_pos and ΔZ_neg are each at most k=256-sparse out of 16384 features.
# The sparse permutation test uses argpartition to extract the top-k entries per sample;
# zero-padded entries (from samples with <k non-zeros) are excluded by the (val>0) guard.

# %%
@torch.no_grad()
def perm_test_gpu_sparse(Z_arr, wt_mask, treat_mask, n_perm, k, device, eps=EPS,
                         batch=GPU_BATCH_SPARSE):
    """Sparse GPU permutation test.  Z_arr assumed to be (N, D) non-negative array."""
    Z_wt    = Z_arr[wt_mask]
    Z_treat = Z_arr[treat_mask]
    n_wt, n_treat, D_ = len(Z_wt), len(Z_treat), Z_arr.shape[1]
    N = n_wt + n_treat

    combined = np.concatenate([Z_wt, Z_treat])

    total_act  = torch.tensor(combined.sum(0),                         dtype=torch.float32, device=device)
    total_fire = torch.tensor((combined > 0).sum(0).astype(np.float32), device=device)

    # Top-k sparse representation (argpartition is O(D_) per row)
    fire_idx_np = np.argpartition(combined, D_ - k, axis=1)[:, D_ - k:]  # (N, k)
    fire_val_np = combined[np.arange(N)[:, None], fire_idx_np]             # (N, k)
    del combined

    fire_idx_t = torch.tensor(fire_idx_np, dtype=torch.long,    device=device)
    fire_val_t = torch.tensor(fire_val_np, dtype=torch.float32, device=device)
    del fire_idx_np, fire_val_np

    null_act  = np.empty((n_perm, D_), dtype=np.float32)
    null_fire = np.empty((n_perm, D_), dtype=np.float32)

    for start in range(0, n_perm, batch):
        bs = min(batch, n_perm - start)
        rand_vals = torch.rand(bs, N, device=device)
        treat_idx = rand_vals.topk(n_treat, dim=1, largest=False).indices
        del rand_vals

        t_fidx = fire_idx_t[treat_idx]
        t_fval = fire_val_t[treat_idx]
        del treat_idx

        flat_idx = t_fidx.view(bs, -1)
        flat_val = t_fval.view(bs, -1)
        del t_fidx, t_fval

        sum_treat_act  = torch.zeros(bs, D_, device=device)
        sum_treat_fire = torch.zeros(bs, D_, device=device)
        sum_treat_act.scatter_add_(1, flat_idx, flat_val)
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
    n_feats = null_act.shape[1]
    ba = 0.05 / n_feats
    p_act  = np.minimum(
        (null_act  >= obs_act [None, :]).mean(0),
        (null_act  <= obs_act [None, :]).mean(0)) * 2
    p_fire = np.minimum(
        (null_fire >= obs_fire[None, :]).mean(0),
        (null_fire <= obs_fire[None, :]).mean(0)) * 2
    lp, up = 100 * ba / 2, 100 * (1 - ba / 2)
    lo_act,  hi_act  = np.percentile(null_act,  [lp, up], axis=0)
    lo_fire, hi_fire = np.percentile(null_fire, [lp, up], axis=0)
    return dict(
        p_act=p_act, p_fire=p_fire,
        sig_enrich_act  =(obs_act  > hi_act),   sig_deplete_act  =(obs_act  < lo_act),
        sig_enrich_fire =(obs_fire > hi_fire),   sig_deplete_fire =(obs_fire < lo_fire),
    )

# %%
RESULTS_COLLAB = {}
USE_LOG2 = True
EPS_PLOT = 1e-6

def _tr(ratio):
    return np.log2(ratio + EPS_PLOT) if USE_LOG2 else ratio

ref_val      = 0 if USE_LOG2 else 1
ratio_ylabel = "log₂(fire_rate: treat / neutral)" if USE_LOG2 else "fire_rate: treat / neutral"
center_label = "log₂(ratio)" if USE_LOG2 else "ratio − 1"

for tag, Z_ms in [("ΔZ_pos (gained)", dz_pos), ("ΔZ_neg (lost)", dz_neg)]:
    print(f"\n=== {tag} ===", flush=True)

    Z_wt_m   = Z_ms[BINS["near neutral"]]
    Z_dest_m = Z_ms[BINS["highly destabilizing"]]
    Z_stab_m = Z_ms[BINS["highly stabilizing"]]

    obs_ratio_act_dest  = Z_dest_m.mean(0)       / (Z_wt_m.mean(0)       + EPS)
    obs_ratio_fire_dest = (Z_dest_m > 0).mean(0) / ((Z_wt_m > 0).mean(0) + EPS)
    null_act_dest, null_fire_dest = perm_test_gpu_sparse(
        Z_ms, BINS["near neutral"], BINS["highly destabilizing"],
        N_PERM, k=K_SAE, device=DEVICE)
    sig_dest = _sig_from_null(obs_ratio_act_dest, obs_ratio_fire_dest,
                               null_act_dest, null_fire_dest)
    print(f"  [dest] act  enriched={sig_dest['sig_enrich_act'].sum():4d}  "
          f"depleted={sig_dest['sig_deplete_act'].sum():4d}")
    print(f"  [dest] fire enriched={sig_dest['sig_enrich_fire'].sum():4d}  "
          f"depleted={sig_dest['sig_deplete_fire'].sum():4d}")

    obs_ratio_act_stab  = Z_stab_m.mean(0)       / (Z_wt_m.mean(0)       + EPS)
    obs_ratio_fire_stab = (Z_stab_m > 0).mean(0) / ((Z_wt_m > 0).mean(0) + EPS)
    null_act_stab, null_fire_stab = perm_test_gpu_sparse(
        Z_ms, BINS["near neutral"], BINS["highly stabilizing"],
        N_PERM, k=K_SAE, device=DEVICE)
    sig_stab = _sig_from_null(obs_ratio_act_stab, obs_ratio_fire_stab,
                               null_act_stab, null_fire_stab)
    print(f"  [stab] act  enriched={sig_stab['sig_enrich_act'].sum():4d}  "
          f"depleted={sig_stab['sig_deplete_act'].sum():4d}")
    print(f"  [stab] fire enriched={sig_stab['sig_enrich_fire'].sum():4d}  "
          f"depleted={sig_stab['sig_deplete_fire'].sum():4d}")

    RESULTS_COLLAB[tag] = dict(
        Z_ms=Z_ms, Z_wt=Z_wt_m, Z_dest=Z_dest_m, Z_stab=Z_stab_m,
        obs_ratio_act =obs_ratio_act_dest,  obs_ratio_fire =obs_ratio_fire_dest,
        null_ratio_act=null_act_dest,        null_ratio_fire=null_fire_dest,
        **{k: v for k, v in sig_dest.items()},
        obs_ratio_act_stab =obs_ratio_act_stab,  obs_ratio_fire_stab =obs_ratio_fire_stab,
        null_ratio_act_stab=null_act_stab,        null_ratio_fire_stab=null_fire_stab,
        **{f"{k}_stab": v for k, v in sig_stab.items()},
        mean_act  =np.stack([Z_ms[mask].mean(0)       for mask in bin_masks], axis=1),
        fire_rate =np.stack([(Z_ms[mask] > 0).mean(0) for mask in bin_masks], axis=1),
    )

# %% [markdown]
# ## 9. Spearman Correlation with ΔΔG

# %%
SPEARMAN_COLLAB = {}

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
        if tag and end % 2048 == 0:
            print(f"  {tag} spearman {end}/{D_}", end="\r", flush=True)

    z_stat = np.arctanh(np.clip(rho, -0.9999, 0.9999)) * np.sqrt(n - 3)
    p_vals = 2 * ndtr(-np.abs(z_stat))
    sig    = p_vals < (0.05 / D_)
    return rho, p_vals, sig

for tag, R in RESULTS_COLLAB.items():
    rho, p_vals, sig = spearman_vs_ddg(R["Z_ms"], ms_ddg_v, tag=tag)
    sig_destab = sig & (rho > 0)
    sig_stab   = sig & (rho < 0)
    max_rho    = np.abs(rho[sig]).max() if sig.any() else 0.0
    print(f"\n{tag}: sig_destab={sig_destab.sum()}  sig_stab={sig_stab.sum()}  max|ρ|={max_rho:.3f}")
    top10 = np.argsort(np.abs(rho))[::-1][:10]
    print(f"  top-10 |ρ|: {[(f'F{n}', f'ρ={rho[n]:.3f}') for n in top10]}")
    SPEARMAN_COLLAB[tag] = dict(rho=rho, p_vals=p_vals, sig=sig,
                                sig_destab=sig_destab, sig_stab=sig_stab)
    RESULTS_COLLAB[tag]["spearman_rho"]        = rho
    RESULTS_COLLAB[tag]["spearman_sig_destab"] = sig_destab
    RESULTS_COLLAB[tag]["spearman_sig_stab"]   = sig_stab

# %% [markdown]
# ## 10. Aggregate Metrics Summary

# %%
print(f"{'Model':<25} {'D':>6} {'De↑':>6} {'De↓':>6} {'St↑':>6} {'St↓':>6} "
      f"{'Joint':>6} {'maxEnr':>7} {'Sp_de':>7} {'Sp_st':>7}")
print("-" * 90)

for tag, R in RESULTS_COLLAB.items():
    D_ = len(R["obs_ratio_fire"])
    n_de = int(R["sig_enrich_fire"].sum())
    n_dd = int(R["sig_deplete_fire"].sum())
    n_se = int(R["sig_enrich_fire_stab"].sum())
    n_sd = int(R["sig_deplete_fire_stab"].sum())
    joint = int(((R["sig_enrich_fire"] & R["sig_deplete_fire_stab"]) |
                 (R["sig_deplete_fire"] & R["sig_enrich_fire_stab"])).sum())
    max_enr = float(R["obs_ratio_fire"][R["sig_enrich_fire"]].max()) \
              if R["sig_enrich_fire"].any() else 0.0
    sp = SPEARMAN_COLLAB.get(tag, {})
    sp_dest = int(sp.get("sig_destab", np.zeros(1, bool)).sum())
    sp_stab = int(sp.get("sig_stab",   np.zeros(1, bool)).sum())
    print(f"{tag:<25} {D_:>6d} {n_de:>6d} {n_dd:>6d} {n_se:>6d} {n_sd:>6d} "
          f"{joint:>6d} {max_enr:>7.2f} {sp_dest:>7d} {sp_stab:>7d}")

# %% [markdown]
# ## 11. Stability Plots (4 × 2 layout, same style as v2)

# %%
for model_name, R in RESULTS_COLLAB.items():
    top_idx_f = np.argsort(
        np.abs((_tr(R["obs_ratio_fire"]) - ref_val))
    )[::-1][:30]

    fig, axes = plt.subplots(4, 2, figsize=(16, 24))
    fig.suptitle(f"Collaborator SAE — {model_name}  (D={DICT_SAE}, k={K_SAE})",
                 fontsize=12, fontweight="bold")

    # row 0: firing-rate heatmap across ΔΔG bins (top 30 by dest enrichment)
    ax = axes[0, 0]
    im = ax.imshow(R["fire_rate"][top_idx_f], aspect="auto", cmap="magma")
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(bin_names, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(top_idx_f)))
    ax.set_yticklabels([f"F{n}" for n in top_idx_f], fontsize=6)
    ax.set_title("Firing rate (top 30 by |dest enrichment|)")
    plt.colorbar(im, ax=ax, fraction=0.03)

    ax = axes[0, 1]
    im = ax.imshow(R["mean_act"][top_idx_f], aspect="auto", cmap="viridis")
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(bin_names, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(top_idx_f)))
    ax.set_yticklabels([f"F{n}" for n in top_idx_f], fontsize=6)
    ax.set_title("Mean activation (top 30)")
    plt.colorbar(im, ax=ax, fraction=0.03)

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
        eligible   = np.where(fire_treat >= 0.05)[0]
        top20      = eligible[np.argsort(np.abs(ratio_vals[eligible] - ref_val))[::-1][:20]]

        ax = axes[row, 0]
        colors = ["tab:red"  if sig_enrich[n] else
                  "tab:blue" if sig_deplete[n] else "lightgray" for n in top20]
        ax.barh(range(len(top20)), ratio_vals[top20[::-1]], color=colors[::-1])
        ax.axvline(ref_val, color="k", lw=0.8)
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels([f"F{n}" for n in top20[::-1]], fontsize=7)
        ax.set_xlabel(ratio_ylabel, fontsize=9)
        ax.set_title(f"Firing ratio: {label} vs neutral\n"
                     f"(≥5% treat firing; red=enriched  blue=depleted  Bonferroni)")

        ax = axes[row, 1]
        eligible_mask = fire_treat >= 0.05
        for cat, base_mask, color, size, zo in [
                ("enriched",  sig_enrich,                "tab:red",   40, 4),
                ("depleted",  sig_deplete,               "tab:blue",  40, 4),
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

    # row 3: combo (destab - stab) analysis
    dest_c = _tr(R["obs_ratio_fire"])      - ref_val
    stab_c = _tr(R["obs_ratio_fire_stab"]) - ref_val
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
    ax.plot([-lim, lim], [-lim, lim], color="gray", lw=0.6, linestyle=":")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"destabilizing {center_label}", fontsize=9)
    ax.set_ylabel(f"stabilizing {center_label}", fontsize=9)
    ax.set_title("Destabilizing vs stabilizing enrichment (anti-diagonal = directional)")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[3, 1]
    top_combo    = np.argsort(np.abs(combo))[::-1][:20]
    sorted_order = top_combo[np.argsort(combo[top_combo])]
    y_pos  = np.arange(len(sorted_order))
    bar_h  = 0.38
    dest_bar_c = ["tab:red"  if (R["sig_enrich_fire"][n] or R["sig_deplete_fire"][n])
                  else "#ffbbbb" for n in sorted_order]
    stab_bar_c = ["tab:blue" if (R["sig_enrich_fire_stab"][n] or R["sig_deplete_fire_stab"][n])
                  else "#bbbbff" for n in sorted_order]
    ax.barh(y_pos + bar_h / 2, dest_c[sorted_order], height=bar_h,
            color=dest_bar_c, label="destabilizing")
    ax.barh(y_pos - bar_h / 2, stab_c[sorted_order], height=bar_h,
            color=stab_bar_c, label="stabilizing", alpha=0.85)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"F{n}" for n in sorted_order], fontsize=7)
    ax.set_xlabel(center_label, fontsize=9)
    ax.set_title("Top 20 features by |combo| score\n"
                 "(bright=Bonferroni sig; sorted: stab-specific → dest-specific)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    safe_tag = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(OUT_DIR / f"collab_sae_ddg_{safe_tag}.png", dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 12. Spearman ρ Histogram and Scatter

# %%
fig, axes = plt.subplots(len(RESULTS_COLLAB), 2, figsize=(14, 4 * len(RESULTS_COLLAB)))
if len(RESULTS_COLLAB) == 1:
    axes = axes[np.newaxis, :]

for i, (tag, R) in enumerate(RESULTS_COLLAB.items()):
    S        = SPEARMAN_COLLAB[tag]
    rho      = S["rho"]
    sig      = S["sig"]
    fire_dest = R["fire_rate"][:, bin_names.index("highly destabilizing")]

    ax = axes[i, 0]
    ax.hist(rho[~sig],             bins=80, color="lightgray", label="n.s.")
    ax.hist(rho[S["sig_destab"]], bins=40, color="tab:red",  alpha=0.8, label="destab (sig)")
    ax.hist(rho[S["sig_stab"]],   bins=40, color="tab:blue", alpha=0.8, label="stab (sig)")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Spearman ρ (SAE feature activation vs ΔΔG)")
    ax.set_ylabel("# features")
    ax.set_title(f"{tag}  destab={S['sig_destab'].sum()}  stab={S['sig_stab'].sum()}")
    ax.legend(fontsize=7)

    ax = axes[i, 1]
    for mask, color, label in [(~sig,            "lightgray", "n.s."),
                                (S["sig_destab"], "tab:red",  "destab (sig)"),
                                (S["sig_stab"],   "tab:blue", "stab (sig)")]:
        ax.scatter(fire_dest[mask], np.abs(rho[mask]),
                   c=color, s=8, alpha=0.6, label=label, rasterized=True)
    ax.set_xlabel("Firing rate (highly destabilizing)")
    ax.set_ylabel("|Spearman ρ|")
    ax.set_title(f"{tag} — |ρ| vs firing rate")
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(OUT_DIR / "collab_sae_spearman.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 13. Top Candidate Features

# %%
print("\n=== Top-10 dest-specific features (sig_enrich_dest & sig_deplete_stab) ===\n")
for tag, R in RESULTS_COLLAB.items():
    anti_dest = R["sig_enrich_fire"] & R["sig_deplete_fire_stab"]
    if not anti_dest.any():
        print(f"{tag}: no dest-specific features\n")
        continue
    rho = SPEARMAN_COLLAB.get(tag, {}).get("rho", np.zeros(len(R["obs_ratio_fire"])))
    enr = R["obs_ratio_fire"]
    scores = (enr * np.abs(rho))[anti_dest]
    cand_i = np.where(anti_dest)[0][np.argsort(scores)[::-1][:10]]
    print(f"{tag} ({anti_dest.sum()} dest-specific features):")
    print(f"  {'Feature':<10} {'fire_ratio':>10} {'Spearman_ρ':>12} {'score':>10}")
    for n in cand_i:
        print(f"  F{n:<9} {enr[n]:>10.3f} {rho[n]:>12.4f} {enr[n]*abs(rho[n]):>10.4f}")
    print()

# %% [markdown]
# ## 14. Neuron Candidate Scatter (enrichment vs Spearman ρ)

# %%
n_models = len(RESULTS_COLLAB)
fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6), constrained_layout=True)
if n_models == 1:
    axes = [axes]

for ax, (tag, R) in zip(axes, RESULTS_COLLAB.items()):
    enr       = _tr(R["obs_ratio_fire"])
    rho       = SPEARMAN_COLLAB.get(tag, {}).get("rho", np.zeros(len(enr)))
    fire_dest = (R["Z_dest"] > 0).mean(0)   # absolute firing rate in dest bin
    eligible  = fire_dest >= 0.05           # same threshold as Section 11 bar/scatter plots

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
    ax.set_title(f"{tag}\nD={DICT_SAE}, k={K_SAE}  (≥5% dest firing)\n"
                 f"dest-spec={anti_dest.sum()}  stab-spec={anti_stab.sum()}", fontsize=10)
    ax.set_xlabel("log₂ fire ratio (destabilizing/neutral)", fontsize=9)
    ax.set_ylabel("Spearman ρ (activation vs ΔΔG)", fontsize=9)
    ax.legend(fontsize=7)

plt.suptitle("Collaborator SAE: feature candidate landscape", fontsize=12, fontweight="bold")
plt.savefig(OUT_DIR / "collab_sae_candidates.png", dpi=150, bbox_inches="tight")
plt.show()
