# %% [markdown]
# # Activity Assay SAE Analysis
#
# Applies two pre-trained sparse autoencoders to functional activity assay variants
# (from DMS datasets), analogous to the MegaScale stability analysis.
#
# **Models:**
# - **D5** (`TopKSAE`, 2048-dim in, dict=8192, k=128): trained on ClinVar WT+VT
#   concat ProtT5 *final-layer* residue embeddings.
# - **Collab SAE** (`AutoEncoderTopK`, 1024-dim in, dict=16384, k=256): pre-trained
#   on UniRef ProtT5 layer-20 residue representations. Applied as ΔZ = z_vt − z_wt.
#
# **Activity bins** (scores averaged across treatments per variant):
#   - LoF  : score < 0.75
#   - wt-like: 0.80 ≤ score ≤ 1.20  (gaps 0.75–0.80 and 1.20–1.25 excluded)
#   - GoF  : score > 1.25

# %%
import os, sys, re, pickle, warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import requests
from io import StringIO
from pathlib import Path
from tqdm import tqdm
from scipy.stats import rankdata
from scipy.special import ndtr
import matplotlib.pyplot as plt

try:
    from Bio import SeqIO
except ImportError:
    raise ImportError("biopython required: conda install -c conda-forge biopython")

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

ACT_CSV   = Path("/data/ross/assay_calibration/labelseq_dataframe_processed.csv.gz")
D5_PATH   = Path("/home/rcstewart/ppi_lossgain/sparse_bottleneck/v2_model_d5_topk.pt")
SAE_PATH  = Path("/data/karna/model_weights/sae_weights/t5/trainer_0/t5_layer20_topk256_ef16.pt")
T5_NAME   = "Rostlab/prot_t5_xl_half_uniref50-enc"
T5_LAYER  = 20          # collab SAE trained on layer 20
# ProtT5-XL has 24 encoder blocks (0-indexed 0–23).
# T5Stack emits hidden_states[0..n_layers]; hidden_states[L+1] = output of block L.
# Verified at runtime in Section 6.
T5_FINAL  = 23          # 0-based index of final encoder block

CACHE_DIR = Path("/data/ross/interp/activity_sae_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR   = Path("/home/rcstewart/ppi_lossgain/sparse_bottleneck")
EXTRACT_BATCH = 16

# Activity bins — gaps 0.75–0.80 and 1.20–1.25 are unclassified and excluded
LOF_MAX = 0.75
WT_MIN  = 0.80
WT_MAX  = 1.20
GOF_MIN = 1.25

# D5 hyperparameters (must match training in v2.py)
EF_TOPK = 4
K_TOPK  = 128

_var_re = re.compile(r'^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$')

USE_LOG2  = True
EPS_PLOT  = 1e-6

np.random.seed(SEED)
torch.manual_seed(SEED)
print(f"device: {DEVICE}")

# %% [markdown]
# ## 2. D5 TopKSAE Architecture
#
# Copied from `clinvar_sparse_bottleneck_v2.py` — do not import that file directly
# as it would execute top-level training cells.

# %%
class TopKSAE(nn.Module):
    """Over-complete SAE with hard TopK sparsity — exact copy of v2.py class."""
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

# %% [markdown]
# ## 3. Load and Filter Activity CSV
#
# Filter to `assay == 'activity'` and missense variants only.
# Multiple rows per variant (across treatments/replicates) are averaged into one score.

# %%
AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}

def parse_variant(v):
    """Return (ref1, pos_int, alt1) from HGVS p. notation, or (None,None,None)."""
    m = _var_re.match(v)
    if m is None:
        return None, None, None
    ref3, pos, alt3 = m.groups()
    return AA3_TO_1.get(ref3), int(pos), AA3_TO_1.get(alt3)

def assign_bin(s):
    if s < LOF_MAX:
        return "LoF"
    if WT_MIN <= s <= WT_MAX:
        return "wt_like"
    if s > GOF_MIN:
        return "GoF"
    return None  # gap / unclassified

print("Loading activity CSV …")
df = pd.read_csv(ACT_CSV, compression="gzip")
print(f"  Total rows: {len(df):,}  |  assay types: {df['assay'].unique().tolist()}")

# Keep activity assay only
df_act = df[df["assay"] == "activity"].copy()
print(f"  Activity rows: {len(df_act):,}")

# Keep missense: variant matches p.AaaXXXBbb (excludes synonymous p.AaaXXX=, frameshifts, etc.)
df_act = df_act[df_act["variant"].str.match(
    r'^p\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2}$', na=False)].copy()
print(f"  Missense rows: {len(df_act):,}")

# Parse variant into 1-letter fields
parsed = [parse_variant(v) for v in df_act["variant"]]
df_act["aa_ref"] = [p[0] for p in parsed]
df_act["aa_pos"] = [p[1] for p in parsed]
df_act["aa_alt"] = [p[2] for p in parsed]
df_act = df_act.dropna(subset=["aa_ref", "aa_pos", "aa_alt"]).copy()
df_act["aa_pos"] = df_act["aa_pos"].astype(int)
print(f"  After parse+dropna: {len(df_act):,}")

# Deduplicate: average score across treatments for each unique variant per protein
# "average score" column already averages replicates within a dataset
df_var = (
    df_act
    .groupby(["uniprot_accession", "Gene", "aa_ref", "aa_pos", "aa_alt"])["average score"]
    .mean()
    .reset_index()
)
df_var.rename(columns={"average score": "score"}, inplace=True)
print(f"  Unique (accession × variant): {len(df_var):,}")

# Assign bins
df_var["bin"] = df_var["score"].map(assign_bin)
n_unclassified = df_var["bin"].isna().sum()
df_var = df_var[df_var["bin"].notna()].copy()
print(f"  Excluded (gap/unclassified): {n_unclassified:,}")
print(f"  Binned variants: {len(df_var):,}")
print(df_var["bin"].value_counts().to_string())

# %% [markdown]
# ## 4. Fetch UniProt Sequences (Cached)
#
# The CSV already contains `uniprot_accession` — use the UniProt REST API
# to retrieve the canonical sequence for each accession.

# %%
FASTA_CACHE = CACHE_DIR / "uniprot_sequences.fasta"

def fetch_uniprot_fasta(accessions):
    seqs = {}
    for acc in tqdm(accessions, desc="UniProt fetch"):
        url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        record = next(SeqIO.parse(StringIO(r.text), "fasta"))
        seqs[acc] = str(record.seq)
    return seqs

if FASTA_CACHE.exists():
    acc_to_seq = {}
    for rec in SeqIO.parse(str(FASTA_CACHE), "fasta"):
        # UniProt FASTA header: >sp|P10398|ARAF_HUMAN ...
        parts = rec.id.split("|")
        acc = parts[1] if len(parts) >= 2 else rec.id
        acc_to_seq[acc] = str(rec.seq)
    print(f"Loaded {len(acc_to_seq)} sequences from {FASTA_CACHE}")
else:
    unique_accs = df_var["uniprot_accession"].unique().tolist()
    print(f"Fetching {len(unique_accs)} UniProt sequences …")
    acc_to_seq = fetch_uniprot_fasta(unique_accs)
    with open(FASTA_CACHE, "w") as fh:
        for acc, seq in acc_to_seq.items():
            fh.write(f">{acc}\n{seq}\n")
    print(f"Saved to {FASTA_CACHE}")

# %% [markdown]
# ## 5. Verify Sequences and Build Variant Arrays
#
# Drop variants where the expected WT residue at the stated position does not match
# the canonical UniProt sequence (isoform mismatches, signal-peptide numbering, etc.).

# %%
valid_rows = []
skipped_no_seq = 0
skipped_mismatch = 0

for _, row in df_var.iterrows():
    acc = row["uniprot_accession"]
    seq = acc_to_seq.get(acc)
    if seq is None:
        skipped_no_seq += 1
        continue
    pos1 = int(row["aa_pos"])   # 1-based
    if pos1 < 1 or pos1 > len(seq):
        skipped_mismatch += 1
        continue
    if seq[pos1 - 1] != row["aa_ref"]:
        skipped_mismatch += 1
        continue
    valid_rows.append(row)

df_valid = pd.DataFrame(valid_rows).reset_index(drop=True)
print(f"Skipped (no seq): {skipped_no_seq}  |  mismatched WT: {skipped_mismatch}")
print(f"Variants passing verification: {len(df_valid):,}")
print(df_valid["bin"].value_counts().to_string())

# Build parallel arrays used throughout
uniprot_accs = df_valid["uniprot_accession"].tolist()
mut_pos0     = (df_valid["aa_pos"].astype(int) - 1).tolist()   # 0-based
aa_ref_list  = df_valid["aa_ref"].tolist()
aa_alt_list  = df_valid["aa_alt"].tolist()
scores       = df_valid["score"].to_numpy(dtype=np.float32)
bins         = df_valid["bin"].tolist()
N_act        = len(df_valid)

mask_gof = np.array([b == "GoF"     for b in bins])
mask_lof = np.array([b == "LoF"     for b in bins])
mask_wt  = np.array([b == "wt_like" for b in bins])
print(f"\nGoF={mask_gof.sum()}  LoF={mask_lof.sum()}  wt-like={mask_wt.sum()}")

# %% [markdown]
# ## 6. ProtT5 Loader and Multi-Layer Extraction
#
# `get_residue_embeddings_multilayer` performs a single forward pass and returns
# the hidden state at each requested block index for each sequence.
# HF T5Stack indexing: hidden_states[L+1] = output of encoder block L (0-based).
# T5_LAYER=20 → hidden_states[21]; T5_FINAL=23 → hidden_states[24].

# %%
def load_prott5(device):
    print(f"Loading ProtT5 ({T5_NAME}) …")
    tokenizer = T5Tokenizer.from_pretrained(T5_NAME, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(T5_NAME).to(device)
    model.eval()
    print("  ProtT5 loaded.")
    return model, tokenizer


@torch.no_grad()
def get_residue_embeddings_multilayer(model, tokenizer, seqs, positions, layers, device):
    """
    Batch ProtT5 forward pass; returns {layer_idx: np.ndarray(N, 1024)}.
    `layers`: list of 0-based block indices.
    `positions`: list of 0-based residue positions (one per sequence in `seqs`).
    """
    t5_inputs = [" ".join(list(s)) for s in seqs]
    inputs = tokenizer(t5_inputs, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs, output_hidden_states=True)
    result = {L: np.empty((len(seqs), 1024), dtype=np.float32) for L in layers}
    for L in layers:
        hidden = outputs.hidden_states[L + 1]   # (batch, max_len, 1024)
        for b, pos in enumerate(positions):
            result[L][b] = hidden[b, pos, :].detach().cpu().float().numpy()
    return result, len(outputs.hidden_states)


@torch.no_grad()
def get_all_positions_single_seq(model, tokenizer, seq, layers, device):
    """
    ONE ProtT5 forward pass for a single sequence; returns all residue positions.
    Returns: {layer_idx: np.ndarray(len(seq), 1024)}, n_hidden_states
    This is used for WT sequences so we pay the cost once per protein, not once
    per unique mutation position.
    """
    t5_input = " ".join(list(seq))
    inputs = tokenizer([t5_input], return_tensors="pt", padding=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs, output_hidden_states=True)
    result = {}
    for L in layers:
        # hidden_states[L+1]: (1, seq_len, 1024) — no padding since batch size = 1
        result[L] = outputs.hidden_states[L + 1][0, :len(seq), :].detach().cpu().float().numpy()
    return result, len(outputs.hidden_states)

# %% [markdown]
# ## 7. Extract ProtT5 Embeddings (Both Layers, Cached)
#
# For each variant we need:
# - Layer-20 WT and VT representations (for collab SAE ΔZ)
# - Final-layer WT and VT representations (for D5 concat input)
#
# WT: ONE forward pass per protein (all positions extracted from that pass).
# VT: batched by position in groups of EXTRACT_BATCH (same-length sequences).

# %%
CACHE_L20_WT  = CACHE_DIR / "layer20_wt.npy"
CACHE_L20_VT  = CACHE_DIR / "layer20_vt.npy"
CACHE_LFN_WT  = CACHE_DIR / "final_layer_wt.npy"
CACHE_LFN_VT  = CACHE_DIR / "final_layer_vt.npy"
CACHE_VALID   = CACHE_DIR / "valid_idx.npy"   # indices in df_valid that were extracted
LAYERS = [T5_LAYER, T5_FINAL]

all_cached = all(p.exists() for p in [CACHE_L20_WT, CACHE_L20_VT, CACHE_LFN_WT, CACHE_LFN_VT,
                                       CACHE_VALID])

if all_cached:
    print("Loading cached ProtT5 representations …")
    valid_idx = np.load(CACHE_VALID)
    h_l20_wt  = np.load(CACHE_L20_WT)
    h_l20_vt  = np.load(CACHE_L20_VT)
    h_lfn_wt  = np.load(CACHE_LFN_WT)
    h_lfn_vt  = np.load(CACHE_LFN_VT)
    # Apply the same valid_idx filter to labels/scores
    scores   = scores[valid_idx]
    mask_gof = mask_gof[valid_idx]
    mask_lof = mask_lof[valid_idx]
    mask_wt  = mask_wt[valid_idx]
    bins     = [bins[i] for i in valid_idx]
    N_act    = h_l20_wt.shape[0]
    _pid_cache = CACHE_DIR / "protein_ids.npy"
    if _pid_cache.exists():
        act_protein_ids = np.load(_pid_cache, allow_pickle=True)
    else:
        act_protein_ids = np.array([uniprot_accs[i] for i in valid_idx], dtype=object)
    print(f"  l20_wt={h_l20_wt.shape}  lfn_wt={h_lfn_wt.shape}")
    print(f"  GoF={mask_gof.sum()}  LoF={mask_lof.sum()}  wt={mask_wt.sum()}")

else:
    print("Extracting ProtT5 representations (first run — will cache) …")
    t5_model, tokenizer = load_prott5(DEVICE)

    h_l20_wt  = np.zeros((N_act, 1024), dtype=np.float32)
    h_l20_vt  = np.zeros((N_act, 1024), dtype=np.float32)
    h_lfn_wt  = np.zeros((N_act, 1024), dtype=np.float32)
    h_lfn_vt  = np.zeros((N_act, 1024), dtype=np.float32)
    valid_mask = np.zeros(N_act, dtype=bool)
    n_hidden_verified = None

    # Group variant indices by UniProt accession
    acc_to_indices: dict[str, list[int]] = {}
    for i, acc in enumerate(uniprot_accs):
        acc_to_indices.setdefault(acc, []).append(i)

    for acc, indices in tqdm(acc_to_indices.items(), desc="proteins"):
        wt_seq = acc_to_seq[acc]

        # ── WT: ONE forward pass for the whole protein sequence ───────────────
        # Extracts all residue positions at once; no need to repeat the sequence
        # per unique mutation position.
        wt_reps, n_hidden = get_all_positions_single_seq(
            t5_model, tokenizer, wt_seq, LAYERS, DEVICE
        )
        if n_hidden_verified is None:
            n_hidden_verified = n_hidden
            actual_final = n_hidden - 2   # hidden_states[n-1] = output of final block
            print(f"\n  [verify] len(hidden_states)={n_hidden} → T5_FINAL={actual_final} "
                  f"(configured T5_FINAL={T5_FINAL})")
            assert actual_final == T5_FINAL, (
                f"T5_FINAL mismatch: expected {T5_FINAL}, got {actual_final}. "
                f"Update T5_FINAL in config.")
        # wt_reps[L]: (len(wt_seq), 1024) — index directly by position
        for i in indices:
            pos = mut_pos0[i]
            h_l20_wt[i] = wt_reps[T5_LAYER][pos]
            h_lfn_wt[i] = wt_reps[T5_FINAL][pos]

        # ── VT: batch by position (same-length seqs, no padding mismatch) ─────
        pos_to_var_indices: dict[int, list[int]] = {}
        for i in indices:
            pos_to_var_indices.setdefault(mut_pos0[i], []).append(i)

        for pos, var_indices in pos_to_var_indices.items():
            if pos >= len(wt_seq):
                continue
            vt_seqs_i = [wt_seq[:pos] + aa_alt_list[i] + wt_seq[pos + 1:]
                         for i in var_indices]

            for batch_start in range(0, len(var_indices), EXTRACT_BATCH):
                b_seqs = vt_seqs_i[batch_start:batch_start + EXTRACT_BATCH]
                b_idx  = var_indices[batch_start:batch_start + EXTRACT_BATCH]
                reps, _ = get_residue_embeddings_multilayer(
                    t5_model, tokenizer,
                    b_seqs, [pos] * len(b_seqs), LAYERS, DEVICE
                )
                for k_b, i in enumerate(b_idx):
                    h_l20_vt[i] = reps[T5_LAYER][k_b]
                    h_lfn_vt[i] = reps[T5_FINAL][k_b]
                    valid_mask[i] = True

    del t5_model
    torch.cuda.empty_cache()

    # Restrict to successfully extracted variants before saving (so cache = filtered arrays)
    valid_idx  = np.where(valid_mask)[0]
    h_l20_wt   = h_l20_wt[valid_idx]
    h_l20_vt   = h_l20_vt[valid_idx]
    h_lfn_wt   = h_lfn_wt[valid_idx]
    h_lfn_vt   = h_lfn_vt[valid_idx]
    scores     = scores[valid_idx]
    mask_gof   = mask_gof[valid_idx]
    mask_lof   = mask_lof[valid_idx]
    mask_wt    = mask_wt[valid_idx]
    bins       = [b for b, v in zip(bins, valid_mask) if v]
    N_act      = h_l20_wt.shape[0]

    np.save(CACHE_L20_WT, h_l20_wt)
    np.save(CACHE_L20_VT, h_l20_vt)
    np.save(CACHE_LFN_WT, h_lfn_wt)
    np.save(CACHE_LFN_VT, h_lfn_vt)
    np.save(CACHE_VALID,  valid_idx)
    # Save per-variant protein IDs for LOPO probing
    act_protein_ids = np.array([uniprot_accs[i] for i in valid_idx], dtype=object)
    np.save(CACHE_DIR / "protein_ids.npy", act_protein_ids)
    print(f"\nSaved.  valid={N_act:,}/{len(valid_mask):,} variants")

print(f"Final: N={N_act:,}  GoF={mask_gof.sum()}  LoF={mask_lof.sum()}  wt={mask_wt.sum()}")

# %% [markdown]
# ## 8. Load SAE Models and Encode
#
# - **D5**: input = concat(WT_final, VT_final) → 2048-dim → TopKSAE → Z_d5 (N, 8192)
# - **Collab SAE**: WT_l20, VT_l20 → SAE encode each → ΔZ_pos/ΔZ_neg (N, 16384)

# %%
# ── D5 ────────────────────────────────────────────────────────────────────────
CACHE_ZD5 = CACHE_DIR / "z_d5.npy"

model_d5 = TopKSAE(in_dim=2048).to(DEVICE)
model_d5.load_state_dict(torch.load(D5_PATH, map_location=DEVICE))
model_d5.eval()
print(f"D5 loaded: dict={model_d5.d}  k={model_d5.k}")

if CACHE_ZD5.exists():
    print("Loading cached D5 encodings …")
    Z_d5 = np.load(CACHE_ZD5)
else:
    print("Encoding through D5 …")
    X_d5 = np.concatenate([h_lfn_wt, h_lfn_vt], axis=1)   # (N, 2048)
    ENCODE_BATCH = 512
    Z_d5 = np.empty((N_act, model_d5.d), dtype=np.float32)
    for start in tqdm(range(0, N_act, ENCODE_BATCH), desc="D5 encode"):
        end = min(start + ENCODE_BATCH, N_act)
        xb  = torch.tensor(X_d5[start:end], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            z, _ = model_d5(xb)
        Z_d5[start:end] = z.cpu().numpy()
    np.save(CACHE_ZD5, Z_d5)
    print("  Saved.")

nnz_d5 = (Z_d5 > 0).sum(1)
print(f"D5 sparsity: mean={nnz_d5.mean():.1f}  (expected ≈{model_d5.k})")

del model_d5
torch.cuda.empty_cache()

# ── Collab SAE ────────────────────────────────────────────────────────────────
CACHE_DZ_POS = CACHE_DIR / "dz_pos.npy"
CACHE_DZ_NEG = CACHE_DIR / "dz_neg.npy"

print(f"\nLoading Collab SAE from {SAE_PATH.name} …")
sae = AutoEncoderTopK.from_pretrained(str(SAE_PATH), device=str(DEVICE))
sae.eval()
K_SAE    = int(sae.k.item())
DICT_SAE = sae.dict_size
print(f"  activation_dim={sae.activation_dim}  dict_size={DICT_SAE}  k={K_SAE}")

if CACHE_DZ_POS.exists() and CACHE_DZ_NEG.exists():
    print("Loading cached ΔZ …")
    dz_pos = np.load(CACHE_DZ_POS)
    dz_neg = np.load(CACHE_DZ_NEG)
else:
    print("Encoding through Collab SAE …")
    ENCODE_BATCH = 512
    dz_pos = np.empty((N_act, DICT_SAE), dtype=np.float32)
    dz_neg = np.empty((N_act, DICT_SAE), dtype=np.float32)
    for start in tqdm(range(0, N_act, ENCODE_BATCH), desc="Collab encode"):
        end = min(start + ENCODE_BATCH, N_act)
        xw  = torch.tensor(h_l20_wt[start:end], dtype=torch.float32, device=DEVICE)
        xv  = torch.tensor(h_l20_vt[start:end], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            zw = sae.encode(xw).cpu().numpy()
            zv = sae.encode(xv).cpu().numpy()
        dz             = zv - zw
        dz_pos[start:end] = np.maximum(dz,  0)
        dz_neg[start:end] = np.maximum(-dz, 0)
    np.save(CACHE_DZ_POS, dz_pos)
    np.save(CACHE_DZ_NEG, dz_neg)
    print("  Saved.")

del sae
torch.cuda.empty_cache()

nnz_pos = (dz_pos > 0).sum(1)
nnz_neg = (dz_neg > 0).sum(1)
print(f"ΔZ_pos sparsity: mean={nnz_pos.mean():.1f}  (expected ≈{K_SAE})")
print(f"ΔZ_neg sparsity: mean={nnz_neg.mean():.1f}  (expected ≈{K_SAE})")

# %% [markdown]
# ## 9. Permutation Test (Sparse, GPU-accelerated)
#
# Two comparisons per model: GoF vs wt-like and LoF vs wt-like.
# Reuses the same sparse permutation test as the stability notebooks.

# %%
@torch.no_grad()
def perm_test_gpu_sparse(Z_arr, wt_mask, treat_mask, n_perm, k, device, eps=EPS,
                         batch=GPU_BATCH_SPARSE):
    """Sparse GPU permutation test. Z_arr: (N, D) non-negative float32 array."""
    Z_wt    = Z_arr[wt_mask]
    Z_treat = Z_arr[treat_mask]
    n_wt, n_treat, D_ = len(Z_wt), len(Z_treat), Z_arr.shape[1]
    N = n_wt + n_treat

    combined = np.concatenate([Z_wt, Z_treat])

    total_act  = torch.tensor(combined.sum(0),                          dtype=torch.float32, device=device)
    total_fire = torch.tensor((combined > 0).sum(0).astype(np.float32), device=device)

    fire_idx_np = np.argpartition(combined, D_ - k, axis=1)[:, D_ - k:]
    fire_val_np = combined[np.arange(N)[:, None], fire_idx_np]
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
        sum_treat_act.scatter_add_ (1, flat_idx, flat_val)
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
    p_act  = np.minimum((null_act  >= obs_act [None, :]).mean(0),
                        (null_act  <= obs_act [None, :]).mean(0)) * 2
    p_fire = np.minimum((null_fire >= obs_fire[None, :]).mean(0),
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
def _tr(ratio):
    return np.log2(ratio + EPS_PLOT) if USE_LOG2 else ratio

ref_val      = 0 if USE_LOG2 else 1
ratio_ylabel = "log₂(fire_rate: treat / wt-like)" if USE_LOG2 else "fire_rate ratio"
center_label = "log₂(ratio)"   if USE_LOG2 else "ratio − 1"

# RESULTS structure: {model_tag: {comparison_tag: result_dict}}
RESULTS = {}

for model_tag, Z_ms, k_model in [
    ("D5",           Z_d5,   K_TOPK),
    ("Collab_ΔZpos", dz_pos, K_SAE),
    ("Collab_ΔZneg", dz_neg, K_SAE),
]:
    RESULTS[model_tag] = {}
    print(f"\n=== {model_tag} ===", flush=True)

    for comp_tag, treat_mask in [("GoF_vs_wt", mask_gof), ("LoF_vs_wt", mask_lof)]:
        Z_wt_m    = Z_ms[mask_wt]
        Z_treat_m = Z_ms[treat_mask]
        n_treat   = treat_mask.sum()

        obs_ratio_act  = Z_treat_m.mean(0)       / (Z_wt_m.mean(0)       + EPS)
        obs_ratio_fire = (Z_treat_m > 0).mean(0) / ((Z_wt_m > 0).mean(0) + EPS)

        null_act, null_fire = perm_test_gpu_sparse(
            Z_ms, mask_wt, treat_mask, N_PERM, k=k_model, device=DEVICE)
        sig = _sig_from_null(obs_ratio_act, obs_ratio_fire, null_act, null_fire)

        print(f"  [{comp_tag}] fire enriched={sig['sig_enrich_fire'].sum():5d}  "
              f"depleted={sig['sig_deplete_fire'].sum():5d}")

        RESULTS[model_tag][comp_tag] = dict(
            Z_ms=Z_ms, Z_wt=Z_wt_m, Z_treat=Z_treat_m, n_treat=n_treat,
            obs_ratio_act=obs_ratio_act, obs_ratio_fire=obs_ratio_fire,
            null_ratio_act=null_act, null_ratio_fire=null_fire,
            fire_treat=(Z_treat_m > 0).mean(0),
            **sig,
        )

# %% [markdown]
# ## 10. Spearman Correlation with Activity Score

# %%
SPEARMAN = {}

def spearman_vs_score(Z_ms_m, act_scores, tag=""):
    n, D_ = Z_ms_m.shape
    r_score = rankdata(act_scores).astype(np.float32)
    r_score -= r_score.mean()
    r_score /= np.linalg.norm(r_score)

    rho   = np.empty(D_, dtype=np.float32)
    chunk = 256
    for start in range(0, D_, chunk):
        end = min(start + chunk, D_)
        rZ  = np.apply_along_axis(rankdata, 0, Z_ms_m[:, start:end]).astype(np.float32)
        rZ -= rZ.mean(0)
        norms = np.linalg.norm(rZ, axis=0)
        norms[norms == 0] = 1.0
        rZ /= norms
        rho[start:end] = rZ.T @ r_score
        if tag and end % 2048 == 0:
            print(f"  {tag} {end}/{D_}", end="\r", flush=True)

    z_stat = np.arctanh(np.clip(rho, -0.9999, 0.9999)) * np.sqrt(n - 3)
    p_vals = 2 * ndtr(-np.abs(z_stat))
    sig    = p_vals < (0.05 / D_)
    return rho, p_vals, sig

for model_tag, Z_ms, _ in [
    ("D5",           Z_d5,   None),
    ("Collab_ΔZpos", dz_pos, None),
    ("Collab_ΔZneg", dz_neg, None),
]:
    rho, p_vals, sig = spearman_vs_score(Z_ms, scores, tag=model_tag)
    sig_gof = sig & (rho > 0)   # higher activation → higher activity (GoF-like)
    sig_lof = sig & (rho < 0)   # higher activation → lower  activity (LoF-like)
    max_rho = float(np.abs(rho[sig]).max()) if sig.any() else 0.0
    print(f"{model_tag}: sig_GoF={sig_gof.sum()}  sig_LoF={sig_lof.sum()}  max|ρ|={max_rho:.3f}")
    SPEARMAN[model_tag] = dict(rho=rho, p_vals=p_vals, sig=sig,
                               sig_gof=sig_gof, sig_lof=sig_lof)
    # Attach to RESULTS for plots
    for comp_tag in RESULTS.get(model_tag, {}):
        RESULTS[model_tag][comp_tag]["spearman_rho"]     = rho
        RESULTS[model_tag][comp_tag]["spearman_sig_gof"] = sig_gof
        RESULTS[model_tag][comp_tag]["spearman_sig_lof"] = sig_lof

# %% [markdown]
# ## 11. Summary Table

# %%
print(f"\n{'Model':<18} {'Comp':<12} {'D':>6} {'Enr↑':>6} {'Enr↓':>6} "
      f"{'Sp_GoF':>8} {'Sp_LoF':>8} {'maxEnrFire':>12}")
print("-" * 85)

for model_tag, comps in RESULTS.items():
    for comp_tag, R in comps.items():
        D_   = len(R["obs_ratio_fire"])
        n_up = int(R["sig_enrich_fire"].sum())
        n_dn = int(R["sig_deplete_fire"].sum())
        sp   = SPEARMAN.get(model_tag, {})
        sp_g = int(sp.get("sig_gof", np.zeros(1, bool)).sum())
        sp_l = int(sp.get("sig_lof", np.zeros(1, bool)).sum())
        max_enr = float(R["obs_ratio_fire"][R["sig_enrich_fire"]].max()) \
                  if R["sig_enrich_fire"].any() else 0.0
        print(f"{model_tag:<18} {comp_tag:<12} {D_:>6d} {n_up:>6d} {n_dn:>6d} "
              f"{sp_g:>8d} {sp_l:>8d} {max_enr:>12.3f}")

# %% [markdown]
# ## 12. Enrichment Bar and Scatter Plots

# %%
for model_tag, comps in RESULTS.items():
    n_comps = len(comps)
    fig, axes = plt.subplots(n_comps, 2, figsize=(14, 6 * n_comps), squeeze=False)
    fig.suptitle(f"{model_tag} — enrichment vs wt-like (Bonferroni, ≥5% firing)",
                 fontsize=12, fontweight="bold")

    for row, (comp_tag, R) in enumerate(comps.items()):
        treat_label = "GoF" if "GoF" in comp_tag else "LoF"
        ratio_vals  = _tr(R["obs_ratio_fire"])
        fire_treat  = R["fire_treat"]
        eligible    = fire_treat >= 0.05
        eligible_i  = np.where(eligible)[0]
        top20       = eligible_i[np.argsort(np.abs(ratio_vals[eligible_i] - ref_val))[::-1][:20]]

        sig_enrich  = R["sig_enrich_fire"]
        sig_deplete = R["sig_deplete_fire"]

        # bar plot
        ax = axes[row, 0]
        colors = ["tab:red"  if sig_enrich[n]  else
                  "tab:blue" if sig_deplete[n] else "lightgray" for n in top20]
        ax.barh(range(len(top20)), ratio_vals[top20[::-1]], color=colors[::-1])
        ax.axvline(ref_val, color="k", lw=0.8)
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels([f"F{n}" for n in top20[::-1]], fontsize=7)
        ax.set_xlabel(ratio_ylabel, fontsize=9)
        ax.set_title(f"Firing ratio: {treat_label} vs wt-like (≥5% {treat_label} firing)\n"
                     f"red=enriched  blue=depleted  Bonferroni")

        # scatter plot
        ax = axes[row, 1]
        for cat, base_mask, color, size, zo in [
                ("n.s.",     ~(sig_enrich | sig_deplete), "lightgray", 10, 2),
                ("enriched",  sig_enrich,                 "tab:red",   40, 4),
                ("depleted",  sig_deplete,                "tab:blue",  40, 4),
        ]:
            mask = base_mask & eligible
            ax.scatter(fire_treat[mask], ratio_vals[mask],
                       c=color, s=size, alpha=0.8, label=cat, zorder=zo)
        ax.axhline(ref_val, color="k", lw=0.8, linestyle="--")
        ax.set_xlabel(f"Firing rate ({treat_label}), ≥5% only", fontsize=9)
        ax.set_ylabel(ratio_ylabel, fontsize=9)
        ax.set_title(f"Enrichment scatter: {treat_label} vs wt-like (Bonferroni)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    safe_tag = model_tag.replace(" ", "_")
    plt.savefig(OUT_DIR / f"act_{safe_tag}_enrichment.png", dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 13. Spearman ρ Histograms

# %%
n_models = len(SPEARMAN)
fig, axes = plt.subplots(n_models, 2, figsize=(14, 4 * n_models), squeeze=False)

for i, (model_tag, S) in enumerate(SPEARMAN.items()):
    rho     = S["rho"]
    sig     = S["sig"]
    sig_gof = S["sig_gof"]
    sig_lof = S["sig_lof"]

    # histogram of ρ
    ax = axes[i, 0]
    ax.hist(rho[~sig],   bins=80, color="lightgray", label="n.s.")
    ax.hist(rho[sig_gof],bins=40, color="tab:red",  alpha=0.8, label="GoF-like (sig)")
    ax.hist(rho[sig_lof],bins=40, color="tab:blue", alpha=0.8, label="LoF-like (sig)")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Spearman ρ (feature activation vs activity score)")
    ax.set_ylabel("# features")
    ax.set_title(f"{model_tag}  GoF-like={sig_gof.sum()}  LoF-like={sig_lof.sum()}")
    ax.legend(fontsize=7)

    # |ρ| vs firing rate in GoF bin
    fire_gof = (RESULTS[model_tag]["GoF_vs_wt"]["Z_treat"] > 0).mean(0) \
               if "GoF_vs_wt" in RESULTS[model_tag] else np.zeros(len(rho))
    ax = axes[i, 1]
    for mask, color, label in [(~sig,   "lightgray", "n.s."),
                                (sig_gof,"tab:red",  "GoF-like (sig)"),
                                (sig_lof,"tab:blue", "LoF-like (sig)")]:
        ax.scatter(fire_gof[mask], np.abs(rho[mask]),
                   c=color, s=8, alpha=0.6, label=label, rasterized=True)
    ax.set_xlabel("Firing rate (GoF bin)")
    ax.set_ylabel("|Spearman ρ|")
    ax.set_title(f"{model_tag} — |ρ| vs GoF firing rate")
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(OUT_DIR / "act_spearman.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 14. Candidate Feature Scatter (log₂ fire ratio vs Spearman ρ)
#
# For each model, one panel per comparison (GoF vs wt, LoF vs wt).
# Firing rate filter ≥5% applied (same threshold as bar plots above).

# %%
for model_tag, comps in RESULTS.items():
    n_comps = len(comps)
    fig, axes = plt.subplots(1, n_comps, figsize=(7 * n_comps, 6), constrained_layout=True)
    if n_comps == 1:
        axes = [axes]

    S = SPEARMAN.get(model_tag, {})
    rho = S.get("rho", np.zeros(1))

    for ax, (comp_tag, R) in zip(axes, comps.items()):
        treat_label = "GoF" if "GoF" in comp_tag else "LoF"
        enr         = _tr(R["obs_ratio_fire"])
        fire_treat  = R["fire_treat"]
        eligible    = fire_treat >= 0.05

        sig_enrich  = R["sig_enrich_fire"]
        sig_deplete = R["sig_deplete_fire"]
        sig_gof_sp  = R.get("spearman_sig_gof", np.zeros(len(enr), bool))
        sig_lof_sp  = R.get("spearman_sig_lof", np.zeros(len(enr), bool))

        # Highlight features with both enrichment and Spearman support
        joint_enrich  = sig_enrich  & sig_gof_sp
        joint_deplete = sig_deplete & sig_lof_sp
        other_sig     = (sig_enrich | sig_deplete) & ~joint_enrich & ~joint_deplete
        neither       = ~(sig_enrich | sig_deplete)

        for mask, color, label, size, zo in [
                (neither   & eligible, "lightgray", "n.s.",                  5,  1),
                (other_sig & eligible, "#ddaaaa",  "sig (enrich only)",     15,  2),
                (joint_deplete & eligible, "tab:blue", "depleted+LoF-like", 40,  4),
                (joint_enrich  & eligible, "tab:red",  "enriched+GoF-like", 40,  4),
        ]:
            ax.scatter(enr[mask], rho[mask], c=color, s=size, alpha=0.8,
                       label=f"{label} (n={mask.sum()})", zorder=zo, rasterized=True)

        ax.axhline(0, color="k", lw=0.8, linestyle="--")
        ax.axvline(0, color="k", lw=0.8, linestyle="--")
        D_ = len(enr)
        k_ = K_SAE if "Collab" in model_tag else K_TOPK
        ax.set_title(f"{model_tag} — {treat_label} vs wt-like\n"
                     f"D={D_}, k={k_}  (≥5% {treat_label} firing)", fontsize=10)
        ax.set_xlabel(f"log₂ fire ratio ({treat_label}/wt-like)", fontsize=9)
        ax.set_ylabel("Spearman ρ (activation vs activity score)", fontsize=9)
        ax.legend(fontsize=7)

    plt.suptitle(f"{model_tag}: activity candidate landscape", fontsize=12, fontweight="bold")
    plt.savefig(OUT_DIR / f"act_{model_tag.replace(' ','_')}_candidates.png",
                dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 15. Top Candidate Features

# %%
print("\n=== Top-10 enriched+GoF-correlated features per model/comparison ===\n")
for model_tag, comps in RESULTS.items():
    S   = SPEARMAN.get(model_tag, {})
    rho = S.get("rho", np.zeros(1))
    for comp_tag, R in comps.items():
        treat_label  = "GoF" if "GoF" in comp_tag else "LoF"
        sig_enrich   = R["sig_enrich_fire"]
        sig_gof_sp   = R.get("spearman_sig_gof", np.zeros(len(sig_enrich), bool))
        joint        = sig_enrich & sig_gof_sp
        if not joint.any():
            print(f"{model_tag} {comp_tag}: no jointly enriched+correlated features\n")
            continue
        enr     = R["obs_ratio_fire"]
        cand_sc = (enr * np.abs(rho))[joint]
        cand_i  = np.where(joint)[0][np.argsort(cand_sc)[::-1][:10]]
        print(f"{model_tag} {comp_tag}  ({joint.sum()} candidates):")
        print(f"  {'Feature':<10} {'fire_ratio':>10} {'Spearman_ρ':>12} {'score':>10}")
        for n in cand_i:
            print(f"  F{n:<9} {enr[n]:>10.3f} {rho[n]:>12.4f} "
                  f"{enr[n]*abs(rho[n]):>10.4f}")
        print()
