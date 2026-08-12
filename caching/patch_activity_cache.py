"""
Patch script: reconstruct protein_ids.npy in the activity SAE cache.

The activity notebook was run before we added the protein_ids save.
The cache already has valid_idx.npy and uniprot_sequences.fasta, so we
can reconstruct df_valid from the CSV + FASTA without re-running ProtT5.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from Bio import SeqIO

CACHE_DIR = Path("/data/ross/interp/activity_sae_cache")
ACT_CSV   = Path("/data/ross/assay_calibration/labelseq_dataframe_processed.csv.gz")
FASTA     = CACHE_DIR / "uniprot_sequences.fasta"

out_path  = CACHE_DIR / "protein_ids.npy"
if out_path.exists():
    print(f"{out_path} already exists — nothing to do.")
    raise SystemExit(0)

# ── Reconstruct df_var (same logic as activity notebook) ─────────────────────
AA3 = {"Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E",
       "Gly":"G","His":"H","Ile":"I","Leu":"L","Lys":"K","Met":"M","Phe":"F",
       "Pro":"P","Ser":"S","Thr":"T","Trp":"W","Tyr":"Y","Val":"V"}
_varre = re.compile(r'^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$')

def parse_variant(v):
    m = _varre.match(v)
    if m is None: return None, None, None
    r3, p, a3 = m.groups()
    return AA3.get(r3), int(p), AA3.get(a3)

def assign_bin(s):
    if s < 0.75:              return "LoF"
    if 0.80 <= s <= 1.20:     return "wt_like"
    if s > 1.25:              return "GoF"
    return None

print("Loading activity CSV …")
df = pd.read_csv(ACT_CSV, compression="gzip")
df_act = df[df["assay"] == "activity"].copy()
df_act = df_act[df_act["variant"].str.match(
    r'^p\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2}$', na=False)]

parsed = [parse_variant(v) for v in df_act["variant"]]
df_act["aa_ref"] = [p[0] for p in parsed]
df_act["aa_pos"] = [p[1] for p in parsed]
df_act["aa_alt"] = [p[2] for p in parsed]
df_act = df_act.dropna(subset=["aa_ref", "aa_pos", "aa_alt"])

df_var = (df_act
          .groupby(["uniprot_accession", "Gene", "aa_ref", "aa_pos", "aa_alt"])["average score"]
          .mean().reset_index())
df_var.rename(columns={"average score": "score"}, inplace=True)
df_var["bin"] = df_var["score"].map(assign_bin)
df_var = df_var[df_var["bin"].notna()].copy().reset_index(drop=True)
print(f"  df_var: {len(df_var):,} binned variants")

# ── Load cached FASTA sequences ───────────────────────────────────────────────
print(f"Loading sequences from {FASTA} …")
acc_to_seq = {}
for rec in SeqIO.parse(str(FASTA), "fasta"):
    parts = rec.id.split("|")
    acc = parts[1] if len(parts) >= 2 else rec.id
    acc_to_seq[acc] = str(rec.seq)
print(f"  {len(acc_to_seq)} sequences loaded")

# ── Sequence-verify variants (same filter as notebook) ───────────────────────
valid_rows = []
for _, row in df_var.iterrows():
    acc = row["uniprot_accession"]
    seq = acc_to_seq.get(acc)
    if seq is None:
        continue
    pos1 = int(row["aa_pos"])
    if pos1 < 1 or pos1 > len(seq):
        continue
    if seq[pos1 - 1] != row["aa_ref"]:
        continue
    valid_rows.append(row)

df_valid = pd.DataFrame(valid_rows).reset_index(drop=True)
print(f"  df_valid: {len(df_valid):,} sequence-verified variants")

uniprot_accs = df_valid["uniprot_accession"].tolist()

# ── Apply valid_idx ───────────────────────────────────────────────────────────
valid_idx = np.load(CACHE_DIR / "valid_idx.npy")
print(f"  valid_idx length: {len(valid_idx)}  max={valid_idx.max()}")

if valid_idx.max() >= len(uniprot_accs):
    raise ValueError(
        f"valid_idx max ({valid_idx.max()}) >= len(df_valid) ({len(uniprot_accs)}). "
        "CSV reconstruction doesn't match original run — check filters.")

act_protein_ids = np.array([uniprot_accs[i] for i in valid_idx], dtype=object)
np.save(out_path, act_protein_ids)
print(f"Saved {out_path}  shape={act_protein_ids.shape}")
print(f"  Unique proteins: {len(np.unique(act_protein_ids))}")
print(f"  Sample: {act_protein_ids[:5]}")
