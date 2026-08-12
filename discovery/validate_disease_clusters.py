"""
Validate disease clusters via:
  Option 1: UniProt gene names + Enrichr pathway enrichment
  Option 2: ClinVar variant_summary condition text
"""
import sys, os, time, json, gzip, re, warnings
import numpy as np
import scipy.sparse as sp
import pandas as pd
import requests
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
import h5py

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
CV_H5       = Path("/data/ross/ppi_lossgain/interaction_loss/clinvar/prott5_subgraphs.h5")
HGMD_H5     = Path("/data/ross/ppi_lossgain/interaction_loss/hgmd/prott5_embeddings.h5")
LABEL_DIR   = Path("/data/ross/ppi_lossgain/interaction_loss/home/data_interaction_loss")
LAT_DIR     = Path("/data/ross/interp/latent_analysis")
OUT_DIR     = LAT_DIR

CLUSTERS_CSV = LAT_DIR / "disease_clusters.csv"
Z_CV_NPZ     = LAT_DIR / "z_cv_concat_ef1_k128.npz"
Z_HG_NPZ     = LAT_DIR / "z_hg_concat_ef1_k128.npz"
CV_LABELS    = Path("/home/rcstewart/ppi_lossgain/sparse_bottleneck/clinvar_labels.npy")

VARSUMMARY_GZ  = LAT_DIR / "variant_summary.txt.gz"
VARSUMMARY_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"

FOCUS_CLUSTERS = [0, 4, 8, 12, 14, 16, 27, 31, 32, 33, 35, 46]

# ── replicate exact H5 filter from clinvar_sparse_bottleneck_v2.py ───────────
_var_re = re.compile(r'^([A-Z])(\d+)([A-Z])$')

def load_label_set(tsv_path):
    s = set()
    with open(tsv_path) as fh:
        for line in fh:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                s.add((parts[0], parts[1]))
    return s

# ─────────────────────────────────────────────────────────────────────────────
# 1. Reconstruct per-variant cluster assignments
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("Step 1: Reconstructing variant keys and cluster assignments")
print("=" * 70)

labels    = np.load(CV_LABELS)
path_mask = (labels == 1)
print(f"  ClinVar pathogenic: {path_mask.sum():,}  benign: {(~path_mask).sum():,}")

print("  Loading sparse Z matrices ...")
Z_cv = sp.load_npz(Z_CV_NPZ)
Z_hg = sp.load_npz(Z_HG_NPZ)

Z_path = sp.vstack([Z_cv[path_mask], Z_hg])
print(f"  Disease set: {Z_path.shape[0]:,}")

print("  Normalising and re-running k-means (same seed -> reproducible) ...")
Z_path_norm = normalize(Z_path, norm="l2")
km = MiniBatchKMeans(n_clusters=50, batch_size=8192, max_iter=300,
                     n_init=5, random_state=42)
cluster_ids = km.fit_predict(Z_path_norm)
counts = np.unique(cluster_ids, return_counts=True)[1]
print(f"  Cluster counts verified: {counts[:5]} ...")

# ── read ClinVar H5 replicating the exact filter from preprocessing ──────────
print("  Loading label sets ...")
pathogenic_set = load_label_set(LABEL_DIR / "clinvar_pathogenic_dirbind_variants.tsv")
benign_set     = load_label_set(LABEL_DIR / "clinvar_benign_dirbind_variants.tsv")
conflicts      = pathogenic_set & benign_set
print(f"  Pathogenic: {len(pathogenic_set):,}  Benign: {len(benign_set):,}  "
      f"Conflicts: {len(conflicts):,}")

print("  Reading ClinVar H5 (applying same filter as preprocessing) ...")
cv_complex_ids = []
cv_variant_1b  = []   # 1-based variant string (for ClinVar cross-reference)
with h5py.File(CV_H5, "r") as f:
    for complex_id in f.keys():
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
            if key not in pathogenic_set and key not in benign_set:
                continue
            cv_complex_ids.append(complex_id)
            cv_variant_1b.append(var_1b)

cv_complex_ids = np.array(cv_complex_ids)
cv_variant_1b  = np.array(cv_variant_1b)
assert len(cv_complex_ids) == len(labels), (
    f"H5 rows {len(cv_complex_ids)} != labels {len(labels)}")
print(f"  Reconstructed {len(cv_complex_ids):,} variant keys OK")

# Subset to pathogenic
cv_path_complex = cv_complex_ids[path_mask]
cv_path_var1b   = cv_variant_1b[path_mask]

# ── HGMD variant keys ─────────────────────────────────────────────────────────
print("  Reading HGMD H5 variant keys ...")
hgmd_prot_ids = []
hgmd_variants = []
with h5py.File(HGMD_H5, "r") as f:
    vt_keys = [k for k in f.keys() if " " in k]
    for vt_key in vt_keys:
        prot_id, var_str = vt_key.split(" ", 1)
        if prot_id in f:
            hgmd_prot_ids.append(prot_id)
            hgmd_variants.append(var_str)

hgmd_prot_ids = np.array(hgmd_prot_ids)
hgmd_variants = np.array(hgmd_variants)
assert len(hgmd_prot_ids) == Z_hg.shape[0], (
    f"HGMD rows {len(hgmd_prot_ids)} != z_hg {Z_hg.shape[0]}")

n_cv_path = path_mask.sum()
cv_clust  = cluster_ids[:n_cv_path]
hg_clust  = cluster_ids[n_cv_path:]
print(f"  CV pathogenic: {len(cv_clust):,}  HGMD: {len(hg_clust):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Build per-cluster protein + variant tables
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Step 2: Building per-cluster protein lists")
print("=" * 70)

def extract_prots(complex_ids):
    protA = np.array([cid.split('_')[0] for cid in complex_ids])
    protB = np.array([cid.split('_')[1] if '_' in cid else cid.split('_')[0]
                      for cid in complex_ids])
    return protA, protB

cv_protA, cv_protB = extract_prots(cv_path_complex)

cluster_data = {}
for c in range(50):
    cv_m = (cv_clust == c)
    hg_m = (hg_clust == c)
    all_prots = (set(cv_protA[cv_m]) | set(cv_protB[cv_m]) |
                 set(hgmd_prot_ids[hg_m]))
    cluster_data[c] = {
        "n_cv": int(cv_m.sum()),
        "n_hg": int(hg_m.sum()),
        "prots": all_prots,
        # gene-symbol keys for Enrichr/ClinVar lookup
        "cv_interactors": list(cv_protA[cv_m]),    # protA = interactor in label set
        "cv_vars":        list(cv_path_var1b[cv_m]),
        "hg_prots":       list(hgmd_prot_ids[hg_m]),
        "hg_vars":        list(hgmd_variants[hg_m]),
    }

for c in FOCUS_CLUSTERS[:5]:
    d = cluster_data[c]
    print(f"  Cluster {c:2d}: n_cv={d['n_cv']:5d}  n_hg={d['n_hg']:4d}  "
          f"n_prots={len(d['prots']):4d}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. UniProt API: gene names for all unique proteins in focus clusters
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Step 3: Fetching gene names from UniProt")
print("=" * 70)

all_prots = sorted({p for c in FOCUS_CLUSTERS
                    for p in cluster_data[c]["prots"]})
print(f"  Querying {len(all_prots)} unique UniProt IDs ...")

gene_name_cache = {}
BATCH = 200
for i in range(0, len(all_prots), BATCH):
    batch = all_prots[i:i+BATCH]
    ids_str = ",".join(batch)
    url = (f"https://rest.uniprot.org/uniprotkb/accessions"
           f"?accessions={ids_str}&fields=accession,gene_names&format=tsv")
    try:
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            for line in r.text.strip().split("\n")[1:]:
                parts = line.split("\t")
                if len(parts) >= 2:
                    acc  = parts[0].strip()
                    gene = parts[1].strip().split()[0] if parts[1].strip() else acc
                    gene_name_cache[acc] = gene
        else:
            print(f"    WARNING: UniProt {r.status_code} for batch {i//BATCH}")
    except Exception as e:
        print(f"    WARNING: UniProt query failed: {e}")
    time.sleep(0.3)

print(f"  Retrieved {len(gene_name_cache)} / {len(all_prots)} gene names")

def prots_to_genes(prot_set):
    return sorted({gene_name_cache.get(p, p) for p in prot_set})

# ─────────────────────────────────────────────────────────────────────────────
# 4. Enrichr pathway enrichment for protein-diverse clusters
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Step 4: Enrichr pathway enrichment")
print("=" * 70)

ENRICHR_BASE = "https://maayanlab.cloud/Enrichr"
GENE_SETS    = ["KEGG_2021_Human", "GO_Biological_Process_2023", "Reactome_2022"]
LARGE_THRESH = 30

def enrichr_query(gene_list, description="query"):
    if len(gene_list) < 3:
        return {}
    genes_str = "\n".join(gene_list)
    try:
        r = requests.post(
            f"{ENRICHR_BASE}/addList",
            files={"list": (None, genes_str), "description": (None, description)},
            timeout=30)
        if r.status_code != 200:
            return {}
        user_list_id = r.json()["userListId"]
        time.sleep(0.5)
    except Exception as e:
        print(f"    Enrichr addList failed: {e}")
        return {}

    results = {}
    for gs in GENE_SETS:
        try:
            r2 = requests.get(
                f"{ENRICHR_BASE}/enrich?userListId={user_list_id}&backgroundType={gs}",
                timeout=30)
            if r2.status_code != 200:
                continue
            data = r2.json().get(gs, [])
            # each entry: [rank, term, pval, zscore, combined_score, genes, adj_pval, ...]
            top = [(d[1], float(d[2]), float(d[6]), d[5])
                   for d in data[:10] if d[6] < 0.05]
            results[gs] = top
        except Exception as e:
            print(f"    Enrichr enrich failed ({gs}): {e}")
        time.sleep(0.3)
    return results

enrichr_results = {}
for c in FOCUS_CLUSTERS:
    n_prots = len(cluster_data[c]["prots"])
    genes   = prots_to_genes(cluster_data[c]["prots"])
    if n_prots >= LARGE_THRESH:
        print(f"  Cluster {c}: {n_prots} prots -> Enrichr ({len(genes)} genes) ...")
        res = enrichr_query(genes, description=f"cluster_{c}")
        enrichr_results[c] = res
        for gs in GENE_SETS:
            terms = res.get(gs, [])
            if terms:
                print(f"    [{gs.split('_')[0]}] {terms[0][0][:65]}  (adj p={terms[0][2]:.2g})")
    else:
        print(f"  Cluster {c}: {n_prots} prots (small) -> listing genes only")
        enrichr_results[c] = {}

# ─────────────────────────────────────────────────────────────────────────────
# 5. Download ClinVar variant_summary.txt.gz
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Step 5: ClinVar variant_summary download")
print("=" * 70)

if not VARSUMMARY_GZ.exists():
    print(f"  Downloading {VARSUMMARY_URL} ...")
    try:
        r = requests.get(VARSUMMARY_URL, stream=True, timeout=600)
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(VARSUMMARY_GZ, "wb") as fout:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fout.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100 * downloaded / total
                    print(f"\r  {pct:.0f}%  ({downloaded/1e6:.0f} / {total/1e6:.0f} MB)",
                          end="", flush=True)
        print(f"\n  Saved to {VARSUMMARY_GZ}")
    except Exception as e:
        print(f"  Download failed: {e}")
        VARSUMMARY_GZ = None
else:
    print(f"  Cached: {VARSUMMARY_GZ}  ({VARSUMMARY_GZ.stat().st_size/1e6:.0f} MB)")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Parse variant_summary and build gene -> condition lookup
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Step 6: Parsing ClinVar variant_summary")
print("=" * 70)

gene_to_phenos = {}
vs_ok = False

if VARSUMMARY_GZ and Path(str(VARSUMMARY_GZ)).exists():
    print("  Parsing (pathogenic GRCh38 only) ...")
    keep_cols = ["GeneSymbol", "ClinicalSignificance", "PhenotypeList",
                 "Assembly", "ProteinChange"]
    chunks = []
    try:
        with gzip.open(VARSUMMARY_GZ, "rt", encoding="latin-1") as f:
            for chunk in pd.read_csv(f, sep="\t", usecols=lambda c: c in keep_cols,
                                     chunksize=100_000, low_memory=False):
                sig_mask = chunk["ClinicalSignificance"].str.contains(
                    r"[Pp]athogenic", na=False)
                ass_mask = chunk.get("Assembly",
                                     pd.Series(["GRCh38"]*len(chunk))) == "GRCh38"
                sub = chunk[sig_mask & ass_mask]
                if len(sub):
                    chunks.append(sub[["GeneSymbol", "PhenotypeList",
                                       "ProteinChange"]].copy())
    except Exception as e:
        print(f"  Parse error: {e}")

    if chunks:
        vs_df = pd.concat(chunks, ignore_index=True)
        print(f"  {len(vs_df):,} pathogenic variants parsed")
        vs_df["gene_upper"] = vs_df["GeneSymbol"].str.upper().str.strip()
        gene_to_phenos = (vs_df.groupby("gene_upper")["PhenotypeList"]
                          .apply(lambda x: list(x.dropna())).to_dict())
        vs_ok = True
else:
    print("  Skipping (no variant_summary file)")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Per-cluster condition enrichment
# ─────────────────────────────────────────────────────────────────────────────
from collections import Counter

condition_results = {}

if vs_ok:
    print("\n" + "=" * 70)
    print("Step 7: Condition enrichment per cluster")
    print("=" * 70)

    for c in FOCUS_CLUSTERS:
        genes        = prots_to_genes(cluster_data[c]["prots"])
        genes_upper  = [g.upper() for g in genes]
        all_phenos   = []
        for g in genes_upper:
            all_phenos.extend(gene_to_phenos.get(g, []))

        terms = Counter()
        for pheno_str in all_phenos:
            for term in str(pheno_str).split("|"):
                term = term.strip()
                if term and term.lower() not in (
                        "not provided", "not specified", ".", "na", "nan"):
                    terms[term] += 1

        top_terms = terms.most_common(10)
        condition_results[c] = top_terms

        print(f"\n  Cluster {c:2d} (n_prots={len(cluster_data[c]['prots'])}):")
        for term, cnt in top_terms[:5]:
            print(f"    [{cnt:4d}] {term[:80]}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Write comprehensive report
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Step 8: Writing cluster validation report")
print("=" * 70)

df_clusters = pd.read_csv(CLUSTERS_CSV)
report_rows = []
for c in FOCUS_CLUSTERS:
    d    = cluster_data[c]
    row  = df_clusters[df_clusters["cluster"] == c]
    cont = float(row["contamination"].iloc[0]) if len(row) else np.nan
    enr  = float(row["top_enr_mean"].iloc[0]) if len(row) else np.nan
    genes = prots_to_genes(d["prots"])

    enr_hit = ""
    for gs in GENE_SETS:
        hits = enrichr_results.get(c, {}).get(gs, [])
        if hits:
            enr_hit = f"{hits[0][0][:60]} (adj_p={hits[0][2]:.2g})"
            break

    top_cond = ""
    if c in condition_results and condition_results[c]:
        top_cond = "; ".join([t for t, _ in condition_results[c][:3]])

    report_rows.append({
        "cluster":              c,
        "n_disease":            d["n_cv"] + d["n_hg"],
        "contamination":        round(cont, 3),
        "top_enr_mean":         round(enr, 3),
        "n_unique_prots":       len(d["prots"]),
        "n_genes":              len(genes),
        "genes":                ", ".join(genes[:30]),
        "top_enrichr_pathway":  enr_hit,
        "top_clinvar_conditions": top_cond,
    })

report_df = pd.DataFrame(report_rows)
report_path = OUT_DIR / "cluster_validation_report.csv"
report_df.to_csv(report_path, index=False)
print(f"  Saved: {report_path}")

# ── print summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CLUSTER VALIDATION SUMMARY")
print("=" * 70)
for _, row in report_df.iterrows():
    c = int(row["cluster"])
    print(f"\nCluster {c:2d}  n={row['n_disease']:5d}  contam={row['contamination']:.3f}"
          f"  n_prots={row['n_unique_prots']:4d}")
    print(f"  Genes: {str(row['genes'])[:110]}")
    if row["top_enrichr_pathway"]:
        print(f"  Pathway: {row['top_enrichr_pathway']}")
    if row["top_clinvar_conditions"]:
        print(f"  Conditions: {str(row['top_clinvar_conditions'])[:120]}")

# ── save Enrichr JSON ─────────────────────────────────────────────────────────
enr_out = {}
for c, res in enrichr_results.items():
    enr_out[str(c)] = {
        gs: [(t, p, a, list(g)) for t, p, a, g in terms]
        for gs, terms in res.items()
    }
with open(OUT_DIR / "cluster_enrichr_results.json", "w") as f:
    json.dump(enr_out, f, indent=2)

# ── save condition counts ─────────────────────────────────────────────────────
cond_rows = [{"cluster": c, "condition": t, "n": n}
             for c, terms in condition_results.items()
             for t, n in terms]
if cond_rows:
    pd.DataFrame(cond_rows).to_csv(OUT_DIR / "cluster_condition_counts.csv", index=False)

print(f"\nAll outputs in {OUT_DIR}")
print("Done.")
