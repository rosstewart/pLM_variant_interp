"""
within_family_analysis.py

Validates whether cluster-defining SAE latents are:
  (A) within-family discriminative: distinguish pathogenic from benign variants
      OF THE SAME PROTEIN, not just "this protein vs. other proteins"
  (B) within-family mechanism-specific: for proteins appearing in multiple clusters
      (e.g. TP53 in clusters 4 and 8), do each cluster's latents specifically fire
      on THAT cluster's TP53 variants and not the other cluster's?

Benign comparison uses ClinVar Benign variants (labels==0, same Z_cv matrix)
as the primary within-family benign group — clinically reviewed benign variants
for the exact same proteins. gnomAD is used as a secondary comparison where
available (COL1A1 only in the PPI-complex gnomAD H5).

Outputs → /data/ross/interp/latent_analysis/validation/within_family/
"""
import sys, re, time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared_infrastructure import (
    LA, LA_LEGACY, DEFAULT_NAME, N_CLUSTERS, RANDOM_SEED,
    load_decoder, load_clinvar_data, load_hgmd_gnomad,
    reconstruct_clinvar_variant_keys, run_disease_kmeans,
)

OUT = LA / "validation" / "within_family"
OUT.mkdir(parents=True, exist_ok=True)

GNOMAD_H5 = Path("/data/ross/ppi_lossgain/interaction_loss/gnomad/prott5_subgraphs.h5")
_VAR_RE   = re.compile(r'^([A-Z])(\d+)([A-Z])$')

# Clusters and their dominant proteins (from LOOG results)
CLUSTER_DOMINANT = {
    0:  "P60484",   # PTEN
    4:  "P04637",   # TP53
    8:  "P04637",   # TP53
    12: "P04637",   # TP53
    14: "P60709",   # ACTB
    16: "P01130",   # LDLR
    27: "P02452",   # COL1A1
    31: "P60484",   # PTEN
    32: "P04637",   # TP53
    33: "P60484",   # PTEN
    35: "P38398",   # BRCA1
    46: "P60709",   # ACTB
}

# TP53 clusters for cross-cluster within-family mechanism test
TP53_CLUSTERS = [4, 8, 12, 32]
PTEN_CLUSTERS = [0, 31, 33]

# Top latents per cluster (from Section 2, highest specificity_ratio)
CLUSTER_TOP_LATS = {
    0:  [1138, 1420, 897],
    4:  [97, 2001, 1253],
    8:  [1494, 1994, 414],
    12: [1279, 2003, 1559],
    14: [1915, 1817, 507],
    16: [60, 338, 221],
    27: [1799, 615, 1263],
    31: [1887, 1757, 1873],
    32: [1279, 1150, 2003],
    33: [1630, 1374, 871],
    35: [1314, 1592, 683],
    46: [518, 943, 1728],
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Reconstruct gnomAD protein IDs (replicate _load_subgraph_h5 traversal)
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_gnomad_prot_ids(cache_path: Path = None):
    """Return protein ID for each row of Z_gn, in same order as Z_gn was encoded.

    Replicates the deduplication logic from _load_subgraph_h5 in
    unsupervised_latent_analysis.py.
    """
    if cache_path and cache_path.exists():
        return np.load(cache_path, allow_pickle=True)

    print("  Traversing gnomAD H5 to reconstruct protein IDs …", flush=True)
    prot_ids = []
    seen = set()
    with h5py.File(str(GNOMAD_H5), "r") as f:
        for complex_key in f.keys():
            prot_id = complex_key.split("_")[0]
            cgrp = f[complex_key]
            for var_key in cgrp.keys():
                uid = (prot_id, var_key)
                if uid in seen:
                    continue
                seen.add(uid)
                # Check that this entry has the required attrs (same filter as encoding)
                try:
                    vgrp = cgrp[var_key]
                    _ = vgrp["node_emb"]
                    _ = vgrp["mut_diff"]
                    _ = vgrp.attrs["mut_local_idx"]
                    prot_ids.append(prot_id)
                except Exception:
                    pass

    prot_ids = np.array(prot_ids)
    if cache_path:
        np.save(cache_path, prot_ids)
        print(f"  Cached → {cache_path}", flush=True)
    return prot_ids


# ─────────────────────────────────────────────────────────────────────────────
# 2. Within-family fire rate breakdown
# ─────────────────────────────────────────────────────────────────────────────

def fire_rate(Z_dense, row_mask, lat):
    """Fraction of rows (selected by row_mask) where latent lat > 0."""
    if row_mask.sum() == 0:
        return 0.0, 0
    vals = Z_dense[row_mask, lat]
    return float((vals > 0).mean()), int(row_mask.sum())


def within_family_breakdown(Z_cv_path_dense, cluster_ids_cv, cv_prots,
                             Z_gn_dense, gn_prots,
                             Z_cv_ben_dense=None, cv_ben_prots=None):
    """For each cluster+latent, compute fire rates across comparison groups.

    Primary benign comparison: ClinVar Benign variants (same protein family).
    Secondary benign comparison: gnomAD variants where available.
    """
    def _fisher(n_path, fr_path, n_ben, fr_ben):
        if n_path == 0 or n_ben == 0:
            return float("nan"), 1.0
        k_p = int(fr_path * n_path)
        k_b = int(fr_ben * n_ben)
        ct = [[k_p, n_path - k_p], [k_b, n_ben - k_b]]
        try:
            return fisher_exact(ct, alternative="greater")
        except Exception:
            return float("nan"), 1.0

    rows = []
    for k, top_lats in CLUSTER_TOP_LATS.items():
        dom_prot = CLUSTER_DOMINANT[k]

        # Masks into Z_cv_path (pathogenic ClinVar)
        in_clus          = cluster_ids_cv == k
        same_fam_cv      = cv_prots == dom_prot
        path_in_same     = in_clus & same_fam_cv    # cluster k, dominant protein
        path_out_same    = ~in_clus & same_fam_cv   # other clusters, dominant protein
        path_in_other    = in_clus & ~same_fam_cv   # cluster k, other proteins

        # Masks into ClinVar Benign (primary benign comparison)
        cv_ben_same  = (cv_ben_prots == dom_prot) if cv_ben_prots is not None else np.zeros(0, dtype=bool)
        cv_ben_other = (cv_ben_prots != dom_prot) if cv_ben_prots is not None else np.zeros(0, dtype=bool)

        # Masks into gnomAD (secondary, where available)
        gn_same  = gn_prots == dom_prot
        gn_other = ~gn_same

        for lat in top_lats:
            fr_path_in,  n_path_in   = fire_rate(Z_cv_path_dense, path_in_same, lat)
            fr_path_out, n_path_out  = fire_rate(Z_cv_path_dense, path_out_same, lat)
            fr_path_oth, _           = fire_rate(Z_cv_path_dense, path_in_other, lat)

            # ClinVar Benign comparison (primary)
            fr_cvb_same, n_cvb_same = (fire_rate(Z_cv_ben_dense, cv_ben_same, lat)
                                        if Z_cv_ben_dense is not None else (0.0, 0))
            fr_cvb_other, _         = (fire_rate(Z_cv_ben_dense, cv_ben_other, lat)
                                        if Z_cv_ben_dense is not None else (0.0, 0))

            # gnomAD comparison (secondary)
            fr_gn_same, n_gn_same   = fire_rate(Z_gn_dense, gn_same, lat)
            fr_gn_other, _          = fire_rate(Z_gn_dense, gn_other, lat)

            # Within-family enrichment: path_in / cvb_same (primary)
            cvb_enr = (fr_path_in / (fr_cvb_same + 1e-9)
                       if n_cvb_same > 0 else float("inf"))
            gn_enr  = (fr_path_in / (fr_gn_same + 1e-9)
                       if n_gn_same > 0 else float("inf"))

            or_cvb, p_cvb = _fisher(n_path_in, fr_path_in, n_cvb_same, fr_cvb_same)
            or_gn,  p_gn  = _fisher(n_path_in, fr_path_in, n_gn_same, fr_gn_same)

            rows.append(dict(
                cluster=k, dominant_prot=dom_prot, latent=lat,
                # group sizes
                n_path_in_clus_same=n_path_in,
                n_path_other_clus_same=n_path_out,
                n_cvben_same=n_cvb_same,
                n_gn_same=n_gn_same,
                # fire rates
                fr_path_in_clus_same=round(fr_path_in, 4),
                fr_path_other_clus_same=round(fr_path_out, 4),
                fr_path_in_clus_other=round(fr_path_oth, 4),
                fr_cvben_same=round(fr_cvb_same, 4),
                fr_cvben_other=round(fr_cvb_other, 4),
                fr_gn_same=round(fr_gn_same, 4),
                fr_gn_other=round(fr_gn_other, 4),
                # within-family discrimination (ClinVar Benign = primary)
                cvben_enrichment=round(cvb_enr, 2),
                cvben_fisher_or=round(or_cvb, 3) if not np.isnan(or_cvb) else float("nan"),
                cvben_fisher_p=round(p_cvb, 6),
                # gnomAD comparison (secondary)
                gn_enrichment=round(gn_enr, 2),
                gn_fisher_or=round(or_gn, 3) if not np.isnan(or_gn) else float("nan"),
                gn_fisher_p=round(p_gn, 6),
            ))

        print(f"  k={k:2d} ({dom_prot}):  "
              f"n_path_in={n_path_in}  n_cvb_same={n_cvb_same}  n_gn_same={n_gn_same}",
              flush=True)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cross-cluster within-family mechanism discrimination (TP53 + PTEN)
# ─────────────────────────────────────────────────────────────────────────────

def cross_cluster_mechanism(Z_cv_path_dense, cluster_ids_cv, cv_prots, label="TP53",
                             protein_id="P04637", clusters=None):
    """For variants of one protein appearing across multiple clusters,
    test whether each cluster's defining latents specifically fire on THAT cluster's
    variants and not the others'.

    Returns a pivot: rows=latent, cols=cluster_k, values=fire_rate.
    """
    if clusters is None:
        clusters = TP53_CLUSTERS

    # All pathogenic variants of this protein
    fam_mask = cv_prots == protein_id

    print(f"\n  {label} variants in each cluster:", flush=True)
    for k in clusters:
        n = ((cluster_ids_cv == k) & fam_mask).sum()
        print(f"    k={k}: {n}", flush=True)

    # All latents used by any of these clusters
    all_lats = []
    for k in clusters:
        all_lats.extend(CLUSTER_TOP_LATS.get(k, []))
    all_lats = sorted(set(all_lats))

    rows = []
    for lat in all_lats:
        row = {"latent": lat}
        # Which cluster(s) list this latent as a top latent?
        source_clusters = [k for k in clusters if lat in CLUSTER_TOP_LATS.get(k, [])]
        row["source_clusters"] = str(source_clusters)

        for k in clusters:
            clus_fam_mask = (cluster_ids_cv == k) & fam_mask
            fr, n = fire_rate(Z_cv_path_dense, clus_fam_mask, lat)
            row[f"fr_k{k}"] = round(fr, 4)
            row[f"n_k{k}"] = n
        rows.append(row)

    df = pd.DataFrame(rows)

    # Specificity score: max(fr_own) / mean(fr_others) for each latent
    spec_rows = []
    for _, row in df.iterrows():
        lat = row["latent"]
        src = eval(row["source_clusters"])
        for k_own in src:
            fr_own = row[f"fr_k{k_own}"]
            others = [row[f"fr_k{k}"] for k in clusters if k != k_own]
            mean_others = np.mean(others) if others else 0
            spec = fr_own / (mean_others + 1e-6)
            spec_rows.append({"latent": lat, "own_cluster": k_own,
                               "fr_own": fr_own, "mean_fr_others": round(mean_others, 4),
                               "within_protein_specificity": round(spec, 2)})

    df_spec = pd.DataFrame(spec_rows).sort_values("within_protein_specificity", ascending=False)
    return df, df_spec


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    print("Loading data …", flush=True)
    _, _, W_dec, b_dec = load_decoder()
    Z_cv, labels, _ = load_clinvar_data()
    Z_hg, Z_gn      = load_hgmd_gnomad()

    path_mask  = labels == 1
    Z_cv_path  = Z_cv[path_mask]
    Z_disease  = sp.vstack([Z_cv_path, Z_hg]).tocsr()

    print("Re-running k-means …", flush=True)
    _, cluster_ids_A, _ = run_disease_kmeans(Z_disease, verbose=True)
    cluster_ids_cv = cluster_ids_A[:Z_cv_path.shape[0]]

    print("Reconstructing ClinVar pathogenic protein IDs …", flush=True)
    cv_key_cache = LA / "cv_variant_keys.npz"
    complex_ids, _ = reconstruct_clinvar_variant_keys(cache_path=cv_key_cache)
    cv_prots = np.array([c.split("_")[0] for c in complex_ids[path_mask]])

    print("Reconstructing gnomAD protein IDs …", flush=True)
    gn_prot_cache = LA / "validation" / "within_family" / "gnomad_prot_ids.npy"
    gn_prots = reconstruct_gnomad_prot_ids(cache_path=gn_prot_cache)
    print(f"  Z_gn shape: {Z_gn.shape}  gn_prots: {len(gn_prots)}", flush=True)

    if len(gn_prots) != Z_gn.shape[0]:
        print(f"  WARNING: length mismatch {len(gn_prots)} vs {Z_gn.shape[0]}", flush=True)
        min_len = min(len(gn_prots), Z_gn.shape[0])
        gn_prots = gn_prots[:min_len]
        Z_gn = Z_gn[:min_len]

    print("Converting to dense …", flush=True)
    Z_cv_path_dense = np.asarray(Z_cv_path.todense(), dtype=np.float32)
    Z_gn_dense      = np.asarray(Z_gn.todense(), dtype=np.float32)

    # ClinVar Benign (primary within-family benign comparison)
    ben_mask   = labels == 0
    Z_cv_ben   = Z_cv[ben_mask]
    cv_ben_prots = np.array([c.split("_")[0] for c in complex_ids[ben_mask]])
    Z_cv_ben_dense = np.asarray(Z_cv_ben.todense(), dtype=np.float32)
    print(f"  ClinVar Benign: {Z_cv_ben.shape[0]:,} variants", flush=True)

    # ── Part A: Within-family fire rate breakdown ──────────────────────────────
    print("\n" + "="*70, flush=True)
    print("Part A: Within-family fire rate breakdown", flush=True)
    print("="*70, flush=True)
    df_wf = within_family_breakdown(
        Z_cv_path_dense, cluster_ids_cv, cv_prots,
        Z_gn_dense, gn_prots,
        Z_cv_ben_dense=Z_cv_ben_dense, cv_ben_prots=cv_ben_prots)
    df_wf.to_csv(OUT / "within_family_fire_rates.csv", index=False)

    print("\n  Within-family enrichment summary (pathogenic vs ClinVar Benign, same protein):", flush=True)
    summary = (df_wf.groupby("cluster")[["cvben_enrichment", "cvben_fisher_p", "gn_enrichment"]]
                    .agg({"cvben_enrichment": "max", "cvben_fisher_p": "min", "gn_enrichment": "max"})
                    .round(3))
    print(summary.to_string(), flush=True)

    # ── Part B: Cross-cluster TP53 mechanism discrimination ────────────────────
    print("\n" + "="*70, flush=True)
    print("Part B: Cross-cluster TP53 mechanism discrimination", flush=True)
    print("="*70, flush=True)
    df_tp53_full, df_tp53_spec = cross_cluster_mechanism(
        Z_cv_path_dense, cluster_ids_cv, cv_prots,
        label="TP53", protein_id="P04637", clusters=TP53_CLUSTERS)
    df_tp53_full.to_csv(OUT / "tp53_cross_cluster_fire_rates.csv", index=False)
    df_tp53_spec.to_csv(OUT / "tp53_within_protein_specificity.csv", index=False)

    print("\n  TP53 within-protein specificity (top 15 by spec ratio):", flush=True)
    print(df_tp53_spec.head(15).to_string(index=False), flush=True)

    # ── Part C: Cross-cluster PTEN mechanism discrimination ────────────────────
    print("\n" + "="*70, flush=True)
    print("Part C: Cross-cluster PTEN mechanism discrimination", flush=True)
    print("="*70, flush=True)
    df_pten_full, df_pten_spec = cross_cluster_mechanism(
        Z_cv_path_dense, cluster_ids_cv, cv_prots,
        label="PTEN", protein_id="P60484", clusters=PTEN_CLUSTERS)
    df_pten_full.to_csv(OUT / "pten_cross_cluster_fire_rates.csv", index=False)
    df_pten_spec.to_csv(OUT / "pten_within_protein_specificity.csv", index=False)

    print("\n  PTEN within-protein specificity (top 15 by spec ratio):", flush=True)
    print(df_pten_spec.head(15).to_string(index=False), flush=True)

    # ── Part D: Detailed per-latent table for key clusters ─────────────────────
    print("\n" + "="*70, flush=True)
    print("Part D: Per-latent within-family summary for key clusters", flush=True)
    print("="*70, flush=True)

    key_clusters = [0, 27, 8, 4, 35, 16]
    for k in key_clusters:
        sub = df_wf[df_wf["cluster"] == k][
            ["latent",
             "n_path_in_clus_same", "fr_path_in_clus_same",
             "n_cvben_same",        "fr_cvben_same",
             "n_gn_same",           "fr_gn_same",
             "cvben_enrichment",    "cvben_fisher_p",
             "gn_enrichment"]].head(5)
        dom = CLUSTER_DOMINANT[k]
        print(f"\n  Cluster {k} ({dom}):", flush=True)
        print(sub.to_string(index=False), flush=True)

    print(f"\nDone. Total time: {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"Outputs in {OUT}", flush=True)


if __name__ == "__main__":
    main()
