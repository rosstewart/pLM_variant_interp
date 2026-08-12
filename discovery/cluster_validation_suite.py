"""
cluster_validation_suite.py — Multi-section validation of disease mechanism clusters.

Imports shared infrastructure. All outputs → /data/ross/interp/latent_analysis/validation/

Sections
--------
 0B  Gating experiment: Approach A (full-space) vs Approach B (disease-enriched subspace) clustering
  1  Probe score distributions per cluster
  2  Latent activation specificity + causal probe shift
  3  Leave-one-gene-out centroid stability
  4  Residualized (protein-agnostic) clustering
  5  Fisher's exact test for ClinVar condition enrichment

Run individual sections with: python cluster_validation_suite.py --sections 1,2,3
Run all sections:            python cluster_validation_suite.py
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as stats
from scipy.spatial.distance import cosine as cosine_dist
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

# ── add sparse_bottleneck to path so shared_infrastructure is importable ────
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared_infrastructure import (
    LA, LA_LEGACY, LABEL_DIR,
    DEFAULT_NAME, N_CLUSTERS, RANDOM_SEED,
    load_decoder, load_clinvar_data, load_hgmd_gnomad, load_phenotype_data,
    train_recon_probes, run_disease_kmeans, disease_enriched_subspace,
    reconstruct_clinvar_variant_keys, enrichr_query, PROBE_TASKS,
)

OUT = LA / "validation"
OUT.mkdir(parents=True, exist_ok=True)

# Clusters to highlight in per-cluster outputs (interpretable / low-contamination)
FOCUS_CLUSTERS = [0, 4, 8, 12, 14, 16, 27, 31, 32, 33, 35, 46]


# ═════════════════════════════════════════════════════════════════════════════
# Shared data loading (called once; results passed into each section)
# ═════════════════════════════════════════════════════════════════════════════

def load_all(name: str = DEFAULT_NAME):
    print("Loading decoder …", flush=True)
    W_dec_diff, b_dec_diff, W_dec, b_dec = load_decoder(name)

    print("Loading ClinVar Z …", flush=True)
    Z_cv, labels, cv_prot_ids = load_clinvar_data(name)

    print("Loading HGMD / gnomAD Z …", flush=True)
    Z_hg, Z_gn = load_hgmd_gnomad(name)

    # Disease = ClinVar pathogenic (label 1) + all HGMD rows
    path_mask = labels == 1
    Z_cv_path  = Z_cv[path_mask]
    Z_disease  = sp.vstack([Z_cv_path, Z_hg]).tocsr()

    print("Loading phenotype labels …", flush=True)
    Z_stab, y_stab, Z_act, y_act, stab_mask, act_mask = load_phenotype_data(name)

    print("Training / loading recon probes …", flush=True)
    probe_cache = OUT / "probes.pkl"
    probes = train_recon_probes(
        Z_stab, y_stab, Z_act, y_act,
        W_dec_diff, b_dec_diff,
        cache_path=probe_cache, verbose=True)

    print("Reconstructing ClinVar variant keys …", flush=True)
    cv_key_cache = LA / "cv_variant_keys.npz"
    complex_ids, variant_1b = reconstruct_clinvar_variant_keys(cache_path=cv_key_cache)

    # Re-run reproducible k-means (Approach A)
    print(f"Running k-means k={N_CLUSTERS} (Approach A, full-space) …", flush=True)
    km_A, cluster_ids_A, Z_disease_norm = run_disease_kmeans(Z_disease, verbose=True)

    # Map the cluster IDs back to ClinVar pathogenic rows only (first n_cv_path rows)
    n_cv_path = Z_cv_path.shape[0]
    cluster_ids_cv = cluster_ids_A[:n_cv_path]

    return dict(
        name=name,
        W_dec_diff=W_dec_diff, b_dec_diff=b_dec_diff,
        W_dec=W_dec, b_dec=b_dec,
        Z_cv=Z_cv, labels=labels, cv_prot_ids=cv_prot_ids,
        Z_hg=Z_hg, Z_gn=Z_gn,
        Z_cv_path=Z_cv_path, Z_disease=Z_disease, Z_disease_norm=Z_disease_norm,
        path_mask=path_mask,
        complex_ids=complex_ids, variant_1b=variant_1b,
        cluster_ids_cv=cluster_ids_cv, cluster_ids_A=cluster_ids_A,
        km_A=km_A, n_cv_path=n_cv_path,
        probes=probes,
        Z_stab=Z_stab, y_stab=y_stab, stab_mask=stab_mask,
        Z_act=Z_act, y_act=y_act, act_mask=act_mask,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Section 0B — Gating experiment: Approach A vs. Approach B
# ═════════════════════════════════════════════════════════════════════════════

def section_0b(d: dict):
    """Compare full-space clustering (A) vs. disease-enriched subspace (B).

    Outputs
    -------
    validation/approach_comparison_summary.txt
    validation/approach_B_clusters.csv
    """
    print("\n" + "="*70)
    print("Section 0B: Approach A vs. Approach B gating experiment")
    print("="*70, flush=True)

    name    = d["name"]
    enr_csv = LA / f"latent_enrichment_{name}.csv"
    if not enr_csv.exists():
        enr_csv = LA_LEGACY / f"latent_enrichment_{name}.csv"
    if not enr_csv.exists():
        print(f"  SKIP: enrichment CSV not found at {enr_csv}", flush=True)
        return

    Z_disease = d["Z_disease"]

    # ── Approach B: disease-enriched subspace ─────────────────────────────────
    print("  Building disease-enriched subspace (threshold > 0.5) …", flush=True)
    Z_sub, enr_lats = disease_enriched_subspace(Z_disease, enr_csv, threshold=0.5)
    print(f"  Subspace: {Z_sub.shape[1]} enriched latents (of {Z_disease.shape[1]})", flush=True)

    print("  Running k-means (Approach B, subspace) …", flush=True)
    km_B, cluster_ids_B, Z_sub_norm = run_disease_kmeans(Z_sub, verbose=True)

    # ── Metric 1: Enrichr pathway enrichment strength ─────────────────────────
    # Use the first N_CLUSTERS×2 runs for speed; query top gene list per cluster
    print("  Metric 1: Enrichr pathway enrichment …", flush=True)
    # pathogenic-only complex_ids, aligned with Z_cv_path / cluster_ids_A[:n_cv_path]
    name_field = d["complex_ids"][d["path_mask"]]
    n_cv_path  = d["n_cv_path"]

    def top_genes_for_cluster(cluster_ids, cids, k, n=50):
        sel = np.where(cluster_ids[:len(cids)] == k)[0]
        from collections import Counter
        gene_counts = Counter(cids[sel])
        return [g.split("_")[0] for g, _ in gene_counts.most_common(n)]

    metrics1_A, metrics1_B = [], []
    for k in range(N_CLUSTERS):
        genes_A = top_genes_for_cluster(d["cluster_ids_A"], name_field, k)
        genes_B = top_genes_for_cluster(cluster_ids_B,       name_field, k)
        res_A = enrichr_query(genes_A, description=f"A_k{k}", n_top=3)
        res_B = enrichr_query(genes_B, description=f"B_k{k}", n_top=3)

        def best_neg_log_p(res):
            best = 0.0
            for terms in res.values():
                for term in terms:
                    if term[2] > 0:
                        best = max(best, -np.log10(term[2]))
            return best

        metrics1_A.append(best_neg_log_p(res_A))
        metrics1_B.append(best_neg_log_p(res_B))
        if (k + 1) % 10 == 0:
            print(f"    {k+1}/{N_CLUSTERS} clusters done", flush=True)

    metrics1_A = np.array(metrics1_A)
    metrics1_B = np.array(metrics1_B)
    stat1, p1 = stats.wilcoxon(metrics1_B, metrics1_A, alternative="greater")

    # ── Metric 6: Disease enrichment of top latents ───────────────────────────
    df_enr = pd.read_csv(enr_csv)
    lat_enr = df_enr["enrichment"].to_numpy()

    def top_lat_enr(cluster_ids, Z_norm_mat, k, n_top=10):
        sel = np.where(cluster_ids == k)[0]
        if len(sel) == 0:
            return 0.0
        centroid = np.asarray(Z_norm_mat[sel].mean(axis=0)).flatten()
        top_lats = np.argsort(centroid)[-n_top:]
        # Only latents within the original space; subspace lats need remapping
        if len(top_lats) > len(lat_enr):
            top_lats = top_lats[top_lats < len(lat_enr)]
        return float(lat_enr[top_lats].mean())

    def remap_lats(enr_lats, top_lats):
        return enr_lats[top_lats]

    metrics6_A, metrics6_B = [], []
    for k in range(N_CLUSTERS):
        metrics6_A.append(top_lat_enr(d["cluster_ids_A"], d["Z_disease_norm"], k))
        # For B, remap subspace latent indices back to full-space
        sel_B = np.where(cluster_ids_B == k)[0]
        if len(sel_B) == 0:
            metrics6_B.append(0.0)
        else:
            centroid_B = np.asarray(Z_sub_norm[sel_B].mean(axis=0)).flatten()
            top_B = np.argsort(centroid_B)[-10:]
            full_lats = remap_lats(enr_lats, top_B)
            metrics6_B.append(float(lat_enr[full_lats].mean()))

    metrics6_A = np.array(metrics6_A)
    metrics6_B = np.array(metrics6_B)
    stat6, p6 = stats.wilcoxon(metrics6_B, metrics6_A, alternative="greater")

    # ── Metric 3: Fire-in / fire-out specificity ──────────────────────────────
    def fire_ratio(cluster_ids, Z_mat, k, n_top=10):
        sel_in  = np.where(cluster_ids == k)[0]
        sel_out = np.where(cluster_ids != k)[0]
        Z_mat_dense = np.asarray(Z_mat.todense() if sp.issparse(Z_mat) else Z_mat)
        centroid = Z_mat_dense[sel_in].mean(axis=0)
        top_lats = np.argsort(centroid)[-n_top:]
        fire_in  = (Z_mat_dense[sel_in][:, top_lats] > 0).mean()
        fire_out = (Z_mat_dense[sel_out][:, top_lats] > 0).mean()
        return fire_in / (fire_out + 1e-6)

    # Compute on dense subsets to avoid memory issues
    Z_dis_dense = np.asarray(d["Z_disease"].todense(), dtype=np.float32)
    Z_sub_dense = np.asarray(Z_sub.todense(), dtype=np.float32)
    metrics3_A, metrics3_B = [], []
    for k in range(N_CLUSTERS):
        sel_in  = np.where(d["cluster_ids_A"] == k)[0]
        sel_out = np.where(d["cluster_ids_A"] != k)[0]
        centroid_A = Z_dis_dense[sel_in].mean(axis=0)
        top_A = np.argsort(centroid_A)[-10:]
        fi_A = (Z_dis_dense[sel_in][:, top_A] > 0).mean()
        fo_A = (Z_dis_dense[sel_out][:, top_A] > 0).mean()
        metrics3_A.append(fi_A / (fo_A + 1e-6))

        sel_in_B  = np.where(cluster_ids_B == k)[0]
        sel_out_B = np.where(cluster_ids_B != k)[0]
        if len(sel_in_B) == 0:
            metrics3_B.append(0.0)
            continue
        centroid_B = Z_sub_dense[sel_in_B].mean(axis=0)
        top_B = np.argsort(centroid_B)[-10:]
        fi_B = (Z_sub_dense[sel_in_B][:, top_B] > 0).mean()
        fo_B = (Z_sub_dense[sel_out_B][:, top_B] > 0).mean()
        metrics3_B.append(fi_B / (fo_B + 1e-6))

    metrics3_A = np.array(metrics3_A)
    metrics3_B = np.array(metrics3_B)
    stat3, p3 = stats.wilcoxon(metrics3_B, metrics3_A, alternative="greater")

    # ── Decision ───────────────────────────────────────────────────────────────
    n_favors_B = sum([p1 < 0.05, p3 < 0.05, p6 < 0.05])
    decision = "ADOPT Approach B" if n_favors_B >= 2 else "RETAIN Approach A"

    summary = (
        f"Gating experiment: Approach A (full-space k=50) vs. Approach B "
        f"(disease-enriched subspace, {Z_sub.shape[1]} latents)\n\n"
        f"Metric 1 (Enrichr -log10 adj-p):       "
        f"A median={np.median(metrics1_A):.2f}  B median={np.median(metrics1_B):.2f}  "
        f"Wilcoxon p={p1:.4f}  {'B wins' if p1 < 0.05 else 'no sig diff'}\n"
        f"Metric 3 (fire_in/out specificity):     "
        f"A median={np.median(metrics3_A):.2f}  B median={np.median(metrics3_B):.2f}  "
        f"Wilcoxon p={p3:.4f}  {'B wins' if p3 < 0.05 else 'no sig diff'}\n"
        f"Metric 6 (disease enr of top latents):  "
        f"A median={np.median(metrics6_A):.2f}  B median={np.median(metrics6_B):.2f}  "
        f"Wilcoxon p={p6:.4f}  {'B wins' if p6 < 0.05 else 'no sig diff'}\n\n"
        f"Metrics favoring B: {n_favors_B}/3 (≥2 required)\n"
        f"DECISION: {decision}\n"
    )
    print(summary, flush=True)
    (OUT / "approach_comparison_summary.txt").write_text(summary)

    # Save Approach B cluster assignments (pathogenic-only rows, aligned with Z_cv_path)
    n_cv_path = d["n_cv_path"]
    cv_path_complex = d["complex_ids"][d["path_mask"]]
    cv_path_variant = d["variant_1b"][d["path_mask"]]
    df_B = pd.DataFrame({
        "complex_id": cv_path_complex,
        "variant_1b": cv_path_variant,
        "prot_id":    [c.split("_")[0] for c in cv_path_complex],
        "cluster_B":  cluster_ids_B[:n_cv_path],
    })
    df_B.to_csv(OUT / "approach_B_clusters.csv", index=False)
    print(f"  Saved → {OUT}/approach_comparison_summary.txt", flush=True)

    # Store B assignments in d for use by other sections if desired
    d["cluster_ids_B"] = cluster_ids_B
    d["Z_sub_dense"]   = Z_sub_dense


# ═════════════════════════════════════════════════════════════════════════════
# Section 1 — Probe score distributions per cluster
# ═════════════════════════════════════════════════════════════════════════════

def section_1(d: dict):
    """KS test + Mann-Whitney U for each probe score between cluster and all-pathogenic."""
    print("\n" + "="*70)
    print("Section 1: Probe score distributions per cluster")
    print("="*70, flush=True)

    W_dec_diff  = d["W_dec_diff"]
    b_dec_diff  = d["b_dec_diff"]
    probes      = d["probes"]
    Z_cv_path   = d["Z_cv_path"]
    cluster_ids_cv = d["cluster_ids_cv"]

    xh_all = (np.asarray(Z_cv_path.dot(W_dec_diff.T), dtype=np.float32)
              + b_dec_diff)

    rows = []
    for task, (pos_cls, neg_cls, ds) in PROBE_TASKS.items():
        clf    = probes[task]
        scores = clf.predict_proba(xh_all)[:, 1]

        for k in range(N_CLUSTERS):
            in_mask  = cluster_ids_cv == k
            out_mask = ~in_mask
            sc_in  = scores[in_mask]
            sc_out = scores[out_mask]
            if len(sc_in) < 5:
                continue
            ks_stat, ks_p = stats.ks_2samp(sc_in, sc_out)
            mw_stat, mw_p = stats.mannwhitneyu(sc_in, sc_out, alternative="two-sided")
            rows.append({
                "cluster": k,
                "task": task,
                "n_in": len(sc_in),
                "mean_in": float(sc_in.mean()),
                "mean_out": float(sc_out.mean()),
                "ks_stat": float(ks_stat),
                "ks_p": float(ks_p),
                "mw_stat": float(mw_stat),
                "mw_p": float(mw_p),
            })

    df_dist = pd.DataFrame(rows)
    df_dist.to_csv(OUT / "probe_distribution_stats.csv", index=False)
    print(f"  Saved → {OUT}/probe_distribution_stats.csv", flush=True)

    sig_rows = df_dist[(df_dist["ks_p"] < 0.05) & (df_dist["cluster"].isin(FOCUS_CLUSTERS))]
    print(f"  Significant (ks_p<0.05) in focus clusters:\n{sig_rows[['cluster','task','mean_in','mean_out','ks_p']].to_string(index=False)}", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# Section 2 — Latent activation specificity
# ═════════════════════════════════════════════════════════════════════════════

def section_2(d: dict):
    """fire_in / fire_out ratio and causal probe shift for each cluster's top latents."""
    print("\n" + "="*70)
    print("Section 2: Latent activation specificity")
    print("="*70, flush=True)

    Z_cv_path   = d["Z_cv_path"]
    cluster_ids_cv = d["cluster_ids_cv"]
    W_dec_diff  = d["W_dec_diff"]
    probes      = d["probes"]

    Z_dense = np.asarray(Z_cv_path.todense(), dtype=np.float32)
    n_lats  = Z_dense.shape[1]

    rows = []
    for k in FOCUS_CLUSTERS:
        in_mask  = cluster_ids_cv == k
        out_mask = ~in_mask
        n_in, n_out = in_mask.sum(), out_mask.sum()
        if n_in < 5:
            continue

        # Top 10 latents by centroid value
        centroid   = Z_dense[in_mask].mean(axis=0)
        top10      = np.argsort(centroid)[-10:][::-1]

        for lat in top10:
            fire_in  = float((Z_dense[in_mask,  lat] > 0).mean())
            fire_out = float((Z_dense[out_mask, lat] > 0).mean())
            spec_ratio = fire_in / (fire_out + 1e-6)

            # Causal probe shift: delta score = probe_coef @ (Z[i,lat] * W_dec_diff[:,lat])
            # Mean over in-cluster and out-of-cluster variants, for each probe task
            w_lat = W_dec_diff[:, lat]   # (half_dim,)
            causal = {}
            for task, (pos_cls, neg_cls, ds) in PROBE_TASKS.items():
                clf     = probes[task]
                coef    = clf.coef_[0]    # (half_dim,)
                delta_w = float(coef @ w_lat)  # scalar: change in linear score per unit activation
                causal[f"delta_{task}_in"]  = delta_w * float(Z_dense[in_mask,  lat].mean())
                causal[f"delta_{task}_out"] = delta_w * float(Z_dense[out_mask, lat].mean())
                causal[f"causal_spec_{task}"] = (causal[f"delta_{task}_in"]
                                                 - causal[f"delta_{task}_out"])

            rows.append({
                "cluster": k, "latent": int(lat),
                "fire_in": fire_in, "fire_out": fire_out,
                "specificity_ratio": spec_ratio,
                **causal,
            })

    df_spec = pd.DataFrame(rows)
    df_spec.to_csv(OUT / "latent_specificity.csv", index=False)
    print(f"  Saved → {OUT}/latent_specificity.csv", flush=True)

    top_by_cluster = (df_spec.sort_values("specificity_ratio", ascending=False)
                              .groupby("cluster").head(3)
                              [["cluster","latent","fire_in","fire_out","specificity_ratio"]])
    print(f"  Top latents by specificity_ratio (focus clusters):\n{top_by_cluster.to_string(index=False)}", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# Section 3 — Leave-one-gene-out centroid stability
# ═════════════════════════════════════════════════════════════════════════════

def section_3(d: dict):
    """Stability of cluster centroids when the dominant gene is removed."""
    print("\n" + "="*70)
    print("Section 3: Leave-one-gene-out centroid stability")
    print("="*70, flush=True)

    Z_dis_norm     = d["Z_disease_norm"]
    cluster_ids_A  = d["cluster_ids_A"]
    n_cv_path      = d["n_cv_path"]
    km_A           = d["km_A"]

    # Pathogenic-only complex_ids, aligned with Z_cv_path / cluster_ids_A[:n_cv_path]
    cv_complex = d["complex_ids"][d["path_mask"]]

    rows = []
    for k in FOCUS_CLUSTERS:
        # Focus on ClinVar pathogenic partition only
        in_idx = np.where(cluster_ids_A[:n_cv_path] == k)[0]
        if len(in_idx) < 10:
            continue

        genes = np.array([cv_complex[i].split("_")[0] for i in in_idx])
        gene_counts = pd.Series(genes).value_counts()
        dominant_gene = gene_counts.index[0]
        n_removed = int(gene_counts.iloc[0])

        # Remove dominant gene rows from the *full disease* set
        all_genes = np.array([cv_complex[i].split("_")[0]
                               if i < n_cv_path else "_HGMD_"
                               for i in range(cluster_ids_A.shape[0])])
        keep_mask = all_genes != dominant_gene
        Z_reduced = Z_dis_norm[keep_mask]
        orig_cluster_ids = cluster_ids_A[keep_mask]

        # Re-run k-means
        km_loo = MiniBatchKMeans(
            n_clusters=N_CLUSTERS, batch_size=8192,
            max_iter=300, n_init=5, random_state=RANDOM_SEED)
        km_loo.fit(Z_reduced)

        # Find nearest new centroid to original centroid k
        orig_centroid = km_A.cluster_centers_[k]
        sims = [1 - cosine_dist(orig_centroid, km_loo.cluster_centers_[j])
                for j in range(N_CLUSTERS)]
        best_new = int(np.argmax(sims))
        centroid_sim = float(sims[best_new])

        # Survival rate: remaining in-cluster variants that map to best_new
        in_loo_idx = np.where((orig_cluster_ids == k) & keep_mask[:len(orig_cluster_ids)])[0]
        if len(in_loo_idx) > 0:
            new_assignments = km_loo.predict(Z_reduced[in_loo_idx])
            survival_rate = float((new_assignments == best_new).mean())
        else:
            survival_rate = 0.0

        # Gene list for Enrichr (before and after removal)
        genes_before = gene_counts.index.tolist()[:50]
        genes_after  = [g for g in genes_before if g != dominant_gene]

        res_before = enrichr_query(genes_before, description=f"loog_k{k}_before", n_top=3)
        res_after  = enrichr_query(genes_after,  description=f"loog_k{k}_after",  n_top=3)

        def top_kegg(res):
            for term in res.get("KEGG_2021_Human", []):
                return term[0]
            return ""

        top_before = top_kegg(res_before)
        top_after  = top_kegg(res_after)
        pathway_stable = (top_before != "" and top_before == top_after)

        rows.append({
            "cluster": k,
            "removed_gene": dominant_gene,
            "n_removed": n_removed,
            "centroid_cosine_sim": centroid_sim,
            "survival_rate": survival_rate,
            "top_kegg_before": top_before,
            "top_kegg_after": top_after,
            "pathway_stable": pathway_stable,
        })
        print(f"  k={k:2d}  gene={dominant_gene:10s}  cosine_sim={centroid_sim:.3f}  "
              f"survival={survival_rate:.2f}  stable={pathway_stable}", flush=True)

    df_loog = pd.DataFrame(rows)
    df_loog.to_csv(OUT / "leave_one_gene_out.csv", index=False)
    print(f"  Saved → {OUT}/leave_one_gene_out.csv", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# Section 4 — Residualized clustering
# ═════════════════════════════════════════════════════════════════════════════

def section_4(d: dict):
    """Cluster in residualized space (subtract per-protein mean Z) to disentangle
    sequence homology from pathomechanism signal."""
    print("\n" + "="*70)
    print("Section 4: Residualized (protein-agnostic) clustering")
    print("="*70, flush=True)

    Z_cv_path   = d["Z_cv_path"]
    cluster_ids_cv = d["cluster_ids_cv"]
    complex_ids    = d["complex_ids"][d["path_mask"]]  # pathogenic-only, aligned with Z_cv_path

    Z_dense = np.asarray(Z_cv_path.todense(), dtype=np.float32)
    prots   = np.array([c.split("_")[0] for c in complex_ids])

    # Subtract per-protein mean
    print("  Computing per-protein residuals …", flush=True)
    Z_res = Z_dense.copy()
    for prot in np.unique(prots):
        mask = prots == prot
        if mask.sum() >= 5:
            Z_res[mask] -= Z_dense[mask].mean(axis=0)

    # Normalise and cluster
    Z_res_norm = normalize(Z_res, norm="l2")
    print("  Running k-means on residualized space …", flush=True)
    km_res = MiniBatchKMeans(
        n_clusters=N_CLUSTERS, batch_size=8192,
        max_iter=300, n_init=5, random_state=RANDOM_SEED)
    cluster_ids_res = km_res.fit_predict(Z_res_norm)

    # Jaccard overlap matrix (original vs. residualized)
    print("  Computing Jaccard overlap matrix …", flush=True)
    jaccard = np.zeros((N_CLUSTERS, N_CLUSTERS), dtype=np.float32)
    for a in range(N_CLUSTERS):
        set_a = set(np.where(cluster_ids_cv == a)[0])
        for b in range(N_CLUSTERS):
            set_b = set(np.where(cluster_ids_res == b)[0])
            inter = len(set_a & set_b)
            union = len(set_a | set_b)
            jaccard[a, b] = inter / union if union > 0 else 0

    pd.DataFrame(jaccard,
                 index=[f"A_{k}" for k in range(N_CLUSTERS)],
                 columns=[f"R_{k}" for k in range(N_CLUSTERS)]).to_csv(
        OUT / "cluster_overlap_matrix.csv")

    # Per-focus-cluster: best matching residualized cluster
    rows = []
    for k in FOCUS_CLUSTERS:
        best_res = int(np.argmax(jaccard[k]))
        best_j   = float(jaccard[k, best_res])

        genes_orig = [complex_ids[i].split("_")[0]
                      for i in np.where(cluster_ids_cv == k)[0]][:50]
        genes_res  = [complex_ids[i].split("_")[0]
                      for i in np.where(cluster_ids_res == best_res)[0]][:50]

        res_orig = enrichr_query(genes_orig, description=f"res_orig_k{k}", n_top=3)
        res_res  = enrichr_query(genes_res,  description=f"res_k{best_res}", n_top=3)

        def top_kegg(r):
            for t in r.get("KEGG_2021_Human", []):
                return t[0]
            return ""

        path_match = (top_kegg(res_orig) != "" and
                      top_kegg(res_orig) == top_kegg(res_res))
        rows.append({
            "orig_cluster": k,
            "best_res_cluster": best_res,
            "jaccard": best_j,
            "top_kegg_orig": top_kegg(res_orig),
            "top_kegg_res": top_kegg(res_res),
            "pathway_match": path_match,
        })
        print(f"  orig k={k:2d} → res k={best_res:2d}  J={best_j:.3f}  "
              f"match={path_match}", flush=True)

    df_res = pd.DataFrame(rows)
    df_res.to_csv(OUT / "residualized_vs_original_summary.csv", index=False)

    # Full residualized cluster summary
    res_summary = []
    for k in range(N_CLUSTERS):
        mask = cluster_ids_res == k
        res_summary.append({
            "cluster": k,
            "n_variants": int(mask.sum()),
            "n_unique_prots": len(set(prots[mask])),
        })
    pd.DataFrame(res_summary).to_csv(OUT / "residualized_clusters.csv", index=False)
    print(f"  Saved → {OUT}/residualized_vs_original_summary.csv", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# Section 5 — Fisher's exact test for ClinVar conditions
# ═════════════════════════════════════════════════════════════════════════════

def section_5(d: dict):
    """Hypergeometric / Fisher's exact test for condition enrichment per cluster.

    Avoids gene-abundance bias from raw condition counts.
    """
    print("\n" + "="*70)
    print("Section 5: Fisher's exact test for ClinVar conditions")
    print("="*70, flush=True)

    import gzip
    from scipy.stats import fisher_exact
    from statsmodels.stats.multitest import multipletests

    varsummary_gz = LA / "variant_summary.txt.gz"
    if not varsummary_gz.exists():
        varsummary_gz = LA_LEGACY / "variant_summary.txt.gz"
    if not varsummary_gz.exists():
        print("  SKIP: variant_summary.txt.gz not found. "
              "Run validate_disease_clusters.py first.", flush=True)
        return

    print("  Parsing variant_summary.txt.gz …", flush=True)
    keep_cols = ["GeneSymbol", "ClinicalSignificance", "PhenotypeList"]
    with gzip.open(str(varsummary_gz), "rt", errors="replace") as f:
        df_vs = pd.read_csv(f, sep="\t", usecols=keep_cols, low_memory=False)

    df_path = df_vs[df_vs["ClinicalSignificance"].str.contains(
        "Pathogenic", case=False, na=False)].copy()
    df_path = df_path[df_path["GeneSymbol"].notna() &
                      (df_path["GeneSymbol"] != "-")]

    # Explode condition list
    rows_expanded = []
    for _, row in df_path.iterrows():
        gene = str(row["GeneSymbol"]).strip()
        for cond in str(row["PhenotypeList"]).split("|"):
            cond = cond.strip()
            if cond and cond != "not provided" and cond != "not specified":
                rows_expanded.append({"gene": gene, "condition": cond})
    df_exp = pd.DataFrame(rows_expanded).drop_duplicates()

    # Background gene set (all ClinVar pathogenic genes)
    bg_genes = set(df_path["GeneSymbol"].unique())
    N_bg     = len(bg_genes)

    # Condition → gene set (background)
    cond2genes = df_exp.groupby("condition")["gene"].apply(set).to_dict()

    # Filter to conditions with ≥ 5 background genes
    cond2genes = {c: gs for c, gs in cond2genes.items() if len(gs) >= 5}

    complex_ids    = d["complex_ids"][d["path_mask"]]  # pathogenic-only, aligned with cluster_ids_cv
    cluster_ids_cv = d["cluster_ids_cv"]

    rows = []
    for k in FOCUS_CLUSTERS:
        clus_mask = cluster_ids_cv == k
        clus_genes = set(complex_ids[clus_mask])
        clus_genes = {g.split("_")[0] for g in clus_genes}   # strip interactor B
        n_clus = len(clus_genes)
        if n_clus < 3:
            continue

        for cond, cond_bg_genes in cond2genes.items():
            k_val = len(clus_genes & cond_bg_genes)
            K_val = len(cond_bg_genes)
            contingency = [
                [k_val,         n_clus - k_val],
                [K_val - k_val, N_bg - K_val - (n_clus - k_val)],
            ]
            try:
                or_val, p_val = fisher_exact(contingency, alternative="greater")
            except Exception:
                continue
            if p_val < 0.1:
                rows.append({
                    "cluster": k,
                    "condition": cond,
                    "k": k_val,
                    "n": n_clus,
                    "K": K_val,
                    "N": N_bg,
                    "odds_ratio": float(or_val),
                    "pval": float(p_val),
                })

    if not rows:
        print("  No nominally significant enrichments found.", flush=True)
        return

    df_fish = pd.DataFrame(rows)
    _, adj_pvals, _, _ = multipletests(df_fish["pval"].to_numpy(),
                                        method="fdr_bh", alpha=0.05)
    df_fish["adj_pval"] = adj_pvals
    df_fish.sort_values(["cluster", "adj_pval"], inplace=True)
    df_fish.to_csv(OUT / "condition_enrichment_fisher.csv", index=False)
    print(f"  Saved → {OUT}/condition_enrichment_fisher.csv", flush=True)

    sig = df_fish[df_fish["adj_pval"] < 0.05]
    print(f"  {len(sig)} significant (adj_p<0.05) cluster-condition pairs:")
    print(sig[["cluster","condition","odds_ratio","adj_pval"]].head(20).to_string(index=False), flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

SECTION_MAP = {
    "0b": section_0b,
    "0B": section_0b,
    "1":  section_1,
    "2":  section_2,
    "3":  section_3,
    "4":  section_4,
    "5":  section_5,
}


def main():
    parser = argparse.ArgumentParser(
        description="Cluster validation suite for MutPred-PPI SAE disease clusters")
    parser.add_argument(
        "--sections", default="0B,1,2,3,4,5",
        help="Comma-separated list of sections to run, e.g. '1,2,5' (default: all)")
    parser.add_argument(
        "--name", default=DEFAULT_NAME,
        help="SAE model name (default: concat_ef1_k128)")
    args = parser.parse_args()

    sections = [s.strip() for s in args.sections.split(",")]

    t0 = time.time()
    d  = load_all(args.name)

    for sec in sections:
        if sec in SECTION_MAP:
            SECTION_MAP[sec](d)
        else:
            print(f"  Unknown section '{sec}', skipping.", flush=True)

    print(f"\nDone. Total time: {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"Outputs in {OUT}", flush=True)


if __name__ == "__main__":
    main()
