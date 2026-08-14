"""
functional_site_analysis.py

Labels disease variants with functional site annotations and tests whether
loss-of-property variants cluster together in the SAE latent space.

Data sources:
  /data/dbs/jose_struct_func_sites/structural_and_functional_sites/*.pos
      PDB-space functional residues (30 site types: catalytic, metal, cofactor, etc.)
      Mapped to UniProt positions via SIFTS.
  /data/dbs/metal_binding/gnomad_filtered_max300_ac10.parquet.csv
      Per-residue metal coordination data already in UniProt space.

Pipeline:
  1. Build (UniProt, pos) → {site_type} map from both sources (cached as parquet).
  2. Label ClinVar pathogenic variants (and HGMD where possible) with site types.
  3. Fisher's exact test for each (cluster, site_type) pair (BH-corrected).
  4. Heatmap + per-cluster summary text.
  5. Mutagenesis variant overlay (jose variants with functional consequence annotations).

Outputs → /data/ross/interp/latent_analysis/validation/
"""

import sys, re, time, argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import fisher_exact

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared_infrastructure import (
    LA, DEFAULT_NAME, N_CLUSTERS, RANDOM_SEED,
    load_clinvar_data, load_hgmd_gnomad,
    reconstruct_clinvar_variant_keys, run_disease_kmeans,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
JOSE_DIR     = Path("/data/dbs/jose_struct_func_sites/structural_and_functional_sites")
JOSE_VAR     = Path("/data/dbs/jose_struct_func_sites/variants/mutagenesis.variants")
METAL_CSV    = Path("/data/dbs/metal_binding/gnomad_filtered_max300_ac10.parquet.csv")
SIFTS_CSV    = Path("/data/dbs/SIFTS/pdb_chain_uniprot_2026_05_02.csv")
OUT          = LA / "validation"
SITE_MAP_CACHE = LA / "functional_site_map.parquet"
OUT.mkdir(parents=True, exist_ok=True)

_ap = argparse.ArgumentParser(description="Functional site enrichment analysis")
_ap.add_argument("--name", default=DEFAULT_NAME)
_ap.add_argument("--rebuild-cache", action="store_true",
                 help="Rebuild SIFTS mapping cache even if it exists")
_args, _ = _ap.parse_known_args()
NAME = _args.name

# ── Site type groupings ────────────────────────────────────────────────────────
JOSE_SITE_MAP = {
    "ZN":       "loss_metal_zn",
    "CA":       "loss_metal_ca",
    "MG":       "loss_metal_mg",
    "MN":       "loss_metal_mn",
    "FE":       "loss_metal_fe",
    "CU":       "loss_metal_cu",
    "CO":       "loss_metal_co",
    "NI":       "loss_metal_ni",
    "HEM":      "loss_metal_hem",
    "Cat":      "loss_catalytic",
    "FAD":      "loss_cofactor_fad",
    "NAD":      "loss_cofactor_nad",
    "ATP":      "loss_cofactor_atp",
    "ADP":      "loss_cofactor_nucleotide",
    "GDP":      "loss_cofactor_nucleotide",
    "GTP":      "loss_cofactor_nucleotide",
    "UDP":      "loss_cofactor_nucleotide",
    "FMN":      "loss_cofactor_fad",
    "PLP":      "loss_cofactor_plp",
    "DNA":      "loss_dna_binding",
    "RNA":      "loss_rna_binding",
    "PPI":      "loss_ppi_interface",
    "Allo":     "loss_allostery",
    "Phos":     "loss_phosphosite",
    "Nglyco":   "loss_glycosite",
    "Stability":"stability_site",
    "Hotspot":  "cancer_hotspot",
    "NA":       "loss_metal_na",
    "K":        "loss_metal_k",
    "CD":       "loss_metal_cd",
}

SITE_TYPE_ORDER = [
    "loss_catalytic", "loss_metal_zn", "loss_metal_ca", "loss_metal_mg",
    "loss_metal_mn", "loss_metal_fe", "loss_metal_cu", "loss_metal_hem",
    "loss_metal_ni", "loss_metal_co", "loss_metal_na", "loss_metal_k",
    "loss_cofactor_fad", "loss_cofactor_nad", "loss_cofactor_atp",
    "loss_cofactor_nucleotide", "loss_cofactor_plp",
    "loss_dna_binding", "loss_rna_binding",
    "loss_ppi_interface", "loss_allostery",
    "loss_phosphosite", "loss_glycosite",
    "stability_site", "cancer_hotspot",
]

_VAR_RE = re.compile(r'^([A-Z*])(\d+)([A-Z*])$')


# ══════════════════════════════════════════════════════════════════════════════
# §1  Build functional site map
# ══════════════════════════════════════════════════════════════════════════════

def _build_sifts_lookup(sifts_csv: Path) -> dict:
    """Return dict[(pdb_upper, chain)] → list of (pdb_beg, pdb_end, sp_beg, sp_end, uniprot)."""
    print("  Loading SIFTS …")
    sifts = pd.read_csv(str(sifts_csv), dtype=str, comment="#")
    sifts.columns = sifts.columns.str.strip()
    needed = {"PDB", "CHAIN", "SP_PRIMARY", "PDB_BEG", "PDB_END", "SP_BEG", "SP_END"}
    missing = needed - set(sifts.columns)
    if missing:
        raise ValueError(f"SIFTS missing columns: {missing}. Got: {list(sifts.columns)}")

    lookup = defaultdict(list)
    for _, row in sifts.iterrows():
        try:
            pdb_beg = int(row["PDB_BEG"])
            pdb_end = int(row["PDB_END"])
            sp_beg  = int(row["SP_BEG"])
            sp_end  = int(row["SP_END"])
        except (ValueError, TypeError):
            continue
        key = (str(row["PDB"]).upper(), str(row["CHAIN"]))
        lookup[key].append((pdb_beg, pdb_end, sp_beg, sp_end, str(row["SP_PRIMARY"])))
    print(f"  SIFTS: {len(lookup):,} (PDB, chain) pairs")
    return lookup


def _pdb_to_uniprot(pdb: str, chain: str, resno_str: str, sifts: dict):
    """Map a PDB author residue number to (UniProt_ID, UniProt_pos). Returns None on failure."""
    try:
        resno = int(resno_str)
    except ValueError:
        return None  # insertion codes like "47A"
    key = (pdb.upper(), chain)
    for pdb_beg, pdb_end, sp_beg, sp_end, uniprot in sifts.get(key, []):
        if pdb_beg <= resno <= pdb_end:
            offset = resno - pdb_beg
            if sp_beg + offset <= sp_end:
                return uniprot, sp_beg + offset
    return None


def build_site_map(rebuild: bool = False) -> dict:
    """Build and cache (UniProt_ID, pos_1b) → frozenset[site_type] from all sources."""
    if not rebuild and SITE_MAP_CACHE.exists():
        print(f"Loading site map from cache: {SITE_MAP_CACHE}")
        df = pd.read_parquet(str(SITE_MAP_CACHE))
        site_map = defaultdict(set)
        for _, row in df.iterrows():
            site_map[(row["uniprot"], int(row["pos"]))] |= set(row["site_types"].split(","))
        print(f"  {len(site_map):,} (UniProt, pos) entries")
        return site_map

    print("Building functional site map …")
    site_map = defaultdict(set)

    # ── Source A: jose .pos files via SIFTS ───────────────────────────────────
    sifts = _build_sifts_lookup(SIFTS_CSV)

    total_jose, mapped_jose = 0, 0
    for pos_file in sorted(JOSE_DIR.glob("*.pos")):
        site_name = pos_file.stem           # e.g. "ZN", "Cat"
        site_label = JOSE_SITE_MAP.get(site_name)
        if site_label is None:
            print(f"  [warn] no label for {site_name}.pos — skipping")
            continue

        with open(pos_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                pdb, chain, resno_str = parts[0].strip(), parts[1].strip(), parts[2].strip()
                total_jose += 1
                result = _pdb_to_uniprot(pdb, chain, resno_str, sifts)
                if result is not None:
                    site_map[result].add(site_label)
                    mapped_jose += 1

    print(f"  Jose: {mapped_jose:,}/{total_jose:,} residues mapped via SIFTS")

    # ── Source B: metal_binding (already UniProt) ─────────────────────────────
    print("  Loading metal_binding CSV …")
    mb = pd.read_csv(str(METAL_CSV), usecols=["UniProt_IDs_clean", "ResidueNo_UniProt", "Metal"],
                     low_memory=False)
    mb = mb.dropna(subset=["UniProt_IDs_clean", "ResidueNo_UniProt"])
    mb["ResidueNo_UniProt"] = pd.to_numeric(mb["ResidueNo_UniProt"], errors="coerce")
    mb = mb.dropna(subset=["ResidueNo_UniProt"])
    mb["ResidueNo_UniProt"] = mb["ResidueNo_UniProt"].astype(int)

    n_metal = 0
    for _, row in mb.iterrows():
        metal = str(row["Metal"]).strip().upper()
        label = f"loss_metal_{metal.lower()}"
        for uid in str(row["UniProt_IDs_clean"]).split(","):
            uid = uid.strip()
            if not uid:
                continue
            key = (uid, int(row["ResidueNo_UniProt"]))
            site_map[key].add(label)
            n_metal += 1
    print(f"  Metal binding: {n_metal:,} (UniProt, pos) entries added")

    # ── Cache ─────────────────────────────────────────────────────────────────
    rows = [{"uniprot": k[0], "pos": k[1], "site_types": ",".join(sorted(v))}
            for k, v in site_map.items()]
    pd.DataFrame(rows).to_parquet(str(SITE_MAP_CACHE), index=False)
    print(f"  Cached {len(site_map):,} entries → {SITE_MAP_CACHE}")
    return site_map


# ══════════════════════════════════════════════════════════════════════════════
# §2  Label variants
# ══════════════════════════════════════════════════════════════════════════════

def label_variants(complex_ids: np.ndarray, variant_1b: np.ndarray,
                   site_map: dict) -> list:
    """Return list of frozensets of site_type labels, aligned to complex_ids."""
    labels_out = []
    for cid, var in zip(complex_ids, variant_1b):
        uniprot = cid.split("_")[0]
        m = _VAR_RE.match(str(var))
        if m:
            pos = int(m.group(2))
            types = site_map.get((uniprot, pos), set())
        else:
            types = set()
        labels_out.append(frozenset(types))
    return labels_out


# ══════════════════════════════════════════════════════════════════════════════
# §3  Fisher's exact enrichment
# ══════════════════════════════════════════════════════════════════════════════

def _bh_correct(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    order = np.argsort(pvals)
    rank  = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1)
    adj = np.minimum(1.0, pvals * n / rank)
    # enforce monotonicity from the right
    for i in range(n - 2, -1, -1):
        adj[order[i]] = min(adj[order[i]], adj[order[i + 1]])
    return adj


def enrichment_analysis(cluster_ids: np.ndarray, var_site_labels: list,
                         all_site_types: list) -> pd.DataFrame:
    N = len(cluster_ids)
    rows = []

    # global counts per site type
    global_counts = {s: sum(1 for fs in var_site_labels if s in fs)
                     for s in all_site_types}

    for k in sorted(set(cluster_ids)):
        mask_k = cluster_ids == k
        n_k    = mask_k.sum()
        fs_k   = [var_site_labels[i] for i in range(N) if mask_k[i]]

        for s in all_site_types:
            n_ks = sum(1 for fs in fs_k if s in fs)
            N_s  = global_counts[s]
            if N_s == 0:
                continue
            # 2×2 contingency: [[n_ks, n_k-n_ks], [N_s-n_ks, N-n_k-(N_s-n_ks)]]
            a = n_ks
            b = n_k - n_ks
            c = N_s - n_ks
            d = N - n_k - c
            if d < 0:
                continue
            _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
            fe = (n_ks / n_k) / (N_s / N) if N_s > 0 and n_k > 0 else 0.0
            rows.append({"cluster": k, "site_type": s,
                         "n_k": n_k, "n_ks": n_ks, "N_s": N_s, "N": N,
                         "fold_enrichment": round(fe, 4), "pval": p})

    df = pd.DataFrame(rows)
    if len(df):
        df["adj_pval"] = _bh_correct(df["pval"].values)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# §4  Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def plot_heatmap(df: pd.DataFrame, out_path: Path):
    """Cluster × site_type heatmap of fold enrichment (masked if adj_p > 0.05)."""
    clusters   = sorted(df["cluster"].unique())
    site_types = [s for s in SITE_TYPE_ORDER if s in df["site_type"].values]

    # Build matrices
    fe_mat   = np.zeros((len(clusters), len(site_types)))
    sig_mat  = np.zeros_like(fe_mat, dtype=bool)

    for i, k in enumerate(clusters):
        for j, s in enumerate(site_types):
            row = df[(df["cluster"] == k) & (df["site_type"] == s)]
            if len(row):
                fe   = float(row["fold_enrichment"].iloc[0])
                ap   = float(row["adj_pval"].iloc[0])
                fe_mat[i, j]  = min(fe, 10.0)
                sig_mat[i, j] = ap < 0.05 and fe > 1.5

    # Mask non-significant cells
    display = np.where(sig_mat, fe_mat, 0.0)

    fig, ax = plt.subplots(figsize=(max(8, len(site_types) * 0.55),
                                    max(6, len(clusters) * 0.25)))
    im = ax.imshow(display, aspect="auto", cmap="YlOrRd", vmin=0, vmax=10)
    ax.set_xticks(range(len(site_types)))
    ax.set_xticklabels([s.replace("loss_", "") for s in site_types],
                       rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels([f"k={k}" for k in clusters], fontsize=7)
    ax.set_xlabel("Functional site type")
    ax.set_ylabel("Cluster")
    ax.set_title(f"{NAME} — Functional site enrichment (fold; masked if adj_p > 0.05)")
    plt.colorbar(im, ax=ax, label="Fold enrichment (capped at 10×)")
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap → {out_path}")


def print_cluster_summary(df: pd.DataFrame, cluster_ids: np.ndarray,
                           var_site_labels: list, cv_prots: np.ndarray,
                           focus_clusters=(0, 4, 8, 12, 14, 16, 27, 31, 32, 33, 35, 46)):
    sig = df[df["adj_pval"] < 0.05].copy()
    print("\n" + "=" * 70)
    print("Per-cluster functional site enrichment (adj_p < 0.05)")
    print("=" * 70)
    for k in focus_clusters:
        sub = sig[sig["cluster"] == k].sort_values("fold_enrichment", ascending=False)
        mask_k = cluster_ids == k
        n_k = mask_k.sum()
        prots = pd.Series(cv_prots[mask_k]).value_counts().head(3)
        prot_str = ", ".join(f"{p}({c})" for p, c in prots.items())
        print(f"\n  Cluster {k:2d}  n={n_k:,}  top_prots=[{prot_str}]")
        if len(sub) == 0:
            print("    (no significant site-type enrichments)")
        for _, row in sub.head(8).iterrows():
            print(f"    {row['site_type']:<30s}  FE={row['fold_enrichment']:.1f}×"
                  f"  n_ks={int(row['n_ks'])}  adj_p={row['adj_pval']:.2e}")


# ══════════════════════════════════════════════════════════════════════════════
# §5  Mutagenesis variant overlay
# ══════════════════════════════════════════════════════════════════════════════

def mutagenesis_overlay(site_map: dict, variant_1b: np.ndarray,
                         complex_ids: np.ndarray, cluster_ids: np.ndarray,
                         sifts_lookup: dict) -> pd.DataFrame:
    """Find jose mutagenesis variants that match existing ClinVar/HGMD variants."""
    if not JOSE_VAR.exists():
        print("  mutagenesis.variants not found — skipping overlay")
        return pd.DataFrame()

    print(f"  Loading mutagenesis.variants …")
    # Columns: PDB Chain ResNo WT_AA Mut_AA [extra] [UniProt_name] [ResNo_again] [description]
    rows_out = []
    with open(JOSE_VAR) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            pdb, chain, resno_str, wt_aa, mut_aa = parts[:5]
            desc = parts[-1] if len(parts) >= 8 else ""
            result = _pdb_to_uniprot(pdb.strip(), chain.strip(), resno_str.strip(), sifts_lookup)
            if result is None:
                continue
            uniprot, upos = result
            var_str = f"{wt_aa}{upos}{mut_aa}"
            # Check if this variant is in our ClinVar dataset
            for i, (cid, v) in enumerate(zip(complex_ids, variant_1b)):
                if cid.split("_")[0] == uniprot and v == var_str:
                    rows_out.append({
                        "cluster": cluster_ids[i],
                        "uniprot": uniprot,
                        "variant": var_str,
                        "consequence": desc,
                        "cluster_id": i,
                    })
                    break

    df = pd.DataFrame(rows_out)
    if len(df):
        print(f"  Matched {len(df)} mutagenesis variants to ClinVar")
        out_path = OUT / f"mutagenesis_overlay_{NAME}.csv"
        df.to_csv(str(out_path), index=False)
        print(f"  → {out_path}")
    else:
        print("  No mutagenesis variants matched to ClinVar")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print(f"Functional site analysis  model={NAME}", flush=True)

    # §1  Build site map
    site_map = build_site_map(rebuild=_args.rebuild_cache)

    # §2  Load variant data and cluster assignments
    print("\nLoading ClinVar data …")
    Z_cv, cv_labels, _ = load_clinvar_data(NAME)
    path_mask = cv_labels == 1
    Z_hg, _   = load_hgmd_gnomad(NAME)
    Z_disease  = sp.vstack([Z_cv[path_mask], Z_hg]).tocsr()

    print("Reconstructing variant keys …")
    cache_path  = LA / "cv_variant_keys.npz"
    complex_ids, variant_1b = reconstruct_clinvar_variant_keys(cache_path=cache_path)
    cv_prots    = np.array([c.split("_")[0] for c in complex_ids[path_mask]])

    # Only label ClinVar pathogenic variants (HGMD keys not available via reconstruct_clinvar_variant_keys)
    cv_path_complex = complex_ids[path_mask]
    cv_path_var1b   = variant_1b[path_mask]

    print("Re-running k-means …")
    _, cluster_ids_all, _ = run_disease_kmeans(Z_disease, verbose=True)
    cluster_ids_cv = cluster_ids_all[:Z_cv[path_mask].shape[0]]

    # §2  Label ClinVar pathogenic variants
    print("\nLabelling variants with functional site annotations …")
    var_site_labels = label_variants(cv_path_complex, cv_path_var1b, site_map)
    n_labelled = sum(1 for fs in var_site_labels if len(fs) > 0)
    print(f"  {n_labelled:,} / {len(var_site_labels):,} ClinVar pathogenic variants "
          f"have at least one functional site label "
          f"({100*n_labelled/len(var_site_labels):.1f}%)")

    # Site type coverage
    all_present = sorted({s for fs in var_site_labels for s in fs})
    print(f"  Site types with at least 1 variant: {len(all_present)}")

    # §3  Enrichment analysis
    print("\nRunning Fisher's exact enrichment tests …")
    df_enr = enrichment_analysis(cluster_ids_cv, var_site_labels, all_present)
    enr_path = OUT / f"functional_site_enrichment_{NAME}.csv"
    df_enr.to_csv(str(enr_path), index=False)
    print(f"  → {enr_path}  ({len(df_enr):,} rows)")

    # §4  Visualisation
    print("\nPlotting heatmap …")
    plot_heatmap(df_enr, OUT / f"functional_site_enrichment_heatmap_{NAME}.png")
    print_cluster_summary(df_enr, cluster_ids_cv, var_site_labels, cv_prots)

    # §5  Mutagenesis overlay (best-effort)
    print("\nMustagenesis variant overlay …")
    sifts_lookup = _build_sifts_lookup(SIFTS_CSV)
    mutagenesis_overlay(site_map, cv_path_var1b, cv_path_complex,
                        cluster_ids_cv, sifts_lookup)

    print(f"\nDone. Total time: {(time.time()-t0)/60:.1f} min")
    print(f"Outputs in {OUT}")


if __name__ == "__main__":
    main()
