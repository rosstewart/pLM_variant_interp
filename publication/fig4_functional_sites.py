"""Fig 4: Functional site enrichment heatmap + TP53 Zn-binding detail."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from plot_utils import (
    set_style, save_fig, VAL,
    SITE_LABELS, get_loog_cluster_map, gene_label,
)

set_style()

df = pd.read_csv(VAL / "functional_site_enrichment_concat_ef1_k128.csv")
loog_map = get_loog_cluster_map()
cluster_to_gene = {}
for uniprot, clus in loog_map.items():
    if clus not in cluster_to_gene:
        cluster_to_gene[clus] = gene_label(uniprot)

# Keep only site types in SITE_LABELS and with ≥1 significant hit
df = df[df["site_type"].isin(SITE_LABELS)]
sig = df[df["adj_pval"] <= 0.05]
if sig.empty:
    keep_sites = list(SITE_LABELS.keys())
else:
    keep_sites = sorted(sig["site_type"].unique(),
                        key=lambda s: list(SITE_LABELS.keys()).index(s)
                        if s in SITE_LABELS else 999)
df = df[df["site_type"].isin(keep_sites)]

all_clusters = sorted(df["cluster"].unique())
col_order    = keep_sites

# Build enrichment and adj_pval matrices
fe_mat  = np.zeros((len(all_clusters), len(col_order)))
sig_mat = np.zeros_like(fe_mat, dtype=bool)

for ri, clus in enumerate(all_clusters):
    sub = df[df["cluster"] == clus]
    for ci, site in enumerate(col_order):
        row = sub[sub["site_type"] == site]
        if not row.empty:
            fe_mat[ri, ci]  = row["fold_enrichment"].iloc[0]
            sig_mat[ri, ci] = row["adj_pval"].iloc[0] <= 0.05

log_fe = np.log2(fe_mat + 0.01)
log_fe = np.clip(log_fe, 0, 6)
masked = np.where(sig_mat, log_fe, np.nan)

fig = plt.figure(figsize=(12, 8), constrained_layout=True)
gs  = fig.add_gridspec(1, 2, width_ratios=[3, 1])
ax_heat = fig.add_subplot(gs[0])
ax_bar  = fig.add_subplot(gs[1])

# ── Panel A: Heatmap ─────────────────────────────────────────────────────────
ax_heat.set_title("A  Functional site enrichment — all clusters",
                  loc="left", fontweight="bold")

cmap = plt.cm.YlOrRd.copy()
cmap.set_bad("#F0F0F0")

im = ax_heat.imshow(masked, aspect="auto", cmap=cmap,
                    vmin=0, vmax=6, interpolation="nearest")

# Annotate cells with significant high enrichment
for ri in range(len(all_clusters)):
    for ci in range(len(col_order)):
        if sig_mat[ri, ci] and fe_mat[ri, ci] >= 3:
            txt = f"{int(round(fe_mat[ri, ci]))}"
            color = "white" if log_fe[ri, ci] > 4 else "black"
            ax_heat.text(ci, ri, txt, ha="center", va="center",
                         fontsize=3.5, color=color, fontweight="bold")

ax_heat.set_xticks(range(len(col_order)))
ax_heat.set_xticklabels([SITE_LABELS.get(s, s) for s in col_order],
                         rotation=45, ha="right", fontsize=5)
ax_heat.set_yticks(range(len(all_clusters)))
ax_heat.set_yticklabels([f"C{c} {cluster_to_gene.get(c, '')}"
                          for c in all_clusters], fontsize=4)

cb = fig.colorbar(im, ax=ax_heat, shrink=0.4, pad=0.01)
cb.set_label("log₂(fold enrichment)", fontsize=6)
cb.ax.tick_params(labelsize=5)

# Highlight row markers for top enrichment clusters
# Find clusters with very high enrichment for any metal
metal_sites = [s for s in col_order if "metal" in s]
if metal_sites:
    metal_idx = [col_order.index(s) for s in metal_sites if s in col_order]
    for ri, clus in enumerate(all_clusters):
        if any(sig_mat[ri, ci] and fe_mat[ri, ci] >= 10 for ci in metal_idx):
            ax_heat.add_patch(mpatches.Rectangle(
                (-0.5 - len(col_order)*0.05, ri - 0.5), len(col_order)*0.05, 1,
                color="#E74C3C", clip_on=False, lw=0))

# ── Panel B: Top-Zn cluster detail (BRCA1 RING domain) ───────────────────────

# Find cluster with max Zn enrichment (adj_pval ≤ 0.05)
zn_rows = df[(df["site_type"] == "loss_metal_zn") & (df["adj_pval"] <= 0.05)]
if not zn_rows.empty:
    zn_cluster = zn_rows.loc[zn_rows["fold_enrichment"].idxmax(), "cluster"]
    max_zn_fe  = zn_rows.loc[zn_rows["fold_enrichment"].idxmax(), "fold_enrichment"]
else:
    zn_cluster = all_clusters[0]
    max_zn_fe  = 0.0

sub = df[(df["cluster"] == zn_cluster) & (df["adj_pval"] <= 0.05)].copy()
sub = sub.nlargest(10, "fold_enrichment")

# Color by site category
def site_color(site):
    if "metal" in site:   return "#F1C40F"
    if "phospho" in site: return "#8E44AD"
    if "nucleotide" in site or "atp" in site or "nad" in site or "fad" in site:
        return "#16A085"
    return "#95A5A6"

colors = [site_color(s) for s in sub["site_type"]]
ax_bar.barh(range(len(sub)), sub["fold_enrichment"], color=colors, height=0.7, alpha=0.85)
ax_bar.axvline(1, color="#333333", lw=0.8, ls="--")
ax_bar.set_yticks(range(len(sub)))
ax_bar.set_yticklabels([SITE_LABELS.get(s, s) for s in sub["site_type"]], fontsize=5.5)
ax_bar.set_xlabel("Fold enrichment", fontsize=6)
gene_name = cluster_to_gene.get(zn_cluster, str(zn_cluster))
# Annotate with domain context: BRCA1 cluster 8 = RING domain Zn fingers
domain_note = "RING domain" if gene_name == "BRCA1" else "Zn finger"
ax_bar.set_title(f"B  C{zn_cluster} ({gene_name} {domain_note}) — Zn²⁺ {max_zn_fe:.0f}×",
                 loc="left", fontweight="bold", fontsize=6)

leg_patches = [
    mpatches.Patch(color="#F1C40F", label="Metal"),
    mpatches.Patch(color="#8E44AD", label="Phospho"),
    mpatches.Patch(color="#16A085", label="Nucleotide"),
    mpatches.Patch(color="#95A5A6", label="Other"),
]
ax_bar.legend(handles=leg_patches, fontsize=5, frameon=False, loc="lower right")

save_fig(fig, "fig4_functional_sites", formats=("pdf",))
print("Done: fig4_functional_sites.pdf")
