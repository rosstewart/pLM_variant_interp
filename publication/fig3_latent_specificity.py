"""Fig 3: Latent specificity — fire rate scatter + causal GoF bar chart."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from plot_utils import (
    set_style, save_fig, VAL,
    PROBE_COLORS, UNIPROT_TO_GENE, GENE_COLORS,
    get_loog_cluster_map, gene_label,
)

set_style()

spec_df = pd.read_csv(VAL / "latent_specificity.csv")
loog_map = get_loog_cluster_map()  # {uniprot: cluster_id}

# Reverse map: cluster_id → gene name (first gene per cluster)
cluster_to_gene = {}
for uniprot, clus in loog_map.items():
    if clus not in cluster_to_gene:
        cluster_to_gene[clus] = gene_label(uniprot)

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

# ── Panel A: fire_in vs fire_out scatter (log10 scale) ───────────────────────
ax_a.set_title("A  Latent fire rate specificity", loc="left", fontweight="bold")

fire_in  = spec_df["fire_in"].clip(lower=1e-7)
fire_out = spec_df["fire_out"].clip(lower=1e-7)

ax_a.scatter(np.log10(fire_out), np.log10(fire_in),
             s=1, color="#CCCCCC", alpha=0.4, linewidths=0, rasterized=True)

# Top-3 latents per cluster by specificity_ratio, colored by gene
top3 = (spec_df
        .groupby("cluster")
        .apply(lambda g: g.nlargest(3, "specificity_ratio"))
        .reset_index(drop=True))

plotted_genes = set()
for _, row in top3.iterrows():
    gene = cluster_to_gene.get(row["cluster"], None)
    color = GENE_COLORS.get(gene, "#888888") if gene else "#888888"
    label = gene if gene and gene not in plotted_genes else None
    if label:
        plotted_genes.add(gene)
    ax_a.scatter(np.log10(max(row["fire_out"], 1e-7)),
                 np.log10(max(row["fire_in"], 1e-7)),
                 s=10, color=color, zorder=3, label=label, linewidths=0.3,
                 edgecolors="white")

# Identity line
lims = (-7, 0.5)
ax_a.plot(lims, lims, "--", color="#999999", lw=0.8, zorder=1)

# Specificity band lines (100×, 1000×, 10000×)
for fold, ls in [(100, ":"), (1000, "--"), (10000, "-.")]:
    xs = np.array(lims)
    ys = xs + np.log10(fold)
    ax_a.plot(xs, np.clip(ys, *lims), color="#AAAAAA", lw=0.6, ls=ls)
    ax_a.text(lims[0] + 0.2, lims[0] + np.log10(fold) + 0.1,
              f"{fold}×", fontsize=4, color="#999999")

# Label key latents
key_latents = {
    1138: ("PTEN\nL1138",  "#3498DB"),
    1799: ("COL1A1\nL1799", "#27AE60"),
    1494: ("TP53\nL1494",  "#E74C3C"),
}
for _, row in spec_df.iterrows():
    if row["latent"] in key_latents:
        lbl, col = key_latents[row["latent"]]
        xp = np.log10(max(row["fire_out"], 1e-7))
        yp = np.log10(max(row["fire_in"],  1e-7))
        ax_a.annotate(lbl, xy=(xp, yp), xytext=(xp + 0.5, yp + 0.4),
                      fontsize=5, color=col,
                      arrowprops=dict(arrowstyle="-", color=col, lw=0.6))

ax_a.set_xlim(*lims)
ax_a.set_ylim(*lims)
ax_a.set_xlabel("log₁₀(fire rate out-of-cluster)", fontsize=7)
ax_a.set_ylabel("log₁₀(fire rate in-cluster)", fontsize=7)
ax_a.text(0.97, 0.03, "Specificity = fire_in / fire_out",
          transform=ax_a.transAxes, fontsize=5, ha="right", color="#666666")
ax_a.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
ax_a.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
ax_a.grid(which="minor", color="#EEEEEE", lw=0.3)

handles = [mpatches.Patch(color=GENE_COLORS.get(g, "#888888"), label=g)
           for g in sorted(plotted_genes) if g in GENE_COLORS]
ax_a.legend(handles=handles, fontsize=5, ncol=2, frameon=False,
            loc="upper left")

# ── Panel B: Causal GoF bar chart ────────────────────────────────────────────
ax_b.set_title("B  Causal GoF effect — top latent per gene cluster",
               loc="left", fontweight="bold")

# Filter to genes we know about
spec_df["gene"] = spec_df["cluster"].map(cluster_to_gene)
known = spec_df[spec_df["gene"].isin(UNIPROT_TO_GENE.values())].copy()

# Top latent per gene (by specificity_ratio)
top1 = (known.groupby("gene")
        .apply(lambda g: g.nlargest(1, "specificity_ratio"))
        .reset_index(drop=True))

top1 = top1.sort_values("causal_spec_gof_vs_wt", ascending=True)

colors = [PROBE_COLORS["gof_vs_wt"] if v >= 0 else PROBE_COLORS["lof_vs_wt"]
          for v in top1["causal_spec_gof_vs_wt"]]

ax_b.barh(range(len(top1)), top1["causal_spec_gof_vs_wt"],
          color=colors, alpha=0.85, height=0.7)
ax_b.axvline(0, color="#333333", lw=0.8)

ax_b.set_yticks(range(len(top1)))
ax_b.set_yticklabels([f"L{int(r.latent)} {r.gene}"
                       for _, r in top1.iterrows()], fontsize=6)
ax_b.set_xlabel("Causal GoF score", fontsize=7)

for i, (_, row) in enumerate(top1.iterrows()):
    v = row["causal_spec_gof_vs_wt"]
    xoff = 0.02 if v >= 0 else -0.02
    ha   = "left" if v >= 0 else "right"
    ax_b.text(v + xoff, i, f"{v:+.2f}", va="center", fontsize=4.5, ha=ha)

ax_b.text(0.5, -0.12,
          "Causal GoF score: effect of injecting latent on GoF probe",
          transform=ax_b.transAxes, ha="center", fontsize=5, color="#666666")

save_fig(fig, "fig3_latent_specificity", formats=("pdf", "png"))
print("Done: fig3_latent_specificity.pdf / .png")
