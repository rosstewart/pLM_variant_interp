"""Fig 5: Within-family latent specificity — 3 panels."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from plot_utils import (
    set_style, save_fig, VAL, WF,
    PROBE_COLORS, GENE_COLORS,
    UNIPROT_TO_GENE, get_loog_cluster_map, gene_label,
)

set_style()

fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
ax_a, ax_b, ax_c = axes

# ── Panel A: Within-family enrichment bar chart ───────────────────────────────
ax_a.set_title("A  Within-family pathogenic discrimination\n(ClinVar benign enrichment)",
               loc="left", fontweight="bold", fontsize=6)

wf = pd.read_csv(WF / "within_family_fire_rates.csv")
# Handle inf strings (already float in this file, but guard anyway)
wf["cvben_fisher_or"] = pd.to_numeric(wf["cvben_fisher_or"], errors="coerce").fillna(1e6)
wf["cvben_fisher_p"]  = pd.to_numeric(wf["cvben_fisher_p"],  errors="coerce").fillna(1.0)

# Best latent per (cluster, dominant_prot): lowest cvben_fisher_p
best = (wf.sort_values("cvben_fisher_p")
          .groupby(["cluster", "dominant_prot"])
          .first()
          .reset_index())

best["gene"] = best["dominant_prot"].map(UNIPROT_TO_GENE).fillna(best["dominant_prot"])
# -log10 p, cap at 50
eps = 1e-300
best["neglog_p"] = -np.log10(best["cvben_fisher_p"].clip(lower=eps)).clip(upper=50)

best = best.sort_values("neglog_p", ascending=False).reset_index(drop=True)

colors = [GENE_COLORS.get(g, "#888888") for g in best["gene"]]
ax_a.bar(range(len(best)), best["neglog_p"], color=colors, alpha=0.85, width=0.7)
ax_a.axhline(-np.log10(0.05), color="#E74C3C", lw=0.8, ls="--",
             label="p=0.05 threshold")

ax_a.set_xticks(range(len(best)))
ax_a.set_xticklabels(best["gene"], rotation=45, ha="right", fontsize=5.5)
ax_a.set_ylabel("−log₁₀(p) ClinVar benign enrichment", fontsize=6)
ax_a.set_ylim(0, 55)

for i, row in best.iterrows():
    if row["neglog_p"] >= 49.9:
        ax_a.text(i, 51, "p=0\n(∞)", ha="center", fontsize=4, color="#333333")
    n = row["n_cvben_same"]
    ax_a.text(i, row["neglog_p"] + 0.8, str(int(n)), ha="center", fontsize=3.5,
              color="#555555")

ax_a.legend(fontsize=5, frameon=False)

# ── Panel B: TP53 within-protein latent specificity ──────────────────────────
ax_b.set_title("B  TP53 within-protein latent specificity\n(own cluster vs others)",
               loc="left", fontweight="bold", fontsize=6)

tp53 = pd.read_csv(WF / "tp53_within_protein_specificity.csv")
# columns: latent, own_cluster, fr_own, mean_fr_others, within_protein_specificity

key_latents_tp53 = {414, 1994, 2001, 1494, 97}
cluster_colors = {c: plt.cm.Set1(i/8)
                  for i, c in enumerate(sorted(tp53["own_cluster"].unique()))}

for _, row in tp53.iterrows():
    c = row["own_cluster"]
    color = cluster_colors.get(c, "#AAAAAA")
    size  = 25 if int(row["latent"]) in key_latents_tp53 else 8
    ax_b.scatter(row["mean_fr_others"], row["fr_own"],
                 color=color, s=size, alpha=0.8,
                 linewidths=0.5 if size > 10 else 0,
                 edgecolors="black" if size > 10 else "none",
                 zorder=3 if size > 10 else 2)
    if int(row["latent"]) in key_latents_tp53:
        ax_b.annotate(f"L{int(row['latent'])}",
                      xy=(row["mean_fr_others"], row["fr_own"]),
                      xytext=(row["mean_fr_others"] + 0.02, row["fr_own"] + 0.02),
                      fontsize=4.5, color="#333333",
                      arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.5))

ax_b.set_xlabel("Mean fire rate (other TP53 clusters)", fontsize=6)
ax_b.set_ylabel("Fire rate (own TP53 cluster)", fontsize=6)
ax_b.set_xlim(-0.05, 1.05)
ax_b.set_ylim(-0.05, 1.05)

# Perfect discriminator region annotation
ax_b.text(0.05, 0.92, "Perfect discriminators\n← top-left", fontsize=4.5,
          transform=ax_b.transAxes, color="#777777")

patches = [mpatches.Patch(color=col, label=f"Cluster {c}")
           for c, col in cluster_colors.items()]
ax_b.legend(handles=patches, fontsize=4, frameon=False, loc="lower right")

# ── Panel C: PTEN cross-cluster fire rates ───────────────────────────────────
ax_c.set_title("C  PTEN cross-cluster latent fire rates\n(3 PTEN clusters)",
               loc="left", fontweight="bold", fontsize=6)

pten = pd.read_csv(WF / "pten_cross_cluster_fire_rates.csv")
# columns: latent, source_clusters, fr_k0, n_k0, fr_k31, n_k31, fr_k33, n_k33

cluster_cols = [c for c in pten.columns if c.startswith("fr_k")]
cluster_names = [c.replace("fr_", "") for c in cluster_cols]

# Scatter: for each latent, plot fr_k0 vs fr_k31 (two biggest PTEN clusters)
if "fr_k0" in pten.columns and "fr_k31" in pten.columns:
    ax_c.scatter(pten["fr_k0"], pten["fr_k31"],
                 c=pten["fr_k33"] if "fr_k33" in pten.columns else "#3498DB",
                 cmap="RdYlGn", s=15, alpha=0.8, vmin=0, vmax=1,
                 edgecolors="none")

    # Label latent 1138
    for _, row in pten.iterrows():
        if row["latent"] == 1138:
            ax_c.annotate(f"L1138\n(146,230×)",
                          xy=(row["fr_k0"], row["fr_k31"]),
                          xytext=(row["fr_k0"] + 0.03, row["fr_k31"] - 0.08),
                          fontsize=4.5, color="#2E86C1",
                          arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.5))

    ax_c.set_xlabel("Fire rate — cluster k0", fontsize=6)
    ax_c.set_ylabel("Fire rate — cluster k31", fontsize=6)
    ax_c.set_xlim(-0.05, 1.05)
    ax_c.set_ylim(-0.05, 1.05)
    ax_c.text(0.5, -0.12,
              "Color = fr_k33; PTEN latents fire in k0 but not k31/k33",
              transform=ax_c.transAxes, ha="center", fontsize=4.5, color="#777777")
else:
    # Fallback: heatmap of latents × clusters
    mat = pten[cluster_cols].values
    im = ax_c.imshow(mat[:20].T, aspect="auto", cmap="RdYlGn",
                     vmin=0, vmax=1, interpolation="nearest")
    ax_c.set_xticks(range(min(20, len(pten))))
    ax_c.set_xticklabels(pten["latent"].values[:20], rotation=90, fontsize=4)
    ax_c.set_yticks(range(len(cluster_cols)))
    ax_c.set_yticklabels(cluster_names, fontsize=5)
    ax_c.set_xlabel("Latent", fontsize=6)
    ax_c.set_ylabel("PTEN cluster", fontsize=6)

save_fig(fig, "fig5_within_family", formats=("pdf",))
print("Done: fig5_within_family.pdf")
