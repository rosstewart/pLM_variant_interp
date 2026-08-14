"""Fig 1: SAE overview — architecture schematic, UMAP, probe profile bar chart."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from plot_utils import (
    set_style, save_fig, VAL, UMAP_PNG,
    PROBE_COLORS, PROBE_LABELS, PROBE_BASELINES,
    UNIPROT_TO_GENE, get_loog_cluster_map, gene_label,
)

set_style()

fig = plt.figure(figsize=(10, 8), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
ax_arch = fig.add_subplot(gs[0, 0])
ax_umap = fig.add_subplot(gs[0, 1])
ax_bar  = fig.add_subplot(gs[1, :])

# ── Panel A: Architecture schematic ──────────────────────────────────────────
ax_arch.set_xlim(0, 1)
ax_arch.set_ylim(0, 1)
ax_arch.axis("off")
ax_arch.set_title("A  SAE architecture", loc="left", fontweight="bold")

boxes = [
    (0.50, 0.93, "WT ProtT5\n(1024-dim)",     "#AED6F1"),
    (0.50, 0.80, "VT ProtT5\n(1024-dim)",     "#AED6F1"),
    (0.50, 0.66, "Concatenate\n[WT; VT]\n(2048-dim)", "#D5E8D4"),
    (0.50, 0.50, "Linear encoder\n(2048→2048)", "#FFF2CC"),
    (0.50, 0.36, "TopK sparsifier\n(K=128 active)", "#FFE6CC"),
    (0.50, 0.22, "Z sparse\n(2048-dim)",       "#F8CECC"),
    (0.50, 0.08, "Reconstruction\n(Linear decoder)", "#E1D5E7"),
]
bw, bh = 0.46, 0.09
for cx, cy, label, color in boxes:
    rect = FancyBboxPatch((cx - bw/2, cy - bh/2), bw, bh,
                          boxstyle="round,pad=0.01", linewidth=0.6,
                          edgecolor="#555555", facecolor=color)
    ax_arch.add_patch(rect)
    ax_arch.text(cx, cy, label, ha="center", va="center", fontsize=5.5,
                 multialignment="center")

# Arrows between boxes
for (_, y0, __, ___), (_, y1, __, ____) in zip(boxes[:-1], boxes[1:]):
    ax_arch.annotate("", xy=(0.50, y1 + bh/2 + 0.005),
                     xytext=(0.50, y0 - bh/2 - 0.005),
                     arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.7))

# Arrow from WT + VT boxes (merge annotation)
ax_arch.annotate("", xy=(0.50, boxes[2][1] + bh/2 + 0.005),
                 xytext=(0.50, boxes[1][1] - bh/2 - 0.005),
                 arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.7))
ax_arch.plot([0.50, 0.50], [boxes[0][1] - bh/2, boxes[1][1] + bh/2],
             lw=0.7, color="#555555")

# Annotation on TopK→Z arrow
ax_arch.text(0.52, (boxes[4][1] + boxes[5][1]) / 2, "K=128/2048 active",
             fontsize=4.5, color="#AA5500", va="center")

# Caption
ax_arch.text(0.50, -0.03,
             "Trained on 901k ClinVar / gnomAD / HGMD variants",
             ha="center", fontsize=5, color="#555555",
             transform=ax_arch.transAxes)

# ── Panel B: UMAP ─────────────────────────────────────────────────────────────
ax_umap.set_title("B  Disease variant clustering (186k variants, k=50)",
                  loc="left", fontweight="bold")
if UMAP_PNG.exists():
    img = plt.imread(str(UMAP_PNG))
    ax_umap.imshow(img, aspect="auto")
else:
    ax_umap.text(0.5, 0.5, f"UMAP PNG not found:\n{UMAP_PNG}",
                 ha="center", va="center", transform=ax_umap.transAxes, fontsize=6)
ax_umap.axis("off")

# ── Panel C: Probe score bar chart ───────────────────────────────────────────
ax_bar.set_title("C  Probe score profile for 6 key disease clusters",
                 loc="left", fontweight="bold")

probe_df = pd.read_csv(VAL / "probe_distribution_stats.csv")
loog_map = get_loog_cluster_map()  # {uniprot: cluster_id}

# UniProt IDs for focus genes (first occurrence per gene)
focus = {
    "P04637": "TP53",
    "P60484": "PTEN",
    "P02452": "COL1A1",
    "P38398": "BRCA1",
    "P60709": "ACTB",
    "P01130": "LDLR",
}

tasks = ["destab_vs_neut", "stab_vs_neut", "gof_vs_wt", "lof_vs_wt"]
n_genes = len(focus)
n_tasks = len(tasks)
group_w = 0.8
bar_w   = group_w / n_tasks
gene_order = list(focus.keys())

for gi, uniprot in enumerate(gene_order):
    if uniprot not in loog_map:
        continue
    cluster = loog_map[uniprot]
    sub = probe_df[probe_df["cluster"] == cluster]
    gene = focus[uniprot]
    for ti, task in enumerate(tasks):
        row = sub[sub["task"] == task]
        if row.empty:
            continue
        val = row["mean_in"].iloc[0]
        x = gi + (ti - n_tasks/2 + 0.5) * bar_w
        ax_bar.bar(x, val, width=bar_w * 0.85,
                   color=PROBE_COLORS[task], alpha=0.85, linewidth=0)

# Baselines
for task, baseline in PROBE_BASELINES.items():
    ax_bar.axhline(baseline, color=PROBE_COLORS[task], lw=0.7,
                   ls="--", alpha=0.5)

ax_bar.set_xticks(range(n_genes))
ax_bar.set_xticklabels([focus[u] for u in gene_order], fontsize=7)
ax_bar.set_ylabel("Mean probe score (in-cluster)", fontsize=7)
ax_bar.set_ylim(0, 1)

legend_patches = [mpatches.Patch(color=PROBE_COLORS[t], label=PROBE_LABELS[t])
                  for t in tasks]
ax_bar.legend(handles=legend_patches, ncol=4, fontsize=6,
              loc="upper right", frameon=False)
ax_bar.text(0.01, 0.97, "Dashed lines = genome-wide baselines",
            transform=ax_bar.transAxes, fontsize=5, va="top", color="#777777")

save_fig(fig, "fig1_overview", formats=("pdf", "png"))
print("Done: fig1_overview.pdf / .png")
