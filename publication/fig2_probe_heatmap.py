"""Fig 2: Probe score heatmap — all 50 clusters × 4 probe tasks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from plot_utils import (
    set_style, save_fig, VAL,
    PROBE_BASELINES, PROBE_LABELS,
    UNIPROT_TO_GENE, get_loog_cluster_map, gene_label,
)

set_style()

probe_df = pd.read_csv(VAL / "probe_distribution_stats.csv")
loog_df  = pd.read_csv(VAL / "leave_one_gene_out.csv")
loog_map = get_loog_cluster_map()  # {uniprot: cluster_id}

tasks = ["destab_vs_neut", "stab_vs_neut", "gof_vs_wt", "lof_vs_wt"]
col_labels = ["Destabilising", "Stabilising", "GoF", "LoF"]

# Build cluster → gene label (take first gene per cluster)
cluster_to_gene = {}
for _, row in loog_df.iterrows():
    c = row["cluster"]
    g = gene_label(row["removed_gene"])
    if c not in cluster_to_gene:
        cluster_to_gene[c] = g

all_clusters = sorted(probe_df["cluster"].unique())

# Matrix of mean_in; matrix of ks_p
mean_mat = np.full((len(all_clusters), len(tasks)), np.nan)
sig_mat  = np.zeros((len(all_clusters), len(tasks)), dtype=bool)

for ci, clus in enumerate(all_clusters):
    sub = probe_df[probe_df["cluster"] == clus]
    for ti, task in enumerate(tasks):
        row = sub[sub["task"] == task]
        if not row.empty:
            mean_mat[ci, ti] = row["mean_in"].iloc[0]
            sig_mat[ci, ti]  = row["ks_p"].iloc[0] < 0.01

baselines = np.array([PROBE_BASELINES[t] for t in tasks])

# Standardised score = (mean_in - baseline) / baseline
std_mat = (mean_mat - baselines) / baselines

# Sort rows by destab column (ascending — most suppressed first)
sort_idx = np.argsort(std_mat[:, 0])

mean_sorted = mean_mat[sort_idx]
std_sorted  = std_mat[sort_idx]
sig_sorted  = sig_mat[sort_idx]
cluster_sorted = [all_clusters[i] for i in sort_idx]

# Row labels
row_labels = [f"C{c} {cluster_to_gene.get(c, '')}" for c in cluster_sorted]

fig, ax = plt.subplots(figsize=(6, 7), constrained_layout=True)

vmax = np.nanmax(np.abs(std_sorted)) * 0.9
im = ax.imshow(std_sorted, aspect="auto", cmap="RdYlBu_r",
               vmin=-vmax, vmax=vmax,
               interpolation="nearest")

# Annotate cells
for ri in range(len(cluster_sorted)):
    for ci in range(len(tasks)):
        val = mean_sorted[ri, ci]
        if np.isnan(val):
            continue
        txt = f"{val:.2f}"
        if sig_sorted[ri, ci]:
            txt += "*"
        color = "white" if abs(std_sorted[ri, ci]) > vmax * 0.65 else "black"
        ax.text(ci, ri, txt, ha="center", va="center",
                fontsize=3.5, color=color)

ax.set_xticks(range(len(tasks)))
ax.set_xticklabels(col_labels, fontsize=7)
ax.set_yticks(range(len(cluster_sorted)))
ax.set_yticklabels(row_labels, fontsize=4)
ax.set_title("Probe score profile — all clusters\n(standardised vs baseline; * = KS p<0.01)",
             fontsize=7)

# Colorbar
cb = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
cb.set_label("(mean_in − baseline) / baseline", fontsize=6)
cb.ax.tick_params(labelsize=5)

# Colour notable y-tick labels
ytick_labels = ax.get_yticklabels()
for ri, (lbl, c) in enumerate(zip(ytick_labels, cluster_sorted)):
    if c in {41, 45}:
        lbl.set_color("#C0392B")   # red — strongest destab+GoF
    elif c == 16:
        lbl.set_color("#16A085")   # teal — highest LoF (ACTB)

# Inline row annotations (right of heatmap)
for ri, c in enumerate(cluster_sorted):
    if c == 16:
        ax.text(len(tasks) + 0.15, ri, "↑ LoF",
                va="center", fontsize=3, color="#16A085",
                clip_on=False)
    elif c in {41, 45}:
        ax.text(len(tasks) + 0.15, ri, "↑ Destab+GoF",
                va="center", fontsize=3, color="#C0392B",
                clip_on=False)

# Caption note for unknown-gene clusters
ax.text(0.5, -0.04,
        "C41, C45 (red): strongest destab+GoF signal — dominant gene not in LOOG set",
        transform=ax.transAxes, ha="center", fontsize=4.5,
        color="#C0392B", style="italic")

save_fig(fig, "fig2_probe_heatmap", formats=("pdf",))
print("Done: fig2_probe_heatmap.pdf")
