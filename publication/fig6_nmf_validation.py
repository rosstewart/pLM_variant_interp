"""Fig 6: NMF vs k-means cluster cosine similarity validation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from plot_utils import (
    set_style, save_fig, VAL,
    get_loog_cluster_map, gene_label,
)

set_style()

nmf_km = pd.read_csv(VAL / "nmf_vs_kmeans_cosine_concat_ef1_k128.csv")
# columns: nmf_component, best_kmeans_cluster, cosine_similarity
print("nmf_vs_kmeans columns:", nmf_km.columns.tolist())
print(nmf_km.head(3).to_string())

nmf_top = pd.read_csv(VAL / "nmf_top_latents_concat_ef1_k128.csv")
print("nmf_top_latents columns:", nmf_top.columns.tolist())

loog_map = get_loog_cluster_map()
cluster_to_gene = {}
for uniprot, clus in loog_map.items():
    if clus not in cluster_to_gene:
        cluster_to_gene[clus] = gene_label(uniprot)

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

# ── Panel A: NMF component → best k-means cluster (lollipop sorted by cosine) ─
ax_a.set_title("A  NMF component → best k-means cluster\n(cosine similarity)",
               loc="left", fontweight="bold")

# Sort by cosine similarity descending, show top 20
top20 = nmf_km.sort_values("cosine_similarity", ascending=False).head(20).reset_index(drop=True)

cmap_vals = cm.Blues(np.linspace(0.4, 1.0, len(top20)))

ax_a.barh(range(len(top20)), top20["cosine_similarity"],
          color=cmap_vals, alpha=0.9, height=0.7)
ax_a.axvline(0.5, color="#E74C3C", lw=0.8, ls="--", label="cosine = 0.5")

ax_a.set_yticks(range(len(top20)))
labels = []
for _, row in top20.iterrows():
    kc  = int(row["best_kmeans_cluster"])
    gene = cluster_to_gene.get(kc, "")
    lbl  = f"NMF-{int(row['nmf_component'])} → C{kc} {gene}"
    labels.append(lbl)
ax_a.set_yticklabels(labels, fontsize=5)
ax_a.set_xlabel("Cosine similarity", fontsize=7)
ax_a.set_xlim(0, 1.05)

for i, (_, row) in enumerate(top20.iterrows()):
    ax_a.text(row["cosine_similarity"] + 0.01, i,
              f"{row['cosine_similarity']:.3f}", va="center", fontsize=4.5)

ax_a.legend(fontsize=6, frameon=False)
ax_a.set_title("A  NMF component → best k-means cluster",
               loc="left", fontweight="bold")

# ── Panel B: Bar chart — max cosine similarity per NMF component ──────────────
ax_b.set_title("B  NMF component purity (max cosine sim to any k-means cluster)",
               loc="left", fontweight="bold")

nmf_sorted = nmf_km.sort_values("cosine_similarity", ascending=False).reset_index(drop=True)

n_above = (nmf_sorted["cosine_similarity"] >= 0.5).sum()
total   = len(nmf_sorted)

colors = [cm.Blues(0.8) if v >= 0.5 else cm.Blues(0.35)
          for v in nmf_sorted["cosine_similarity"]]
ax_b.bar(range(len(nmf_sorted)), nmf_sorted["cosine_similarity"],
         color=colors, alpha=0.9, width=0.85)
ax_b.axhline(0.5, color="#E74C3C", lw=0.8, ls="--")

ax_b.set_xlabel("NMF component (sorted by cosine)", fontsize=7)
ax_b.set_ylabel("Max cosine similarity\nto best k-means cluster", fontsize=7)
ax_b.set_xlim(-0.5, len(nmf_sorted) - 0.5)
ax_b.set_ylim(0, 1.05)
ax_b.set_xticks([])

ax_b.text(0.97, 0.95, f"{n_above}/{total} ≥ 0.50",
          transform=ax_b.transAxes, ha="right", fontsize=8, fontweight="bold",
          color="#E74C3C")
ax_b.text(0.97, 0.88, "NMF components with strong\nk-means correspondence",
          transform=ax_b.transAxes, ha="right", fontsize=6, color="#555555")

save_fig(fig, "fig6_nmf_validation", formats=("pdf",))
print("Done: fig6_nmf_validation.pdf")
