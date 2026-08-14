"""
nmf_analysis.py

NMF decomposition of disease-variant SAE activations as a complement to cosine k-means.
Non-negative Matrix Factorisation is well-matched to TopK SAE activations:
  - activations are non-negative by construction (ReLU before TopK)
  - NMF basis vectors are interpretable as additive mechanism prototypes
  - component activation profiles can be compared to k-means centroids

Outputs → /data/ross/interp/latent_analysis/validation/
"""

import sys, time, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared_infrastructure import (
    LA, DEFAULT_NAME, N_CLUSTERS, RANDOM_SEED,
    load_clinvar_data, load_hgmd_gnomad, run_disease_kmeans,
)

_ap = argparse.ArgumentParser(description="NMF vs k-means comparison")
_ap.add_argument("--name",         default=DEFAULT_NAME)
_ap.add_argument("--n-components", type=int, default=50)
_ap.add_argument("--max-iter",     type=int, default=500)
_args, _ = _ap.parse_known_args()

NAME         = _args.name
N_COMPONENTS = _args.n_components
MAX_ITER     = _args.max_iter
OUT          = LA / "validation"
OUT.mkdir(parents=True, exist_ok=True)


def load_disease_matrix(name: str):
    """Return (Z_disease_csr, n_cv_path) — rows: ClinVar pathogenic then HGMD."""
    Z_cv, cv_labels, _ = load_clinvar_data(name)
    path_mask = cv_labels == 1
    Z_hg, _   = load_hgmd_gnomad(name)
    Z_disease  = sp.vstack([Z_cv[path_mask], Z_hg]).tocsr()
    return Z_disease, int(path_mask.sum())


def run_nmf(Z: sp.csr_matrix) -> tuple:
    """Fit NMF; return (W, H, model)."""
    print(f"  Fitting NMF (n_components={N_COMPONENTS}, max_iter={MAX_ITER}) …")
    model = NMF(
        n_components=N_COMPONENTS,
        init="nndsvda",
        random_state=RANDOM_SEED,
        max_iter=MAX_ITER,
        tol=1e-4,
    )
    W = model.fit_transform(Z)  # (n_variants, n_components)
    H = model.components_       # (n_components, n_latents)
    recon_err = model.reconstruction_err_
    print(f"  Reconstruction error: {recon_err:.4f}")
    print(f"  W shape: {W.shape}, H shape: {H.shape}")
    return W, H, model


def compare_to_kmeans(H: np.ndarray, Z_disease: sp.csr_matrix) -> pd.DataFrame:
    """Compare NMF basis vectors (H) to k-means centroids via cosine similarity."""
    print("  Re-running k-means for centroid comparison …")
    km, cluster_ids, _ = run_disease_kmeans(Z_disease, verbose=False)
    centroids = km.cluster_centers_  # (n_clusters, n_latents)

    # L2-normalise both for cosine similarity
    H_norm   = H   / (np.linalg.norm(H,        axis=1, keepdims=True) + 1e-10)
    C_norm   = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)

    sim_mat  = cosine_similarity(H_norm, C_norm)  # (n_components, n_clusters)
    best_cluster = np.argmax(sim_mat, axis=1)
    best_sim     = sim_mat[np.arange(N_COMPONENTS), best_cluster]

    rows = []
    for comp_i in range(N_COMPONENTS):
        rows.append({
            "nmf_component":    comp_i,
            "best_kmeans_cluster": int(best_cluster[comp_i]),
            "cosine_similarity": round(float(best_sim[comp_i]), 4),
        })
    df = pd.DataFrame(rows).sort_values("cosine_similarity", ascending=False)
    return df, sim_mat, centroids, cluster_ids


def plot_similarity_heatmap(sim_mat: np.ndarray, out_path: Path):
    """Heatmap: NMF components (rows) × k-means clusters (cols), colour = cosine sim."""
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(sim_mat, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("k-means cluster")
    ax.set_ylabel("NMF component")
    ax.set_title(f"{NAME} — NMF component vs k-means centroid cosine similarity")
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_xticks(range(sim_mat.shape[1]))
    ax.set_xticklabels([str(i) for i in range(sim_mat.shape[1])], fontsize=6)
    ax.set_yticks(range(sim_mat.shape[0]))
    ax.set_yticklabels([str(i) for i in range(sim_mat.shape[0])], fontsize=6)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap → {out_path}")


def top_latents_per_component(H: np.ndarray, top_n: int = 10) -> pd.DataFrame:
    """For each NMF component, find the top-n latents by H weight."""
    rows = []
    for comp_i in range(H.shape[0]):
        top_idx = np.argsort(H[comp_i])[::-1][:top_n]
        for rank, latent_idx in enumerate(top_idx):
            rows.append({
                "nmf_component": comp_i,
                "rank":          rank + 1,
                "latent_idx":    int(latent_idx),
                "weight":        round(float(H[comp_i, latent_idx]), 6),
            })
    return pd.DataFrame(rows)


def component_cluster_assignments(W: np.ndarray) -> np.ndarray:
    """Assign each variant to the NMF component with the highest activation weight."""
    return np.argmax(W, axis=1)


def main():
    t0 = time.time()
    print(f"NMF analysis  model={NAME}  n_components={N_COMPONENTS}", flush=True)

    # Load
    print("\nLoading disease variant matrix …")
    Z_disease, n_cv_path = load_disease_matrix(NAME)
    print(f"  Z_disease shape: {Z_disease.shape}  (ClinVar path: {n_cv_path:,})")

    # NMF
    print("\nRunning NMF …")
    W, H, model = run_nmf(Z_disease)

    # Save W and H
    np.save(str(OUT / f"nmf_W_{NAME}.npy"), W)
    np.save(str(OUT / f"nmf_H_{NAME}.npy"), H)
    print(f"  W → {OUT / f'nmf_W_{NAME}.npy'}")
    print(f"  H → {OUT / f'nmf_H_{NAME}.npy'}")

    # Compare to k-means
    print("\nComparing NMF components to k-means centroids …")
    df_comp, sim_mat, centroids, cluster_ids = compare_to_kmeans(H, Z_disease)

    comp_path = OUT / f"nmf_vs_kmeans_cosine_{NAME}.csv"
    df_comp.to_csv(str(comp_path), index=False)
    print(f"  → {comp_path}")

    # Print top-matched pairs
    print("\nTop NMF↔k-means matches (cosine ≥ 0.50):")
    high = df_comp[df_comp["cosine_similarity"] >= 0.50]
    for _, row in high.iterrows():
        print(f"  NMF {int(row['nmf_component']):2d}  ↔  cluster {int(row['best_kmeans_cluster']):2d}"
              f"   cos={row['cosine_similarity']:.3f}")
    if len(high) == 0:
        print("  (none above 0.50 — NMF structure may differ from k-means)")

    # Heatmap
    print("\nPlotting similarity heatmap …")
    plot_similarity_heatmap(sim_mat,
                            OUT / f"nmf_vs_kmeans_cosine_heatmap_{NAME}.png")

    # Top latents per component
    df_top = top_latents_per_component(H, top_n=20)
    top_path = OUT / f"nmf_top_latents_{NAME}.csv"
    df_top.to_csv(str(top_path), index=False)
    print(f"  Top latents per component → {top_path}")

    # Component-level cluster overlap
    nmf_assign = component_cluster_assignments(W)
    km_assign  = cluster_ids

    # Per NMF component: mode k-means cluster and purity
    print("\nNMF component ↔ k-means cluster overlap (top 5 components by purity):")
    comp_purity = []
    for comp_i in range(N_COMPONENTS):
        mask   = nmf_assign == comp_i
        n_comp = mask.sum()
        if n_comp == 0:
            continue
        km_in_comp = km_assign[mask]
        mode_km    = int(pd.Series(km_in_comp).mode()[0])
        purity     = (km_in_comp == mode_km).mean()
        comp_purity.append((comp_i, mode_km, purity, n_comp))

    comp_purity.sort(key=lambda x: -x[2])
    for comp_i, mode_km, purity, n in comp_purity[:10]:
        print(f"  NMF {comp_i:2d}  → km_cluster {mode_km:2d}  purity={purity:.2f}  n={n:,}")

    print(f"\nDone. Total time: {(time.time()-t0)/60:.1f} min")
    print(f"Outputs in {OUT}")


if __name__ == "__main__":
    main()
