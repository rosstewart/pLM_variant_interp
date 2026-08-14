"""
Shared plotting utilities for ProtT5 Variant SAE publication figures.
All figure scripts import from here.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ────────────────────────────────────────────────────────────────────
LA   = Path("/data/ross/interp/latent_analysis")
VAL  = LA / "validation"
WF   = VAL / "within_family"
FIGS = Path("/data/ross/interp/paper/figures")
FIGS.mkdir(parents=True, exist_ok=True)

# Existing PNG (reuse directly in LaTeX — no regeneration needed)
UMAP_PNG = LA / "umap_disease_clusters_concat_ef1_k128.png"

# ── UniProt → gene symbol mapping ────────────────────────────────────────────
UNIPROT_TO_GENE = {
    "P04637": "TP53",
    "P60484": "PTEN",
    "P02452": "COL1A1",
    "P38398": "BRCA1",
    "P60709": "ACTB",
    "P01130": "LDLR",
    "P02461": "COL3A1",
    "P35222": "CTNNB1",
    "O00429": "DNM1L",
    "P00441": "SOD1",
    "P10275": "AR",
    "Q9Y6N9": "USH2A",
    "P40692": "MLH1",
    "P55072": "VCP",
    "P06400": "RB1",
    "P14923": "JUP",
    "Q86YC2": "BPIFB4",
    "P25054": "APC",
    "P40337": "VHL",
}

# Cluster mechanism labels (for leave_one_gene_out.csv alignment)
GENE_TO_MECHANISM = {
    "TP53":   "TP53 Zn-binding / dominant-neg",
    "PTEN":   "PTEN phosphatase",
    "COL1A1": "Collagen triple-helix",
    "BRCA1":  "BRCA1 RING/BRCT",
    "ACTB":   "Actin cytoskeleton",
    "LDLR":   "LDLR receptor",
}

# ── Colour schemes ────────────────────────────────────────────────────────────
# Semantic probe colours
PROBE_COLORS = {
    "destab_vs_neut": "#E74C3C",   # red
    "stab_vs_neut":   "#2ECC71",   # green
    "gof_vs_wt":      "#E67E22",   # orange
    "lof_vs_wt":      "#3498DB",   # blue
}
PROBE_LABELS = {
    "destab_vs_neut": "Destabilising",
    "stab_vs_neut":   "Stabilising",
    "gof_vs_wt":      "Gain-of-function",
    "lof_vs_wt":      "Loss-of-function",
}
PROBE_BASELINES = {
    "destab_vs_neut": 0.323,
    "stab_vs_neut":   0.292,
    "gof_vs_wt":      0.520,
    "lof_vs_wt":      0.498,
}

# Site type display names
SITE_LABELS = {
    "loss_metal_zn":          "Zn²⁺",
    "loss_metal_ca":          "Ca²⁺",
    "loss_metal_mg":          "Mg²⁺",
    "loss_metal_fe":          "Fe²⁺",
    "loss_metal_cu":          "Cu²⁺",
    "loss_metal_mn":          "Mn²⁺",
    "loss_metal_co":          "Co²⁺",
    "loss_metal_ni":          "Ni²⁺",
    "loss_metal_na":          "Na⁺",
    "loss_metal_k":           "K⁺",
    "loss_metal_cd":          "Cd²⁺",
    "loss_cofactor_nucleotide": "Nucleotide",
    "loss_cofactor_atp":      "ATP",
    "loss_cofactor_nad":      "NAD",
    "loss_cofactor_fad":      "FAD",
    "loss_cofactor_plp":      "PLP",
    "loss_catalytic":         "Catalytic",
    "loss_dna_binding":       "DNA binding",
    "loss_rna_binding":       "RNA binding",
    "loss_ppi_interface":     "PPI interface",
    "loss_allostery":         "Allosteric",
    "loss_phosphosite":       "Phosphosite",
    "loss_glycosite":         "Glycosite",
    "cancer_hotspot":         "Cancer hotspot",
}

# Gene-level colour palette for 12 focus genes (tab10 + extras)
import matplotlib.cm as cm
_tab20 = cm.get_cmap("tab20")
FOCUS_GENES = ["TP53", "PTEN", "COL1A1", "BRCA1", "ACTB", "LDLR",
               "COL3A1", "MLH1", "VCP", "RB1", "AR", "DNM1L"]
GENE_COLORS = {g: _tab20(i / len(FOCUS_GENES)) for i, g in enumerate(FOCUS_GENES)}

# ── Style setup ───────────────────────────────────────────────────────────────
def set_style():
    """Apply publication-quality matplotlib rcParams."""
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":          7,
        "axes.titlesize":     8,
        "axes.labelsize":     7,
        "xtick.labelsize":    6,
        "ytick.labelsize":    6,
        "legend.fontsize":    6,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.6,
        "xtick.major.width":  0.6,
        "ytick.major.width":  0.6,
        "lines.linewidth":    1.0,
        "patch.linewidth":    0.5,
        "pdf.fonttype":       42,   # embeds fonts as TrueType
        "ps.fonttype":        42,
    })


# ── Save helper ───────────────────────────────────────────────────────────────
def save_fig(fig, name: str, formats=("pdf", "png")):
    """Save figure to /data/ross/interp/paper/figures/ in requested formats."""
    for fmt in formats:
        out = FIGS / f"{name}.{fmt}"
        fig.savefig(str(out), bbox_inches="tight", dpi=300)
    print(f"  Saved: {[str(FIGS / f'{name}.{fmt}') for fmt in formats]}")


# ── Helper: map removed_gene → cluster ID from leave_one_gene_out ─────────────
def get_loog_cluster_map() -> dict:
    """Returns {uniprot_id: cluster_id} from leave_one_gene_out.csv.
    All 4 CSVs from cluster_validation_suite.py share the same cluster numbering."""
    import pandas as pd
    df = pd.read_csv(VAL / "leave_one_gene_out.csv")
    return dict(zip(df["removed_gene"], df["cluster"]))


def gene_label(uniprot: str) -> str:
    """Convert UniProt accession to readable gene label."""
    return UNIPROT_TO_GENE.get(uniprot, uniprot)
