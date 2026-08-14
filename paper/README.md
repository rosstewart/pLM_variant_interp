# ProtT5 Variant SAE Paper

## Compilation

LaTeX is not installed on the analysis server. To compile:

### Option A — Overleaf
1. Upload `main.tex`, `refs.bib`, and all figures from `/data/ross/interp/paper/figures/`
2. Set compiler to pdflatex
3. Compile

### Option B — Local machine (with TeX Live / MacTeX)
```bash
# Copy figures to local paper dir (or symlink)
scp /data/ross/interp/paper/figures/*.pdf ./figures/
scp /data/ross/interp/paper/figures/*.png ./figures/
scp /data/ross/interp/latent_analysis/umap_disease_clusters_concat_ef1_k128.png ./figures/

# Compile
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Option C — Docker
```bash
docker run --rm -v $(pwd):/workdir -w /workdir texlive/texlive \
  bash -c "pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex"
```

## Generate Figures
```bash
cd /home/rcstewart/ppi_lossgain/sparse_bottleneck
/home/rcstewart/miniconda3/envs/ppi/bin/python publication/generate_all_figures.py \
  > /data/ross/interp/paper/figures/generate_all.log 2>&1
```

## File Map
- `main.tex` — LaTeX source
- `refs.bib` — BibTeX bibliography
- Figures generated to `/data/ross/interp/paper/figures/`
- Figure scripts in `../publication/fig*.py`
