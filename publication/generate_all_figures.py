#!/usr/bin/env python
"""Run all publication figure scripts in sequence."""
import subprocess, sys, time
from pathlib import Path

PYTHON = "/home/rcstewart/miniconda3/envs/ppi/bin/python"
PUB = Path(__file__).parent

scripts = [
    "fig1_overview.py",
    "fig2_probe_heatmap.py",
    "fig3_latent_specificity.py",
    "fig4_functional_sites.py",
    "fig5_within_family.py",
    "fig6_nmf_validation.py",
    "fig7_concrete_examples.py",
]

failed = []
for s in scripts:
    t0 = time.time()
    print(f"\n{'='*60}\nRunning {s} ...")
    r = subprocess.run([PYTHON, str(PUB / s)], capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(f"ERROR in {s}:\n{r.stderr}", file=sys.stderr)
        failed.append(s)
    else:
        print(f"  Done in {time.time()-t0:.1f}s")

print(f"\n{'='*60}")
if failed:
    print(f"FAILED ({len(failed)}/{len(scripts)}): {failed}", file=sys.stderr)
    sys.exit(1)
else:
    print(f"All {len(scripts)} figures generated successfully.")
