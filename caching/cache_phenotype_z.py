"""
cache_phenotype_z.py

Encode pre-computed MegaScale (stability) and DMS (activity) WT/VT embeddings
through a named TopK SAE and save z_stab_{name}.npz and z_act_{name}.npz.

No new ProtT5 inference required — embeddings are already stored in:
  /data/ross/interp/collab_sae_cache/    layer20_wt.npy / layer20_vt.npy  (stability)
  /data/ross/interp/activity_sae_cache/  final_layer_wt.npy / final_layer_vt.npy (activity)

Outputs → /data/ross/interp/combined_sae_cache/
  z_stab_{name}.npz
  z_act_{name}.npz

Usage:
  python -u caching/cache_phenotype_z.py --name diff_ef4_k256
  python -u caching/cache_phenotype_z.py --name concat_ef1_k128  # already exists, skipped by default
  python -u caching/cache_phenotype_z.py --name diff_ef4_k256 --force  # overwrite existing
"""

import sys, argparse, time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared_infrastructure import (
    MODEL_REGISTRY, DEFAULT_NAME, TopKSAE, COMBINED, COMBINED_CACHE,
)

_ap = argparse.ArgumentParser(description="Encode phenotype embeddings through a named SAE")
_ap.add_argument("--name",   default=DEFAULT_NAME, help="Model name from MODEL_REGISTRY")
_ap.add_argument("--device", default="cuda:2")
_ap.add_argument("--batch",  type=int, default=4096)
_ap.add_argument("--force",  action="store_true", help="Overwrite existing files")
_args = _ap.parse_args()

NAME   = _args.name
DEVICE = torch.device(_args.device if torch.cuda.is_available() else "cpu")
BATCH  = _args.batch

if NAME not in MODEL_REGISTRY:
    print(f"ERROR: '{NAME}' not in MODEL_REGISTRY. Known: {list(MODEL_REGISTRY)}")
    sys.exit(1)

input_type, in_dim, ef, k = MODEL_REGISTRY[NAME]
dict_size = ef * in_dim
CHECKPOINT = COMBINED / f"combined_{NAME}.pt"

STAB_WT  = Path("/data/ross/interp/collab_sae_cache/layer20_wt.npy")
STAB_VT  = Path("/data/ross/interp/collab_sae_cache/layer20_vt.npy")
ACT_WT   = Path("/data/ross/interp/activity_sae_cache/final_layer_wt.npy")
ACT_VT   = Path("/data/ross/interp/activity_sae_cache/final_layer_vt.npy")

OUT_STAB = COMBINED_CACHE / f"z_stab_{NAME}.npz"
OUT_ACT  = COMBINED_CACHE / f"z_act_{NAME}.npz"
COMBINED_CACHE.mkdir(parents=True, exist_ok=True)


def build_input(wt: np.ndarray, vt: np.ndarray) -> np.ndarray:
    """Construct SAE input from WT and VT embeddings."""
    if input_type == "diff":
        return (vt - wt).astype(np.float32)        # (N, 1024)
    else:
        return np.concatenate([wt, vt], axis=1).astype(np.float32)  # (N, 2048)


def encode(X: np.ndarray, model: TopKSAE) -> sp.csr_matrix:
    """Encode X through SAE in batches; return sparse CSR."""
    N = X.shape[0]
    parts = []
    model.eval()
    with torch.no_grad():
        for i in range(0, N, BATCH):
            xb = torch.tensor(X[i:i+BATCH], dtype=torch.float32).to(DEVICE)
            z, _ = model(xb)
            parts.append(z.cpu().numpy())
    Z = np.concatenate(parts, axis=0)
    sparsity = 1 - np.count_nonzero(Z) / Z.size
    print(f"  Encoded {N:,} variants — sparsity={sparsity:.3%}")
    return sp.csr_matrix(Z.astype(np.float32))


def main():
    t0 = time.time()
    print(f"cache_phenotype_z  name={NAME}  input_type={input_type}  "
          f"in_dim={in_dim}  dict_size={dict_size}")
    print(f"  Checkpoint: {CHECKPOINT}")
    print(f"  Device: {DEVICE}")

    # Load SAE
    print("\nLoading SAE …")
    model = TopKSAE(in_dim=in_dim, ef=ef, k=k).to(DEVICE)
    state = torch.load(str(CHECKPOINT), map_location=DEVICE)
    model.load_state_dict(state)
    print(f"  Loaded: dict_size={dict_size}  k={k}")

    # ── Stability (MegaScale) ────────────────────────────────────────────────
    if OUT_STAB.exists() and not _args.force:
        print(f"\n[skip] {OUT_STAB.name} already exists (use --force to overwrite)")
    else:
        print("\nEncoding MegaScale stability variants …")
        wt_stab = np.load(str(STAB_WT))
        vt_stab = np.load(str(STAB_VT))
        print(f"  Loaded WT/VT: {wt_stab.shape}")
        X_stab = build_input(wt_stab, vt_stab)
        del wt_stab, vt_stab
        Z_stab = encode(X_stab, model)
        del X_stab
        sp.save_npz(str(OUT_STAB), Z_stab)
        print(f"  → {OUT_STAB}  ({OUT_STAB.stat().st_size/1e6:.0f} MB)")

    # ── Activity (DMS) ───────────────────────────────────────────────────────
    if OUT_ACT.exists() and not _args.force:
        print(f"\n[skip] {OUT_ACT.name} already exists (use --force to overwrite)")
    else:
        print("\nEncoding DMS activity variants …")
        wt_act = np.load(str(ACT_WT))
        vt_act = np.load(str(ACT_VT))
        print(f"  Loaded WT/VT: {wt_act.shape}")
        X_act = build_input(wt_act, vt_act)
        del wt_act, vt_act
        Z_act = encode(X_act, model)
        del X_act
        sp.save_npz(str(OUT_ACT), Z_act)
        print(f"  → {OUT_ACT}  ({OUT_ACT.stat().st_size/1e6:.0f} MB)")

    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
