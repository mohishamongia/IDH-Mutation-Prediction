"""
harmonize_and_cache.py

Combines run_harmonize_all.py + build_npy_cache.py into a single pass.

For each subject: N4 -> Nyul (in memory) -> save .nii.gz for reference ->
resize + stack -> save float16 .npy cache directly. Avoids writing the
harmonized nii.gz and then re-reading it back off disk just to resize it,
which was the extra round-trip in the two-script version.

Requires no_ws_harmonization_.py in the same folder.

Run: python3 harmonize_and_cache.py
"""

import os
import json
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import zoom
from tqdm import tqdm

from no_ws_harmonization_ import (
    MODALITY_FILES, fit_nyul_scales, load_nyul_scales, harmonize_subject,
)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG — edit before running
# ═══════════════════════════════════════════════════════════════════════════
RAW_DATA_DIR   = "/workspace/DATASETS/UTSW_Glioma_data/UTSW-Glioma"
TSV_PATH       = "/workspace/DATASETS/UTSW_Glioma_data/UTSW_Glioma_Metadata-2-1 (2).tsv"
NII_OUT_DIR    = "/workspace/harmonized_utsw_full_v2"     # harmonized .nii.gz, kept for reference
CACHE_DIR      = "/workspace/cache_npy_harmonized_v2"     # resized float16 .npy, what training reads
NYUL_SCALE_DIR = "/workspace/nyul_scales_v2"
MODALITIES     = ("t1", "t2", "flair")
N_FIT_SUBJECTS = 200
IMG_SIZE       = 96
SEED           = 42

os.makedirs(NII_OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# BUILD SUBJECT LIST (same filter as the CV training script)
# ═══════════════════════════════════════════════════════════════════════════
df = pd.read_csv(TSV_PATH, sep="\t")
df = df[df["IDH"].isin(["mutated", "wild type"])]
df = df[df["T1"] == "Available"]
df = df[df["Subject ID"].apply(
    lambda s: os.path.isdir(os.path.join(RAW_DATA_DIR, s)))]

all_subjects = df["Subject ID"].tolist()
print(f"Total subjects to harmonize + cache: {len(all_subjects)}")

subject_dirs = [os.path.join(RAW_DATA_DIR, s) for s in all_subjects]


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — FIT NYUL SCALES ONCE (see leakage note in no_ws_harmonization_.py)
# ═══════════════════════════════════════════════════════════════════════════
if all(os.path.exists(os.path.join(NYUL_SCALE_DIR, f"{m}_nyul.npy")) for m in MODALITIES):
    print(f"Nyul scales already exist in {NYUL_SCALE_DIR}, loading instead of re-fitting.")
    nyul_normalizers = load_nyul_scales(NYUL_SCALE_DIR, modalities=MODALITIES)
else:
    print(f"Fitting Nyul scales on {N_FIT_SUBJECTS or len(subject_dirs)} subjects...")
    nyul_normalizers = fit_nyul_scales(
        subject_dirs, NYUL_SCALE_DIR,
        modalities=MODALITIES, n_fit_subjects=N_FIT_SUBJECTS, seed=SEED,
    )


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — PER SUBJECT: HARMONIZE -> SAVE NII (reference) -> RESIZE -> CACHE
# ═══════════════════════════════════════════════════════════════════════════
manifest = {"ok": [], "skipped": []}

for sid, sdir in tqdm(list(zip(all_subjects, subject_dirs)), desc="Harmonize + cache"):
    cache_path = os.path.join(CACHE_DIR, f"{sid}.npy")
    if os.path.exists(cache_path):
        manifest["ok"].append(sid)
        continue

    try:
        harmonized = harmonize_subject(sdir, nyul_normalizers)  # dict: mod -> np.ndarray, bg=0.0
    except Exception as e:
        print(f"  ✗ {sid}: {type(e).__name__}: {e}")
        manifest["skipped"].append({"subject_id": sid, "error": str(e)})
        continue

    # Save harmonized nii.gz for reference/debugging (cheap relative to N4 itself)
    out_subject_dir = os.path.join(NII_OUT_DIR, sid)
    os.makedirs(out_subject_dir, exist_ok=True)
    ref = nib.load(os.path.join(sdir, MODALITY_FILES[MODALITIES[0]]))
    for mod_name, vol in harmonized.items():
        nib.save(nib.Nifti1Image(vol, ref.affine, ref.header),
                  os.path.join(out_subject_dir, f"{mod_name}_n4_nyul.nii.gz"))

    # Resize + stack directly from the in-memory harmonized arrays -- no
    # re-reading from disk. Channel order fixed as (t1, t2, flair) to match
    # the training script's expected indices 0/1/2.
    channels = []
    for mod in MODALITIES:
        vol = harmonized[mod]
        factors = [IMG_SIZE / s for s in vol.shape]
        channels.append(zoom(vol, factors, order=1))

    volume = np.stack(channels, axis=0).astype(np.float16)
    np.save(cache_path, volume)
    manifest["ok"].append(sid)

print(f"\nDone. OK: {len(manifest['ok'])}  Skipped: {len(manifest['skipped'])}")
with open(os.path.join(CACHE_DIR, "harmonize_cache_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

if manifest["skipped"]:
    print("\n⚠ Some subjects failed -- check harmonize_cache_manifest.json")

# Sanity check: background should be exactly 0.0 after harmonization+resize.
if manifest["ok"]:
    sample_sid = manifest["ok"][0] if manifest["ok"][0] != "" else manifest["ok"][-1]
    sample_path = os.path.join(CACHE_DIR, f"{sample_sid}.npy")
    if os.path.exists(sample_path):
        sample_vol = np.load(sample_path).astype(np.float32)
        bg_mask = sample_vol == 0
        print(f"\nSanity check on {sample_sid}: {bg_mask.sum()} exact-zero voxels "
              f"out of {sample_vol.size} ({100 * bg_mask.mean():.1f}%) -- "
              f"should be a large fraction if background survived correctly.")