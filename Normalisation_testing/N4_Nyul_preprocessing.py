"""
Standalone Preprocessing Test — N4 + Nyul, no training involved
==================================================================
Run this BEFORE touching idh_cv_train.py. Point it at a small handful of
subjects per site (e.g. 5-10 each), and it will:
  1. Load raw volumes
  2. Apply N4 bias correction
  3. Fit Nyul landmarks on the UTSW subset, apply to all sites
  4. Print validation numbers (CV, octant range, cross-site divergence)
  5. Save before/after histogram + slice comparison plots to disk

Usage:
    python test_preprocessing.py
(edit SUBJECTS_PER_SITE dict below to point at your real subject folders)
"""

import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")  # headless — saving to file, not displaying
import matplotlib.pyplot as plt

from n4_correction import n4_correct, validate_correction, print_validation_report
from nyul_normalization import fit_nyul_landmarks, apply_nyul, validate_nyul, print_nyul_report

OUT_DIR = "preprocessing_test_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# EDIT THIS — point at a SMALL number of real subjects per site
# ═══════════════════════════════════════════════════════════════════════════
# Each entry: site_name -> {"data_dir": ..., "subject_ids": [...], "t1_filename": ...}
# Keep subject_ids short (5-10) — this script loads full volumes into memory
# and is meant to run in a minute or two, not a full preprocessing pass.
SITE_CONFIG = {
    "UTSW": {
        "data_dir": "DATASETS/UTSW_Glioma_data/UTSW-Glioma",
        "subject_ids": ["BT0001", "BT0002", "BT0003", "BT0004", "BT0005"],
        # UTSW filenames are generic (identical name every subject) — exact match is fine.
        "t1_glob": "brain_t1_ants.nii.gz",
    },
    "TCGA": {
        "data_dir": "DATASETS/BraTS_TCGA_GBM/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations",
        "subject_ids": ["TCGA-02-0006", "TCGA-02-0009", "TCGA-02-0011", "TCGA-02-0027"],
        # TCGA filenames embed subject ID + scan date (e.g. TCGA-02-0006_1996.08.23_t1.nii),
        # and the date isn't known ahead of time — use a wildcard, and match on "_t1."
        # specifically so it doesn't also grab t1ce/t2/flair/seg files sharing the prefix.
        "t1_glob": "{sid}_*_t1.nii*",
    },
    "UCSF": {
        "data_dir": "DATASETS/UCSF-PDGM-v5",
        "subject_ids": ["UCSF-PDGM-0005_nifti", "UCSF-PDGM-0007_nifti", "UCSF-PDGM-0010_nifti"],
        # CONFIRMED from directory listing: folder is "..._nifti" but internal filenames
        # are NOT (e.g. UCSF-PDGM-0005_T1.nii.gz, not UCSF-PDGM-0005_nifti_T1.nii.gz).
        # strip_from_sid removes that suffix before building the filename pattern.
        "strip_from_sid": "_nifti",
        # CONFIRMED: this dataset ships BOTH raw (T1.nii.gz) and pre-bias-corrected
        # (T1_bias.nii.gz) versions. Use the RAW version so your own N4 step does
        # comparable work here to what it does on UTSW/TCGA's raw scans — using
        # T1_bias would mean double-correcting (N4 on top of their existing correction),
        # which would understate this site's true bias field vs the other two sites.
        # The trailing ".nii.gz" (not a wildcard) + no wildcard before it is deliberate:
        # avoids accidentally matching T1c.nii.gz or T1_bias.nii.gz, which "T1*.nii*"
        # would have matched too.
        "t1_glob": "{sid}_T1.nii.gz",
    },
}

REFERENCE_SITE = "UTSW"  # fit Nyul landmarks on this site's subjects


# ═══════════════════════════════════════════════════════════════════════════
# LOAD
# ═══════════════════════════════════════════════════════════════════════════
import glob as globmod

def print_first_subject_files(site_name, config):
    """Debug helper: lists every file under the first subject's folder so you
    can confirm the real T1 filename pattern before trusting t1_glob."""
    if not config["subject_ids"]:
        return
    sid = config["subject_ids"][0]
    folder = os.path.join(config["data_dir"], sid)
    if not os.path.isdir(folder):
        print(f"  [{site_name}] folder not found: {folder} — check data_dir/subject_ids")
        return
    print(f"  [{site_name}] files under {folder}:")
    for f in sorted(os.listdir(folder)):
        print(f"      {f}")


def find_t1_file(data_dir, sid, t1_glob, strip_from_sid=None):
    """Resolve the T1 file for one subject via glob pattern (supports {sid}
    substitution and wildcards for unknown parts like scan dates).

    strip_from_sid: substring to remove from sid ONLY when building the
    filename pattern — use this when the folder name and the internal
    filename prefix differ (e.g. UCSF-PDGM's "..._nifti" folder suffix
    that isn't part of the files inside it). The folder path itself still
    uses the full, un-stripped sid.
    """
    file_sid = sid.replace(strip_from_sid, "") if strip_from_sid else sid
    pattern = t1_glob.format(sid=file_sid)
    matches = sorted(globmod.glob(os.path.join(data_dir, sid, pattern)))
    if not matches:
        # fall back to a recursive search in case files are nested one level deeper
        matches = sorted(globmod.glob(os.path.join(data_dir, sid, "**", pattern), recursive=True))
    return matches[0] if matches else None


def load_site_volumes(site_name, config):
    vols, masks, ids_loaded = [], [], []
    strip_from_sid = config.get("strip_from_sid")
    for sid in config["subject_ids"]:
        path = find_t1_file(config["data_dir"], sid, config["t1_glob"], strip_from_sid=strip_from_sid)
        if path is None:
            file_sid = sid.replace(strip_from_sid, "") if strip_from_sid else sid
            print(f"  [skip] {site_name}/{sid}: no file matched pattern "
                  f"'{config['t1_glob'].format(sid=file_sid)}'")
            continue
        print(f"  [{site_name}/{sid}] matched: {path}")
        vol = nib.load(path).get_fdata(dtype=np.float32)
        mask = vol > 0
        vols.append(vol)
        masks.append(mask)
        ids_loaded.append(sid)
    print(f"  Loaded {len(vols)}/{len(config['subject_ids'])} subjects for {site_name}")
    return vols, masks, ids_loaded


# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════
def plot_histograms(raw_by_site, n4_by_site, nyul_by_site, masks_by_site, out_path):
    """Overlay intensity histograms across sites, at each preprocessing stage."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    stages = [("Raw", raw_by_site), ("After N4", n4_by_site), ("After N4+Nyul", nyul_by_site)]

    colors = {"UTSW": "tab:blue", "TCGA": "tab:orange", "UCSF": "tab:green"}

    for ax, (stage_name, vols_by_site) in zip(axes, stages):
        for site, vols in vols_by_site.items():
            if not vols:
                continue
            masks = masks_by_site[site]
            all_vals = np.concatenate([v[m] for v, m in zip(vols, masks)])
            ax.hist(all_vals, bins=80, alpha=0.5, density=True,
                    label=site, color=colors.get(site))
        ax.set_title(stage_name)
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Density")
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved histogram comparison: {out_path}")


def plot_slice_comparison(raw_vol, n4_vol, nyul_vol, site_name, subject_id, out_path):
    """Side-by-side middle-slice view: raw vs N4 vs N4+Nyul, same subject."""
    mid = raw_vol.shape[0] // 2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, vol, title in zip(
        axes, [raw_vol, n4_vol, nyul_vol], ["Raw", "After N4", "After N4+Nyul"]
    ):
        ax.imshow(vol[mid], cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{site_name} / {subject_id} — middle axial slice")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved slice comparison: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("Checking actual filenames in each site's first subject folder...\n")
    print("(confirm t1_glob patterns above actually match these before proceeding)\n")
    for site, config in SITE_CONFIG.items():
        print_first_subject_files(site, config)
    print()

    print("Loading raw volumes per site...\n")
    raw_by_site, masks_by_site, ids_by_site = {}, {}, {}
    for site, config in SITE_CONFIG.items():
        vols, masks, ids_loaded = load_site_volumes(site, config)
        raw_by_site[site] = vols
        masks_by_site[site] = masks
        ids_by_site[site] = ids_loaded

    if not any(raw_by_site.values()):
        print("\nNo volumes loaded — edit SITE_CONFIG at the top of this script "
              "with real subject_ids and paths, then re-run.")
        return

    # ── Step 1: N4 correction, per site ─────────────────────────────────
    print("\nApplying N4 bias correction...\n")
    n4_by_site = {}
    for site, vols in raw_by_site.items():
        masks = masks_by_site[site]
        corrected = [n4_correct(v, mask=m) for v, m in zip(vols, masks)]
        n4_by_site[site] = corrected

        if vols:  # per-site N4 validation on the first subject as a spot check
            report = validate_correction(vols[0], corrected[0], mask=masks[0])
            print(f"  [{site} / {ids_by_site[site][0]}] N4 check:")
            print_validation_report(report)

    # ── Step 2: Fit Nyul landmarks on reference site (post-N4) ───────────
    print(f"\nFitting Nyul landmarks on {REFERENCE_SITE} (post-N4)...")
    ref_vols = n4_by_site[REFERENCE_SITE]
    ref_masks = masks_by_site[REFERENCE_SITE]
    if not ref_vols:
        print(f"  No {REFERENCE_SITE} subjects loaded — cannot fit reference landmarks. Stopping.")
        return
    landmarks = fit_nyul_landmarks(ref_vols, masks=ref_masks)
    print(f"  Landmarks fitted from {len(ref_vols)} {REFERENCE_SITE} subjects.")

    # ── Step 3: Apply Nyul to all sites ──────────────────────────────────
    print("\nApplying Nyul matching to all sites...")
    nyul_by_site = {}
    for site, vols in n4_by_site.items():
        masks = masks_by_site[site]
        matched = [apply_nyul(v, landmarks, mask=m) for v, m in zip(vols, masks)]
        nyul_by_site[site] = matched

    # ── Step 4: Cross-site divergence check ──────────────────────────────
    print("\nChecking cross-site convergence after N4+Nyul...")
    nonempty_sites = {s: v for s, v in n4_by_site.items() if v}
    nonempty_masks = {s: masks_by_site[s] for s in nonempty_sites}
    nyul_report = validate_nyul(nonempty_sites, landmarks, masks_by_site=nonempty_masks)
    print_nyul_report(nyul_report)

    # ── Step 5: Plots ─────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_histograms(raw_by_site, n4_by_site, nyul_by_site, masks_by_site,
                     os.path.join(OUT_DIR, "histogram_comparison.png"))

    for site in raw_by_site:
        if raw_by_site[site]:
            plot_slice_comparison(
                raw_by_site[site][0], n4_by_site[site][0], nyul_by_site[site][0],
                site, ids_by_site[site][0],
                os.path.join(OUT_DIR, f"slices_{site}_{ids_by_site[site][0]}.png"),
            )

    print(f"\nDone. Check {OUT_DIR}/ for histogram_comparison.png and per-site slice images.")
    print("If histograms visually converge across sites in the 3rd panel, and the")
    print("cross-site divergence number above dropped substantially, you're good")
    print("to wire n4_correct() + apply_nyul() into preprocess_and_cache().")


if __name__ == "__main__":
    main()