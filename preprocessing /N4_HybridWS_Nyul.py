"""
build_n4_hybridws_nyul_3site.py

Full pipeline: N4 -> Hybrid WhiteStripe (T1+T2+FLAIR, graceful fallback)
                  -> Nyul histogram matching

Run across 30 subjects each from UTSW, UCSF-PDGM, and BraTS-GLI (90 total).

This does three things:
  1. Harmonizes every subject through all three stages.
  2. Fits Nyul on the POOLED hybrid-WhiteStripe output across all three
     sites (not per-site), then applies that single fitted normalizer to
     every subject -- this is what actually tests whether the pipeline
     harmonizes across sites, rather than just standardizing within one.
  3. Logs a per-subject summary CSV (site, fallback level, WhiteStripe
     stats, post-Nyul percentiles) that plugs directly into the cross-site
     evaluation metrics (CV / Wasserstein / site-classifier) discussed
     separately -- no tissue segmentation needed, this reuses the
     WhiteStripe reference-region stats and masked-brain percentiles.
  4. Saves before/after-Nyul histogram overlays, colored by site, so you
     can eyeball cross-site alignment before running the full metrics.

Does NOT touch idh_cv_train.py, does NOT build the full 622-subject cache,
does NOT train anything.

Requires, alongside this script (same as your existing check scripts):
  - Harmonization.py                    (n4_bias_correct, NyulNormalizer)
  - hybrid_whitestripe_T1T2Flair.py     (fit_whitestripe_hybrid, apply_whitestripe)

Run: python3 build_n4_hybridws_nyul_3site.py
"""

import os
import csv
import glob
import random

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")   # no display on DGX; saves PNGs instead
import matplotlib.pyplot as plt

from Harmonization import n4_bias_correct, NyulNormalizer
from hybrid_whitestripe_T1T2Flair import fit_whitestripe_hybrid, apply_whitestripe

# ══════════════════════════════════════════════════════════════════════════
# EDIT THIS BLOCK -- dataset paths, subject counts, and modality filename
# patterns. UTSW is filled in from your existing scripts; UCSF and BraTS
# patterns below are my best guess at each dataset's standard naming and
# almost certainly need adjusting to match your actual folder layout.
# ══════════════════════════════════════════════════════════════════════════

N_SUBJECTS_PER_SITE = 30
RANDOM_SEED = 42          # set to None for a fresh random sample each run
SAVE_VOLUMES = False      # True -> also writes harmonized .nii.gz per subject
OUT_DIR = "harmonization_3stage_3site"

MODALITY_ORDER = ["t1", "t2", "flair"]

DATASET_CONFIGS = {
    "UTSW": {
        "data_dir": "DATASETS/UTSW_Glioma_data/UTSW-Glioma",
        # exact filenames, one subfolder per subject -- same as your other scripts
        "modality_patterns": {
            "t1": "brain_t1_ants.nii.gz",
            "t2": "brain_t2_ants.nii.gz",
            "flair": "brain_fl_ants.nii.gz",
        },
    },
    "UCSF": {
        "data_dir": "DATASETS/UCSF-PDGM-v5",
        # EDIT: glob patterns (subject ID is usually embedded in the filename
        # for UCSF-PDGM, e.g. "UCSF-PDGM-0004_T1.nii.gz")
        "modality_patterns": {
            "t1": "*T1.nii.gz",
            "t2": "*T2.nii.gz",
            "flair": "*FLAIR.nii.gz",
        },
    },
    "BraTS": {
        "data_dir": "DATASETS/BraTS2023_GLI/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
        # EDIT: BraTS-GLI naming convention, e.g.
        # "BraTS-GLI-00000-000-t1n.nii.gz", "...-t2w.nii.gz", "...-t2f.nii.gz"
        # NOTE: BraTS calls FLAIR "t2f" and native T1 "t1n" -- double check
        # you want t1n (not t1c, the contrast-enhanced one) to match how
        # you're treating T1 in UTSW/UCSF.
        "modality_patterns": {
            "t1": "*t1n.nii.gz",
            "t2": "*t2w.nii.gz",
            "flair": "*t2f.nii.gz",
        },
    },
}

os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# Subject discovery
# ══════════════════════════════════════════════════════════════════════════

def resolve_modality_path(subject_dir, pattern):
    """Exact filename or glob pattern -> single matching path, or None."""
    if "*" not in pattern:
        candidate = os.path.join(subject_dir, pattern)
        return candidate if os.path.exists(candidate) else None
    matches = glob.glob(os.path.join(subject_dir, pattern))
    return matches[0] if matches else None


def get_subjects(data_dir, n_subjects, seed):
    """Lists subject subfolders, returns a reproducible sample of n_subjects."""
    all_subjects = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )
    if len(all_subjects) <= n_subjects:
        print(f"  [WARN] only {len(all_subjects)} subjects found in {data_dir}, "
              f"using all of them (wanted {n_subjects})")
        return all_subjects
    rng = random.Random(seed)
    return sorted(rng.sample(all_subjects, n_subjects))


# ══════════════════════════════════════════════════════════════════════════
# Stage 1+2: N4 -> Hybrid WhiteStripe (T1+T2+FLAIR, graceful fallback)
# ══════════════════════════════════════════════════════════════════════════

def harmonize_subject_stage12(site, sid, data_dir, patterns):
    """
    Runs N4 then hybrid WhiteStripe for one subject.
    Returns dict with raw/n4/ws/masks (each keyed by modality) + ws_stats,
    or None if the subject was skipped.
    """
    subject_dir = os.path.join(data_dir, sid)
    paths = {mod: resolve_modality_path(subject_dir, pat)
             for mod, pat in patterns.items()}
    missing = [mod for mod, p in paths.items() if p is None]
    if missing:
        print(f"  [SKIP] {site}/{sid}: missing {missing}")
        return None

    raw = {mod: nib.load(p).get_fdata(dtype=np.float32) for mod, p in paths.items()}
    shapes = {mod: v.shape for mod, v in raw.items()}
    if len(set(shapes.values())) > 1:
        print(f"  [SKIP] {site}/{sid}: modality shape mismatch {shapes} "
              f"-- are they co-registered?")
        return None

    try:
        n4 = {mod: n4_bias_correct(paths[mod]) for mod in paths}
    except Exception as e:
        print(f"  [SKIP] {site}/{sid}: N4 failed -- {e}")
        return None
    masks = {mod: n4[mod] > 0 for mod in n4}

    try:
        ws_stats = fit_whitestripe_hybrid(
            n4["t1"], n4["t2"], masks["t1"], masks["t2"],
            flair_vol=n4["flair"], flair_mask=masks["flair"],
        )
    except ValueError as e:
        print(f"  [SKIP] {site}/{sid}: hybrid WhiteStripe failed -- {e}")
        return None

    ws = {
        "t1": apply_whitestripe(n4["t1"], ws_stats["t1_mean"], ws_stats["t1_std"]),
        "t2": apply_whitestripe(n4["t2"], ws_stats["t2_mean"], ws_stats["t2_std"]),
        "flair": apply_whitestripe(n4["flair"], ws_stats["flair_mean"], ws_stats["flair_std"]),
    }

    return {"raw": raw, "n4": n4, "ws": ws, "masks": masks, "ws_stats": ws_stats}


def harmonize_all_sites():
    """Runs stage 1+2 for all subjects across all sites. Returns nested dict."""
    per_site = {}
    for site, cfg in DATASET_CONFIGS.items():
        print(f"\n{'='*60}\n{site}: selecting {N_SUBJECTS_PER_SITE} subjects\n{'='*60}")
        subjects = get_subjects(cfg["data_dir"], N_SUBJECTS_PER_SITE, RANDOM_SEED)
        print(f"  subjects: {subjects}")

        per_subject = {}
        for sid in subjects:
            result = harmonize_subject_stage12(site, sid, cfg["data_dir"], cfg["modality_patterns"])
            if result is not None:
                per_subject[sid] = result
        print(f"  {len(per_subject)}/{len(subjects)} subjects harmonized successfully "
              f"(N4 + hybrid WhiteStripe)")
        per_site[site] = per_subject
    return per_site


# ══════════════════════════════════════════════════════════════════════════
# Stage 3: Nyul, fit on the POOLED hybrid-WS output across all sites
# ══════════════════════════════════════════════════════════════════════════

def fit_pooled_nyul(per_site):
    """
    Fits one NyulNormalizer per modality on hybrid-WS volumes pooled across
    ALL sites (not per-site) -- this is the harmonization step that's
    actually supposed to pull the three datasets onto a common scale.
    Returns dict: modality -> fitted NyulNormalizer.
    """
    print(f"\n{'='*60}\nFitting pooled Nyul (all sites combined)\n{'='*60}")
    normalizers = {}
    for mod in MODALITY_ORDER:
        volumes, masks = [], []
        for site, per_subject in per_site.items():
            for sid, data in per_subject.items():
                volumes.append(data["ws"][mod])
                masks.append(data["masks"][mod])
        print(f"  {mod}: fitting on {len(volumes)} pooled subjects")
        normalizers[mod] = NyulNormalizer().fit(volumes, masks)
    return normalizers


def apply_nyul_all(per_site, normalizers):
    """Applies the pooled-fit Nyul normalizer to every subject's WS volumes."""
    for site, per_subject in per_site.items():
        for sid, data in per_subject.items():
            data["nyul"] = {
                mod: normalizers[mod].transform(data["ws"][mod], data["masks"][mod])
                for mod in MODALITY_ORDER
            }


# ══════════════════════════════════════════════════════════════════════════
# Summary CSV -- feeds the cross-site evaluation metrics (no segmentation
# needed: WhiteStripe reference-region stats + masked-brain percentiles)
# ══════════════════════════════════════════════════════════════════════════

PERCENTILES = [10, 25, 50, 75, 90, 99]


def write_summary_csv(per_site, out_path):
    rows = []
    for site, per_subject in per_site.items():
        for sid, data in per_subject.items():
            ws_stats = data["ws_stats"]
            row = {
                "site": site,
                "subject_id": sid,
                "fallback_level": ws_stats.get("fallback_level"),
                "modalities_used": ws_stats.get("modalities_used"),
                "n_wm_voxels": ws_stats.get("n_voxels"),
                "t1_ws_mean": ws_stats.get("t1_mean"),
                "t1_ws_std": ws_stats.get("t1_std"),
                "t2_ws_mean": ws_stats.get("t2_mean"),
                "t2_ws_std": ws_stats.get("t2_std"),
                "flair_ws_mean": ws_stats.get("flair_mean"),
                "flair_ws_std": ws_stats.get("flair_std"),
            }
            for mod in MODALITY_ORDER:
                vol = data["nyul"][mod]
                mask = data["masks"][mod]
                vals = vol[mask]
                for p in PERCENTILES:
                    row[f"{mod}_nyul_p{p}"] = np.percentile(vals, p)
                row[f"{mod}_nyul_mean"] = vals.mean()
                row[f"{mod}_nyul_std"] = vals.std()
            rows.append(row)

    if not rows:
        print("  [WARN] no rows to write -- nothing harmonized successfully")
        return

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved summary CSV: {out_path} ({len(rows)} subjects)")
    print("  -> use this for cross-site CV / Wasserstein distance on the "
          "*_nyul_p* columns, and as features for the site-classifier check.")


# ══════════════════════════════════════════════════════════════════════════
# Overlay plots -- before/after Nyul, colored by site
# ══════════════════════════════════════════════════════════════════════════

SITE_COLORS = {"UTSW": "tab:blue", "UCSF": "tab:orange", "BraTS": "tab:green"}


def plot_cross_site_overlay(per_site, out_dir):
    for mod in MODALITY_ORDER:
        fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(14, 5))
        for site, per_subject in per_site.items():
            color = SITE_COLORS.get(site, None)
            first = True
            for sid, data in per_subject.items():
                mask = data["masks"][mod]
                before_vals = data["ws"][mod][mask].ravel()
                after_vals = data["nyul"][mod][mask].ravel()
                label = site if first else None
                ax_before.hist(before_vals, bins=100, alpha=0.25, color=color,
                                density=True, label=label)
                ax_after.hist(after_vals, bins=100, alpha=0.25, color=color,
                               density=True, label=label)
                first = False

        ax_before.set_title(f"{mod} — before Nyul (hybrid-WS only), by site")
        ax_after.set_title(f"{mod} — after pooled Nyul, by site")
        ax_before.legend()
        ax_after.legend()
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"cross_site_overlay_{mod}.png")
        plt.savefig(out_path, dpi=100)
        plt.close()
        print(f"  Saved: {out_path}  (right panel should show tighter "
              f"cross-site overlap than the left if harmonization is helping)")


# ══════════════════════════════════════════════════════════════════════════
# Optional: save harmonized volumes to disk
# ══════════════════════════════════════════════════════════════════════════

def save_volumes(per_site, out_dir):
    for site, per_subject in per_site.items():
        site_dir = os.path.join(out_dir, "harmonized", site)
        os.makedirs(site_dir, exist_ok=True)
        for sid, data in per_subject.items():
            subj_dir = os.path.join(site_dir, sid)
            os.makedirs(subj_dir, exist_ok=True)
            for mod in MODALITY_ORDER:
                # Use T1's affine/header as reference since all modalities
                # for a subject should already be co-registered.
                ref_path_img = nib.Nifti1Image(data["nyul"][mod].astype(np.float32), np.eye(4))
                nib.save(ref_path_img, os.path.join(subj_dir, f"{mod}_n4_hybridws_nyul.nii.gz"))
    print(f"  Saved harmonized volumes under {os.path.join(out_dir, 'harmonized')}/")


# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    per_site = harmonize_all_sites()

    normalizers = fit_pooled_nyul(per_site)
    apply_nyul_all(per_site, normalizers)

    print(f"\n{'='*60}\nSaving outputs\n{'='*60}")
    write_summary_csv(per_site, os.path.join(OUT_DIR, "harmonization_summary.csv"))
    plot_cross_site_overlay(per_site, OUT_DIR)

    if SAVE_VOLUMES:
        save_volumes(per_site, OUT_DIR)

    print(f"\nDone. Per-site subject counts:")
    for site, per_subject in per_site.items():
        n_fallback = sum(
            1 for d in per_subject.values()
            if d["ws_stats"].get("fallback_level", 0) != 0
        )
        print(f"  {site}: {len(per_subject)} harmonized, {n_fallback} used a fallback")
    print(f"\nCheck {OUT_DIR}/ -- pull it off the DGX to view "
          f"(e.g. scp or rsync to your local machine).")
