"""
check_hybrid_whitestripe_t1t2flair.py

Standalone sanity check for the N4 -> Hybrid WhiteStripe (T1 + T2 + FLAIR,
with graceful fallback) pipeline. Does NOT touch idh_cv_train.py, does NOT
build the full cache, does NOT train anything. Runs harmonization on a
handful of subjects and saves visual + numeric checks so you can confirm
it's working before committing DGX time to the full cohort cache build.

Requires, alongside this script:
  - Harmonization.py       (n4_bias_correct)
  - hybrid_whitestripe_T1T2Flair.py  (fit_whitestripe_hybrid, apply_whitestripe --
                             the version with optional flair_vol/flair_mask
                             args and graceful T1+T2+FLAIR -> T1+T2 -> T1
                             fallback)

Run: python3 check_hybrid_whitestripe_t1t2flair.py
"""

import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")   # no display on DGX; saves PNGs instead
import matplotlib.pyplot as plt

from Harmonization import n4_bias_correct
from hybrid_whitestripe_T1T2Flair import fit_whitestripe_hybrid, apply_whitestripe

# ── EDIT THESE THREE LINES ───────────────────────────────────────────────────
DATA_DIR = "/workspace/DATASETS/UTSW_Glioma_data/UTSW-Glioma"
CHECK_SUBJECTS = ["BT0001", "BT0002", "BT0003"]  # picked from your BT000X naming convention
OUT_DIR = "Normalisation_testing/N4_hybrid_whitestripe_T1T2Flair_check"
# ──────────────────────────────────────────────────────────────────────────────

# Confirm these match your real filenames before running -- same caveat as
# your T1/T2 patterns had earlier.
MODALITY_FILENAMES = {
    "t1": "brain_t1_ants.nii.gz",
    "t2": "brain_t2_ants.nii.gz",
    "flair": "brain_fl_ants.nii.gz",
}
MODALITY_ORDER = ["t1", "t2", "flair"]

os.makedirs(OUT_DIR, exist_ok=True)


def _harmonize_subject(sid):
    """
    Runs N4 -> Hybrid WhiteStripe (T1+T2+FLAIR, with graceful fallback) for
    one subject.
    Returns (raw, n4, ws, masks, ws_stats) -- each of raw/n4/ws/masks a dict
    keyed by modality -- or None if this subject was skipped.
    """
    folder = os.path.join(DATA_DIR, sid)

    paths = {mod: os.path.join(folder, fname) for mod, fname in MODALITY_FILENAMES.items()}
    missing = [mod for mod, p in paths.items() if not os.path.exists(p)]
    if missing:
        print(f"  [SKIP] {sid}: missing {missing} file(s)")
        return None

    raw = {mod: nib.load(p).get_fdata(dtype=np.float32) for mod, p in paths.items()}

    shapes = {mod: v.shape for mod, v in raw.items()}
    if len(set(shapes.values())) > 1:
        print(f"  [SKIP] {sid}: modality shape mismatch {shapes} "
              f"-- are they co-registered?")
        return None

    # ── N4 per modality (Harmonization.py -- takes a filepath, not an
    # already-loaded array) ──────────────────────────────────────────────
    n4 = {mod: n4_bias_correct(paths[mod]) for mod in paths}
    masks = {mod: n4[mod] > 0 for mod in n4}

    # ── Hybrid WhiteStripe (T1+T2+FLAIR, graceful fallback) ───────────────
    try:
        ws_stats = fit_whitestripe_hybrid(
            n4["t1"], n4["t2"], masks["t1"], masks["t2"],
            flair_vol=n4["flair"], flair_mask=masks["flair"],
        )
    except ValueError as e:
        print(f"  [SKIP] {sid}: hybrid WhiteStripe failed -- {e}")
        return None

    ws = {
        "t1": apply_whitestripe(n4["t1"], ws_stats["t1_mean"], ws_stats["t1_std"]),
        "t2": apply_whitestripe(n4["t2"], ws_stats["t2_mean"], ws_stats["t2_std"]),
        "flair": apply_whitestripe(n4["flair"], ws_stats["flair_mean"], ws_stats["flair_std"]),
    }

    return raw, n4, ws, masks, ws_stats


def check_one_subject(sid, per_subject):
    print(f"\n{'='*60}\nSubject: {sid}\n{'='*60}")

    result = per_subject.get(sid)
    if result is None:
        print(f"  [SKIP] {sid}: no harmonized data")
        return

    raw, n4, ws, masks, ws_stats = result

    print(f"  modalities used: {ws_stats['modalities_used']} "
          f"(fallback level {ws_stats['fallback_level']}, "
          f"{'FULL T1+T2+FLAIR' if ws_stats['fallback_level'] == 0 else 'FELL BACK'})")
    print(f"  hybrid WM reference voxels: {ws_stats['n_voxels']}")
    for mod in MODALITY_ORDER:
        print(f"  {mod:6s} | raw range [{raw[mod].min():.1f}, {raw[mod].max():.1f}] "
              f"| N4 range [{n4[mod].min():.1f}, {n4[mod].max():.1f}] "
              f"| hybrid-WS range [{ws[mod].min():.2f}, {ws[mod].max():.2f}]")

    # ── plot: raw, N4, hybrid-WS image, hybrid-WS DENSITY histogram ───────
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    for row, mod in enumerate(MODALITY_ORDER):
        mid = raw[mod].shape[2] // 2

        axes[row, 0].imshow(raw[mod][:, :, mid].T, cmap="gray", origin="lower")
        axes[row, 0].set_title(f"{mod} — raw")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(n4[mod][:, :, mid].T, cmap="gray", origin="lower")
        axes[row, 1].set_title(f"{mod} — N4 corrected")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(ws[mod][:, :, mid].T, cmap="gray", origin="lower")
        axes[row, 2].set_title(f"{mod} — Hybrid WhiteStripe")
        axes[row, 2].axis("off")

        axes[row, 3].hist(ws[mod][masks[mod]].ravel(), bins=100, color="steelblue")
        axes[row, 3].set_title(f"{mod} — Hybrid WS histogram")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{sid}_check.png")
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"  Saved visual check: {out_path}")


def check_hybrid_ws_overlay(subjects):
    """
    Harmonizes the given subjects (small subset, just for this check) and
    overlays hybrid-WhiteStripe DENSITY histograms across subjects, one
    panel per modality (T1/T2/FLAIR), so you can eyeball whether they land
    on top of each other.

    Returns the per-subject harmonized data (raw/N4/WS/masks/ws_stats) so
    check_one_subject() can reuse it without re-running N4 + WhiteStripe.
    """
    print(f"\n{'='*60}\nHybrid WhiteStripe overlay check "
          f"({len(subjects)} subjects)\n{'='*60}")

    per_subject = {}
    for sid in subjects:
        per_subject[sid] = _harmonize_subject(sid)

    results = [(sid, per_subject[sid]) for sid in subjects if per_subject[sid] is not None]
    if len(results) < 2:
        print("  [SKIP] need at least 2 successfully-processed subjects to overlay")
        return per_subject

    for mod in MODALITY_ORDER:
        fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(12, 4))
        for sid, (raw, n4, ws, masks, ws_stats) in results:
            ax_before.hist(n4[mod][masks[mod]].ravel(), bins=100, alpha=0.5, label=sid)
            ax_after.hist(ws[mod][masks[mod]].ravel(), bins=100, alpha=0.5, label=sid)

        ax_before.set_title(f"{mod} — before hybrid WS (N4 only)")
        ax_after.set_title(f"{mod} — after hybrid WS")
        ax_before.legend(fontsize=7)
        ax_after.legend(fontsize=7)
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"hybrid_ws_overlay_{mod}.png")
        plt.savefig(out_path, dpi=100)
        plt.close()
        print(f"  Saved: {out_path}  (compare left vs right — histograms should "
              f"align more tightly on the right if harmonization is helping)")

    print("  NOTE: if all CHECK_SUBJECTS are from the same site, this only "
          "checks within-site consistency, not cross-site harmonization -- "
          "add a TCGA/UCSF subject to test that instead.")

    print("\n  Fallback level counts (0 = full T1+T2+FLAIR used):")
    levels = [ws_stats["fallback_level"] for _, (raw, n4, ws, masks, ws_stats) in results]
    for lvl in sorted(set(levels)):
        count = levels.count(lvl)
        print(f"    level {lvl}: {count}/{len(results)} subjects "
              f"({100 * count / len(results):.0f}%)")

    return per_subject


if __name__ == "__main__":
    # Harmonize once on the check subjects and get the overlay + fallback
    # summary first (mirrors check_hybrid_whitestripe.py's fit-then-check
    # ordering).
    per_subject = check_hybrid_ws_overlay(CHECK_SUBJECTS)

    for sid in CHECK_SUBJECTS:
        check_one_subject(sid, per_subject)

    n_ok = sum(1 for r in per_subject.values() if r is not None)
    n_fallback = sum(1 for r in per_subject.values() if r is not None and r[-1]["fallback_used"])
    print(f"\nDone. {n_ok}/{len(CHECK_SUBJECTS)} subjects processed successfully, "
          f"{n_fallback} used a fallback (T1+T2 or T1-only).")
    print(f"Check the PNGs in {OUT_DIR}/ — pull them off the DGX to view, "
          f"e.g. scp or rsync to your local machine.")