"""

Requires whitestripe_normalization.py (fit_whitestripe_hybrid,
apply_whitestripe) alongside this script.


"""

import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")   # no display on DGX; saves PNGs instead
import matplotlib.pyplot as plt

from Harmonization import (
    n4_bias_correct,
    MODALITY_FILES,
)
from hybrid_whitestripe import fit_whitestripe_hybrid, apply_whitestripe

# ── EDIT THESE THREE LINES ───────────────────────────────────────────────────
DATA_DIR = "/workspace/DATASETS/UTSW_Glioma_data/UTSW-Glioma"
CHECK_SUBJECTS = ["BT0001", "BT0002", "BT0003"]
OUT_DIR = "Normalisation_testing/N4_hybrid_whitestripe_check"
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)


def check_one_subject(sid):
    """
    Runs N4 -> Hybrid WhiteStripe (T1 ∩ T2) for one subject and saves a
    visual check: raw / N4 / hybrid-WS images plus a density histogram,
    for T1 and T2 (the two modalities the hybrid reference mask is built
    from).

    Returns a dict with the masked hybrid-WS voxel values + QC info for
    the cross-subject overlay step, or None if this subject was skipped.
    """
    print(f"\n{'='*60}\nSubject: {sid}\n{'='*60}")
    folder = os.path.join(DATA_DIR, sid)

    t1_path = os.path.join(folder, MODALITY_FILES["t1"])
    t2_path = os.path.join(folder, MODALITY_FILES["t2"])

    if not os.path.exists(t1_path) or not os.path.exists(t2_path):
        print(f"  [SKIP] {sid}: missing T1 or T2 file")
        return None

    t1_raw = nib.load(t1_path).get_fdata(dtype=np.float32)
    t2_raw = nib.load(t2_path).get_fdata(dtype=np.float32)

    if t1_raw.shape != t2_raw.shape:
        print(f"  [SKIP] {sid}: T1/T2 shape mismatch {t1_raw.shape} vs "
              f"{t2_raw.shape} -- are they co-registered?")
        return None

    # ── N4 ───────────────────────────────────────────────────────────────
    t1_n4 = n4_bias_correct(t1_path)
    t2_n4 = n4_bias_correct(t2_path)

    t1_mask = t1_n4 > 0
    t2_mask = t2_n4 > 0

    # ── Hybrid WhiteStripe (T1 ∩ T2) ─────────────────────────────────────
    try:
        ws_stats = fit_whitestripe_hybrid(t1_n4, t2_n4, t1_mask, t2_mask)
    except ValueError as e:
        print(f"  [SKIP] {sid}: hybrid WhiteStripe failed -- {e}")
        return None

    t1_ws = apply_whitestripe(t1_n4, ws_stats["t1_mean"], ws_stats["t1_std"])
    t2_ws = apply_whitestripe(t2_n4, ws_stats["t2_mean"], ws_stats["t2_std"])

    flag = " <-- fallback to T1-only (small T1∩T2 intersection)" if ws_stats["fallback_used"] else ""
    print(f"  hybrid WM reference voxels: {ws_stats['n_voxels']}{flag}")
    print(f"  t1 | raw range [{t1_raw.min():.1f}, {t1_raw.max():.1f}] "
          f"| N4 range [{t1_n4.min():.1f}, {t1_n4.max():.1f}] "
          f"| hybrid-WS range [{t1_ws.min():.2f}, {t1_ws.max():.2f}]")
    print(f"  t2 | raw range [{t2_raw.min():.1f}, {t2_raw.max():.1f}] "
          f"| N4 range [{t2_n4.min():.1f}, {t2_n4.max():.1f}] "
          f"| hybrid-WS range [{t2_ws.min():.2f}, {t2_ws.max():.2f}]")

    # ── plot: raw, N4, hybrid-WS image, hybrid-WS DENSITY histogram ───────
    rows = [
        ("t1", t1_raw, t1_n4, t1_ws, t1_mask),
        ("t2", t2_raw, t2_n4, t2_ws, t2_mask),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for row, (mod_name, raw, n4, ws, mask) in enumerate(rows):
        mid = raw.shape[2] // 2

        axes[row, 0].imshow(raw[:, :, mid].T, cmap="gray", origin="lower")
        axes[row, 0].set_title(f"{mod_name} — raw")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(n4[:, :, mid].T, cmap="gray", origin="lower")
        axes[row, 1].set_title(f"{mod_name} — N4 corrected")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(ws[:, :, mid].T, cmap="gray", origin="lower")
        axes[row, 2].set_title(f"{mod_name} — Hybrid WhiteStripe")
        axes[row, 2].axis("off")

        # density=True so peak height reflects distribution SHAPE, not
        # this subject's brain/mask voxel count -- raw counts would make
        # a subject with a bigger brain look like it has a "bigger" peak
        # even with an identical intensity distribution.
        axes[row, 3].hist(ws[mask].ravel(), bins=100, color="steelblue", density=True)
        axes[row, 3].set_title(f"{mod_name} — Hybrid WS histogram (density)")
        axes[row, 3].axvline(0, color="red", linestyle="--", linewidth=0.8)
        axes[row, 3].set_ylabel("Density")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{sid}_check.png")
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"  Saved visual check: {out_path}")

    return {
        "sid": sid,
        "t1_ws_vals": t1_ws[t1_mask],
        "t2_ws_vals": t2_ws[t2_mask],
        "n_hybrid_voxels": ws_stats["n_voxels"],
        "fallback_used": ws_stats["fallback_used"],
    }


def check_cross_subject_overlay(results):
    """
    Overlay hybrid-WhiteStripe DENSITY histograms across the checked
    subjects, for T1 and T2, so you can eyeball whether they land on
    top of each other.
    """
    results = [r for r in results if r is not None]
    print(f"\n{'='*60}\nCross-subject Hybrid-WhiteStripe overlay "
          f"({len(results)} subjects)\n{'='*60}")

    if len(results) < 2:
        print("  [SKIP] need at least 2 successfully-processed subjects to overlay")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for r in results:
        axes[0].hist(r["t1_ws_vals"], bins=100, alpha=0.5, density=True, label=r["sid"])
        axes[1].hist(r["t2_ws_vals"], bins=100, alpha=0.5, density=True, label=r["sid"])

    axes[0].set_title("t1 — Hybrid WhiteStripe (density)")
    axes[1].set_title("t2 — Hybrid WhiteStripe (density)")
    axes[0].set_ylabel("Density")
    axes[1].set_ylabel("Density")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "hybrid_ws_overlay.png")
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"  Saved: {out_path}")
    print("  NOTE: if all CHECK_SUBJECTS are from the same site, this only "
          "checks within-site consistency, not cross-site harmonization -- "
          "add a TCGA/UCSF subject to CHECK_SUBJECTS/DATA_DIR handling to "
          "actually test cross-site convergence.")


if __name__ == "__main__":
    results = [check_one_subject(sid) for sid in CHECK_SUBJECTS]
    check_cross_subject_overlay(results)

    n_ok = sum(1 for r in results if r is not None)
    n_fallback = sum(1 for r in results if r is not None and r["fallback_used"])
    print(f"\nDone. {n_ok}/{len(CHECK_SUBJECTS)} subjects processed successfully, "
          f"{n_fallback} used the T1-only fallback.")
    print(f"Check the PNGs in {OUT_DIR}/ — pull them off the DGX to view, "
          f"e.g. scp or rsync to your local machine.")
