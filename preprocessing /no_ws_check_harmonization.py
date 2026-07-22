"""
check_harmonization.py

Standalone sanity check for the N4 -> Nyul pipeline (WhiteStripe stage
removed -- Nyul is fit and applied directly on N4-corrected volumes).
Does NOT touch idh_cv_train.py, does NOT build the full cache, does NOT
train anything. Just runs harmonization on a handful of subjects and
prints/plots enough for you to eyeball whether it's working correctly
before committing DGX time to the full 622-subject cache build.

Run: python3 check_harmonization.py
"""

import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")   # no display on DGX; saves PNGs instead
import matplotlib.pyplot as plt

from no_ws_harmonization import (
    n4_bias_correct,
    NyulNormalizer,
    MODALITY_FILES,
)

# ── EDIT THESE THREE LINES ───────────────────────────────────────────────────
DATA_DIR = "/workspace/DATASETS/UTSW_Glioma_data/UTSW-Glioma"
CHECK_SUBJECTS = ["BT0001", "BT0002", "BT0003"]  # picked from your BT000X naming convention
OUT_DIR = "no_ws_harmonization_check"
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)


def _n4_for_subject(sid):
    """Runs N4 on all 4 modalities for one subject, returns dict + masks."""
    folder = os.path.join(DATA_DIR, sid)
    n4_vols, masks, raws = {}, {}, {}
    for mod_name, fname in MODALITY_FILES.items():
        in_path = os.path.join(folder, fname)
        if not os.path.exists(in_path):
            print(f"  [SKIP] {fname} not found for {sid}")
            continue
        raws[mod_name] = nib.load(in_path).get_fdata(dtype=np.float32)
        vol_n4 = n4_bias_correct(in_path)
        n4_vols[mod_name] = vol_n4
        masks[mod_name] = (vol_n4 > 0).astype(np.uint8)
    return raws, n4_vols, masks


def check_one_subject(sid, nyul_normalizers):
    print(f"\n{'='*60}\nSubject: {sid}\n{'='*60}")
    raws, n4_vols, masks = _n4_for_subject(sid)

    fig, axes = plt.subplots(len(MODALITY_FILES), 4, figsize=(16, 4 * len(MODALITY_FILES)))

    for row, mod_name in enumerate(MODALITY_FILES):
        if mod_name not in n4_vols:
            continue

        raw = raws[mod_name]
        vol_n4 = n4_vols[mod_name]
        mask = masks[mod_name]
        mid = raw.shape[2] // 2

        # Nyul, applied directly on the N4-corrected volume (no WhiteStripe)
        vol_nyul = nyul_normalizers[mod_name].transform(vol_n4, mask)

        print(f"  {mod_name:6s} | raw range [{raw.min():.1f}, {raw.max():.1f}] "
              f"| N4 range [{vol_n4.min():.1f}, {vol_n4.max():.1f}] "
              f"| N4+Nyul range [{vol_nyul.min():.2f}, {vol_nyul.max():.2f}]")

        # ── plot: raw, N4, N4+Nyul image, N4+Nyul histogram ─────────────────
        axes[row, 0].imshow(raw[:, :, mid].T, cmap="gray", origin="lower")
        axes[row, 0].set_title(f"{mod_name} — raw")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(vol_n4[:, :, mid].T, cmap="gray", origin="lower")
        axes[row, 1].set_title(f"{mod_name} — N4 corrected")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(vol_nyul[:, :, mid].T, cmap="gray", origin="lower")
        axes[row, 2].set_title(f"{mod_name} — N4 + Nyul")
        axes[row, 2].axis("off")

        axes[row, 3].hist(vol_nyul[mask > 0].ravel(), bins=100, color="steelblue")
        axes[row, 3].set_title(f"{mod_name} — N4+Nyul histogram")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{sid}_check.png")
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"  Saved visual check: {out_path}")


def check_nyul_overlay(subjects):
    """
    Fits Nyul on the given subjects (small subset, just for this check) and
    plots the raw-N4 histograms BEFORE Nyul vs AFTER Nyul, overlaid across
    subjects. If harmonization is working, the "after" histograms should
    line up much more closely than the "before" ones.

    Returns the fitted normalizers so check_one_subject() can reuse them
    without re-fitting.
    """
    print(f"\n{'='*60}\nNyul overlay check ({len(subjects)} subjects)\n{'='*60}")

    fitted_normalizers = {}

    for mod_name, fname in MODALITY_FILES.items():
        volumes, masks = [], []
        for sid in subjects:
            in_path = os.path.join(DATA_DIR, sid, fname)
            if not os.path.exists(in_path):
                continue
            vol_n4 = n4_bias_correct(in_path)
            mask = (vol_n4 > 0).astype(np.uint8)
            volumes.append(vol_n4)
            masks.append(mask)

        if len(volumes) < 2:
            print(f"  [SKIP] {mod_name}: need at least 2 subjects with this modality")
            continue

        normalizer = NyulNormalizer().fit(volumes, masks)
        fitted_normalizers[mod_name] = normalizer

        fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(12, 4))
        for vol, mask, sid in zip(volumes, masks, subjects):
            ax_before.hist(vol[mask > 0].ravel(), bins=100, alpha=0.5, label=sid)
            vol_nyul = normalizer.transform(vol, mask)
            ax_after.hist(vol_nyul[mask > 0].ravel(), bins=100, alpha=0.5, label=sid)

        ax_before.set_title(f"{mod_name} — before Nyul (N4 only)")
        ax_after.set_title(f"{mod_name} — after Nyul")
        ax_before.legend(fontsize=7)
        ax_after.legend(fontsize=7)
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"nyul_overlay_{mod_name}.png")
        plt.savefig(out_path, dpi=100)
        plt.close()
        print(f"  Saved: {out_path}  (compare left vs right — histograms should "
              f"align more tightly on the right if harmonization is helping)")

    return fitted_normalizers


if __name__ == "__main__":
    # Fit Nyul once on the check subjects (small-sample fit, just for this
    # sanity check -- your real fit_nyul_scales() run uses ~100 subjects).
    nyul_normalizers = check_nyul_overlay(CHECK_SUBJECTS)

    for sid in CHECK_SUBJECTS:
        check_one_subject(sid, nyul_normalizers)

    print(f"\nDone. Check the PNGs in {OUT_DIR}/ — pull them off the DGX to view, "
          f"e.g. scp or rsync to your local machine.")