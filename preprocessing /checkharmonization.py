"""
check_harmonization.py

Standalone sanity check for the N4 -> WhiteStripe -> Nyul pipeline.
Does NOT touch idh_cv_train.py, does NOT build the full cache, does NOT
train anything. Just runs harmonization on a handful of subjects and
prints/plots enough for you to eyeball whether it's working correctly
before committing DGX time to the full 622-subject cache build.

Run: python3 check_harmonization.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # no display on DGX; saves PNGs instead
import matplotlib.pyplot as plt

from Harmonization import (
    n4_bias_correct,
    whitestripe_normalize,
    NyulNormalizer,
    fit_nyul_scales,
    MODALITY_FILES,
    MODALITY_KIND,
)

# ── EDIT THESE THREE LINES ───────────────────────────────────────────────────
DATA_DIR = "/workspace/DATASETS/UTSW_Glioma_data/UTSW-Glioma"
CHECK_SUBJECTS = ["BT0001", "BT0002", "BT0003"]
OUT_DIR = "ws_harmonization_check"
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)


def check_one_subject(sid):
    print(f"\n{'='*60}\nSubject: {sid}\n{'='*60}")
    folder = os.path.join(DATA_DIR, sid)

    fig, axes = plt.subplots(len(MODALITY_FILES), 4, figsize=(16, 4 * len(MODALITY_FILES)))

    for row, (mod_name, fname) in enumerate(MODALITY_FILES.items()):
        in_path = os.path.join(folder, fname)
        if not os.path.exists(in_path):
            print(f"  [SKIP] {fname} not found for {sid}")
            continue

        # ── raw ──────────────────────────────────────────────────────────
        import nibabel as nib
        raw = nib.load(in_path).get_fdata(dtype=np.float32)
        mid = raw.shape[2] // 2

        # ── N4 ───────────────────────────────────────────────────────────
        vol_n4 = n4_bias_correct(in_path)
        mask = (vol_n4 > 0).astype(np.uint8)

        # ── WhiteStripe ──────────────────────────────────────────────────
        vol_ws = whitestripe_normalize(vol_n4, mask, modality=MODALITY_KIND[mod_name])

        print(f"  {mod_name:6s} | raw range [{raw.min():.1f}, {raw.max():.1f}] "
              f"| N4 range [{vol_n4.min():.1f}, {vol_n4.max():.1f}] "
              f"| WhiteStripe range [{vol_ws.min():.2f}, {vol_ws.max():.2f}]")

        # ── plot: raw, N4, WhiteStripe image, WhiteStripe histogram ────────
        axes[row, 0].imshow(raw[:, :, mid].T, cmap="gray", origin="lower")
        axes[row, 0].set_title(f"{mod_name} — raw")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(vol_n4[:, :, mid].T, cmap="gray", origin="lower")
        axes[row, 1].set_title(f"{mod_name} — N4 corrected")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(vol_ws[:, :, mid].T, cmap="gray", origin="lower")
        axes[row, 2].set_title(f"{mod_name} — WhiteStripe")
        axes[row, 2].axis("off")

        axes[row, 3].hist(vol_ws[mask > 0].ravel(), bins=100, color="steelblue")
        axes[row, 3].set_title(f"{mod_name} — WhiteStripe histogram")
        axes[row, 3].axvline(0, color="red", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{sid}_check.png")
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"  Saved visual check: {out_path}")


def check_nyul_overlay(subjects):
    """
    Fits Nyul on the given subjects (small subset, just for this check) and
    plots the WhiteStripe histograms BEFORE Nyul vs AFTER Nyul, overlaid
    across subjects. If harmonization is working, the "after" histograms
    should line up much more closely than the "before" ones.
    """
    print(f"\n{'='*60}\nNyul overlay check ({len(subjects)} subjects)\n{'='*60}")

    for mod_name, fname in MODALITY_FILES.items():
        volumes, masks = [], []
        for sid in subjects:
            in_path = os.path.join(DATA_DIR, sid, fname)
            if not os.path.exists(in_path):
                continue
            vol_n4 = n4_bias_correct(in_path)
            mask = (vol_n4 > 0).astype(np.uint8)
            vol_ws = whitestripe_normalize(vol_n4, mask, modality=MODALITY_KIND[mod_name])
            volumes.append(vol_ws)
            masks.append(mask)

        if len(volumes) < 2:
            print(f"  [SKIP] {mod_name}: need at least 2 subjects with this modality")
            continue

        normalizer = NyulNormalizer().fit(volumes, masks)

        fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(12, 4))
        for vol, mask, sid in zip(volumes, masks, subjects):
            ax_before.hist(vol[mask > 0].ravel(), bins=100, alpha=0.5, label=sid)
            vol_nyul = normalizer.transform(vol, mask)
            ax_after.hist(vol_nyul[mask > 0].ravel(), bins=100, alpha=0.5, label=sid)

        ax_before.set_title(f"{mod_name} — before Nyul (WhiteStripe only)")
        ax_after.set_title(f"{mod_name} — after Nyul")
        ax_before.legend(fontsize=7)
        ax_after.legend(fontsize=7)
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"nyul_overlay_{mod_name}.png")
        plt.savefig(out_path, dpi=100)
        plt.close()
        print(f"  Saved: {out_path}  (compare left vs right — histograms should "
              f"align more tightly on the right if harmonization is helping)")


if __name__ == "__main__":
    for sid in CHECK_SUBJECTS:
        check_one_subject(sid)

    check_nyul_overlay(CHECK_SUBJECTS)

    print(f"\nDone. Check the PNGs in {OUT_DIR}/ — pull them off the DGX to view, "
          f"e.g. scp or rsync to your local machine.")