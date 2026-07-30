"""
check_hybrid_whitestripe.py

Standalone sanity check for the N4 -> Hybrid WhiteStripe (T1 ∩ T2) pipeline.
Does NOT touch idh_cv_train.py, does NOT build the full cache, does NOT
train anything. Just runs harmonization on a handful of subjects and
prints/plots enough for you to eyeball whether it's working correctly
before committing DGX time to the full 622-subject cache build.

Requires whitestripe_normalization.py (fit_whitestripe_hybrid,
apply_whitestripe) alongside this script.

Run: python3 check_hybrid_whitestripe.py
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
CHECK_SUBJECTS = ["BT0001", "BT0002", "BT0003"]  # picked from your BT000X naming convention
OUT_DIR = "Normalisation_testing/hybrid_whitestripe/T1_T2_Check"
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)


def _harmonize_subject(sid):
    """
    Runs N4 -> Hybrid WhiteStripe (T1 ∩ T2) for one subject.
    Returns (t1_raw, t2_raw, t1_n4, t2_n4, t1_ws, t2_ws, t1_mask, t2_mask,
    ws_stats), or None if this subject was skipped.
    """
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

    return t1_raw, t2_raw, t1_n4, t2_n4, t1_ws, t2_ws, t1_mask, t2_mask, ws_stats


def check_one_subject(sid, per_subject):
    print(f"\n{'='*60}\nSubject: {sid}\n{'='*60}")

    result = per_subject.get(sid)
    if result is None:
        print(f"  [SKIP] {sid}: no harmonized data")
        return

    (t1_raw, t2_raw, t1_n4, t2_n4, t1_ws, t2_ws,
     t1_mask, t2_mask, ws_stats) = result

    flag = " <-- fallback to T1-only (small T1∩T2 intersection)" if ws_stats["fallback_used"] else ""
    print(f"  hybrid WM reference voxels: {ws_stats['n_voxels']}{flag}")
    print(f"  t1 | raw range [{t1_raw.min():.1f}, {t1_raw.max():.1f}] "
          f"| N4 range [{t1_n4.min():.1f}, {t1_n4.max():.1f}] "
          f"| hybrid-WS range [{t1_ws.min():.2f}, {t1_ws.max():.2f}]")
    print(f"  t2 | raw range [{t2_raw.min():.1f}, {t2_raw.max():.1f}] "
          f"| N4 range [{t2_n4.min():.1f}, {t2_n4.max():.1f}] "
          f"| hybrid-WS range [{t2_ws.min():.2f}, {t2_ws.max():.2f}]")

    # ── plot: raw, N4, hybrid-WS image, hybrid-WS histogram ────────────────
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

        axes[row, 3].hist(ws[mask].ravel(), bins=100, color="steelblue")
        axes[row, 3].set_title(f"{mod_name} — Hybrid WS histogram")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{sid}_check.png")
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"  Saved visual check: {out_path}")


def check_hybrid_ws_overlay(subjects):
    """
    Harmonizes the given subjects (small subset, just for this check) and
    plots the raw-N4 histograms BEFORE hybrid WhiteStripe vs AFTER, overlaid
    across subjects. If harmonization is working, the "after" histograms
    should line up much more closely than the "before" ones.

    Returns the per-subject harmonized volumes/masks/stats so
    check_one_subject() can reuse them without re-running N4 + WhiteStripe.
    """
    print(f"\n{'='*60}\nHybrid WhiteStripe overlay check "
          f"({len(subjects)} subjects)\n{'='*60}")

    per_subject = {}
    for mod_name in ["t1", "t2"]:
        volumes, masks, sids = [], [], []
        for sid in subjects:
            if sid not in per_subject:
                per_subject[sid] = _harmonize_subject(sid)
            result = per_subject[sid]
            if result is None:
                continue
            (t1_raw, t2_raw, t1_n4, t2_n4, t1_ws, t2_ws,
             t1_mask, t2_mask, ws_stats) = result
            n4_vol = t1_n4 if mod_name == "t1" else t2_n4
            ws_vol = t1_ws if mod_name == "t1" else t2_ws
            mask = t1_mask if mod_name == "t1" else t2_mask
            volumes.append(n4_vol)
            masks.append((ws_vol, mask))
            sids.append(sid)

        if len(volumes) < 2:
            print(f"  [SKIP] {mod_name}: need at least 2 subjects with this modality")
            continue

        fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(12, 4))
        for n4_vol, (ws_vol, mask), sid in zip(volumes, masks, sids):
            ax_before.hist(n4_vol[mask].ravel(), bins=100, alpha=0.5, label=sid)
            ax_after.hist(ws_vol[mask].ravel(), bins=100, alpha=0.5, label=sid)

        ax_before.set_title(f"{mod_name} — before hybrid WS (N4 only)")
        ax_after.set_title(f"{mod_name} — after hybrid WS")
        ax_before.legend(fontsize=7)
        ax_after.legend(fontsize=7)
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"hybrid_ws_overlay_{mod_name}.png")
        plt.savefig(out_path, dpi=100)
        plt.close()
        print(f"  Saved: {out_path}  (compare left vs right — histograms should "
              f"align more tightly on the right if harmonization is helping)")

    return per_subject


if __name__ == "__main__":
    # Harmonize once on the check subjects and get the before/after overlay
    # (small-sample check, just for this sanity check).
    per_subject = check_hybrid_ws_overlay(CHECK_SUBJECTS)

    for sid in CHECK_SUBJECTS:
        check_one_subject(sid, per_subject)

    n_ok = sum(1 for r in per_subject.values() if r is not None)
    n_fallback = sum(1 for r in per_subject.values() if r is not None and r[-1]["fallback_used"])
    print(f"\nDone. {n_ok}/{len(CHECK_SUBJECTS)} subjects processed successfully, "
          f"{n_fallback} used the T1-only fallback.")
    print(f"Check the PNGs in {OUT_DIR}/ — pull them off the DGX to view, "
          f"e.g. scp or rsync to your local machine.")