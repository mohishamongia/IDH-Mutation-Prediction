"""
N4 Bias Field Correction + Validation
======================================
Drop-in addition to preprocess_and_cache() in idh_cv_train.py.

Usage:
    from n4_correction import n4_correct, validate_correction

    vol_corrected = n4_correct(vol_path_or_array)
    report = validate_correction(vol_before, vol_after, mask=brain_mask)
"""

import numpy as np
import SimpleITK as sitk


# ═══════════════════════════════════════════════════════════════════════════
# CORE CORRECTION
# ═══════════════════════════════════════════════════════════════════════════
def n4_correct(vol, mask=None, shrink_factor=4, num_iterations=(50, 50, 50, 50)):
    """
    Apply N4ITK bias field correction to a single 3D volume.

    Parameters
    ----------
    vol : np.ndarray (D, H, W), float
        Raw volume (e.g. loaded via nib.load(...).get_fdata()).
    mask : np.ndarray (D, H, W), bool, optional
        Brain/foreground mask. Strongly recommended — without it N4 estimates
        bias over background/air too, which wastes iterations and can distort
        the estimate near the skull-strip boundary. If you already skull-strip
        (your `brain_t1_ants.nii.gz` files suggest you do), a simple `vol > 0`
        mask is normally enough.
    shrink_factor : int
        Downsample factor for the internal bias-field estimation (speed vs.
        precision trade-off). 4 is the SimpleITK default and standard choice
        for 96-256^3 MRI volumes.
    num_iterations : tuple of int
        Iterations per resolution level in the multi-resolution fitting.

    Returns
    -------
    np.ndarray (D, H, W), float32 — bias-corrected volume, same shape as input.
    """
    sitk_img = sitk.GetImageFromArray(vol.astype(np.float32))

    if mask is None:
        mask_arr = (vol > 0).astype(np.uint8)
    else:
        mask_arr = mask.astype(np.uint8)
    sitk_mask = sitk.GetImageFromArray(mask_arr)

    # Shrink for speed, correct, then the filter's log-bias-field is resampled
    # back up to full resolution internally when we call .Execute at full res
    # below — the shrink only affects fitting speed, not output resolution.
    shrunk_img  = sitk.Shrink(sitk_img,  [shrink_factor] * 3)
    shrunk_mask = sitk.Shrink(sitk_mask, [shrink_factor] * 3)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(list(num_iterations))
    corrector.Execute(shrunk_img, shrunk_mask)

    # Get the estimated bias field (log domain) and apply it to the FULL
    # resolution original image — this is the standard N4 recipe for doing
    # fast estimation + full-res correction.
    log_bias_field = corrector.GetLogBiasFieldAsImage(sitk_img)
    corrected = sitk_img / sitk.Exp(log_bias_field)

    return sitk.GetArrayFromImage(corrected).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION — did it actually do anything?
# ═══════════════════════════════════════════════════════════════════════════
def validate_correction(vol_before, vol_after, mask=None, n_bins=8):
    """
    Quantitative check that N4 correction reduced intensity inhomogeneity.

    Core metric: coefficient of variation (CV = std/mean) within a tissue
    region that SHOULD be intensity-homogeneous. If the correction worked,
    CV after should be lower than CV before, since we're removing a spatial
    shading pattern that was inflating the spread of an otherwise-uniform
    tissue's intensities.

    Also reports a spatial check: mean intensity in 8 octants of the volume.
    A real bias field shows up as octants with systematically different
    means; after correction, octant means should converge toward each other.

    Parameters
    ----------
    vol_before, vol_after : np.ndarray (D, H, W)
    mask : np.ndarray bool, optional — restricts CV calc to brain tissue
           (foreground). Without it, background zeros dominate the stats.

    Returns
    -------
    dict with cv_before, cv_after, cv_reduction_pct, octant_means_before,
    octant_means_after, octant_range_before, octant_range_after
    """
    if mask is None:
        mask = vol_before > 0

    vb = vol_before[mask]
    va = vol_after[mask]

    cv_before = float(vb.std() / (vb.mean() + 1e-8))
    cv_after  = float(va.std() / (va.mean() + 1e-8))
    cv_reduction_pct = 100 * (cv_before - cv_after) / (cv_before + 1e-8)

    # Octant means: split volume into 8 spatial blocks, compare mean
    # intensity across them. Real bias field => octants differ; after
    # correction the spread across octants should shrink.
    def octant_means(vol, mask):
        D, H, W = vol.shape
        dm, hm, wm = D // 2, H // 2, W // 2
        means = []
        for d0, d1 in [(0, dm), (dm, D)]:
            for h0, h1 in [(0, hm), (hm, H)]:
                for w0, w1 in [(0, wm), (wm, W)]:
                    block_v = vol[d0:d1, h0:h1, w0:w1]
                    block_m = mask[d0:d1, h0:h1, w0:w1]
                    if block_m.sum() > 0:
                        means.append(float(block_v[block_m].mean()))
                    else:
                        means.append(float("nan"))
        return means

    oct_before = octant_means(vol_before, mask)
    oct_after  = octant_means(vol_after, mask)
    range_before = float(np.nanmax(oct_before) - np.nanmin(oct_before))
    range_after  = float(np.nanmax(oct_after) - np.nanmin(oct_after))

    report = {
        "cv_before": round(cv_before, 4),
        "cv_after": round(cv_after, 4),
        "cv_reduction_pct": round(cv_reduction_pct, 2),
        "octant_means_before": [round(m, 4) for m in oct_before],
        "octant_means_after": [round(m, 4) for m in oct_after],
        "octant_range_before": round(range_before, 4),
        "octant_range_after": round(range_after, 4),
        "octant_range_reduction_pct": round(
            100 * (range_before - range_after) / (range_before + 1e-8), 2
        ),
    }
    return report


def print_validation_report(report):
    print("N4 Correction Validation")
    print("=" * 50)
    print(f"  CV (tissue homogeneity, lower=better)")
    print(f"    before : {report['cv_before']:.4f}")
    print(f"    after  : {report['cv_after']:.4f}")
    print(f"    change : {report['cv_reduction_pct']:+.2f}%  "
          f"({'improved' if report['cv_reduction_pct'] > 0 else 'WORSE — check mask/params'})")
    print()
    print(f"  Octant intensity range (spatial shading, lower=better)")
    print(f"    before : {report['octant_range_before']:.4f}")
    print(f"    after  : {report['octant_range_after']:.4f}")
    print(f"    change : {report['octant_range_reduction_pct']:+.2f}%")
    print("=" * 50)


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST on synthetic data with a KNOWN bias field
# ═══════════════════════════════════════════════════════════════════════════
def _make_synthetic_test_volume(size=64):
    """
    Builds a synthetic volume: uniform-intensity 'tissue' sphere with a known
    smooth multiplicative bias field applied, so we know the ground truth
    (before correction: has bias; after: should approach uniform again).
    """
    rng = np.random.default_rng(0)
    zz, yy, xx = np.meshgrid(
        np.linspace(-1, 1, size),
        np.linspace(-1, 1, size),
        np.linspace(-1, 1, size),
        indexing="ij",
    )
    sphere_mask = (zz**2 + yy**2 + xx**2) < 0.6**2

    # "True" tissue: uniform intensity ~100 plus small noise
    true_tissue = np.zeros((size, size, size), dtype=np.float32)
    true_tissue[sphere_mask] = 100.0 + rng.normal(0, 3, sphere_mask.sum())

    # Known smooth multiplicative bias field: stronger on one side
    bias_field = 1.0 + 0.6 * (xx + 1) / 2  # ranges 1.0 -> 1.6 across x-axis

    biased = true_tissue * bias_field
    return biased.astype(np.float32), sphere_mask, true_tissue


if __name__ == "__main__":
    print("Running self-test on synthetic volume with known bias field...\n")
    biased_vol, mask, ground_truth = _make_synthetic_test_volume(size=64)

    corrected_vol = n4_correct(biased_vol, mask=mask)

    report = validate_correction(biased_vol, corrected_vol, mask=mask)
    print_validation_report(report)

    # Sanity check against ground truth (only possible because this is
    # synthetic — you won't have ground truth on real scans, that's why
    # the CV/octant checks above are the ones you'll actually use)
    gt_vals = ground_truth[mask]
    corrected_vals = corrected_vol[mask]
    # Correlate shapes rather than compare absolute scale (N4 doesn't
    # guarantee matching the original absolute intensity scale)
    corrected_normalized = corrected_vals / corrected_vals.mean() * gt_vals.mean()
    residual = np.abs(corrected_normalized - gt_vals).mean()
    print(f"\n[Synthetic-only check] Mean abs residual vs. known ground truth: "
          f"{residual:.4f}  (uncorrected residual was "
          f"{np.abs(biased_vol[mask] - gt_vals).mean():.4f})")