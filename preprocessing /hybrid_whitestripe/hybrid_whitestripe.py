"""
Hybrid WhiteStripe Normalization (T1 + T2)
============================================
"""

import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema


def _find_tissue_mode(intensities, side="high", n_grid=1000,
                       bw_method=None, prominence_frac=0.1):
    """
    Fit a KDE to `intensities` and return the location of the WM peak.

    side="high": WM is the highest-intensity major mode (T1-like contrast,
                 WM brighter than GM/CSF).
    side="low":  WM is the lowest-intensity major mode (T2-like contrast,
                 WM darker than GM/CSF).

    prominence_frac filters out minor/noisy local maxima: only peaks with
    density >= prominence_frac * max_density are considered "major" tissue
    peaks (as opposed to small bumps from partial-volume voxels).
    """
    intensities = intensities[np.isfinite(intensities)]
    if intensities.size < 100:
        raise ValueError("Not enough voxels to fit a KDE reliably "
                          f"(got {intensities.size}).")

    kde = gaussian_kde(intensities, bw_method=bw_method)
    xs = np.linspace(intensities.min(), intensities.max(), n_grid)
    dens = kde(xs)

    peak_idx = argrelextrema(dens, np.greater)[0]
    if peak_idx.size == 0:
        # Degenerate/unimodal case -- fall back to the global max.
        peak_idx = np.array([np.argmax(dens)])

    major = peak_idx[dens[peak_idx] >= prominence_frac * dens.max()]
    if major.size == 0:
        major = peak_idx

    if side == "high":
        chosen = major[np.argmax(xs[major])]
    elif side == "low":
        chosen = major[np.argmin(xs[major])]
    else:
        raise ValueError("side must be 'high' or 'low'")

    return xs[chosen], xs, dens


def _window_around_peak(peak_val, width_pct=0.06):
    """
    Define the WM 'stripe' window as +/- width_pct of the peak value.
    width_pct is a fraction of the peak intensity itself (not the full
    histogram range), which keeps the window sensible across very
    different intensity scales between modalities/scanners.

    If your QC plots show the hybrid mask is too sparse or too broad,
    this is the parameter to tune (passed through from
    fit_whitestripe_hybrid).
    """
    half = width_pct * abs(peak_val)
    return peak_val - half, peak_val + half


def fit_whitestripe_hybrid(t1_vol, t2_vol, t1_mask, t2_mask,
                            width_pct=0.06, min_voxels=200):
    """
    Compute the hybrid WhiteStripe reference voxel set and normalization
    stats for one subject's T1 + T2 pair. Assumes T1/T2 are already
    co-registered to the same voxel grid, skull-stripped, and N4-corrected
    (do N4 BEFORE calling this).

    Returns a dict:
      {
        "hybrid_mask": bool array (same shape as t1_vol/t2_vol),
        "n_voxels": int,
        "t1_mean": float, "t1_std": float,
        "t2_mean": float, "t2_std": float,
        "t1_peak": float, "t2_peak": float,
        "fallback_used": bool  # True if intersection collapsed and we
                                # relaxed to T1-only WM voxels
      }
    """
    t1_vals = t1_vol[t1_mask]
    t2_vals = t2_vol[t2_mask]

    t1_peak, _, _ = _find_tissue_mode(t1_vals, side="high")
    t2_peak, _, _ = _find_tissue_mode(t2_vals, side="low")

    t1_lo, t1_hi = _window_around_peak(t1_peak, width_pct)
    t2_lo, t2_hi = _window_around_peak(t2_peak, width_pct)

    t1_window_mask = t1_mask & (t1_vol >= t1_lo) & (t1_vol <= t1_hi)
    t2_window_mask = t2_mask & (t2_vol >= t2_lo) & (t2_vol <= t2_hi)

    hybrid_mask = t1_window_mask & t2_window_mask
    n_voxels = int(hybrid_mask.sum())

    fallback_used = False
    if n_voxels < min_voxels:
        # Intersection collapsed -- likely large tumor/edema burden wiping
        # out overlap between the two windows. Fall back to T1-only so the
        # subject doesn't just fail outright, but flag it for QC review.
        fallback_used = True
        hybrid_mask = t1_window_mask
        n_voxels = int(hybrid_mask.sum())
        if n_voxels < min_voxels:
            raise ValueError(
                f"WhiteStripe reference set too small even after fallback "
                f"({n_voxels} voxels) -- check masks/registration for this subject."
            )

    t1_mean = float(t1_vol[hybrid_mask].mean())
    t1_std = float(t1_vol[hybrid_mask].std())
    t2_mean = float(t2_vol[hybrid_mask].mean())
    t2_std = float(t2_vol[hybrid_mask].std())

    if t1_std == 0 or t2_std == 0:
        raise ValueError("Zero variance in WhiteStripe reference set -- "
                          "check for constant/corrupted input volume.")

    return {
        "hybrid_mask": hybrid_mask,
        "n_voxels": n_voxels,
        "t1_mean": t1_mean, "t1_std": t1_std,
        "t2_mean": t2_mean, "t2_std": t2_std,
        "t1_peak": float(t1_peak), "t2_peak": float(t2_peak),
        "fallback_used": fallback_used,
    }


def apply_whitestripe(vol, mean, std):
    """Z-score the full volume using the WM reference mean/std."""
    return (vol - mean) / std
