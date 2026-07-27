"""
Hybrid WhiteStripe Normalization (T1 + T2, optionally + FLAIR)
================================================================
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
    side="low":  WM is the lowest-intensity major mode (T2/FLAIR-like
                 contrast, WM darker than GM/CSF).

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
                            flair_vol=None, flair_mask=None,
                            width_pct=0.06, min_voxels=200):
    """
    Compute the hybrid WhiteStripe reference voxel set and normalization
    stats for one subject. Assumes all volumes are already co-registered to
    the same voxel grid, skull-stripped, and N4-corrected (do N4 BEFORE
    calling this).

    If flair_vol/flair_mask are provided, tries a 3-way T1 ∩ T2 ∩ FLAIR
    intersection first (the strongest WM reference), and falls back
    gracefully if it collapses:

        level 0: T1 ∩ T2 ∩ FLAIR   (only tried if flair_vol is given)
        level 1: T1 ∩ T2           (dropped FLAIR first -- FLAIR is
                                     T2-weighted with CSF suppressed, so it
                                     mostly duplicates T2's contrast
                                     direction rather than adding an
                                     independent check the way T1 does)
        level 2: T1 only

    If flair_vol is None, behaves exactly as before: tries T1 ∩ T2, then
    falls back to T1-only (this is level 1 -> level 2 in the numbering
    above, but reported as fallback_level 0 -> 1 for backward
    compatibility with code that checks `fallback_used`).

    Returns a dict:
      {
        "hybrid_mask": bool array (same shape as t1_vol/t2_vol),
        "n_voxels": int,
        "modalities_used": list[str],   # e.g. ["t1","t2","flair"] or ["t1"]
        "fallback_level": int,          # 0 = best case for whatever inputs
                                         # were given, higher = more fallback
        "fallback_used": bool,          # True if fallback_level > 0
        "t1_mean": float, "t1_std": float, "t1_peak": float,
        "t2_mean": float, "t2_std": float, "t2_peak": float,
        # present only if flair_vol was given:
        "flair_mean": float, "flair_std": float, "flair_peak": float,
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

    have_flair = flair_vol is not None and flair_mask is not None
    flair_peak = None
    if have_flair:
        flair_vals = flair_vol[flair_mask]
        flair_peak, _, _ = _find_tissue_mode(flair_vals, side="low")
        flair_lo, flair_hi = _window_around_peak(flair_peak, width_pct)
        flair_window_mask = flair_mask & (flair_vol >= flair_lo) & (flair_vol <= flair_hi)

    # Build the ordered list of (modalities, mask) candidates to try.
    if have_flair:
        candidates = [
            (["t1", "t2", "flair"], t1_window_mask & t2_window_mask & flair_window_mask),
            (["t1", "t2"], t1_window_mask & t2_window_mask),
            (["t1"], t1_window_mask),
        ]
    else:
        candidates = [
            (["t1", "t2"], t1_window_mask & t2_window_mask),
            (["t1"], t1_window_mask),
        ]

    hybrid_mask, n_voxels, modalities_used, fallback_level = None, 0, None, None
    for level, (mods, mask) in enumerate(candidates):
        n = int(mask.sum())
        if n >= min_voxels:
            hybrid_mask, n_voxels, modalities_used, fallback_level = mask, n, mods, level
            break

    if hybrid_mask is None:
        raise ValueError(
            f"WhiteStripe reference set too small even at the last fallback "
            f"level ({candidates[-1][0]}) -- check masks/registration for "
            f"this subject."
        )

    t1_mean = float(t1_vol[hybrid_mask].mean())
    t1_std = float(t1_vol[hybrid_mask].std())
    t2_mean = float(t2_vol[hybrid_mask].mean())
    t2_std = float(t2_vol[hybrid_mask].std())

    if t1_std == 0 or t2_std == 0:
        raise ValueError("Zero variance in WhiteStripe reference set -- "
                          "check for constant/corrupted input volume.")

    result = {
        "hybrid_mask": hybrid_mask,
        "n_voxels": n_voxels,
        "modalities_used": modalities_used,
        "fallback_level": fallback_level,
        "fallback_used": fallback_level > 0,
        "t1_mean": t1_mean, "t1_std": t1_std, "t1_peak": float(t1_peak),
        "t2_mean": t2_mean, "t2_std": t2_std, "t2_peak": float(t2_peak),
    }

    if have_flair:
        flair_mean = float(flair_vol[hybrid_mask].mean())
        flair_std = float(flair_vol[hybrid_mask].std())
        if flair_std == 0:
            raise ValueError("Zero variance in FLAIR over the WhiteStripe "
                              "reference set -- check for constant/corrupted "
                              "input volume.")
        result["flair_mean"] = flair_mean
        result["flair_std"] = flair_std
        result["flair_peak"] = float(flair_peak)

    return result


def apply_whitestripe(vol, mean, std):
    """Z-score the full volume using the WM reference mean/std."""
    return (vol - mean) / std
