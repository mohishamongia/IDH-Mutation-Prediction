"""
Nyul & Udupa Piecewise-Linear Histogram Matching
==================================================
Standard MRI intensity harmonization: fit "landmark" percentiles on a
reference distribution (your UTSW training set), then map any new volume's
intensity histogram onto those same landmarks.

Unlike N4 (which fixes WITHIN-scan smooth shading), this fixes ACROSS-site
differences in overall intensity scale/contrast — e.g. TCGA's T1 sequences
having a systematically different white-matter/gray-matter intensity ratio
than UTSW's, purely due to different scanner vendors/protocols.

Usage:
    landmarks = fit_nyul_landmarks(reference_volumes, masks=reference_masks)
    vol_matched = apply_nyul(new_vol, landmarks, mask=new_mask)
"""

import numpy as np


DEFAULT_PERCENTILES = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]


# ═══════════════════════════════════════════════════════════════════════════
# FIT — build the standard histogram landmarks from a reference set
# ═══════════════════════════════════════════════════════════════════════════
def fit_nyul_landmarks(volumes, masks=None, percentiles=DEFAULT_PERCENTILES,
                        standard_range=(0, 100)):
    """
    Fit standard-scale landmarks from a set of reference volumes (typically
    a subset of your UTSW training set, since that's your "home" distribution
    the model was originally built around).

    Parameters
    ----------
    volumes : list of np.ndarray (D, H, W)
        Reference volumes to build the standard histogram from.
    masks : list of np.ndarray bool, optional
        Foreground/brain masks matching each volume. Strongly recommended —
        without it, background zeros skew the low percentiles badly.
    percentiles : list of float
        Percentile landmarks to track. The defaults (1-99, every 10) are the
        standard Nyul choice — dense enough to capture the histogram shape,
        sparse enough to be robust to noise at any single percentile.
    standard_range : tuple
        The common scale all volumes get mapped onto (min/max landmark
        positions). (0, 100) is an arbitrary but conventional choice — the
        actual numbers don't matter, only that everything shares the same
        scale afterward.

    Returns
    -------
    dict: {
        "percentiles": percentiles used,
        "standard_landmarks": the common target landmark values (same for
            every volume once mapped),
        "individual_landmarks": list of each reference volume's own raw
            landmark values (mean of these defines mapping in apply_nyul)
    }
    """
    if masks is None:
        masks = [v > 0 for v in volumes]

    all_landmarks = []
    for vol, mask in zip(volumes, masks):
        vals = vol[mask]
        lm = np.percentile(vals, percentiles)
        all_landmarks.append(lm)

    all_landmarks = np.array(all_landmarks)  # (n_volumes, n_percentiles)

    # Standard scale: map the mean min/max landmark positions onto
    # standard_range, and use the MEAN of each intermediate percentile
    # across the reference set as the common target for that percentile.
    mean_landmarks_raw = all_landmarks.mean(axis=0)

    p_lo, p_hi = standard_range
    lo_val, hi_val = mean_landmarks_raw[0], mean_landmarks_raw[-1]
    standard_landmarks = p_lo + (mean_landmarks_raw - lo_val) * (p_hi - p_lo) / (hi_val - lo_val + 1e-8)

    return {
        "percentiles": percentiles,
        "standard_landmarks": standard_landmarks.tolist(),
        "individual_landmarks": all_landmarks.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# APPLY — map a new volume onto the fitted standard scale
# ═══════════════════════════════════════════════════════════════════════════
def apply_nyul(vol, landmarks, mask=None):
    """
    Map a volume's intensities onto the standard scale via piecewise-linear
    interpolation between its OWN landmarks and the fitted standard ones.

    Parameters
    ----------
    vol : np.ndarray (D, H, W)
    landmarks : dict — output of fit_nyul_landmarks()
    mask : np.ndarray bool, optional

    Returns
    -------
    np.ndarray (D, H, W), float32 — intensity-matched volume. Background
    (outside mask) is passed through the same piecewise mapping via
    np.interp's edge behavior, which is fine since it gets zeroed/ignored
    downstream by your existing min-max + masking steps anyway.
    """
    if mask is None:
        mask = vol > 0

    percentiles = landmarks["percentiles"]
    standard_landmarks = np.array(landmarks["standard_landmarks"])

    vals = vol[mask]
    own_landmarks = np.percentile(vals, percentiles)

    # Piecewise-linear map: this volume's own landmark values -> standard
    # landmark values. np.interp handles the piecewise part automatically
    # given sorted x (own_landmarks) and y (standard_landmarks) control points.
    flat = vol.flatten()
    mapped_flat = np.interp(flat, own_landmarks, standard_landmarks)
    mapped = mapped_flat.reshape(vol.shape)

    return mapped.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION — did landmarks actually converge across "sites"?
# ═══════════════════════════════════════════════════════════════════════════
def validate_nyul(volumes_by_site, landmarks, masks_by_site=None):
    """
    Checks whether, after applying Nyul matching, different sites'
    landmark positions actually converge onto the same standard scale.

    Parameters
    ----------
    volumes_by_site : dict {site_name: [volumes]}
    landmarks : dict — fitted landmarks (from fit_nyul_landmarks on
                reference/UTSW data)
    masks_by_site : dict {site_name: [masks]}, optional

    Returns
    -------
    dict with per-site landmark spread before/after, and the max
    cross-site landmark divergence before/after (the key summary number:
    lower after = harmonization is working)
    """
    if masks_by_site is None:
        masks_by_site = {s: [v > 0 for v in vs] for s, vs in volumes_by_site.items()}

    percentiles = landmarks["percentiles"]
    site_names = list(volumes_by_site.keys())

    before_means, after_means = {}, {}
    for site in site_names:
        vols = volumes_by_site[site]
        masks = masks_by_site[site]

        raw_lms, matched_lms = [], []
        for vol, mask in zip(vols, masks):
            raw_lms.append(np.percentile(vol[mask], percentiles))
            matched = apply_nyul(vol, landmarks, mask=mask)
            matched_lms.append(np.percentile(matched[mask], percentiles))

        before_means[site] = np.mean(raw_lms, axis=0)
        after_means[site] = np.mean(matched_lms, axis=0)

    # Cross-site divergence at each percentile: max - min across sites
    before_arr = np.array([before_means[s] for s in site_names])  # (n_sites, n_perc)
    after_arr = np.array([after_means[s] for s in site_names])

    before_divergence = (before_arr.max(axis=0) - before_arr.min(axis=0))
    after_divergence = (after_arr.max(axis=0) - after_arr.min(axis=0))

    report = {
        "site_names": site_names,
        "percentiles": percentiles,
        "landmarks_before": {s: before_means[s].tolist() for s in site_names},
        "landmarks_after": {s: after_means[s].tolist() for s in site_names},
        "max_cross_site_divergence_before": float(before_divergence.max()),
        "max_cross_site_divergence_after": float(after_divergence.max()),
        "mean_cross_site_divergence_before": float(before_divergence.mean()),
        "mean_cross_site_divergence_after": float(after_divergence.mean()),
    }
    return report


def print_nyul_report(report):
    print("Nyul Cross-Site Harmonization Validation")
    print("=" * 55)
    print(f"  Sites: {report['site_names']}")
    print()
    print(f"  Cross-site landmark divergence (lower = better harmonized)")
    print(f"    max divergence  before: {report['max_cross_site_divergence_before']:.4f}")
    print(f"    max divergence  after : {report['max_cross_site_divergence_after']:.4f}")
    print(f"    mean divergence before: {report['mean_cross_site_divergence_before']:.4f}")
    print(f"    mean divergence after : {report['mean_cross_site_divergence_after']:.4f}")
    reduction = 100 * (report['mean_cross_site_divergence_before'] - report['mean_cross_site_divergence_after']) / (report['mean_cross_site_divergence_before'] + 1e-8)
    print(f"    -> {reduction:+.1f}% change in mean divergence "
          f"({'improved' if reduction > 0 else 'WORSE — check reference set / masks'})")
    print("=" * 55)


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST — synthetic "3 sites" with different intensity scales
# ═══════════════════════════════════════════════════════════════════════════
def _make_synthetic_site_volume(size=48, scale=1.0, shift=0.0, seed=0):
    """Same underlying anatomy, but different intensity scale/shift per
    'site' — simulating scanner-to-scanner intensity differences."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.meshgrid(
        np.linspace(-1, 1, size), np.linspace(-1, 1, size), np.linspace(-1, 1, size),
        indexing="ij",
    )
    sphere = (zz**2 + yy**2 + xx**2) < 0.6**2
    inner = (zz**2 + yy**2 + xx**2) < 0.3**2  # simulate a second "tissue type"

    vol = np.zeros((size, size, size), dtype=np.float32)
    vol[sphere] = 60 + rng.normal(0, 4, sphere.sum())
    vol[inner] = 120 + rng.normal(0, 4, inner.sum())

    vol = vol * scale + shift
    vol[vol < 0] = 0
    return vol, sphere


if __name__ == "__main__":
    print("Running self-test: 3 synthetic 'sites' with different intensity scales...\n")

    # Simulate UTSW (reference), TCGA, UCSF each with a different scale/shift
    site_params = {
        "UTSW": {"scale": 1.0, "shift": 0.0, "seed": 1},
        "TCGA": {"scale": 1.8, "shift": 15.0, "seed": 2},
        "UCSF": {"scale": 0.6, "shift": -5.0, "seed": 3},
    }

    volumes_by_site, masks_by_site = {}, {}
    for site, p in site_params.items():
        vols, masks = [], []
        for i in range(5):
            v, m = _make_synthetic_site_volume(scale=p["scale"], shift=p["shift"], seed=p["seed"] * 10 + i)
            vols.append(v)
            masks.append(m)
        volumes_by_site[site] = vols
        masks_by_site[site] = masks

    # Fit landmarks on UTSW only (the reference site)
    landmarks = fit_nyul_landmarks(volumes_by_site["UTSW"], masks=masks_by_site["UTSW"])

    report = validate_nyul(volumes_by_site, landmarks, masks_by_site=masks_by_site)
    print_nyul_report(report)