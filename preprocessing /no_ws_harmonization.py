"""
harmonization.py

Intensity harmonization pipeline for multi-site MRI: N4 bias correction ->
WhiteStripe normalization -> Nyul histogram matching.

Designed to slot in before your existing preprocess_and_cache() step:
raw NIfTI -> [THIS MODULE] -> harmonized NIfTI (or in-memory array) -> your
existing zoom/cache logic.

Dependencies: SimpleITK, numpy, scipy, nibabel
    pip install SimpleITK --break-system-packages
"""

import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter1d


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1 — N4 BIAS FIELD CORRECTION
# ═══════════════════════════════════════════════════════════════════════════
def n4_bias_correct(in_path, out_path=None, mask_path=None,
                     shrink_factor=4, num_iterations=(50, 50, 50, 50)):
    """
    Runs N4ITK bias field correction on a single NIfTI volume.

    Parameters
    ----------
    in_path : str
        Path to input .nii.gz (e.g. brain_t1_ants.nii.gz)
    out_path : str or None
        If given, writes corrected volume here. If None, returns array only.
    mask_path : str or None
        Optional brain mask. If None, a simple Otsu mask is computed
        internally (fine for skull-stripped / brain-extracted volumes,
        which yours already are given the _ants naming convention).
    shrink_factor : int
        Downsamples the image before field estimation to speed things up;
        the field is then upsampled back. 4 is the standard default.
    num_iterations : tuple of int
        Max iterations per resolution level (coarse to fine).

    Returns
    -------
    np.ndarray (float32), same shape as input
    """
    img = sitk.ReadImage(in_path, sitk.sitkFloat32)

    if mask_path is not None:
        mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)
    else:
        # Otsu threshold as a stand-in brain mask. Since your volumes are
        # already skull-stripped (_ants suggests ANTs brain extraction),
        # this just excludes zero-background voxels from field estimation.
        mask = sitk.OtsuThreshold(img, 0, 1, 200)

    img_shrunk = sitk.Shrink(img, [shrink_factor] * img.GetDimension())
    mask_shrunk = sitk.Shrink(mask, [shrink_factor] * img.GetDimension())

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(list(num_iterations))
    corrected_shrunk = corrector.Execute(img_shrunk, mask_shrunk)

    # Recover the full-resolution bias field and apply it to the original
    log_bias_field = corrector.GetLogBiasFieldAsImage(img)
    corrected_full = img / sitk.Exp(log_bias_field)

    arr = sitk.GetArrayFromImage(corrected_full).astype(np.float32)
    # sitk array axis order is (z,y,x); transpose back to (x,y,z) to match nibabel
    arr = np.transpose(arr, (2, 1, 0))

    if out_path is not None:
        ref = nib.load(in_path)
        nib.save(nib.Nifti1Image(arr, ref.affine, ref.header), out_path)

    return arr


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2 — WHITESTRIPE NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════
def _smoothed_histogram(voxels, n_bins=200, sigma=2.0):
    """Histogram of in-mask voxels, lightly smoothed to find stable modes."""
    counts, edges = np.histogram(voxels, bins=n_bins)
    counts = gaussian_filter1d(counts.astype(np.float64), sigma=sigma)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


def whitestripe_normalize(volume, brain_mask, modality="t1", width=0.05):
    """
    WhiteStripe normalization (Shinohara et al., 2014).

    Finds the white-matter-associated mode of the intensity histogram,
    takes a narrow "stripe" of voxels around it, and z-scores the whole
    volume using only that stripe's mean/std.

    Parameters
    ----------
    volume : np.ndarray
        3D array, already N4-corrected.
    brain_mask : np.ndarray (bool or 0/1)
        Same shape as volume. Excludes background from mode-finding.
    modality : str
        "t1" -> WM is the highest-intensity mode (T1-weighted contrast).
        "t2" or "flair" -> WM is typically the lower-intensity mode
        relative to CSF/edema; here we take the mode in the upper-middle
        of the in-mask distribution as a practical proxy. For your 4
        modalities (T1, T1CE, T2, FLAIR) run this per-channel with the
        matching flag.
    width : float
        Half-width of the stripe as a fraction of the mode location
        (e.g. 0.05 = +/-5% around the mode). Matches the "narrow band"
        parameter in the original WhiteStripe formulation.

    Returns
    -------
    np.ndarray, same shape as input, WhiteStripe-normalized
    """
    voxels = volume[brain_mask > 0]
    centers, counts = _smoothed_histogram(voxels)

    if modality == "t1":
        # WM peak = the highest mode above the median in-mask intensity
        candidate = centers > np.median(voxels)
    else:
        # T2 / FLAIR: WM peak tends to sit below the CSF/edema peak but
        # above the lowest background/tissue noise; take modes in the
        # 25th-75th percentile range as candidates.
        lo, hi = np.percentile(voxels, [25, 75])
        candidate = (centers > lo) & (centers < hi)

    candidate_counts = np.where(candidate, counts, -np.inf)
    mode_idx = np.argmax(candidate_counts)
    mu_wm = centers[mode_idx]

    stripe_lo, stripe_hi = mu_wm * (1 - width), mu_wm * (1 + width)
    stripe_voxels = voxels[(voxels >= stripe_lo) & (voxels <= stripe_hi)]

    if stripe_voxels.size < 10:
        # Fallback if the stripe is too thin (can happen on noisy/small
        # volumes) -- widen it once rather than silently failing.
        stripe_lo, stripe_hi = mu_wm * (1 - 2 * width), mu_wm * (1 + 2 * width)
        stripe_voxels = voxels[(voxels >= stripe_lo) & (voxels <= stripe_hi)]

    mu_ws, sigma_ws = stripe_voxels.mean(), stripe_voxels.std()
    if sigma_ws < 1e-6:
        sigma_ws = 1e-6  # guard against degenerate stripes

    normalized = (volume - mu_ws) / sigma_ws
    return normalized.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3 — NYUL & UDUPA HISTOGRAM MATCHING (piecewise-linear landmark scale)
# ═══════════════════════════════════════════════════════════════════════════
class NyulNormalizer:
    """
    Nyul & Udupa (1999) piecewise-linear histogram standardization.

    Two-phase use:
        1. fit(training_volumes, training_masks)  -- learns the standard
           scale ONCE from a representative training set.
        2. transform(volume, mask)                -- applied to every
           volume (train or test) at inference/preprocessing time.

    Do NOT re-fit per-site or per-subject -- the whole point is one shared
    standard scale across all sites, learned from your training data only,
    then applied uniformly (including to held-out cross-site data). This
    mirrors standard practice and avoids test-time leakage.
    """

    def __init__(self, percentiles=None, standard_scale=None):
        # Default landmarks: 1st, 99th percentile as endpoints, deciles between.
        self.percentiles = percentiles if percentiles is not None else \
            np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99])
        self.standard_scale = standard_scale  # learned in fit()

    def _get_landmarks(self, volume, mask):
        voxels = volume[mask > 0]
        return np.percentile(voxels, self.percentiles)

    def fit(self, volumes, masks):
        """
        volumes, masks : lists of np.ndarray (same length)
            Training set volumes (already N4 + WhiteStripe processed) and
            their brain masks.
        """
        all_landmarks = np.stack(
            [self._get_landmarks(v, m) for v, m in zip(volumes, masks)],
            axis=0
        )
        # Standard scale = mean landmark location across the training set
        self.standard_scale = all_landmarks.mean(axis=0)
        return self

    def transform(self, volume, mask):
        if self.standard_scale is None:
            raise RuntimeError("Call fit() before transform(), or load a "
                                "saved standard_scale.")

        src_landmarks = self._get_landmarks(volume, mask)
        tgt_landmarks = self.standard_scale

        out = np.zeros_like(volume, dtype=np.float32)
        flat_vol = volume.ravel()
        flat_out = out.ravel()

        # Piecewise-linear mapping across each landmark interval
        for i in range(len(src_landmarks) - 1):
            p_i, p_ip1 = src_landmarks[i], src_landmarks[i + 1]
            s_i, s_ip1 = tgt_landmarks[i], tgt_landmarks[i + 1]

            if i == 0:
                in_range = flat_vol <= p_ip1
            elif i == len(src_landmarks) - 2:
                in_range = flat_vol > p_i
            else:
                in_range = (flat_vol > p_i) & (flat_vol <= p_ip1)

            denom = (p_ip1 - p_i) if (p_ip1 - p_i) != 0 else 1e-6
            flat_out[in_range] = s_i + (flat_vol[in_range] - p_i) * \
                                  (s_ip1 - s_i) / denom

        return flat_out.reshape(volume.shape)

    def save(self, path):
        np.save(path, {"percentiles": self.percentiles,
                        "standard_scale": self.standard_scale})

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True).item()
        return cls(percentiles=d["percentiles"], standard_scale=d["standard_scale"])


# ═══════════════════════════════════════════════════════════════════════════
# FULL PIPELINE — wraps N4 + Nyul per-modality (WhiteStripe stage removed)
# ═══════════════════════════════════════════════════════════════════════════
MODALITY_FILES = {
    "t1":    "brain_t1_ants.nii.gz",
    "t1ce":  "brain_t1ce_ants.nii.gz",
    "t2":    "brain_t2_ants.nii.gz",
    "flair": "brain_fl_ants.nii.gz",
}
# Kept for reference / in case you want to re-add WhiteStripe later --
# not used by harmonize_subject() or fit_nyul_scales() anymore.
MODALITY_KIND = {"t1": "t1", "t1ce": "t1", "t2": "t2", "flair": "t2"}


def harmonize_subject(subject_dir, nyul_normalizers, n4_kwargs=None):
    """
    Runs N4 -> Nyul for all 4 modalities of one subject.
    (WhiteStripe stage removed -- Nyul is applied directly to the
    N4-corrected volume.)

    Parameters
    ----------
    subject_dir : str
        Folder containing the 4 _ants.nii.gz files for one subject.
    nyul_normalizers : dict[str, NyulNormalizer]
        One FITTED NyulNormalizer per modality (t1, t1ce, t2, flair),
        each fit once on your training cohort (see fit_nyul_scales below).
    n4_kwargs : dict or None
        Extra kwargs forwarded to n4_bias_correct.

    Returns
    -------
    dict[str, np.ndarray] : harmonized volumes keyed by modality name
    """
    n4_kwargs = n4_kwargs or {}
    harmonized = {}

    for mod_name, fname in MODALITY_FILES.items():
        in_path = os.path.join(subject_dir, fname)

        # Stage 1: N4
        vol_n4 = n4_bias_correct(in_path, **n4_kwargs)

        # Brain mask: nonzero voxels (volumes are already skull-stripped)
        mask = (vol_n4 > 0).astype(np.uint8)

        # Stage 2: Nyul (uses the pre-fitted normalizer for this modality),
        # applied directly on the N4-corrected volume.
        vol_nyul = nyul_normalizers[mod_name].transform(vol_n4, mask)

        harmonized[mod_name] = vol_nyul

    return harmonized


def fit_nyul_scales(subject_dirs, save_dir, n4_kwargs=None, n_fit_subjects=None):
    """
    One-time fitting step: runs N4 on a training subset, then fits a
    NyulNormalizer per modality directly on the N4-corrected volumes
    (no WhiteStripe stage), and saves the standard scales to disk so
    harmonize_subject() can reuse them for every subject (train AND
    cross-site test) without re-fitting.

    Parameters
    ----------
    subject_dirs : list[str]
        Training subject folders (UTSW cohort -- fit on train split only).
    save_dir : str
        Where to write t1_nyul.npy, t1ce_nyul.npy, t2_nyul.npy, flair_nyul.npy
    n_fit_subjects : int or None
        Optionally subsample (e.g. 100) for speed; fitting on your full
        ~500-subject training split isn't necessary for a stable scale.
    """
    os.makedirs(save_dir, exist_ok=True)
    n4_kwargs = n4_kwargs or {}

    if n_fit_subjects is not None:
        subject_dirs = subject_dirs[:n_fit_subjects]

    normalizers = {}
    for mod_name, fname in MODALITY_FILES.items():
        volumes, masks = [], []
        for sdir in subject_dirs:
            in_path = os.path.join(sdir, fname)
            vol_n4 = n4_bias_correct(in_path, **n4_kwargs)
            mask = (vol_n4 > 0).astype(np.uint8)
            volumes.append(vol_n4)
            masks.append(mask)

        normalizer = NyulNormalizer().fit(volumes, masks)
        normalizer.save(os.path.join(save_dir, f"{mod_name}_nyul.npy"))
        normalizers[mod_name] = normalizer
        print(f"[Nyul fit] {mod_name}: standard_scale = {normalizer.standard_scale}")

    return normalizers


def load_nyul_scales(save_dir):
    """Loads previously-fitted NyulNormalizers for all 4 modalities."""
    return {
        mod_name: NyulNormalizer.load(os.path.join(save_dir, f"{mod_name}_nyul.npy"))
        for mod_name in MODALITY_FILES
    }