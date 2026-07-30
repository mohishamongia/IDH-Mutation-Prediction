"""
harmonization.py  (v2 — N4 + Nyul, WhiteStripe removed)

Intensity harmonization pipeline for multi-site MRI: N4 bias correction ->
Nyul histogram matching. Drops WhiteStripe entirely and explicitly re-masks
background to 0.0 after every transform stage.

WHY THESE TWO CHANGES (context for anyone reading this later):

1. WhiteStripe removed.
   The original T2/FLAIR WM-mode heuristic picked a peak from the 25th-75th
   percentile band of in-mask intensity. That's a reasonable proxy for
   normal-appearing white matter in healthy controls, but on a glioma cohort
   a large, IDH-status-correlated fraction of brain volume can be T2/FLAIR
   hyperintense (edema, infiltrative/non-enhancing tumor). The "WM peak"
   this function locks onto can silently drift onto edema/tumor signal,
   with the amount of drift depending on tumor burden per subject -- i.e.
   it injects noise correlated with the very thing we're trying to predict,
   concentrated in the two modalities (T2/FLAIR) the model relies on most.
   Simplest fix for now: don't use it. N4 + Nyul alone is a well-established,
   safer harmonization baseline for pathological brains.

2. Explicit re-masking after every stage.
   Your training pipeline's modality dropout hardcodes "missing" channels to
   literal 0.0. Nyul's piecewise-linear transform is applied to the WHOLE
   volume (including background voxels outside the brain mask), and
   extrapolates below the 1st percentile landmark using the first segment's
   linear coefficients -- there's no guarantee raw background (0.0) maps
   back to 0.0 after that. If background drifts off zero, "missing modality
   = 0.0" in the dropout code no longer matches what real background looks
   like post-harmonization, and the model can't learn from it as a clean
   absence signal anymore. Fix: after N4 and after Nyul, forcibly zero
   everything outside the brain mask. Background is then guaranteed to be
   exactly 0.0 everywhere, downstream cache/training code needs no changes.

Dependencies: SimpleITK, numpy, scipy, nibabel
    pip install SimpleITK --break-system-packages
"""

import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1 — N4 BIAS FIELD CORRECTION
# ═══════════════════════════════════════════════════════════════════════════
def n4_bias_correct(in_path, out_path=None, mask_path=None,
                     shrink_factor=4, num_iterations=(50, 50, 50, 50)):
    """
    Runs N4ITK bias field correction on a single NIfTI volume.

    Returns
    -------
    (arr, mask) : (np.ndarray float32, np.ndarray uint8)
        Bias-corrected volume AND the brain mask used, both same shape.
        Background (mask == 0) is forced to exactly 0.0 in `arr` before
        returning -- N4's division by exp(bias_field) can otherwise leave
        tiny nonzero noise outside the brain on some volumes.
    """
    img = sitk.ReadImage(in_path, sitk.sitkFloat32)

    if mask_path is not None:
        mask_img = sitk.ReadImage(mask_path, sitk.sitkUInt8)
    else:
        # FIXED: the previous sitk.OtsuThreshold(img, 0, 1, 200) call passed
        # insideValue=0, outsideValue=1 -- that's backwards, and OtsuThreshold's
        # inside/outside convention is easy to get wrong. It silently labeled
        # brain tissue as 0 and background as 1, so the re-masking step below
        # (`arr = arr * mask`) was zeroing out the ENTIRE BRAIN and keeping
        # only background. That's why training collapsed to a constant
        # prediction -- the network was being fed blank volumes.
        #
        # Since these volumes are already skull-stripped, background is
        # already exactly 0 -- a plain, unambiguous threshold is both safer
        # and correct here, no Otsu convention to get backwards.
        mask_img = sitk.BinaryThreshold(img, lowerThreshold=1e-6,
                                         upperThreshold=1e10,
                                         insideValue=1, outsideValue=0)

    img_shrunk = sitk.Shrink(img, [shrink_factor] * img.GetDimension())
    mask_shrunk = sitk.Shrink(mask_img, [shrink_factor] * img.GetDimension())

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(list(num_iterations))
    corrector.Execute(img_shrunk, mask_shrunk)

    log_bias_field = corrector.GetLogBiasFieldAsImage(img)
    corrected_full = img / sitk.Exp(log_bias_field)

    arr = sitk.GetArrayFromImage(corrected_full).astype(np.float32)
    arr = np.transpose(arr, (2, 1, 0))  # sitk (z,y,x) -> nibabel (x,y,z)

    mask = sitk.GetArrayFromImage(mask_img).astype(np.uint8)
    mask = np.transpose(mask, (2, 1, 0))

    # Explicit re-mask: background must be exactly 0.0, not "close to it".
    arr = arr * mask

    if out_path is not None:
        ref = nib.load(in_path)
        nib.save(nib.Nifti1Image(arr, ref.affine, ref.header), out_path)

    return arr, mask


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2 — NYUL & UDUPA HISTOGRAM MATCHING (piecewise-linear landmark scale)
# ═══════════════════════════════════════════════════════════════════════════
class NyulNormalizer:
    """
    Nyul & Udupa (1999) piecewise-linear histogram standardization.

    Two-phase use:
        1. fit(volumes, masks)      -- learns the standard scale ONCE from
                                        a designated training subset.
        2. transform(volume, mask)  -- applied to every volume (the fitting
                                        subjects included, plus everyone
                                        else) at preprocessing time.

    Do NOT re-fit per-site or per-subject -- the whole point is one shared
    standard scale, learned once, applied uniformly everywhere.

    NOTE ON LEAKAGE: because your 5-fold CV rotates every subject through
    validation in some fold, any subject used to fit these landmarks has,
    in that fold, had its own intensities inform a preprocessing choice
    later applied to itself. This is a much milder leak than leaking labels
    (it only affects an intensity-rescaling constant, not the network's
    supervision signal), and is standard practice in the harmonization
    literature -- but it is worth flagging explicitly rather than silently
    assuming it away. If you want a fully leakage-free setup for the paper,
    fit a separate NyulNormalizer per CV fold (on that fold's train split
    only) and produce 5 separate harmonized caches. Not done here given
    time -- tracked as a TODO.
    """

    def __init__(self, percentiles=None, standard_scale=None):
        self.percentiles = percentiles if percentiles is not None else \
            np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99])
        self.standard_scale = standard_scale  # learned in fit()

    def _get_landmarks(self, volume, mask):
        voxels = volume[mask > 0]
        return np.percentile(voxels, self.percentiles)

    def fit(self, volumes, masks):
        all_landmarks = np.stack(
            [self._get_landmarks(v, m) for v, m in zip(volumes, masks)],
            axis=0
        )
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

        out = out.reshape(volume.shape)

        # Explicit re-mask: this is the critical fix. The piecewise-linear
        # map above is applied to every voxel including background, and
        # extrapolates outside [src_landmarks[0], src_landmarks[-1]] for
        # anything below the 1st or above the 99th percentile -- background
        # (raw 0.0) falls in the first segment and gets mapped to whatever
        # that segment's linear extrapolation gives, which is NOT
        # guaranteed to be 0.0 and varies slightly per subject. Force it
        # back to exactly 0.0 so "missing modality = 0.0" in the training
        # pipeline's dropout code stays meaningful.
        out = out * mask

        return out.astype(np.float32)

    def save(self, path):
        np.save(path, {"percentiles": self.percentiles,
                        "standard_scale": self.standard_scale})

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True).item()
        return cls(percentiles=d["percentiles"], standard_scale=d["standard_scale"])


# ═══════════════════════════════════════════════════════════════════════════
# FULL PIPELINE — wraps both stages per-modality
# ═══════════════════════════════════════════════════════════════════════════
MODALITY_FILES = {
    "t1":    "brain_t1_ants.nii.gz",
    "t1ce":  "brain_t1ce_ants.nii.gz",
    "t2":    "brain_t2_ants.nii.gz",
    "flair": "brain_fl_ants.nii.gz",
}


def harmonize_subject(subject_dir, nyul_normalizers, n4_kwargs=None):
    """
    Runs N4 -> Nyul for each modality of one subject.

    Parameters
    ----------
    subject_dir : str
        Folder containing the _ants.nii.gz files for one subject.
    nyul_normalizers : dict[str, NyulNormalizer]
        One FITTED NyulNormalizer per modality present in MODALITY_FILES
        (or a subset -- pass only the modalities you actually need, e.g.
        {"t1", "t2", "flair"} for the 3-modality pipeline).
    n4_kwargs : dict or None
        Extra kwargs forwarded to n4_bias_correct.

    Returns
    -------
    dict[str, np.ndarray] : harmonized volumes keyed by modality name,
        background guaranteed exactly 0.0 in every returned volume.
    """
    n4_kwargs = n4_kwargs or {}
    harmonized = {}

    for mod_name in nyul_normalizers:
        fname = MODALITY_FILES[mod_name]
        in_path = os.path.join(subject_dir, fname)

        vol_n4, mask = n4_bias_correct(in_path, **n4_kwargs)
        vol_nyul = nyul_normalizers[mod_name].transform(vol_n4, mask)

        harmonized[mod_name] = vol_nyul

    return harmonized


def fit_nyul_scales(subject_dirs, save_dir, modalities=("t1", "t2", "flair"),
                     n4_kwargs=None, n_fit_subjects=None, seed=42):
    """
    One-time fitting step: runs N4 on a designated subject list, then fits
    a NyulNormalizer per modality and saves the standard scales to disk so
    harmonize_subject() can reuse them for every subject without re-fitting.

    Parameters
    ----------
    subject_dirs : list[str]
        The subject folders to fit landmarks on. See the leakage note on
        NyulNormalizer above before deciding what to pass here -- if you
        want the strict per-fold version later, call this once per fold
        with that fold's train subjects only, into separate save_dirs.
    save_dir : str
        Where to write {modality}_nyul.npy for each modality.
    modalities : tuple[str]
        Which modalities to fit. Defaults to the 3-modality set (no T1CE),
        matching the current training pipeline.
    n_fit_subjects : int or None
        Optionally subsample for speed (e.g. 150-200 is plenty for a
        stable scale on 622 subjects); None uses everyone passed in.
    """
    os.makedirs(save_dir, exist_ok=True)
    n4_kwargs = n4_kwargs or {}

    if n_fit_subjects is not None:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(subject_dirs), size=min(n_fit_subjects, len(subject_dirs)),
                          replace=False)
        subject_dirs = [subject_dirs[i] for i in idx]

    normalizers = {}
    for mod_name in modalities:
        fname = MODALITY_FILES[mod_name]
        volumes, masks = [], []
        for sdir in subject_dirs:
            in_path = os.path.join(sdir, fname)
            vol_n4, mask = n4_bias_correct(in_path, **n4_kwargs)
            volumes.append(vol_n4)
            masks.append(mask)

        normalizer = NyulNormalizer().fit(volumes, masks)
        normalizer.save(os.path.join(save_dir, f"{mod_name}_nyul.npy"))
        normalizers[mod_name] = normalizer
        print(f"[Nyul fit] {mod_name}: standard_scale = {normalizer.standard_scale}")

    return normalizers


def load_nyul_scales(save_dir, modalities=("t1", "t2", "flair")):
    """Loads previously-fitted NyulNormalizers for the given modalities."""
    return {
        mod_name: NyulNormalizer.load(os.path.join(save_dir, f"{mod_name}_nyul.npy"))
        for mod_name in modalities
    }