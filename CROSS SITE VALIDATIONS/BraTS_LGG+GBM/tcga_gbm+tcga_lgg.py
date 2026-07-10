"""
Cross-Site Evaluation: UTSW-trained Baseline_ResNet18 → TCGA-GBM + TCGA-LGG
=============================================================================
Train site : UTSW Glioma (~622 subjects, 5-fold CV)
Test  site : TCGA-GBM (102 subjects, ~95% WT) + TCGA-LGG (65 subjects, ~94% mut)
             Combined: ~167 subjects, reasonably balanced
Labels     : TCGA-LGG → cBioPortal API (IDH1_MUTATION)
             TCGA-GBM → all wild-type (GBM is >95% IDH-WT clinically)

Modality mapping (same suffix for both GBM and LGG):
  UTSW brain_t1_ants.nii.gz   → TCGA {id}_{date}_t1.nii.gz
  UTSW brain_t1ce_ants.nii.gz → TCGA {id}_{date}_t1Gd.nii.gz
  UTSW brain_t2_ants.nii.gz   → TCGA {id}_{date}_t2.nii.gz
  UTSW brain_fl_ants.nii.gz   → TCGA {id}_{date}_flair.nii.gz

Run: python tcga_combined_eval.py
"""

import os, json, time, glob
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import requests
from tqdm import tqdm
from scipy.ndimage import zoom
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, f1_score,
    confusion_matrix, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

TCGA_GBM_DIR = "/workspace/BraTS_TCGA_GBM/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations"
TCGA_LGG_DIR = "/workspace/BraTS_TCGA_LGG/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations"

CHECKPOINT_DIR = "/workspace/results_final"
MODEL_NAME     = "Baseline_ResNet18"
GLOBAL_CKPT    = "/workspace/idh_resnet18_global_best.pth"

SAVE_DIR = "/workspace/tcga_combined_results"

IMG_SIZE = 96
BATCH    = 8
GPU_ID   = 0

# ═══════════════════════════════════════════════════════════════════════════════
# END CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Device   : {DEVICE}")
print(f"TCGA-GBM : {TCGA_GBM_DIR}")
print(f"TCGA-LGG : {TCGA_LGG_DIR}")
print(f"Output   : {SAVE_DIR}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL — exact copy from ResNet18_Final.py
# ═══════════════════════════════════════════════════════════════════════════════
class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)
        self.skip  = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch))

    def forward(self, x):
        return self.relu(
            self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.skip(x))


class ResNet18_3D(nn.Module):
    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv3d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1))
        self.layer1 = nn.Sequential(ResBlock3D(64,  64),            ResBlock3D(64,  64))
        self.layer2 = nn.Sequential(ResBlock3D(64,  128, stride=2), ResBlock3D(128, 128))
        self.layer3 = nn.Sequential(ResBlock3D(128, 256, stride=2), ResBlock3D(256, 256))
        self.layer4 = nn.Sequential(ResBlock3D(256, 512, stride=2), ResBlock3D(512, 512))
        self.pool   = nn.AdaptiveAvgPool3d(1)
        self.drop   = nn.Dropout(0.5)
        self.fc     = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.fc(self.drop(self.pool(x).flatten(1)))


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — FETCH IDH LABELS FROM cBioPortal (LGG only)
# ═══════════════════════════════════════════════════════════════════════════════
def fetch_lgg_idh_labels():
    """Fetch IDH1_MUTATION for TCGA-LGG from cBioPortal API."""
    print("Fetching TCGA-LGG IDH labels from cBioPortal...")
    idh_labels = {}

    r = requests.get(
        "https://www.cbioportal.org/api/studies/lgg_tcga/clinical-data",
        params={"clinicalAttributeId": "IDH1_MUTATION", "type": "PATIENT"},
        headers={"Accept": "application/json"}, timeout=60)

    if r.status_code == 200:
        for record in r.json():
            pid   = record.get("patientId", "")
            value = record.get("value", "").upper()
            if value in ["YES", "MUTANT", "MUTATED", "1", "TRUE"]:
                idh_labels[pid] = 1
            elif value in ["NO", "WT", "WILD TYPE", "WILD-TYPE", "0", "FALSE"]:
                idh_labels[pid] = 0
        mut = sum(v == 1 for v in idh_labels.values())
        print(f"  {len(idh_labels)} patients | mutated: {mut} | WT: {len(idh_labels)-mut}\n")
    else:
        print(f"  WARNING: API failed (status {r.status_code})\n")

    return idh_labels


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — BUILD COMBINED SUBJECT + LABEL LIST
# ═══════════════════════════════════════════════════════════════════════════════
def build_combined_subjects(gbm_dir, lgg_dir, lgg_idh_labels):
    """
    Combines TCGA-GBM (all WT) + TCGA-LGG (IDH from cBioPortal).
    Returns:
        subjects  : list of subject IDs
        labels    : list of IDH labels (1=mutated, 0=WT)
        data_dirs : dict {subject_id: data_dir} so Dataset knows which folder
        cohorts   : dict {subject_id: 'GBM'|'LGG'} for per-cohort reporting
    """
    subjects  = []
    labels    = []
    data_dirs = {}
    cohorts   = {}

    # ── TCGA-GBM — all wild-type ──────────────────────────────────────────────
    print("Loading TCGA-GBM subjects (all labeled wild-type)...")
    gbm_subjects = sorted([
        d for d in os.listdir(gbm_dir)
        if os.path.isdir(os.path.join(gbm_dir, d)) and d.startswith('TCGA-')])

    for sid in gbm_subjects:
        subjects.append(sid)
        labels.append(0)          # GBM → wild-type
        data_dirs[sid] = gbm_dir
        cohorts[sid]   = 'GBM'

    print(f"  GBM subjects: {len(gbm_subjects)}\n")

    # ── TCGA-LGG — IDH from cBioPortal ───────────────────────────────────────
    print("Loading TCGA-LGG subjects (IDH from cBioPortal)...")
    lgg_subjects = sorted([
        d for d in os.listdir(lgg_dir)
        if os.path.isdir(os.path.join(lgg_dir, d)) and d.startswith('TCGA-')])

    lgg_matched  = 0
    lgg_skipped  = 0
    for sid in lgg_subjects:
        if sid in lgg_idh_labels:
            subjects.append(sid)
            labels.append(lgg_idh_labels[sid])
            data_dirs[sid] = lgg_dir
            cohorts[sid]   = 'LGG'
            lgg_matched += 1
        else:
            lgg_skipped += 1

    print(f"  LGG subjects: {lgg_matched} matched | {lgg_skipped} skipped\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_mut = sum(labels)
    n_wt  = len(labels) - n_mut
    print(f"Combined dataset:")
    print(f"  Total    : {len(subjects)}")
    print(f"  Mutated  : {n_mut} ({100*n_mut/len(labels):.1f}%)")
    print(f"  Wild-type: {n_wt} ({100*n_wt/len(labels):.1f}%)\n")

    return subjects, labels, data_dirs, cohorts


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — DATASET
# ═══════════════════════════════════════════════════════════════════════════════
class TCGACombinedDataset(Dataset):
    """
    Loads TCGA-GBM and TCGA-LGG volumes with same preprocessing as UTSW:
    - Zoom to IMG_SIZE^3
    - Min-max normalization per channel to [0, 1]

    Modality order matches UTSW training:
      ch0: T1    → *_t1.nii.gz
      ch1: T1CE  → *_t1Gd.nii.gz
      ch2: T2    → *_t2.nii.gz
      ch3: FLAIR → *_flair.nii.gz

    Files are found via glob since filenames include scan dates.
    """
    MODALITY_SUFFIXES = ['_t1.nii.gz', '_t1Gd.nii.gz',
                         '_t2.nii.gz', '_flair.nii.gz']

    def __init__(self, subjects, labels, data_dirs, size=96, cache_dir=None):
        self.subjects  = subjects
        self.labels    = labels
        self.data_dirs = data_dirs   # dict {sid: dir}
        self.size      = size
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _find_file(self, folder, subject_id, suffix):
        """Glob for file ending in suffix — handles date in filename."""
        matches = glob.glob(os.path.join(folder, f"{subject_id}_*{suffix}"))
        if len(matches) == 0:
            raise FileNotFoundError(
                f"No file matching '{subject_id}_*{suffix}' in {folder}")
        return matches[0]

    def load_volume(self, subject_id):
        folder   = os.path.join(self.data_dirs[subject_id], subject_id)
        channels = []
        for suffix in self.MODALITY_SUFFIXES:
            fpath   = self._find_file(folder, subject_id, suffix)
            vol     = nib.load(fpath).get_fdata(dtype=np.float32)
            factors = [self.size / s for s in vol.shape]
            vol     = zoom(vol, factors, order=1)
            v_min, v_max = vol.min(), vol.max()
            vol     = (vol - v_min) / (v_max - v_min + 1e-8)
            channels.append(vol)
        return np.stack(channels, axis=0).astype(np.float32)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        sid = self.subjects[idx]
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"{sid}.npy")
            if os.path.exists(cache_path):
                volume = np.load(cache_path)
            else:
                volume = self.load_volume(sid)
                np.save(cache_path, volume)
        else:
            volume = self.load_volume(sid)
        return (torch.tensor(volume, dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.long))


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — LOAD CHECKPOINTS
# ═══════════════════════════════════════════════════════════════════════════════
def load_checkpoints(checkpoint_dir, model_name, global_ckpt):
    checkpoints = []
    for fold in range(1, 6):
        ckpt_path = os.path.join(checkpoint_dir,
                                  f"fold{fold}_{model_name}_best.pth")
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location='cpu')
            checkpoints.append((f"fold{fold}", state))
            print(f"  Loaded: {ckpt_path}")

    if len(checkpoints) == 0 and os.path.exists(global_ckpt):
        state = torch.load(global_ckpt, map_location='cpu')
        checkpoints.append(("global", state))
        print(f"  Loaded global: {global_ckpt}")

    if len(checkpoints) == 0:
        raise FileNotFoundError(f"No checkpoints found.")

    print(f"  Using {len(checkpoints)} checkpoint(s) for ensemble\n")
    return checkpoints


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — INFERENCE + ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════
def run_inference(checkpoints, loader, device):
    all_fold_probs = []
    all_true       = []

    for fold_name, state_dict in checkpoints:
        model = ResNet18_3D(in_channels=4, num_classes=2)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        fold_probs = []
        fold_true  = []

        with torch.no_grad():
            for imgs, lbls in tqdm(loader,
                                   desc=f"  Inference [{fold_name}]",
                                   leave=False):
                imgs   = imgs.to(device)
                logits = model(imgs)
                probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                fold_probs.extend(probs.tolist())
                fold_true.extend(lbls.numpy().tolist())

        all_fold_probs.append(fold_probs)
        all_true = fold_true   # same across folds
        del model
        torch.cuda.empty_cache()

    ensemble_probs = np.mean(all_fold_probs, axis=0)
    return ensemble_probs, np.array(all_true)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — COMPUTE METRICS
# ═══════════════════════════════════════════════════════════════════════════════
def compute_metrics(all_true, all_probs, subjects, threshold=0.5):
    all_preds       = (all_probs >= threshold).astype(int)
    auc             = float(roc_auc_score(all_true, all_probs))
    bal_acc         = float(balanced_accuracy_score(all_true, all_preds))
    f1              = float(f1_score(all_true, all_preds, pos_label=1, zero_division=0))
    cm              = confusion_matrix(all_true, all_preds, labels=[0, 1])
    tn, fp, fn, tp  = cm.ravel()
    sens            = float(tp / (tp + fn + 1e-8))
    spec            = float(tn / (tn + fp + 1e-8))
    ppv             = float(tp / (tp + fp + 1e-8))
    npv             = float(tn / (tn + fn + 1e-8))
    fpr, tpr, thr   = roc_curve(all_true, all_probs)

    per_subject = [
        {
            "subject_id": subjects[i],
            "true_label": int(all_true[i]),
            "pred_label": int(all_preds[i]),
            "pred_prob" : round(float(all_probs[i]), 4),
            "correct"   : bool(int(all_true[i]) == int(all_preds[i])),
        }
        for i in range(len(subjects))
    ]

    return {
        "metrics": {
            "AUC"        : round(auc,     4),
            "BalAcc"     : round(bal_acc, 4),
            "Sensitivity": round(sens,    4),
            "Specificity": round(spec,    4),
            "F1"         : round(f1,      4),
            "PPV"        : round(ppv,     4),
            "NPV"        : round(npv,     4),
            "TP": int(tp), "TN": int(tn),
            "FP": int(fp), "FN": int(fn),
        },
        "confusion_matrix": cm.tolist(),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                      "thresholds": thr.tolist()},
        "per_subject": per_subject,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    start = time.time()

    # Step 1: Fetch LGG IDH labels
    lgg_idh_labels = fetch_lgg_idh_labels()

    # Step 2: Build combined subject list
    subjects, labels, data_dirs, cohorts = build_combined_subjects(
        TCGA_GBM_DIR, TCGA_LGG_DIR, lgg_idh_labels)

    if len(subjects) == 0:
        print("ERROR: No subjects found.")
        return

    # Step 3: Dataset + loader
    cache_dir = os.path.join(SAVE_DIR, "tcga_combined_cache")
    dataset   = TCGACombinedDataset(subjects, labels, data_dirs,
                                     size=IMG_SIZE, cache_dir=cache_dir)
    loader    = DataLoader(dataset, batch_size=BATCH, shuffle=False,
                           num_workers=2, pin_memory=True)
    print(f"Dataset ready: {len(dataset)} subjects | "
          f"Batches: {len(loader)} | Cache: {cache_dir}\n")

    # Step 4: Load checkpoints
    print("Loading UTSW checkpoints...")
    checkpoints = load_checkpoints(CHECKPOINT_DIR, MODEL_NAME, GLOBAL_CKPT)

    # Step 5: Inference
    print("Running inference...")
    all_probs, all_true = run_inference(checkpoints, loader, DEVICE)
    print(f"  Inference complete: {len(all_probs)} subjects\n")

    # Step 6: Overall metrics
    print("Computing metrics...")
    results = compute_metrics(all_true, all_probs, subjects)
    m = results["metrics"]

    print(f"\n{'='*55}")
    print(f"  CROSS-SITE: UTSW → TCGA-GBM + TCGA-LGG (Combined)")
    print(f"{'='*55}")
    print(f"  Subjects   : {len(subjects)} "
          f"(mut={sum(labels)}, wt={len(labels)-sum(labels)})")
    print(f"  AUC        : {m['AUC']:.4f}")
    print(f"  BalAcc     : {m['BalAcc']:.4f}")
    print(f"  Sensitivity: {m['Sensitivity']:.4f}  (TP={m['TP']}, FN={m['FN']})")
    print(f"  Specificity: {m['Specificity']:.4f}  (TN={m['TN']}, FP={m['FP']})")
    print(f"  F1         : {m['F1']:.4f}")
    print(f"  PPV        : {m['PPV']:.4f}")
    print(f"  NPV        : {m['NPV']:.4f}")
    print(f"{'='*55}\n")

    # Per-cohort breakdown
    print("Per-cohort breakdown:")
    for cohort_name in ['GBM', 'LGG']:
        idxs = [i for i, s in enumerate(subjects) if cohorts[s] == cohort_name]
        if len(idxs) == 0:
            continue
        c_true  = all_true[idxs]
        c_probs = all_probs[idxs]
        c_subs  = [subjects[i] for i in idxs]
        try:
            c_res = compute_metrics(c_true, c_probs, c_subs)
            cm    = c_res["metrics"]
            print(f"  {cohort_name:5s} ({len(idxs)} subjects) | "
                  f"AUC: {cm['AUC']:.4f} | "
                  f"Sens: {cm['Sensitivity']:.4f} | "
                  f"Spec: {cm['Specificity']:.4f}")
        except Exception as e:
            print(f"  {cohort_name}: could not compute metrics ({e})")

    # Save results
    output = {
        "experiment"              : "cross_site_UTSW_to_TCGA_GBM_LGG_combined",
        "train_site"              : "UTSW Glioma",
        "test_site"               : "TCGA-GBM + TCGA-LGG (combined)",
        "model"                   : MODEL_NAME,
        "n_checkpoints_ensembled" : len(checkpoints),
        "checkpoint_names"        : [name for name, _ in checkpoints],
        "n_subjects"              : len(subjects),
        "n_mutated"               : int(sum(labels)),
        "n_wild_type"             : int(len(labels) - sum(labels)),
        "cohort_breakdown"        : {
            s: cohorts[s] for s in subjects
        },
        "img_size": IMG_SIZE,
        **results,
    }

    out_path = os.path.join(SAVE_DIR, "cross_site_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

    csv_path = os.path.join(SAVE_DIR, "cross_site_per_subject.csv")
    df = pd.DataFrame(results["per_subject"])
    df["cohort"] = df["subject_id"].map(cohorts)
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    elapsed = (time.time() - start) / 60
    print(f"\nDone in {elapsed:.1f} minutes")


if __name__ == "__main__":
    main()