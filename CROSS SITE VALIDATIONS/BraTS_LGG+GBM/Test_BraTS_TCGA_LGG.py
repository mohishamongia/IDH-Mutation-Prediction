"""
Cross-Site Evaluation: UTSW-trained Baseline_ResNet18 -> TCGA-LGG (TCIA)
=========================================================================
Train site : UTSW Glioma (~622 subjects, 5-fold CV)
Test  site : TCGA-LGG Pre-operative NIfTI collection (108 subjects, TCIA)
Labels     : Fetched from cBioPortal API (lgg_tcga study, IDH1_MUTATION)
             matched DIRECTLY by TCGA patient ID (folder name == patient ID)

Modality mapping (per UTSW training order: T1, T1CE, T2, FLAIR):
  UTSW brain_t1_ants.nii.gz   -> TCGA-LGG {id}_{date}_t1.nii.gz
  UTSW brain_t1ce_ants.nii.gz -> TCGA-LGG {id}_{date}_t1Gd.nii.gz
  UTSW brain_t2_ants.nii.gz   -> TCGA-LGG {id}_{date}_t2.nii.gz
  UTSW brain_fl_ants.nii.gz   -> TCGA-LGG {id}_{date}_flair.nii.gz

Run: python brats_cross_site_eval.py
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
# CONFIG  — edit paths to match your workspace
# ═══════════════════════════════════════════════════════════════════════════════

# TCGA-LGG Pre-operative NIfTI + Segmentations folder
# Each subfolder is named exactly like the TCGA patient ID, e.g. TCGA-CS-4942
TCGA_LGG_DIR ="BraTS_TCGA_LGG/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations"

# Your UTSW trained checkpoints — all 5 folds
# Script will ensemble predictions across all available folds
CHECKPOINT_DIR = "/workspace/results_final"   # folder with fold*_Baseline_ResNet18_best.pth
MODEL_NAME     = "Baseline_ResNet18"

# If you have a single global best checkpoint, set this path
# (will be used if fold checkpoints are not found)
GLOBAL_CKPT = "/workspace/idh_resnet18_global_best.pth"

# Output directory
SAVE_DIR = "/workspace/tcga_lgg_results"

# Preprocessing
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
print(f"TCGA-LGG : {TCGA_LGG_DIR}")
print(f"Output   : {SAVE_DIR}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL  — exact copy from ResNet18_Final.py
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
        self.layer1 = nn.Sequential(ResBlock3D(64,  64),             ResBlock3D(64,  64))
        self.layer2 = nn.Sequential(ResBlock3D(64,  128, stride=2),  ResBlock3D(128, 128))
        self.layer3 = nn.Sequential(ResBlock3D(128, 256, stride=2),  ResBlock3D(256, 256))
        self.layer4 = nn.Sequential(ResBlock3D(256, 512, stride=2),  ResBlock3D(512, 512))
        self.pool   = nn.AdaptiveAvgPool3d(1)
        self.drop   = nn.Dropout(0.5)
        self.fc     = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.fc(self.drop(self.pool(x).flatten(1)))


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — FETCH IDH LABELS FROM cBioPortal API (lgg_tcga study only)
# ═══════════════════════════════════════════════════════════════════════════════
def fetch_tcga_lgg_idh_labels():
    """
    Fetches IDH1 mutation status for TCGA-LGG patients from cBioPortal API.
    Returns: dict {TCGA_patient_id: idh_label}
              idh_label = 1 (mutated) | 0 (wild type)
    """
    print("Fetching IDH labels from cBioPortal API (lgg_tcga)...")
    base = "https://www.cbioportal.org/api"
    idh_labels = {}

    url = f"{base}/studies/lgg_tcga/clinical-data"
    params = {"clinicalAttributeId": "IDH1_MUTATION", "type": "PATIENT"}
    r = requests.get(url, params=params,
                     headers={"Accept": "application/json"}, timeout=60)

    if r.status_code == 200:
        data = r.json()
        for record in data:
            pid   = record.get("patientId", "")      # e.g. TCGA-CS-4942
            value = record.get("value", "").upper()   # YES / NO / MUTANT / WT
            if value in ["YES", "MUTANT", "MUTATED", "1", "TRUE"]:
                idh_labels[pid] = 1
            elif value in ["NO", "WT", "WILD TYPE", "WILD-TYPE", "0", "FALSE"]:
                idh_labels[pid] = 0
        print(f"  LGG: {len(idh_labels)} patients with IDH labels")
        lgg_mut = sum(v == 1 for v in idh_labels.values())
        print(f"  LGG: {lgg_mut} mutated | {len(idh_labels)-lgg_mut} wild type\n")
    else:
        print(f"  WARNING: LGG API call failed (status {r.status_code})\n")

    return idh_labels


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — MATCH TCGA-LGG SUBJECT FOLDERS TO IDH LABELS (direct ID match)
# ═══════════════════════════════════════════════════════════════════════════════
def build_tcga_lgg_label_map(tcga_lgg_dir, idh_labels):
    """
    Matches TCGA-LGG subject folders directly to IDH labels.
    Folder name (e.g. TCGA-CS-4942) IS the cBioPortal patient ID --
    no separate mapping file is needed.

    Returns:
        subjects : list of TCGA-LGG folder names with known IDH labels
        labels   : corresponding IDH labels (1=mutated, 0=WT)
    """
    print("Building TCGA-LGG label map (direct folder-name match)...")

    all_subjects = sorted([
        d for d in os.listdir(tcga_lgg_dir)
        if os.path.isdir(os.path.join(tcga_lgg_dir, d)) and d.startswith('TCGA-')
    ])
    print(f"  Subject folders found: {len(all_subjects)}")

    subjects_with_labels = []
    labels_list          = []
    skipped_no_label      = []

    for sid in all_subjects:
        if sid in idh_labels:
            subjects_with_labels.append(sid)
            labels_list.append(idh_labels[sid])
        else:
            skipped_no_label.append(sid)

    n_mut = sum(labels_list)
    n_wt  = len(labels_list) - n_mut
    print(f"  Subjects with IDH labels : {len(subjects_with_labels)}")
    print(f"  Mutated: {n_mut} | Wild type: {n_wt}")
    print(f"  Skipped (no cBioPortal match): {len(skipped_no_label)}")
    if skipped_no_label:
        print(f"    e.g. {skipped_no_label[:5]}")
    print()

    return subjects_with_labels, labels_list


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — TCGA-LGG DATASET
# ═══════════════════════════════════════════════════════════════════════════════
class TCGALGGDataset(Dataset):
    """
    Loads TCGA-LGG volumes with same preprocessing as UTSW training:
    - Zoom to IMG_SIZE^3
    - Min-max normalization per channel to [0, 1]

    Modality order matches UTSW training:
      ch0: T1    -> *_t1.nii.gz
      ch1: T1CE  -> *_t1Gd.nii.gz
      ch2: T2    -> *_t2.nii.gz
      ch3: FLAIR -> *_flair.nii.gz

    Each TCGA-LGG subject folder contains exactly one file per suffix,
    but the filename also embeds a scan date (e.g.
    TCGA-CS-4942_1997.02.22_t1.nii.gz), so files are located with glob
    rather than a fixed filename pattern.
    """
    MODALITY_SUFFIXES = ['_t1.nii.gz', '_t1Gd.nii.gz', '_t2.nii.gz', '_flair.nii.gz']

    def __init__(self, subjects, labels, tcga_lgg_dir, size=96, cache_dir=None):
        self.subjects     = subjects
        self.labels       = labels
        self.tcga_lgg_dir = tcga_lgg_dir
        self.size         = size
        self.cache_dir    = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _find_modality_file(self, folder, subject_id, suffix):
        """Find the file ending in `suffix` inside `folder` (date varies per subject)."""
        matches = glob.glob(os.path.join(folder, f"{subject_id}_*{suffix}"))
        if len(matches) == 0:
            raise FileNotFoundError(
                f"No file matching '*{suffix}' found for {subject_id} in {folder}")
        if len(matches) > 1:
            print(f"  WARNING: multiple files matched '*{suffix}' for {subject_id}, "
                  f"using {matches[0]}")
        return matches[0]

    def load_volume(self, subject_id):
        """Load and preprocess 4-channel volume for one TCGA-LGG subject."""
        folder   = os.path.join(self.tcga_lgg_dir, subject_id)
        channels = []

        for suffix in self.MODALITY_SUFFIXES:
            fpath = self._find_modality_file(folder, subject_id, suffix)
            vol   = nib.load(fpath).get_fdata(dtype=np.float32)

            # Resize to IMG_SIZE^3
            factors = [self.size / s for s in vol.shape]
            vol     = zoom(vol, factors, order=1)

            # Min-max normalization — same as UTSW training preprocessing
            v_min, v_max = vol.min(), vol.max()
            vol = (vol - v_min) / (v_max - v_min + 1e-8)

            channels.append(vol)

        return np.stack(channels, axis=0).astype(np.float32)  # (4, S, S, S)

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
    """
    Tries to load all 5 fold checkpoints.
    Falls back to global best checkpoint if fold checkpoints not found.
    Returns list of (fold_name, state_dict) tuples.
    """
    checkpoints = []

    # Try fold checkpoints first
    for fold in range(1, 6):
        ckpt_path = os.path.join(checkpoint_dir,
                                  f"fold{fold}_{model_name}_best.pth")
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location='cpu')
            checkpoints.append((f"fold{fold}", state))
            print(f"  Loaded: {ckpt_path}")

    # Fall back to global checkpoint
    if len(checkpoints) == 0 and os.path.exists(global_ckpt):
        state = torch.load(global_ckpt, map_location='cpu')
        checkpoints.append(("global", state))
        print(f"  Loaded global checkpoint: {global_ckpt}")

    if len(checkpoints) == 0:
        raise FileNotFoundError(
            f"No checkpoints found in {checkpoint_dir} or {global_ckpt}")

    print(f"  Using {len(checkpoints)} checkpoint(s) for ensemble\n")
    return checkpoints


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — INFERENCE + ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════
def run_inference(checkpoints, loader, device):
    """
    Runs inference with each checkpoint and averages probabilities (ensemble).
    Returns: all_probs (N,), all_true (N,)
    """
    all_fold_probs = []
    fold_true = []

    for fold_name, state_dict in checkpoints:
        model = ResNet18_3D(in_channels=4, num_classes=2)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        fold_probs = []
        fold_true  = []

        with torch.no_grad():
            for imgs, lbls in tqdm(loader, desc=f"  Inference [{fold_name}]",
                                   leave=False):
                imgs   = imgs.to(device)
                logits = model(imgs)
                probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                fold_probs.extend(probs.tolist())
                fold_true.extend(lbls.numpy().tolist())

        all_fold_probs.append(fold_probs)
        del model
        torch.cuda.empty_cache()

    # Ensemble: average probabilities across all folds
    ensemble_probs = np.mean(all_fold_probs, axis=0)
    return ensemble_probs, np.array(fold_true)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — COMPUTE METRICS
# ═══════════════════════════════════════════════════════════════════════════════
def compute_metrics(all_true, all_probs, subjects, threshold=0.5):
    """Computes full set of evaluation metrics."""
    all_preds = (all_probs >= threshold).astype(int)

    auc     = float(roc_auc_score(all_true, all_probs))
    bal_acc = float(balanced_accuracy_score(all_true, all_preds))
    f1      = float(f1_score(all_true, all_preds, pos_label=1, zero_division=0))
    cm      = confusion_matrix(all_true, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens    = float(tp / (tp + fn + 1e-8))
    spec    = float(tn / (tn + fp + 1e-8))
    ppv     = float(tp / (tp + fp + 1e-8))   # precision
    npv     = float(tn / (tn + fn + 1e-8))   # negative predictive value
    fpr, tpr, thresholds = roc_curve(all_true, all_probs)

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
        "roc_curve": {
            "fpr"       : fpr.tolist(),
            "tpr"       : tpr.tolist(),
            "thresholds": thresholds.tolist(),
        },
        "per_subject": per_subject,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    start = time.time()

    # ── Step 1: Fetch IDH labels from cBioPortal (lgg_tcga study) ─────────────
    idh_labels = fetch_tcga_lgg_idh_labels()

    # ── Step 2: Match TCGA-LGG subject folders to IDH labels (direct) ─────────
    subjects, labels = build_tcga_lgg_label_map(TCGA_LGG_DIR, idh_labels)

    if len(subjects) == 0:
        print("ERROR: No subjects with IDH labels found. Cannot evaluate.")
        print("Check that TCGA_LGG_DIR folder names match cBioPortal patient IDs,")
        print("and that the cBioPortal API request succeeded.")
        return

    print(f"Final evaluation set: {len(subjects)} subjects")
    print(f"  Mutated : {sum(labels)} ({100*sum(labels)/len(labels):.1f}%)")
    print(f"  Wild type: {len(labels)-sum(labels)} "
          f"({100*(len(labels)-sum(labels))/len(labels):.1f}%)\n")

    # ── Step 3: Build dataset and dataloader ──────────────────────────────────
    cache_dir = os.path.join(SAVE_DIR, "tcga_lgg_cache")
    dataset   = TCGALGGDataset(subjects, labels, TCGA_LGG_DIR,
                                size=IMG_SIZE, cache_dir=cache_dir)
    loader    = DataLoader(dataset, batch_size=BATCH, shuffle=False,
                           num_workers=2, pin_memory=True)
    print(f"Dataset ready: {len(dataset)} subjects | "
          f"Batches: {len(loader)} | Cache: {cache_dir}\n")

    # ── Step 4: Load checkpoints ───────────────────────────────────────────────
    print("Loading UTSW checkpoints...")
    checkpoints = load_checkpoints(CHECKPOINT_DIR, MODEL_NAME, GLOBAL_CKPT)

    # ── Step 5: Run inference ──────────────────────────────────────────────────
    print("Running inference on TCGA-LGG...")
    all_probs, all_true = run_inference(checkpoints, loader, DEVICE)
    print(f"  Inference complete: {len(all_probs)} subjects\n")

    # ── Step 6: Compute metrics ────────────────────────────────────────────────
    print("Computing metrics...")
    results = compute_metrics(all_true, all_probs, subjects)

    m = results["metrics"]
    print(f"\n{'='*55}")
    print(f"  CROSS-SITE EVALUATION: UTSW -> TCGA-LGG")
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

    # ── Save results ───────────────────────────────────────────────────────────
    output = {
        "experiment": "cross_site_UTSW_to_TCGA_LGG",
        "train_site" : "UTSW Glioma",
        "test_site"  : "TCGA-LGG (TCIA Pre-operative NIfTI collection)",
        "model"      : MODEL_NAME,
        "n_checkpoints_ensembled": len(checkpoints),
        "checkpoint_names": [name for name, _ in checkpoints],
        "n_subjects" : len(subjects),
        "n_mutated"  : int(sum(labels)),
        "n_wild_type": int(len(labels) - sum(labels)),
        "img_size"   : IMG_SIZE,
        **results,
    }

    out_path = os.path.join(SAVE_DIR, "cross_site_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_path}")

    # Save per-subject CSV for easy inspection
    csv_path = os.path.join(SAVE_DIR, "cross_site_per_subject.csv")
    pd.DataFrame(results["per_subject"]).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    elapsed = (time.time() - start) / 60
    print(f"\nDone in {elapsed:.1f} minutes")


if __name__ == "__main__":
    main()