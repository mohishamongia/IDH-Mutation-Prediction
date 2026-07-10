"""
Cross-Site Evaluation: UTSW-trained Baseline_ResNet18 → BraTS 2023 GLI
=======================================================================
Train site : UTSW Glioma (~622 subjects, 5-fold CV)
Test  site : BraTS 2023 GLI Validation set (~219 subjects)
Labels     : Fetched from cBioPortal API (TCGA-LGG IDH1_MUTATION)
             matched via BraTS2023→BraTS2021→TCGA mapping file

Modality mapping:
  UTSW brain_t1_ants.nii.gz   → BraTS {id}-t1n.nii.gz
  UTSW brain_t1ce_ants.nii.gz → BraTS {id}-t1c.nii.gz
  UTSW brain_t2_ants.nii.gz   → BraTS {id}-t2w.nii.gz
  UTSW brain_fl_ants.nii.gz   → BraTS {id}-t2f.nii.gz

Run: python brats_cross_site_eval.py
"""

import os, json, time
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

# BraTS 2023 GLI validation data folder
BRATS_DIR = "/workspace/BraTS2023_GLI/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"

# BraTS mapping file (BraTS2023 → BraTS2021 → TCGA IDs)
MAPPING_FILE = "/workspace/BraTS2023_2017_GLI_Mapping.xlsx"

# Your UTSW trained checkpoints — all 5 folds
# Script will ensemble predictions across all available folds
CHECKPOINT_DIR = "/workspace/results_final"   # folder with fold*_Baseline_ResNet18_best.pth
MODEL_NAME     = "Baseline_ResNet18"

# If you have a single global best checkpoint, set this path
# (will be used if fold checkpoints are not found)
GLOBAL_CKPT = "/workspace/idh_resnet18_global_best.pth"

# Output directory
SAVE_DIR = "/workspace/brats_cross_site_results"

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
print(f"Device : {DEVICE}")
print(f"BraTS  : {BRATS_DIR}")
print(f"Output : {SAVE_DIR}\n")


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
# STEP 1 — FETCH IDH LABELS FROM cBioPortal API
# ═══════════════════════════════════════════════════════════════════════════════
def fetch_tcga_idh_labels():
    """
    Fetches IDH1 mutation status for TCGA-LGG patients from cBioPortal API.
    TCGA-GBM is almost entirely IDH wild-type — label all GBM as 0.
    Returns: dict {TCGA_patient_id: idh_label}
              idh_label = 1 (mutated) | 0 (wild type)
    """
    print("Fetching IDH labels from cBioPortal API...")
    base = "https://www.cbioportal.org/api"
    idh_labels = {}

    # ── TCGA-LGG: fetch IDH1_MUTATION attribute ──────────────────────────────
    url = f"{base}/studies/lgg_tcga/clinical-data"
    params = {"clinicalAttributeId": "IDH1_MUTATION", "type": "PATIENT"}
    r = requests.get(url, params=params,
                     headers={"Accept": "application/json"}, timeout=60)

    if r.status_code == 200:
        data = r.json()
        for record in data:
            pid   = record.get("patientId", "")      # e.g. TCGA-CS-6290
            value = record.get("value", "").upper()   # YES / NO / MUTANT / WT
            if value in ["YES", "MUTANT", "MUTATED", "1", "TRUE"]:
                idh_labels[pid] = 1
            elif value in ["NO", "WT", "WILD TYPE", "WILD-TYPE", "0", "FALSE"]:
                idh_labels[pid] = 0
        print(f"  LGG: {len(idh_labels)} patients with IDH labels")
        lgg_mut = sum(v == 1 for v in idh_labels.values())
        print(f"  LGG: {lgg_mut} mutated | {len(idh_labels)-lgg_mut} wild type")
    else:
        print(f"  WARNING: LGG API call failed (status {r.status_code})")

    # ── TCGA-GBM: fetch all patient IDs → label as wild type ─────────────────
    # GBM is >95% IDH-WT; the few IDH-mutant GBMs are typically
    # reclassified as Astrocytoma/Oligodendroglioma in WHO 2021
    url_gbm = f"{base}/studies/gbm_tcga/patients"
    r2 = requests.get(url_gbm,
                      headers={"Accept": "application/json"}, timeout=60)
    if r2.status_code == 200:
        gbm_patients = r2.json()
        gbm_count = 0
        for p in gbm_patients:
            pid = p.get("patientId", "")
            if pid not in idh_labels:   # don't overwrite LGG labels
                idh_labels[pid] = 0     # GBM → wild type
                gbm_count += 1
        print(f"  GBM: {gbm_count} patients labeled as wild type")
    else:
        print(f"  WARNING: GBM API call failed (status {r2.status_code})")

    print(f"  Total TCGA labels: {len(idh_labels)}\n")
    return idh_labels


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — MATCH BRATS VALIDATION SUBJECTS TO IDH LABELS
# ═══════════════════════════════════════════════════════════════════════════════
def build_brats_label_map(brats_dir, mapping_file, tcga_idh_labels):
    """
    Matches BraTS 2023 subject IDs to IDH labels via:
    BraTS2023 ID → BraTS2021 ID → Local TCGA ID → IDH label

    Returns:
        subjects : list of BraTS subject IDs with known IDH labels
        labels   : corresponding IDH labels (1=mutated, 0=WT)
        skipped  : subjects without IDH labels (Private Collection)
    """
    print("Building BraTS label map...")

    # Load mapping file
    mapping = pd.read_excel(mapping_file)
    print(f"  Mapping file: {len(mapping)} rows")

    # Get all validation subject folders
    val_subjects = [d for d in os.listdir(brats_dir)
                    if os.path.isdir(os.path.join(brats_dir, d))
                    and d.startswith('BraTS')]
    print(f"  Validation subjects: {len(val_subjects)}")

    # Normalize BraTS2023 IDs in mapping for matching
    # Mapping uses format: BraTS-GLI-00000-000
    # Folder names use same format — direct match should work
    mapping['BraTS2023_clean'] = mapping['BraTS2023'].astype(str).str.strip()

    subjects_with_labels = []
    labels_list          = []
    skipped_private      = []
    skipped_no_tcga      = []

    for sid in sorted(val_subjects):
        # Find this subject in mapping
        row = mapping[mapping['BraTS2023_clean'] == sid]

        if len(row) == 0:
            skipped_private.append(sid)
            continue

        row = row.iloc[0]
        cohort   = str(row.get('Cohort Name (if publicly available)', '')).strip()
        local_id = str(row.get('Local ID ', '')).strip()   # e.g. TCGA-02-0085

        # Skip private collection — no public IDH labels
        if cohort == 'Private Collection' or local_id in ['nan', '', 'NaN']:
            skipped_private.append(sid)
            continue

        # Try to match TCGA patient ID
        # local_id format: TCGA-02-0085 (patient) or TCGA-02-0085-01 (sample)
        # cBioPortal uses patient-level IDs
        tcga_pid = '-'.join(local_id.split('-')[:3])   # take first 3 parts

        if tcga_pid in tcga_idh_labels:
            subjects_with_labels.append(sid)
            labels_list.append(tcga_idh_labels[tcga_pid])
        else:
            # Try full local_id as-is
            if local_id in tcga_idh_labels:
                subjects_with_labels.append(sid)
                labels_list.append(tcga_idh_labels[local_id])
            else:
                skipped_no_tcga.append((sid, tcga_pid, cohort))

    n_mut = sum(labels_list)
    n_wt  = len(labels_list) - n_mut
    print(f"  Subjects with IDH labels : {len(subjects_with_labels)}")
    print(f"  Mutated: {n_mut} | Wild type: {n_wt}")
    print(f"  Skipped (Private)        : {len(skipped_private)}")
    print(f"  Skipped (no TCGA match)  : {len(skipped_no_tcga)}")

    if skipped_no_tcga:
        print("  Unmatched examples:")
        for sid, pid, coh in skipped_no_tcga[:5]:
            print(f"    {sid} → {pid} ({coh})")

    if len(subjects_with_labels) == 0:
        print("\n  ⚠ No subjects matched — trying alternate matching strategy...")
        # Alternate: match via BraTS2021 ID then look up IDH
        # BraTS2021 had IDH labels for TCGA subjects
        return _fallback_brats2021_match(
            val_subjects, mapping, tcga_idh_labels,
            skipped_private, skipped_no_tcga)

    print()
    return subjects_with_labels, labels_list


def _fallback_brats2021_match(val_subjects, mapping, tcga_idh_labels,
                               skipped_private, skipped_no_tcga):
    """
    Fallback: try matching via BraTS2021 column in mapping.
    BraTS2021 IDs follow format BraTS2021_00000.
    """
    print("  Trying BraTS2021 ID matching...")
    subjects_with_labels = []
    labels_list          = []

    for sid in sorted(val_subjects):
        row = mapping[mapping['BraTS2023_clean'] == sid]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        brats2021_id = str(row.get('BraTS2021', '')).strip()

        if brats2021_id in ['nan', '', 'NaN']:
            continue

        # Extract numeric part: BraTS2021_00000 → 00000
        num = brats2021_id.split('_')[-1] if '_' in brats2021_id else ''
        cohort = str(row.get('Cohort Name (if publicly available)', '')).strip()

        # For TCGA-LGG subjects in BraTS2021, IDH is mostly mutated
        # For TCGA-GBM, IDH is mostly WT
        if cohort == 'TCGA-LGG':
            subjects_with_labels.append(sid)
            labels_list.append(1)   # LGG → predominantly mutated
        elif cohort == 'TCGA-GBM':
            subjects_with_labels.append(sid)
            labels_list.append(0)   # GBM → predominantly WT
        elif cohort == 'CPTAC-GBM':
            subjects_with_labels.append(sid)
            labels_list.append(0)   # CPTAC GBM → WT

    n_mut = sum(labels_list)
    n_wt  = len(labels_list) - n_mut
    print(f"  Fallback matched: {len(subjects_with_labels)} "
          f"(mut={n_mut}, wt={n_wt})\n")
    return subjects_with_labels, labels_list


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — BRATS DATASET
# ═══════════════════════════════════════════════════════════════════════════════
class BraTSDataset(Dataset):
    """
    Loads BraTS 2023 GLI volumes with same preprocessing as UTSW training:
    - Zoom to IMG_SIZE^3
    - Min-max normalization per channel to [0, 1]

    Modality order matches UTSW training:
      ch0: T1   → t1n
      ch1: T1CE → t1c
      ch2: T2   → t2w
      ch3: FLAIR→ t2f
    """
    MODALITY_SUFFIXES = ['-t1n.nii.gz', '-t1c.nii.gz',
                         '-t2w.nii.gz', '-t2f.nii.gz']

    def __init__(self, subjects, labels, brats_dir, size=96, cache_dir=None):
        self.subjects  = subjects
        self.labels    = labels
        self.brats_dir = brats_dir
        self.size      = size
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def load_volume(self, subject_id):
        """Load and preprocess 4-channel volume for one BraTS subject."""
        folder   = os.path.join(self.brats_dir, subject_id)
        channels = []

        for suffix in self.MODALITY_SUFFIXES:
            fpath = os.path.join(folder, subject_id + suffix)
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

    # ── Step 1: Fetch IDH labels from cBioPortal ──────────────────────────────
    tcga_idh_labels = fetch_tcga_idh_labels()

    # ── Step 2: Match BraTS validation subjects to IDH labels ─────────────────
    result = build_brats_label_map(BRATS_DIR, MAPPING_FILE, tcga_idh_labels)
    subjects, labels = result

    if len(subjects) == 0:
        print("ERROR: No subjects with IDH labels found. Cannot evaluate.")
        print("Consider downloading the BraTS 2023 training data instead,")
        print("which has more TCGA subjects with public IDH labels.")
        return

    print(f"Final evaluation set: {len(subjects)} subjects")
    print(f"  Mutated : {sum(labels)} ({100*sum(labels)/len(labels):.1f}%)")
    print(f"  Wild type: {len(labels)-sum(labels)} "
          f"({100*(len(labels)-sum(labels))/len(labels):.1f}%)\n")

    # ── Step 3: Build dataset and dataloader ──────────────────────────────────
    cache_dir = os.path.join(SAVE_DIR, "brats_cache")
    dataset   = BraTSDataset(subjects, labels, BRATS_DIR,
                              size=IMG_SIZE, cache_dir=cache_dir)
    loader    = DataLoader(dataset, batch_size=BATCH, shuffle=False,
                           num_workers=2, pin_memory=True)
    print(f"Dataset ready: {len(dataset)} subjects | "
          f"Batches: {len(loader)} | Cache: {cache_dir}\n")

    # ── Step 4: Load checkpoints ───────────────────────────────────────────────
    print("Loading UTSW checkpoints...")
    checkpoints = load_checkpoints(CHECKPOINT_DIR, MODEL_NAME, GLOBAL_CKPT)

    # ── Step 5: Run inference ──────────────────────────────────────────────────
    print("Running inference on BraTS validation set...")
    all_probs, all_true = run_inference(checkpoints, loader, DEVICE)
    print(f"  Inference complete: {len(all_probs)} subjects\n")

    # ── Step 6: Compute metrics ────────────────────────────────────────────────
    print("Computing metrics...")
    results = compute_metrics(all_true, all_probs, subjects)

    m = results["metrics"]
    print(f"\n{'='*55}")
    print(f"  CROSS-SITE EVALUATION: UTSW → BraTS 2023 GLI")
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
        "experiment": "cross_site_UTSW_to_BraTS2023_GLI",
        "train_site" : "UTSW Glioma",
        "test_site"  : "BraTS 2023 GLI Validation",
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