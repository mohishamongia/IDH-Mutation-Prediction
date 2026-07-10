"""
Cross-Site Evaluation: UTSW-trained Baseline_ResNet18 → UCSF-PDGM
=============================================================================
Train site : UTSW Glioma (~622 subjects, 5-fold CV)
Test  site : UCSF-PDGM (v5)
Labels     : UCSF-PDGM-metadata_v5.csv → "IDH" column
             ("wildtype" -> 0, anything containing "mutat" -> 1)

Folder layout (per screenshots):
  UCSF-PDGM-v5/
    UCSF-PDGM-0004_nifti/
      UCSF-PDGM-0004_T1.nii.gz
      UCSF-PDGM-0004_T1c.nii.gz
      UCSF-PDGM-0004_T2.nii.gz
      UCSF-PDGM-0004_FLAIR.nii.gz
      UCSF-PDGM-0004_ADC.nii.gz   (not used)
      ...
    UCSF-PDGM-metadata_v5.csv


Run: python ucsf_pdgm_eval.py
"""

import os, json, time, glob, re
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
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

UCSF_PDGM_DIR     = "/workspace/UCSF-PDGM-v5"                                   # contains the per-subject *_nifti folders
UCSF_METADATA_CSV = "/workspace/UCSF-PDGM-v5/UCSF-PDGM-metadata_v5.csv"         # has an "IDH" column

CHECKPOINT_DIR = "/workspace/results_final"
MODEL_NAME     = "Baseline_ResNet18"
GLOBAL_CKPT    = "/workspace/idh_resnet18_global_best.pth"

SAVE_DIR = "/workspace/ucsf_pdgm_results"

IMG_SIZE = 96
BATCH    = 8
GPU_ID   = 0

# Column in the metadata CSV that holds the subject ID (e.g. "UCSF-PDGM-0004")
# and the column that holds IDH status. Adjust these two if your CSV header differs.
ID_COLUMN  = "ID"
IDH_COLUMN = "IDH"

# ═══════════════════════════════════════════════════════════════════════════════
# END CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Device     : {DEVICE}")
print(f"UCSF-PDGM  : {UCSF_PDGM_DIR}")
print(f"Metadata   : {UCSF_METADATA_CSV}")
print(f"Output     : {SAVE_DIR}\n")


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
# ID NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
def normalize_ucsf_id(raw_id):
   
    match = re.search(r'(\d+)', str(raw_id))
    if not match:
        return None
    num = int(match.group(1))
    return f"UCSF-PDGM-{num:04d}"



def fetch_ucsf_idh_labels(csv_path, id_col=ID_COLUMN, idh_col=IDH_COLUMN):
    """
    Reads UCSF-PDGM-metadata_v5.csv and maps subject_id -> 0/1 IDH label.
    Based on the screenshot, the IDH column contains strings like
    'wildtype' or 'mutated (NOS)' (possibly other 'mutated, ...' variants).
    """
    print("Loading UCSF-PDGM IDH labels from metadata CSV...")
    df = pd.read_csv(csv_path)

    if id_col not in df.columns:
        raise KeyError(
            f"Column '{id_col}' not found in CSV. Available columns: {list(df.columns)}\n"
            f"Set ID_COLUMN at the top of the script to the correct subject-ID column name.")
    if idh_col not in df.columns:
        raise KeyError(
            f"Column '{idh_col}' not found in CSV. Available columns: {list(df.columns)}")

    idh_labels = {}
    for _, row in df.iterrows():
        raw_sid = str(row[id_col]).strip()
        sid     = normalize_ucsf_id(raw_sid)
        val     = str(row[idh_col]).strip().lower()

        if sid is None:
            print(f"  WARNING: could not parse subject number from '{raw_sid}' — skipping")
            continue
        if val in ("", "nan", "unknown", "indeterminate"):
            continue  # skip subjects with no usable IDH call
        elif "mutat" in val:          # e.g. "mutated (NOS)", "mutated, IDH1"
            idh_labels[sid] = 1
        elif "wildtype" in val or val == "wt":
            idh_labels[sid] = 0
        else:
            print(f"  WARNING: unrecognized IDH value '{val}' for subject {sid} — skipping")

    mut = sum(v == 1 for v in idh_labels.values())
    print(f"  {len(idh_labels)} labeled subjects | mutated: {mut} | WT: {len(idh_labels)-mut}\n")
    return idh_labels


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — BUILD SUBJECT + LABEL LIST FROM ON-DISK FOLDERS
# ═══════════════════════════════════════════════════════════════════════════════
def build_ucsf_subjects(data_dir, idh_labels):
    """
    Matches subject folders (e.g. 'UCSF-PDGM-0004_nifti') to IDH labels keyed
    by the normalized subject ID (e.g. 'UCSF-PDGM-0004'), regardless of
    zero-padding differences between the CSV and the on-disk folder names.

    Returns subjects/labels keyed by the ACTUAL on-disk folder's subject id
    (so the Dataset can build correct file paths), plus a normalized lookup
    used only for matching against idh_labels.
    """
    subjects  = []
    labels    = []
    data_dirs = {}   # sid (on-disk form) -> parent dir

    print("Scanning UCSF-PDGM subject folders...")
    folder_names = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('UCSF-PDGM-')])

    matched, skipped = 0, 0
    for folder in folder_names:
        sid_ondisk = folder.replace("_nifti", "")   # e.g. 'UCSF-PDGM-0004' (matches actual files)
        sid_norm   = normalize_ucsf_id(sid_ondisk)  # e.g. 'UCSF-PDGM-0004' (canonical match key)
        if sid_norm is not None and sid_norm in idh_labels:
            subjects.append(sid_ondisk)
            labels.append(idh_labels[sid_norm])
            data_dirs[sid_ondisk] = data_dir
            matched += 1
        else:
            skipped += 1

    print(f"  Matched: {matched} | Skipped (no label / no match): {skipped}\n")

    n_mut = sum(labels)
    n_wt  = len(labels) - n_mut
    print(f"UCSF-PDGM eval set:")
    print(f"  Total    : {len(subjects)}")
    print(f"  Mutated  : {n_mut} ({100*n_mut/max(len(labels),1):.1f}%)")
    print(f"  Wild-type: {n_wt} ({100*n_wt/max(len(labels),1):.1f}%)\n")

    return subjects, labels, data_dirs


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — DATASET
# ═══════════════════════════════════════════════════════════════════════════════
class UCSFPDGMDataset(Dataset):
    """
    Loads UCSF-PDGM volumes with same preprocessing as UTSW:
    - Zoom to IMG_SIZE^3
    - Min-max normalization per channel to [0, 1]

    Modality order matches UTSW training:
      ch0: T1    → *_T1.nii.gz
      ch1: T1CE  → *_T1c.nii.gz
      ch2: T2    → *_T2.nii.gz
      ch3: FLAIR → *_FLAIR.nii.gz

    Subject's files live in: {data_dir}/{subject_id}_nifti/
    Files are found via glob, so exact case/suffix variants are tolerated
    as long as MODALITY_SUFFIXES below match what's on disk.
    """
    # EDIT THESE if your filenames differ (e.g. '_T1_bias.nii.gz')
    MODALITY_SUFFIXES = ['_T1.nii.gz', '_T1c.nii.gz',
                         '_T2.nii.gz', '_FLAIR.nii.gz']

    def __init__(self, subjects, labels, data_dirs, size=96, cache_dir=None):
        self.subjects  = subjects
        self.labels    = labels
        self.data_dirs = data_dirs   # dict {sid: parent dir}
        self.size      = size
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _find_file(self, folder, subject_id, suffix):
        """Glob for file ending in suffix."""
        matches = glob.glob(os.path.join(folder, f"{subject_id}*{suffix}"))
        if len(matches) == 0:
            raise FileNotFoundError(
                f"No file matching '{subject_id}*{suffix}' in {folder}")
        return matches[0]

    def load_volume(self, subject_id):
        folder   = os.path.join(self.data_dirs[subject_id], f"{subject_id}_nifti")
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

    # Step 1: Load IDH labels from metadata CSV
    idh_labels = fetch_ucsf_idh_labels(UCSF_METADATA_CSV)

    # Step 2: Build subject list matched against on-disk folders
    subjects, labels, data_dirs = build_ucsf_subjects(UCSF_PDGM_DIR, idh_labels)

    if len(subjects) == 0:
        print("ERROR: No subjects found/matched. Check ID_COLUMN / IDH_COLUMN "
              "and that folder names start with 'UCSF-PDGM-'.")
        return

    # Step 3: Dataset + loader
    cache_dir = os.path.join(SAVE_DIR, "ucsf_pdgm_cache")
    dataset   = UCSFPDGMDataset(subjects, labels, data_dirs,
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

    # Step 6: Metrics
    print("Computing metrics...")
    results = compute_metrics(all_true, all_probs, subjects)
    m = results["metrics"]

    print(f"\n{'='*55}")
    print(f"  CROSS-SITE: UTSW → UCSF-PDGM")
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

    # Save results
    output = {
        "experiment"              : "cross_site_UTSW_to_UCSF_PDGM",
        "train_site"              : "UTSW Glioma",
        "test_site"               : "UCSF-PDGM",
        "model"                   : MODEL_NAME,
        "n_checkpoints_ensembled" : len(checkpoints),
        "checkpoint_names"        : [name for name, _ in checkpoints],
        "n_subjects"              : len(subjects),
        "n_mutated"               : int(sum(labels)),
        "n_wild_type"             : int(len(labels) - sum(labels)),
        "img_size": IMG_SIZE,
        **results,
    }

    out_path = os.path.join(SAVE_DIR, "ucsf_pdgm_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

    csv_path = os.path.join(SAVE_DIR, "ucsf_pdgm_per_subject.csv")
    df = pd.DataFrame(results["per_subject"])
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    elapsed = (time.time() - start) / 60
    print(f"\nDone in {elapsed:.1f} minutes")


if __name__ == "__main__":
    main()