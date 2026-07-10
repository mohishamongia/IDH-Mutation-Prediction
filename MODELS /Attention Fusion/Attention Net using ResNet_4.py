"""
Generalized N-Fold Training — ResNet-based Attention Fusion
Saves    : results_resnet_attn/ (configurable)
Run      : python idh_train_resnet_attn.py
"""

import os, json, time
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# ─── EASILY CONFIGURABLE SECTION ────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

GPU_IDS = [0]   # ✏️ edit per run

# --- Modality configuration -----------------------------------------------------
ALL_MODALITIES = {
    "T1"   : "brain_t1_ants.nii.gz",
    "T1CE" : "brain_t1ce_ants.nii.gz",
    "T2"   : "brain_t2_ants.nii.gz",
    "FLAIR": "brain_fl_ants.nii.gz",
}

# ✏️ Choose which modalities to use as input channels (order = channel order)
ACTIVE_MODALITIES = ["T1", "T1CE", "T2", "FLAIR"]

# --- Fold configuration -----------------------------------------------------------
N_FOLDS  = 5          # ✏️ total number of CV folds to define the split
RUN_FOLDS = [1,2,3,4,5]       # ✏️ which fold(s) to actually train/eval (1-indexed)

# --- Training configuration (SAME AS ORIGINAL) ------------------------------------
SAVE_DIR   = "results_resnet_attn_4"
IMG_SIZE   = 96          # ← original value
BATCH      = 8 * len(GPU_IDS)   # ← original value
EPOCHS     = 30           # ← original value
LR         = 5e-4         # ← original value
PATIENCE   = 10             # ← original value
DROPOUT_P  = 0.3           # ← original modality dropout (constant, no curriculum)

# --- Evaluation scenarios ----------------------------------------------------------
TEST_SCENARIOS = {
    "All": list(range(len(ACTIVE_MODALITIES))),
    # Add more if needed, indices refer to positions in ACTIVE_MODALITIES
}

# --- Class weights -------------------------------------------------------------------
CLASS_WEIGHTS = [1.0, 3.0]   # ← original value

# --- Caching (for speed) --------------------------------------------------------------
USE_CACHE = True
CACHE_DIR = f"/workspace/cache_{IMG_SIZE}_{'_'.join(ACTIVE_MODALITIES)}"

# ═══════════════════════════════════════════════════════════════════════════════
# END CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in GPU_IDS)
DEVICE    = torch.device("cuda:0")
MULTI_GPU = len(GPU_IDS) > 1
NUM_CHANNELS = len(ACTIVE_MODALITIES)

DATA_DIR = "/workspace/UTSW_Glioma_data/UTSW-Glioma"
TSV_PATH = "UTSW_Glioma_data/UTSW_Glioma_Metadata-2-1 (2).tsv"

os.makedirs(SAVE_DIR, exist_ok=True)
if USE_CACHE:
    os.makedirs(CACHE_DIR, exist_ok=True)

print(f"GPUs: {GPU_IDS}  |  Multi-GPU: {MULTI_GPU}")
print(f"Active modalities ({NUM_CHANNELS}): {ACTIVE_MODALITIES}")
print(f"N_FOLDS: {N_FOLDS}  |  Running folds: {RUN_FOLDS}")
print(f"Results will be saved to: {SAVE_DIR}/")
print(f"Batch: {BATCH}  |  Image size: {IMG_SIZE}³  |  Cache: {USE_CACHE} ({CACHE_DIR})\n")


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════
class GliomaDataset(Dataset):
    def __init__(self, subjects, labels, data_dir, modality_files, size=96,
                 dropout_p=0.0, missing=None, cache_dir=None):
        self.subjects       = subjects
        self.labels         = labels
        self.data_dir       = data_dir
        self.modality_files = modality_files
        self.size           = size
        self.dropout_p      = dropout_p
        self.missing        = missing
        self.cache_dir      = cache_dir

    def load_volume(self, path):
        from scipy.ndimage import zoom
        vol     = nib.load(path).get_fdata(dtype=np.float32)
        factors = [self.size / s for s in vol.shape]
        vol     = zoom(vol, factors, order=1)
        vol     = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
        return vol

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        sid = self.subjects[idx]

        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"{sid}.npy")
            if os.path.exists(cache_path):
                volume = np.load(cache_path)
            else:
                folder   = os.path.join(self.data_dir, sid)
                channels = [self.load_volume(os.path.join(folder, f)) for f in self.modality_files]
                volume   = np.stack(channels, axis=0).astype(np.float32)
                np.save(cache_path, volume)
        else:
            folder   = os.path.join(self.data_dir, sid)
            channels = [self.load_volume(os.path.join(folder, f)) for f in self.modality_files]
            volume   = np.stack(channels, axis=0)

        volume = volume.copy()
        n_ch   = volume.shape[0]

        # Training: random modality dropout
        if self.dropout_p > 0:
            for c in range(n_ch):
                if np.random.rand() < self.dropout_p:
                    volume[c] = 0.0

        # Test: zero out channels not in `missing`
        if self.missing is not None:
            for c in range(n_ch):
                if c not in self.missing:
                    volume[c] = 0.0

        return torch.tensor(volume), torch.tensor(self.labels[idx], dtype=torch.long)


# ═══════════════════════════════════════════════════════════════════════════════
# MODELS
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
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.skip(x))


class ResNetModalityEncoder(nn.Module):
    """Single-channel ResNet18-3D-style encoder for one modality."""
    def __init__(self, feat_dim=256):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv3d(1, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1))
        self.layer1 = nn.Sequential(ResBlock3D(32,  32),               ResBlock3D(32,  32))
        self.layer2 = nn.Sequential(ResBlock3D(32,  64,  stride=2),     ResBlock3D(64,  64))
        self.layer3 = nn.Sequential(ResBlock3D(64,  128, stride=2),     ResBlock3D(128, 128))
        self.layer4 = nn.Sequential(ResBlock3D(128, feat_dim, stride=2), ResBlock3D(feat_dim, feat_dim))
        self.pool   = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten())

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.pool(x)  # (B, feat_dim)


class ResNetAttentionFusionNet(nn.Module):
    """
    Attention fusion with ResNet18-3D modality encoders.
    Works with any number of modalities >= 1.
    """
    def __init__(self, num_modalities, feat_dim=256, num_classes=2, dropout=0.5):
        super().__init__()
        self.num_modalities = num_modalities
        self.feat_dim = feat_dim

        self.encoders = nn.ModuleList([ResNetModalityEncoder(feat_dim) for _ in range(num_modalities)])

        self.attention = nn.Sequential(
            nn.Linear(feat_dim * num_modalities, 64), nn.ReLU(),
            nn.Linear(64, num_modalities), nn.Softmax(dim=1))

        # ← original classifier shape: Linear(feat_dim,64) -> ReLU -> Dropout(0.5) -> Linear(64,num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, num_classes))

    def forward(self, x, return_weights=False):
        feats   = [self.encoders[i](x[:, i:i+1]) for i in range(self.num_modalities)]
        concat  = torch.cat(feats, dim=1)
        weights = self.attention(concat)
        feats_s = torch.stack(feats, dim=1)
        fused   = (feats_s * weights.unsqueeze(2)).sum(1)
        out     = self.classifier(fused)
        if return_weights:
            return out, weights
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN ONE FOLD — same hyperparameters as original
# ═══════════════════════════════════════════════════════════════════════════════
def train_one_fold(model_fn, X_train, y_train, X_val, y_val, modality_files):
    model = model_fn()
    if MULTI_GPU:
        model = nn.DataParallel(model)
    model = model.to(DEVICE)

    optimizer     = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    class_weights = torch.tensor(CLASS_WEIGHTS).to(DEVICE)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', patience=3, factor=0.5)

    train_ds = GliomaDataset(X_train, y_train, DATA_DIR, modality_files, IMG_SIZE,
                              dropout_p=DROPOUT_P,
                              cache_dir=CACHE_DIR if USE_CACHE else None)
    val_ds   = GliomaDataset(X_val, y_val, DATA_DIR, modality_files, IMG_SIZE,
                              cache_dir=CACHE_DIR if USE_CACHE else None)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                               num_workers=2, pin_memory=True,
                               persistent_workers=True, prefetch_factor=4)
    val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                               num_workers=8, pin_memory=True,
                               persistent_workers=True, prefetch_factor=4)

    best_auc, patience_counter = 0.0, 0
    best_state = None
    history = {"train_loss": [], "val_loss": [],
               "auc": [], "bal_acc": [], "f1": [],
               "sensitivity": [], "specificity": []}

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        batch_losses = []
        for imgs, lbls in tqdm(train_loader, desc=f"  Ep {epoch+1}/{EPOCHS}", leave=False):
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(loss.item())
        history["train_loss"].append(float(np.mean(batch_losses)))

        # --- Validate ---
        model.eval()
        all_preds, all_probs, all_true, v_losses = [], [], [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                logits = model(imgs)
                v_losses.append(criterion(logits, lbls).item())
                all_probs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_true.extend(lbls.cpu().numpy())

        history["val_loss"].append(float(np.mean(v_losses)))
        auc     = float(roc_auc_score(all_true, all_probs))
        bal_acc = float(balanced_accuracy_score(all_true, all_preds))
        f1      = float(f1_score(all_true, all_preds, pos_label=1, zero_division=0))
        cm      = confusion_matrix(all_true, all_preds)
        tn, fp, fn, tp = cm.ravel()
        sens = float(tp / (tp + fn + 1e-8))
        spec = float(tn / (tn + fp + 1e-8))

        history["auc"].append(auc)
        history["bal_acc"].append(bal_acc)
        history["f1"].append(f1)
        history["sensitivity"].append(sens)
        history["specificity"].append(spec)

        print(f"  Ep {epoch+1:02d} | Loss: {history['train_loss'][-1]:.4f}/"
              f"{history['val_loss'][-1]:.4f} | AUC: {auc:.4f} | "
              f"BalAcc: {bal_acc:.4f} | Sens: {sens:.4f} | Spec: {spec:.4f}",
              flush=True)

        scheduler.step(history["val_loss"][-1])

        if auc > best_auc:
            best_auc   = auc
            best_state = {k: v.cpu() for k, v in
                          (model.module.state_dict() if MULTI_GPU
                           else model.state_dict()).items()}
            patience_counter = 0
            print(f"  ✓ Best AUC: {auc:.4f}", flush=True)
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    return model, best_state, history, best_auc


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATE ON CONFIGURED SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_scenarios(model, subjects, labels, modality_files, is_attention=True):
    model.eval()
    scenario_results = {}

    for sc_name, avail_ch in TEST_SCENARIOS.items():
        ds     = GliomaDataset(subjects, labels, DATA_DIR, modality_files, IMG_SIZE,
                                missing=avail_ch, cache_dir=CACHE_DIR if USE_CACHE else None)
        loader = DataLoader(ds, batch_size=BATCH, shuffle=False,
                            num_workers=8, pin_memory=True)

        all_probs, all_preds, all_true = [], [], []
        all_attn_weights = []

        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(DEVICE)
                if is_attention:
                    logits, attn_w = model(imgs, return_weights=True)
                    all_attn_weights.extend(attn_w.cpu().numpy().tolist())
                else:
                    logits = model(imgs)
                probs = torch.softmax(logits, 1)[:, 1].cpu().numpy()
                preds = logits.argmax(1).cpu().numpy()
                all_probs.extend(probs.tolist())
                all_preds.extend(preds.tolist())
                all_true.extend(lbls.numpy().tolist())

        auc     = float(roc_auc_score(all_true, all_probs))
        bal_acc = float(balanced_accuracy_score(all_true, all_preds))
        f1      = float(f1_score(all_true, all_preds, pos_label=1, zero_division=0))
        cm      = confusion_matrix(all_true, all_preds)
        tn, fp, fn, tp = cm.ravel()
        sens = float(tp / (tp + fn + 1e-8))
        spec = float(tn / (tn + fp + 1e-8))

        fpr, tpr, thresholds = roc_curve(all_true, all_probs)

        scenario_results[sc_name] = {
            "metrics": {
                "AUC"        : round(auc,     4),
                "BalAcc"     : round(bal_acc, 4),
                "Sensitivity": round(sens,    4),
                "Specificity": round(spec,    4),
                "F1"         : round(f1,      4),
            },
            "confusion_matrix": cm.tolist(),
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist()
            },
            "per_subject": [
                {
                    "subject_id"  : subjects[i],
                    "true_label"  : int(all_true[i]),
                    "pred_label"  : int(all_preds[i]),
                    "pred_prob"   : round(float(all_probs[i]), 4),
                    "correct"     : bool(all_true[i] == all_preds[i]),
                    "attn_weights": all_attn_weights[i] if all_attn_weights else None
                }
                for i in range(len(subjects))
            ]
        }

        print(f"    {sc_name:20s} | AUC: {auc:.4f} | "
              f"Sens: {sens:.4f} | Spec: {spec:.4f}", flush=True)

    return scenario_results


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PREP
# ═══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(TSV_PATH, sep='\t')
df = df[df['IDH'].isin(['mutated', 'wild type'])]
df = df[df['T1'] == 'Available']
df['label'] = (df['IDH'] == 'mutated').astype(int)
df = df[df['Subject ID'].apply(
    lambda s: os.path.isdir(os.path.join(DATA_DIR, s)))]

df['scanner'] = df['Scanner Make'] + '_' + df['Scanner Strength'].astype(str) + 'T'

subjects     = df['Subject ID'].tolist()
labels       = df['label'].tolist()
scanner_info = dict(zip(df['Subject ID'], df['scanner']))
grade_info   = dict(zip(df['Subject ID'], df['Tumor Grade']))

n_mut = sum(labels)
n_wt  = len(labels) - n_mutmplemented via simple operations, such as summation or concatenation, but this might not be the best choice. In this work, we propose a uniform and general scheme, namely attentional feature fusion, which is applicable for most common scenarios, including feature fusion induced by short and long skip connections as well as within Inception layers. To better fuse features of inconsistent semantics and scales, we propose a multi-scale channel attention module, which addresses issues that arise when fusing features given at different scales. We also demons
print(f"Total: {len(subjects)} | Mutated: {n_mut} ({100*n_mut/len(labels):.1f}%) "
      f"| Wild type: {n_wt} ({100*n_wt/len(labels):.1f}%)\n")

dataset_info = {
    "total": len(subjects),
    "mutated": n_mut,
    "wild_type": n_wt,
    "scanner_info": scanner_info,
    "grade_info": {k: int(v) if pd.notna(v) else None
                   for k, v in grade_info.items()},
    "active_modalities": ACTIVE_MODALITIES,
}
with open(f"{SAVE_DIR}/dataset_info.json", "w") as f:
    json.dump(dataset_info, f, indent=2)
print(f"Saved: {SAVE_DIR}/dataset_info.json")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN SELECTED FOLDS — ResNet Attention Fusion
# ═══════════════════════════════════════════════════════════════════════════════
skf         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
subjects_np = np.array(subjects)
labels_np   = np.array(labels)

MODEL_NAME = "Attention_Fusion_ResNet"
MODALITY_FILES = [ALL_MODALITIES[m] for m in ACTIVE_MODALITIES]

splits = list(skf.split(subjects_np, labels_np))

for rf in RUN_FOLDS:
    if not (1 <= rf <= N_FOLDS):
        raise ValueError(f"RUN_FOLDS contains {rf}, must be in 1..{N_FOLDS}")

all_fold_results = []
overall_start = time.time()

for fold in RUN_FOLDS:
    train_idx, val_idx = splits[fold - 1]

    X_train = subjects_np[train_idx].tolist()
    y_train = labels_np[train_idx].tolist()
    X_val   = subjects_np[val_idx].tolist()
    y_val   = labels_np[val_idx].tolist()

    print(f"\n{'#'*60}")
    print(f"FOLD {fold} / {N_FOLDS}  (modalities: {ACTIVE_MODALITIES})")
    print(f"{'#'*60}")
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")

    save_path = f"{SAVE_DIR}/fold{fold}_{MODEL_NAME}.json"
    if os.path.exists(save_path):
        print(f"\n  [{MODEL_NAME}] Already done — skipping (found {save_path})")
        with open(save_path) as f:
            all_fold_results.append(json.load(f))
        continue

    print(f"\n  ── {MODEL_NAME} ──")

    fold_start = time.time()

    model, best_state, history, best_auc = train_one_fold(
        lambda: ResNetAttentionFusionNet(num_modalities=NUM_CHANNELS),
        X_train, y_train, X_val, y_val, MODALITY_FILES)

    core = model.module if MULTI_GPU else model
    core.load_state_dict(best_state)

    print(f"\n  Evaluating scenarios...")
    scenario_results = evaluate_scenarios(model, X_val, y_val, MODALITY_FILES, is_attention=True)

    ckpt_path = f"{SAVE_DIR}/fold{fold}_{MODEL_NAME}_best.pth"
    torch.save(best_state, ckpt_path)

    fold_data = {
        "fold"            : fold,
        "model"           : MODEL_NAME,
        "train_subjects"  : X_train,
        "val_subjects"    : X_val,
        "train_labels"    : y_train,
        "val_labels"      : y_val,
        "best_auc"        : best_auc,
        "history"         : history,
        "scenario_results": scenario_results,
        "checkpoint_path" : ckpt_path,
        "config": {
            "img_size"          : IMG_SIZE,
            "batch"             : BATCH,
            "epochs"            : EPOCHS,
            "lr"                : LR,
            "patience"          : PATIENCE,
            "dropout_p"         : DROPOUT_P,
            "class_weights"     : CLASS_WEIGHTS,
            "gpu_ids"           : GPU_IDS,
            "feat_dim"          : 256,
            "active_modalities" : ACTIVE_MODALITIES,
            "n_folds"           : N_FOLDS,
        }
    }

    with open(save_path, "w") as f:
        json.dump(fold_data, f, indent=2)
    print(f"  ✓ Saved: {save_path}")

    all_fold_results.append(fold_data)

    del model
    torch.cuda.empty_cache()

    fold_elapsed = (time.time() - fold_start) / 60
    print(f"  Fold {fold} done in {fold_elapsed:.1f} min | Best AUC: {best_auc:.4f}")

overall_elapsed = (time.time() - overall_start) / 60
print(f"\n{'='*60}")
print(f"All {len(RUN_FOLDS)} requested fold(s) complete in {overall_elapsed:.1f} minutes")
print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE (over RUN_FOLDS only)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\nSUMMARY — Mean ± Std AUC across run folds {RUN_FOLDS}\n")

summary = {MODEL_NAME: {}, "run_folds": RUN_FOLDS, "active_modalities": ACTIVE_MODALITIES}
folds_data = []
for fold in RUN_FOLDS:
    fp = f"{SAVE_DIR}/fold{fold}_{MODEL_NAME}.json"
    if os.path.exists(fp):
        with open(fp) as f:
            folds_data.append(json.load(f))

for sc_name in TEST_SCENARIOS:
    aucs = [fd["scenario_results"][sc_name]["metrics"]["AUC"]
            for fd in folds_data if sc_name in fd["scenario_results"]]
    if not aucs:
        continue
    summary[MODEL_NAME][sc_name] = {
        "mean": round(float(np.mean(aucs)), 4),
        "std" : round(float(np.std(aucs)),  4)
    }
    print(f"  {MODEL_NAME:25s} | {sc_name:20s} | "
          f"AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

with open(f"{SAVE_DIR}/summary_folds_{'_'.join(map(str, RUN_FOLDS))}.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {SAVE_DIR}/summary_folds_{'_'.join(map(str, RUN_FOLDS))}.json")

print("\n✓ Run complete!")