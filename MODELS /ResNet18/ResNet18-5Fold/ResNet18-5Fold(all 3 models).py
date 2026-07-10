"""
5-Fold Cross Validation — Missing Modality Robust IDH Prediction
Models   : (1) Baseline ResNet18  (2) Dropout ResNet18  (3) Attention Fusion Net
Saves    : All metrics, predictions, attention weights, ROC data per fold
Run      : python idh_cv_train.py
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
    confusion_matrix, classification_report, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# ─── GPU CONFIG ───────────────────────────────────────────────────────────────
GPU_IDS = [0]   # ✏️ edit this every run

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in GPU_IDS)
DEVICE    = torch.device("cuda:0")
MULTI_GPU = len(GPU_IDS) > 1
print(f"GPUs: {GPU_IDS}  |  Multi-GPU: {MULTI_GPU}")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR   = "/workspace/UTSW_Glioma_data/UTSW-Glioma"
TSV_PATH   = "UTSW_Glioma_data/UTSW_Glioma_Metadata-2-1 (2).tsv"
SAVE_DIR   = "results"
IMG_SIZE   = 96
BATCH      = 8 * len(GPU_IDS)
EPOCHS     = 30
LR         = 1e-4
PATIENCE   = 7
N_FOLDS    = 5
MODALITIES = ["T1", "T1CE", "T2", "FLAIR"]

TEST_SCENARIOS = {
    "All_4"         : [0, 1, 2, 3],
    "T1CE_missing"  : [0,    2, 3],
    "FLAIR_missing" : [0, 1, 2   ],
    "T2_missing"    : [0, 1,    3],
    "T1_T2_only"    : [0,    2   ],
    "T1_only"       : [0         ],
}

os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Results will be saved to: {SAVE_DIR}/")
print(f"Batch: {BATCH}  |  Image size: {IMG_SIZE}³  |  Folds: {N_FOLDS}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════
class GliomaDataset(Dataset):
    def __init__(self, subjects, labels, data_dir, size=96,
                 dropout_p=0.0, missing=None):
        self.subjects  = subjects
        self.labels    = labels
        self.data_dir  = data_dir
        self.size      = size
        self.dropout_p = dropout_p
        self.missing   = missing

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
        sid    = self.subjects[idx]
        folder = os.path.join(self.data_dir, sid)
        files  = [
            "brain_t1_ants.nii.gz",    # ch 0
            "brain_t1ce_ants.nii.gz",  # ch 1
            "brain_t2_ants.nii.gz",    # ch 2
            "brain_fl_ants.nii.gz",    # ch 3
        ]
        channels = [self.load_volume(os.path.join(folder, f)) for f in files]
        volume   = np.stack(channels, axis=0).copy()

        # Training: random modality dropout
        if self.dropout_p > 0:
            for c in range(4):
                if np.random.rand() < self.dropout_p:
                    volume[c] = 0.0

        # Test: zero out missing channels
        if self.missing is not None:
            for c in range(4):
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
        self.head   = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Dropout(0.5), nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.head(x)


class ModalityEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 32, 3, stride=2, padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64, 128, 3, stride=2, padding=1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)


class AttentionFusionNet(nn.Module):
    def __init__(self, num_modalities=4, feat_dim=128, num_classes=2):
        super().__init__()
        self.encoders  = nn.ModuleList([ModalityEncoder() for _ in range(num_modalities)])
        self.attention = nn.Sequential(
            nn.Linear(feat_dim * num_modalities, 64), nn.ReLU(),
            nn.Linear(64, num_modalities), nn.Softmax(dim=1))
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(64, num_classes))

    def forward(self, x, return_weights=False):
        feats   = [self.encoders[i](x[:, i:i+1]) for i in range(4)]
        concat  = torch.cat(feats, dim=1)
        weights = self.attention(concat)                    # (B, 4)
        feats_s = torch.stack(feats, dim=1)                # (B, 4, 128)
        fused   = (feats_s * weights.unsqueeze(2)).sum(1)  # (B, 128)
        out     = self.classifier(fused)
        if return_weights:
            return out, weights
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN ONE FOLD
# ═══════════════════════════════════════════════════════════════════════════════
def train_one_fold(model, train_loader, val_loader):
    if MULTI_GPU:
        model = nn.DataParallel(model)
    model = model.to(DEVICE)

    optimizer     = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    class_weights = torch.tensor([1.0, 2.5]).to(DEVICE)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', patience=3, factor=0.5)

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
# EVALUATE ON MISSING MODALITY SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_scenarios(model, subjects, labels, is_attention=False):
    """Returns metrics + per-subject predictions for all scenarios."""
    model.eval()
    scenario_results = {}

    for sc_name, avail_ch in TEST_SCENARIOS.items():
        ds     = GliomaDataset(subjects, labels, DATA_DIR, IMG_SIZE, missing=avail_ch)
        loader = DataLoader(ds, batch_size=BATCH, shuffle=False,
                            num_workers=4, pin_memory=True)

        all_probs, all_preds, all_true = [], [], []
        all_attn_weights = []   # only for attention model

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

        # Metrics
        auc     = float(roc_auc_score(all_true, all_probs))
        bal_acc = float(balanced_accuracy_score(all_true, all_preds))
        f1      = float(f1_score(all_true, all_preds, pos_label=1, zero_division=0))
        cm      = confusion_matrix(all_true, all_preds)
        tn, fp, fn, tp = cm.ravel()
        sens = float(tp / (tp + fn + 1e-8))
        spec = float(tn / (tn + fp + 1e-8))

        # ROC curve data
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
            # Per-subject predictions — for Grad-CAM selection later
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

# Merge scanner metadata for later analysis
df['scanner'] = df['Scanner Make'] + '_' + df['Scanner Strength'].astype(str) + 'T'

subjects     = df['Subject ID'].tolist()
labels       = df['label'].tolist()
scanner_info = dict(zip(df['Subject ID'], df['scanner']))
grade_info   = dict(zip(df['Subject ID'], df['Tumor Grade']))

n_mut = sum(labels)
n_wt  = len(labels) - n_mut
print(f"Total: {len(subjects)} | Mutated: {n_mut} ({100*n_mut/len(labels):.1f}%) "
      f"| Wild type: {n_wt} ({100*n_wt/len(labels):.1f}%)\n")

# Save dataset info
dataset_info = {
    "total": len(subjects),
    "mutated": n_mut,
    "wild_type": n_wt,
    "scanner_info": scanner_info,
    "grade_info": {k: int(v) if pd.notna(v) else None
                   for k, v in grade_info.items()}
}
with open(f"{SAVE_DIR}/dataset_info.json", "w") as f:
    json.dump(dataset_info, f, indent=2)
print(f"Saved: {SAVE_DIR}/dataset_info.json")


# ═══════════════════════════════════════════════════════════════════════════════
# 5-FOLD CROSS VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
skf         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
subjects_np = np.array(subjects)
labels_np   = np.array(labels)

MODEL_CONFIGS = [
    ("Baseline_ResNet18",  lambda: ResNet18_3D(),        0.0,  False),
    ("Dropout_ResNet18",   lambda: ResNet18_3D(),        0.3,  False),
    ("Attention_Fusion",   lambda: AttentionFusionNet(), 0.3,  True),
]

# Master results store
all_fold_results = {name: [] for name, _, _, _ in MODEL_CONFIGS}

start_time = time.time()

for fold, (train_idx, val_idx) in enumerate(skf.split(subjects_np, labels_np)):
    print(f"\n{'#'*60}")
    print(f"FOLD {fold+1} / {N_FOLDS}")
    print(f"{'#'*60}")

    X_train = subjects_np[train_idx].tolist()
    y_train = labels_np[train_idx].tolist()
    X_val   = subjects_np[val_idx].tolist()
    y_val   = labels_np[val_idx].tolist()

    print(f"Train: {len(X_train)} | Val: {len(X_val)}")

    for model_name, model_fn, dropout_p, is_attention in MODEL_CONFIGS:

        # Skip if already saved (crash recovery)
        save_path = f"{SAVE_DIR}/fold{fold+1}_{model_name}.json"
        if os.path.exists(save_path):
            print(f"\n  [{model_name}] Already done — skipping (found {save_path})")
            continue

        print(f"\n  ── {model_name} ──")

        # Dataloaders
        train_ds = GliomaDataset(X_train, y_train, DATA_DIR, IMG_SIZE,
                                  dropout_p=dropout_p)
        val_ds   = GliomaDataset(X_val, y_val, DATA_DIR, IMG_SIZE)
        train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                                   num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                                   num_workers=4, pin_memory=True)

        # Train
        model, best_state, history, best_auc = train_one_fold(
            model_fn(), train_loader, val_loader)

        # Load best weights for evaluation
        core = model.module if MULTI_GPU else model
        core.load_state_dict(best_state)

        # Evaluate on all missing modality scenarios
        print(f"\n  Evaluating scenarios...")
        scenario_results = evaluate_scenarios(
            model, X_val, y_val, is_attention=is_attention)

        # Save model checkpoint
        ckpt_path = f"{SAVE_DIR}/fold{fold+1}_{model_name}_best.pth"
        torch.save(best_state, ckpt_path)

        # Build complete fold result
        fold_data = {
            "fold"            : fold + 1,
            "model"           : model_name,
            "train_subjects"  : X_train,
            "val_subjects"    : X_val,
            "train_labels"    : y_train,
            "val_labels"      : y_val,
            "best_auc"        : best_auc,
            "history"         : history,
            "scenario_results": scenario_results,
            "checkpoint_path" : ckpt_path,
            "config": {
                "img_size"  : IMG_SIZE,
                "batch"     : BATCH,
                "epochs"    : EPOCHS,
                "lr"        : LR,
                "dropout_p" : dropout_p,
                "gpu_ids"   : GPU_IDS,
            }
        }

        # Save JSON — crash safe
        with open(save_path, "w") as f:
            json.dump(fold_data, f, indent=2)
        print(f"  ✓ Saved: {save_path}")

        all_fold_results[model_name].append(fold_data)

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

elapsed = (time.time() - start_time) / 60
print(f"\n{'='*60}")
print(f"All folds complete in {elapsed:.1f} minutes")
print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\nSUMMARY — Mean ± Std AUC across folds\n")

summary = {}
for model_name, _, _, _ in MODEL_CONFIGS:
    summary[model_name] = {}
    folds_data = []
    for fold_f in range(1, N_FOLDS + 1):
        fp = f"{SAVE_DIR}/fold{fold_f}_{model_name}.json"
        if os.path.exists(fp):
            with open(fp) as f:
                folds_data.append(json.load(f))

    for sc_name in TEST_SCENARIOS:
        aucs = [fd["scenario_results"][sc_name]["metrics"]["AUC"]
                for fd in folds_data]
        summary[model_name][sc_name] = {
            "mean": round(float(np.mean(aucs)), 4),
            "std" : round(float(np.std(aucs)),  4)
        }
        print(f"  {model_name:25s} | {sc_name:20s} | "
              f"AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

with open(f"{SAVE_DIR}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {SAVE_DIR}/summary.json")

print("\n✓ Training complete! Run idh_cv_plot.py to generate all figures.")
print(f"\nFiles saved in {SAVE_DIR}/:")
print("  dataset_info.json")
print("  summary.json")
print("  fold[1-5]_[model].json          ← all metrics + predictions")
print("  fold[1-5]_[model]_best.pth      ← model checkpoints")