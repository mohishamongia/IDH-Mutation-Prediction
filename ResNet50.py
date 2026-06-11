"""
3D ResNet50 for IDH Mutation Prediction — 5-Fold Stratified Cross-Validation
Dataset : UTSW-Glioma
Input   : 4-channel MRI (T1, T1CE, T2, FLAIR) resized to 96x96x96
Output  : IDH status  0=wild type  1=mutated
Metrics : AUC, Balanced Accuracy, Sensitivity, Specificity, F1
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score,
    f1_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

# ─── GPU CONFIG ───────────────────────────────────────────────────────────────
# ✏️  Edit this list to whichever GPUs are free today
# Examples:
#   GPU_IDS = [3]          # single GPU
#   GPU_IDS = [2, 5, 7]    # three GPUs
#   GPU_IDS = [1, 3, 6]    # different three GPUs
GPU_IDS = [0]              # ← change this after checking with admin

# Setup
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in GPU_IDS)
DEVICE     = torch.device("cuda:0")   # always 0 after CUDA_VISIBLE_DEVICES
MULTI_GPU  = len(GPU_IDS) > 1

print(f"Requested GPUs : {GPU_IDS}")
print(f"Multi-GPU mode : {MULTI_GPU}")
for i, gid in enumerate(GPU_IDS):
    mem = torch.cuda.get_device_properties(i).total_memory // 1024**2
    print(f"  cuda:{i}  →  Physical GPU {gid}  |  {mem} MB total")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR  = "/workspace/utsw_glioma/images"
TSV_PATH  = "/workspace/utsw_glioma/UTSW_Glioma_Metadata.tsv"
IMG_SIZE  = 96       # larger size = better features, A100 handles easily
BATCH     = 4 * len(GPU_IDS)   # smaller batch than ResNet18 (more params → more VRAM)
EPOCHS    = 30
LR        = 1e-4
PATIENCE  = 7
N_FOLDS   = 5
SEED      = 42

print(f"Batch size     : {BATCH}")
print(f"Image size     : {IMG_SIZE}³")
print(f"Cross-val folds: {N_FOLDS}")

# ─── DATASET ──────────────────────────────────────────────────────────────────
class GliomaDataset(Dataset):
    def __init__(self, subjects, labels, data_dir, size=96):
        self.subjects = subjects
        self.labels   = labels
        self.data_dir = data_dir
        self.size     = size

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
            "brain_t1_ants.nii.gz",
            "brain_t1ce_ants.nii.gz",
            "brain_t2_ants.nii.gz",
            "brain_fl_ants.nii.gz",
        ]
        channels = [self.load_volume(os.path.join(folder, f)) for f in files]
        volume   = np.stack(channels, axis=0)          # (4, 96, 96, 96)
        return torch.tensor(volume), torch.tensor(self.labels[idx], dtype=torch.long)


# ─── 3D RESNET50 ─────────────────────────────────────────────────────────────

class Bottleneck3D(nn.Module):
    """
    Bottleneck residual block for ResNet50/101/152.
    Structure: 1x1x1 (reduce) → 3x3x3 (convolve) → 1x1x1 (expand)
    Expansion factor = 4: output channels = out_ch * 4
    """
    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        # 1x1x1 — reduce channels
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)

        # 3x3x3 — spatial convolution
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)

        # 1x1x1 — expand channels
        self.conv3 = nn.Conv3d(out_ch, out_ch * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm3d(out_ch * self.expansion)

        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class ResNet50_3D(nn.Module):
    """
    3D ResNet-50 for volumetric classification.
    Layer config: [3, 4, 6, 3] Bottleneck blocks
    Channels:     64 → 256 → 512 → 1024 → 2048
    """
    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()
        self.in_ch = 64   # tracks current channel count

        # Stem: 7x7x7 conv + BN + ReLU + MaxPool
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        # ResNet-50 layer config: [3, 4, 6, 3]
        self.layer1 = self._make_layer(64,  num_blocks=3, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=4, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=6, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=3, stride=2)

        # Classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512 * Bottleneck3D.expansion, num_classes)  # 2048 → num_classes
        )

        # Weight initialisation (He / Kaiming)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_ch, num_blocks, stride):
        """Build a residual stage with `num_blocks` Bottleneck blocks."""
        downsample = None
        expanded = out_ch * Bottleneck3D.expansion  # e.g. 64*4 = 256

        # Need downsample if spatial stride > 1 OR channel mismatch
        if stride != 1 or self.in_ch != expanded:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_ch, expanded, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm3d(expanded)
            )

        layers = [Bottleneck3D(self.in_ch, out_ch, stride, downsample)]
        self.in_ch = expanded   # update channel count after first block

        for _ in range(1, num_blocks):
            layers.append(Bottleneck3D(self.in_ch, out_ch))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)


# ─── DATA PREP ────────────────────────────────────────────────────────────────
df = pd.read_csv(TSV_PATH, sep='\t')
df = df[df['IDH'].isin(['mutated', 'wild type'])]
df = df[df['T1'] == 'Available']
df['label'] = (df['IDH'] == 'mutated').astype(int)
df = df[df['Subject ID'].apply(
    lambda s: os.path.isdir(os.path.join(DATA_DIR, s)))]

subjects = np.array(df['Subject ID'].tolist())
labels   = np.array(df['label'].tolist())

n_mut = labels.sum()
n_wt  = len(labels) - n_mut
print(f"\nTotal subjects : {len(subjects)}")
print(f"IDH mutated    : {n_mut}  ({100*n_mut/len(labels):.1f}%)")
print(f"IDH wild type  : {n_wt}  ({100*n_wt/len(labels):.1f}%)")


# ─── 5-FOLD CROSS-VALIDATION ─────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Collect per-fold results
fold_results = {
    'auc': [], 'bal_acc': [], 'sensitivity': [],
    'specificity': [], 'f1': []
}

# Collect per-fold training curves for plotting
all_fold_train_losses = []
all_fold_val_losses   = []
all_fold_val_aucs     = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(subjects, labels)):
    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx + 1} / {N_FOLDS}")
    print(f"{'='*60}")
    print(f"  Train: {len(train_idx)} subjects  |  Val: {len(val_idx)} subjects")

    X_train, y_train = subjects[train_idx].tolist(), labels[train_idx].tolist()
    X_val,   y_val   = subjects[val_idx].tolist(),   labels[val_idx].tolist()

    train_ds     = GliomaDataset(X_train, y_train, DATA_DIR, IMG_SIZE)
    val_ds       = GliomaDataset(X_val,   y_val,   DATA_DIR, IMG_SIZE)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ─── MODEL SETUP (fresh model each fold) ─────────────────────────────────
    model = ResNet50_3D().to(DEVICE)

    if MULTI_GPU:
        model = nn.DataParallel(model)
        if fold_idx == 0:
            print(f"  DataParallel enabled across {len(GPU_IDS)} GPUs")

    if fold_idx == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    class_weights = torch.tensor([1.0, 2.5]).to(DEVICE)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5)

    # ─── TRACKING ────────────────────────────────────────────────────────────
    train_losses, val_losses        = [], []
    val_aucs, val_bal_accs, val_f1s = [], [], []
    best_auc         = 0.0
    patience_counter = 0

    # ─── TRAINING LOOP ───────────────────────────────────────────────────────
    for epoch in range(EPOCHS):

        # --- Training ---
        model.train()
        batch_losses = []
        for imgs, lbls in tqdm(train_loader,
                                desc=f"Fold {fold_idx+1} Epoch {epoch+1}/{EPOCHS} [Train]"):
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(loss.item())
        train_losses.append(np.mean(batch_losses))

        # --- Validation ---
        model.eval()
        all_preds, all_probs, all_true, v_losses = [], [], [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                logits = model(imgs)
                v_losses.append(criterion(logits, lbls).item())
                probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds  = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_probs.extend(probs)
                all_true.extend(lbls.cpu().numpy())

        val_losses.append(np.mean(v_losses))

        # ── Metrics ──────────────────────────────────────────────────────────
        bal_acc = balanced_accuracy_score(all_true, all_preds)
        auc     = roc_auc_score(all_true, all_probs)
        f1      = f1_score(all_true, all_preds, pos_label=1, zero_division=0)
        cm      = confusion_matrix(all_true, all_preds)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)

        val_aucs.append(auc)
        val_bal_accs.append(bal_acc)
        val_f1s.append(f1)

        print(f"  Fold {fold_idx+1} Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Loss T/V: {train_losses[-1]:.4f}/{val_losses[-1]:.4f} | "
              f"AUC: {auc:.4f} | "
              f"BalAcc: {bal_acc:.4f} | "
              f"Sens: {sensitivity:.4f} | "
              f"Spec: {specificity:.4f} | "
              f"F1: {f1:.4f}", flush=True)

        scheduler.step(val_losses[-1])

        # ── Save best model for this fold ────────────────────────────────────
        if auc > best_auc:
            best_auc = auc
            state = model.module.state_dict() if MULTI_GPU else model.state_dict()
            torch.save(state, f"idh_resnet50_best_fold{fold_idx+1}.pth")
            print(f"    ✓ Best model saved!  AUC: {auc:.4f}", flush=True)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch+1}")
            break

    # ─── FOLD REPORT ─────────────────────────────────────────────────────────
    # Load best model for final evaluation on this fold's val set
    best_model = ResNet50_3D().to(DEVICE)
    best_model.load_state_dict(
        torch.load(f"idh_resnet50_best_fold{fold_idx+1}.pth", map_location=DEVICE)
    )
    best_model.eval()

    all_preds, all_probs, all_true = [], [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            logits = best_model(imgs)
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_true.extend(lbls.cpu().numpy())

    fold_auc  = roc_auc_score(all_true, all_probs)
    fold_bacc = balanced_accuracy_score(all_true, all_preds)
    fold_f1   = f1_score(all_true, all_preds, pos_label=1, zero_division=0)
    cm        = confusion_matrix(all_true, all_preds)
    tn, fp, fn, tp = cm.ravel()
    fold_sens = tp / (tp + fn + 1e-8)
    fold_spec = tn / (tn + fp + 1e-8)

    fold_results['auc'].append(fold_auc)
    fold_results['bal_acc'].append(fold_bacc)
    fold_results['sensitivity'].append(fold_sens)
    fold_results['specificity'].append(fold_spec)
    fold_results['f1'].append(fold_f1)

    print(f"\n  Fold {fold_idx+1} Best Results:")
    print(f"    AUC         : {fold_auc:.4f}")
    print(f"    BalAcc      : {fold_bacc:.4f}")
    print(f"    Sensitivity : {fold_sens:.4f}")
    print(f"    Specificity : {fold_spec:.4f}")
    print(f"    F1          : {fold_f1:.4f}")
    print(classification_report(all_true, all_preds,
          target_names=['Wild Type', 'Mutated']))

    # Store curves for plotting
    all_fold_train_losses.append(train_losses)
    all_fold_val_losses.append(val_losses)
    all_fold_val_aucs.append(val_aucs)


# ─── CROSS-VALIDATION SUMMARY ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  5-FOLD CROSS-VALIDATION SUMMARY  (ResNet-50)")
print("=" * 60)

print(f"\n{'Metric':<20} {'Mean':>8} {'Std':>8}   Per-Fold Values")
print("-" * 70)
for metric_name, values in fold_results.items():
    mean_val = np.mean(values)
    std_val  = np.std(values)
    per_fold = "  ".join(f"{v:.4f}" for v in values)
    print(f"{metric_name:<20} {mean_val:>8.4f} {std_val:>8.4f}   [{per_fold}]")

print(f"\n{'='*60}")
print(f"  Overall AUC             : {np.mean(fold_results['auc']):.4f} ± {np.std(fold_results['auc']):.4f}")
print(f"  Overall Balanced Acc    : {np.mean(fold_results['bal_acc']):.4f} ± {np.std(fold_results['bal_acc']):.4f}")
print(f"  Overall Sensitivity     : {np.mean(fold_results['sensitivity']):.4f} ± {np.std(fold_results['sensitivity']):.4f}")
print(f"  Overall Specificity     : {np.mean(fold_results['specificity']):.4f} ± {np.std(fold_results['specificity']):.4f}")
print(f"  Overall F1              : {np.mean(fold_results['f1']):.4f} ± {np.std(fold_results['f1']):.4f}")
print(f"{'='*60}")


# ─── PLOTS ────────────────────────────────────────────────────────────────────

# --- Plot 1: Per-fold training curves ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("ResNet-50  —  5-Fold Cross-Validation Training Curves",
             fontsize=14, fontweight='bold')

for f_idx in range(N_FOLDS):
    epochs_ran = range(1, len(all_fold_train_losses[f_idx]) + 1)

    axes[0].plot(epochs_ran, all_fold_train_losses[f_idx],
                 label=f'Fold {f_idx+1} Train', linestyle='--', alpha=0.7)
    axes[0].plot(epochs_ran, all_fold_val_losses[f_idx],
                 label=f'Fold {f_idx+1} Val', alpha=0.9)

    axes[1].plot(epochs_ran, all_fold_val_aucs[f_idx],
                 label=f'Fold {f_idx+1}', alpha=0.9)

axes[0].set_title('Loss Curves')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend(fontsize=7)

axes[1].set_title('Validation AUC')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('AUC')
axes[1].legend(fontsize=8)

# --- Plot 2 (axes[2]): Summary bar chart ---
metric_names = ['AUC', 'BalAcc', 'Sens', 'Spec', 'F1']
metric_keys  = ['auc', 'bal_acc', 'sensitivity', 'specificity', 'f1']
means = [np.mean(fold_results[k]) for k in metric_keys]
stds  = [np.std(fold_results[k])  for k in metric_keys]

bars = axes[2].bar(metric_names, means, yerr=stds, capsize=5,
                   color=['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974'],
                   edgecolor='black', linewidth=0.5)
axes[2].set_title('CV Summary (mean ± std)')
axes[2].set_ylabel('Score')
axes[2].set_ylim(0, 1.05)
for bar, m, s in zip(bars, means, stds):
    axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.02,
                 f'{m:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("resnet50_5fold_cv_curves.png", dpi=150)
plt.show()
print("Saved: resnet50_5fold_cv_curves.png")

# --- Save per-fold results to CSV ---
results_df = pd.DataFrame(fold_results)
results_df.index = [f"Fold {i+1}" for i in range(N_FOLDS)]
results_df.loc["Mean"] = results_df.mean()
results_df.loc["Std"]  = results_df.iloc[:N_FOLDS].std()
results_df.to_csv("resnet50_5fold_cv_results.csv")
print("Saved: resnet50_5fold_cv_results.csv")
print(f"\nBest models saved as: idh_resnet50_best_fold{{1..{N_FOLDS}}}.pth")
