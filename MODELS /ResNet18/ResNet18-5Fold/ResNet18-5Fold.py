"""
3D ResNet18 for IDH Mutation Prediction — with Stratified K-Fold Cross-Validation
Dataset : UTSW-Glioma
Input   : 4-channel MRI (T1, T1CE, T2, FLAIR) resized to 96x96x96
Output  : IDH status 0=wild type 1=mutated
Metrics : AUC, Balanced Accuracy, Sensitivity, Specificity, F1
CV      : Stratified K-Fold (default K=5)
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
GPU_IDS = [0]  # ← change this every run

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in GPU_IDS)
DEVICE    = torch.device("cuda:0")
MULTI_GPU = len(GPU_IDS) > 1

print(f"Requested GPUs : {GPU_IDS}")
print(f"Multi-GPU mode : {MULTI_GPU}")
for i, gid in enumerate(GPU_IDS):
    mem = torch.cuda.get_device_properties(i).total_memory // 1024**2
    print(f"  cuda:{i} → Physical GPU {gid} | {mem} MB total")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR = "/workspace/UTSW_Glioma_data/UTSW-Glioma"
TSV_PATH = "UTSW_Glioma_data/UTSW_Glioma_Metadata-2-1 (2).tsv"
IMG_SIZE = 96
BATCH    = 8 * len(GPU_IDS)
EPOCHS   = 30
LR       = 1e-4
PATIENCE = 7
N_FOLDS  = 5        # ← number of CV folds; change to 3 for faster runs

print(f"Batch size : {BATCH}")
print(f"Image size : {IMG_SIZE}³")
print(f"CV folds   : {N_FOLDS}")


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
        volume   = np.stack(channels, axis=0)  # (4, 96, 96, 96)
        return torch.tensor(volume), torch.tensor(self.labels[idx], dtype=torch.long)


# ─── 3D RESNET18 ──────────────────────────────────────────────────────────────
class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)

        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.skip(x))


class ResNet18_3D(nn.Module):
    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = nn.Sequential(ResBlock3D(64, 64),             ResBlock3D(64, 64))
        self.layer2 = nn.Sequential(ResBlock3D(64, 128, stride=2),  ResBlock3D(128, 128))
        self.layer3 = nn.Sequential(ResBlock3D(128, 256, stride=2), ResBlock3D(256, 256))
        self.layer4 = nn.Sequential(ResBlock3D(256, 512, stride=2), ResBlock3D(512, 512))
        self.head   = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)


# ─── HELPER: build a fresh model + optimizer + criterion ──────────────────────
def build_model():
    m = ResNet18_3D().to(DEVICE)
    if MULTI_GPU:
        m = nn.DataParallel(m)
    opt  = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.5]).to(DEVICE))
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', patience=3, factor=0.5)
    return m, opt, crit, sch


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
print(f"IDH mutated    : {n_mut} ({100*n_mut/len(labels):.1f}%)")
print(f"IDH wild type  : {n_wt} ({100*n_wt/len(labels):.1f}%)")


# ─── CROSS-VALIDATION LOOP ────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Per-fold result storage
fold_results = []                      # list of dicts, one per fold
all_fold_train_losses = []             # shape: [fold][epoch]
all_fold_val_losses   = []
all_fold_val_aucs     = []

global_best_auc  = 0.0
global_best_fold = -1

for fold, (train_idx, val_idx) in enumerate(skf.split(subjects, labels)):
    print(f"\n{'='*60}")
    print(f"  FOLD {fold+1} / {N_FOLDS}")
    print(f"{'='*60}")
    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)}")

    X_train, y_train = subjects[train_idx].tolist(), labels[train_idx].tolist()
    X_val,   y_val   = subjects[val_idx].tolist(),   labels[val_idx].tolist()

    train_ds = GliomaDataset(X_train, y_train, DATA_DIR, IMG_SIZE)
    val_ds   = GliomaDataset(X_val,   y_val,   DATA_DIR, IMG_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                              num_workers=4, pin_memory=True)

    model, optimizer, criterion, scheduler = build_model()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {total_params:,}")

    # Per-epoch tracking for this fold
    train_losses, val_losses = [], []
    val_aucs                 = []
    best_fold_auc            = 0.0
    patience_counter         = 0
    fold_best_preds          = None   # saved at best epoch for this fold
    fold_best_probs          = None
    fold_best_true           = None

    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        batch_losses = []
        for imgs, lbls in tqdm(train_loader,
                               desc=f"Fold {fold+1} Epoch {epoch+1}/{EPOCHS} [Train]"):
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
                logits     = model(imgs)
                v_losses.append(criterion(logits, lbls).item())
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_probs.extend(probs)
                all_true.extend(lbls.cpu().numpy())

        val_losses.append(np.mean(v_losses))

        bal_acc     = balanced_accuracy_score(all_true, all_preds)
        auc         = roc_auc_score(all_true, all_probs)
        f1          = f1_score(all_true, all_preds, pos_label=1, zero_division=0)
        cm          = confusion_matrix(all_true, all_preds)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)

        val_aucs.append(auc)

        print(f"  Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Loss T/V: {train_losses[-1]:.4f}/{val_losses[-1]:.4f} | "
              f"AUC: {auc:.4f} | BalAcc: {bal_acc:.4f} | "
              f"Sens: {sensitivity:.4f} | Spec: {specificity:.4f} | "
              f"F1: {f1:.4f}", flush=True)

        scheduler.step(val_losses[-1])

        # Save best checkpoint for this fold
        if auc > best_fold_auc:
            best_fold_auc    = auc
            fold_best_preds  = list(all_preds)
            fold_best_probs  = list(all_probs)
            fold_best_true   = list(all_true)
            state = model.module.state_dict() if MULTI_GPU else model.state_dict()
            torch.save(state, f"idh_resnet18_fold{fold+1}_best.pth")
            print(f"  ✓ Fold {fold+1} best model saved (AUC={auc:.4f})", flush=True)
            patience_counter = 0

            # Track global best
            if auc > global_best_auc:
                global_best_auc  = auc
                global_best_fold = fold + 1
                torch.save(state, "idh_resnet18_global_best.pth")
                print(f"  ★ New global best! AUC={auc:.4f} (Fold {fold+1})", flush=True)
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch+1} (fold {fold+1})")
            break

    # Compute final metrics at best epoch for this fold
    cm_best      = confusion_matrix(fold_best_true, fold_best_preds)
    tn, fp, fn, tp = cm_best.ravel()
    fold_results.append({
        "fold":        fold + 1,
        "auc":         roc_auc_score(fold_best_true, fold_best_probs),
        "bal_acc":     balanced_accuracy_score(fold_best_true, fold_best_preds),
        "sensitivity": tp / (tp + fn + 1e-8),
        "specificity": tn / (tn + fp + 1e-8),
        "f1":          f1_score(fold_best_true, fold_best_preds,
                                pos_label=1, zero_division=0),
    })

    all_fold_train_losses.append(train_losses)
    all_fold_val_losses.append(val_losses)
    all_fold_val_aucs.append(val_aucs)

    print(f"\n  Fold {fold+1} summary  →  "
          f"AUC: {fold_results[-1]['auc']:.4f} | "
          f"BalAcc: {fold_results[-1]['bal_acc']:.4f} | "
          f"Sens: {fold_results[-1]['sensitivity']:.4f} | "
          f"Spec: {fold_results[-1]['specificity']:.4f} | "
          f"F1: {fold_results[-1]['f1']:.4f}")

    # Free GPU memory before next fold
    del model, optimizer, criterion, scheduler
    torch.cuda.empty_cache()


# ─── AGGREGATE CROSS-VALIDATION RESULTS ──────────────────────────────────────
print("\n" + "="*60)
print("CROSS-VALIDATION SUMMARY")
print("="*60)

metrics = ["auc", "bal_acc", "sensitivity", "specificity", "f1"]
labels_pretty = ["AUC", "Bal. Acc.", "Sensitivity", "Specificity", "F1"]

results_df = pd.DataFrame(fold_results).set_index("fold")
print(results_df.to_string())

print("\n--- Mean ± Std across folds ---")
for m, lbl in zip(metrics, labels_pretty):
    vals = results_df[m].values
    print(f"  {lbl:14s}: {vals.mean():.4f} ± {vals.std():.4f}")

print(f"\nGlobal best AUC: {global_best_auc:.4f} (Fold {global_best_fold})")
print("Best model saved: idh_resnet18_global_best.pth")


# ─── PLOTS ────────────────────────────────────────────────────────────────────
max_epochs = max(len(x) for x in all_fold_train_losses)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"Cross-Validation Training Curves ({N_FOLDS}-Fold)", fontsize=14)

# 1. Loss curves
ax = axes[0]
for f_idx, (tl, vl) in enumerate(zip(all_fold_train_losses, all_fold_val_losses)):
    ep = range(1, len(tl) + 1)
    ax.plot(ep, tl, alpha=0.35, color='steelblue',   linewidth=0.9)
    ax.plot(ep, vl, alpha=0.35, color='darkorange',  linewidth=0.9)
# Mean lines (pad shorter folds with NaN for averaging)
def pad_nan(lst):
    out = np.full((len(lst), max_epochs), np.nan)
    for i, l in enumerate(lst):
        out[i, :len(l)] = l
    return out

mean_tl = np.nanmean(pad_nan(all_fold_train_losses), axis=0)
mean_vl = np.nanmean(pad_nan(all_fold_val_losses),   axis=0)
ep_mean = range(1, max_epochs + 1)
ax.plot(ep_mean, mean_tl, color='steelblue',  linewidth=2.2, label='Train (mean)')
ax.plot(ep_mean, mean_vl, color='darkorange', linewidth=2.2, label='Val (mean)')
ax.set_title('Loss')
ax.set_xlabel('Epoch')
ax.legend()

# 2. AUC per fold
ax = axes[1]
for f_idx, aucs in enumerate(all_fold_val_aucs):
    ep = range(1, len(aucs) + 1)
    ax.plot(ep, aucs, alpha=0.45, linewidth=0.9,
            label=f'Fold {f_idx+1}')
mean_auc = np.nanmean(pad_nan(all_fold_val_aucs), axis=0)
ax.plot(ep_mean, mean_auc, color='black', linewidth=2.2, label='Mean')
ax.set_title('Validation AUC per Fold')
ax.set_xlabel('Epoch')
ax.legend(fontsize=8)

# 3. Per-fold final metrics bar chart
ax = axes[2]
fold_labels = [f"Fold {r['fold']}" for r in fold_results]
x    = np.arange(len(fold_results))
w    = 0.15
cols = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336']
for i, (m, lbl) in enumerate(zip(metrics, labels_pretty)):
    vals = [r[m] for r in fold_results]
    ax.bar(x + i * w, vals, width=w, label=lbl, color=cols[i], alpha=0.85)
ax.set_xticks(x + 2 * w)
ax.set_xticklabels(fold_labels, fontsize=8)
ax.set_ylim(0, 1.05)
ax.set_title('Final Metrics per Fold')
ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("resnet18_kfold_curves.png", dpi=150)
plt.show()
print("Saved: resnet18_kfold_curves.png")