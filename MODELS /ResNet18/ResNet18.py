"""
3D ResNet18 for IDH Mutation Prediction
Dataset : UTSW-Glioma
Input : 4-channel MRI (T1, T1CE, T2, FLAIR) resized to 96x96x96
Output : IDH status 0=wild type 1=mutated
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score,
    f1_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

# ─── GPU CONFIG ───────────────────────────────────────────────────────────────
# ✏️ Edit this list to whichever GPUs are free today
# Examples:
# GPU_IDS = [3]        # single GPU
# GPU_IDS = [2, 5, 7]  # three GPUs
# GPU_IDS = [1, 3, 6]  # different three GPUs
GPU_IDS = [0]  # ← change this every run

# Setup
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in GPU_IDS)
DEVICE = torch.device("cuda:0")  # always 0 after CUDA_VISIBLE_DEVICES
MULTI_GPU = len(GPU_IDS) > 1

print(f"Requested GPUs : {GPU_IDS}")
print(f"Multi-GPU mode : {MULTI_GPU}")
for i, gid in enumerate(GPU_IDS):
    mem = torch.cuda.get_device_properties(i).total_memory // 1024**2
    print(f"  cuda:{i} → Physical GPU {gid} | {mem} MB total")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR =  "/workspace/UTSW_Glioma_data/UTSW-Glioma"
TSV_PATH = "/workspace/UTSW_Glioma_Metadata-2-1 (2).tsv"
IMG_SIZE = 96        # larger size = better features, A100 handles easily
BATCH    = 8 * len(GPU_IDS)  # scale batch with number of GPUs
EPOCHS   = 30
LR       = 1e-4
PATIENCE = 7

print(f"Batch size : {BATCH}")
print(f"Image size : {IMG_SIZE}³")


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
    """Basic residual block: two 3x3x3 convs + skip connection."""
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
        return self.relu(out + self.skip(x))  # residual add


class ResNet18_3D(nn.Module):
    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = nn.Sequential(ResBlock3D(64, 64),              ResBlock3D(64, 64))
        self.layer2 = nn.Sequential(ResBlock3D(64, 128, stride=2),   ResBlock3D(128, 128))
        self.layer3 = nn.Sequential(ResBlock3D(128, 256, stride=2),  ResBlock3D(256, 256))
        self.layer4 = nn.Sequential(ResBlock3D(256, 512, stride=2),  ResBlock3D(512, 512))
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


# ─── DATA PREP ────────────────────────────────────────────────────────────────
df = pd.read_csv(TSV_PATH, sep='\t')
df = df[df['IDH'].isin(['mutated', 'wild type'])]
df = df[df['T1'] == 'Available']
df['label'] = (df['IDH'] == 'mutated').astype(int)
df = df[df['Subject ID'].apply(
    lambda s: os.path.isdir(os.path.join(DATA_DIR, s)))]

subjects = df['Subject ID'].tolist()
labels   = df['label'].tolist()

n_mut = sum(labels)
n_wt  = len(labels) - n_mut
print(f"\nTotal subjects : {len(subjects)}")
print(f"IDH mutated    : {n_mut} ({100*n_mut/len(labels):.1f}%)")
print(f"IDH wild type  : {n_wt} ({100*n_wt/len(labels):.1f}%)")

X_train, X_val, y_train, y_val = train_test_split(
    subjects, labels, test_size=0.2, stratify=labels, random_state=42)

train_ds = GliomaDataset(X_train, y_train, DATA_DIR, IMG_SIZE)
val_ds   = GliomaDataset(X_val,   y_val,   DATA_DIR, IMG_SIZE)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                          num_workers=4, pin_memory=True)


# ─── MODEL SETUP ──────────────────────────────────────────────────────────────
model = ResNet18_3D().to(DEVICE)

if MULTI_GPU:
    # DataParallel splits each batch across all specified GPUs automatically
    model = nn.DataParallel(model)
    print(f"\nDataParallel enabled across {len(GPU_IDS)} GPUs")

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {total_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

class_weights = torch.tensor([1.0, 2.5]).to(DEVICE)
criterion     = nn.CrossEntropyLoss(weight=class_weights)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.5)

# ─── TRACKING ─────────────────────────────────────────────────────────────────
train_losses, val_losses          = [], []
val_aucs, val_bal_accs, val_f1s   = [], [], []
best_auc        = 0.0
patience_counter = 0

# ─── TRAINING LOOP ────────────────────────────────────────────────────────────
for epoch in range(EPOCHS):

    # --- Training ---
    model.train()
    batch_losses = []
    for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
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

    # ── Metrics ──────────────────────────────────────────────────────────────
    bal_acc     = balanced_accuracy_score(all_true, all_preds)
    auc         = roc_auc_score(all_true, all_probs)
    f1          = f1_score(all_true, all_preds, pos_label=1, zero_division=0)
    cm          = confusion_matrix(all_true, all_preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    val_aucs.append(auc)
    val_bal_accs.append(bal_acc)
    val_f1s.append(f1)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Loss T/V: {train_losses[-1]:.4f}/{val_losses[-1]:.4f} | "
          f"AUC: {auc:.4f} | "
          f"BalAcc: {bal_acc:.4f} | "
          f"Sens: {sensitivity:.4f} | "
          f"Spec: {specificity:.4f} | "
          f"F1: {f1:.4f}", flush=True)

    scheduler.step(val_losses[-1])

    # ── Save best model ───────────────────────────────────────────────────────
    if auc > best_auc:
        best_auc = auc
        # Save underlying model (unwrap DataParallel if needed)
        state = model.module.state_dict() if MULTI_GPU else model.state_dict()
        torch.save(state, "idh_resnet18_best.pth")
        print(f"  ✓ Best model saved! AUC: {auc:.4f}", flush=True)
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break


# ─── FINAL REPORT ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(classification_report(all_true, all_preds,
                             target_names=['Wild Type', 'Mutated']))
cm = confusion_matrix(all_true, all_preds)
tn, fp, fn, tp = cm.ravel()
print(f"Confusion Matrix:\n{cm}")
print(f"Best AUC    : {best_auc:.4f}")
print(f"Sensitivity : {tp/(tp+fn+1e-8):.4f}")
print(f"Specificity : {tn/(tn+fp+1e-8):.4f}")


# ─── PLOTS ────────────────────────────────────────────────────────────────────
epochs_ran = range(1, len(train_losses) + 1)
fig, axes  = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(epochs_ran, train_losses, label='Train Loss')
axes[0].plot(epochs_ran, val_losses,   label='Val Loss')
axes[0].set_title('Loss Curve')
axes[0].set_xlabel('Epoch')
axes[0].legend()

axes[1].plot(epochs_ran, val_aucs,     label='AUC',              color='blue')
axes[1].plot(epochs_ran, val_bal_accs, label='Balanced Accuracy', color='orange')
axes[1].set_title('AUC & Balanced Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].legend()

axes[2].plot(epochs_ran, val_f1s, label='F1 (Mutated)', color='green')
axes[2].set_title('F1 Score (Mutated Class)')
axes[2].set_xlabel('Epoch')
axes[2].legend()

plt.tight_layout()
plt.savefig("resnet18_training_curves.png", dpi=150)
plt.show()
print("Saved: resnet18_training_curves.png")

state = model.module.state_dict() if MULTI_GPU else model.state_dict()
torch.save(state, "idh_resnet18_final.pth")
print("Saved: idh_resnet18_final.pth")
print("Saved: idh_resnet18_best.pth")