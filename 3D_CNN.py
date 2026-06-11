"""
Simple 3D CNN for IDH Mutation Prediction
Dataset : UTSW-Glioma
Input   : 4-channel MRI (T1, T1CE, T2, FLAIR) resized to 48x48x48
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    roc_auc_score, f1_score, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR  =  "/workspace/UTSW_Glioma_data/UTSW-Glioma"
TSV_PATH  = "/workspace/UTSW_Glioma_Metadata-2-1 (2).tsv"
IMG_SIZE  = 48
BATCH     = 16
EPOCHS    = 30
LR        = 1e-4
PATIENCE  = 7        # early stopping patience
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ─── DATASET ──────────────────────────────────────────────────────────────────
class GliomaDataset(Dataset):
    def __init__(self, subjects, labels, data_dir, size=48):
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
        volume   = np.stack(channels, axis=0)          # (4, 48, 48, 48)
        return torch.tensor(volume), torch.tensor(self.labels[idx], dtype=torch.long)


# ─── MODEL ────────────────────────────────────────────────────────────────────
class Simple3DCNN(nn.Module):
    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16), nn.ReLU(), nn.MaxPool3d(2),   # 48→24

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d(2),   # 24→12

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(), nn.MaxPool3d(2),   # 12→6
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),   # works for any input size → (64,1,1,1)
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


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
print(f"Total subjects : {len(subjects)}")
print(f"IDH mutated    : {n_mut}  ({100*n_mut/len(labels):.1f}%)")
print(f"IDH wild type  : {n_wt}  ({100*n_wt/len(labels):.1f}%)")

X_train, X_val, y_train, y_val = train_test_split(
    subjects, labels, test_size=0.2, stratify=labels, random_state=42)

train_ds     = GliomaDataset(X_train, y_train, DATA_DIR, IMG_SIZE)
val_ds       = GliomaDataset(X_val,   y_val,   DATA_DIR, IMG_SIZE)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=8)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=2)


# ─── TRAIN SETUP ──────────────────────────────────────────────────────────────
model     = Simple3DCNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Class weights: penalise mutated class more (minority)
class_weights = torch.tensor([1.0, 2.5]).to(DEVICE)
criterion     = nn.CrossEntropyLoss(weight=class_weights)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.5, verbose=True)

# ─── TRACKING ─────────────────────────────────────────────────────────────────
train_losses, val_losses = [], []
val_accs, val_aucs, val_bal_accs, val_f1s = [], [], [], []

best_auc         = 0.0
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

    # ── Metrics ──────────────────────────────────────────────────────────────
    acc      = accuracy_score(all_true, all_preds)
    bal_acc  = balanced_accuracy_score(all_true, all_preds)
    auc      = roc_auc_score(all_true, all_probs)
    f1       = f1_score(all_true, all_preds, pos_label=1, zero_division=0)
    cm       = confusion_matrix(all_true, all_preds)

    # Sensitivity = TP/(TP+FN),  Specificity = TN/(TN+FP)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    val_accs.append(acc)
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

    # ── LR Scheduler ─────────────────────────────────────────────────────────
    scheduler.step(val_losses[-1])

    # ── Save best model (based on AUC) ───────────────────────────────────────
    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), "idh_3dcnn_best.pth")
        print(f"  ✓ Best model saved!  AUC: {auc:.4f}", flush=True)
        patience_counter = 0
    else:
        patience_counter += 1

    # ── Early Stopping ───────────────────────────────────────────────────────
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping triggered at epoch {epoch+1}")
        break

# ─── FINAL REPORT ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL RESULTS (last epoch)")
print("="*60)
print(classification_report(all_true, all_preds,
      target_names=['Wild Type', 'Mutated']))
print(f"Best AUC achieved : {best_auc:.4f}")
print(f"Confusion Matrix  :\n{confusion_matrix(all_true, all_preds)}")


# ─── PLOTS ────────────────────────────────────────────────────────────────────
epochs_ran = range(1, len(train_losses) + 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

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
plt.savefig("training_curves.png", dpi=150)
plt.show()
print("Saved: training_curves.png")

# ─── SAVE FINAL MODEL ─────────────────────────────────────────────────────────
torch.save(model.state_dict(), "idh_3dcnn_final.pth")
print("Saved: idh_3dcnn_final.pth")
print("Saved: idh_3dcnn_best.pth  (best AUC checkpoint)")
