import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR  =  "/workspace/UTSW_Glioma_data/UTSW-Glioma"
TSV_PATH  = "/workspace/UTSW_Glioma_Metadata-2-1 (2).tsv"
IMG_SIZE  = 48        # resize all volumes to 64³
BATCH     = 16
EPOCHS    = 30
LR        = 1e-4
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── DATASET ──────────────────────────────────────────────────────────────────
class GliomaDataset(Dataset):
    def __init__(self, subjects, labels, data_dir, size=64):  
        self.subjects = subjects
        self.labels   = labels
        self.data_dir = data_dir
        self.size     = size

    def load_volume(self, path):
        vol = nib.load(path).get_fdata(dtype=np.float32)
        # simple resize via slicing/zoom to size³
        from scipy.ndimage import zoom
        factors = [self.size / s for s in vol.shape]
        vol = zoom(vol, factors, order=1)
        # normalize to [0,1]
        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
        return vol

    def __len__(self):  # FIXED: was _len_
        return len(self.subjects)

    def __getitem__(self, idx):  # FIXED: was _getitem_
        sid = self.subjects[idx]
        folder = os.path.join(self.data_dir, sid)

        # 4 channels: T1, T1CE, T2, FLAIR (all _ants = co-registered)
        files = [
            "brain_t1_ants.nii.gz",
            "brain_t1ce_ants.nii.gz",
            "brain_t2_ants.nii.gz",
            "brain_fl_ants.nii.gz",
        ]
        channels = [self.load_volume(os.path.join(folder, f)) for f in files]
        volume = np.stack(channels, axis=0)               # (4, 64, 64, 64)
        return torch.tensor(volume), torch.tensor(self.labels[idx], dtype=torch.long)


# ─── MODEL ────────────────────────────────────────────────────────────────────
class Simple3DCNN(nn.Module):
    def __init__(self, in_channels=4, num_classes=2):  # FIXED: was _init_
        super().__init__()  # FIXED: was super()._init_()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16), nn.ReLU(), nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(), nn.MaxPool3d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),   # ← always outputs 64x1x1x1
            nn.Flatten(),
            nn.Linear(64, 128),        # ← always 64, regardless of input size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─── DATA PREP ────────────────────────────────────────────────────────────────
df = pd.read_csv(TSV_PATH, sep='\t')

# Keep only subjects with IDH label and available images
df = df[df['IDH'].isin(['mutated', 'wild type'])]
df = df[df['T1'] == 'Available']

# Encode label: mutated=1, wild type=0
df['label'] = (df['IDH'] == 'mutated').astype(int)

# Keep only subjects that exist on disk
df = df[df['Subject ID'].apply(lambda s: os.path.isdir(os.path.join(DATA_DIR, s)))]

subjects = df['Subject ID'].tolist()
labels   = df['label'].tolist()

print(f"Total subjects: {len(subjects)}")
print(f"IDH mutated: {sum(labels)}  |  wild type: {len(labels)-sum(labels)}")

# Train / Val split (80/20, stratified)
X_train, X_val, y_train, y_val = train_test_split(
    subjects, labels, test_size=0.2, stratify=labels, random_state=42
)

train_ds = GliomaDataset(X_train, y_train, DATA_DIR, IMG_SIZE)
val_ds   = GliomaDataset(X_val,   y_val,   DATA_DIR, IMG_SIZE)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=2)


# ─── TRAIN ────────────────────────────────────────────────────────────────────
model     = Simple3DCNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# FIXED: correct class weight formula — total / (n_classes * count)
class_counts  = [446, 176]  # wild type, mutated
class_weights = torch.tensor(
    [622 / (2 * 446), 622 / (2 * 176)],
    dtype=torch.float32
).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)

train_losses, val_losses, val_accs = [], [], []

for epoch in range(EPOCHS):
    # --- training ---
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

    # --- validation ---
    model.eval()
    all_preds, all_probs, all_true = [], [], []
    batch_losses = []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            logits = model(imgs)
            loss   = criterion(logits, lbls)
            batch_losses.append(loss.item())
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_true.extend(lbls.cpu().numpy())

    val_losses.append(np.mean(batch_losses))
    acc = accuracy_score(all_true, all_preds)
    val_accs.append(acc)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Train Loss: {train_losses[-1]:.4f} | "
          f"Val Loss: {val_losses[-1]:.4f} | "
          f"Val Acc: {acc:.4f}")

# AUC on final epoch
auc = roc_auc_score(all_true, all_probs)
print(f"\nFinal Val AUC: {auc:.4f}")


# ─── PLOT ─────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, label='Train Loss')
ax1.plot(val_losses,   label='Val Loss')
ax1.set_title('Loss Curve')
ax1.set_xlabel('Epoch')
ax1.legend()

ax2.plot(val_accs, color='green', label='Val Accuracy')
ax2.set_title('Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()
print("Saved: training_curves.png")


# ─── SAVE MODEL ───────────────────────────────────────────────────────────────
torch.save(model.state_dict(), "idh_3dcnn.pth")
print("Saved: idh_3dcnn.pth")