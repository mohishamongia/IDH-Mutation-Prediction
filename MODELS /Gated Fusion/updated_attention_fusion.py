"""
Generalized N-Fold Training — ResNet-based Gated Modality Fusion  [v3]
Changes vs v2:
  1. GatedModalityFusion  — SE-style sigmoid gating (replaces unstable MultiheadAttention)
  2. FocalLoss            — replaces CrossEntropy+class_weights (fixes majority-class collapse)
  3. LR warmup (5 ep)     — LinearLR → CosineAnnealingLR via SequentialLR
  4. LR lowered to 2e-4   — was 5e-4, reduces early instability
  5. Augmentation + dual normalization kept from v2

Run : python idh_train_resnet_attn_v3.py
"""

import os, json, time
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# ─── CONFIGURABLE SECTION ──────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

GPU_IDS = [0]   # ✏️ edit per run

ALL_MODALITIES = {
    "T1"   : "brain_t1_ants.nii.gz",
    "T1CE" : "brain_t1ce_ants.nii.gz",
    "T2"   : "brain_t2_ants.nii.gz",
    "FLAIR": "brain_fl_ants.nii.gz",
}

ACTIVE_MODALITIES = ["T1", "T1CE", "T2", "FLAIR"]

N_FOLDS   = 5
RUN_FOLDS = [1, 2, 3, 4, 5]

SAVE_DIR   = "results_resnet_attn_v3"
IMG_SIZE   = 96
BATCH      = 8 * len(GPU_IDS)
EPOCHS     = 60
LR         = 2e-4        # ↓ lowered from 5e-4
WARMUP_EP  = 5           # epochs of linear warmup before cosine kicks in
PATIENCE   = 15
DROPOUT_P  = 0.3         # modality dropout probability (training only)

FEAT_DIM   = 256

# Focal loss hyperparams
FOCAL_ALPHA = 0.75       # down-weights easy negatives (WT majority class)
FOCAL_GAMMA = 2.0        # focusing parameter

TEST_SCENARIOS = {
    "All": list(range(len(ACTIVE_MODALITIES))),
}

USE_CACHE = True
CACHE_DIR = f"/workspace/cache_{IMG_SIZE}_{'_'.join(ACTIVE_MODALITIES)}"

# ═══════════════════════════════════════════════════════════════════════════════
# END CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in GPU_IDS)
DEVICE       = torch.device("cuda:0")
MULTI_GPU    = len(GPU_IDS) > 1
NUM_CHANNELS = len(ACTIVE_MODALITIES)

DATA_DIR = "/workspace/UTSW_Glioma_data/UTSW-Glioma"
TSV_PATH = "UTSW_Glioma_data/UTSW_Glioma_Metadata-2-1 (2).tsv"

os.makedirs(SAVE_DIR, exist_ok=True)
if USE_CACHE:
    os.makedirs(CACHE_DIR, exist_ok=True)

print(f"GPUs          : {GPU_IDS}  |  Multi-GPU: {MULTI_GPU}")
print(f"Modalities ({NUM_CHANNELS}): {ACTIVE_MODALITIES}")
print(f"N_FOLDS       : {N_FOLDS}  |  Running folds: {RUN_FOLDS}")
print(f"LR            : {LR}  |  Warmup: {WARMUP_EP} ep  |  Patience: {PATIENCE}")
print(f"FocalLoss     : alpha={FOCAL_ALPHA}  gamma={FOCAL_GAMMA}")
print(f"Results dir   : {SAVE_DIR}/\n")


# ═══════════════════════════════════════════════════════════════════════════════
# FOCAL LOSS  — fixes majority-class collapse
# ═══════════════════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    """
    Focal Loss: down-weights easy (well-classified) examples so the model
    is forced to focus on hard minority-class samples.
    alpha=0.75 → extra weight on positive (mutated) class
    gamma=2.0  → standard focusing exponent
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')  # (B,)
        pt      = torch.exp(-ce_loss)                                 # prob of correct class
        # alpha weighting: alpha for positives, (1-alpha) for negatives
        alpha_t = torch.where(targets == 1,
                              torch.tensor(self.alpha,       device=logits.device),
                              torch.tensor(1.0 - self.alpha, device=logits.device))
        focal   = alpha_t * (1.0 - pt) ** self.gamma * ce_loss
        return focal.mean()


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════
class GliomaDataset(Dataset):
    def __init__(self, subjects, labels, data_dir, modality_files, size=96,
                 dropout_p=0.0, missing=None, cache_dir=None, augment=False):
        self.subjects       = subjects
        self.labels         = labels
        self.data_dir       = data_dir
        self.modality_files = modality_files
        self.size           = size
        self.dropout_p      = dropout_p
        self.missing        = missing
        self.cache_dir      = cache_dir
        self.augment        = augment

    def load_volume(self, path):
        from scipy.ndimage import zoom
        vol     = nib.load(path).get_fdata(dtype=np.float32)
        factors = [self.size / s for s in vol.shape]
        vol     = zoom(vol, factors, order=1)

        # Step 1 — min-max to [0, 1]
        v_min, v_max = vol.min(), vol.max()
        vol = (vol - v_min) / (v_max - v_min + 1e-8)

        # Step 2 — foreground z-score (avoids background zeros skewing stats)
        mask = vol > 0.01
        if mask.sum() > 100:
            mu, sigma = vol[mask].mean(), vol[mask].std()
            vol[mask] = (vol[mask] - mu) / (sigma + 1e-8)
            vol = np.clip(vol, -5.0, 5.0)

        return vol

    def _augment(self, volume):
        """volume: (C, D, H, W) — applied during training only"""
        # Random flips along each spatial axis
        for axis in [1, 2, 3]:
            if np.random.rand() > 0.5:
                volume = np.flip(volume, axis=axis).copy()
        # Per-channel intensity jitter
        for c in range(volume.shape[0]):
            if volume[c].max() > 0:
                volume[c] += np.random.uniform(-0.05, 0.05)
                volume[c] *= np.random.uniform(0.95, 1.05)
                volume[c]  = np.clip(volume[c], -5.0, 5.0)
        return volume

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
                channels = [self.load_volume(os.path.join(folder, f))
                            for f in self.modality_files]
                volume   = np.stack(channels, axis=0).astype(np.float32)
                np.save(cache_path, volume)
        else:
            folder   = os.path.join(self.data_dir, sid)
            channels = [self.load_volume(os.path.join(folder, f))
                        for f in self.modality_files]
            volume   = np.stack(channels, axis=0)

        volume = volume.copy()
        n_ch   = volume.shape[0]

        # Modality dropout (training)
        if self.dropout_p > 0:
            for c in range(n_ch):
                if np.random.rand() < self.dropout_p:
                    volume[c] = 0.0

        # Spatial + intensity augmentation (training only)
        if self.augment:
            volume = self._augment(volume)

        # Test scenario: zero out unavailable channels
        if self.missing is not None:
            for c in range(n_ch):
                if c not in self.missing:
                    volume[c] = 0.0

        return torch.tensor(volume, dtype=torch.float32), \
               torch.tensor(self.labels[idx], dtype=torch.long)


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
    """Single-channel ResNet18-3D encoder — one per modality."""
    def __init__(self, feat_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1))
        self.layer1 = nn.Sequential(ResBlock3D(32,  32),                 ResBlock3D(32,  32))
        self.layer2 = nn.Sequential(ResBlock3D(32,  64,  stride=2),      ResBlock3D(64,  64))
        self.layer3 = nn.Sequential(ResBlock3D(64,  128, stride=2),      ResBlock3D(128, 128))
        self.layer4 = nn.Sequential(ResBlock3D(128, feat_dim, stride=2), ResBlock3D(feat_dim, feat_dim))
        self.pool   = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten())

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.pool(x)   # (B, feat_dim)


# ─── FIX 1: Gated Modality Fusion — stable SE-style gating ────────────────────
class GatedModalityFusion(nn.Module):
    """
    SE-style sigmoid gating across modalities.

    Why better than v1 softmax:
      - Sigmoid (not softmax) → gates are independent, can suppress any
        modality to ~0 without forcing others to compensate
      - Learned projection + LayerNorm after fusion stabilises gradients
      - Far fewer parameters than MultiheadAttention → stable on small datasets

    forward() returns (fused, gates) so gates can be logged for interpretability.
    """
    def __init__(self, feat_dim, num_modalities):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * num_modalities, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_modalities),
            nn.Sigmoid()            # independent gate per modality [0, 1]
        )
        self.proj = nn.Linear(feat_dim, feat_dim)
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, feats):
        """feats: list of (B, feat_dim) tensors — one per modality"""
        concat  = torch.cat(feats, dim=1)           # (B, feat_dim * num_mod)
        gates   = self.gate(concat)                  # (B, num_mod)
        feats_s = torch.stack(feats, dim=1)          # (B, num_mod, feat_dim)
        gated   = feats_s * gates.unsqueeze(2)       # broadcast gate per modality
        fused   = gated.sum(dim=1)                   # (B, feat_dim)
        return self.norm(self.proj(fused)), gates


class ResNetGatedFusionNet(nn.Module):
    """
    ResNet18-3D encoders + Gated Modality Fusion + wider classifier.
    """
    def __init__(self, num_modalities, feat_dim=256, num_classes=2, dropout=0.4):
        super().__init__()
        self.num_modalities = num_modalities
        self.feat_dim       = feat_dim

        self.encoders    = nn.ModuleList(
            [ResNetModalityEncoder(feat_dim) for _ in range(num_modalities)])
        self.fusion      = GatedModalityFusion(feat_dim, num_modalities)

        # Wider classifier: feat_dim → 256 → 64 → 2
        self.classifier  = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256,      64),  nn.ReLU(), nn.Dropout(dropout * 0.75),
            nn.Linear(64, num_classes))

    def forward(self, x, return_weights=False):
        feats          = [self.encoders[i](x[:, i:i+1]) for i in range(self.num_modalities)]
        fused, gates   = self.fusion(feats)
        out            = self.classifier(fused)
        if return_weights:
            return out, gates
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN ONE FOLD
# ═══════════════════════════════════════════════════════════════════════════════
def train_one_fold(model_fn, X_train, y_train, X_val, y_val, modality_files):
    model = model_fn()
    if MULTI_GPU:
        model = nn.DataParallel(model)
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    # FIX 2: Focal loss — prevents collapse to majority class
    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)

    # FIX 3: Warmup → Cosine annealing
    warmup    = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                         total_iters=WARMUP_EP)
    cosine    = CosineAnnealingLR(optimizer, T_max=max(EPOCHS - WARMUP_EP, 1),
                                  eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                             milestones=[WARMUP_EP])

    train_ds = GliomaDataset(X_train, y_train, DATA_DIR, modality_files, IMG_SIZE,
                              dropout_p=DROPOUT_P, augment=True,
                              cache_dir=CACHE_DIR if USE_CACHE else None)
    val_ds   = GliomaDataset(X_val,   y_val,   DATA_DIR, modality_files, IMG_SIZE,
                              augment=False,
                              cache_dir=CACHE_DIR if USE_CACHE else None)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                               num_workers=2, pin_memory=True,
                               persistent_workers=True, prefetch_factor=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                               num_workers=8, pin_memory=True,
                               persistent_workers=True, prefetch_factor=4)

    best_auc, patience_counter = 0.0, 0
    best_state = None
    history = {"train_loss": [], "val_loss": [],
               "auc": [], "bal_acc": [], "f1": [],
               "sensitivity": [], "specificity": [], "lr": []}

    for epoch in range(EPOCHS):
        # ── Train ──────────────────────────────────────────────────────────────
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
        current_lr = scheduler.get_last_lr()[0]
        history["lr"].append(float(current_lr))

        # ── Validate ───────────────────────────────────────────────────────────
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

        print(f"  Ep {epoch+1:02d} | LR: {current_lr:.2e} | "
              f"Loss: {history['train_loss'][-1]:.4f}/{history['val_loss'][-1]:.4f} | "
              f"AUC: {auc:.4f} | BalAcc: {bal_acc:.4f} | "
              f"Sens: {sens:.4f} | Spec: {spec:.4f}", flush=True)

        scheduler.step()

        if auc > best_auc:
            best_auc   = auc
            best_state = {k: v.cpu() for k, v in
                          (model.module.state_dict() if MULTI_GPU
                           else model.state_dict()).items()}
            patience_counter = 0
            print(f"  ✓ New best AUC: {auc:.4f}", flush=True)
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  ⏹ Early stopping at epoch {epoch+1} "
                  f"(no improvement for {PATIENCE} epochs)")
            break

    return model, best_state, history, best_auc


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATE SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_scenarios(model, subjects, labels, modality_files):
    model.eval()
    scenario_results = {}

    for sc_name, avail_ch in TEST_SCENARIOS.items():
        ds = GliomaDataset(subjects, labels, DATA_DIR, modality_files, IMG_SIZE,
                           missing=avail_ch, augment=False,
                           cache_dir=CACHE_DIR if USE_CACHE else None)
        loader = DataLoader(ds, batch_size=BATCH, shuffle=False,
                            num_workers=8, pin_memory=True)

        all_probs, all_preds, all_true = [], [], []
        all_gate_weights = []

        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(DEVICE)
                logits, gates = model(imgs, return_weights=True)
                all_gate_weights.extend(gates.cpu().numpy().tolist())
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

        # Mean gate values per modality across all subjects
        mean_gates = np.mean(all_gate_weights, axis=0).tolist()
        print(f"    Mean modality gates: "
              + " | ".join(f"{ACTIVE_MODALITIES[i]}: {mean_gates[i]:.3f}"
                           for i in range(NUM_CHANNELS)))

        scenario_results[sc_name] = {
            "metrics": {
                "AUC"        : round(auc,     4),
                "BalAcc"     : round(bal_acc, 4),
                "Sensitivity": round(sens,    4),
                "Specificity": round(spec,    4),
                "F1"         : round(f1,      4),
            },
            "confusion_matrix": cm.tolist(),
            "mean_modality_gates": {ACTIVE_MODALITIES[i]: round(mean_gates[i], 4)
                                    for i in range(NUM_CHANNELS)},
            "roc_curve": {
                "fpr"       : fpr.tolist(),
                "tpr"       : tpr.tolist(),
                "thresholds": thresholds.tolist()
            },
            "per_subject": [
                {
                    "subject_id"  : subjects[i],
                    "true_label"  : int(all_true[i]),
                    "pred_label"  : int(all_preds[i]),
                    "pred_prob"   : round(float(all_probs[i]), 4),
                    "correct"     : bool(all_true[i] == all_preds[i]),
                    "gate_weights": all_gate_weights[i],
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
n_wt  = len(labels) - n_mut
print(f"Total: {len(subjects)} | Mutated: {n_mut} ({100*n_mut/len(labels):.1f}%) "
      f"| Wild type: {n_wt} ({100*n_wt/len(labels):.1f}%)\n")

dataset_info = {
    "total"            : len(subjects),
    "mutated"          : n_mut,
    "wild_type"        : n_wt,
    "scanner_info"     : scanner_info,
    "grade_info"       : {k: int(v) if pd.notna(v) else None
                          for k, v in grade_info.items()},
    "active_modalities": ACTIVE_MODALITIES,
}
with open(f"{SAVE_DIR}/dataset_info.json", "w") as f:
    json.dump(dataset_info, f, indent=2)
print(f"Saved: {SAVE_DIR}/dataset_info.json")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN SELECTED FOLDS
# ═══════════════════════════════════════════════════════════════════════════════
skf         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
subjects_np = np.array(subjects)
labels_np   = np.array(labels)

MODEL_NAME     = "Gated_Fusion_ResNet"
MODALITY_FILES = [ALL_MODALITIES[m] for m in ACTIVE_MODALITIES]

splits = list(skf.split(subjects_np, labels_np))

for rf in RUN_FOLDS:
    if not (1 <= rf <= N_FOLDS):
        raise ValueError(f"RUN_FOLDS contains {rf}, must be in 1..{N_FOLDS}")

all_fold_results = []
overall_start    = time.time()

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

    # Class balance per fold (useful to watch)
    n_mut_tr = sum(y_train); n_mut_vl = sum(y_val)
    print(f"Train mut/wt: {n_mut_tr}/{len(y_train)-n_mut_tr} | "
          f"Val mut/wt: {n_mut_vl}/{len(y_val)-n_mut_vl}")

    save_path = f"{SAVE_DIR}/fold{fold}_{MODEL_NAME}.json"
    if os.path.exists(save_path):
        print(f"\n  [{MODEL_NAME}] Already done — skipping (found {save_path})")
        with open(save_path) as f:
            all_fold_results.append(json.load(f))
        continue

    print(f"\n  ── {MODEL_NAME} ──")
    fold_start = time.time()

    model, best_state, history, best_auc = train_one_fold(
        lambda: ResNetGatedFusionNet(num_modalities=NUM_CHANNELS, feat_dim=FEAT_DIM),
        X_train, y_train, X_val, y_val, MODALITY_FILES)

    core = model.module if MULTI_GPU else model
    core.load_state_dict(best_state)

    print(f"\n  Evaluating scenarios...")
    scenario_results = evaluate_scenarios(model, X_val, y_val, MODALITY_FILES)

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
            "warmup_ep"         : WARMUP_EP,
            "patience"          : PATIENCE,
            "dropout_p"         : DROPOUT_P,
            "focal_alpha"       : FOCAL_ALPHA,
            "focal_gamma"       : FOCAL_GAMMA,
            "gpu_ids"           : GPU_IDS,
            "feat_dim"          : FEAT_DIM,
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
print(f"All {len(RUN_FOLDS)} fold(s) complete in {overall_elapsed:.1f} minutes")
print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\nSUMMARY — Mean ± Std AUC across run folds {RUN_FOLDS}\n")

summary    = {MODEL_NAME: {}, "run_folds": RUN_FOLDS, "active_modalities": ACTIVE_MODALITIES}
folds_data = []
for fold in RUN_FOLDS:
    fp = f"{SAVE_DIR}/fold{fold}_{MODEL_NAME}.json"
    if os.path.exists(fp):
        with open(fp) as f:
            folds_data.append(json.load(f))

for sc_name in TEST_SCENARIOS:
    aucs  = [fd["scenario_results"][sc_name]["metrics"]["AUC"]
              for fd in folds_data if sc_name in fd["scenario_results"]]
    senss = [fd["scenario_results"][sc_name]["metrics"]["Sensitivity"]
              for fd in folds_data if sc_name in fd["scenario_results"]]
    specs = [fd["scenario_results"][sc_name]["metrics"]["Specificity"]
              for fd in folds_data if sc_name in fd["scenario_results"]]
    if not aucs:
        continue
    summary[MODEL_NAME][sc_name] = {
        "auc_mean" : round(float(np.mean(aucs)),  4),
        "auc_std"  : round(float(np.std(aucs)),   4),
        "sens_mean": round(float(np.mean(senss)), 4),
        "spec_mean": round(float(np.mean(specs)), 4),
    }
    print(f"  {MODEL_NAME:30s} | {sc_name:10s} | "
          f"AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f} | "
          f"Sens: {np.mean(senss):.4f} | Spec: {np.mean(specs):.4f}")

with open(f"{SAVE_DIR}/summary_folds_{'_'.join(map(str, RUN_FOLDS))}.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {SAVE_DIR}/summary_folds_{'_'.join(map(str, RUN_FOLDS))}.json")
print("\n✓ Run complete!")