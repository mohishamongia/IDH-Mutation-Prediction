"""
5-Fold Cross-Validation — Missing-Modality Robust IDH Prediction
Models : Baseline_ResNet18 | Dropout30_ResNet18 | Dropout50_ResNet18
Run    : python idh_cv_train.py

"""

# ── CUDA env var MUST precede 'import torch' ──────────────────────────────────
import os

GPU_IDS = [0]   # ✏️ edit before every run
# CVD remaps physical GPUs to logical cuda:0…cuda:N-1.
# Always use cuda:0 as primary; never reference the raw physical index.
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in GPU_IDS)

# ── Standard imports (safe after env var is set) ──────────────────────────────
import json, time, random
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
from scipy import stats as scipy_stats
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, roc_curve
)
import warnings
warnings.filterwarnings("ignore")

# ─── REPRODUCIBILITY ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ─── GPU ──────────────────────────────────────────────────────────────────────
DEVICE    = torch.device("cuda:0")
MULTI_GPU = len(GPU_IDS) > 1
print(f"Physical GPUs : {GPU_IDS}")
print(f"Logical range : cuda:0 … cuda:{len(GPU_IDS)-1}  |  DataParallel: {MULTI_GPU}")

# ─── PATHS & CONFIG ───────────────────────────────────────────────────────────
DATA_DIR  = "/workspace/UTSW_Glioma_data/UTSW-Glioma"
TSV_PATH  = "UTSW_Glioma_data/UTSW_Glioma_Metadata-2-1 (2).tsv"
SAVE_DIR  = "results_final"
CACHE_DIR = "cache_npy"   # Opt-1: preprocessed volume cache
IMG_SIZE  = 96
BATCH     = 24            # lowered from 48; ~26 updates/epoch on 622 subjects
EPOCHS    = 100
LR        = 1e-4
PATIENCE  = 7
N_FOLDS   = 5

TEST_SCENARIOS = {
    "All_4"         : [0, 1, 2, 3],
    "T1CE_missing"  : [0,    2, 3],
    "FLAIR_missing" : [0, 1, 2   ],
    "T2_missing"    : [0, 1,    3],
    "T1_T2_only"    : [0,    2   ],
    "T1_only"       : [0         ],
}
SCENARIO_KEYS = list(TEST_SCENARIOS.keys())

os.makedirs(SAVE_DIR,  exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
print(f"\nSAVE_DIR={SAVE_DIR}  CACHE_DIR={CACHE_DIR}")
print(f"Batch={BATCH}  IMG={IMG_SIZE}³  Folds={N_FOLDS}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def safe_auc(y_true, y_prob):
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")

def safe_cm(y_true, y_pred):
    """Always returns 2×2; safe even when fold predicts single class."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1])

def make_loader(dataset, shuffle, batch=BATCH):
    g = torch.Generator()
    g.manual_seed(SEED)
    return DataLoader(
        dataset, batch_size=batch, shuffle=shuffle,
        num_workers=4, pin_memory=True,
        worker_init_fn=seed_worker, generator=g
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Opt-1 — PREPROCESSING CACHE
# ═══════════════════════════════════════════════════════════════════════════════
def preprocess_and_cache(subjects, data_dir, cache_dir, size=96):
    """
    Zoom + normalise each volume once; save as float16 .npy.
    Subsequent epochs load from disk in microseconds.
    Returns: dict {subject_id: cache_path}
    """
    from scipy.ndimage import zoom

    cache_map = {}
    skipped = 0

    for sid in tqdm(subjects, desc="Building cache"):
        out_path = os.path.join(cache_dir, f"{sid}.npy")
        cache_map[sid] = out_path
        if os.path.exists(out_path):
            skipped += 1
            continue

        folder = os.path.join(data_dir, sid)
        files  = [
            "brain_t1_ants.nii.gz",
            "brain_t1ce_ants.nii.gz",
            "brain_t2_ants.nii.gz",
            "brain_fl_ants.nii.gz",
        ]
        channels = []
        for fname in files:
            vol     = nib.load(os.path.join(folder, fname)).get_fdata(dtype=np.float32)
            factors = [size / s for s in vol.shape]
            vol     = zoom(vol, factors, order=1)
            vol     = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
            channels.append(vol)

        volume = np.stack(channels, axis=0).astype(np.float16)  # (4, 96, 96, 96)
        np.save(out_path, volume)

    print(f"Cache ready: {len(subjects)} subjects  "
          f"({skipped} already cached, {len(subjects)-skipped} newly written)")
    return cache_map


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET  (loads from cache; applies dropout / missing-modality masking)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Scenario sampling weights for structured dropout (Problem 1 fix) ──────────
# dropout_p is NOT used in structured mode — instead, the scenario distribution
# itself controls how aggressively modalities are dropped during training.
#
# Dropout30 → mild robustness: model still sees all 4 modalities ~40% of the time
# Dropout50 → aggressive robustness: hard single-modality cases dominate
#
# Weights correspond to TEST_SCENARIOS keys in order:
#   All_4 | T1CE_missing | FLAIR_missing | T2_missing | T1_T2_only | T1_only
STRUCTURED_WEIGHTS = {
    "Dropout30_ResNet18": [0.40, 0.15, 0.15, 0.15, 0.10, 0.05],
    "Dropout50_ResNet18": [0.10, 0.20, 0.20, 0.20, 0.15, 0.15],
}
# ─────────────────────────────────────────────────────────────────────────────

class GliomaDataset(Dataset):
    """
    dropout_mode:
      'random'     — each channel independently dropped with prob dropout_p
                     (dropout_p IS used here)
      'structured' — sample a test scenario according to model-specific weights
                     in STRUCTURED_WEIGHTS (dropout_p ignored — distribution
                     controls aggressiveness; model_name must be supplied)
      None / 0.0   — no dropout (validation / test)
    missing: list of channel indices to KEEP (test-time explicit scenario)
    """
    def __init__(self, subjects, labels, cache_map,
                 dropout_p=0.0, dropout_mode="random",
                 missing=None, model_name=None):
        self.subjects     = subjects
        self.labels       = labels
        self.cache_map    = cache_map
        self.dropout_p    = dropout_p
        self.dropout_mode = dropout_mode
        self.missing      = missing
        self._scenario_lists   = list(TEST_SCENARIOS.values())
        # Resolve per-model weights; fall back to uniform if name not found
        raw_w = STRUCTURED_WEIGHTS.get(model_name, [1/6]*6)
        total = sum(raw_w)
        self._scenario_weights = [w / total for w in raw_w]

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        sid    = self.subjects[idx]
        volume = np.load(self.cache_map[sid]).astype(np.float32).copy()  # (4,96,96,96)

        if self.dropout_p > 0 and self.dropout_mode == "random":
            # Independent per-channel dropout with all-zero guardrail
            mask = np.random.rand(4) < self.dropout_p
            if mask.all():
                mask[np.random.randint(4)] = False
            volume[mask] = 0.0

        elif self.dropout_mode == "structured":
            # Sample scenario according to model-specific weights.
            # dropout_p is intentionally NOT used here — aggressiveness is
            # controlled by STRUCTURED_WEIGHTS, making Dropout30 vs Dropout50
            # meaningfully different training distributions.
            avail = random.choices(self._scenario_lists,
                                   weights=self._scenario_weights, k=1)[0]
            for c in range(4):
                if c not in avail:
                    volume[c] = 0.0

        if self.missing is not None:
            for c in range(4):
                if c not in self.missing:
                    volume[c] = 0.0

        return torch.tensor(volume), torch.tensor(self.labels[idx], dtype=torch.long)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL  (Fix-S2: single forward pass returns logits + optional embedding)
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
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.skip(x))


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
        """Standard forward — logits only. DataParallel-safe (no extra args)."""
        x   = self.stem(x)
        x   = self.layer1(x); x = self.layer2(x)
        x   = self.layer3(x); x = self.layer4(x)
        emb = self.pool(x).flatten(1)          # (B, 512)
        return self.fc(self.drop(emb))

    def embed(self, x):
        """
        512-d pre-dropout embedding.  Always call on the UNWRAPPED core model
        (model.module when DataParallel), never on the wrapper itself — avoids
        the kwargs-positional-arg crash that triggered this fix.
        Single-GPU call during eval; batch=24 makes this fast enough.
        """
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.pool(x).flatten(1)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN ONE FOLD
# ═══════════════════════════════════════════════════════════════════════════════
def train_one_fold(model, train_loader, val_loader, class_weights_tensor):
    if MULTI_GPU:
        model = nn.DataParallel(model)
    model = model.to(DEVICE)

    core = model.module if MULTI_GPU else model

    # Bug-4 fix: initialise best_state before training loop
    best_state = {k: v.cpu().clone() for k, v in core.state_dict().items()}

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(DEVICE))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", patience=3, factor=0.5)

    best_auc, patience_ctr = 0.0, 0
    history = {k: [] for k in ["train_loss","val_loss","auc",
                                "bal_acc","f1","sensitivity","specificity"]}

    for epoch in range(EPOCHS):
        # --- train ---
        model.train()
        batch_losses = []
        for imgs, lbls in tqdm(train_loader, desc=f"  Ep{epoch+1:03d}", leave=False):
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(loss.item())
        history["train_loss"].append(float(np.mean(batch_losses)))

        # --- validate ---
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
        auc     = safe_auc(all_true, all_probs)
        bal_acc = float(balanced_accuracy_score(all_true, all_preds))
        f1      = float(f1_score(all_true, all_preds, pos_label=1, zero_division=0))
        tn, fp, fn, tp = safe_cm(all_true, all_preds).ravel()
        sens = float(tp / (tp + fn + 1e-8))
        spec = float(tn / (tn + fp + 1e-8))

        for k, v in zip(["auc","bal_acc","f1","sensitivity","specificity"],
                        [auc, bal_acc, f1, sens, spec]):
            history[k].append(v)

        auc_s = f"{auc:.4f}" if not np.isnan(auc) else "NaN "
        print(f"  Ep{epoch+1:03d} | L {history['train_loss'][-1]:.4f}/"
              f"{history['val_loss'][-1]:.4f} | AUC {auc_s} | "
              f"BA {bal_acc:.4f} | Se {sens:.4f} | Sp {spec:.4f}", flush=True)

        scheduler.step(history["val_loss"][-1])

        if not np.isnan(auc) and auc > best_auc:
            best_auc   = auc
            best_state = {k: v.cpu().clone() for k, v in core.state_dict().items()}
            patience_ctr = 0
            print(f"  ✓ Best AUC {auc:.4f}", flush=True)
        else:
            patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"  Early stop @ epoch {epoch+1}")
            break

    return model, best_state, history, best_auc


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATE — all scenarios  |  Fix-S1: embeddings → .npy  |  Fix-S2: one pass
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_scenarios(model, subjects, labels, cache_map, fold, model_name):
    """
    Returns scenario_results dict (JSON-safe, no embeddings inline).
    Embeddings saved as:  results/embeddings/fold{F}_{model}_{scenario}.npy
    Shape: (n_subjects, 512), dtype float32

    Two-pass design (DataParallel fix):
      Pass 1 — model(imgs)       : logits via DataParallel across all GPUs
      Pass 2 — core.embed(imgs)  : 512-d embeddings on primary GPU only
    Passing keyword args through DataParallel is unreliable across PyTorch
    versions; two explicit passes are simpler and crash-free.
    """
    model.eval()
    core    = model.module if MULTI_GPU else model   # unwrapped for embed()
    emb_dir = os.path.join(SAVE_DIR, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)

    scenario_results = {}

    for sc_name, avail_ch in TEST_SCENARIOS.items():
        ds     = GliomaDataset(subjects, labels, cache_map, missing=avail_ch)
        loader = make_loader(ds, shuffle=False)

        all_probs, all_preds, all_true = [], [], []
        all_embeds = []

        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(DEVICE)
                # Pass 1: logits — DataParallel distributes across all GPUs
                logits = model(imgs)
                all_probs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_true.extend(lbls.numpy())
                # Pass 2: embeddings — core model on primary GPU, no DP wrapper
                emb = core.embed(imgs)
                all_embeds.append(emb.cpu().numpy().astype(np.float32))

        # Fix-S1: save embeddings as compressed .npy — NOT in JSON
        emb_array  = np.concatenate(all_embeds, axis=0)   # (N, 512)
        emb_path   = os.path.join(emb_dir,
                        f"fold{fold}_{model_name}_{sc_name}.npy")
        np.save(emb_path, emb_array)

        auc     = safe_auc(all_true, all_probs)
        bal_acc = float(balanced_accuracy_score(all_true, all_preds))
        f1      = float(f1_score(all_true, all_preds, pos_label=1, zero_division=0))
        tn, fp, fn, tp = safe_cm(all_true, all_preds).ravel()
        sens = float(tp / (tp + fn + 1e-8))
        spec = float(tn / (tn + fp + 1e-8))

        if not np.isnan(auc):
            fpr, tpr, thr = roc_curve(all_true, all_probs)
            roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                        "thresholds": thr.tolist()}
        else:
            roc_data = {"fpr": [], "tpr": [], "thresholds": []}

        scenario_results[sc_name] = {
            "metrics": {
                "AUC"        : round(auc,     4) if not np.isnan(auc) else None,
                "BalAcc"     : round(bal_acc, 4),
                "Sensitivity": round(sens,    4),
                "Specificity": round(spec,    4),
                "F1"         : round(f1,      4),
            },
            "confusion_matrix": safe_cm(all_true, all_preds).tolist(),
            "roc_curve"       : roc_data,
            "embedding_path"  : emb_path,     # pointer only — no data in JSON
            "per_subject": [
                {
                    "subject_id": subjects[i],
                    "true_label": int(all_true[i]),
                    "pred_label": int(all_preds[i]),
                    "pred_prob" : round(float(all_probs[i]), 4),
                    "correct"   : bool(int(all_true[i]) == int(all_preds[i])),
                }
                for i in range(len(subjects))
            ]
        }

        auc_s = f"{auc:.4f}" if not np.isnan(auc) else "NaN "
        print(f"    {sc_name:20s} | AUC {auc_s} | "
              f"Se {sens:.4f} | Sp {spec:.4f} | emb→{os.path.basename(emb_path)}",
              flush=True)

    return scenario_results


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PREP
# ═══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(TSV_PATH, sep="\t")
df = df[df["IDH"].isin(["mutated", "wild type"])]
df = df[df["T1"] == "Available"]
df["label"] = (df["IDH"] == "mutated").astype(int)
df = df[df["Subject ID"].apply(
    lambda s: os.path.isdir(os.path.join(DATA_DIR, s)))]
df["scanner"] = df["Scanner Make"] + "_" + df["Scanner Strength"].astype(str) + "T"

subjects     = df["Subject ID"].tolist()
labels       = df["label"].tolist()
scanner_info = dict(zip(df["Subject ID"], df["scanner"]))
grade_info   = dict(zip(df["Subject ID"], df["Tumor Grade"]))

n_mut = sum(labels); n_wt = len(labels) - n_mut
print(f"Subjects: {len(subjects)} | Mutated: {n_mut} ({100*n_mut/len(labels):.1f}%) "
      f"| WT: {n_wt} ({100*n_wt/len(labels):.1f}%)\n")

with open(f"{SAVE_DIR}/dataset_info.json", "w") as f:
    json.dump({
        "total": len(subjects), "mutated": n_mut, "wild_type": n_wt,
        "scanner_info": scanner_info,
        "grade_info": {k: int(v) if pd.notna(v) else None
                       for k, v in grade_info.items()}
    }, f, indent=2)
print(f"Saved: {SAVE_DIR}/dataset_info.json")

# Opt-1: build preprocessing cache once (skips already-cached subjects)
cache_map = preprocess_and_cache(subjects, DATA_DIR, CACHE_DIR, IMG_SIZE)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL CONFIGS
# ═══════════════════════════════════════════════════════════════════════════════
# dropout_mode: 'structured' (Opt-2) for robust models; None for baseline
MODEL_CONFIGS = [
    # (name,                  model_factory,         dropout_p, dropout_mode)
    ("Baseline_ResNet18",  lambda: ResNet18_3D(), 0.0, None),
    ("Dropout30_ResNet18", lambda: ResNet18_3D(), 0.3, "structured"),
    ("Dropout50_ResNet18", lambda: ResNet18_3D(), 0.5, "structured"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 5-FOLD CV  +  OOF accumulation  (Opt-3)
# ═══════════════════════════════════════════════════════════════════════════════
skf         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
subjects_np = np.array(subjects)
labels_np   = np.array(labels)

# OOF store: model_name → list of dicts
oof_store = {name: [] for name, *_ in MODEL_CONFIGS}

start_time = time.time()

for fold, (train_idx, val_idx) in enumerate(skf.split(subjects_np, labels_np)):
    fold_n = fold + 1
    print(f"\n{'#'*62}\nFOLD {fold_n}/{N_FOLDS}\n{'#'*62}")

    X_train = subjects_np[train_idx].tolist()
    y_train = labels_np[train_idx].tolist()
    X_val   = subjects_np[val_idx].tolist()
    y_val   = labels_np[val_idx].tolist()
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")

    # Fold-specific class weights
    y_tr       = np.array(y_train)
    n_wt_f     = int(np.sum(y_tr == 0))
    n_mut_f    = int(np.sum(y_tr == 1))
    w_mut      = n_wt_f / (n_mut_f + 1e-8)
    fold_w     = torch.tensor([1.0, w_mut], dtype=torch.float32)
    print(f"Weights → WT: 1.00 | Mut: {w_mut:.3f}  (n_wt={n_wt_f}, n_mut={n_mut_f})")

    for model_name, model_fn, dropout_p, dropout_mode in MODEL_CONFIGS:

        save_path = f"{SAVE_DIR}/fold{fold_n}_{model_name}.json"
        if os.path.exists(save_path):
            print(f"\n  [{model_name}] Skipping — already saved")
            # Still reload OOF predictions for stats at end
            with open(save_path) as f_in:
                fd = json.load(f_in)
            sc0 = fd["scenario_results"]["All_4"]["per_subject"]
            for row in sc0:
                oof_store[model_name].append({
                    "subject_id": row["subject_id"],
                    "true_label": row["true_label"],
                    "pred_prob" : row["pred_prob"],
                    "fold"      : fold_n,
                })
            continue

        print(f"\n  ── {model_name}  (p={dropout_p}, mode={dropout_mode}) ──")

        train_ds = GliomaDataset(X_train, y_train, cache_map,
                                 dropout_p=dropout_p, dropout_mode=dropout_mode,
                                 model_name=model_name)
        val_ds   = GliomaDataset(X_val, y_val, cache_map)

        model, best_state, history, best_auc = train_one_fold(
            model_fn(),
            make_loader(train_ds, shuffle=True),
            make_loader(val_ds,   shuffle=False),
            fold_w
        )

        # Load best weights
        core = model.module if MULTI_GPU else model
        core.load_state_dict(best_state)

        print(f"\n  Evaluating scenarios...")
        scenario_results = evaluate_scenarios(
            model, X_val, y_val, cache_map, fold_n, model_name)

        ckpt_path = f"{SAVE_DIR}/fold{fold_n}_{model_name}_best.pth"
        torch.save(best_state, ckpt_path)

        fold_data = {
            "fold": fold_n, "model": model_name,
            "train_subjects": X_train, "val_subjects": X_val,
            "train_labels": y_train,   "val_labels": y_val,
            "best_auc": best_auc, "history": history,
            "scenario_results": scenario_results,
            "checkpoint_path": ckpt_path,
            "config": {
                "img_size": IMG_SIZE, "batch": BATCH, "epochs": EPOCHS,
                "lr": LR, "dropout_p": dropout_p, "dropout_mode": dropout_mode,
                "gpu_ids": GPU_IDS, "seed": SEED,
                "class_weights": [1.0, round(float(w_mut), 4)],
            }
        }

        with open(save_path, "w") as f_out:
            json.dump(fold_data, f_out, indent=2)
        print(f"  ✓ Saved: {save_path}")

        # Opt-3: accumulate OOF predictions (All_4 scenario = full modality)
        for row in scenario_results["All_4"]["per_subject"]:
            oof_store[model_name].append({
                "subject_id": row["subject_id"],
                "true_label": row["true_label"],
                "pred_prob" : row["pred_prob"],
                "fold"      : fold_n,
            })

        del model
        torch.cuda.empty_cache()

elapsed = (time.time() - start_time) / 60
print(f"\n{'='*62}\nAll folds done in {elapsed:.1f} min\n{'='*62}")


# ═══════════════════════════════════════════════════════════════════════════════
# Opt-3 — SAVE OOF PREDICTIONS CSV
# ═══════════════════════════════════════════════════════════════════════════════
print("\nSaving OOF predictions...")
oof_rows = []
for model_name, rows in oof_store.items():
    for r in rows:
        oof_rows.append({**r, "model": model_name})
oof_df = pd.DataFrame(oof_rows)
oof_df.to_csv(f"{SAVE_DIR}/all_oof_predictions.csv", index=False)
print(f"Saved: {SAVE_DIR}/all_oof_predictions.csv  "
      f"({len(oof_df)} rows)")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY  +  STATISTICAL TESTS  (Stat recommendation)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nSUMMARY — Mean ± Std AUC\n")
summary = {}

for model_name, *_ in MODEL_CONFIGS:
    summary[model_name] = {}
    folds_data = []
    for ff in range(1, N_FOLDS + 1):
        fp = f"{SAVE_DIR}/fold{ff}_{model_name}.json"
        if os.path.exists(fp):
            with open(fp) as f_in:
                folds_data.append(json.load(f_in))

    for sc_name in TEST_SCENARIOS:
        aucs = [fd["scenario_results"][sc_name]["metrics"]["AUC"]
                for fd in folds_data
                if fd["scenario_results"][sc_name]["metrics"]["AUC"] is not None]
        if aucs:
            summary[model_name][sc_name] = {
                "mean": round(float(np.mean(aucs)), 4),
                "std" : round(float(np.std(aucs)),  4),
                "fold_aucs": [round(a, 4) for a in aucs],
            }
            print(f"  {model_name:28s} | {sc_name:20s} | "
                  f"AUC {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        else:
            summary[model_name][sc_name] = {"mean": None, "std": None, "fold_aucs": []}
            print(f"  {model_name:28s} | {sc_name:20s} | AUC N/A")

# ── Stat: paired Wilcoxon signed-rank test across folds ──────────────────────
print("\nStatistical tests (Wilcoxon signed-rank, paired across folds):\n")
stats_results = {}
model_names   = [name for name, *_ in MODEL_CONFIGS]

for sc_name in TEST_SCENARIOS:
    stats_results[sc_name] = {}
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2   = model_names[i], model_names[j]
            aucs1    = summary[m1][sc_name].get("fold_aucs", [])
            aucs2    = summary[m2][sc_name].get("fold_aucs", [])
            pair_key = f"{m1}_vs_{m2}"

            if len(aucs1) == len(aucs2) == N_FOLDS:
                stat, p = scipy_stats.wilcoxon(aucs1, aucs2,
                                               alternative="two-sided")
                stars = "***" if p < 0.001 else "**" if p < 0.01 else \
                        "*"   if p < 0.05  else "ns"
                stats_results[sc_name][pair_key] = {
                    "statistic": round(float(stat), 4),
                    "p_value"  : round(float(p),    6),
                    "sig"      : stars,
                }
                print(f"  {sc_name:20s} | {pair_key}  p={p:.4f} {stars}")
            else:
                stats_results[sc_name][pair_key] = {"note": "insufficient folds"}

# ── Pooled OOF stats: Bootstrap AUC CI + Brier score ─────────────────────────
# NOTE: This is bootstrap-based confidence interval estimation, NOT a DeLong
# test.  DeLong requires a separate implementation (e.g. fastDeLong).
# In the paper, report as: "Bootstrap 95% CI (n=1000 stratified resamples)."
# For pairwise significance between models, use the Wilcoxon results above.
print("\nPooled OOF — Bootstrap AUC CI (1000 iter) + Brier Score:\n")
from sklearn.metrics import brier_score_loss

bootstrap_results = {}
for model_name in model_names:
    sub = oof_df[oof_df["model"] == model_name]
    if len(sub) == 0:
        continue
    y_true = sub["true_label"].values
    y_prob = sub["pred_prob"].values
    pooled_auc = safe_auc(y_true, y_prob)

    # Problem 4 fix: Brier score (lower = better calibrated)
    brier = float(brier_score_loss(y_true, y_prob))

    # Bootstrap AUC CI (stratified resampling preserves class ratio)
    rng    = np.random.default_rng(SEED)
    b_aucs = []
    idx0   = np.where(y_true == 0)[0]
    idx1   = np.where(y_true == 1)[0]
    for _ in range(1000):
        # Stratified resample: same class proportions as original
        b0  = rng.choice(idx0, size=len(idx0), replace=True)
        b1  = rng.choice(idx1, size=len(idx1), replace=True)
        bi  = np.concatenate([b0, b1])
        bt  = y_true[bi]; bp = y_prob[bi]
        b_aucs.append(safe_auc(bt, bp))
    b_aucs  = [a for a in b_aucs if not np.isnan(a)]
    ci_lo, ci_hi = np.percentile(b_aucs, [2.5, 97.5])

    bootstrap_results[model_name] = {
        "pooled_auc" : round(pooled_auc, 4),
        "ci_95"      : [round(ci_lo, 4), round(ci_hi, 4)],
        "brier_score": round(brier, 4),        # Problem 4
    }
    print(f"  {model_name:28s} | AUC {pooled_auc:.4f} [{ci_lo:.4f}–{ci_hi:.4f}] "
          f"| Brier {brier:.4f}")

# Save all stats
stats_out = {
    "wilcoxon_per_scenario": stats_results,
    "bootstrap_oof"        : bootstrap_results,
}
with open(f"{SAVE_DIR}/stats_summary.json", "w") as f:
    json.dump(stats_out, f, indent=2)
print(f"\nSaved: {SAVE_DIR}/stats_summary.json")

with open(f"{SAVE_DIR}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {SAVE_DIR}/summary.json")

print("\n✓ Training complete!  Run idh_cv_plot.py for figures.")
print(f"\nOutputs in {SAVE_DIR}/:")
print("  dataset_info.json")
print("  summary.json")
print("  stats_summary.json              ← Wilcoxon + bootstrap AUC CI + Brier score")
print("  all_oof_predictions.csv         ← full-cohort ROC source")
print("  fold[1-5]_[model].json          ← metrics + predictions (no embeddings)")
print("  fold[1-5]_[model]_best.pth      ← model checkpoints")
print(f"\nOutputs in {SAVE_DIR}/embeddings/:")
print("  fold[1-5]_[model]_[scenario].npy  ← 512-d float32 embeddings")
