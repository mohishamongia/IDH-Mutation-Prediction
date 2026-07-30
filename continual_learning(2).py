"""
5-Fold Cross-Validation — Missing-Modality Robust IDH Prediction
Models : Baseline_ResNet18 | Dropout30_ResNet18 | Dropout50_ResNet18
Run    : python idh_cv_train.py

"""

# ── CUDA env var MUST precede 'import torch' ──────────────────────────────────
import os

GPU_IDS = [0, 1, 2, 3, 5, 7]   # ✏️ edit before every run
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
import torch.nn.functional as F
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
DATA_DIR  = "DATASETS/UTSW_Glioma_data/UTSW-Glioma"
TSV_PATH  = "DATASETS/UTSW_Glioma_data/UTSW_Glioma_Metadata-2-1 (2).tsv"
SAVE_DIR  = "results(2)"
CACHE_DIR = "cache_npy"   # preprocessed volume cache
IMG_SIZE  = 96
BATCH     = 24            # lowered from 48; ~26 updates/epoch on 622 subjects
EPOCHS    = 100
LR        = 1e-4
PATIENCE  = 7
N_FOLDS   = 5
# Repeated k-fold gives N_REPEATS*N_FOLDS effective folds — the main lever for
# statistical power. 1 = original; 3 trades compute for power, scaling linearly.
N_REPEATS = 1

# ── Continual-learning (EWC) config ────────────────────────────────────────
# Smaller than EPOCHS/PATIENCE: the continual model trains 4 sequential tasks
# per fold. Cut to save runtime; raise if fold results look under-converged.
TASK_EPOCHS   = 40
TASK_PATIENCE = 5
# Acts on normalised Fisher (~1.0), so it lives in a small range. Raise if
# All_4 AUC collapses across tasks; lower if T1_only stays near chance.
EWC_LAMBDA    = 5.0
# EWC doesn't cover BN running stats (they're buffers, not params), so freezing
# them after Task1 keeps All_4 retention from measuring BN drift as forgetting.
FREEZE_BN_AFTER_TASK1 = True
# Curriculum easiest→hardest. Tier 2 groups the three one-missing scenarios into
# one task (same difficulty tier) so it's 4 tasks, not 6.
TASK_SEQUENCE = [
    ("Task1_AllModalities", ["All_4"]),
    ("Task2_OneMissing",    ["T1CE_missing", "FLAIR_missing", "T2_missing"]),
    ("Task3_TwoAvailable",  ["T1_T2_only"]),
    ("Task4_SingleModality",["T1_only"]),
]

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

def json_safe(o):
    # nan/inf → None and numpy scalars → python, so every dumped file is valid JSON
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, dict):
        return {k: json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    return o

def safe_cm(y_true, y_pred):
    """Always returns 2×2; safe even when fold predicts single class."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1])

def make_loader(dataset, shuffle, batch=BATCH):
    g = torch.Generator()
    g.manual_seed(SEED)
    return DataLoader(
        dataset, batch_size=batch, shuffle=shuffle,
        num_workers=2, pin_memory=True,
        worker_init_fn=seed_worker, generator=g
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING CACHE
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

# Scenario sampling weights for structured dropout. dropout_p is unused here —
# the scenario distribution sets how aggressively modalities drop.
# Order matches TEST_SCENARIOS: All_4 | T1CE_missing | FLAIR_missing | T2_missing | T1_T2_only | T1_only
STRUCTURED_WEIGHTS = {
    "Dropout30_ResNet18": [0.40, 0.15, 0.15, 0.15, 0.10, 0.05],
    "Dropout50_ResNet18": [0.10, 0.20, 0.20, 0.20, 0.15, 0.15],
}
# ─────────────────────────────────────────────────────────────────────────────

class GliomaDataset(Dataset):
    """
    dropout_mode:
      'random'          — each channel independently dropped with prob dropout_p
                          (dropout_p IS used here)
      'structured'      — sample a test scenario according to model-specific
                          weights in STRUCTURED_WEIGHTS (dropout_p ignored —
                          distribution controls aggressiveness; model_name
                          must be supplied)
      'task_scenarios'  — uniformly sample among a restricted scenario subset
                          (task_scenarios param) each __getitem__ call. Used
                          for one task in the continual-learning (EWC)
                          curriculum — e.g. Task2_OneMissing uniformly rotates
                          through T1CE_missing/FLAIR_missing/T2_missing so the
                          model doesn't overfit to one specific channel being
                          the one that's gone.
      None / 0.0        — no dropout (validation / test)
    missing: list of channel indices to KEEP (test-time explicit scenario)
    """
    def __init__(self, subjects, labels, cache_map,
                 dropout_p=0.0, dropout_mode="random",
                 missing=None, model_name=None, task_scenarios=None):
        self.subjects     = subjects
        self.labels       = labels
        self.cache_map    = cache_map
        self.dropout_p    = dropout_p
        self.dropout_mode = dropout_mode
        self.missing      = missing
        self.epoch        = 0
        self._scenario_lists   = list(TEST_SCENARIOS.values())
        # Resolve per-model weights; fall back to uniform if name not found
        raw_w = STRUCTURED_WEIGHTS.get(model_name, [1/6]*6)
        total = sum(raw_w)
        self._scenario_weights = [w / total for w in raw_w]
        # For 'task_scenarios' mode: restrict sampling to this scenario subset
        self._task_channel_lists = (
            [TEST_SCENARIOS[s] for s in task_scenarios]
            if task_scenarios is not None else None
        )

    def __len__(self):
        return len(self.subjects)

    def set_epoch(self, epoch):
        # folded into the mask seed so masks change each epoch — default workers
        # re-seed identically every epoch, which froze the old global-RNG draws
        self.epoch = int(epoch)

    def __getitem__(self, idx):
        sid    = self.subjects[idx]
        volume = np.load(self.cache_map[sid]).astype(np.float32).copy()  # (4,96,96,96)

        # local RNG seeded on (seed, epoch, idx): reproducible but fresh per epoch,
        # independent of how workers are seeded
        mask_seed = (SEED * 2654435761 + self.epoch * 40503 + idx) & 0x7FFFFFFF
        np_rng    = np.random.default_rng(mask_seed)
        py_rng    = random.Random(mask_seed)

        if self.dropout_p > 0 and self.dropout_mode == "random":
            # Independent per-channel dropout with all-zero guardrail
            mask = np_rng.random(4) < self.dropout_p
            if mask.all():
                mask[np_rng.integers(4)] = False
            volume[mask] = 0.0

        elif self.dropout_mode == "structured":
            # scenario sampled by STRUCTURED_WEIGHTS (dropout_p unused in this mode)
            avail = py_rng.choices(self._scenario_lists,
                                   weights=self._scenario_weights, k=1)[0]
            for c in range(4):
                if c not in avail:
                    volume[c] = 0.0

        elif self.dropout_mode == "task_scenarios":
            # Continual-learning curriculum: uniformly sample among this
            # task's scenario subset (e.g. the 3 one-missing scenarios).
            avail = py_rng.choice(self._task_channel_lists)
            for c in range(4):
                if c not in avail:
                    volume[c] = 0.0

        if self.missing is not None:
            for c in range(4):
                if c not in self.missing:
                    volume[c] = 0.0

        return torch.tensor(volume), torch.tensor(self.labels[idx], dtype=torch.long)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
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


class SelfAttention3D(nn.Module):
    """
    Non-local self-attention block (Wang et al., "Non-local Neural Networks",
    2018 — embedded-Gaussian variant), adapted to 3D feature maps.

    Inserted after layer3 (256 channels), NOT after layer4, deliberately:
    at IMG_SIZE=96 the spatial size after layer3 is 6³=216, so the N×N
    attention matrix is 216×216 (cheap). After layer4 it's 3³=27, small
    enough that self-attention has almost no receptive-field benefit left
    over the conv's own field of view — the block would cost compute for
    little gain. layer3 is the sweet spot for this input resolution.

    query/key are projected to in_ch//reduction to keep the attention matrix
    affordable; value stays at full in_ch so the output channel count matches
    the input with no extra projection needed for the residual add.

    gamma starts at 0 (verified in proto_attn.py), so this block is the
    identity function at initialisation — it can only ever help relative to
    the pre-attention ResNet18_3D, never destabilise the existing training
    dynamics your Baseline/Dropout30/Dropout50 models already rely on.
    """
    def __init__(self, in_ch, reduction=8):
        super().__init__()
        self.inter_ch = max(in_ch // reduction, 1)
        self.theta  = nn.Conv3d(in_ch, self.inter_ch, 1, bias=False)  # query
        self.phi    = nn.Conv3d(in_ch, self.inter_ch, 1, bias=False)  # key
        self.g      = nn.Conv3d(in_ch, in_ch,        1, bias=False)   # value
        self.out    = nn.Conv3d(in_ch, in_ch,        1, bias=False)
        self.bn_out = nn.BatchNorm3d(in_ch)
        self.gamma  = nn.Parameter(torch.zeros(1))   # residual-safe init

    def forward(self, x):
        B, C, D, H, W = x.shape
        N = D * H * W

        q = self.theta(x).view(B, self.inter_ch, N).permute(0, 2, 1)  # (B, N, C')
        k = self.phi(x).view(B, self.inter_ch, N)                     # (B, C', N)
        v = self.g(x).view(B, C, N).permute(0, 2, 1)                  # (B, N, C)

        attn = torch.bmm(q, k)
        attn = torch.softmax(attn / (self.inter_ch ** 0.5), dim=-1)

        out = torch.bmm(attn, v)
        out = out.permute(0, 2, 1).contiguous().view(B, C, D, H, W)
        out = self.bn_out(self.out(out))

        return x + self.gamma * out


class SEBlock3D(nn.Module):
    # Squeeze-and-Excitation channel attention (Hu et al. 2018): global-pool each
    # channel, learn a per-channel gate, rescale. ~1% of a non-local block's params.
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c = x.shape[:2]
        s = x.mean(dim=(2, 3, 4))            # global average pool
        s = self.fc(s).view(b, c, 1, 1, 1)
        return x * s


class CBAM3D(nn.Module):
    # CBAM (Woo et al. 2018): channel gate (avg+max pooled through a shared MLP)
    # then a light spatial gate. Slightly heavier than SE, still very cheap.
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False))
        self.spatial = nn.Conv3d(2, 1, spatial_kernel,
                                 padding=spatial_kernel // 2, bias=False)

    def forward(self, x):
        b, c = x.shape[:2]
        ch = torch.sigmoid(self.mlp(x.mean(dim=(2, 3, 4))) +
                           self.mlp(x.amax(dim=(2, 3, 4)))).view(b, c, 1, 1, 1)
        x = x * ch
        sp = torch.sigmoid(self.spatial(torch.cat(
            [x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)))
        return x * sp


def _make_attention(attention_type, channels):
    # single place that maps an attention name to a module, so the ablation
    # (none / nonlocal / se / cbam) is a one-word switch at the same insertion point
    if attention_type == "nonlocal":
        return SelfAttention3D(channels)
    if attention_type == "se":
        return SEBlock3D(channels)
    if attention_type == "cbam":
        return CBAM3D(channels)
    return nn.Identity()


class ResNet18_3D(nn.Module):
    def __init__(self, in_channels=4, num_classes=2, use_attention=True, attention_type=None):
        super().__init__()
        # attention_type wins when given; otherwise fall back to the old boolean
        # (True→nonlocal, False→none) so existing configs/checkpoints are unchanged
        if attention_type is None:
            attention_type = "nonlocal" if use_attention else "none"
        self.attention_type = attention_type
        self.use_attention  = (attention_type != "none")
        self.stem   = nn.Sequential(
            nn.Conv3d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1))
        self.layer1 = nn.Sequential(ResBlock3D(64,  64),            ResBlock3D(64,  64))
        self.layer2 = nn.Sequential(ResBlock3D(64,  128, stride=2), ResBlock3D(128, 128))
        self.layer3 = nn.Sequential(ResBlock3D(128, 256, stride=2), ResBlock3D(256, 256))
        self.attn   = _make_attention(attention_type, 256)   # after layer3, same spot for every type
        self.layer4 = nn.Sequential(ResBlock3D(256, 512, stride=2), ResBlock3D(512, 512))
        self.pool   = nn.AdaptiveAvgPool3d(1)
        self.drop   = nn.Dropout(0.5)
        self.fc     = nn.Linear(512, num_classes)

    def forward(self, x):
        """Standard forward — logits only. DataParallel-safe (no extra args)."""
        x   = self.stem(x)
        x   = self.layer1(x); x = self.layer2(x)
        x   = self.layer3(x); x = self.attn(x); x = self.layer4(x)
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
        x = self.layer3(x); x = self.attn(x); x = self.layer4(x)
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
        train_loader.dataset.set_epoch(epoch)   # new masks this epoch; before workers fork
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
# CONTINUAL LEARNING (EWC) — sequential task curriculum, verified in proto_ewc.py
# ═══════════════════════════════════════════════════════════════════════════════
def _freeze_bn_running_stats(core_model):
    # BN.eval() stops running-stat updates while weights still train; re-called
    # after every model.train(), which would otherwise switch BN back on
    for m in core_model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eval()


def _quick_validate_auc(model, loader):
    """
    Lightweight validation: AUC only, no confusion matrix / embeddings.
    Used for per-epoch monitoring during the continual-learning curriculum
    (called far more often than the full evaluate_scenarios(), which writes
    embeddings to disk and would be far too slow to call every epoch).
    """
    model.eval()
    all_probs, all_true = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            all_probs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
            all_true.extend(lbls.numpy())
    return safe_auc(all_true, all_probs)


class EWCState:
    """
    Online EWC (Schwarz et al., "Progress & Compress", 2018): a single running
    Fisher matrix accumulated across tasks, and a single anchor (star_params)
    updated to the model's weights at the end of each task.

    Fisher is normalised (mean-rescaled to ~1.0) BEFORE accumulation — verified
    in proto_ewc.py to be necessary: raw empirical Fisher values are tiny and
    scale-dependent on architecture/loss, so without normalisation EWC_LAMBDA
    would need re-sweeping by orders of magnitude for this ResNet18+attention
    vs. whatever toy problem it was tuned on. Normalised, EWC_LAMBDA stays in
    a small human-tunable range (~0.1-100) regardless of architecture.
    """
    def __init__(self):
        self.fisher = None       # dict[name -> tensor], accumulated across tasks
        self.star_params = None  # dict[name -> tensor], anchor = previous task's final weights

    def penalty(self, core_model):
        if self.fisher is None:
            return torch.tensor(0.0, device=DEVICE)
        loss = 0.0
        for n, p in core_model.named_parameters():
            f = self.fisher[n].to(p.device)
            s = self.star_params[n].to(p.device)
            loss = loss + (f * (p - s) ** 2).sum()
        return EWC_LAMBDA * 0.5 * loss

    def consolidate(self, core_model, loader, n_batches=20):
        """
        Call at the END of each task: compute this task's Fisher information
        (empirical Fisher via the model's own predicted labels — no extra
        label leakage needed), normalise it, and accumulate into the running
        Fisher. Anchor (star_params) becomes this task's final weights.
        """
        core_model.eval()
        new_fisher = {n: torch.zeros_like(p) for n, p in core_model.named_parameters()}
        seen = 0
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)
            core_model.zero_grad()
            logits = core_model(imgs)
            pseudo_labels = logits.argmax(1).detach()
            loss = F.cross_entropy(logits, pseudo_labels)
            loss.backward()
            for n, p in core_model.named_parameters():
                if p.grad is not None:
                    new_fisher[n] += p.grad.detach() ** 2
            seen += 1
            if seen >= n_batches:
                break
        for n in new_fisher:
            new_fisher[n] /= max(seen, 1)

        all_vals = torch.cat([f.flatten() for f in new_fisher.values()])
        scale = all_vals.mean().clamp_min(1e-12)
        new_fisher = {n: f / scale for n, f in new_fisher.items()}

        if self.fisher is None:
            self.fisher = new_fisher
        else:
            for n in self.fisher:
                self.fisher[n] = self.fisher[n] + new_fisher[n]

        self.star_params = {n: p.clone().detach()
                             for n, p in core_model.named_parameters()}
        core_model.zero_grad()


def train_continual_fold(model, X_train, y_train, X_val, y_val,
                          cache_map, class_weights_tensor, model_name):
    """
    Sequential-task continual learning curriculum (TASK_SEQUENCE), with EWC
    protecting earlier tasks' weights while later, harder tasks are learned.

    Returns (model, best_state, history, best_auc) — same signature as
    train_one_fold() so the main loop / checkpoint saving / JSON structure
    downstream needs no changes.

    history["retention_curve"]: All_4-scenario val AUC measured every epoch
    across the WHOLE curriculum (all 4 tasks) — this is the key plot for the
    paper: does the model actually keep its full-modality performance while
    learning to handle missing modalities, or does it quietly erode?
    history["tasks"]: per-task sub-histories (train_loss, val_auc per epoch).
    """
    if MULTI_GPU:
        model = nn.DataParallel(model)
    model = model.to(DEVICE)
    core = model.module if MULTI_GPU else model

    ewc = EWCState()
    history = {"tasks": {}, "retention_curve": []}
    best_state = {k: v.cpu().clone() for k, v in core.state_dict().items()}

    # Fixed loader to measure All_4 retention every epoch, regardless of
    # which task is currently training.
    all4_val_ds     = GliomaDataset(X_val, y_val, cache_map, missing=TEST_SCENARIOS["All_4"])
    all4_val_loader = make_loader(all4_val_ds, shuffle=False)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(DEVICE))

    for task_idx, (task_name, task_scenarios) in enumerate(TASK_SEQUENCE):
        print(f"\n  ── [{model_name}] {task_name}  (scenarios: {task_scenarios}) ──")
        freeze_bn = FREEZE_BN_AFTER_TASK1 and task_idx >= 1
        if freeze_bn:
            print(f"    (BatchNorm running stats FROZEN for {task_name} — "
                  f"pinned to full-modality Task1 estimate)")

        train_ds = GliomaDataset(X_train, y_train, cache_map,
                                  dropout_mode="task_scenarios",
                                  task_scenarios=task_scenarios)
        train_loader = make_loader(train_ds, shuffle=True)

        # Task validation: average AUC across this task's own scenario(s)
        task_val_loaders = [
            make_loader(GliomaDataset(X_val, y_val, cache_map,
                                       missing=TEST_SCENARIOS[s]), shuffle=False)
            for s in task_scenarios
        ]

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode="max", patience=3, factor=0.5)

        task_hist = {"train_loss": [], "val_auc": []}
        best_task_auc, patience_ctr = -1.0, 0
        task_best_state = {k: v.cpu().clone() for k, v in core.state_dict().items()}

        for epoch in range(TASK_EPOCHS):
            train_loader.dataset.set_epoch(epoch)   # fresh masks per epoch
            model.train()
            if freeze_bn:
                _freeze_bn_running_stats(core)      # re-freeze BN after .train()
            batch_losses = []
            for imgs, lbls in tqdm(train_loader,
                                    desc=f"    {task_name} Ep{epoch+1:03d}", leave=False):
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(imgs), lbls) + ewc.penalty(core)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                batch_losses.append(loss.item())
            task_hist["train_loss"].append(float(np.mean(batch_losses)))

            task_aucs = [_quick_validate_auc(model, ld) for ld in task_val_loaders]
            task_aucs = [a for a in task_aucs if not np.isnan(a)]
            task_auc  = float(np.mean(task_aucs)) if task_aucs else float("nan")
            task_hist["val_auc"].append(task_auc)

            all4_auc = _quick_validate_auc(model, all4_val_loader)
            history["retention_curve"].append({
                "task": task_name, "epoch": epoch + 1,
                "task_auc": task_auc, "all4_auc": all4_auc,
            })

            auc_s  = f"{task_auc:.4f}" if not np.isnan(task_auc) else "NaN "
            all4_s = f"{all4_auc:.4f}" if not np.isnan(all4_auc) else "NaN "
            print(f"    Ep{epoch+1:03d} | L {task_hist['train_loss'][-1]:.4f} | "
                  f"TaskAUC {auc_s} | All4AUC(retention) {all4_s}", flush=True)

            if not np.isnan(task_auc):
                scheduler.step(task_auc)
                if task_auc > best_task_auc:
                    best_task_auc  = task_auc
                    task_best_state = {k: v.cpu().clone() for k, v in core.state_dict().items()}
                    patience_ctr = 0
                    print(f"    ✓ Best {task_name} AUC {task_auc:.4f}", flush=True)
                else:
                    patience_ctr += 1
            else:
                patience_ctr += 1
            if patience_ctr >= TASK_PATIENCE:
                print(f"    Early stop {task_name} @ epoch {epoch+1}")
                break

        # Load this task's best weights before consolidating Fisher / moving on
        core.load_state_dict(task_best_state)
        ewc.consolidate(core, train_loader)

        history["tasks"][task_name] = task_hist
        best_state = task_best_state

    # report best_auc on All_4 like the other models, not Task4's single-modality
    # number, so all four models' best_auc mean the same thing
    core.load_state_dict(best_state)
    best_auc = _quick_validate_auc(model, all4_val_loader)
    history["all4_best_auc"]       = best_auc
    history["final_task_best_auc"] = float(best_task_auc)   # single-modality, kept for reference

    return model, best_state, history, best_auc


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATE — all scenarios; embeddings saved to .npy, not JSON
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

        # embeddings → .npy, not JSON (too big to inline)
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

# build preprocessing cache once (skips already-cached subjects)
cache_map = preprocess_and_cache(subjects, DATA_DIR, CACHE_DIR, IMG_SIZE)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL CONFIGS
# ═══════════════════════════════════════════════════════════════════════════════
# dropout_mode: 'structured' for robust models; None for baseline
MODEL_CONFIGS = [
    # (name,                  model_factory,                             dropout_p, dropout_mode)
    # Baseline vs Attention differ only in use_attention — that pair is the ablation.
    ("Baseline_ResNet18",     lambda: ResNet18_3D(use_attention=False), 0.0, None),
    ("Attention_ResNet18",    lambda: ResNet18_3D(use_attention=True),  0.0, None),
    # Lightweight attention alternatives to the (null) non-local block above —
    # same insertion point, so Baseline vs these four is a clean "which attention
    # helps, if any" study. DeLong on pooled OOF decides significance.
    ("SE_ResNet18",           lambda: ResNet18_3D(attention_type="se"),   0.0, None),
    ("CBAM_ResNet18",         lambda: ResNet18_3D(attention_type="cbam"), 0.0, None),
    ("Dropout30_ResNet18",    lambda: ResNet18_3D(use_attention=True),  0.3, "structured"),
    ("Dropout50_ResNet18",    lambda: ResNet18_3D(use_attention=True),  0.5, "structured"),
    # "continual_ewc" is a sentinel: the main loop calls train_continual_fold()
    # instead of train_one_fold(). dropout_p unused (masking from TASK_SEQUENCE).
    ("ContinualEWC_ResNet18", lambda: ResNet18_3D(use_attention=True),  0.0, "continual_ewc"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 5-FOLD CV + OOF accumulation
# ═══════════════════════════════════════════════════════════════════════════════
subjects_np = np.array(subjects)
labels_np   = np.array(labels)
TOTAL_RUNS  = N_REPEATS * N_FOLDS

# All (train, val) splits across repeats, flattened into one list so the loop
# body below is unchanged and fold_n stays unique (keeps checkpoint naming +
# resume-on-skip working).
splits = []
for _rep in range(N_REPEATS):
    _skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + _rep)
    splits.extend(_skf.split(subjects_np, labels_np))

# OOF store: model_name → list of dicts
oof_store = {name: [] for name, *_ in MODEL_CONFIGS}

start_time = time.time()

for fold, (train_idx, val_idx) in enumerate(splits):
    fold_n = fold + 1
    print(f"\n{'#'*62}\nRUN {fold_n}/{TOTAL_RUNS}\n{'#'*62}")

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

        if dropout_mode == "continual_ewc":
            # continual model builds its own per-task datasets internally
            # (each task needs a scenario-restricted dataset, not one fixed train_ds)
            model, best_state, history, best_auc = train_continual_fold(
                model_fn(), X_train, y_train, X_val, y_val,
                cache_map, fold_w, model_name
            )
        else:
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
            json.dump(json_safe(fold_data), f_out, indent=2)
        print(f"  ✓ Saved: {save_path}")

        # accumulate OOF predictions (All_4 = full modality)
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
# SAVE OOF PREDICTIONS CSV
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
# ANALYSIS HELPERS — DeLong paired AUC test, calibration, operating points
# ═══════════════════════════════════════════════════════════════════════════════
def _midrank(x):
    order = np.argsort(x)
    ranked = x[order]
    n = len(x)
    out = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and ranked[j] == ranked[i]:
            j += 1
        out[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    res = np.empty(n)
    res[order] = out
    return res


def delong_auc_test(y_true, prob_a, prob_b):
    # Paired AUC comparison on the same subjects (fast DeLong, Sun & Xu 2014).
    # Returns (auc_a, auc_b, p) for H0: AUC_a == AUC_b. Uses every subject's
    # prediction, so it's far more powerful than Wilcoxon over a handful of folds.
    y_true = np.asarray(y_true); prob_a = np.asarray(prob_a); prob_b = np.asarray(prob_b)
    order = np.argsort(-y_true, kind="stable")
    m = int(y_true.sum())                 # positives first
    preds = np.vstack((prob_a, prob_b))[:, order]
    n = preds.shape[1] - m
    if m == 0 or n == 0:
        return float("nan"), float("nan"), float("nan")
    pos, neg = preds[:, :m], preds[:, m:]
    tx = np.vstack([_midrank(pos[r]) for r in range(2)])
    ty = np.vstack([_midrank(neg[r]) for r in range(2)])
    tz = np.vstack([_midrank(preds[r]) for r in range(2)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n     # 2x2
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2 * scipy_stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(p)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    # Mean gap between confidence and accuracy across probability bins.
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        sel = (y_prob > edges[b]) & (y_prob <= edges[b + 1])
        if sel.sum() == 0:
            continue
        ece += sel.mean() * abs(y_true[sel].mean() - y_prob[sel].mean())
    return float(ece)


def fit_temperature(y_true, y_prob):
    # Post-hoc temperature scaling: one scalar T on the logits, chosen to
    # minimise NLL. Doesn't touch training; sharpens/softens probabilities only.
    eps = 1e-6
    y_true = np.asarray(y_true, float)
    logit = np.log(np.clip(y_prob, eps, 1 - eps) / (1 - np.clip(y_prob, eps, 1 - eps)))

    def nll(T):
        p = 1.0 / (1.0 + np.exp(-logit / T))
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    from scipy.optimize import minimize_scalar
    T = float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)
    p_cal = 1.0 / (1.0 + np.exp(-logit / T))
    return T, p_cal


def operating_points(y_true, y_prob):
    # Youden's J (max sens+spec-1) and the best sensitivity at spec >= 0.90.
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    thr = np.clip(thr, 0.0, 1.0)          # roc_curve prepends inf; keep thresholds finite
    j = int(np.argmax(tpr - fpr))
    youden = {"threshold": float(thr[j]), "sensitivity": float(tpr[j]),
              "specificity": float(1 - fpr[j])}
    hi = np.where((1 - fpr) >= 0.90)[0]
    high_spec = None
    if len(hi):
        k = hi[int(np.argmax(tpr[hi]))]
        high_spec = {"threshold": float(thr[k]), "sensitivity": float(tpr[k]),
                     "specificity": float(1 - fpr[k])}
    return youden, high_spec


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY + STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\nSUMMARY — Mean ± Std AUC\n")
summary = {}

for model_name, *_ in MODEL_CONFIGS:
    summary[model_name] = {}
    folds_data = []
    for ff in range(1, TOTAL_RUNS + 1):
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

# paired Wilcoxon across folds (per scenario); DeLong on pooled OOF is the primary test
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

            if len(aucs1) == len(aucs2) == TOTAL_RUNS:
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

# Pooled OOF: bootstrap AUC CI (stratified, 1000 resamples) + Brier. Pairwise
# significance is the DeLong test below, which supersedes the fold-level Wilcoxon.
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

    # Brier score (lower = better calibrated)
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
        "brier_score": round(brier, 4),
    }
    print(f"  {model_name:28s} | AUC {pooled_auc:.4f} [{ci_lo:.4f}–{ci_hi:.4f}] "
          f"| Brier {brier:.4f}")

# ── DeLong pairwise AUC test (paired, pooled OOF) ────────────────────────────
# Aggregate each subject to one prediction (mean across repeats) so pairs align
# 1:1 and each subject counts once — this is the primary significance test.
print("\nDeLong pairwise AUC test (pooled OOF, paired by subject):\n")
agg = {}
for model_name in model_names:
    sub = oof_df[oof_df["model"] == model_name]
    if len(sub) == 0:
        continue
    g = sub.groupby("subject_id").agg(true=("true_label", "first"),
                                      prob=("pred_prob", "mean"))
    agg[model_name] = g

delong_results = {}
present = [m for m in model_names if m in agg]
for i in range(len(present)):
    for j in range(i + 1, len(present)):
        m1, m2 = present[i], present[j]
        common = agg[m1].index.intersection(agg[m2].index)
        if len(common) < 10:
            continue
        yt = agg[m1].loc[common, "true"].values
        a1 = agg[m1].loc[common, "prob"].values
        a2 = agg[m2].loc[common, "prob"].values
        auc1, auc2, p = delong_auc_test(yt, a1, a2)
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        delong_results[f"{m1}_vs_{m2}"] = {
            "auc_a": round(auc1, 4), "auc_b": round(auc2, 4),
            "p_value": round(p, 6), "sig": stars, "n": int(len(common)),
        }
        print(f"  {m1:24s} vs {m2:24s} | ΔAUC {auc1 - auc2:+.4f} | p={p:.4f} {stars}")

# ── Calibration + operating points (pooled OOF) ──────────────────────────────
print("\nCalibration (temperature-scaled) + operating points (pooled OOF):\n")
calibration_results = {}
for model_name in present:
    yt = agg[model_name]["true"].values
    yp = agg[model_name]["prob"].values
    ece_raw = expected_calibration_error(yt, yp)
    T, yp_cal = fit_temperature(yt, yp)
    ece_cal = expected_calibration_error(yt, yp_cal)
    brier_raw = float(brier_score_loss(yt, yp))
    brier_cal = float(brier_score_loss(yt, yp_cal))
    youden, high_spec = operating_points(yt, yp)
    calibration_results[model_name] = {
        "ece_raw": round(ece_raw, 4), "ece_calibrated": round(ece_cal, 4),
        "brier_raw": round(brier_raw, 4), "brier_calibrated": round(brier_cal, 4),
        "temperature": round(T, 4),
        "youden": youden, "high_spec_0.90": high_spec,
    }
    js = f"J thr={youden['threshold']:.3f} sens={youden['sensitivity']:.3f} spec={youden['specificity']:.3f}"
    print(f"  {model_name:24s} | ECE {ece_raw:.3f}→{ece_cal:.3f} (T={T:.2f}) | "
          f"Brier {brier_raw:.3f}→{brier_cal:.3f} | {js}")

# Save all stats
# Save all stats — non-finite floats (nan/inf) are sanitised to null via
# json_safe (defined up top) so the JSON is always valid.
stats_out = {
    "wilcoxon_per_scenario": stats_results,
    "bootstrap_oof"        : bootstrap_results,
    "delong_pairwise_oof"  : delong_results,
    "calibration_oof"      : calibration_results,
}
with open(f"{SAVE_DIR}/stats_summary.json", "w") as f:
    json.dump(json_safe(stats_out), f, indent=2)
print(f"\nSaved: {SAVE_DIR}/stats_summary.json")

with open(f"{SAVE_DIR}/summary.json", "w") as f:
    json.dump(json_safe(summary), f, indent=2)
print(f"Saved: {SAVE_DIR}/summary.json")

print("\n✓ Training complete!  Run idh_cv_plot.py for figures.")
print(f"\nOutputs in {SAVE_DIR}/:")
print("  dataset_info.json")
print("  summary.json")
print("  stats_summary.json              ← Wilcoxon + DeLong + bootstrap CI + calibration")
print("  all_oof_predictions.csv         ← full-cohort ROC source")
print("  fold[1-5]_[model].json          ← metrics + predictions (no embeddings)")
print("  fold[1-5]_[model]_best.pth      ← model checkpoints")
print(f"\nOutputs in {SAVE_DIR}/embeddings/:")
print("  fold[1-5]_[model]_[scenario].npy  ← 512-d float32 embeddings")