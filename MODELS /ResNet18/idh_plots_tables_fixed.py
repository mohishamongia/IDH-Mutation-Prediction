"""idh_plots_tables.py
===================
Publication-quality figures + tables for:
  "Missing-Modality Robust IDH Prediction — UTSW-Glioma"

Figures produced
----------------
  Fig 1 — ROC curves (per model, per scenario)
  Fig 2 — Robustness degradation curve (AUC vs #modalities)
  Fig 3 — Calibration curves (OOF predictions)
  Fig 4 — t-SNE / UMAP of fold embeddings
  Fig 5 — Subgroup bar charts (scanner + tumor grade)

Tables produced
---------------
  Table 1 — Full metric table (AUC ± SD, BalAcc, Se, Sp, F1) per model × scenario  [CSV + LaTeX]
  Table 2 — Wilcoxon pairwise significance                                          [CSV + LaTeX]
  Table 3 — Bootstrap AUC CI + Brier score                                          [CSV + LaTeX]
  Table 4 — Scanner subgroup AUC                                                    [CSV + LaTeX]
  Table 5 — Tumor-grade subgroup AUC                                                [CSV + LaTeX]

Usage
-----
  python idh_plots_tables.py [--results_dir results] [--cache_dir cache_npy]
                             [--out_dir paper_figures] [--no_umap] [--folds 5]

Dependencies
------------
  pip install matplotlib seaborn scikit-learn pandas numpy umap-learn tqdm
  (umap-learn is optional — skipped gracefully if absent)
"""

import os, json, argparse, warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc as sklearn_auc
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

# ── UMAP optional ─────────────────────────────────────────────────────────────
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[WARN] umap-learn not installed — UMAP panel will be skipped.")

# ─── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", default="results")
parser.add_argument("--cache_dir",   default="cache_npy")
parser.add_argument("--out_dir",     default="paper_figures")
parser.add_argument("--no_umap",     action="store_true")
parser.add_argument("--folds",       type=int, default=5)
args = parser.parse_args()

RESULTS = Path(args.results_dir)
OUT     = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "tables").mkdir(exist_ok=True)
N_FOLDS = args.folds

# ─── STYLE ────────────────────────────────────────────────────────────────────
PALETTE = {
    "Baseline_ResNet18" : "#2166ac",   # blue
    "Dropout30_ResNet18": "#1a9641",   # green
    "Dropout50_ResNet18": "#d7191c",   # red
}
LINESTYLE = {
    "Baseline_ResNet18" : "-",
    "Dropout30_ResNet18": "--",
    "Dropout50_ResNet18": ":",
}
MODEL_LABELS = {
    "Baseline_ResNet18" : "Baseline",
    "Dropout30_ResNet18": "Dropout-30",
    "Dropout50_ResNet18": "Dropout-50",
}
SCENARIO_LABELS = {
    "All_4"         : "All 4",
    "T1CE_missing"  : "No T1CE",
    "FLAIR_missing" : "No FLAIR",
    "T2_missing"    : "No T2",
    "T1_T2_only"    : "T1+T2 only",
    "T1_only"       : "T1 only",
}
SCENARIO_N_MOD = {          # number of available modalities per scenario
    "All_4"         : 4,
    "T1CE_missing"  : 3,
    "FLAIR_missing" : 3,
    "T2_missing"    : 3,
    "T1_T2_only"    : 2,
    "T1_only"       : 1,
}
SCENARIO_ORDER = list(SCENARIO_LABELS.keys())
MODEL_ORDER    = list(PALETTE.keys())

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "font.size"        : 10,
    "axes.titlesize"   : 11,
    "axes.labelsize"   : 10,
    "xtick.labelsize"  : 9,
    "ytick.labelsize"  : 9,
    "legend.fontsize"  : 9,
    "figure.dpi"       : 150,
    "savefig.dpi"      : 300,
    "savefig.bbox"     : "tight",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
def load_json(path):
    with open(path) as f:
        return json.load(f)

print("Loading summary files …")
summary      = load_json(RESULTS / "summary.json")
stats        = load_json(RESULTS / "stats_summary.json")
dataset_info = load_json(RESULTS / "dataset_info.json")
oof_df       = pd.read_csv(RESULTS / "all_oof_predictions.csv")

# Load per-fold JSONs for ROC raw data + subgroup info
fold_data = {}   # model → fold_n → dict
for model in MODEL_ORDER:
    fold_data[model] = {}
    for f in range(1, N_FOLDS + 1):
        fp = RESULTS / f"fold{f}_{model}.json"
        if fp.exists():
            fold_data[model][f] = load_json(fp)

scanner_map = dataset_info.get("scanner_info", {})
grade_map   = dataset_info.get("grade_info",   {})

print("Data loaded.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def mean_std(model, scenario):
    d = summary.get(model, {}).get(scenario, {})
    m = d.get("mean"); s = d.get("std")
    return m, s

def get_fold_aucs(model, scenario):
    return summary.get(model, {}).get(scenario, {}).get("fold_aucs", [])

def save_fig(fig, name, tight=True):
    path = OUT / name
    fig.savefig(path, dpi=300, bbox_inches="tight" if tight else None)
    print(f"  ✓ {path}")
    plt.close(fig)

def latex_table(df, caption, label, col_format=None):
    """Return LaTeX booktabs table string."""
    if col_format is None:
        col_format = "l" + "c" * (len(df.columns))
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{" + col_format + "}",
        r"\toprule",
    ]
    # header
    lines.append(" & ".join([""] + list(df.columns)) + r" \\")
    lines.append(r"\midrule")
    for idx, row in df.iterrows():
        cells = [str(idx)] + [str(v) for v in row.values]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)

def save_table(df, name, caption, label):
    csv_path = OUT / "tables" / f"{name}.csv"
    tex_path = OUT / "tables" / f"{name}.tex"
    df.to_csv(csv_path)
    with open(tex_path, "w") as f:
        f.write(latex_table(df, caption, label))
    print(f"  ✓ {csv_path}")
    print(f"  ✓ {tex_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — ROC CURVES
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 1 — ROC curves …")

n_sc  = len(SCENARIO_ORDER)
ncols = 3
nrows = (n_sc + ncols - 1) // ncols   # = 2

fig, axes = plt.subplots(nrows, ncols, figsize=(14, 9))
axes = axes.flatten()

for si, sc_name in enumerate(SCENARIO_ORDER):
    ax = axes[si]
    ax.plot([0,1],[0,1],"k--", lw=0.8, alpha=0.5)

    for model in MODEL_ORDER:
        fpr_list, tpr_list = [], []
        for fn, fdata in fold_data[model].items():
            rc = fdata["scenario_results"][sc_name]["roc_curve"]
            if len(rc["fpr"]) < 2:
                continue
            fpr_list.append(np.array(rc["fpr"]))
            tpr_list.append(np.array(rc["tpr"]))

        if not fpr_list:
            continue

        # Interpolate to common FPR grid and average
        mean_fpr = np.linspace(0, 1, 200)
        interp_tprs = [np.interp(mean_fpr, fpr, tpr)
                       for fpr, tpr in zip(fpr_list, tpr_list)]
        mean_tpr = np.mean(interp_tprs, axis=0)
        std_tpr  = np.std(interp_tprs,  axis=0)

        m, s = mean_std(model, sc_name)
        lbl  = f"{MODEL_LABELS[model]}  AUC={m:.3f}±{s:.3f}" if (m and s) else MODEL_LABELS[model]

        ax.plot(mean_fpr, mean_tpr,
                color=PALETTE[model], ls=LINESTYLE[model], lw=1.8, label=lbl)
        ax.fill_between(mean_fpr,
                         np.clip(mean_tpr - std_tpr, 0, 1),
                         np.clip(mean_tpr + std_tpr, 0, 1),
                         color=PALETTE[model], alpha=0.10)

    ax.set_title(SCENARIO_LABELS[sc_name], fontweight="bold")
    ax.set_xlabel("1 − Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_xlim([0,1]); ax.set_ylim([0,1])
    ax.legend(fontsize=7.5, loc="lower right")

# Hide unused axes
for ax in axes[n_sc:]:
    ax.set_visible(False)

fig.suptitle("ROC Curves — Missing-Modality IDH Prediction\n(5-fold CV, shaded = ±1 SD)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
save_fig(fig, "fig1_roc_curves.pdf")
save_fig(fig, "fig1_roc_curves.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — ROBUSTNESS DEGRADATION CURVE
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 2 — Robustness degradation curve …")

# For 3-modality scenarios, average AUC across all three
def robustness_series(model):
    """Returns {n_mod: mean_AUC} averaged over scenarios with same n_mod."""
    bucket = {}
    for sc in SCENARIO_ORDER:
        n  = SCENARIO_N_MOD[sc]
        m, _ = mean_std(model, sc)
        if m is not None:
            bucket.setdefault(n, []).append(m)
    return {k: np.mean(v) for k, v in bucket.items()}

fig, ax = plt.subplots(figsize=(7, 5))

for model in MODEL_ORDER:
    series = robustness_series(model)
    xs = sorted(series.keys())
    ys = [series[x] for x in xs]
    ax.plot(xs, ys, "o-",
            color=PALETTE[model], ls=LINESTYLE[model],
            lw=2.2, ms=7, label=MODEL_LABELS[model])

    # Individual scenario points (smaller, semi-transparent)
    for sc in SCENARIO_ORDER:
        n = SCENARIO_N_MOD[sc]
        m, _ = mean_std(model, sc)
        if m is not None:
            ax.scatter(n, m, color=PALETTE[model], s=18, alpha=0.45, zorder=3)

ax.set_xlabel("Number of Available Modalities")
ax.set_ylabel("Mean AUC (5-fold CV)")
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(["1\n(T1 only)", "2\n(T1+T2)", "3\n(one missing)", "4\n(all)"])
ax.set_ylim([0.45, 1.0])
ax.set_title("Robustness Degradation: AUC vs. Available Modalities",
             fontweight="bold")
ax.legend(title="Model", frameon=True)
ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.6, label="Chance")
plt.tight_layout()
save_fig(fig, "fig2_robustness_curve.pdf")
save_fig(fig, "fig2_robustness_curve.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — CALIBRATION CURVES
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 3 — Calibration curves …")

fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(15, 5), sharey=True)

for mi, model in enumerate(MODEL_ORDER):
    ax  = axes[mi]
    sub = oof_df[oof_df["model"] == model]

    if len(sub) == 0:
        ax.set_visible(False)
        continue

    y_true = sub["true_label"].values
    y_prob = sub["pred_prob"].values

    # Calibration curve
    frac_pos, mean_pred = calibration_curve(y_true, y_prob,
                                            n_bins=10, strategy="uniform")
    ax.plot(mean_pred, frac_pos, "s-",
            color=PALETTE[model], lw=2, ms=6, label="Model")
    ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.6, label="Perfect")
    ax.fill_between([0,1],[0,1],[0,1], alpha=0)

    # Histogram of predicted probabilities
    ax2 = ax.twinx()
    ax2.hist(y_prob[y_true==0], bins=20, alpha=0.25,
             color="#377eb8", label="WT")
    ax2.hist(y_prob[y_true==1], bins=20, alpha=0.25,
             color="#e41a1c", label="Mutant")
    ax2.set_ylabel("Count" if mi == len(MODEL_ORDER)-1 else "")
    ax2.set_ylim([0, len(sub)])

    # Brier score
    bs = stats["bootstrap_oof"].get(model, {}).get("brier_score", None)
    bs_txt = f"Brier={bs:.3f}" if bs is not None else ""
    ax.set_title(f"{MODEL_LABELS[model]}\n{bs_txt}", fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability")
    if mi == 0:
        ax.set_ylabel("Fraction of Positives")
    ax.set_xlim([0,1]); ax.set_ylim([0,1])
    ax.legend(loc="upper left", fontsize=8)

fig.suptitle("Calibration Curves — Pooled OOF Predictions\n(Histogram: WT=blue, Mutant=red)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
save_fig(fig, "fig3_calibration.pdf")
save_fig(fig, "fig3_calibration.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — t-SNE / UMAP  (All_4 scenario, all folds concatenated)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 4 — t-SNE / UMAP embeddings …")

def load_embeddings_for_scenario(model, scenario="All_4"):
    """
    Load fold embeddings + per-subject labels from fold JSONs.
    Returns (emb_array [N,512], labels [N], subject_ids [N]).
    """
    embs, labs, sids = [], [], []
    emb_dir = RESULTS / "embeddings"
    for fn in range(1, N_FOLDS + 1):
        npy_path = emb_dir / f"fold{fn}_{model}_{scenario}.npy"
        json_path = RESULTS / f"fold{fn}_{model}.json"
        if not (npy_path.exists() and json_path.exists()):
            continue
        emb = np.load(npy_path)          # (n_val, 512)
        fd  = load_json(json_path)
        ps  = fd["scenario_results"][scenario]["per_subject"]
        y   = [p["true_label"] for p in ps]
        sid = [p["subject_id"] for p in ps]
        if len(emb) != len(y):
            print(f"  [WARN] Shape mismatch fold{fn} {model}: emb={len(emb)} labels={len(y)}")
            continue
        embs.append(emb); labs.extend(y); sids.extend(sid)
    if not embs:
        return None, None, None
    return np.concatenate(embs, axis=0), np.array(labs), sids

DO_UMAP = HAS_UMAP and not args.no_umap
n_panels = 2 if DO_UMAP else 1
n_rows   = len(MODEL_ORDER)
n_cols   = n_panels

fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(6 * n_cols, 4.5 * n_rows))
if n_rows == 1:
    axes = axes[np.newaxis, :]
if n_cols == 1:
    axes = axes[:, np.newaxis]

for ri, model in enumerate(MODEL_ORDER):
    emb, lab, _ = load_embeddings_for_scenario(model, "All_4")

    if emb is None:
        print(f"  [WARN] No embeddings found for {model} — skipping")
        for ci in range(n_cols):
            axes[ri, ci].set_visible(False)
        continue

    print(f"  {model}: {emb.shape[0]} subjects loaded")

    # Sub-sample if very large for speed (>1000)
    if emb.shape[0] > 1000:
        idx = np.random.choice(emb.shape[0], 1000, replace=False)
        emb_s, lab_s = emb[idx], lab[idx]
    else:
        emb_s, lab_s = emb, lab

    colors = ["#377eb8" if l == 0 else "#e41a1c" for l in lab_s]

    # t-SNE
    print(f"    Running t-SNE …")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    Z    = tsne.fit_transform(emb_s)

    ax = axes[ri, 0]
    ax.scatter(Z[lab_s==0, 0], Z[lab_s==0, 1], c="#377eb8", s=12,
               alpha=0.65, label="WT", edgecolors="none")
    ax.scatter(Z[lab_s==1, 0], Z[lab_s==1, 1], c="#e41a1c", s=12,
               alpha=0.65, label="Mutant", edgecolors="none")
    ax.set_title(f"{MODEL_LABELS[model]} — t-SNE", fontweight="bold")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=2, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

    # UMAP
    if DO_UMAP:
        print(f"    Running UMAP …")
        reducer = umap.UMAP(n_components=2, random_state=42)
        U       = reducer.fit_transform(emb_s)

        ax2 = axes[ri, 1]
        ax2.scatter(U[lab_s==0, 0], U[lab_s==0, 1], c="#377eb8", s=12,
                    alpha=0.65, label="WT", edgecolors="none")
        ax2.scatter(U[lab_s==1, 0], U[lab_s==1, 1], c="#e41a1c", s=12,
                    alpha=0.65, label="Mutant", edgecolors="none")
        ax2.set_title(f"{MODEL_LABELS[model]} — UMAP", fontweight="bold")
        ax2.set_xlabel("UMAP 1"); ax2.set_ylabel("UMAP 2")
        ax2.legend(markerscale=2, fontsize=8)
        ax2.set_xticks([]); ax2.set_yticks([])

fig.suptitle("Embedding Visualisation — All_4 Scenario (OOF validation subjects)\n"
             "Blue = IDH WT · Red = IDH Mutant",
             fontsize=12, fontweight="bold")
plt.tight_layout()
save_fig(fig, "fig4_tsne_umap.pdf")
save_fig(fig, "fig4_tsne_umap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — SUBGROUP ANALYSIS  (scanner + tumor grade)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 5 — Subgroup analysis …")

def build_subgroup_auc(group_map, group_name):
    """
    group_map: {subject_id: group_label}
    Returns DataFrame: rows=group, cols=model, values=AUC
    """
    from sklearn.metrics import roc_auc_score

    groups  = sorted(set(v for v in group_map.values() if v is not None))
    records = {}

    for model in MODEL_ORDER:
        sub = oof_df[oof_df["model"] == model].copy()
        sub[group_name] = sub["subject_id"].map(group_map)
        sub = sub.dropna(subset=[group_name])
        col_aucs = {}
        for g in groups:
            grp = sub[sub[group_name] == g]
            if len(grp) < 5 or grp["true_label"].nunique() < 2:
                col_aucs[g] = np.nan
            else:
                try:
                    col_aucs[g] = roc_auc_score(grp["true_label"], grp["pred_prob"])
                except Exception:
                    col_aucs[g] = np.nan
        records[MODEL_LABELS[model]] = col_aucs

    return pd.DataFrame(records, index=groups)

scanner_df = build_subgroup_auc(scanner_map, "scanner")
grade_map2 = {k: (f"Grade {int(v)}" if v is not None else None)
              for k, v in grade_map.items()}
grade_df   = build_subgroup_auc(grade_map2, "grade")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, df, title in [
    (axes[0], scanner_df, "AUC by Scanner"),
    (axes[1], grade_df,   "AUC by Tumor Grade"),
]:
    df_clean = df.dropna(how="all")
    x        = np.arange(len(df_clean))
    width    = 0.25
    for bi, model_lbl in enumerate(df_clean.columns):
        vals = df_clean[model_lbl].values
        color = list(PALETTE.values())[bi]
        ax.bar(x + bi * width, vals, width,
               label=model_lbl, color=color, alpha=0.85, edgecolor="white")

    ax.set_xticks(x + width)
    ax.set_xticklabels(df_clean.index, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("AUC")
    ax.set_ylim([0, 1.05])
    ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.set_title(title, fontweight="bold")
    ax.legend(title="Model", fontsize=8)

fig.suptitle("Subgroup AUC — Pooled OOF Predictions", fontsize=12, fontweight="bold")
plt.tight_layout()
save_fig(fig, "fig5_subgroups.pdf")
save_fig(fig, "fig5_subgroups.png")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 1 — Full metric table
# ═══════════════════════════════════════════════════════════════════════════════
print("Table 1 — Full metric table …")

rows = {}
for model in MODEL_ORDER:
    for sc in SCENARIO_ORDER:
        key = (MODEL_LABELS[model], SCENARIO_LABELS[sc])
        fd_list = [fold_data[model][fn]
                   for fn in range(1, N_FOLDS+1)
                   if fn in fold_data[model]]
        if not fd_list:
            rows[key] = {m: "–" for m in ["AUC","BalAcc","Sensitivity","Specificity","F1"]}
            continue

        collect = {m: [] for m in ["AUC","BalAcc","Sensitivity","Specificity","F1"]}
        for fd in fd_list:
            met = fd["scenario_results"][sc]["metrics"]
            for k in collect:
                v = met.get(k)
                if v is not None:
                    collect[k].append(v)

        row = {}
        for k, vals in collect.items():
            if vals:
                row[k] = f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"
            else:
                row[k] = "–"
        rows[key] = row

idx  = pd.MultiIndex.from_tuples(rows.keys(), names=["Model","Scenario"])
tab1 = pd.DataFrame(rows.values(), index=idx)
save_table(tab1, "table1_full_metrics",
           "Mean ± SD performance across 5-fold CV for each model and modality scenario.",
           "tab:metrics")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 2 — Wilcoxon pairwise significance
# ═══════════════════════════════════════════════════════════════════════════════
print("Table 2 — Wilcoxon significance …")

wilcoxon = stats.get("wilcoxon_per_scenario", {})
wrows = {}
for sc in SCENARIO_ORDER:
    sc_d = wilcoxon.get(sc, {})
    for pair, res in sc_d.items():
        key = (SCENARIO_LABELS[sc], pair.replace("_vs_", " vs "))
        wrows[key] = {
            "Statistic": res.get("statistic", "–"),
            "p-value"  : res.get("p_value",   "–"),
            "Sig."     : res.get("sig",        res.get("note","–")),
        }

idx2  = pd.MultiIndex.from_tuples(wrows.keys(), names=["Scenario","Pair"])
tab2  = pd.DataFrame(wrows.values(), index=idx2)
save_table(tab2, "table2_wilcoxon",
           "Pairwise Wilcoxon signed-rank test (fold-wise AUC, two-sided).",
           "tab:wilcoxon")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 3 — Bootstrap AUC CI + Brier score
# ═══════════════════════════════════════════════════════════════════════════════
print("Table 3 — Bootstrap CI + Brier …")

boot = stats.get("bootstrap_oof", {})
brows = {}
for model in MODEL_ORDER:
    d   = boot.get(model, {})
    ci  = d.get("ci_95", [None, None])
    brows[MODEL_LABELS[model]] = {
        "Pooled AUC"    : d.get("pooled_auc", "–"),
        "95% CI (Boot)" : f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci[0] is not None else "–",
        "Brier Score"   : d.get("brier_score", "–"),
    }

tab3 = pd.DataFrame(brows).T
save_table(tab3, "table3_bootstrap",
           "Pooled OOF AUC with 1000-iteration stratified bootstrap 95\\% CI and Brier score.",
           "tab:bootstrap")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 4 — Scanner subgroup AUC
# ═══════════════════════════════════════════════════════════════════════════════
print("Table 4 — Scanner subgroup …")

scanner_df.index.name = "Scanner"
scanner_df_r = scanner_df.apply(lambda x: x.map(lambda v: f"{v:.3f}" if not np.isnan(v) else "–"))
save_table(scanner_df_r, "table4_scanner_subgroup",
           "AUC by scanner (pooled OOF, All\\_4 scenario).",
           "tab:scanner")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 5 — Tumor-grade subgroup AUC
# ═══════════════════════════════════════════════════════════════════════════════
print("Table 5 — Tumor grade subgroup …")

grade_df.index.name = "Grade"
grade_df_r = grade_df.apply(lambda x: x.map(lambda v: f"{v:.3f}" if not np.isnan(v) else "–"))
save_table(grade_df_r, "table5_grade_subgroup",
           "AUC by tumor grade (pooled OOF, All\\_4 scenario).",
           "tab:grade")


# ═══════════════════════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════════════════════
print(f"""
╔══════════════════════════════════════════════════════╗
║  idh_plots_tables.py — Complete                     ║
╠══════════════════════════════════════════════════════╣
║  Figures → {str(OUT):<40} ║
║  Tables  → {str(OUT / 'tables'):<40} ║
╚══════════════════════════════════════════════════════╝

Figures
  fig1_roc_curves.pdf/png
  fig2_robustness_curve.pdf/png
  fig3_calibration.pdf/png
  fig4_tsne_umap.pdf/png
  fig5_subgroups.pdf/png

Tables (CSV + LaTeX .tex)
  table1_full_metrics
  table2_wilcoxon
  table3_bootstrap
  table4_scanner_subgroup
  table5_grade_subgroup
""")