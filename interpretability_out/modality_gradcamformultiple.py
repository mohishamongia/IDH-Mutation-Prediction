"""
Modality Attribution Analysis for 3D ResNet-18 IDH Classifier
================================================================

Two complementary interpretability methods:

1. Grad-CAM (3D)        -> WHERE in the brain the model looks (spatial heatmap)
2. Per-modality ablation -> WHICH modality (T1/T1CE/T2/FLAIR) the model relies on

Run this on the DGX with the trained checkpoint(s). Designed to be dropped into
your existing training codebase with minimal changes -- see the "PLUG IN HERE"
comments below where you need to point it at your actual Dataset / Model classes.

Usage:
    python modality_gradcam_analysis.py \
        --checkpoint /workspace/checkpoints/baseline_fold0.pt \
        --subject_id <subject_id> \
        --data_root /workspace/data/UTSW_Glioma \
        --outdir /workspace/interpretability_out

Outputs per subject:
    <outdir>/<subject_id>_gradcam_overlay.png   -> Grad-CAM overlaid on axial/coronal/sagittal slices
    <outdir>/<subject_id>_modality_importance.png -> bar chart of per-modality AUC/logit drop
    <outdir>/<subject_id>_results.json           -> raw numbers
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

MODALITY_NAMES = ["T1", "T1CE", "T2", "FLAIR"]  # channel order -- MATCH THIS to your Dataset's channel order


# --------------------------------------------------------------------------- #
# 1. GRAD-CAM (3D)
# --------------------------------------------------------------------------- #
class GradCAM3D:
    """
    Standard Grad-CAM adapted for a 3D CNN. Hooks the last residual block
    (layer4 in a standard ResNet-18) since that's where the spatial feature
    maps still retain some resolution before global average pooling.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=1):
        """
        input_tensor: (1, 4, 96, 96, 96) -- single subject, 4 modality channels
        class_idx: which class's logit to backprop from. Model outputs 2 raw
                   logits (softmax over [wildtype, mutant]) -- class_idx=1 means
                   "explain the mutant-class score" (default, usually what you want).
        Returns a (96, 96, 96) heatmap normalized to [0, 1]
        """
        self.model.zero_grad()
        output = self.model(input_tensor)  # shape (1, 2) -- raw logits, softmax not yet applied

        score = output[:, class_idx].squeeze()
        score.backward(retain_graph=True)

        # activations/gradients shape: (1, C, D, H, W)
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)  # global-avg-pool the gradients -> channel weights
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # weighted sum over channels
        cam = F.relu(cam)

        # upsample CAM (currently at layer4's spatial resolution) back to input resolution
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="trilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def save_gradcam_overlay(volume_4ch, cam, out_path, modality_for_bg=2):
    """
    volume_4ch: (4, 96, 96, 96) numpy array, the input MRI volumes (already z-score normalized)
    cam: (96, 96, 96) Grad-CAM heatmap, normalized [0,1]
    modality_for_bg: which channel to show as the grayscale background (default: T2, index 2)
    Saves a 3-panel (axial/coronal/sagittal) mid-slice overlay figure.
    """
    bg = volume_4ch[modality_for_bg]
    mid = np.array(bg.shape) // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    slices_bg = [bg[mid[0], :, :], bg[:, mid[1], :], bg[:, :, mid[2]]]
    slices_cam = [cam[mid[0], :, :], cam[:, mid[1], :], cam[:, :, mid[2]]]
    titles = ["Axial", "Coronal", "Sagittal"]

    for ax, s_bg, s_cam, title in zip(axes, slices_bg, slices_cam, titles):
        ax.imshow(s_bg, cmap="gray")
        ax.imshow(s_cam, cmap="jet", alpha=0.45)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 2. PER-MODALITY OCCLUSION IMPORTANCE
# --------------------------------------------------------------------------- #
@torch.no_grad()
def per_modality_importance(model, volume_4ch, device, true_label=None):
    """
    volume_4ch: (4, 96, 96, 96) numpy array for ONE subject
    Zeroes each modality channel one at a time (same as your structured dropout
    zero-fill convention) and measures the change in predicted probability.

    Returns dict: {modality_name: {"full_prob":.., "ablated_prob":.., "delta":..}}
    """
    model.eval()
    x_full = torch.from_numpy(volume_4ch).unsqueeze(0).float().to(device)  # (1,4,96,96,96)

    full_logits = model(x_full)  # (1, 2) raw logits -> [wildtype, mutant]
    full_prob = torch.softmax(full_logits, dim=1)[0, 1].item()  # P(mutant)

    results = {}
    for i, name in enumerate(MODALITY_NAMES):
        x_ablated = x_full.clone()
        x_ablated[:, i, :, :, :] = 0.0  # zero-fill, matches your training-time missing-modality convention

        ablated_logits = model(x_ablated)
        ablated_prob = torch.softmax(ablated_logits, dim=1)[0, 1].item()

        results[name] = {
            "full_prob": round(full_prob, 4),
            "ablated_prob": round(ablated_prob, 4),
            "delta": round(full_prob - ablated_prob, 4),  # positive delta = modality was pushing prob UP; removing it drops confidence
            "abs_delta": round(abs(full_prob - ablated_prob), 4),  # magnitude of reliance, regardless of direction
        }

    if true_label is not None:
        for name in results:
            results[name]["true_label"] = true_label

    return results, full_prob


def save_importance_barplot(results, out_path, subject_id):
    names = list(results.keys())
    abs_deltas = [results[n]["abs_delta"] for n in names]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, abs_deltas, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    ax.set_ylabel("|Δ predicted probability| when modality removed")
    ax.set_title(f"Per-modality importance -- subject {subject_id}")
    for bar, val in zip(bars, abs_deltas):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f"{val:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def load_model_and_dataset_classes(checkpoint_path, device):
    """
    Loads the ResNet18_3D model from checkpoint_path, plus returns the
    UCSFPDGMDataset class -- both pulled from your existing inference file
    via importlib (handles the space in the filename).
    """
    import importlib.util
    candidate_paths = [
        "/workspace/UCSF_PDGM Cross Site Eval.py",
        os.path.join(os.getcwd(), "UCSF_PDGM Cross Site Eval.py"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "UCSF_PDGM Cross Site Eval.py"),
    ]
    inference_file_path = next((p for p in candidate_paths if os.path.exists(p)), None)
    if inference_file_path is None:
        raise FileNotFoundError(
            f"Could not find 'UCSF_PDGM Cross Site Eval.py' in: {candidate_paths}. "
            f"Edit candidate_paths above to point at its actual location."
        )
    spec = importlib.util.spec_from_file_location("ucsf_pdgm_eval_mod", inference_file_path)
    ucsf_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ucsf_mod)
    ResNet18_3D = ucsf_mod.ResNet18_3D
    UCSFPDGMDataset = ucsf_mod.UCSFPDGMDataset

    model = ResNet18_3D(in_channels=4, num_classes=2)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, UCSFPDGMDataset


def analyze_single_subject(model, UCSFPDGMDataset, subject_id, true_label, data_root, outdir, device,
                            save_gradcam=True):
    """Runs Grad-CAM + per-modality ablation for one subject. Returns the results dict."""
    data_dirs = {subject_id: data_root}
    ds = UCSFPDGMDataset([subject_id], [true_label if true_label is not None else -1],
                         data_dirs, size=96, cache_dir=None)
    volume_4ch = ds.load_volume(subject_id)

    if save_gradcam:
        target_layer = model.layer4[-1]
        gradcam = GradCAM3D(model, target_layer)
        x = torch.from_numpy(volume_4ch).unsqueeze(0).float().to(device)
        x.requires_grad_(True)
        cam = gradcam.generate(x)
        gradcam_path = os.path.join(outdir, f"{subject_id}_gradcam_overlay.png")
        save_gradcam_overlay(volume_4ch, cam, gradcam_path)

    importance_results, full_prob = per_modality_importance(model, volume_4ch, device, true_label)

    barplot_path = os.path.join(outdir, f"{subject_id}_modality_importance.png")
    save_importance_barplot(importance_results, barplot_path, subject_id)

    json_path = os.path.join(outdir, f"{subject_id}_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "subject_id": subject_id,
            "full_prediction_prob": round(full_prob, 4),
            "true_label": true_label,
            "modality_importance": importance_results,
        }, f, indent=2)

    return importance_results, full_prob


def run_batch(model, UCSFPDGMDataset, subjects_csv, data_root, outdir, device,
              id_col="subject_id", label_col="true_label", save_gradcam_all=False):
    """
    subjects_csv must have at least: subject_id, true_label (0=wildtype, 1=mutant)
    Runs per-modality ablation (and optionally Grad-CAM) for every row, then
    aggregates mean +/- std importance per modality, split by true_label group.
    """
    import pandas as pd

    df = pd.read_csv(subjects_csv)
    rows = []
    print(f"Running batch analysis on {len(df)} subjects...\n")

    for _, row in df.iterrows():
        sid = str(row[id_col])
        label = int(row[label_col])
        try:
            importance_results, full_prob = analyze_single_subject(
                model, UCSFPDGMDataset, sid, label, data_root, outdir, device,
                save_gradcam=save_gradcam_all
            )
            for modality, vals in importance_results.items():
                rows.append({
                    "subject_id": sid,
                    "true_label": label,
                    "modality": modality,
                    "abs_delta": vals["abs_delta"],
                    "delta": vals["delta"],
                })
            print(f"  [OK] {sid} (label={label}, pred_prob={full_prob:.3f})")
        except Exception as e:
            print(f"  [SKIP] {sid} -- {e}")

    long_df = pd.DataFrame(rows)
    long_csv_path = os.path.join(outdir, "batch_modality_importance_long.csv")
    long_df.to_csv(long_csv_path, index=False)
    print(f"\nSaved per-subject long-format results -> {long_csv_path}")

    # Aggregate: mean +/- std abs_delta per modality, split by true_label group
    summary = long_df.groupby(["true_label", "modality"])["abs_delta"].agg(["mean", "std", "count"]).reset_index()
    summary_csv_path = os.path.join(outdir, "batch_modality_importance_summary.csv")
    summary.to_csv(summary_csv_path, index=False)
    print(f"Saved group summary -> {summary_csv_path}")

    # Grouped bar plot: mutant vs wildtype, per modality
    _save_grouped_summary_plot(summary, os.path.join(outdir, "batch_modality_importance_summary.png"))
    print(f"Saved summary plot -> {os.path.join(outdir, 'batch_modality_importance_summary.png')}")

    return long_df, summary


def _save_grouped_summary_plot(summary_df, out_path):
    modalities = MODALITY_NAMES
    label_names = {0: "Wildtype", 1: "Mutant"}
    x = np.arange(len(modalities))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, label_val in enumerate(sorted(summary_df["true_label"].unique())):
        sub = summary_df[summary_df["true_label"] == label_val].set_index("modality").reindex(modalities)
        means = sub["mean"].values
        stds = sub["std"].fillna(0).values
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               label=label_names.get(label_val, str(label_val)))

    ax.set_xticks(x)
    ax.set_xticklabels(modalities)
    ax.set_ylabel("Mean |Δ predicted probability| when modality removed")
    ax.set_title("Per-modality importance by IDH status (group mean ± SD)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to trained model .pt checkpoint")
    parser.add_argument("--data_root", required=True, help="Root path to preprocessed subject volumes")
    parser.add_argument("--outdir", required=True)

    # single-subject mode
    parser.add_argument("--subject_id", default=None, help="Subject ID to analyze (single-subject mode)")
    parser.add_argument("--true_label", type=int, default=None, help="0=wildtype, 1=mutant, if known")

    # batch mode
    parser.add_argument("--subjects_csv", default=None,
                        help="CSV with columns subject_id,true_label -- runs batch analysis over all rows")
    parser.add_argument("--id_col", default="subject_id")
    parser.add_argument("--label_col", default="true_label")
    parser.add_argument("--save_gradcam_all", action="store_true",
                        help="Also save Grad-CAM overlay PNGs for every subject in batch mode (slower)")

    args = parser.parse_args()

    if not args.subject_id and not args.subjects_csv:
        parser.error("Provide either --subject_id (single subject) or --subjects_csv (batch mode).")

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, UCSFPDGMDataset = load_model_and_dataset_classes(args.checkpoint, device)

    if args.subjects_csv:
        run_batch(model, UCSFPDGMDataset, args.subjects_csv, args.data_root, args.outdir, device,
                  id_col=args.id_col, label_col=args.label_col, save_gradcam_all=args.save_gradcam_all)
    else:
        importance_results, full_prob = analyze_single_subject(
            model, UCSFPDGMDataset, args.subject_id, args.true_label, args.data_root, args.outdir, device,
            save_gradcam=True
        )
        print(f"Saved Grad-CAM overlay, importance barplot, and results JSON to {args.outdir}")
        print(f"Predicted P(mutant) = {full_prob:.4f}")


if __name__ == "__main__":
    main()