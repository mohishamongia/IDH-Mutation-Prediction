def"""
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


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to trained model .pt checkpoint")
    parser.add_argument("--subject_id", required=True, help="Subject ID to analyze")
    parser.add_argument("--data_root", required=True, help="Root path to preprocessed subject volumes")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--true_label", type=int, default=None, help="0=wildtype, 1=mutant, if known")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading via importlib since the inference file's name has spaces
    # ("UCSF_PDGM Cross Site Eval.py"), which a plain `import` statement can't handle.
    import importlib.util
    candidate_paths = [
        "/workspace/UCSF_PDGM Cross Site Eval.py",
        os.path.join(os.getcwd(), "UCSF_PDGM Cross Site Eval.py"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "UCSF_PDGM Cross Site Eval.py"),
    ]
    INFERENCE_FILE_PATH = next((p for p in candidate_paths if os.path.exists(p)), None)
    if INFERENCE_FILE_PATH is None:
        raise FileNotFoundError(
            f"Could not find 'UCSF_PDGM Cross Site Eval.py' in: {candidate_paths}. "
            f"Edit candidate_paths above to point at its actual location."
        )
    spec = importlib.util.spec_from_file_location("ucsf_pdgm_eval_mod", INFERENCE_FILE_PATH)
    ucsf_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ucsf_mod)
    ResNet18_3D = ucsf_mod.ResNet18_3D
    UCSFPDGMDataset = ucsf_mod.UCSFPDGMDataset

    model = ResNet18_3D(in_channels=4, num_classes=2)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # Build a single-subject dataset instance just to reuse its loading/
    # preprocessing logic (zoom + min-max norm + correct T1/T1CE/T2/FLAIR order).
    # data_dirs must map subject_id -> the parent folder containing "<id>_nifti/"
    data_dirs = {args.subject_id: args.data_root}
    ds = UCSFPDGMDataset([args.subject_id], [args.true_label if args.true_label is not None else -1],
                         data_dirs, size=96, cache_dir=None)
    volume_4ch = ds.load_volume(args.subject_id)  # (4, 96, 96, 96) numpy array, already preprocessed

    # 1. Grad-CAM
    target_layer = model.layer4[-1]  # last block of layer4 -- adjust if your ResNet3D naming differs
    gradcam = GradCAM3D(model, target_layer)

    x = torch.from_numpy(volume_4ch).unsqueeze(0).float().to(device)
    x.requires_grad_(True)
    cam = gradcam.generate(x)

    gradcam_path = os.path.join(args.outdir, f"{args.subject_id}_gradcam_overlay.png")
    save_gradcam_overlay(volume_4ch, cam, gradcam_path)
    print(f"Saved Grad-CAM overlay -> {gradcam_path}")

    # 2. Per-modality occlusion
    importance_results, full_prob = per_modality_importance(model, volume_4ch, device, args.true_label)

    barplot_path = os.path.join(args.outdir, f"{args.subject_id}_modality_importance.png")
    save_importance_barplot(importance_results, barplot_path, args.subject_id)
    print(f"Saved modality importance barplot -> {barplot_path}")

    json_path = os.path.join(args.outdir, f"{args.subject_id}_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "subject_id": args.subject_id,
            "full_prediction_prob": round(full_prob, 4),
            "true_label": args.true_label,
            "modality_importance": importance_results,
        }, f, indent=2)
    print(f"Saved results JSON -> {json_path}")


if __name__ == "__main__":
    main()