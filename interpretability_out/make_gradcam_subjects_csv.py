"""
Auto-generates gradcam_subjects.csv for the batch modality-importance analysis.

Reuses your existing label-parsing + folder-matching logic from
"UCSF_PDGM Cross Site Eval.py" (fetch_ucsf_idh_labels + build_ucsf_subjects),
then randomly samples N mutant + N wildtype subjects and writes them to a CSV
with the exact columns Modality_gradcam.py expects: subject_id, true_label.

Usage:
    python make_gradcam_subjects_csv.py \
        --n_per_group 10 \
        --outdir /workspace/UCSF-PDGM-v5

Output:
    <outdir>/gradcam_subjects.csv
"""

import argparse
import os
import random
import csv
import importlib.util


def load_inference_module():
    candidate_paths = [
        "/workspace/UCSF_PDGM Cross Site Eval.py",
        os.path.join(os.getcwd(), "UCSF_PDGM Cross Site Eval.py"),
    ]
    path = next((p for p in candidate_paths if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(f"Could not find 'UCSF_PDGM Cross Site Eval.py' in: {candidate_paths}")
    spec = importlib.util.spec_from_file_location("ucsf_pdgm_eval_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_group", type=int, default=10,
                        help="How many mutant AND how many wildtype subjects to sample (default 10 each = 20 total)")
    parser.add_argument("--outdir", default="/workspace/UCSF-PDGM-v5")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # NOTE: importing the module runs its top-level CONFIG code but NOT main(),
    # since main() only executes inside `if __name__ == "__main__":` in that file.
    mod = load_inference_module()

    idh_labels = mod.fetch_ucsf_idh_labels(mod.UCSF_METADATA_CSV)
    subjects, labels, data_dirs = mod.build_ucsf_subjects(mod.UCSF_PDGM_DIR, idh_labels)

    mutant_ids = [s for s, l in zip(subjects, labels) if l == 1]
    wildtype_ids = [s for s, l in zip(subjects, labels) if l == 0]

    n_mut = min(args.n_per_group, len(mutant_ids))
    n_wt = min(args.n_per_group, len(wildtype_ids))
    if n_mut < args.n_per_group:
        print(f"WARNING: only {len(mutant_ids)} mutant subjects available, using all of them.")
    if n_wt < args.n_per_group:
        print(f"WARNING: only {len(wildtype_ids)} wildtype subjects available, using all of them.")

    sampled_mutant = random.sample(mutant_ids, n_mut)
    sampled_wildtype = random.sample(wildtype_ids, n_wt)

    out_path = os.path.join(args.outdir, "gradcam_subjects.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id", "true_label"])
        for sid in sampled_mutant:
            writer.writerow([sid, 1])
        for sid in sampled_wildtype:
            writer.writerow([sid, 0])

    print(f"\nWrote {n_mut} mutant + {n_wt} wildtype subjects -> {out_path}")


if __name__ == "__main__":
    main()