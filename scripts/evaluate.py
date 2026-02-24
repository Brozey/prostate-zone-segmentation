"""
Evaluate nnU-Net predictions against ground-truth segmentation masks.

Computes per-case and per-dataset Dice scores for TZ (label 1) and PZ (label 2).

Usage:
    python scripts/evaluate.py \
        --predictions /path/to/predictions \
        --ground-truth /path/to/labelsTs \
        --id-mapping /path/to/id_mapping.json
"""

import argparse
import json
import os
from collections import defaultdict

import nibabel as nib
import numpy as np


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice similarity coefficient for binary masks."""
    intersection = np.sum(pred * gt)
    denom = np.sum(pred) + np.sum(gt)
    if denom == 0:
        return 1.0 if np.sum(pred) == 0 else 0.0
    return 2.0 * intersection / denom


def main():
    parser = argparse.ArgumentParser(description="Evaluate prostate zone segmentation.")
    parser.add_argument("--predictions", required=True, help="Folder with predicted .nii.gz masks.")
    parser.add_argument("--ground-truth", required=True, help="Folder with ground-truth .nii.gz masks.")
    parser.add_argument("--id-mapping", default=None, help="JSON mapping original IDs → nnU-Net case names.")
    parser.add_argument("--output", default=None, help="Save results to JSON file.")
    args = parser.parse_args()

    # Reverse mapping for readable output
    rev_mapping = {}
    if args.id_mapping and os.path.exists(args.id_mapping):
        with open(args.id_mapping) as f:
            fwd = json.load(f)
        rev_mapping = {v: k for k, v in fwd.items()}

    results = []

    for fname in sorted(os.listdir(args.predictions)):
        if not fname.endswith(".nii.gz"):
            continue
        case = fname.replace(".nii.gz", "")
        gt_path = os.path.join(args.ground_truth, fname)
        if not os.path.exists(gt_path):
            print(f"  SKIP {case}: no matching ground truth")
            continue

        pred = nib.load(os.path.join(args.predictions, fname)).get_fdata().astype(np.int16)
        gt = nib.load(gt_path).get_fdata().astype(np.int16)

        tz = dice_score((pred == 1).astype(np.uint8), (gt == 1).astype(np.uint8))
        pz = dice_score((pred == 2).astype(np.uint8), (gt == 2).astype(np.uint8))
        mean = (tz + pz) / 2.0

        original_id = rev_mapping.get(case, case)

        # Determine source dataset
        if original_id.startswith("Prostate158"):
            domain = "Prostate158"
        elif original_id.startswith("MSD"):
            domain = "MSD"
        elif original_id.startswith("BMC"):
            domain = "BMC"
        else:
            domain = "Other"

        results.append(dict(id=original_id, domain=domain, tz=tz, pz=pz, mean=mean))
        print(f"  {original_id:<35} TZ={tz:.4f}  PZ={pz:.4f}  Mean={mean:.4f}")

    if not results:
        print("No cases evaluated.")
        return

    # Per-domain summary
    domain_agg = defaultdict(lambda: dict(tz=[], pz=[], mean=[]))
    for r in results:
        domain_agg[r["domain"]]["tz"].append(r["tz"])
        domain_agg[r["domain"]]["pz"].append(r["pz"])
        domain_agg[r["domain"]]["mean"].append(r["mean"])

    print(f"\n{'='*65}")
    print(f"  {'Dataset':<15} {'N':>4} {'TZ Dice':>10} {'PZ Dice':>10} {'Mean':>10}")
    print(f"  {'-'*55}")

    all_tz, all_pz, all_mean = [], [], []
    for domain in ["Prostate158", "MSD", "BMC", "Other"]:
        if domain not in domain_agg:
            continue
        d = domain_agg[domain]
        n = len(d["tz"])
        print(f"  {domain:<15} {n:>4} {np.mean(d['tz']):>10.4f} {np.mean(d['pz']):>10.4f} {np.mean(d['mean']):>10.4f}")
        all_tz.extend(d["tz"])
        all_pz.extend(d["pz"])
        all_mean.extend(d["mean"])

    print(f"  {'-'*55}")
    print(f"  {'OVERALL':<15} {len(all_tz):>4} {np.mean(all_tz):>10.4f} {np.mean(all_pz):>10.4f} {np.mean(all_mean):>10.4f}")

    # Save
    if args.output:
        output = {
            "summary": {"tz": float(np.mean(all_tz)), "pz": float(np.mean(all_pz)), "mean": float(np.mean(all_mean))},
            "per_dataset": {d: {k: float(np.mean(v)) for k, v in vals.items()} for d, vals in domain_agg.items()},
            "per_case": results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
