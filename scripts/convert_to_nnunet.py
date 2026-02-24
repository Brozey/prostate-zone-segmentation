"""
Convert Prostate158 + MSD Task05 + BMC datasets into nnU-Net raw format.

Creates Dataset501_ProstateZones with the following structure:
    nnUNet_raw/Dataset501_ProstateZones/
        imagesTr/   (train + val images)
        labelsTr/   (train + val labels)
        imagesTs/   (held-out test images)
        labelsTs/   (held-out test labels)
        dataset.json

Labels: 0=background, 1=TZ, 2=PZ

NOTE: MSD Task05 has TZ and PZ labels swapped relative to Prostate158.
      This script handles the swap automatically.
"""

import argparse
import json
import os
import shutil

import nibabel as nib
import numpy as np
from PIL import Image


def load_tiff_as_nifti(tiff_path, spacing=(0.625, 0.625, 3.0)):
    """Load a multi-frame TIFF volume and return a NIfTI image."""
    img = Image.open(tiff_path)
    frames = []
    for i in range(getattr(img, "n_frames", 1)):
        img.seek(i)
        frames.append(np.array(img))
    vol = np.stack(frames, axis=0)   # (Z, H, W)
    vol = np.moveaxis(vol, 0, -1)    # (H, W, Z)

    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    return nib.Nifti1Image(vol, affine)


def swap_labels_1_2(nifti_img):
    """Swap label values 1 ↔ 2 (corrects MSD Task05 convention)."""
    data = nifti_img.get_fdata().copy()
    mask1 = data == 1
    mask2 = data == 2
    data[mask1] = 2
    data[mask2] = 1
    return nib.Nifti1Image(data.astype(np.int16), nifti_img.affine, nifti_img.header)


def squeeze_4d(nifti_img):
    """Remove trailing singleton dimension from 4-D NIfTI images."""
    data = nifti_img.get_fdata()
    if data.ndim == 4:
        data = data[..., 0]
        return nib.Nifti1Image(data, nifti_img.affine, nifti_img.header)
    return nifti_img


def main():
    parser = argparse.ArgumentParser(
        description="Convert Prostate158 + MSD Task05 + BMC → nnU-Net raw format."
    )
    parser.add_argument("--prostate158", required=True,
                        help="Path to Prostate158 train/ directory.")
    parser.add_argument("--msd", required=True,
                        help="Path to MSD Task05_Prostate directory.")
    parser.add_argument("--bmc", required=True,
                        help="Path to BMC prostate_dataset directory.")
    parser.add_argument("--output", required=True,
                        help="Path to nnUNet_raw output directory.")
    parser.add_argument("--splits", required=True,
                        help="Path to split JSON (split_seed42.json).")
    args = parser.parse_args()

    dataset_id = 501
    dataset_name = f"Dataset{dataset_id}_ProstateZones"
    dataset_dir = os.path.join(args.output, dataset_name)

    images_tr = os.path.join(dataset_dir, "imagesTr")
    labels_tr = os.path.join(dataset_dir, "labelsTr")
    images_ts = os.path.join(dataset_dir, "imagesTs")
    labels_ts = os.path.join(dataset_dir, "labelsTs")
    for d in [images_tr, labels_tr, images_ts, labels_ts]:
        os.makedirs(d, exist_ok=True)

    # Load split
    with open(args.splits) as f:
        split_data = json.load(f)
    train_ids = set(split_data["train_ids"])
    val_ids = set(split_data["val_ids"])
    test_ids = set(split_data["test_ids"])
    tr_ids = train_ids | val_ids  # nnU-Net does its own cross-val

    # Collect all cases: id → {image, label, swap, tiff}
    all_cases = {}

    # --- Prostate158 ---
    for sid in sorted(os.listdir(args.prostate158)):
        img = os.path.join(args.prostate158, sid, "t2.nii.gz")
        lbl = os.path.join(args.prostate158, sid, "t2_anatomy_reader1.nii.gz")
        if os.path.exists(img) and os.path.exists(lbl):
            all_cases[f"Prostate158_{sid}"] = dict(image=img, label=lbl, swap=False, tiff=False)

    # --- MSD Task05 (labels are swapped: TZ=2,PZ=1 → we fix to TZ=1,PZ=2) ---
    msd_img_dir = os.path.join(args.msd, "imagesTr_T2")
    msd_lbl_dir = os.path.join(args.msd, "labelsTr")
    if os.path.exists(msd_img_dir):
        for f in sorted(os.listdir(msd_img_dir)):
            if f.endswith((".nii", ".nii.gz")):
                lbl = os.path.join(msd_lbl_dir, f)
                if os.path.exists(lbl):
                    all_cases[f"MSD_{f}"] = dict(
                        image=os.path.join(msd_img_dir, f), label=lbl, swap=True, tiff=False
                    )

    # --- BMC (TIFF format) ---
    bmc_img_dir = os.path.join(args.bmc, "T2_IMAGES")
    bmc_lbl_dir = os.path.join(args.bmc, "LABELS")
    if os.path.exists(bmc_img_dir):
        for f in sorted(os.listdir(bmc_img_dir)):
            if f.endswith(".tiff"):
                lbl = os.path.join(bmc_lbl_dir, f.replace(".tiff", "_SEG.tiff"))
                if os.path.exists(lbl):
                    all_cases[f"BMC_{f}"] = dict(
                        image=os.path.join(bmc_img_dir, f), label=lbl, swap=False, tiff=True
                    )

    print(f"Total cases found: {len(all_cases)}")

    case_counter = 0
    id_mapping = {}
    num_training = 0

    for case_id in sorted(all_cases):
        info = all_cases[case_id]
        case_counter += 1
        nn_name = f"case_{case_counter:04d}"
        id_mapping[case_id] = nn_name

        is_train = case_id in tr_ids
        out_img = os.path.join(images_tr if is_train else images_ts, f"{nn_name}_0000.nii.gz")
        out_lbl = os.path.join(labels_tr if is_train else labels_ts, f"{nn_name}.nii.gz")

        if is_train:
            num_training += 1

        if os.path.exists(out_img) and os.path.exists(out_lbl):
            continue

        print(f"  [{case_counter:3d}] {case_id:<35} → {nn_name} ({'train' if is_train else 'test'})")

        if info["tiff"]:
            img_nii = load_tiff_as_nifti(info["image"])
            lbl_nii = load_tiff_as_nifti(info["label"])
            lbl_nii = nib.Nifti1Image(lbl_nii.get_fdata().astype(np.int16), lbl_nii.affine)
        else:
            img_nii = squeeze_4d(nib.load(info["image"]))
            lbl_nii = nib.load(info["label"])
            lbl_data = squeeze_4d(lbl_nii).get_fdata().astype(np.int16)
            lbl_nii = nib.Nifti1Image(lbl_data, lbl_nii.affine, lbl_nii.header)

        if info["swap"]:
            lbl_nii = swap_labels_1_2(lbl_nii)

        nib.save(img_nii, out_img)
        nib.save(lbl_nii, out_lbl)

    # dataset.json
    dataset_json = {
        "channel_names": {"0": "T2"},
        "labels": {"background": 0, "TZ": 1, "PZ": 2},
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "name": "ProstateZones",
        "description": (
            "Prostate zone segmentation (TZ + PZ) from T2w MRI. "
            "Combined Prostate158 + MSD Task05 + BMC datasets."
        ),
        "reference": "Combined multi-site dataset",
        "licence": "Research use only",
        "release": "1.0",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    with open(os.path.join(dataset_dir, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)

    # ID mapping
    with open(os.path.join(dataset_dir, "id_mapping.json"), "w") as f:
        json.dump(id_mapping, f, indent=2)

    # Custom splits for nnU-Net (preserves our train/val split as fold 0)
    tr_nn = [id_mapping[c] for c in sorted(train_ids) if c in id_mapping]
    val_nn = [id_mapping[c] for c in sorted(val_ids) if c in id_mapping]
    with open(os.path.join(dataset_dir, "custom_splits.json"), "w") as f:
        json.dump([{"train": tr_nn, "val": val_nn}], f, indent=2)

    print(f"\nDataset created at: {dataset_dir}")
    print(f"  Training: {num_training}  |  Test: {case_counter - num_training}")
    print(f"\nNext steps:")
    print(f"  1. export nnUNet_raw='{args.output}'")
    print(f"  2. nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity")
    print(f"  3. Copy custom_splits.json → nnUNet_preprocessed/{dataset_name}/splits_final.json")
    print(f"  4. nnUNetv2_train {dataset_id} 3d_fullres 0")


if __name__ == "__main__":
    main()
