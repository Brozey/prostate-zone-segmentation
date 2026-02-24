"""
Prostate Zone Segmentation — nnU-Net Inference
================================================
Segments T2-weighted prostate MRI into:
  Label 1 = Transition Zone (TZ)
  Label 2 = Peripheral Zone (PZ)

Usage:
    python predict.py  -i INPUT  -o OUTPUT  [--gpu 0]

INPUT can be a folder of .nii.gz files or a single .nii.gz file.
Segmentation masks are saved in OUTPUT with the same filenames + _seg suffix.
"""

import argparse
import os
import sys
import shutil
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run nnU-Net prostate zone segmentation on T2w NIfTI images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py -i /data/t2w_images/ -o /data/segmentations/
  python predict.py -i patient.nii.gz -o results/ --gpu -1
        """,
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Folder with .nii.gz T2w images, or a single .nii.gz file.",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Folder where segmentation masks will be saved.",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device ID (default: 0). Use -1 for CPU inference.",
    )
    parser.add_argument(
        "--save-probabilities", action="store_true",
        help="Also save softmax probability maps (.npz).",
    )
    args = parser.parse_args()

    # ── Resolve paths ─────────────────────────────────────────────
    script_dir = Path(__file__).resolve().parent
    model_base = script_dir / "nnUNet_results"
    model_dir = (
        model_base
        / "Dataset501_ProstateZones"
        / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    )
    checkpoint = model_dir / "fold_0" / "checkpoint_final.pth"

    if not checkpoint.exists():
        sys.exit(
            f"Model checkpoint not found at:\n  {checkpoint}\n\n"
            "Download it with:  python download_weights.py\n"
            "Or manually place checkpoint_final.pth in the fold_0/ directory."
        )

    # Point nnU-Net environment at local model copy
    os.environ["nnUNet_results"] = str(model_base)
    os.environ.setdefault("nnUNet_raw", str(script_dir / "_unused_raw"))
    os.environ.setdefault("nnUNet_preprocessed", str(script_dir / "_unused_preproc"))
    # Avoid OpenMP duplicate-library crash (Windows)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # ── Prepare input ─────────────────────────────────────────────
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    single_file = input_path.is_file()
    tmp_dirs = []

    if single_file:
        tmp_in = Path(tempfile.mkdtemp(prefix="nnunet_in_"))
        tmp_dirs.append(tmp_in)
        stem = input_path.name.replace(".nii.gz", "").replace(".nii", "")
        shutil.copy2(str(input_path), str(tmp_in / f"{stem}_0000.nii.gz"))
        nnunet_input = str(tmp_in)
    else:
        nii_files = sorted(input_path.glob("*.nii.gz"))
        if not nii_files:
            sys.exit(f"No .nii.gz files found in {input_path}")

        has_channel_suffix = any("_0000.nii.gz" in f.name for f in nii_files)
        if has_channel_suffix:
            nnunet_input = str(input_path)
        else:
            tmp_in = Path(tempfile.mkdtemp(prefix="nnunet_in_"))
            tmp_dirs.append(tmp_in)
            for f in nii_files:
                stem = f.name.replace(".nii.gz", "")
                shutil.copy2(str(f), str(tmp_in / f"{stem}_0000.nii.gz"))
            nnunet_input = str(tmp_in)

    # ── Run prediction ────────────────────────────────────────────
    tmp_out = Path(tempfile.mkdtemp(prefix="nnunet_out_"))
    tmp_dirs.append(tmp_out)

    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    except ImportError:
        sys.exit(
            "nnU-Net v2 is not installed.\n"
            "Install it with:  pip install -r requirements.txt"
        )

    device_str = "cuda" if args.gpu >= 0 else "cpu"

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_mirroring=True,
        use_gaussian=True,
        perform_everything_on_device=True,
        device=device_str,
        verbose=True,
        verbose_preprocessing=True,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_dir),
        use_folds=(0,),
        checkpoint_name="checkpoint_final.pth",
    )
    predictor.predict_from_files(
        list_of_lists_or_source_folder=nnunet_input,
        output_folder_or_list_of_truncated_output_files=str(tmp_out),
        save_probabilities=args.save_probabilities,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )

    # ── Copy results ──────────────────────────────────────────────
    for pred_file in sorted(tmp_out.glob("*.nii.gz")):
        shutil.copy2(str(pred_file), str(output_path / pred_file.name))
    if args.save_probabilities:
        for npz_file in sorted(tmp_out.glob("*.npz")):
            shutil.copy2(str(npz_file), str(output_path / npz_file.name))

    # Rename output for single-file mode
    if single_file:
        stem = input_path.name.replace(".nii.gz", "").replace(".nii", "")
        pred_files = list(output_path.glob("*.nii.gz"))
        if len(pred_files) == 1:
            pred_files[0].rename(output_path / f"{stem}_seg.nii.gz")

    # ── Cleanup ───────────────────────────────────────────────────
    for d in tmp_dirs:
        shutil.rmtree(str(d), ignore_errors=True)

    n_out = len(list(output_path.glob("*.nii.gz")))
    print(f"\nDone — {n_out} segmentation(s) saved to {output_path}")


if __name__ == "__main__":
    main()
