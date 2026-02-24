#!/usr/bin/env python3
"""
Generate visual comparisons: MRI with segmentation overlays.
Creates:
  - Side-by-side PNGs: T2w MRI | Ground Truth overlay | Prediction overlay
  - Animated GIF scrolling through slices for selected cases
"""

import argparse
import os
import numpy as np
import nibabel as nib
from PIL import Image, ImageDraw, ImageFont


# Colours: TZ = blue, PZ = green (semi-transparent)
COLORS = {
    1: np.array([0, 120, 255]),   # TZ — blue
    2: np.array([0, 220, 80]),    # PZ — green
}
ALPHA = 0.35  # overlay opacity


def load_nifti(path):
    """Load a NIfTI file, return 3-D numpy array (squeeze 4-D)."""
    data = nib.load(path).get_fdata()
    if data.ndim == 4:
        data = data[..., 0]
    return data


def normalize_image(vol):
    """Percentile-clip + scale to 0-255 uint8."""
    p1, p99 = np.percentile(vol, [1, 99])
    vol = np.clip(vol, p1, p99)
    vol = (vol - p1) / max(p99 - p1, 1e-8) * 255
    return vol.astype(np.uint8)


def overlay_seg(img_slice, seg_slice, alpha=ALPHA):
    """Blend segmentation colours onto a grayscale MRI slice.

    Args:
        img_slice: 2-D uint8 grayscale (H, W)
        seg_slice: 2-D int labels (H, W)  — 0=bg, 1=TZ, 2=PZ
        alpha: overlay opacity

    Returns:
        RGB uint8 (H, W, 3)
    """
    rgb = np.stack([img_slice] * 3, axis=-1).astype(np.float32)
    for label, color in COLORS.items():
        mask = seg_slice == label
        if mask.any():
            rgb[mask] = rgb[mask] * (1 - alpha) + color.astype(np.float32) * alpha
    return np.clip(rgb, 0, 255).astype(np.uint8)


def mid_slice_idx(seg_vol):
    """Find the axial slice with the largest prostate area."""
    area = np.sum(seg_vol > 0, axis=(0, 1))  # per-slice area (axis 2)
    if area.max() == 0:
        return seg_vol.shape[2] // 2
    return int(np.argmax(area))


def make_comparison_png(img_path, gt_path, pred_path, out_path, slice_idx=None):
    """Create a 3-panel comparison image for one case."""
    img_vol = normalize_image(load_nifti(img_path))
    gt_vol = load_nifti(gt_path).astype(int)
    pred_vol = load_nifti(pred_path).astype(int)

    if slice_idx is None:
        slice_idx = mid_slice_idx(gt_vol)

    # Extract axial slices and rotate for display
    img_s = np.rot90(img_vol[:, :, slice_idx])
    gt_s = np.rot90(gt_vol[:, :, slice_idx])
    pred_s = np.rot90(pred_vol[:, :, slice_idx])

    # Build 3 panels
    mri_rgb = np.stack([img_s] * 3, axis=-1)
    gt_overlay = overlay_seg(img_s, gt_s)
    pred_overlay = overlay_seg(img_s, pred_s)

    # Add labels
    panels = [mri_rgb, gt_overlay, pred_overlay]
    labels = ["T2w MRI", "Ground Truth", "Prediction"]

    # Compute uniform panel size
    h, w = img_s.shape[:2]
    gap = 4
    total_w = w * 3 + gap * 2
    canvas = np.zeros((h + 30, total_w, 3), dtype=np.uint8)

    for i, (panel, label) in enumerate(zip(panels, labels)):
        x0 = i * (w + gap)
        canvas[30:30 + h, x0:x0 + w] = panel

    # Convert to PIL, add text labels
    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i, label in enumerate(labels):
        x0 = i * (w + gap) + w // 2 - len(label) * 4
        draw.text((x0, 6), label, fill=(255, 255, 255), font=font)

    # Add legend
    legend_y = 6
    legend_x = total_w - 140
    draw.rectangle([legend_x, legend_y, legend_x + 12, legend_y + 12],
                    fill=(0, 120, 255))
    draw.text((legend_x + 16, legend_y - 2), "TZ", fill=(255, 255, 255), font=font)
    draw.rectangle([legend_x + 50, legend_y, legend_x + 62, legend_y + 12],
                    fill=(0, 220, 80))
    draw.text((legend_x + 66, legend_y - 2), "PZ", fill=(255, 255, 255), font=font)

    pil_img.save(out_path, dpi=(150, 150))
    return pil_img


def make_slice_gif(img_path, pred_path, out_path, fps=4,
                   gt_path=None, n_context=3):
    """Create an animated GIF scrolling through slices with overlay.

    Shows ±n_context slices around the largest prostate slice.
    Left panel: MRI. Right panel: MRI + prediction overlay.
    If gt_path given: 3-panel (MRI | GT | Pred).
    """
    img_vol = normalize_image(load_nifti(img_path))
    pred_vol = load_nifti(pred_path).astype(int)

    ref_vol = pred_vol
    if gt_path and os.path.exists(gt_path):
        gt_vol = load_nifti(gt_path).astype(int)
        ref_vol = gt_vol
    else:
        gt_vol = None

    center = mid_slice_idx(ref_vol)
    n_slices = img_vol.shape[2]
    s_start = max(0, center - n_context)
    s_end = min(n_slices, center + n_context + 1)

    frames = []
    for z in range(s_start, s_end):
        img_s = np.rot90(img_vol[:, :, z])
        pred_s = np.rot90(pred_vol[:, :, z])
        h, w = img_s.shape[:2]

        mri_rgb = np.stack([img_s] * 3, axis=-1)
        pred_ov = overlay_seg(img_s, pred_s)

        if gt_vol is not None:
            gt_s = np.rot90(gt_vol[:, :, z])
            gt_ov = overlay_seg(img_s, gt_s)
            gap = 4
            canvas = np.zeros((h + 30, w * 3 + gap * 2, 3), dtype=np.uint8)
            canvas[30:30 + h, 0:w] = mri_rgb
            canvas[30:30 + h, w + gap:2 * w + gap] = gt_ov
            canvas[30:30 + h, 2 * (w + gap):2 * (w + gap) + w] = pred_ov
            labels = ["T2w MRI", "Ground Truth", "Prediction"]
        else:
            gap = 4
            canvas = np.zeros((h + 30, w * 2 + gap, 3), dtype=np.uint8)
            canvas[30:30 + h, 0:w] = mri_rgb
            canvas[30:30 + h, w + gap:2 * w + gap] = pred_ov
            labels = ["T2w MRI", "Prediction"]

        pil_frame = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_frame)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()

        total_w = canvas.shape[1]
        n_panels = len(labels)
        pw = w
        for i, label in enumerate(labels):
            x0 = i * (pw + gap) + pw // 2 - len(label) * 3
            draw.text((x0, 6), label, fill=(255, 255, 255), font=font)

        # Slice counter
        draw.text((4, 6), f"z={z}/{n_slices-1}", fill=(200, 200, 200), font=font)

        frames.append(pil_frame)

    # Forward + backward for smooth loop
    frames_loop = frames + frames[-2:0:-1]
    duration = int(1000 / fps)
    frames_loop[0].save(
        out_path, save_all=True, append_images=frames_loop[1:],
        duration=duration, loop=0, optimize=True
    )
    return len(frames_loop)


def main():
    parser = argparse.ArgumentParser(
        description="Generate overlay comparison images and GIFs."
    )
    parser.add_argument("--images", required=True,
                        help="Directory with test T2w images (*_0000.nii.gz)")
    parser.add_argument("--labels", required=True,
                        help="Directory with ground-truth labels (*.nii.gz)")
    parser.add_argument("--predictions", required=True,
                        help="Directory with predicted segmentations (*.nii.gz)")
    parser.add_argument("--output", required=True,
                        help="Output directory for generated visuals")
    parser.add_argument("--num-cases", type=int, default=5,
                        help="Number of cases for overview PNG (default: 5)")
    parser.add_argument("--gif-cases", type=int, default=3,
                        help="Number of cases for GIF (default: 3)")
    parser.add_argument("--gif-context", type=int, default=4,
                        help="Slices above/below center for GIF (default: 4)")
    parser.add_argument("--gif-fps", type=int, default=3,
                        help="GIF frame rate (default: 3)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Collect matching cases
    pred_files = sorted([f for f in os.listdir(args.predictions)
                         if f.endswith(".nii.gz")])
    cases = []
    for pf in pred_files:
        case_id = pf.replace(".nii.gz", "")
        img_file = os.path.join(args.images, case_id + "_0000.nii.gz")
        gt_file = os.path.join(args.labels, pf)
        pred_file = os.path.join(args.predictions, pf)
        if os.path.exists(img_file) and os.path.exists(gt_file):
            cases.append((case_id, img_file, gt_file, pred_file))

    if not cases:
        print("ERROR: No matching cases found. Check paths.")
        return

    print(f"Found {len(cases)} test cases with images + GT + predictions.\n")

    # --- Per-case PNGs ---
    png_cases = cases[:args.num_cases]
    print(f"Generating {len(png_cases)} comparison PNGs...")
    for case_id, img_f, gt_f, pred_f in png_cases:
        out = os.path.join(args.output, f"{case_id}_comparison.png")
        make_comparison_png(img_f, gt_f, pred_f, out)
        print(f"  {case_id}_comparison.png")

    # --- Multi-case grid: the hero image for README ---
    print("\nGenerating hero overview image...")
    hero_cases = cases[:min(4, len(cases))]
    panels = []
    for case_id, img_f, gt_f, pred_f in hero_cases:
        img_vol = normalize_image(load_nifti(img_f))
        gt_vol = load_nifti(gt_f).astype(int)
        pred_vol = load_nifti(pred_f).astype(int)
        z = mid_slice_idx(gt_vol)

        img_s = np.rot90(img_vol[:, :, z])
        gt_s = np.rot90(gt_vol[:, :, z])
        pred_s = np.rot90(pred_vol[:, :, z])

        mri_rgb = np.stack([img_s] * 3, axis=-1)
        gt_ov = overlay_seg(img_s, gt_s)
        pred_ov = overlay_seg(img_s, pred_s)

        h, w = img_s.shape[:2]
        gap = 3
        row = np.zeros((h, w * 3 + gap * 2, 3), dtype=np.uint8)
        row[:, 0:w] = mri_rgb
        row[:, w + gap:2 * w + gap] = gt_ov
        row[:, 2 * (w + gap):2 * (w + gap) + w] = pred_ov
        panels.append(row)

    # Resize all to same width
    max_w = max(p.shape[1] for p in panels)
    resized = []
    for p in panels:
        if p.shape[1] < max_w:
            pad = np.zeros((p.shape[0], max_w - p.shape[1], 3), dtype=np.uint8)
            p = np.concatenate([p, pad], axis=1)
        resized.append(p)

    row_gap = 3
    header_h = 35
    total_h = sum(r.shape[0] for r in resized) + row_gap * (len(resized) - 1) + header_h
    hero = np.zeros((total_h, max_w, 3), dtype=np.uint8)

    y = header_h
    for r in resized:
        hero[y:y + r.shape[0], :r.shape[1]] = r
        y += r.shape[0] + row_gap

    pil_hero = Image.fromarray(hero)
    draw = ImageDraw.Draw(pil_hero)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        font_sm = ImageFont.truetype("arial.ttf", 13)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_sm = font

    col_w = panels[0].shape[1] // 3
    col_labels = ["T2w MRI", "Ground Truth", "Prediction"]
    for i, label in enumerate(col_labels):
        x = i * (col_w + 3) + col_w // 2 - len(label) * 5
        draw.text((x, 8), label, fill=(255, 255, 255), font=font)

    # Legend
    lx = max_w - 150
    draw.rectangle([lx, 10, lx + 14, 24], fill=(0, 120, 255))
    draw.text((lx + 18, 8), "TZ", fill=(255, 255, 255), font=font_sm)
    draw.rectangle([lx + 55, 10, lx + 69, 24], fill=(0, 220, 80))
    draw.text((lx + 73, 8), "PZ", fill=(255, 255, 255), font=font_sm)

    hero_path = os.path.join(args.output, "example_prediction.png")
    pil_hero.save(hero_path, dpi=(150, 150))
    print(f"  example_prediction.png ({len(hero_cases)} cases)")

    # --- GIFs ---
    gif_cases = cases[:args.gif_cases]
    print(f"\nGenerating {len(gif_cases)} slice-scroll GIFs...")
    for case_id, img_f, gt_f, pred_f in gif_cases:
        out = os.path.join(args.output, f"{case_id}_slices.gif")
        n = make_slice_gif(img_f, pred_f, out, fps=args.gif_fps,
                           gt_path=gt_f, n_context=args.gif_context)
        print(f"  {case_id}_slices.gif  ({n} frames)")

    print(f"\nDone! All visuals saved to: {args.output}")


if __name__ == "__main__":
    main()
