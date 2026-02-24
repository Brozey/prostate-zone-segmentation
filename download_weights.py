"""
Download pre-trained model weights from GitHub Releases.

Usage:
    python download_weights.py

The checkpoint will be saved to:
    nnUNet_results/Dataset501_ProstateZones/
        nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth
"""

import os
import sys
import urllib.request
from pathlib import Path

# Update this URL after uploading to GitHub Releases
RELEASE_URL = (
    "https://github.com/Brozey/prostate-zone-segmentation"
    "/releases/download/v1.0/checkpoint_final.pth"
)

CHECKPOINT_REL = (
    "nnUNet_results/Dataset501_ProstateZones/"
    "nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
)


def download(url: str, dest: Path) -> None:
    """Download a file with progress indicator."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model weights …")
    print(f"  URL:  {url}")
    print(f"  Dest: {dest}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / 1024 / 1024
            total_mb = total_size / 1024 / 1024
            sys.stdout.write(f"\r  {mb:.1f} / {total_mb:.1f} MB ({pct:.0f}%)")
        else:
            mb = downloaded / 1024 / 1024
            sys.stdout.write(f"\r  {mb:.1f} MB downloaded")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, str(dest), reporthook=_progress)
    print("\n  Download complete.")


def main():
    root = Path(__file__).resolve().parent
    dest = root / CHECKPOINT_REL

    if dest.exists():
        size_mb = dest.stat().st_size / 1024 / 1024
        print(f"Checkpoint already exists ({size_mb:.0f} MB): {dest}")
        ans = input("Re-download? [y/N] ").strip().lower()
        if ans != "y":
            return

    try:
        download(RELEASE_URL, dest)
    except Exception as e:
        print(f"\nDownload failed: {e}", file=sys.stderr)
        print(
            "\nYou can download the checkpoint manually from:",
            f"\n  {RELEASE_URL}",
            f"\n\nPlace it at:\n  {dest}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Verify file size
    size_mb = dest.stat().st_size / 1024 / 1024
    if size_mb < 100:
        print(f"WARNING: File is only {size_mb:.1f} MB — expected ~341 MB.")
        print("The download may have failed. Please check the file.")
    else:
        print(f"Model checkpoint ready ({size_mb:.0f} MB).")


if __name__ == "__main__":
    main()
