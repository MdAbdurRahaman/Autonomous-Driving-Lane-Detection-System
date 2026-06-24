#!/usr/bin/env python3
"""
src/data_download.py
Downloads and sets up the TuSimple lane detection dataset.
Supports a quick 'sample' mode for verification and a 'full' mode for the official dataset.
"""

import os
import sys
import json
import zipfile
import urllib.request
import argparse
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Official TuSimple AWS S3 download links
OFFICIAL_URLS = {
    "train_set": "https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/train_set.zip",
    "test_set": "https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/test_set.zip",
    "test_label": "https://s3.us-east-2.amazonaws.com/benchmark-frontend/truth/1/test_label.json"
}

# Curated high-quality highway images with lanes from public lane detection repos
SAMPLE_IMAGE_URLS = [
    "https://raw.githubusercontent.com/MaybeShewill-CV/lanenet-lane-detection/master/data/tusimple_test_image/0.jpg",
    "https://raw.githubusercontent.com/MaybeShewill-CV/lanenet-lane-detection/master/data/tusimple_test_image/1.jpg",
    "https://raw.githubusercontent.com/MaybeShewill-CV/lanenet-lane-detection/master/data/tusimple_test_image/2.jpg",
    "https://raw.githubusercontent.com/MaybeShewill-CV/lanenet-lane-detection/master/data/tusimple_test_image/3.jpg",
    "https://raw.githubusercontent.com/MaybeShewill-CV/lanenet-lane-detection/master/data/tusimple_test_image/4.jpg"
]


def download_file(url: str, dest_path: str) -> None:
    """
    Downloads a file from a URL to a local destination, showing progress.
    """
    logger.info(f"Downloading {url} to {dest_path}...")
    try:
        def progress_callback(block_num: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                percent = min(100, int(block_num * block_size * 100 / total_size))
                if percent % 10 == 0:  # Print every 10%
                    sys.stdout.write(f"\rDownloading... {percent}%")
                    sys.stdout.flush()

        urllib.request.urlretrieve(url, dest_path, reporthook=progress_callback)
        sys.stdout.write("\n")
        logger.info(f"Successfully downloaded to {dest_path}")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        raise e


def create_sample_dataset(data_dir: str) -> None:
    """
    Creates a small, fully functional sample TuSimple-like dataset
    for verification and testing.
    """
    logger.info("Setting up sample TuSimple dataset...")
    
    # Define directories
    clips_root = os.path.join(data_dir, "clips", "0530")
    os.makedirs(clips_root, exist_ok=True)
    
    h_samples = list(range(240, 720, 10))  # 48 points
    
    label_entries: List[Dict[str, Any]] = []
    
    for idx, url in enumerate(SAMPLE_IMAGE_URLS):
        clip_dir = os.path.join(clips_root, f"1492626047222176189_{idx}")
        os.makedirs(clip_dir, exist_ok=True)
        
        dest_img_path = os.path.join(clip_dir, "20.jpg")
        
        # Download sample road image
        try:
            download_file(url, dest_img_path)
        except Exception:
            logger.warning(f"Failed to download sample image {idx}, creating empty placeholder image...")
            # Fallback to create a black placeholder image if download fails
            try:
                import numpy as np
                from PIL import Image
                dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
                # Paint mock lanes with correct perspective
                for y in range(350, 720):
                    # Left lane line (drifts left as it goes down)
                    xl = int(550 - (y - 300) * 0.85)
                    dummy_img[y, xl-5:xl+5] = [255, 255, 255]
                    # Right lane line (drifts right as it goes down)
                    xr = int(730 + (y - 300) * 0.85)
                    dummy_img[y, xr-5:xr+5] = [255, 255, 255]
                Image.fromarray(dummy_img).save(dest_img_path)
                logger.info(f"Saved black placeholder with mock lanes to {dest_img_path}")
            except Exception as placeholder_err:
                logger.error(f"Cannot generate placeholder image: {placeholder_err}")
                raise placeholder_err
        
        # Generate TuSimple-compatible annotations matching real highway perspective
        # Left lane converges near center (550 at y=300) and diverges at bottom (202 at y=710)
        # Right lane converges near center (730 at y=300) and diverges at bottom (1078 at y=710)
        left_lane = [int(550 - (h - 300) * 0.85) if h >= 350 else -2 for h in h_samples]
        right_lane = [int(730 + (h - 300) * 0.85) if h >= 350 else -2 for h in h_samples]
        
        # Hide the lane points at the top (horizon) as in real data
        for k in range(len(h_samples)):
            if h_samples[k] < 300:
                left_lane[k] = -2
                right_lane[k] = -2
        
        rel_img_path = f"clips/0530/1492626047222176189_{idx}/20.jpg"
        
        entry = {
            "raw_file": rel_img_path,
            "lanes": [left_lane, right_lane],
            "h_samples": h_samples
        }
        label_entries.append(entry)

    # Write training, validation, and test split JSONs
    # In sample mode we will write all of them with these entries
    for filename in ["label_data_0313.json", "label_data_0531.json", "label_data_0601.json", "test_label.json"]:
        json_path = os.path.join(data_dir, filename)
        with open(json_path, "w") as f:
            for entry in label_entries:
                f.write(json.dumps(entry) + "\n")
        logger.info(f"Wrote sample annotation file: {json_path}")
        
    logger.info("Sample TuSimple dataset setup complete!")


def setup_full_dataset(data_dir: str) -> None:
    """
    Downloads and extracts the full official TuSimple dataset.
    """
    logger.info("Starting download of the FULL TuSimple Lane Detection dataset (~12GB total)...")
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Download & extract train_set.zip
    train_zip = os.path.join(data_dir, "train_set.zip")
    if not os.path.exists(train_zip):
        download_file(OFFICIAL_URLS["train_set"], train_zip)
    else:
        logger.info(f"Found existing train_set.zip at {train_zip}")
        
    logger.info("Extracting train_set.zip...")
    with zipfile.ZipFile(train_zip, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    logger.info("Extracted train_set.zip successfully.")
    
    # 2. Download & extract test_set.zip
    test_zip = os.path.join(data_dir, "test_set.zip")
    if not os.path.exists(test_zip):
        download_file(OFFICIAL_URLS["test_set"], test_zip)
    else:
        logger.info(f"Found existing test_set.zip at {test_zip}")
        
    logger.info("Extracting test_set.zip...")
    with zipfile.ZipFile(test_zip, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    logger.info("Extracted test_set.zip successfully.")
    
    # 3. Download test_label.json
    test_label_dest = os.path.join(data_dir, "test_label.json")
    if not os.path.exists(test_label_dest):
        download_file(OFFICIAL_URLS["test_label"], test_label_dest)
    else:
        logger.info(f"Found existing test_label.json at {test_label_dest}")

    logger.info("Full TuSimple dataset setup complete!")


def main() -> None:
    parser = argparse.ArgumentParser(description="TuSimple Dataset Downloader & Setup")
    parser.add_argument(
        "--mode",
        choices=["sample", "full"],
        default="sample",
        help="Download mode: 'sample' for a small test set (5 images) or 'full' for official TuSimple dataset."
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Local directory to store dataset."
    )
    args = parser.parse_args()
    
    # Resolve absolute path for clarity
    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    
    logger.info(f"Dataset target folder: {data_dir}")
    
    if args.mode == "sample":
        create_sample_dataset(data_dir)
    else:
        setup_full_dataset(data_dir)


if __name__ == "__main__":
    main()
