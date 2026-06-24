#!/usr/bin/env python3
"""
src/dataset.py
Custom PyTorch Dataset for loading the TuSimple lane detection dataset,
generating binary segmentation masks from lane coordinates, and applying augmentations.
"""

import os
import json
import logging
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)


class TuSimpleDataset(Dataset):
    """
    PyTorch Dataset for TuSimple Lane Detection.
    Converts list of lane coordinates to binary semantic segmentation masks.
    """
    def __init__(
        self,
        data_dir: str,
        json_file: str,
        transform: Optional[A.Compose] = None,
        lane_thickness: int = 10
    ) -> None:
        """
        Args:
            data_dir: Root path to the dataset.
            json_file: Filename of the annotation JSON (e.g. label_data_0313.json).
            transform: Albumentations composition for image/mask transforms.
            lane_thickness: Thickness of lane lines drawn on the mask.
        """
        self.data_dir = os.path.abspath(data_dir)
        self.json_path = os.path.join(self.data_dir, json_file)
        self.transform = transform
        self.lane_thickness = lane_thickness
        self.samples: List[Dict[str, Any]] = []

        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Annotation file not found: {self.json_path}")

        self._load_annotations()

    def _load_annotations(self) -> None:
        """
        Reads annotations from JSON file. Each line is a JSON object.
        """
        logger.info(f"Loading annotations from {self.json_path}...")
        count = 0
        with open(self.json_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    # Ensure image path is valid
                    full_img_path = os.path.join(self.data_dir, entry["raw_file"])
                    if os.path.exists(full_img_path):
                        self.samples.append(entry)
                        count += 1
                    else:
                        logger.warning(f"Image not found, skipping: {full_img_path}")
                except Exception as e:
                    logger.error(f"Error parsing line in {self.json_path}: {e}")
        logger.info(f"Loaded {count} valid samples from {self.json_path}.")

    def _generate_mask(self, entry: Dict[str, Any], img_shape: Tuple[int, int]) -> np.ndarray:
        """
        Renders lane coordinates onto a binary segmentation mask.
        
        Args:
            entry: Dictionary containing 'lanes' and 'h_samples'.
            img_shape: (height, width) of the original image.
            
        Returns:
            Binary mask (numpy array) of the same spatial dimensions, dtype=uint8 (0=bg, 1=lane).
        """
        h, w = img_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        lanes = entry["lanes"]
        h_samples = entry["h_samples"]
        
        for lane in lanes:
            points = []
            for x, y in zip(lane, h_samples):
                # Filter out invalid values (-2 represents absent points in TuSimple)
                if x >= 0 and y >= 0:
                    points.append((int(x), int(y)))
            
            if len(points) >= 2:
                points_arr = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                # Draw lanes as continuous polylines on the mask
                cv2.polylines(
                    mask,
                    [points_arr],
                    isClosed=False,
                    color=1,
                    thickness=self.lane_thickness
                )
        return mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image_tensor: Normalized float tensor of shape (3, H, W)
            mask_tensor: Float tensor of shape (1, H, W) containing binary mask
        """
        entry = self.samples[idx]
        img_path = os.path.join(self.data_dir, entry["raw_file"])
        
        # Load image (BGR to RGB)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Generate target mask
        mask = self._generate_mask(entry, image.shape[:2])
        
        # Apply data augmentations & resizing
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].to(torch.float32)
            # Ensure mask has channel dimension (1, H, W)
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
        else:
            # Simple conversion to tensor if no transforms provided
            image = torch.from_numpy(image).permute(2, 0, 1).to(torch.float32) / 255.0
            mask = torch.from_numpy(mask).to(torch.float32).unsqueeze(0)
            
        return image, mask


def get_transforms(img_height: int, img_width: int, is_train: bool = True) -> A.Compose:
    """
    Returns standard Albumentations transforms for train / validation splits.
    """
    if is_train:
        return A.Compose([
            A.Resize(height=img_height, width=img_width),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(height=img_height, width=img_width),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
