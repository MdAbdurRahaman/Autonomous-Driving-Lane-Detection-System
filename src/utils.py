#!/usr/bin/env python3
"""
src/utils.py
Helper functions for configuration, file handling, logging, and visualization.
Includes image overlay utility to combine masks with raw camera frames.
"""

import os
import yaml
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Loads configuration settings from a YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_logger(log_level: int = logging.INFO) -> None:
    """
    Sets up the global logging configuration.
    """
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def overlay_lane_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4
) -> np.ndarray:
    """
    Overlay a binary lane segmentation mask onto the original road image.
    
    Args:
        image: Numpy array of shape (H, W, 3), RGB format.
        mask: Binary mask of shape (H, W) or (H, W, 1) with values in {0, 1} or {0, 255}.
        color: RGB color tuple for the lane overlay. Default is green (0, 255, 0).
        alpha: Transparency value for blending (0.0 = fully transparent, 1.0 = opaque).
        
    Returns:
        Numpy array (H, W, 3) representing the blended image.
    """
    # Ensure mask is 2D
    if len(mask.shape) == 3:
        mask = mask.squeeze(-1)
        
    # Resize mask if dimensions don't match image
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    # Convert mask to binary (0 and 1)
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    # Create colored mask overlay
    colored_mask = np.zeros_like(image)
    colored_mask[binary_mask == 1] = color
    
    # Perform alpha blending where the lane is detected
    overlay_img = image.copy()
    mask_indices = binary_mask == 1
    overlay_img[mask_indices] = cv2.addWeighted(
        image[mask_indices],
        1.0 - alpha,
        colored_mask[mask_indices],
        alpha,
        0
    )
    
    return overlay_img


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    train_ious: list,
    val_ious: list,
    save_path: str
) -> None:
    """
    Generates and saves training curves for loss and IoU metrics.
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss Curve
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # IoU Curve
    ax2.plot(epochs, train_ious, 'b-', label='Training IoU')
    ax2.plot(epochs, val_ious, 'r-', label='Validation IoU')
    ax2.set_title('Training & Validation Mean IoU')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('IoU')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved training curves plot to {save_path}")


def plot_confusion_matrix(
    tp: float,
    fp: float,
    fn: float,
    tn: float,
    save_path: str
) -> None:
    """
    Plots a 2x2 confusion matrix for binary lane segmentation at pixel level.
    """
    matrix = np.array([[tn, fp], [fn, tp]])
    labels = [["True Neg\n(Background)", "False Pos\n(False Lane)"], 
              ["False Neg\n(Missed Lane)", "True Pos\n(Detected Lane)"]]
              
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Normalize values for display clarity
    total = matrix.sum() + 1e-7
    matrix_normalized = matrix / total
    
    im = ax.imshow(matrix_normalized, cmap=plt.cm.Greens)
    plt.colorbar(im, ax=ax)
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Negatives (0)', 'Positives (1)'])
    ax.set_yticklabels(['Negatives (0)', 'Positives (1)'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Pixel-Level Confusion Matrix')
    
    # Annotate the matrix blocks
    for i in range(2):
        for j in range(2):
            count_str = f"{matrix[i, j]:,.0f}"
            pct_str = f"{matrix_normalized[i, j]*100:.2f}%"
            text_color = "white" if matrix_normalized[i, j] > 0.5 else "black"
            ax.text(
                j, i, f"{labels[i][j]}\n\nCount: {count_str}\nRatio: {pct_str}",
                ha="center", va="center", color=text_color, fontweight='bold'
            )
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved pixel-level confusion matrix to {save_path}")
