#!/usr/bin/env python3
"""
src/metrics.py
Defines evaluation metrics for semantic segmentation:
IoU, Dice Score (F1), Precision, Recall, and Pixel Accuracy.
"""

import torch
from typing import Dict


def get_segmentation_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7
) -> Dict[str, float]:
    """
    Computes semantic segmentation metrics for a batch of predictions and targets.
    
    Args:
        preds: Raw model logits or probabilities. Tensor of any shape.
        targets: Binary ground truth mask. Same shape as preds.
        threshold: Decision threshold to binarize predictions.
        eps: Small epsilon to prevent division by zero.
        
    Returns:
        Dict containing IoU, Dice/F1, Precision, Recall, and Accuracy.
    """
    # Apply sigmoid and binarize predictions
    if preds.max() > 1.0 or preds.min() < 0.0:
        probs = torch.sigmoid(preds)
    else:
        probs = preds
        
    binary_preds = (probs > threshold).float()
    binary_targets = (targets > threshold).float()
    
    # Flatten tensors
    binary_preds = binary_preds.view(-1)
    binary_targets = binary_targets.view(-1)
    
    # Calculate TP, FP, FN, TN
    tp = torch.sum(binary_preds * binary_targets)
    fp = torch.sum(binary_preds * (1.0 - binary_targets))
    fn = torch.sum((1.0 - binary_preds) * binary_targets)
    tn = torch.sum((1.0 - binary_preds) * (1.0 - binary_targets))
    
    # Compute metrics
    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    
    return {
        "iou": float(iou.item()),
        "dice": float(dice.item()),
        "precision": float(precision.item()),
        "recall": float(recall.item()),
        "f1_score": float(dice.item()),  # F1 is mathematically identical to Dice coefficient
        "accuracy": float(accuracy.item())
    }
