#!/usr/bin/env python3
"""
src/evaluate.py
Loads trained model checkpoints, evaluates them on the test set,
generates evaluation metrics, creates comparison plots, and saves results to artifacts/.
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add workspace root to python path to resolve local src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, Tuple, List

from src.utils import load_config, setup_logger, overlay_lane_mask
from src.dataset import TuSimpleDataset, get_transforms
from src.models import build_model
from src.metrics import get_segmentation_metrics
import cv2

logger = logging.getLogger(__name__)


def evaluate_model(
    model_type: str,
    checkpoint_path: str,
    loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any]
) -> Tuple[Dict[str, float], List[np.ndarray]]:
    """
    Evaluates a specific model on the loader and gathers metric averages.
    
    Returns:
        Dict of metrics, and a list of output masks for visualization.
    """
    model = build_model(model_type, pretrained=False)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found for evaluation: {checkpoint_path}")
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded {model_type} weights from {checkpoint_path}. Starting evaluation...")
    
    accumulated_metrics = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0}
    total_samples = 0
    predictions_list = []
    
    with torch.no_grad():
        for images, masks in loader:
            images_dev = images.to(device)
            masks_dev = masks.to(device)
            
            outputs = model(images_dev)
            logits = outputs["out"] if isinstance(outputs, dict) else outputs
            probs = torch.sigmoid(logits)
            
            batch_size = images.size(0)
            total_samples += batch_size
            
            # Batch metrics calculation
            batch_metrics = get_segmentation_metrics(logits, masks_dev)
            for k in accumulated_metrics:
                accumulated_metrics[k] += batch_metrics[k] * batch_size
                
            # Store predictions for saving samples later (first batch only)
            if len(predictions_list) < 5:
                for idx in range(batch_size):
                    prob_mask = probs[idx, 0].cpu().numpy()
                    pred_mask = (prob_mask > 0.5).astype(np.uint8)
                    predictions_list.append(pred_mask)
                    if len(predictions_list) >= 5:
                        break
                        
    # Average metrics
    avg_metrics = {k: v / total_samples for k, v in accumulated_metrics.items()}
    logger.info(f"Finished evaluating {model_type}. Mean IoU: {avg_metrics['iou']:.4f}")
    return avg_metrics, predictions_list


def generate_comparison_plots(df: pd.DataFrame, save_path: str) -> None:
    """
    Generates a comparative bar chart for the models across standard segmentation metrics.
    """
    # Set metrics to plot
    metrics = ["IoU", "Dice / F1", "Precision", "Recall", "Accuracy"]
    models = df.index.tolist()
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars for each model
    for idx, model_name in enumerate(models):
        row = df.loc[model_name]
        values = [row["IoU"], row["Dice / F1"], row["Precision"], row["Recall"], row["Accuracy"]]
        ax.bar(x + (idx - 0.5) * width, values, width, label=model_name.upper())
        
    ax.set_ylabel('Scores')
    ax.set_title('Lane Detection Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Annotate bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(p.get_x() + p.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved performance comparison chart to {save_path}")


def main() -> None:
    setup_logger()
    
    parser = argparse.ArgumentParser(description="Lane Detection System Evaluator")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    device_name = config["training"]["device"]
    device = torch.device("cuda" if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    
    artifact_dir = config["paths"]["artifact_dir"]
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Load dataset
    img_h = config["dataset"]["img_height"]
    img_w = config["dataset"]["img_width"]
    data_dir = config["dataset"]["root_dir"]
    
    test_dataset = TuSimpleDataset(
        data_dir=data_dir,
        json_file=config["dataset"]["test_json"],
        transform=get_transforms(img_h, img_w, is_train=False)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"]
    )
    
    logger.info(f"Loaded {len(test_dataset)} test samples.")
    
    # Evaluate models
    results = {}
    unet_preds = []
    deeplab_preds = []
    
    # Check if U-Net checkpoint exists
    unet_ckpt = os.path.join(checkpoint_dir, "best_unet.pth")
    if os.path.exists(unet_ckpt):
        try:
            results["unet"], unet_preds = evaluate_model("unet", unet_ckpt, test_loader, device, config)
        except Exception as e:
            logger.error(f"Failed to evaluate U-Net: {e}")
    else:
        logger.warning(f"U-Net checkpoint not found at {unet_ckpt}, skipping evaluation.")
        
    # Check if DeepLabV3 checkpoint exists
    deeplab_ckpt = os.path.join(checkpoint_dir, "best_deeplabv3.pth")
    if os.path.exists(deeplab_ckpt):
        try:
            results["deeplabv3"], deeplab_preds = evaluate_model("deeplabv3", deeplab_ckpt, test_loader, device, config)
        except Exception as e:
            logger.error(f"Failed to evaluate DeepLabV3: {e}")
    else:
        logger.warning(f"DeepLabV3 checkpoint not found at {deeplab_ckpt}, skipping evaluation.")
        
    if not results:
        logger.error("No model checkpoints were found. Please train models first using src/train.py.")
        sys.exit(1)
        
    # Process results into dataframe
    data_dict = {
        model_name: {
            "IoU": metrics["iou"],
            "Dice / F1": metrics["dice"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "Accuracy": metrics["accuracy"]
        }
        for model_name, metrics in results.items()
    }
    
    df = pd.DataFrame.from_dict(data_dict, orient='index')
    
    # Save comparison table markdown
    table_path = os.path.join(artifact_dir, "model_comparison.md")
    with open(table_path, "w") as f:
        f.write("# Model Performance Comparison\n\n")
        f.write(df.to_markdown())
        f.write("\n")
    logger.info(f"Saved comparison table to {table_path}")
    
    # Generate and save comparison plots
    plot_path = os.path.join(artifact_dir, "model_comparison.png")
    generate_comparison_plots(df, plot_path)
    
    # Save comparison prediction visual images
    # We will generate visual comparison of the first 3 images in the test loader
    logger.info("Generating comparison visualization samples...")
    count = 0
    with torch.no_grad():
        for images, masks in test_loader:
            for idx in range(images.size(0)):
                if count >= 3:
                    break
                # Denormalize original image
                img_np = images[idx].permute(1, 2, 0).numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = (img_np * std + mean) * 255.0
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                
                true_mask = masks[idx, 0].numpy().astype(np.uint8)
                overlay_true = overlay_lane_mask(img_np, true_mask, color=(0, 0, 255))  # Red for Ground Truth
                
                parts = [img_np, overlay_true]
                labels = ["Original", "Ground Truth"]
                
                if "unet" in results and count < len(unet_preds):
                    unet_mask = unet_preds[count]
                    overlay_unet = overlay_lane_mask(img_np, unet_mask, color=(0, 255, 0))  # Green for U-Net
                    parts.append(overlay_unet)
                    labels.append("U-Net Prediction")
                    
                if "deeplabv3" in results and count < len(deeplab_preds):
                    deeplab_mask = deeplab_preds[count]
                    overlay_dl = overlay_lane_mask(img_np, deeplab_mask, color=(255, 165, 0))  # Orange for DeepLab
                    parts.append(overlay_dl)
                    labels.append("DeepLabV3 Prediction")
                    
                # Merge side-by-side
                combined = np.hstack(parts)
                h_c, w_c = combined.shape[:2]
                w_single = w_c // len(parts)
                
                # Annotate labels
                for p_idx, label in enumerate(labels):
                    cv2.putText(
                        combined, label, (p_idx * w_single + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
                    )
                    
                save_visual_path = os.path.join(artifact_dir, f"comparison_sample_{count}.png")
                cv2.imwrite(save_visual_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                count += 1
            if count >= 3:
                break
                
    logger.info("Evaluation complete! All reports and plots generated in the artifacts directory.")


if __name__ == "__main__":
    main()
