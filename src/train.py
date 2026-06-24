#!/usr/bin/env python3
"""
src/train.py
Trains U-Net and DeepLabV3 models, tracks experiments in MLflow,
and outputs model checkpoints and training diagnostic plots.
"""

import os
import sys
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import cv2

# Add workspace root to python path to resolve local src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Tuple, Any

from src.utils import load_config, setup_logger, overlay_lane_mask, plot_training_curves, plot_confusion_matrix
from src.dataset import TuSimpleDataset, get_transforms
from src.models import build_model
from src.metrics import get_segmentation_metrics

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Sets random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Runs one epoch of model training.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0
    epoch_metrics = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0}
    
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        # Handle dict outputs from torchvision models like DeepLabV3
        if isinstance(outputs, dict):
            logits = outputs["out"]
        else:
            logits = outputs
            
        loss = criterion(logits, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track statistics
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # Batch metrics
        batch_metrics = get_segmentation_metrics(logits, masks)
        for k in epoch_metrics:
            epoch_metrics[k] += batch_metrics[k] * batch_size
            
    epoch_loss = running_loss / total_samples
    mean_iou = epoch_metrics["iou"] / total_samples
    
    return epoch_loss, mean_iou


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float], Tuple[float, float, float, float]]:
    """
    Validates model performance on the validation set.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    val_metrics = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0}
    
    # Store aggregated pixel confusions for confusion matrix plotting
    tp_tot, fp_tot, fn_tot, tn_tot = 0.0, 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            if isinstance(outputs, dict):
                logits = outputs["out"]
            else:
                logits = outputs
                
            loss = criterion(logits, masks)
            
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Binary predictions
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            # Confusion matrix accumulation
            tp_tot += float(torch.sum(preds * masks).item())
            fp_tot += float(torch.sum(preds * (1.0 - masks)).item())
            fn_tot += float(torch.sum((1.0 - preds) * masks).item())
            tn_tot += float(torch.sum((1.0 - preds) * (1.0 - masks)).item())
            
            # Batch metrics
            batch_metrics = get_segmentation_metrics(logits, masks)
            for k in val_metrics:
                val_metrics[k] += batch_metrics[k] * batch_size
                
    epoch_loss = running_loss / total_samples
    avg_metrics = {k: v / total_samples for k, v in val_metrics.items()}
    
    return epoch_loss, avg_metrics, (tp_tot, fp_tot, fn_tot, tn_tot)


def run_experiment(
    model_type: str,
    epochs: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    config: Dict[str, Any]
) -> None:
    """
    Runs a single training and tracking experiment.
    """
    # Allow local file store tracking in newer MLflow versions
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    
    set_seed(config["training"]["seed"])
    
    device_name = config["training"]["device"]
    device = torch.device("cuda" if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    logger.info(f"Using execution device: {device}")
    
    # Establish paths
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    artifact_dir = config["paths"]["artifact_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Setup data loaders
    img_h = config["dataset"]["img_height"]
    img_w = config["dataset"]["img_width"]
    data_dir = config["dataset"]["root_dir"]
    
    train_dataset = TuSimpleDataset(
        data_dir=data_dir,
        json_file=config["dataset"]["train_json"],
        transform=get_transforms(img_h, img_w, is_train=True)
    )
    val_dataset = TuSimpleDataset(
        data_dir=data_dir,
        json_file=config["dataset"]["val_json"],
        transform=get_transforms(img_h, img_w, is_train=False)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["dataset"]["num_workers"],
        drop_last=(len(train_dataset) > batch_size)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["dataset"]["num_workers"]
    )
    
    logger.info(f"Loaded {len(train_dataset)} training and {len(val_dataset)} validation samples.")
    
    # Build model
    model = build_model(model_type, pretrained=True)
    model = model.to(device)
    
    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    
    # Initialize MLflow tracking
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    run_name = f"{model_type.upper()}_LR_{lr}_BS_{batch_size}"
    
    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"Started MLflow run: {run_name} (ID: {run.info.run_id})")
        
        # Log parameters
        mlflow.log_params({
            "model_type": model_type,
            "epochs": epochs,
            "learning_rate": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "img_height": img_h,
            "img_width": img_w,
            "optimizer": "Adam",
            "loss_function": "BCEWithLogitsLoss"
        })
        
        train_losses, val_losses = [], []
        train_ious, val_ious = [], []
        best_val_iou = 0.0
        
        # Confusion matrix variables (final epoch)
        final_conf_matrix = (0.0, 0.0, 0.0, 0.0)
        
        # Training loops
        for epoch in range(1, epochs + 1):
            train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_metrics, conf_matrix = validate(model, val_loader, criterion, device)
            val_iou = val_metrics["iou"]
            
            # Record statistics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_ious.append(train_iou)
            val_ious.append(val_iou)
            final_conf_matrix = conf_matrix
            
            # Log epoch metrics to MLflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_iou": train_iou,
                "val_loss": val_loss,
                "val_iou": val_iou,
                "val_dice": val_metrics["dice"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_accuracy": val_metrics["accuracy"]
            }, step=epoch)
            
            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} - Train IoU: {train_iou:.4f} | "
                f"Val Loss: {val_loss:.4f} - Val IoU: {val_iou:.4f}"
            )
            
            # Save checkpoints
            best_ckpt_path = os.path.join(checkpoint_dir, f"best_{model_type}.pth")
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                torch.save(model.state_dict(), best_ckpt_path)
                logger.info(f"New best validation IoU ({best_val_iou:.4f}). Saved checkpoint to {best_ckpt_path}")
                
        # Save final checkpoint
        final_ckpt_path = os.path.join(checkpoint_dir, f"final_{model_type}.pth")
        torch.save(model.state_dict(), final_ckpt_path)
        logger.info(f"Saved final checkpoint to {final_ckpt_path}")
        
        # Generate and save diagnostic plots
        curves_plot_path = os.path.join(artifact_dir, f"training_curves_{model_type}.png")
        plot_training_curves(train_losses, val_losses, train_ious, val_ious, curves_plot_path)
        mlflow.log_artifact(curves_plot_path)
        
        conf_plot_path = os.path.join(artifact_dir, f"confusion_matrix_{model_type}.png")
        tp, fp, fn, tn = final_conf_matrix
        plot_confusion_matrix(tp, fp, fn, tn, conf_plot_path)
        mlflow.log_artifact(conf_plot_path)
        
        # Generate visual prediction overlays (save 3 validation samples)
        logger.info("Generating validation overlay samples for MLflow tracking...")
        model.eval()
        samples_logged = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images_dev = images.to(device)
                outputs = model(images_dev)
                logits = outputs["out"] if isinstance(outputs, dict) else outputs
                probs = torch.sigmoid(logits).cpu().numpy()
                
                for idx in range(images.size(0)):
                    if samples_logged >= 3:
                        break
                    # Denormalize image for overlay
                    img_np = images[idx].permute(1, 2, 0).numpy()
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_np = (img_np * std + mean) * 255.0
                    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                    
                    pred_mask = (probs[idx, 0] > 0.5).astype(np.uint8)
                    true_mask = masks[idx, 0].numpy().astype(np.uint8)
                    
                    # Generate overlays
                    overlay_pred = overlay_lane_mask(img_np, pred_mask, color=(0, 255, 0))  # Green
                    overlay_true = overlay_lane_mask(img_np, true_mask, color=(0, 0, 255))  # Red
                    
                    # Merge side-by-side
                    combined = np.hstack((img_np, overlay_true, overlay_pred))
                    # Add labels
                    h_c, w_c = combined.shape[:2]
                    cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(combined, "Ground Truth", (w_c // 3 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(combined, f"Prediction ({model_type.upper()})", (2 * w_c // 3 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    sample_img_path = os.path.join(artifact_dir, f"sample_val_{model_type}_{samples_logged}.png")
                    cv2.imwrite(sample_img_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                    mlflow.log_artifact(sample_img_path)
                    samples_logged += 1
                if samples_logged >= 3:
                    break
                    
        # Log model file directly to MLflow using pickling instead of pt2 graph tracing
        # to bypass shape tracing limitations on torchvision segmentation layers.
        mlflow.pytorch.log_model(model, "model", serialization_format="pickle")
        logger.info(f"Model and artifacts logged for {model_type} run.")


def main() -> None:
    setup_logger()
    
    parser = argparse.ArgumentParser(description="Lane Detection System Trainer")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML file")
    parser.add_argument("--model_type", choices=["unet", "deeplabv3"], help="Override model type to train")
    parser.add_argument("--epochs", type=int, help="Override training epochs count")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument(
        "--run_all",
        action="store_true",
        help="Orchestrates and runs the three official experiments sequentially: 1) U-Net, 2) DeepLabV3, 3) Tuned DeepLabV3."
    )
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.run_all:
        logger.info("Executing all 3 certification experiments sequentially...")
        
        # Experiment 1: U-Net baseline
        logger.info("========================================")
        logger.info("EXPERIMENT 1: U-Net Baseline")
        logger.info("========================================")
        run_experiment(
            model_type="unet",
            epochs=config["training"]["epochs"],
            lr=config["training"]["learning_rate"],
            batch_size=config["dataset"]["batch_size"],
            weight_decay=config["training"]["weight_decay"],
            config=config
        )
        
        # Experiment 2: DeepLabV3 baseline
        logger.info("========================================")
        logger.info("EXPERIMENT 2: DeepLabV3 Baseline")
        logger.info("========================================")
        run_experiment(
            model_type="deeplabv3",
            epochs=config["training"]["epochs"],
            lr=config["training"]["learning_rate"],
            batch_size=config["dataset"]["batch_size"],
            weight_decay=config["training"]["weight_decay"],
            config=config
        )
        
        # Experiment 3: DeepLabV3 with tuned hyperparameters (smaller learning rate and added weight decay)
        logger.info("========================================")
        logger.info("EXPERIMENT 3: DeepLabV3 Tuned Hyperparameters")
        logger.info("========================================")
        run_experiment(
            model_type="deeplabv3",
            epochs=config["training"]["epochs"],
            lr=0.0002,               # Tuned LR
            batch_size=config["dataset"]["batch_size"],
            weight_decay=0.0005,       # Tuned Weight Decay
            config=config
        )
        logger.info("All 3 certification experiments complete and logged in MLflow!")
    else:
        # Run a single experiment
        model_type = args.model_type if args.model_type else config["model"]["type"]
        epochs = args.epochs if args.epochs else config["training"]["epochs"]
        lr = args.lr if args.lr else config["training"]["learning_rate"]
        batch_size = args.batch_size if args.batch_size else config["dataset"]["batch_size"]
        weight_decay = config["training"]["weight_decay"]
        
        run_experiment(
            model_type=model_type,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            weight_decay=weight_decay,
            config=config
        )


if __name__ == "__main__":
    main()
