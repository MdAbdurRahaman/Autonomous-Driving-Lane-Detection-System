#!/usr/bin/env python3
"""
src/predict.py
Performs lane detection inference on a single image, a folder of images, or a video file.
Outputs predictions (lane mask, visual overlay, video file) to outputs/.
"""

import os
import sys
import time
import argparse
import logging
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
# Add workspace root to python path to resolve local src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Tuple

from src.utils import load_config, setup_logger, overlay_lane_mask
from src.models import build_model

logger = logging.getLogger(__name__)


def preprocess_image(image_rgb: np.ndarray, height: int, width: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Resizes, normalizes, and converts image to torch tensor.
    
    Returns:
        tensor: Shape (1, 3, height, width)
        orig_size: Tuple (original_height, original_width)
    """
    orig_h, orig_w = image_rgb.shape[:2]
    
    transform = A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    augmented = transform(image=image_rgb)
    tensor = augmented["image"].unsqueeze(0)  # Add batch dimension
    
    return tensor, (orig_h, orig_w)


def predict_mask(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    height: int,
    width: int,
    device: torch.device
) -> np.ndarray:
    """
    Runs model inference on an image and returns the binary mask.
    """
    orig_h, orig_w = image_rgb.shape[:2]
    
    # Preprocess
    tensor, _ = preprocess_image(image_rgb, height, width)
    tensor = tensor.to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(tensor)
        logits = output["out"] if isinstance(output, dict) else output
        prob = torch.sigmoid(logits).cpu().squeeze().numpy()
        
    # Resize probability map back to original size
    prob_resized = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    
    # Threshold to create binary mask
    binary_mask = (prob_resized > 0.5).astype(np.uint8)
    return binary_mask


def process_image(
    model: torch.nn.Module,
    image_path: str,
    output_dir: str,
    height: int,
    width: int,
    device: torch.device
) -> None:
    """
    Runs inference on a single image, saves binary mask and visual overlay.
    """
    logger.info(f"Processing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not open image: {image_path}")
        return
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Predict
    mask = predict_mask(model, image_rgb, height, width, device)
    
    # Create overlay
    overlay = overlay_lane_mask(image_rgb, mask, color=(0, 255, 0), alpha=0.4)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    
    # Prepare save names
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_save_path = os.path.join(output_dir, f"{base_name}_mask.png")
    overlay_save_path = os.path.join(output_dir, f"{base_name}_overlay.png")
    
    # Save results
    cv2.imwrite(mask_save_path, mask * 255)
    cv2.imwrite(overlay_save_path, overlay_bgr)
    logger.info(f"Saved mask to {mask_save_path}")
    logger.info(f"Saved overlay to {overlay_save_path}")


def process_video(
    model: torch.nn.Module,
    video_path: str,
    output_path: str,
    height: int,
    width: int,
    device: torch.device
) -> None:
    """
    Processes video frame-by-frame, performs lane detection, overlays mask, and exports output video.
    """
    logger.info(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file: {video_path}")
        return
        
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Video Writer
    # Use MP4V or XVID codec (standard on Windows/Linux)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    logger.info(f"Video specs: {frame_width}x{frame_height} @ {fps} fps. Total frames: {total_frames}")
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Inference
        t0 = time.time()
        mask = predict_mask(model, frame_rgb, height, width, device)
        inference_time = (time.time() - t0) * 1000  # in ms
        
        # Overlay
        overlay = overlay_lane_mask(frame_rgb, mask, color=(0, 255, 0), alpha=0.4)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        
        # Add telemetry metadata text overlay
        fps_info = f"Inference: {inference_time:.1f}ms ({1000 / (inference_time + 1e-5):.1f} FPS)"
        cv2.putText(
            overlay_bgr, fps_info, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
        )
        cv2.putText(
            overlay_bgr, "Autonomous Lane Detection System", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
        )
        
        # Write frame
        out.write(overlay_bgr)
        
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            avg_fps = frame_count / elapsed
            logger.info(f"Processed frame {frame_count}/{total_frames} (Avg FPS: {avg_fps:.1f})")
            
    cap.release()
    out.release()
    logger.info(f"Video processing finished. Saved to {output_path}")


def main() -> None:
    setup_logger()
    
    parser = argparse.ArgumentParser(description="Lane Detection System Predictor")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML file")
    parser.add_argument("--image", help="Path to single input image")
    parser.add_argument("--dir", help="Path to folder containing images")
    parser.add_argument("--video", help="Path to input driving video file")
    parser.add_argument("--model_type", choices=["unet", "deeplabv3"], help="Model type to run inference")
    parser.add_argument("--checkpoint", help="Path to specific model weight checkpoint (.pth)")
    parser.add_argument("--output_dir", default="outputs", help="Directory to save predictions")
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Detect device
    device_name = config["training"]["device"]
    device = torch.device("cuda" if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    logger.info(f"Running inference on device: {device}")
    
    model_type = args.model_type if args.model_type else config["model"]["type"]
    
    # Resolve weights path
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoint_path = os.path.join(config["paths"]["checkpoint_dir"], f"best_{model_type}.pth")
        
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint path not found: {checkpoint_path}")
        logger.error("Please train the model first or provide a valid --checkpoint path.")
        sys.exit(1)
        
    # Build and load model
    model = build_model(model_type, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    model = model.to(device)
    model.eval()
    logger.info(f"Successfully loaded {model_type} weights from {checkpoint_path}")
    
    img_h = config["dataset"]["img_height"]
    img_w = config["dataset"]["img_width"]
    
    if args.image:
        process_image(model, args.image, output_dir, img_h, img_w, device)
        
    elif args.dir:
        logger.info(f"Scanning directory: {args.dir}")
        for filename in os.listdir(args.dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(args.dir, filename)
                process_image(model, img_path, output_dir, img_h, img_w, device)
                
    elif args.video:
        vid_name = os.path.splitext(os.path.basename(args.video))[0]
        out_vid_path = os.path.join(output_dir, f"{vid_name}_processed.mp4")
        process_video(model, args.video, out_vid_path, img_h, img_w, device)
        
    else:
        logger.error("Please supply one of: --image, --dir, or --video for prediction.")
        parser.print_help()


if __name__ == "__main__":
    main()
