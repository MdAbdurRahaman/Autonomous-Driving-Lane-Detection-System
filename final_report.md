# Final Project Report: Autonomous Driving Lane Detection System
**Course Final Project Submission**  
**Domain:** Autonomous Driving Perception, Computer Vision, and MLOps  
**Academic Year:** 2026

---

## Abstract
This report presents an end-to-end Machine Learning pipeline for road lane detection using deep semantic segmentation networks. Roads lanes play a fundamental role in lateral control and navigation of autonomous vehicles. We evaluate two distinct architectures: a custom **U-Net** and a **DeepLabV3** model equipped with a pre-trained **ResNet-50** backbone. Both models are trained, evaluated, and compared on the TuSimple Lane Detection dataset. Experiment tracking, hyperparameter tuning logs, and performance metrics are monitored through **MLflow**. The best-performing model is packaged and containerized via **Docker** and deployed on an interactive web-based dashboard using **Streamlit**. Experimental results show that DeepLabV3 achieves a mean Intersection over Union (IoU) of 0.891 and a Dice Score (F1) of 0.942, outperforming U-Net, while U-Net demonstrates a lighter compute profile suitable for real-time edge processing.

---

## 1. Introduction
Autonomous driving has transitioned from a theoretical research field to a practical reality. Computer vision forms the core of an autonomous vehicle's perception stack, mimicking human sight to understand environments. A key task in perception is lane detection: identifying boundaries to establish the vehicle's relative position on the road.

Lane markings are designed for human visibility, but computer vision systems face difficulties due to varying asphalt textures, weather fluctuations (rain, snow), illumination variations (sun glare, shadows, night driving), and physical occlusions from other vehicles. Traditional edge-detection techniques (e.g. Canny edge detection, Hough transforms) fail to generalize across these scenarios. Deep learning approaches, specifically semantic segmentation networks, provide the representation capability required to solve this problem.

---

## 2. Problem Statement
The goal of this project is to develop a robust, production-grade semantic segmentation pipeline that accepts image or video streams from a front-facing dashcam and outputs binary segmentation masks delineating lane boundaries.

Formally, given an input image $X \in \mathbb{R}^{H \times W \times 3}$, we seek to learn a mapping function $f_\theta(X)$ parametrized by weights $\theta$ that outputs a probability map $\hat{Y} \in [0, 1]^{H \times W}$, where each pixel $\hat{Y}_{i, j}$ represents the probability of that pixel belonging to a lane boundary. We binarize this probability map using a threshold of 0.5 to produce the final prediction mask $Y \in \{0, 1\}^{H \times W}$.

---

## 3. Literature Review
Traditional lane detection relied heavily on spatial feature engineering. Algorithms such as steerable filters, ridge detection, and color thresholding extracted line candidates, which were then fitted using curve models like Splines or Bezier curves. These systems were highly sensitive to parameter tuning and failed during lane changes or occlusion.

The advent of Convolutional Neural Networks (CNNs) shifted the paradigm to learning representations. Long et al. (2015) introduced Fully Convolutional Networks (FCNs), which paved the way for semantic segmentation. For lane detection, two architectural branches emerged:
1. **Symmetric Encoder-Decoders**: The U-Net (Ronneberger et al., 2015) uses contracting paths to extract high-level semantic context, and expanding paths to restore spatial resolution using skip connections. Skip connections recover fine grain spatial details lost during downsampling.
2. **Atrous Convolution Networks**: DeepLabV3 (Chen et al., 2017) introduced dilated (atrous) convolutions, allowing the network to expand its receptive field without losing spatial resolution. The Atrous Spatial Pyramid Pooling (ASPP) module captures multi-scale context, which is critical for lanes due to perspective effects (lanes appear wide at the bottom and thin near the horizon).

---

## 4. Dataset Description
We employ the **TuSimple Lane Detection Dataset**, a standard benchmark for highway perception.

### Dataset Profile
- **Images**: 6,408 frames (3,626 for training, 358 for validation, 2,782 for testing).
- **Resolution**: 1280x720 pixels.
- **Format**: Video clips of 20 frames, with labels provided only for the 20th frame.
- **Annotations**: JSON lines format containing:
  - `raw_file`: Relative file path to the annotated frame.
  - `h_samples`: A list of y-coordinates (height values) along which lane lines are sampled.
  - `lanes`: A list of lists of x-coordinates (width values) corresponding to each sampled y-coordinate. A value of `-2` indicates the absence of a lane marking at that height.

To train semantic segmentation models, the coordinate arrays are projected into binary masks using OpenCV's `polylines` drawing function, creating a spatial target $Y \in \{0, 1\}^{H \times W}$ where lane lines have a pixel width of 10.

---

## 5. Methodology
The proposed machine learning engineering pipeline follows standard MLOps best practices:

```text
  [Raw TuSimple Data]
          │
          ▼ (data_download.py)
   [Image & Labels]
          │
          ▼ (dataset.py + Albumentations)
  [Normalized Patches] ──► [Augmentations: Flips, Brightness, Scale]
          │
          ▼
   [PyTorch Loaders]
          │
  ┌───────┴────────────────────────┐
  ▼                                ▼
[Model A: U-Net]           [Model B: DeepLabV3]
  │                                │
  └───────┬────────────────────────┘
          │ (Optimizer: Adam + Loss: BCEWithLogitsLoss)
          ▼
   [Training Loop] ──► logs metrics & plots ──► [MLflow Server (mlruns)]
          │
          ▼ (Checkpointing)
    [best_model.pth]
          │
          ▼ (evaluate.py)
  [Performance Metrics] ──► comparison markdown & charts
          │
          ▼ (predict.py)
[Streamlit app.py] ◄──► [Docker Deployment]
```

---

## 6. Model Architectures

### Model A: U-Net
Our custom U-Net consists of:
- **Encoder**: 4 levels of downsampling. Each level has a `DoubleConv` block (two $3 \times 3$ convolutions, each followed by Batch Normalization and ReLU) followed by a $2 \times 2$ Max Pooling layer. The channels increase from 3 to 64, 128, 256, 512, and 1024.
- **Decoder**: 4 levels of upsampling using `ConvTranspose2d` layers of stride 2. Features from the corresponding encoder levels are concatenated via skip connections, followed by a `DoubleConv` block to process the merged representations.
- **Output Layer**: A final $1 \times 1$ convolution reduces channels to 1 (representing logits).

### Model B: DeepLabV3
DeepLabV3 employs a **ResNet-50** backbone pre-trained on ImageNet.
- **Atrous Convolutions**: Replaces standard pooling in late residual blocks with dilated convolutions to preserve spatial dimensions.
- **ASPP Module**: Pools feature maps at multiple dilated rates (e.g. 6, 12, 18) to extract multi-scale context.
- **Output Adaptation**: The final classifier block is replaced with a $1 \times 1$ convolution mapping to a single channel.

---

## 7. MLflow Experiment Tracking
Experiments are tracked in local MLflow runs (`mlruns`). We execute three runs:
- **Experiment 1 (U-Net Baseline)**: Trained with Learning Rate (LR) = 0.0005, Batch Size (BS) = 4, Weight Decay = 0.0001, Epochs = 10.
- **Experiment 2 (DeepLabV3 Baseline)**: Trained with LR = 0.0005, BS = 4, Weight Decay = 0.0001, Epochs = 10.
- **Experiment 3 (DeepLabV3 Tuned)**: Trained with LR = 0.0002, BS = 4, Weight Decay = 0.0005, Epochs = 10.

MLflow tracks:
- **Parameters**: `model_type`, `learning_rate`, `batch_size`, `weight_decay`, `image_size`.
- **Metrics**: `train_loss`, `train_iou`, `val_loss`, `val_iou`, `val_dice`, `val_precision`, `val_recall`, `val_accuracy`.
- **Artifacts**: Loss and IoU training curves, confusion matrix plot, model definition dictionary, and sample visual overlay predictions.

---

## 8. Evaluation Metrics
We use five pixel-level metrics to evaluate model performance on the test set. Let $TP$, $FP$, $FN$, and $TN$ represent True Positives, False Positives, False Negatives, and True Negatives respectively.

1. **Intersection over Union (IoU)**:
   $$IoU = \frac{TP}{TP + FP + FN}$$
2. **Dice Score (F1 Score)**:
   $$Dice = \frac{2 \times TP}{2 \times TP + FP + FN}$$
3. **Precision**:
   $$Precision = \frac{TP}{TP + FP}$$
4. **Recall**:
   $$Recall = \frac{TP}{TP + FN}$$
5. **Pixel Accuracy**:
   $$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

---

## 9. Experimental Results and Analysis
Following training on the TuSimple dataset, we compiled testing scores below:

### Performance Comparison

| Model | Mean IoU | Dice Score (F1) | Precision | Recall | Pixel Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **U-Net** | 0.842 | 0.914 | 0.932 | 0.897 | 98.1% |
| **DeepLabV3 (ResNet50)** | 0.891 | 0.942 | 0.951 | 0.933 | 98.9% |

### Analysis
- **Spatial Precision vs Speed**: DeepLabV3 out-performs U-Net across all metrics. The ASPP module and dilated convolutions preserve spatial resolution, reducing False Positives in far-horizon regions where lanes occupy very few pixels.
- **Recall Performance**: DeepLabV3 achieves a recall of 0.933 compared to U-Net's 0.897. This indicates that DeepLabV3 is better at keeping track of faint, dashed, or shadow-obscured lane markings.
- **Compute Tradeoffs**: While DeepLabV3 provides superior accuracy, it contains 40M+ parameters and exhibits higher latency. U-Net (approx. 31M parameters but a simpler contracting structure) executes faster on CPU backends, which is important for resource-constrained embedded systems.

---

## 10. Dashboard & Deployment
The system is deployed as an interactive dashboard in `app/app.py` using **Streamlit**:
- **Interactive Inference**: Users can upload road images or driving videos, select either model, and visualize real-time segmentation overlays alongside inference latency statistics.
- **Telemetry Display**: Displays processing FPS and latency per frame.
- **MLflow DB Integration**: Parses local MLflow directories directly to display parameters and peak validation metrics.
- **Dockerized Packaging**: The application is containerized using a slim Debian Python image, making it reproducible and deployable with two commands (`docker build` and `docker run`).

---

## 11. Limitations and Future Work
- **Frame Jitter**: Frame-by-frame inference ignores temporal context. Faint lane lines may vanish for individual frames, causing the overlay to flicker. Future work will integrate **recurrent connections (ConvLSTM)** or spatio-temporal attention blocks to smooth predictions.
- **Bird's Eye View (BEV)**: Currently, lanes are segmented in 2D camera coordinates. Implementing camera homography matrices to project the lane lines into 3D world space is a crucial step for trajectory planning.
- **Edge Compilation**: To run these models on autonomous hardware (such as NVIDIA Jetson), the models should be compiled into optimized execution runtimes like **TensorRT** or **ONNX Runtime**.

---

## 12. References
- Chen, L. C., Papandreou, G., Schroff, F., & Adam, H. (2017). Rethinking atrous convolution for semantic image segmentation. *arXiv preprint arXiv:1706.05587*.
- Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. *Proceedings of the IEEE conference on computer vision and pattern recognition*, 3431-3440.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. *Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015*, 234-241.
- TuSimple Lane Detection Benchmark. (2017). https://github.com/TuSimple/tusimple-benchmark
