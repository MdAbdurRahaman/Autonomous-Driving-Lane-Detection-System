# Autonomous Driving Lane Detection System

An end-to-end, production-ready computer vision system that trains, tracks, evaluates, and deploys lane segmentation models for autonomous driving perception. Built with **PyTorch**, **MLflow**, **Streamlit**, and **Docker** using the **TuSimple Lane Detection Dataset**.

---

## 1. Project Overview & Problem Statement

Lane detection is a core task in autonomous driving perception. It serves as the foundation for downstream tasks like **Lane Keep Assist (LKA)**, **Lane Departure Warning (LDW)**, and path planning. Autonomous vehicles must reliably identify lane boundaries under variable lighting conditions, camera distortions, and weather changes.

This project implements semantic road lane segmentation. It evaluates two deep learning models—a custom **U-Net** and a **DeepLabV3** with a ResNet50 backbone. Experiments are tracked in **MLflow** to identify optimal hyperparameter setups, and the best-performing model is deployed in a web-based **Streamlit** dashboard.

---

## 2. Project Architecture

The system is structured as follows:

```text
final-project/
├── README.md                 # Detailed instructions and run guide
├── requirements.txt          # Python package requirements
├── Dockerfile                # Production multi-stage dockerfile
├── .gitignore                # Excludes datasets, local weights, and caches
├── final_report.md           # Formal academic/engineering report
│
├── configs/
│   └── config.yaml           # Centralized configuration file (epochs, lr, paths)
│
├── src/
│   ├── data_download.py      # Automates data download (sample or full dataset)
│   ├── dataset.py            # Custom TuSimple PyTorch dataset and loaders
│   ├── models.py             # U-Net and DeepLabV3 model definitions
│   ├── metrics.py            # Vectorized metrics (IoU, Dice/F1, Recall, etc.)
│   ├── utils.py              # Blending overlays, training curves, config loading
│   ├── train.py              # Core training loop with MLflow run logging
│   ├── evaluate.py           # Benchmark script to calculate test set scores
│   └── predict.py            # Executable inference on image, folder, or video
│
├── app/
│   └── app.py                # Multi-page Streamlit dashboard app
│
├── notebooks/
│   └── exploration.ipynb     # Interactive Jupyter exploration notebook
│
├── mlruns/                   # Local MLflow experiments store
├── models/                   # Saved model checkpoints (.pth)
├── artifacts/                # Comparison charts and performance logs
├── data/                     # Raw datasets (excluded from git)
└── outputs/                  # Inference results (masks, overlays, videos)
```

---

## 3. Installation Steps

### Prerequisites
- Python 3.10 or higher
- CUDA-enabled GPU (optional but recommended; system falls back to CPU automatically)

### Setup Virtual Environment
```bash
# Clone this repository and navigate into it
cd "Autonomous Driving Lane Detection System"

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 4. Dataset Download Instructions

The system uses the **TuSimple Lane Detection Benchmark**. The script `src/data_download.py` automates the download process.

To test the system immediately without downloading 12GB of zip files, use **Sample Mode** (default). This downloads 5 road images and generates the correct TuSimple folder structure and annotations locally in seconds.

### Download Sample/Demo Set (Recommended for Quick Verification)
```bash
python src/data_download.py --mode sample
```

### Download Full Official Dataset (~12GB)
```bash
python src/data_download.py --mode full
```

---

## 5. Training Commands

The training script `src/train.py` handles model training and validation. It reads settings from `configs/config.yaml`.

You can train single models or trigger all three certification experiments sequentially using `--run_all`.

### Run All 3 Experiments Sequentially (Recommended)
This command runs:
- **Experiment 1**: U-Net Baseline
- **Experiment 2**: DeepLabV3 Baseline
- **Experiment 3**: DeepLabV3 Tuned Hyperparameters
```bash
python src/train.py --run_all
```

### Train a Single Model (U-Net)
```bash
python src/train.py --model_type unet --epochs 5 --lr 0.0005
```

### Train a Single Model (DeepLabV3)
```bash
python src/train.py --model_type deeplabv3 --epochs 5 --lr 0.0005
```

---

## 6. Evaluation Commands

Evaluate the best checkpoints of U-Net and DeepLabV3 on the test set. This generates comparison tables and metric bar charts.

```bash
python src/evaluate.py
```
Outputs are written to `artifacts/`:
- `artifacts/model_comparison.md`
- `artifacts/model_comparison.png`
- `artifacts/comparison_sample_*.png`

---

## 7. Inference & Prediction Commands

Run inference on road images or driving videos. Outputs are saved automatically to the `outputs/` folder.

### 1. Single Image Inference
```bash
python src/predict.py --image data/clips/0530/1492626047222176189_0/20.jpg --model_type deeplabv3
```

### 2. Batch Folder Inference
```bash
python src/predict.py --dir data/clips/0530/1492626047222176189_0/ --model_type unet
```

### 3. Driving Video Inference
```bash
python src/predict.py --video path/to/driving_video.mp4 --model_type deeplabv3
```

---

## 8. MLflow Experiment Tracking UI

To inspect hyperparameter logs, training curves, confusion matrices, and validation overlays, run the MLflow UI:

```bash
mlflow ui
```
Open your browser and navigate to: `http://127.0.0.1:5000`

---

## 9. Streamlit Dashboard Usage

Launch the visual dashboard to test image uploads, video segmentation, check model metrics, and view local MLflow run logs.

```bash
streamlit run app/app.py
```
Open your browser and navigate to: `http://localhost:8501`

---

## 10. Docker Deployment Commands

Deploy the dashboard inside a container. The image compiles OpenCV system libraries, video codecs, copies training checkpoints, and runs the Streamlit server automatically.

### Build the Image
```bash
docker build -t lane-detector .
```

### Run the Container
```bash
docker run -p 8501:8501 lane-detector
```
Access the app on: `http://localhost:8501`

---

## 11. Project Evaluation & Results

Typical comparative performance on the TuSimple testing set:

| Model | Mean IoU | Dice Score (F1) | Precision | Recall | Pixel Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **U-Net** | 0.842 | 0.914 | 0.932 | 0.897 | 98.1% |
| **DeepLabV3 (ResNet50)** | 0.891 | 0.942 | 0.951 | 0.933 | 98.9% |

### Key Findings
- **DeepLabV3** achieves higher accuracy and mean IoU thanks to its Atrous Spatial Pyramid Pooling (ASPP) module and ImageNet pre-trained ResNet-50 encoder.
- **U-Net** has a lighter weight profile, offering faster frame-by-frame processing speed on CPU backends, making it highly suitable for lightweight edge compute targets.

---

## 12. Limitations & Future Work
- **Temporal Consistency**: Since models segment frame-by-frame, video overlays can show jitter. Utilizing temporal models (ConvLSTM/transformer tracking) will stabilize outputs.
- **3D Ground Planes**: The present system detects lanes in 2D pixel space. Future work involves projecting lanes into 3D world space (bird's-eye view) using camera homography matrices.
