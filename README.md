# OncoSight-AI

OncoSight AI is a multi-modal AI research platform that analyzes medical images and clinical data to support cancer-related research using machine learning and deep learning models.

# OncoSight AI – Lung CT Segmentation Module

**An AI-powered research prototype for lung nodule segmentation using CT scans and deep learning.**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-CUDA%20Enabled-ee4c2c?logo=pytorch" />
  <img src="https://img.shields.io/badge/Model-2D%20U--Net-purple" />
  <img src="https://img.shields.io/badge/Status-In%20Development-orange" />
</p>

## 1. Project Overview

OncoSight AI is a medical imaging research project focused on developing an AI-based lung nodule segmentation system.

This module uses CT scan volumes from the **LIDC-IDRI dataset** to identify and segment lung nodule regions using deep learning techniques.

The primary objective is to develop a pipeline that can:

- Load and process lung CT scan volumes.
- Prepare corresponding segmentation masks.
- Train a U-Net-based deep learning model.
- Predict lung nodule regions from CT images.
- Evaluate segmentation performance using Dice Score and Intersection over Union (IoU).
- Generate visual segmentation overlays for analysis.

**Important:** This project is an academic research prototype and is not intended for clinical diagnosis or medical decision-making.

## 2. Problem Statement

Manual identification and segmentation of lung nodules in CT scans can be time-consuming and challenging.

The objective of this project is to develop a deep learning-based segmentation system that learns from annotated CT images and automatically identifies candidate lung nodule regions.

The system aims to support medical imaging research by providing automated segmentation predictions and quantitative evaluation.

## 3. Technology Stack

| Component                | Technology         |
| ------------------------ | ------------------ |
| Programming Language     | Python             |
| Deep Learning            | PyTorch            |
| Model Architecture       | 2D U-Net           |
| Medical Image Processing | NiBabel            |
| Numerical Processing     | NumPy              |
| Data Analysis            | Pandas             |
| Image Processing         | SciPy              |
| Visualization            | Matplotlib         |
| Dataset                  | LIDC-IDRI          |
| GPU Acceleration         | NVIDIA CUDA        |
| Development Environment  | Visual Studio Code |

## 4. Dataset

The project uses the **LIDC-IDRI (Lung Image Database Consortium and Image Database Resource Initiative)** dataset.

Dataset source: [MedOtter/LIDC-IDRI – Hugging Face](https://huggingface.co/datasets/MedOtter/LIDC-IDRI)

### Dataset information

- Approximately 1,018 CT scans.
- CT images stored in NIfTI format (`.nii.gz`).
- Corresponding segmentation masks.
- Multiple radiologist annotations available.
- Default masks represent consensus nodule annotations.

### Current dataset selection

A pilot dataset of 10 selected CT scans and their corresponding masks is being used to develop and test the pipeline.

The pilot dataset is intended for initial experimentation, debugging and model development. It is not sufficient to establish reliable generalization performance.

## 5. Project Architecture

```text
                 LIDC-IDRI DATASET
                         |
                         v
              CT SCANS + SEGMENTATION MASKS
                         |
                         v
                 DATA VALIDATION
                         |
                         v
                  PREPROCESSING
             (HU Windowing and Normalization)
                         |
                         v
                PROCESSED CT VOLUMES
                         |
                         v
              PATIENT-LEVEL DATA SPLIT
                         |
                         v
                  2D U-NET MODEL
                         |
                         v
                 MODEL TRAINING
                         |
                         v
                SEGMENTATION PREDICTION
                         |
                         v
                 MODEL EVALUATION
                         |
                         v
               DICE SCORE AND IoU
                         |
                         v
                 VISUAL OVERLAYS
```

## 6. Project Directory Structure

```text
OncoSight-AI/
│
├── datasets/
│   ├── raw/
│   │   ├── images/
│   │   └── masks/
│   │
│   ├── processed/
│   │   └── lung_ct/
│   │
│   ├── validation/
│   │   └── pilot_overlays/
│   │
│   └── manifests/
│       ├── pilot_cases.json
│       ├── batch_validation.json
│       └── preprocessing_report.json
│
├── scripts/
│   ├── inspect_lidc_hf.py
│   ├── preprocess_lung_ct.py
│   ├── preprocess_pilot.py
│   ├── probe_row_order.py
│   ├── select_pilot_cases.py
│   ├── split_pilot_dataset.py
│   ├── validate_all_cases.py
│   ├── validate_image_mask.py
│   └── validate_nifti_case.py
│
├── README.md
└── .gitignore
```

Additional training and evaluation scripts will be added as development progresses.

## 7. Data Preprocessing

The preprocessing pipeline performs the following operations:

1. Load CT images and segmentation masks from NIfTI files.
2. Verify image and mask dimensions.
3. Check affine alignment between images and masks.
4. Check for invalid image values.
5. Verify that segmentation masks contain binary values.
6. Clip CT intensity values to the selected Hounsfield Unit window of -1000 to 400 HU.
7. Normalize the image intensities.
8. Identify slices containing annotated nodules.
9. Save processed volumes as compressed NumPy `.npz` files.
10. Generate segmentation overlay images for visual inspection.

### Preprocessing command

Run from the project root:

```bash
python -u scripts/preprocess_lung_ct.py --repo-root .
```

Processed volumes are saved in:

```text
datasets/processed/lung_ct/
```

Overlay images are saved in:

```text
datasets/validation/pilot_overlays/
```

The preprocessing report is generated at:

```text
datasets/manifests/preprocessing_report.json
```

## 8. Deep Learning Model

The planned segmentation architecture is **2D U-Net**.

U-Net is a convolutional neural network architecture commonly used for biomedical image segmentation.

It consists of:

- Encoder: extracts image features.
- Bottleneck: captures deeper contextual information.
- Decoder: reconstructs spatial information.
- Skip connections: preserve fine image details.

The model will take preprocessed CT slices as input and predict pixel-level lung nodule segmentation masks.

### Planned training configuration

| Parameter        | Initial configuration            |
| ---------------- | -------------------------------- |
| Architecture     | 2D U-Net                         |
| Input resolution | 256 × 256                        |
| Batch size       | 2 (initial setting)              |
| Loss function    | Dice Loss + Binary Cross Entropy |
| Optimizer        | Adam                             |
| Hardware         | NVIDIA RTX 3050 Laptop GPU       |
| Framework        | PyTorch                          |
| Device           | CUDA                             |

These are initial experimental settings and may be adjusted during training.

## 9. Model Evaluation

The segmentation model will be evaluated using:

### Dice Similarity Coefficient

Measures the overlap between the predicted segmentation and the ground-truth mask.

### Intersection over Union (IoU)

Measures the ratio of the intersection to the union of the predicted and ground-truth regions.

Additional evaluation may include:

- Precision
- Recall
- Training and validation loss
- Visual comparison of predicted masks and ground-truth masks

## 10. Hardware Configuration

Development environment:

| Component        | Configuration                      |
| ---------------- | ---------------------------------- |
| GPU              | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM             | 4 GB                               |
| PyTorch          | 2.14.0+cu126                       |
| CUDA Runtime     | 12.6                               |
| GPU Acceleration | Verified                           |

CUDA tensor operations have been successfully tested.

GPU acceleration will primarily be used for model training and inference.

## 11. Installation and Setup

### Step 1: Clone the repository

```bash
git clone <https://github.com/OncoSight-AI/OncoSight-AI.git>
cd OncoSight-AI
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
pip install numpy pandas nibabel scipy matplotlib huggingface_hub torch
```

Install the PyTorch build compatible with your NVIDIA GPU and CUDA environment if needed. Verify CUDA availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Step 4: Run validation

```bash
python -u scripts/validate_all_cases.py --repo-root .
```

### Step 5: Run preprocessing

```bash
python -u scripts/preprocess_lung_ct.py --repo-root .
```

## 12. Current Development Status

| Development Stage               | Status                        |
| ------------------------------- | ----------------------------- |
| Dataset selection               | Completed                     |
| Pilot dataset download          | Completed                     |
| Image-mask validation           | Completed – 10/10 pilot cases |
| Preprocessing implementation    | Available                     |
| Pilot preprocessing             | In progress                   |
| Patient-level dataset splitting | Pending completion            |
| 2D U-Net implementation         | Planned                       |
| GPU model training              | Planned                       |
| Model evaluation                | Planned                       |
| Prediction visualization        | Planned                       |

## 13. Future Improvements

- Expand the pilot dataset to a larger number of eligible CT scans.
- Implement and optimize the 2D U-Net architecture.
- Introduce data augmentation.
- Improve segmentation performance through hyperparameter tuning.
- Evaluate model generalization on patient-level held-out data.
- Develop an interactive interface for uploading CT images and viewing segmentation results.
- Generate downloadable research reports and segmentation visualizations.

## 14. Research and Disclaimer

This project uses the LIDC-IDRI public research dataset for academic experimentation.

The segmentation outputs are experimental predictions. They should not be interpreted as confirmed medical findings, cancer diagnoses or recommendations for treatment.

All model performance claims must be supported by experimental evaluation on appropriately separated patient-level datasets.

---

**OncoSight AI – Lung CT Segmentation Module**

_Building an AI-assisted approach to medical image segmentation through deep learning._
