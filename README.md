# Salt-and-Pepper Noise Removal Using Spatial Filters

## Overview

This experiment investigates the removal of **Salt-and-Pepper noise** from digital images using three different approaches:

1. Classical spatial filtering (Median Filter)
2. Learning-based denoising (CNN using PyTorch)
3. A research-inspired adaptive spatial filter

The goal is to compare these methods both **quantitatively** (PSNR, SSIM) and **qualitatively** (visual inspection).

---

## Noise Model

Salt-and-Pepper noise is an impulsive noise model where random pixels are corrupted to extreme values (0 or 255).  
Noise is synthetically added at different densities (10%, 30%, 50%, 70%) to ensure controlled experimentation.

---

## Methods Implemented

### 1. Median Filter (Baseline)

- Nonlinear spatial filter
- Replaces each pixel with the median of its neighborhood
- Effective at low noise levels
- Tends to blur edges at higher noise densities

Implemented using **OpenCV and NumPy**.

---

### 2. CNN-Based Denoising

- Uses a convolutional neural network trained on noisy-clean image pairs
- Learns noise patterns directly from data
- Preserves edges and textures better than classical filters

Implemented using **PyTorch**.

---

### 3. Adaptive Spatial Filter (Paper-Inspired)

- Two-stage approach:
  1. Initial median filtering
  2. Adaptive averaging applied only to noisy pixels
- Preserves clean pixels while correcting corrupted ones
- Inspired by recent open-access research on detail-aware spatial filtering

Implemented using **NumPy and OpenCV**.

---

## Evaluation Metrics

- **PSNR (Peak Signal-to-Noise Ratio)**
- **SSIM (Structural Similarity Index)**

Metrics are computed between the denoised image and the original clean image.

---

## Folder Structure
```
salt_pepper_noise_removal/
│
├── requirements.txt
├── README.md
│
├── data/
│   ├── original/
│   │   ├── grayscale/
│   │   └── colored/
│   │
│   ├── noisy/
│   │   ├── grayscale/
│   │   └── colored/
│   │
│   └── clean/
│       ├── median/
│       │   ├── grayscale/
│       │   └── colored/
│       │
│       ├── cnn/
│       │   ├── grayscale/
│       │   └── colored/
│       │
│       └── adaptive/
│           ├── grayscale/
│           └── colored/
│
├── baseline/
│   └── median_filter.py
│
├── cnn/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── infer.py
│
├── paper-method/
│   └── adaptive_filter.py
│
├── utils/
│   ├── add_noise.py
│   └── metrics.py
│
└── run_experiment.py
```

---

## How to Run

1. Add clean images to:
