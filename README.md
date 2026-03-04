# SH-GAN: Enhanced Image Inpainting via Self-Similarity Haar Transform and Multi-Scale Feature Feedback GAN

This repository contains the official implementation of the paper **"Enhanced Image Inpainting via Self-Similarity Haar Transform and Multi-Scale Feature Feedback GAN"**, submitted to *The Visual Computer* (Submission ID: [你的投稿ID，如e90976bc-7e1e-4aa0-a21a-b1596b01d166]).

## 📋 Overview
SH-GAN addresses the challenges of texture distortion, edge blurring, and color inconsistency in complex damaged region image inpainting by integrating:
- A **Self-similarity Haar Transform Module (SHTM)** for noise-robust structural initialization of damaged regions.
- A **Multi-scale Feature Feedback GAN (MFFGAN)** with hierarchical multi-head attention for cross-scale feature fusion.
- A **hierarchical weighted adaptive loss mechanism** to balance global structural consistency and local texture naturalness.

Experimental results on Facade, CelebA, and Places2 datasets demonstrate that SH-GAN outperforms six state-of-the-art methods in LPIPS, FID, PSNR, and SSIM metrics.

## 🛠️ Environment Setup
### Prerequisites
- Python 3.8+
- PyTorch 1.12.0+
- CUDA 11.3+ (recommended for GPU acceleration)
- Other dependencies:
  ```bash
  pip install opencv-python==4.6.0.66 numpy==1.23.5 scipy==1.9.3 pillow==9.3.0 torchvision==0.13.1 tqdm==4.64.1
