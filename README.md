# SH-GAN: Enhanced Image Inpainting via Self-Similarity Haar Transform and Multi-Scale Feature Feedback GAN

This repository contains the official implementation of the paper **"Enhanced Image Inpainting via Self-Similarity Haar Transform and Multi-Scale Feature Feedback GAN"**, submitted to *The Visual Computer* 

## Overview
SH-GAN addresses the challenges of texture distortion, edge blurring, and color inconsistency in complex damaged region image inpainting by integrating:
- A **Self-similarity Haar Transform Module (SHTM)** for noise-robust structural initialization of damaged regions.
- A **Multi-scale Feature Feedback GAN (MFFGAN)** with hierarchical multi-head attention for cross-scale feature fusion.
- A **hierarchical weighted adaptive loss mechanism** to balance global structural consistency and local texture naturalness.

##  Environment Setup
### Prerequisites
- Python 3.8+
- PyTorch 1.12.0+
- CUDA 11.3+ (recommended for GPU acceleration)
- Other dependencies:
  ```bash
  pip install opencv-python==4.6.0.66 numpy==1.23.5 scipy==1.9.3 pillow==9.3.0 torchvision==0.13.1 tqdm==4.64.1
##  Datasets
The experiments are conducted on three public benchmark datasets:
Places2: http://places2.csail.mit.edu/
CelebA: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
Facade: https://cmp.felk.cvut.cz/~tylecr1/facade/

## Training
Set your dataset path in configs/default.yaml
Run:python train.py --config configs/default.yaml

## Testing
Run:python test.py --config configs/default.yaml --checkpoint path/to/your/model.pth

