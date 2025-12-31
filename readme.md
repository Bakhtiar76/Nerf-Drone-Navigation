# NeRF — Representing Scenes as Neural Radiance Fields for View Synthesis  
### Implementation by S. Bakhtiar Ahmed (24K-7622)

---

## Overview
This repository contains the implementation of **NeRF (Neural Radiance Fields)** for view synthesis, following the paper:

> **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis**  
> *Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng*  

Our implementation reproduces the paper’s results on the **Lego dataset** using **PyTorch**, trained for 200K iterations on an **NVIDIA RTX 3060 GPU** (≈8 hours).

---

## Setup Instructions

### Create and activate environment
```bash
conda create -n nerf_simple python=3.8 -y
conda activate nerf_simple
pip install -r requirements.txt
```


## Training Instructions

### To train NeRF on the Lego dataset
```bash
python run_nerf.py --config configs/lego.txt
```
### Training runs for 200,000 iterations and saves model checkpoints & render video to:
```bash
logs\lego\200000.tar
logs\lego\blender_paper_lego_spiral_200000_rgb.mp4
```


## Test Rendering
### To render test views from the trained model
```bash
python run_nerf.py --config configs/lego.txt --render_only --render_test
```


## Evaluation
### To evaluate the trained model and compute PSNR, SSIM, and LPIPS
```bash
python evaluation.py
```
