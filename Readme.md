# AI-Enhanced 3D Reconstruction from 2D Drone Imagery for Autonomous Drone Navigation in Complex Environments

This repository implements a hybrid pipeline that combines **reconstructs accurate 3D scenes from 2D drone imagery** using **Instant-NGP** and enables **safe autonomous navigation** in complex environments via **Reinforcement Learning-based differentiable trajectory optimization**.

The entire system runs in a Linux environment (WSL2) with full GPU acceleration, ensuring reproducible setup.

---

## Dataset Downloads

Download the datasets and place them inside a `data/nerf_synthetic` folder at the root of the repository.

### 1. LEGO Dataset (for 3D reconstruction testing)
Link: (https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4)  
Place in:  
`data/nerf_synthetic/lego/`

### 2. Stonehenge Synthetic Dataset (for navigation tasks)
Link: (https://drive.google.com/drive/folders/1yPItq_6C_zqV2WyzqMZdMcJnXXRODYx3)  
Place in:  
`data/nerf_synthetic/stonehenge/`

---
## Environment Setup

### 1. WSL2 Environment Setup
Enable WSL2:
```bash
wsl --install
```

Verify GPU in WSL
```bash
nvidia-smi
```

### 2. Create Conda Environment
```bash
# Create the environment from the provided yaml file
conda env create -f environment.yml
conda activate nerfnav
```

### 3. tiny-cuda-nn Installation
Since tiny-cuda-nn requires specific CUDA bindings, install it manually from the source:

```bash

git clone https://github.com/NVlabs/tiny-cuda-nn
cd tiny-cuda-nn/bindings/torch
python setup.py install
cd ../../../
```

### 4. Rebuild Compiled NeRF Modules
These modules are required for Instant-NGP speed and CUDA raymarching.

```bash
# 1. Raymarching (CUDA volume rendering)
cd raymarching
python setup.py install
cd ..

# 2. GridEncoder (HashGrid encoding)
cd gridencoder
python setup.py install
cd ..

# 3. SHEncoder (Spherical Harmonics)
cd shencoder
python setup.py install
cd ..

``` 

### 5. Create Required Directories
```bash
mkdir -p sim_img_cache paths
```


### 6. Blender Simulation Setup
Blender is used as a photorealistic ground-truth renderer for pose estimation.

*   Install the latest version of Blender
*   Ensure blender is available in your terminal/PATH
*   Scene must contain a Camera object

Install Blender in WSL
```bash
sudo apt update
sudo apt install -y blender libgl1 libxi6 libxrender1 libxkbcommon0 libsm6 libice6
```

Verify:
```bash
blender --version
```

### Install NumPy for Blender’s Python
Blender uses its own Python runtime.
```bash
sudo /usr/bin/python3.12 -m pip install numpy --break-system-packages
```

---

## Running the Pipeline

### 1. Train the NeRF Model (Stonehenge)
This step learns the neural 3D scene representation used for navigation.
```bash
python main_nerf.py data/nerf_synthetic/stonehenge --workspace stonehenge_nerf -O --bound 2 --scale 1.0 --dt_gamma 0
```
Notes:
- -O enables fp16, CUDA raymarching, and data preloading.
- Training stops automatically after --iters (default: 30,000).

Outputs are saved in:
- `stonehenge_nerf`

### 2. Run Simulation & Visualization
This runs the closed-loop navigation system using:

- NeRF density gradients
- Visual pose estimation
- Differentiable trajectory optimization
```bash
python simulate.py data/nerf_synthetic/stonehenge --workspace stonehenge_nerf -O --bound 2 --scale 1.0 --dt_gamma 0
```
Results (planned vs actual trajectories, renders) are saved in:
- `paths`

---

## System Requirements

* NVIDIA GPU (tested on RTX 3060 / 3080 / 4090)
* Docker with NVIDIA Container Toolkit (--gpus all support)
* CUDA 11.7+ and compatible drivers

---

## Key Components
- Trajectory Optimization: Uses the NeRF's density field to calculate gradients for collision-free paths.
- Feature-Based Estimator: Uses visual landmarks from Blender renders to correct the drone's pose.
- Visualization: Results, including planned vs. actual trajectories, are saved in the paths/ directory.