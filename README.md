# AdaDPIGU

This repository provides the official implementation of the paper:  
**AdaDPIGU: Differentially Private SGD with Adaptive Clipping and Importance-Based Gradient Updates for Deep Neural Networks**.

---

## Installation

**Requirements:**  
- Python 3.8 or above  
- Linux system (tested), CUDA 11.0 recommended

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/adadpigu.git
   cd adadpigu
2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
# Environment
This code is tested on Linux system with CUDA version 11.0

To run the source code, please first install the following packages:
```
torch>=2.0.0
torchvision>=0.15.0
backpack-for-pytorch>=1.7.0
opacus>=1.5.0
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
pandas>=1.3.0
tqdm>=4.0.0
```
## Quick Start

Below are example commands for reproducing the experimental results on MNIST, FMNIST, and CIFAR-10:

### MNIST
```
python main.py --dataset mnist  --private --eps 4 --delta 1e-5 --sess adadpigu_mnist
```
```
python main.py --dataset fmnist  --private --eps 4 --delta 1e-5 --sess adadpigu_fmnist
```
```
python main.py --dataset cifar10  --private --eps 4 --delta 1e-5 --sess adadpigu_cifar10
```
