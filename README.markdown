# 23M1079_MTP - Neural Collapse Analysis with Multiple Architectures

## Overview

This repository, developed by Devayan (23M1079), contains Python code for analyzing Neural Collapse (NC) properties across various neural network architectures on multiple datasets. The architectures include ResNet, Vision Transformer (ViT), Swin Transformer, ConvNeXt, WaveMix, MambaOut Kobe, Vision Mamba, RegNet, MobileNetV3, ShuffleNet, and EfficientNetV2. The experiments evaluate NC behavior by applying NC loss on the penultimate layer and multiple intermediate layers, compared against a baseline with standard cross-entropy loss. The datasets used are CIFAR-10, EuroSAT, Food-101, and DTD.

The code is designed to:

- Train models on specified datasets with cross-entropy loss (baseline).
- Apply NC loss on the penultimate layer.
- Apply NC loss on a combination of intermediate and final layers.
- Analyze where NC enforcement improves or hurts performance.

## Repository Structure

```
23M1079_MTP/
├── resnet.py                      # NC analysis with ResNet
├── vit.py                         # NC analysis with Vision Transformer
├── swin_transformer.py            # NC analysis with Swin Transformer
├── convnext.py                    # NC analysis with ConvNeXt
├── wavemix.py                     # NC analysis with WaveMix
├── mambaout_kobe.py               # NC analysis with MambaOut Kobe
├── vision_mamba.py                # NC analysis with Vision Mamba
├── regnet.py                      # NC analysis with RegNet
├── mobilenetv3.py                 # NC analysis with MobileNetV3
├── shufflenet.py                  # NC analysis with ShuffleNet
├── efficientnetv2.py              # NC analysis with EfficientNetV2
├── requirements.txt               # Python dependencies
├── README.md                      # This file
```

## Datasets

The code supports the following datasets (downloaded automatically via `torchvision.datasets`):

- **CIFAR-10**: 10 classes, 50,000 training images.
- **EuroSAT**: 10 classes, satellite imagery.
- **Food-101**: 101 classes, food images (commented out by default).
- **DTD**: 47 classes, texture images (commented out by default).

To enable Food-101 and DTD, uncomment the relevant lines in the `datasets` dictionary in each script.

## Requirements

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

See `requirements.txt` for the list of dependencies.

## Usage

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Devayan21/23M1079_MTP.git
   cd 23M1079_MTP
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run an Experiment**: Each architecture has its own script. To run an experiment, execute:

   ```bash
   python convnext.py
   ```

   Replace `convnext` with the desired architecture (e.g., `resnet`, `vit`, `mobilenetv3`).

4. **Output**:

   - Log files are saved in the current directory with timestamps (e.g., `convnextbase_nc_analysis_20250629_2121.log`).
   - The console and log files display training progress, epoch-wise loss, accuracy, and NC loss (if applicable).
   - The final analysis compares baseline, penultimate NC, and multi-layer NC performance for each dataset.

## Experiment Details

Each script performs three experiments:

- **Baseline**: Trains only the final classifier layer with cross-entropy loss.
- **Penultimate NC**: Trains the entire model with NC loss applied on the penultimate layer (`nc_lambda=0.3`).
- **Multi-Layer NC**: Trains the entire model with NC loss applied on multiple layers (e.g., `stem`, `stages`, `norm`, `head.global_pool` for ConvNeXt) with layer-specific weights (`nc_layer_lambdas`).

The NC loss consists of:

- **Within-Class Loss**: Minimizes variance within each class.
- **ETF Loss**: Encourages between-class orthogonality.
- **Norm Loss**: Ensures equal norms for class means.

The analysis section reports whether NC enforcement improves or hurts accuracy compared to the baseline.

## Notes

- The default configuration uses 1 epoch for quick experimentation. Increase `num_epochs` in the script for better convergence.
- Food-101 and DTD are commented out due to their larger size and number of classes, which may require more computational resources.
- The code uses mixed precision training (`torch.amp`) for efficiency on CUDA-enabled GPUs.
- Potential issues (e.g., empty class handling, gradient accumulation) are not addressed in the default scripts to preserve the original logic. Check log files for warnings.

## Example Output

For ConvNeXt on CIFAR-10 and EuroSAT:

```
=== Analysis: Where Does NC Enforcement Help? ===

CIFAR-10:
Penultimate NC improves performance (Baseline: 85.20%, Penultimate NC: 85.70%, Change: 0.50%)
Multi-Layer NC hurts performance (Baseline: 85.20%, Multi-Layer NC: 84.90%, Change: -0.30%)

EuroSAT:
Penultimate NC improves performance (Baseline: 87.80%, Penultimate NC: 88.30%, Change: 0.50%, NC Loss: 0.0125)
Multi-Layer NC improves performance (Baseline: 87.80%, Multi-Layer NC: 88.10%, Change: 0.30%, NC Loss: 0.0118)
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for bugs, improvements, or additional architectures. Contact Devayan (23M1079) for collaboration.