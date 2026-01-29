# ZECO: ZeroFusion Guided 3D MRI Conditional Generation

Official PyTorch implementation of **ZECO: ZeroFusion Guided 3D MRI Conditional Generation**.

## Overview

ZECO is a novel framework for 3D MRI conditional generation that leverages zero-shot fusion techniques to generate high-quality medical images. This repository contains the training and inference code for the VQVAE autoencoder, Latent Diffusion Models (LDM), and 3D Multi-modal Conditional Generation (3MCG) components.

**Star this repository if you find it helpful!** 🌟


## Features

- **3D VQVAE**: Vector Quantized Variational AutoEncoder for 3D medical image compression
- **Latent Diffusion Models**: High-quality latent space diffusion for both FLAIR and T1 modalities
- **3MCG Framework**: Multi-modal conditional generation with mask-based control
- **BraTS Dataset Support**: Built-in support for BraTS 2020 dataset


## Installation

### Prerequisites

- Python >= 3.8
- CUDA >= 11.0 (for GPU training)
- 16GB+ GPU memory recommended

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/zeco.git
cd zeco
```

2. Create a virtual environment (recommended):
```bash
conda create -n zeco python=3.9
conda activate zeco
```

3. Install PyTorch (choose the appropriate version for your CUDA):
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Install MONAI (if not already installed):
```bash
pip install monai[all]
```


This will check all dependencies and ensure your environment is ready for training.

## Data Preparation

### BraTS 2020 Dataset

1. Download the BraTS 2020 dataset from the [official website](https://www.med.upenn.edu/cbica/brats2020/data.html)

2. Extract the dataset to your data directory:
```bash
# Expected structure:
# /path/to/dataset/brat20/MICCAI_BraTS2020_TrainingData/
#   ├── BraTS20_Training_001/
#   │   ├── BraTS20_Training_001_flair.nii
#   │   ├── BraTS20_Training_001_t1.nii
#   │   ├── BraTS20_Training_001_t1ce.nii
#   │   ├── BraTS20_Training_001_t2.nii
#   │   └── BraTS20_Training_001_seg.nii
#   ├── BraTS20_Training_002/
#   └── ...
```

3. Update the data path in training scripts:
```python
# In scripts/train/*.py, modify:
train_data_dir = "/path/to/your/dataset/brat20/MICCAI_BraTS2020_TrainingData"
```

### Data Format

- **Image files**: `.nii` or `.nii.gz` format
- **Expected modalities**: FLAIR, T1, T1CE, T2
- **Segmentation masks**: Multi-class tumor segmentation
- **Spatial size**: Original images are resampled to 96×96×64

## Training

### Stage 1: Train VQVAE

Train the vector quantized autoencoder first:

```bash
# For FLAIR modality (channel=0)
python scripts/train/train_vqvae.py

# For T1 modality, modify channel parameter in the script:
# channel = 1  # 0=FLAIR, 1=T1, 2=T1CE, 3=T2
```

**Key hyperparameters:**
- Batch size: 2
- Epochs: 20
- Learning rate: 1e-4
- Input size: 96×96×64
- Latent space: 12×12×8

### Stage 2: Train Latent Diffusion Model

Train the LDM using the pre-trained VQVAE:

```bash
python scripts/train/train_ldm.py
```

**Key hyperparameters:**
- Batch size: 10-16
- Epochs: 200
- Validation interval: 25
- DDPM steps: 1000
- Learning rate: 1e-4

### Stage 3: Train 3MCG (Full Model)

Train the full 3MCG model with ControlNet:

```bash
python scripts/train/train_3mcg.py
```

**Key hyperparameters:**
- Batch size: 16
- AutoEncoder epochs: 200
- DDPM epochs: 400
- ControlNet epochs: 400
- Validation interval: 25

### Training Tips

1. **GPU Memory**: Reduce batch size if you encounter OOM errors
2. **Checkpointing**: Models are saved every validation interval
3. **Monitoring**: Training visualizations are saved in the model directory
4. **Resume Training**: Load checkpoint and continue from saved epoch

## Testing and Evaluation

### Generate Samples

Use the trained models to generate synthetic MRI images:

```bash
# Update the checkpoint paths in the test script
python scripts/test/fid_test.py
```

### Evaluation Metrics

The framework supports multiple evaluation metrics:

- **FID (Fréchet Inception Distance)**: Image quality and diversity
- **MS-SSIM (Multi-Scale Structural Similarity)**: Structural similarity
- **SSIM**: Structural similarity index
- **MMD (Maximum Mean Discrepancy)**: Distribution matching

Example usage:
```python
from generative.metrics import FIDMetric, MultiScaleSSIMMetric, SSIMMetric, MMDMetric

ms_ssim = MultiScaleSSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=2)
ssim = SSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=2)
mmd = MMDMetric()
```

## Results

![Sample Results](docs/sample_results.png)

Our method achieves state-of-the-art performance on 3D brain tumor MRI generation tasks. See the paper for detailed quantitative results.

## Citation

If you find this work useful, please consider citing our paper and giving us a 🌟:

```bibtex
@article{wang2025zeco,
  title={ZECO: ZeroFusion Guided 3D MRI Conditional Generation},
  author={Wang, Feiran and Duan, Bin and Tao, Jiachen and Sharma, Nikhil and Cai, Dawen and Yan, Yan},
  journal={arXiv preprint arXiv:2503.18246},
  year={2025}
}

@article{feiran2025zeco,
  title={ZECO: ZeroFusion Guided 3D MRI Conditional Generation},
  author={Feiran, Wang and Bin, Duan and Jiachen, Tao and Nikhil, Sharma and Dawen, Cai and Yan, Yan},
  journal={IEICE Proceedings Series},
  volume={93},
  number={O1-2-2},
  year={2025},
  publisher={The Institute of Electronics, Information and Communication Engineers}
}
```

## Acknowledgments

This work builds upon:
- [MONAI](https://monai.io/): Medical Open Network for AI
- [BraTS Challenge](https://www.med.upenn.edu/cbica/brats2020/): Brain Tumor Segmentation Challenge
- Latent Diffusion Models and ControlNet architectures

## License

This project is released under the MIT License. See [LICENSE](LICENSE) file for details.

