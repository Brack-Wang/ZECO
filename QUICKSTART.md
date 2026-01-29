# Quick Start Guide

This guide will help you get started with ZECO quickly.

## Installation (5 minutes)

```bash
# Clone the repository
git clone https://github.com/yourusername/zeco.git
cd zeco

# Create virtual environment
conda create -n zeco python=3.9
conda activate zeco

# Install PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Verify installation
python verify_installation.py
```

## Data Preparation (30 minutes)

1. Download BraTS 2020 dataset from [official website](https://www.med.upenn.edu/cbica/brats2020/data.html)

2. Extract to your preferred location:
```bash
/path/to/dataset/brat20/MICCAI_BraTS2020_TrainingData/
```

3. Update data path in training scripts:
```bash
# Edit scripts/train/*.py
# Change: train_data_dir = "/path/to/your/dataset/brat20/MICCAI_BraTS2020_TrainingData"
```

## Training Pipeline

### Option 1: Train from Scratch (Recommended for Research)

**Step 1: Train VQVAE (1-2 days)**
```bash
python scripts/train/train_vqvae.py
```

**Step 2: Train LDM (3-5 days)**
```bash
# After VQVAE training completes
python scripts/train/train_ldm.py
```

**Step 3: Train 3MCG (5-7 days)**
```bash
# After LDM training completes
python scripts/train/train_3mcg.py
```

### Training Notes

- Training will create checkpoints automatically in the model output directories
- Monitor GPU memory usage and adjust batch size if needed
- Each stage builds on the previous one, so train in order

## Key Configuration

### VQVAE Training
- **Batch size**: 2 (adjust based on GPU memory)
- **Epochs**: 20
- **Input size**: 96×96×64
- **GPU Memory**: ~10GB

### LDM Training
- **Batch size**: 10-16 (adjust based on GPU memory)
- **Epochs**: 200
- **Validation interval**: 25 epochs
- **GPU Memory**: ~12-16GB

### 3MCG Training
- **Batch size**: 16 (adjust based on GPU memory)
- **Total epochs**: 400
- **GPU Memory**: ~16GB

## Troubleshooting

### Out of Memory (OOM)
```python
# Reduce batch_size in the training script
batch_size = 2  # or even 1
```

### CUDA Error
```bash
# Check PyTorch CUDA compatibility
python -c "import torch; print(torch.cuda.is_available())"
```

### Import Error
```bash
# Ensure paths are correct
cd /data/wangfeiran/code/zeco
python -c "import sys; sys.path.append('generative'); from generative.networks.nets import VQVAE"
```

## Expected Results

After training, you should see:
- Reconstructed images in model output directories
- Training curves (loss vs. epochs)
- Checkpoints saved at validation intervals
- Generated samples for evaluation

## Next Steps

1. Monitor training progress in output directories
2. Evaluate using FID, MS-SSIM metrics
3. Generate synthetic samples
4. Fine-tune hyperparameters for your specific use case

## Support

- Issues: [GitHub Issues](https://github.com/yourusername/zeco/issues)
- Documentation: See [README.md](README.md)
