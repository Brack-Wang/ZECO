#!/bin/bash

# ZECO Setup Script
echo "Setting up ZECO environment..."

# Create necessary directories
mkdir -p checkpoints
mkdir -p outputs
mkdir -p logs
mkdir -p data

echo "✓ Directories created"

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name --format=csv,noheader
else
    echo "⚠ No NVIDIA GPU detected (CPU mode only)"
fi

# Check PyTorch installation
if python -c "import torch" &> /dev/null; then
    echo "✓ PyTorch is installed"
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
else
    echo "✗ PyTorch is not installed. Please install it first."
    exit 1
fi

# Check MONAI installation
if python -c "import monai" &> /dev/null; then
    echo "✓ MONAI is installed"
    python -c "import monai; print(f'MONAI version: {monai.__version__}')"
else
    echo "⚠ MONAI is not installed. Installing..."
    pip install monai
fi

echo ""
echo "Setup complete! You can now:"
echo "1. Download BraTS 2020 dataset"
echo "2. Update data paths in scripts/train/*.py"
echo "3. Start training with: python scripts/train/train_vqvae.py"
