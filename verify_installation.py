#!/usr/bin/env python
"""
ZECO Installation Verification Script
Run this script to verify your environment is properly set up.
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'generative'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MONAI'))

def check_python_version():
    """Check Python version"""
    print("=" * 70)
    print("1. Checking Python Version")
    print("=" * 70)
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major >= 3 and version.minor >= 8:
        print("✓ Python version is compatible (>= 3.8)")
        return True
    else:
        print("✗ Python version too old. Please use Python 3.8 or higher")
        return False

def check_core_packages():
    """Check core packages"""
    print("\n" + "=" * 70)
    print("2. Checking Core Packages")
    print("=" * 70)

    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'tqdm': 'TQDM',
        'sklearn': 'Scikit-learn',
        'PIL': 'Pillow',
        'scipy': 'SciPy'
    }

    all_ok = True
    for pkg, name in packages.items():
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {name:15s} : {version}")
        except ImportError:
            print(f"✗ {name:15s} : NOT INSTALLED")
            all_ok = False

    return all_ok

def check_pytorch_cuda():
    """Check PyTorch CUDA availability"""
    print("\n" + "=" * 70)
    print("3. Checking PyTorch CUDA Support")
    print("=" * 70)

    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")

        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("⚠ CUDA not available. Training will be slow on CPU.")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def check_monai():
    """Check MONAI installation"""
    print("\n" + "=" * 70)
    print("4. Checking MONAI Framework")
    print("=" * 70)

    try:
        import monai
        print(f"✓ MONAI version: {monai.__version__}")

        # Check if we can import from local MONAI
        from monai.networks.blocks import Convolution
        print("✓ MONAI imports working correctly")
        return True
    except ImportError as e:
        print(f"✗ MONAI not installed or import failed: {e}")
        print("\nTo install MONAI, run:")
        print("  pip install monai")
        return False

def check_generative_models():
    """Check generative models can be imported"""
    print("\n" + "=" * 70)
    print("5. Checking Generative Models")
    print("=" * 70)

    models = [
        ('VQVAE', 'generative.networks.nets'),
        ('DiffusionModelUNet', 'generative.networks.nets'),
        ('ControlNet', 'generative.networks.nets'),
        ('DDPMScheduler', 'generative.networks.schedulers'),
    ]

    all_ok = True
    for model_name, module_path in models:
        try:
            module = __import__(module_path, fromlist=[model_name])
            model_class = getattr(module, model_name)
            print(f"✓ {model_name:25s} from {module_path}")
        except Exception as e:
            print(f"✗ {model_name:25s} : Failed to import ({e})")
            all_ok = False

    return all_ok

def check_training_scripts():
    """Check training scripts exist"""
    print("\n" + "=" * 70)
    print("6. Checking Training Scripts")
    print("=" * 70)

    scripts = [
        'scripts/train/train_vqvae.py',
        'scripts/train/train_ldm.py',
        'scripts/train/train_3mcg.py',
    ]

    all_ok = True
    for script in scripts:
        if os.path.exists(script):
            print(f"✓ {script}")
        else:
            print(f"✗ {script} NOT FOUND")
            all_ok = False

    return all_ok

def main():
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "ZECO Installation Verification" + " " * 23 + "║")
    print("╚" + "=" * 68 + "╝\n")

    results = []
    results.append(("Python Version", check_python_version()))
    results.append(("Core Packages", check_core_packages()))
    results.append(("PyTorch CUDA", check_pytorch_cuda()))
    results.append(("MONAI Framework", check_monai()))
    results.append(("Generative Models", check_generative_models()))
    results.append(("Training Scripts", check_training_scripts()))

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:25s} : {status}")

    all_passed = all(r[1] for r in results)

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 All checks passed! Your environment is ready for training.")
        print("\nNext steps:")
        print("  1. Download BraTS 2020 dataset")
        print("  2. Update data paths in training scripts")
        print("  3. Run: python scripts/train/train_vqvae.py")
    else:
        print("⚠ Some checks failed. Please install missing packages:")
        print("\nTo install all dependencies:")
        print("  pip install -r requirements.txt")
        print("  pip install monai")
    print("=" * 70 + "\n")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
