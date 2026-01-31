"""
ZECO Unified Training Script
Entry point for training VQVAE, LDM, and 3MCG models

Usage:
    python train.py --model vqvae --channel 0
    python train.py --model ldm
    python train.py --model 3mcg

This script serves as a unified interface that calls the appropriate training script
from the scripts/ directory with user-specified parameters.
"""

import os
import sys
import argparse
import subprocess


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="ZECO Unified Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train VQVAE on FLAIR modality
  python train.py --model vqvae --channel 0

  # Train LDM
  python train.py --model ldm

  # Train 3MCG (full model)
  python train.py --model 3mcg

Note: This script calls the corresponding training script in scripts/ directory.
For advanced configuration, you can directly edit and run the scripts:
  - scripts/train_vqvae.py
  - scripts/train_ldm.py
  - scripts/train_3mcg.py
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["vqvae", "ldm", "3mcg"],
        help="Model to train: vqvae, ldm, or 3mcg"
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="MRI channel for VQVAE: 0=FLAIR, 1=T1, 2=T1CE, 3=T2 (default: 0)"
    )

    return parser.parse_args()


def main():
    """Main function to run training"""

    args = parse_args()

    # Map model names to script paths
    script_map = {
        "vqvae": "scripts/train_vqvae.py",
        "ldm": "scripts/train_ldm.py",
        "3mcg": "scripts/train_3mcg.py"
    }

    script_path = script_map[args.model]

    # Check if script exists
    if not os.path.exists(script_path):
        print(f"Error: Training script not found: {script_path}")
        sys.exit(1)

    print("="*70)
    print(f"ZECO Training: {args.model.upper()}")
    print("="*70)
    print(f"Model: {args.model}")
    if args.model == "vqvae":
        channel_names = ["FLAIR", "T1", "T1CE", "T2"]
        print(f"Channel: {args.channel} ({channel_names[args.channel]})")
    print(f"Script: {script_path}")
    print("="*70)
    print()

    # Display information message
    print("ℹ️  Note:")
    print(f"   This script will run: python {script_path}")
    print()
    print("   To customize training parameters (batch size, epochs, etc.),")
    print(f"   please edit the configuration in: {script_path}")
    print()

    if args.model == "vqvae":
        print(f"   For VQVAE, the channel is set to {args.channel} ({channel_names[args.channel]})")
        print(f"   To change the channel, edit 'channel = {args.channel}' in {script_path}")
        print()

    # Ask for confirmation
    response = input("Press Enter to continue or Ctrl+C to cancel...")
    print()

    print("="*70)
    print("Starting training...")
    print("="*70)
    print()

    # Run the training script
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )

        print()
        print("="*70)
        print("✅ Training completed successfully!")
        print("="*70)

    except subprocess.CalledProcessError as e:
        print()
        print("="*70)
        print(f"❌ Training failed with exit code {e.returncode}")
        print("="*70)
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print()
        print("="*70)
        print("⚠️  Training interrupted by user")
        print("="*70)
        sys.exit(1)


if __name__ == "__main__":
    main()
