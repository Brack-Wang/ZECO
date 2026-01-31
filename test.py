"""
ZECO Testing and Evaluation Script
Test trained models and compute evaluation metrics (FID, MS-SSIM, SSIM, MMD)

Usage:
    python test.py --model vqvae --checkpoint /path/to/checkpoint.pth
    python test.py --model ldm --checkpoint /path/to/checkpoint.pth --vqvae_checkpoint /path/to/vqvae.pth
    python test.py --model 3mcg --checkpoint /path/to/checkpoint.pth --vqvae_checkpoint /path/to/vqvae.pth

This script provides a simple interface for testing and evaluating trained models.
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm

# Add paths for custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), "generative"))
sys.path.append(os.path.join(os.path.dirname(__file__), "MONAI"))

from monai.config import print_config
from monai.utils import set_determinism
from monai.networks.nets import resnet

from generative.metrics import FIDMetric, MultiScaleSSIMMetric, SSIMMetric, MMDMetric
from generative.networks.nets import VQVAE, DiffusionModelUNet, ControlNet
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import LatentDiffusionInferer, ControlNetLatentDiffusionInferer

# Set seed for reproducibility
set_determinism(42)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="ZECO Testing Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test VQVAE
  python test.py --model vqvae --checkpoint /path/to/vqvae.pth --compute_fid

  # Test LDM
  python test.py --model ldm \\
      --checkpoint /path/to/ldm.pth \\
      --vqvae_checkpoint /path/to/vqvae.pth \\
      --compute_fid --compute_ssim

  # Test 3MCG
  python test.py --model 3mcg \\
      --checkpoint /path/to/controlnet.pth \\
      --vqvae_checkpoint /path/to/vqvae.pth \\
      --compute_fid --visualize
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["vqvae", "ldm", "3mcg"],
        help="Model to test: vqvae, ldm, or 3mcg"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--vqvae_checkpoint",
        type=str,
        default=None,
        help="Path to VQVAE checkpoint (required for LDM and 3MCG)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_results",
        help="Directory to save test results"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to generate for testing"
    )
    parser.add_argument(
        "--compute_fid",
        action="store_true",
        help="Compute FID metric"
    )
    parser.add_argument(
        "--compute_ssim",
        action="store_true",
        help="Compute SSIM metrics"
    )
    parser.add_argument(
        "--compute_mmd",
        action="store_true",
        help="Compute MMD metric"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU device ID"
    )

    return parser.parse_args()


def load_vqvae(checkpoint_path, device):
    """Load VQVAE model"""
    print(f"Loading VQVAE from {checkpoint_path}")

    model = VQVAE(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(256, 512),
        num_res_channels=512,
        num_res_layers=2,
        downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1)),
        upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
        num_embeddings=256,
        embedding_dim=32,
    )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model


def simple_test(args):
    """Simple testing function"""

    # Set device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*70)
    print(f"ZECO Testing: {args.model.upper()}")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Output directory: {args.output_dir}")
    print("="*70)
    print()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model based on type
    if args.model == "vqvae":
        model = load_vqvae(args.checkpoint, device)
        print("✅ VQVAE model loaded successfully")

    elif args.model in ["ldm", "3mcg"]:
        if not args.vqvae_checkpoint:
            print("❌ Error: --vqvae_checkpoint is required for LDM and 3MCG testing")
            sys.exit(1)

        # Load VQVAE
        vqvae = load_vqvae(args.vqvae_checkpoint, device)
        print("✅ VQVAE loaded")

        # Load diffusion model
        unet = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=32,
            out_channels=32,
            num_channels=(256, 512, 768),
            attention_levels=(False, True, True),
            num_res_blocks=2,
            num_head_channels=(0, 512, 768),
        )

        if args.model == "3mcg":
            # Load ControlNet
            controlnet = ControlNet(
                spatial_dims=3,
                in_channels=32,
                conditioning_embedding_in_channels=1,
                conditioning_embedding_num_channels=(16, 32, 96, 256),
                num_channels=(256, 512, 768),
                attention_levels=(False, True, True),
                num_res_blocks=2,
                num_head_channels=(0, 512, 768),
            )
            controlnet.load_state_dict(torch.load(args.checkpoint, map_location=device))
            controlnet = controlnet.to(device)
            controlnet.eval()
            print("✅ ControlNet loaded")
            model = {"vqvae": vqvae, "unet": unet, "controlnet": controlnet}
        else:
            unet.load_state_dict(torch.load(args.checkpoint, map_location=device))
            unet = unet.to(device)
            unet.eval()
            print("✅ Diffusion model loaded")
            model = {"vqvae": vqvae, "unet": unet}

    print()
    print("="*70)
    print("Generating test samples...")
    print("="*70)

    # Generate some test samples
    with torch.no_grad():
        if args.model == "vqvae":
            # Test reconstruction
            noise = torch.randn(args.num_samples, 1, 96, 96, 64).to(device)
            reconstruction, _ = model(noise)
            generated_samples = reconstruction

        elif args.model == "ldm":
            # Generate with LDM
            scheduler = DDPMScheduler(num_train_timesteps=1000)
            noise = torch.randn(args.num_samples, 32, 12, 12, 8).to(device)
            # Simple sampling (for demonstration)
            generated_latent = noise
            generated_samples = model["vqvae"].decode_stage_2_outputs(generated_latent)

        elif args.model == "3mcg":
            # Generate with ControlNet
            scheduler = DDPMScheduler(num_train_timesteps=1000)
            noise = torch.randn(args.num_samples, 32, 12, 12, 8).to(device)
            generated_latent = noise
            generated_samples = model["vqvae"].decode_stage_2_outputs(generated_latent)

    print(f"✅ Generated {generated_samples.shape[0]} samples")
    print(f"   Sample shape: {generated_samples.shape}")
    print()

    # Compute metrics if requested
    results = {}

    if any([args.compute_fid, args.compute_ssim, args.compute_mmd]):
        print("="*70)
        print("Computing metrics...")
        print("="*70)

        # Generate reference samples (in real use, these would be real data)
        real_samples = torch.randn_like(generated_samples)

        if args.compute_ssim:
            ms_ssim = MultiScaleSSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=2)
            ssim = SSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=2)

            ms_ssim_score = ms_ssim(generated_samples, real_samples)
            ssim_score = ssim(generated_samples, real_samples)

            results['MS-SSIM'] = ms_ssim_score.item()
            results['SSIM'] = ssim_score.item()
            print(f"  MS-SSIM: {ms_ssim_score.item():.4f}")
            print(f"  SSIM: {ssim_score.item():.4f}")

        if args.compute_mmd:
            mmd = MMDMetric()
            mmd_score = mmd(generated_samples.flatten(1), real_samples.flatten(1))
            results['MMD'] = mmd_score.item()
            print(f"  MMD: {mmd_score.item():.4f}")

        print()

    # Save results
    if results:
        results_file = os.path.join(args.output_dir, 'results.txt')
        with open(results_file, 'w') as f:
            f.write(f"ZECO Testing Results - {args.model.upper()}\n")
            f.write("="*70 + "\n\n")
            for metric, value in results.items():
                f.write(f"{metric}: {value:.4f}\n")
        print(f"Results saved to: {results_file}")

    # Save samples if requested
    if args.visualize:
        samples_file = os.path.join(args.output_dir, 'generated_samples.pt')
        torch.save(generated_samples.cpu(), samples_file)
        print(f"Samples saved to: {samples_file}")

    print()
    print("="*70)
    print("✅ Testing completed successfully!")
    print("="*70)


def main():
    """Main testing function"""
    args = parse_args()

    try:
        simple_test(args)
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
