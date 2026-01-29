
import os
import shutil
import tempfile
import sys
from tqdm import tqdm
import numpy as np
import torch
sys.path.append("/data/wangfeiran/code/all_you_control/generative")
sys.path.append('/data/wangfeiran/code/all_you_control/MONAI')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss
from tqdm import tqdm

from generative.inferers import LatentDiffusionInferer, ControlNetDiffusionInferer, ControlNetLatentDiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator,ControlNet
from generative.networks.schedulers import DDPMScheduler
from generative.metrics import FIDMetric, MMDMetric, MultiScaleSSIMMetric, SSIMMetric
import torch


"""
2D FID
"""
# def subtract_mean(x: torch.Tensor) -> torch.Tensor:
#     mean = [0.406, 0.456, 0.485]
#     x[:, 0, :, :] -= mean[0]
#     x[:, 1, :, :] -= mean[1]
#     x[:, 2, :, :] -= mean[2]
#     return x

# def spatial_average(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
#     return x.mean([2, 3], keepdim=keepdim)

# def get_features(image):
#     # If input has just 1 channel, repeat channel to have 3 channels
#     if image.shape[1] == 1:
#         image = image.repeat(1, 3, 1, 1)

#     # Change order from 'RGB' to 'BGR'
#     image = image[:, [2, 1, 0], ...]

#     # Subtract mean used during training
#     image = subtract_mean(image)

#     # Get model outputs
#     with torch.no_grad():
#         feature_image = radnet.forward(image)
#         # flattens the image spatially
#         feature_image = spatial_average(feature_image, keepdim=False)

#     return feature_image

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# radnet = torch.hub.load("Warvito/radimagenet-models", model="radimagenet_resnet50", verbose=True)
# radnet.to(device)
# radnet.eval()

# # Here, we will load the real and generate synthetic images from noise and compute the FID of these two groups of images.

# synth_features = []
# real_features = []

# n_synthetic_images = 3
# noise_1 = torch.randn((n_synthetic_images, 1, 64, 64)).to(device)
# noise_2 = torch.randn((n_synthetic_images, 1, 64, 64)).to(device)

# real_eval_feats = get_features(noise_1)
# real_features.append(real_eval_feats)

# # Get the features for the synthetic data
# synth_eval_feats = get_features(noise_2)
# synth_features.append(synth_eval_feats)
# synth_features = torch.vstack(synth_features)
# real_features = torch.vstack(real_features)
# fid = FIDMetric()
# fid_res = fid(synth_features, real_features)
# print(f"FID Score: {fid_res.item():.4f}")



"""
3D of resnet
"""

# import torch
# from monai.networks.nets import resnet
# from torchmetrics.image.fid import FIDMetric

# def subtract_mean(x: torch.Tensor) -> torch.Tensor:
#     mean = [0.406]
#     x[:, 0, :] -= mean[0]
#     return x

# def get_features(image):
#     # Subtract mean used during training
#     image = subtract_mean(image)

#     # Get model outputs
#     with torch.no_grad():
#         feature_image = radnet(image)
#         # No need to apply spatial average since the shape is already [batch_size, 2048]
#         return feature_image

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load a pretrained 3D ResNet model from MONAI with MedicalNet weights
# radnet = resnet.resnet50(
#     spatial_dims=3,
#     n_input_channels=1,
#     num_classes=1,
#     pretrained=True,
#     progress=True,
#     feed_forward=False,
#     shortcut_type="B",
#     bias_downsample=False
# )
# radnet.to(device)
# radnet.eval()

# # Accumulate features over multiple batches
# synth_features = []
# real_features = []

# n_synthetic_images = 10  # Increase the number of samples if possible
# batch_size = 3

# for _ in range(n_synthetic_images // batch_size):
#     noise_1 = torch.randn((batch_size, 1, 96, 96, 64)).to(device)
#     noise_2 = torch.randn((batch_size, 1, 96, 96, 64)).to(device)

#     real_eval_feats = get_features(noise_1)
#     real_features.append(real_eval_feats)

#     synth_eval_feats = get_features(noise_2)
#     synth_features.append(synth_eval_feats)

# # Convert the list of features to a single tensor
# synth_features = torch.cat(synth_features, dim=0)
# real_features = torch.cat(real_features, dim=0)

# fid = FIDMetric()
# fid_res = fid(synth_features, real_features)
# print(f"FID Score: {fid_res.item():.4f}")


# import torch
# from torch.utils.data import DataLoader, TensorDataset
# from monai.networks.nets import resnet
# from scipy import linalg
# import numpy as np
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # Assuming the FIDMetric class and helper functions are already defined as you provided

# def extract_features(model, data, device):
#     model.eval()

#     with torch.no_grad():
#         data = data.to(device)
#         features = model(data)
#         features = features.view(features.size(0), -1)  # Flatten the features

#     return features

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load a pretrained 3D ResNet model from MONAI with MedicalNet weights
# model = resnet.resnet50(
#     spatial_dims=3,
#     n_input_channels=1,
#     num_classes=1,
#     pretrained=True,
#     progress=True,
#     feed_forward=False,
#     shortcut_type="B",
#     bias_downsample=False
# )
# model.to(device)

# # Generate synthetic noise data
# batch_size = 3
# noise_1 = torch.randn((batch_size, 1, 96, 96, 64)).to(device)
# noise_2 = torch.randn((batch_size, 1, 96, 96, 64)).to(device)

# # Extract features from both noise tensors
# real_features = extract_features(model, noise_1, device)
# synth_features = extract_features(model, noise_2, device)

# # Calculate the FID
# fid_metric = FIDMetric()
# fid_score = fid_metric(synth_features, real_features)
# print(f"FID Score: {fid_score.item():.4f}")


import torch
from torch.utils.data import DataLoader, TensorDataset
from monai.networks.nets import resnet
from scipy import linalg
import numpy as np

# Assuming the FIDMetric class and helper functions are already defined as you provided

def extract_features(model, data, device):
    model.eval()

    with torch.no_grad():
        data = data.to(device)
        features = model(data)
        features = features.view(features.size(0), -1)  # Flatten the features

    return features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pretrained 3D ResNeSt200 model from MONAI
model = resnet.resnet200(
    spatial_dims=3,
    n_input_channels=1,  # Replace 'in_channels' with 'n_input_channels'
    num_classes=1,
    pretrained=True,  # Use pretrained weights if available
    progress=True,
    feed_forward=False,
    shortcut_type="B",
    bias_downsample=False
)
model.to(device)

# Generate synthetic noise data
batch_size = 3
noise_1 = torch.randn((batch_size, 1, 96, 96, 64)).to(device)
noise_2 = torch.randn((batch_size, 1, 96, 96, 64)).to(device)

# Extract features from both noise tensors
real_features = extract_features(model, noise_1, device)
synth_features = extract_features(model, noise_2, device)

# Calculate the FID
fid_metric = FIDMetric()
fid_score = fid_metric(synth_features, real_features)
print(f"FID Score: {fid_score.item():.4f}")