"""
Description: Train autoencoderkl + ddpm + Controlnet with 

"""
# +
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
from torch.utils.tensorboard import SummaryWriter
from generative.inferers import LatentDiffusionInferer, ControlNetDiffusionInferer, ControlNetLatentDiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator,ControlNet
from generative.networks.schedulers import DDPMScheduler

print_config()
# -

# for reproducibility purposes set a seed
set_determinism(42)

# ### Setup a data directory and download dataset
# Specify a MONAI_DATA_DIRECTORY variable, where the data will be downloaded. If not specified a temporary directory will be used.

root_dir = "/data/wangfeiran/dataset/monai"
print(root_dir)
# Directory to read dataset
root_dir = "/data/wangfeiran/dataset/monai"
os.makedirs(root_dir, exist_ok=True)
# Directory to save the models
model_dir = "/data/wangfeiran/result/monai_results/3mg_text"
os.makedirs(model_dir, exist_ok=True)
# Define paths and other necessary variables
ddpm_dir = model_dir + "/ddpm"
ddpm_checkpoint_dir = ddpm_dir + "/checkpoints"
os.makedirs(ddpm_checkpoint_dir, exist_ok=True)
controlnet_dir = model_dir + "/controlnet"
controlnet_checkpoint_dir = controlnet_dir + "/checkpoints"
os.makedirs(controlnet_checkpoint_dir, exist_ok=True)
autoencoder_dir = model_dir + "/autoencoder"
autoencoder_checkpoint_dir = autoencoder_dir + "/checkpoints"
os.makedirs(autoencoder_checkpoint_dir, exist_ok=True)
# ### Prepare data loader for the training set
# Here we will download the Brats dataset using MONAI's `DecathlonDataset` class, and we prepare the data loader for the training set.

# +
# batch_size
batch_size = 4

#channel
channel = 0  # 0 = Flair
assert channel in [0, 1, 2, 3], "Choose a valid channel"

train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.Lambdad(keys="image", func=lambda x: x[channel, :, :, :]),
        transforms.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        transforms.EnsureTyped(keys=["image"]),
        transforms.Orientationd(keys=["image"], axcodes="RAS"),
        transforms.Spacingd(keys=["image"], pixdim=(2.4, 2.4, 2.2), mode=("bilinear")),
        transforms.CenterSpatialCropd(keys=["image"], roi_size=(96, 96, 64)),
        transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
        
        transforms.EnsureChannelFirstd(keys=["label"]),
        transforms.Lambdad(keys="label", func=lambda x: x[channel, :, :, :]),
        transforms.EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        transforms.Orientationd(keys=["label"], axcodes="RAS"),
        transforms.Spacingd(keys=["label"], pixdim=(2.4, 2.4, 2.2), mode=("bilinear")),
        transforms.CenterSpatialCropd(keys=["label"], roi_size=(96, 96, 64)),
    ]

)
train_ds = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    section="training",  # validation
    # cache_rate=0.1,  # you may need a few Gb of RAM... Set to 0 otherwise
    # num_workers=8,
    download=True,  # Set download to True if the dataset hasnt been downloaded yet
    seed=0,
    transform=train_transforms,
)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True)

val_ds = DecathlonDataset(
    root_dir=root_dir, task="Task01_BrainTumour", transform=train_transforms, section="validation", download=True
)

val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True)


print(f'Image shape {train_ds[0]["image"].shape}')
print(f'Image shape {train_ds[0]["label"].shape}')