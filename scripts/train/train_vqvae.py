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
sys.path.append(os.path.join(os.path.dirname(__file__), "../..", "generative"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../..", "MONAI"))
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
import time
from generative.inferers import LatentDiffusionInferer, ControlNetDiffusionInferer, ControlNetLatentDiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, VQVAE, PatchDiscriminator,ControlNet
from generative.networks.schedulers import DDPMScheduler
from generative.metrics import FIDMetric, MMDMetric, MultiScaleSSIMMetric, SSIMMetric, MMDMetric
ms_ssim = MultiScaleSSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=2)
ssim = SSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=2)
mmd = MMDMetric()
print_config()
# -

# for reproducibility purposes set a seed
set_determinism(42)

# ### Setup a data directory and download dataset
# Specify a MONAI_DATA_DIRECTORY variable, where the data will be downloaded. If not specified a temporary directory will be used.

# Directory to read dataset
root_dir = "/data/wangfeiran/dataset/monai_full"
os.makedirs(root_dir, exist_ok=True)
# Directory to save the models
model_dir = "/data/wangfeiran/result/monai_results/vqvae_t1"
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
batch_size = 2

#channel
channel = 1  # 0 = Flair
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
        transforms.Lambdad(keys="label", func=lambda x: x[0, :, :, :]),
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


print(f'Image shape train_ds {train_ds[0]["image"].shape}')
print(f'label shape train_ds {train_ds[0]["label"].shape}')

print(f'Image shape val_ds {val_ds[0]["image"].shape}')
print(f'label shape val_ds {val_ds[0]["label"].shape}')
# -

# ### Visualise examples from the training set

# +
# Plot axial, coronal and sagittal slices of a training sample
check_data = first(train_loader)
idx = 0

img = check_data["image"][idx, 0]

print("img_size", img.shape)
plt.subplots(1, 4, figsize=(10, 6))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(train_ds[i]["image"][0, :, :, img.shape[2] // 2].detach().cpu(), vmin=0, vmax=1, cmap="gray")
    plt.axis("off")
plt.tight_layout()
image_path = os.path.join(model_dir, f"Origin_image.png")
plt.savefig(image_path)
plt.close()


plt.subplots(1, 4, figsize=(10, 6))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(train_ds[i]["label"][0, :, :, img.shape[2] // 2].detach().cpu(), vmin=0, vmax=1, cmap="gray")
    plt.axis("off")
plt.tight_layout()
image_path = os.path.join(model_dir, f"Origin_label.png")
plt.savefig(image_path)
plt.close()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# autoencoder = VQVAE(
#     spatial_dims=3,
#     in_channels=1,
#     out_channels=1,
#     num_channels=(256, 512, 512),
#     num_res_channels=256,
#     num_res_layers=3,
#     downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1), (1, 4, 1, 1)),  # Keep third dim stride as 1
#     upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (1, 4, 1, 1, 0)),  # Ensure symmetry
#     num_embeddings=256,
#     embedding_dim=64,
# )

autoencoder = VQVAE(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_channels=(256, 512, 512),  # Defining layers for depth in the network
    num_res_channels=256,
    num_res_layers=3,
    
    # Downsampling: Let's aim for a factor that progressively reduces the spatial dimensions symmetrically
    # (stride, kernel_size, dilation, padding)
    downsample_parameters=(
        (2, 4, 1, 1),  # Reduces the size by 2 in each dimension (96x96x64 -> 48x48x32)
        (2, 4, 1, 1),  # Further reduces to 24x24x16
        (2, 4, 1, 1)   # Further reduces to 12x12x8
    ),
    
    # Upsampling: Reversing the downsampling procedure (stride, kernel_size, dilation, padding, output_padding)
    upsample_parameters=(
        (2, 4, 1, 1, 0),  # Restores from 12x12x8 to 24x24x16
        (2, 4, 1, 1, 0),  # Restores from 24x24x16 to 48x48x32
        (2, 4, 1, 1, 0)   # Restores from 48x48x32 to 96x96x64 (original size)
    ),

    num_embeddings=256,  # Number of embeddings
    embedding_dim=64,    # Embedding dimension
)

autoencoder.to(device)

optimizer = torch.optim.Adam(params=autoencoder.parameters(), lr=1e-4)
l1_loss = L1Loss()


n_epochs = 20
val_interval = 2
epoch_recon_loss_list = []
epoch_quant_loss_list = []
val_recon_epoch_loss_list = []
intermediary_images = []
n_example_images = 4

total_start = time.time()
for epoch in range(n_epochs):
    autoencoder.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch["image"].to(device)
        optimizer.zero_grad(set_to_none=True)
        
        # model outputs reconstruction and the quantization error
        reconstruction, quantization_loss = autoencoder(images=images)
        # reconstruction = F.interpolate(reconstruction, size=(96, 96, 64))
        print("images", images.shape)
        print("reconstruction", reconstruction.shape)
        recons_loss = l1_loss(reconstruction.float(), images.float())

        loss = recons_loss + quantization_loss

        loss.backward()
        optimizer.step()

        epoch_loss += recons_loss.item()

        progress_bar.set_postfix(
            {"recons_loss": epoch_loss / (step + 1), "quantization_loss": quantization_loss.item() / (step + 1)}
        )
    epoch_recon_loss_list.append(epoch_loss / (step + 1))
    epoch_quant_loss_list.append(quantization_loss.item() / (step + 1))

    if (epoch + 1) % val_interval == 0:
        autoencoder.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(val_loader, start=1):
                images = batch["image"].to(device)

                reconstruction, quantization_loss = autoencoder(images=images)
                # reconstruction = F.interpolate(reconstruction, size=(96, 96, 64))
                # get the first sample from the first validation batch for
                # visualizing how the training evolves
                print("images 1", images.shape)
                print("reconstruction 1", reconstruction.shape)
                if val_step == 1:
                    intermediary_images.append(reconstruction[:n_example_images, 0])
                print("images 2", images.shape)
                print("reconstruction 2", reconstruction.shape)
                recons_loss = l1_loss(reconstruction.float(), images.float())

                val_loss += recons_loss.item()

        val_loss /= val_step
        val_recon_epoch_loss_list.append(val_loss)
                # Select the slice to visualize (e.g., the middle slice along the depth dimension)
        slice_index = img.shape[2] // 2

        # Get the original image slice
        original_img = images[0, 0].detach().cpu().numpy()
        original_slice = original_img[..., slice_index]

        # Get the reconstructed image slice
        reconstructed_img = reconstruction[0, 0].detach().cpu().numpy()
        reconstructed_slice = reconstructed_img[..., slice_index]

        # Create a figure with two subplots: one for the original and one for the reconstruction
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the original image slice
        ax[0].imshow(original_slice, cmap="gray")
        ax[0].axis("off")
        ax[0].title.set_text("Original Image")

        # Plot the reconstructed image slice
        ax[1].imshow(reconstructed_slice, cmap="gray")
        ax[1].axis("off")
        ax[1].title.set_text("Reconstruction")

        # Construct the full path for the image file
        image_path = os.path.join(autoencoder_dir, f"comparison_{epoch+1}.png")

        # Save the figure
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close()


        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,  # Save the current epoch
            'autoencoder_state_dict': autoencoder.state_dict(),  # Save autoencoder weights
            'epoch_recon_loss_list': epoch_recon_loss_list,  # Save the reconstruction loss history
            'val_recon_epoch_loss_list': val_recon_epoch_loss_list,  # Save the validation loss history
            'intermediary_images': intermediary_images  # Save intermediary images for visual inspection
        }
        checkpoint_path = os.path.join(autoencoder_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(checkpoint, checkpoint_path)
        