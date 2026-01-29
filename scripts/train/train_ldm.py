"""
Description: Train autoencoderkl + ddpm + Controlnet With mask condition on Flair modalities

"""
# +
import os
import shutil
import tempfile
import sys
from tqdm import tqdm
import numpy as np
import time
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

from generative.inferers import LatentDiffusionInferer, ControlNetDiffusionInferer, ControlNetLatentDiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator,ControlNet, VQVAE
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
model_dir = "/data/wangfeiran/result/monai_results/20_flair_2"
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
batch_size = 10

n_epochs = 200
val_interval = 25
ddpm_n_epochs = 200
ddpm_val_interval= 1


controlnet_n_epochs = 200

controlnet_val_interval = 25


# +
import os
import glob
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, ScaleIntensityd, ToTensord, Resized, EnsureChannelFirstd
)
from sklearn.model_selection import train_test_split

# Define the paths for training data
train_data_dir = "/data/wangfeiran/dataset/brat20/MICCAI_BraTS2020_TrainingData"

# Get list of paths for flair and seg files in training data
train_images = sorted(glob.glob(os.path.join(train_data_dir, "*/*_flair.nii")))
train_labels = sorted(glob.glob(os.path.join(train_data_dir, "*/*_seg.nii")))

# Create a dictionary for the data (both images and labels)
train_files = [{"image": img, "label": seg} for img, seg in zip(train_images, train_labels)]

# Split the data into train and validation sets (80% train, 20% val)
train_files, val_files = train_test_split(train_files, test_size=0.2, random_state=42)

# Print the number of training and validation samples
print(f"Number of training samples: {len(train_files)}")
print(f"Number of validation samples: {len(val_files)}")


# Define the transformations for training and validation datasets
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),  # Ensure channels are in the first position
    ScaleIntensityd(keys=["image"]),  # Optionally, normalize intensity
    Resized(keys=["image", "label"], spatial_size=(96, 96, 64)),  # Resize to (96, 96, 64)
    transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
    ToTensord(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"]),
    Resized(keys=["image", "label"], spatial_size=(96, 96, 64)),
    transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
    ToTensord(keys=["image", "label"]),
])

# Create MONAI datasets and data loaders
train_ds = Dataset(data=train_files, transform=train_transforms)
val_ds = Dataset(data=val_files, transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=False, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=False, drop_last=True)

# Example of iterating through the training data
print(f'Image shape {train_ds[0]["image"].shape}')
print(f'label shape {train_ds[0]["label"].shape}')


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


# plt.savefig("training_examples.png")
# -

# ## Autoencoder KL
#
# ### Define Autoencoder KL network
#
# In this section, we will define an autoencoder with KL-regularization for the LDM. The autoencoder's primary purpose is to transform input images into a latent representation that the diffusion model will subsequently learn. By doing so, we can decrease the computational resources required to train the diffusion component, making this approach suitable for learning high-resolution medical images.
#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")


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

discriminator = PatchDiscriminator(spatial_dims=3, in_channels=1, num_layers_d=3, num_channels=64)
discriminator.to(device)

perceptual_loss = PerceptualLoss(spatial_dims=3, network_type="alex")
perceptual_loss.to(device)

optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=5e-5)
optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=3e-4)
# -

# ### Defining Losses
#
# We will also specify the perceptual and adversarial losses, including the involved networks, and the optimizers to use during the training process.

# +
l1_loss = L1Loss()
adv_loss = PatchAdversarialLoss(criterion="least_squares")
loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
adv_weight = 0.01
perceptual_weight = 0.001
loss_perceptual.to(device)


scaler_g = torch.cuda.amp.GradScaler()
scaler_d = torch.cuda.amp.GradScaler()
# ### Train model

# +

SSIM_vae_list= []
MSSSIM_vae_list = []


epoch_recon_loss_list = []
epoch_gen_loss_list = []
epoch_disc_loss_list = []
val_recon_epoch_loss_list = []
intermediary_images = []
n_example_images = 4

import os
import torch
start_epoch = 0
# Specify the checkpoint path
checkpoint_path = "/data/wangfeiran/result/monai_results/20_flair_2/autoencoder/checkpoint_epoch_200.pth"

print("Train autoencoder")

# Check if the checkpoint exists
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Load model and optimizer states
    autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    
    # Load additional training details
    start_epoch = checkpoint['epoch']
    epoch_recon_loss_list = checkpoint['epoch_recon_loss_list']
    epoch_gen_loss_list = checkpoint['epoch_gen_loss_list']
    epoch_disc_loss_list = checkpoint['epoch_disc_loss_list']
    val_recon_epoch_loss_list = checkpoint['val_recon_epoch_loss_list']
    intermediary_images = checkpoint.get('intermediary_images', [])  # Load intermediary images if available
    
    print(f"Resuming training from epoch {start_epoch}")
else:
    # If the specified checkpoint doesn't exist, handle it
    raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found.")
scratch_flag = start_epoch


total_start = time.time()

for epoch in range(start_epoch, n_epochs):
    autoencoder.train()
    discriminator.train()
    epoch_loss = 0
    gen_epoch_loss = 0
    disc_epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    
    for step, batch in progress_bar:
        images = batch["image"].to(device)
        masks = batch["label"].to(device)
        # images_cond = torch.cat([images, masks], dim=1).to(device)  
        optimizer_g.zero_grad(set_to_none=True)

        # Generator part
        reconstruction, quantization_loss = autoencoder(images=images)
        logits_fake = discriminator(reconstruction.contiguous().float())[-1]
        # reconstruction = F.interpolate(reconstruction, size=(96, 96, 64))
        recons_loss = l1_loss(reconstruction.float(), images.float())
        p_loss = perceptual_loss(reconstruction.float(), images.float())
        generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
        loss_g = recons_loss + quantization_loss + perceptual_weight * p_loss + adv_weight * generator_loss

        loss_g.backward()
        optimizer_g.step()

        # Discriminator part
        optimizer_d.zero_grad(set_to_none=True)

        logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
        loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = discriminator(images.contiguous().detach())[-1]
        loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

        loss_d = adv_weight * discriminator_loss

        loss_d.backward()
        optimizer_d.step()

        epoch_loss += recons_loss.item()
        gen_epoch_loss += generator_loss.item()
        disc_epoch_loss += discriminator_loss.item()

        progress_bar.set_postfix(
            {
                "recons_loss": epoch_loss / (step + 1),
                "gen_loss": gen_epoch_loss / (step + 1),
                "disc_loss": disc_epoch_loss / (step + 1),
            }
        )
    
    epoch_recon_loss_list.append(epoch_loss / (step + 1))
    epoch_gen_loss_list.append(gen_epoch_loss / (step + 1))
    epoch_disc_loss_list.append(disc_epoch_loss / (step + 1))

    if (epoch + 1) % val_interval == 0:
        autoencoder.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(val_loader, start=1):
                images = batch["image"].to(device)
                masks = batch["label"].to(device)
                # images_cond = torch.cat([images, masks], dim=1).to(device)  

                reconstruction, quantization_loss = autoencoder(images=images)
                # Get the first sample from the first validation batch for visualization purposes
                if val_step == 1:
                    intermediary_images.append(reconstruction[:n_example_images, 0])

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
            'discriminator_state_dict': discriminator.state_dict(),  # Save discriminator weights
            'optimizer_g_state_dict': optimizer_g.state_dict(),  # Save generator optimizer state
            'optimizer_d_state_dict': optimizer_d.state_dict(),  # Save discriminator optimizer state
            'epoch_recon_loss_list': epoch_recon_loss_list,  # Save the reconstruction loss history
            'epoch_gen_loss_list': epoch_gen_loss_list,  # Save the generator loss history
            'epoch_disc_loss_list': epoch_disc_loss_list,  # Save the discriminator loss history
            'val_recon_epoch_loss_list': val_recon_epoch_loss_list,  # Save the validation loss history
            'intermediary_images': intermediary_images  # Save intermediary images for visual inspection
        }
        checkpoint_path = os.path.join(autoencoder_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(checkpoint, checkpoint_path)
        

total_time = time.time() - total_start
print(f"Training completed, total time: {total_time}.")

if scratch_flag == 0: 
    # Plot and save the learning curves
    plt.style.use("seaborn-v0_8")
    plt.title("Learning Curves", fontsize=20)
    plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_recon_loss_list, color="C0", linewidth=2.0, label="Train")
    plt.plot(
        np.linspace(val_interval, n_epochs, int(n_epochs / val_interval)),
        val_recon_epoch_loss_list,
        color="C1",
        linewidth=2.0,
        label="Validation",
    )
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(prop={"size": 14})
    plt.savefig(os.path.join(model_dir, "learning_curves.png"))  # Save the plot
    plt.close()

    # Plot and save the adversarial training curves
    plt.title("Adversarial Training Curves", fontsize=20)
    plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_gen_loss_list, color="C0", linewidth=2.0, label="Generator")
    plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_disc_loss_list, color="C1", linewidth=2.0, label="Discriminator")
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(prop={"size": 14})
    plt.savefig(os.path.join(model_dir, "adversarial_training_curves.png"))  # Save the plot
    plt.close()



del discriminator
del loss_perceptual
torch.cuda.empty_cache()
# -

# ### Define diffusion model and scheduler
#
# In this section, we will define the diffusion model that will learn data distribution of the latent representation of the autoencoder. Together with the diffusion model, we define a beta scheduler responsible for defining the amount of noise tahat is added across the diffusion's model Markov chain.


'''
DDPM


'''
def calculate_3d_psnr(original, generated):
    mse = F.mse_loss(generated, original, reduction='mean')
    max_pixel_value = original.max()
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()

print("Train DDPM")

unet = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=64,  # Match the embedding_dim from the modified VQVAE
    out_channels=64,  # Match the embedding_dim from the modified VQVAE
    num_res_blocks=8,  # Keep the number of residual blocks
    num_channels=(256, 512, 512),  # Match the new autoencoder's number of channels
    attention_levels=(False, True, True),  # Keep the attention levels consistent
    num_head_channels=(0, 512, 512),  # Adjust num_head_channels to match the attention levels
)

#     transformer_num_layers: number of layers of Transformer blocks to use.
unet.to(device)


scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195)
# -

# ### Scaling factor
#
# As mentioned in Rombach et al. [1] Section 4.3.2 and D.1, the signal-to-noise ratio (induced by the scale of the latent space) can affect the results obtained with the LDM, if the standard deviation of the latent space distribution drifts too much from that of a Gaussian. For this reason, it is best practice to use a scaling factor to adapt this standard deviation.
#
# _Note: In case where the latent space is close to a Gaussian distribution, the scaling factor will be close to one, and the results will not differ from those obtained when it is not used._
#

# +
with torch.no_grad():
    with autocast(enabled=True):
        z = autoencoder.encode_stage_2_inputs(check_data["image"].to(device))

print(f"Scaling factor set to {1/torch.std(z)}")
scale_factor = 1 
# -

# We define the inferer using the scale factor:

inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

optimizer_diff = torch.optim.Adam(params=unet.parameters(), lr=1e-4)

# ### Train diffusion model

# +

epoch_loss_list = []
autoencoder.eval()
scaler = GradScaler()

first_batch = first(train_loader)
z = autoencoder.encode_stage_2_inputs(first_batch["image"].to(device))



# Path to the specific checkpoint you want to load
specific_checkpoint_path = "/data/wangfeiran/result/monai_results/20_flair_2/ddpm/checkpoints/checkpoint_epoch_100.pth"

# Check if the checkpoint exists
if os.path.exists(specific_checkpoint_path):
    # Load the specific checkpoint
    checkpoint = torch.load(specific_checkpoint_path)

    # Load the model and optimizer states
    unet.load_state_dict(checkpoint['unet_state_dict'])
    autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
    optimizer_diff.load_state_dict(checkpoint['optimizer_diff_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # Set the start epoch to the one in the checkpoint
    start_epoch = checkpoint['epoch']
    
    # Load the loss history
    epoch_loss_list = checkpoint['epoch_loss_list']

    print(f"Resuming training from epoch {start_epoch} using checkpoint {specific_checkpoint_path}")
else:
    # If the checkpoint doesn't exist, start from scratch
    start_epoch = 0
    epoch_loss_list = []
    print(f"No checkpoint found at {specific_checkpoint_path}, starting from scratch.")

autoencoder.eval()
SSIM_ddpm_list = []
MSSSIM_ddpm_list = []
for epoch in range(start_epoch, ddpm_n_epochs):
    unet.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch["image"].to(device)
        optimizer_diff.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            # Generate random noise
            noise = torch.randn_like(z).to(device)

            # Create timesteps
            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
            ).long()

            # Get model prediction
            noise_pred = inferer(
                inputs=images, autoencoder_model=autoencoder, diffusion_model=unet, noise=noise, timesteps=timesteps
            )

            loss = F.mse_loss(noise_pred.float(), noise.float())

        
        scaler.scale(loss).backward()
        scaler.step(optimizer_diff)
        scaler.update()

        epoch_loss += loss.item()

        progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
    epoch_loss_list.append(epoch_loss / (step + 1))
    if (epoch + 1) % ddpm_val_interval == 0:
        unet.eval()
        noise = torch.randn_like(z).to(device)
        print("z", z.shape)
        scheduler.set_timesteps(num_inference_steps=1000)
        synthetic_images = inferer.sample(
            input_noise=noise, autoencoder_model=autoencoder, diffusion_model=unet, scheduler=scheduler
        )
        # print("images",images.shape)
        # print("synthetic_images",synthetic_images.shape)

        # # Extract the image slice from the synthetic images tensor
        # img = synthetic_images[idx, channel].detach().cpu().numpy()

        # # Plot the middle slice of the 3D volume along the third axis (depth)
        # plt.imshow(img[..., img.shape[2] // 2], cmap="gray")
        # plt.axis("off")

        # # Construct the full path for the image file
        # image_path = os.path.join(ddpm_dir, f"synthetic_{epoch+1}.png")

        # # Save the image
        # plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        # plt.close()

        # Select a slice to visualize from the 3D mask, sample, and original image
        slice_index = 30  # You can change this index to visualize different slices
        plt.figure(figsize=(8, 4))  # Increase figsize to accommodate both images side by side

        # Plot the original image corresponding to the mask
        plt.subplot(1, 2, 1)
        plt.imshow(images[0, 0, :, :, slice_index].cpu(), vmin=0, vmax=1, cmap="gray")
        plt.axis("off")
        plt.title("Original Image")

        # Plot the generated sample
        plt.subplot(1, 2, 2)
        plt.imshow(synthetic_images[0, 0, :, :, slice_index].cpu(), vmin=0, vmax=1, cmap="gray")
        plt.axis("off")
        plt.title("Sample Image")

        plt.tight_layout()

        # Save the images with epoch number as the name
        image_save_path = os.path.join(ddpm_dir, f"ddpm_image_epoch_{epoch + 1}.png")
        plt.savefig(image_save_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the plot to avoid displaying it during training

        print("Sample shape:", synthetic_images.shape)
        print("Images shape:", images.shape)

        # Calculate SSIM and MS-SSIM
        if images.shape[0] == 1 and synthetic_images.shape[0] > 1:
            images = images.expand(synthetic_images.shape[0], -1, -1, -1, -1)
        ssim_now = ssim(synthetic_images.float().cpu(), images.float().cpu())
        ms_ssim_now = ms_ssim(synthetic_images.float().cpu(), images.float().cpu())
        mean_ssim = ssim_now.mean().item()
        mean_ms_ssim = ms_ssim_now.mean().item()
        print("SSIM:", mean_ssim)
        print("MSSSIM:", mean_ms_ssim)
        mmd_now = mmd(synthetic_images.float().cpu(), images.float().cpu())
        psnr_now = calculate_3d_psnr(synthetic_images.float().cpu(), images.float().cpu())
        mean_mmd = mmd_now.mean().item()
        mean_psnr = psnr_now
        print("mean_mmd:", mean_mmd)
        print("mean_psnr:", mean_psnr)
        # Function to calculate and print covariance for a metric
        def print_covariance(metric_values, metric_name):
            # Calculate the covariance matrix of the metric
            cov_matrix = np.cov(metric_values)
            print(f"Covariance of {metric_name}:")
            print(cov_matrix)
            print()  # Add an empty line for readability

        # Calculate and print covariance for each metric
        print_covariance(ssim_now, "SSIM")
        print_covariance(ms_ssim_now, "MS-SSIM")
        print_covariance(mmd_now, "MMD")
        print_covariance(psnr_now, "PSNR")
        SSIM_ddpm_list.append(mean_ssim)
        MSSSIM_ddpm_list.append(mean_ms_ssim)

        checkpoint_path = os.path.join(ddpm_checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'unet_state_dict': unet.state_dict(),
            'autoencoder_state_dict': autoencoder.state_dict(),
            'optimizer_diff_state_dict': optimizer_diff.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'epoch_loss_list': epoch_loss_list,
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    progress_bar.close()

# plt.style.use("ggplot")
# plt.title("Learning Curves", fontsize=20)
# plt.plot(epoch_loss_list)
# plt.yticks(fontsize=12)
# plt.xticks(fontsize=12)
# plt.xlabel("Epochs", fontsize=16)
# plt.ylabel("Loss", fontsize=16)
# plt.legend(prop={"size": 14})
# # Construct the full path for the image file
# image_path = os.path.join(ddpm_dir, f"Learning Curves.png")
# plt.savefig(image_save_path, bbox_inches='tight', pad_inches=0)
# plt.close() 
# torch.cuda.empty_cache()

# -
# -

# plt.plot(epoch_loss_list)
# plt.title("Learning Curves", fontsize=20)
# plt.plot(epoch_loss_list)
# plt.yticks(fontsize=12)
# plt.xticks(fontsize=12)
# plt.xlabel("Epochs", fontsize=16)
# plt.ylabel("Loss", fontsize=16)
# plt.legend(prop={"size": 14})
# image_path = os.path.join(model_dir, f"Learning Curves.png")
# plt.savefig(image_path)
# plt.close()




from generative.metrics import FIDMetric, MMDMetric, MultiScaleSSIMMetric, SSIMMetric, MMDMetric
ms_ssim = MultiScaleSSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=2)
ssim = SSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=2)
mmd = MMDMetric()
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# Directory to save the 2D slices
output_dir = model_dir+  "/inference_slices_2"
os.makedirs(output_dir, exist_ok=True)

# Lists to store metrics for the inferred images
control_SSIM_list = []
control_MSSSIM_list = []
control_mmd_list = []
psnr_values_list = []

# Lists to store metrics for the inferred images
control_SSIM_list = []
control_MSSSIM_list = []
control_mmd_list = []
psnr_values_list = []
# Set the model to evaluation mode
autoencoder.eval()
unet.eval()
# Lists to store the results
ssim_list = []
ms_ssim_list = []
mmd_list = []
psnr_list = []

# Set the model to evaluation mode
autoencoder.eval()
unet.eval()

def calculate_3d_psnr(original, generated):
    mse = F.mse_loss(generated, original, reduction='mean')
    max_pixel_value = original.max()
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()

# Loop over the training data for inference
progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
progress_bar.set_description("Inference")


for step, batch in progress_bar:
        images = batch["image"].to(device)
        masks = batch["label"].to(device)
        images_cond = torch.cat([images, masks], dim=1).to(device)  
        optimizer_diff.zero_grad(set_to_none=True)

        noise = torch.randn_like(z).to(device)
        print("z", z.shape)
        scheduler.set_timesteps(num_inference_steps=1000)
        generated_images = inferer.sample(
            input_noise=noise, autoencoder_model=autoencoder, diffusion_model=unet, scheduler=scheduler
        )
        

        print("generated_images",generated_images.shape)
        print("images",images.shape)


        # Assuming images and generated_images are already loaded as tensors with the shape [16, 1, height, width, depth]
        batch_size = images.shape[0]

        # Loop over the batch (each image pair)
        for i in range(batch_size):
            new_generated_image_single = generated_images[i].unsqueeze(0)
            print(new_generated_image_single.shape)
            new_images = images[i].unsqueeze(0)
            print(new_images.shape)
            # Calculate metrics for each pair of images
            ssim_now = ssim(new_generated_image_single.float().cpu(), new_images.float().cpu())
            ms_ssim_now = ms_ssim(new_generated_image_single.float().cpu(), new_images.float().cpu())
            mmd_now = mmd(new_generated_image_single.float().cpu(), new_images.float().cpu())
            psnr = calculate_3d_psnr(new_images, new_generated_image_single)
            
            print("ssim_now", ssim_now)
            print("ms_ssim_now", ms_ssim_now)
            print("mmd_now", mmd_now)
            print("psnr", psnr)
            
            # Append the calculated metrics to their respective lists
            ssim_list.append(ssim_now.mean().item())
            ms_ssim_list.append(ms_ssim_now.mean().item())
            mmd_list.append(mmd_now.mean().item())
            psnr_list.append(psnr)

        # Convert lists to tensors for calculating mean and variance
        ssim_tensor = torch.tensor(ssim_list)
        ms_ssim_tensor = torch.tensor(ms_ssim_list)
        mmd_tensor = torch.tensor(mmd_list)
        psnr_tensor = torch.tensor(psnr_list)

        # Calculate mean and variance
        mean_ssim = ssim_tensor.mean().item()
        var_ssim = ssim_tensor.var(unbiased=False).item()

        mean_ms_ssim = ms_ssim_tensor.mean().item()
        var_ms_ssim = ms_ssim_tensor.var(unbiased=False).item()

        mean_mmd = mmd_tensor.mean().item()
        var_mmd = mmd_tensor.var(unbiased=False).item()

        mean_psnr = psnr_tensor.mean().item()
        var_psnr = psnr_tensor.var(unbiased=False).item()

        # Print the final results
        print(f"Mean SSIM: {mean_ssim}, Variance SSIM: {var_ssim}")
        print(f"Mean MS-SSIM: {mean_ms_ssim}, Variance MS-SSIM: {var_ms_ssim}")
        print(f"Mean MMD: {mean_mmd}, Variance MMD: {var_mmd}")
        print(f"Mean PSNR: {mean_psnr}, Variance PSNR: {var_psnr}")

   







