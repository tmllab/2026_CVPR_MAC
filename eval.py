import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import torch
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid, save_image
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms
from pytorch_lightning import seed_everything

from models.unet.unet import UNetModelWrapper
from models.DiT import DiT_B_2
from diffusers.models import AutoencoderKL

class CelebaHQDataset(Dataset):
    def __init__(self, parquet_path, transform=None, size=256):
        self.parquet_path = parquet_path
        parquet_names = sorted([
            f for f in os.listdir(parquet_path)
            if f.endswith(".parquet")
        ])
        parquet_paths = [os.path.join(parquet_path, name) for name in parquet_names]
        
        if len(parquet_paths) < 1:
            raise FileNotFoundError("No parquet files found in the given path.")

        # Merge all parquet shards.
        parquets = [pd.read_parquet(p) for p in parquet_paths]
        # print(parquet_paths[:5])
        # print(parquets[:5])
        self.data = pd.concat(parquets, axis=0, ignore_index=True)
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.loc[idx, "image_path"].split('/')[-1]
        image_path = self.parquet_path + '/' + image_path
        label = self.data.loc[idx, "label"]

        try:
            image = Image.open(image_path).convert("RGB").resize((self.size, self.size))
        except Exception as e:
            print(f"Error loading image at {image_path}, skipping. Error: {e}")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        return image, label


def convert_to_rgb(images):
    """Convert grayscale images to RGB by repeating the channel 3 times."""
    if images.shape[1] == 1:
        return images.repeat(1, 3, 1, 1)
    return images

def calculate_fid(real_images, generated_images, device='cuda', feature_dims=2048, batch_size=100):
    """
    Calculate FID score between real and generated images in batches
    
    Args:
        real_images: Tensor of real images [N, C, H, W] in range [0, 1]
        generated_images: Tensor of generated images [N, C, H, W] in range [0, 1]
        device: Device to run FID calculation on
        feature_dims: Feature dimensions for FID calculation
        batch_size: Batch size to avoid OOM
    
    Returns:
        fid_score: The FID score
    """
    # Convert to RGB if needed - Inception model requires 3 channels
    real_images = convert_to_rgb(real_images)
    generated_images = convert_to_rgb(generated_images)

    # Initialize FID metric
    fid = FrechetInceptionDistance(feature=feature_dims, normalize=True).to(device)

    real_loader = DataLoader(TensorDataset(real_images), batch_size=batch_size)
    gen_loader = DataLoader(TensorDataset(generated_images), batch_size=batch_size)

    # Update real image stats
    for batch in real_loader:
        imgs = batch[0].to(device)
        fid.update(imgs, real=True)
        del imgs
        torch.cuda.empty_cache()

    # Update generated image stats
    for batch in gen_loader:
        imgs = batch[0].to(device)
        fid.update(imgs, real=False)
        del imgs
        torch.cuda.empty_cache()

    # Compute FID score
    fid_score = fid.compute()
    return fid_score

if __name__ == '__main__':
    seed = 42
    seed_everything(seed, workers=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',      type=str,   default='cifar', choices=['cifar', 'celeba', 'afhq'])
    parser.add_argument('--percent',      type=float, default=0.4)
    parser.add_argument('--add_weight',      type=float, default=0.5)
    parser.add_argument("--sample_steps", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument('--temp',      type=float, default=0.01)
    parser.add_argument('--batch_size',   type=int,   default=128)
    parser.add_argument("--calculate_fid", action="store_true")
    parser.add_argument('--lr',           type=float, default=5e-4)

    parser.add_argument('--dim',          type=int,   default=256)
    parser.add_argument('--n_layers',     type=int,   default=10)
    parser.add_argument('--n_heads',      type=int,   default=8)
    parser.add_argument('--max_epochs',   type=int,   default=200)
    parser.add_argument('--gpus',         type=int,   default=1)
    parser.add_argument('--noncond', action='store_true')
    parser.add_argument('--method', type=str, default='meanflow', choices=['meanflow', 'flow_matching', 'shortcut', 'batchot'])
    parser.add_argument('--model_type', type=str, default='select', choices=['select', 'full'])
    args = parser.parse_args()

    # data
    if args.dataset=='cifar':
        dataset_name = "cifar"
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        val_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        
        channels = 3
        num_classes = 10
        input_shape = (3,32,32)
        latent_shape = input_shape

        class_cond = not args.noncond

        model = UNetModelWrapper(
                    dim=input_shape,
                    num_res_blocks=2,
                    num_channels=128,
                    class_cond=class_cond,
                    num_classes=num_classes+1,
                    channel_mult=[1, 2, 2, 2],
                    input_step=True,
                    num_heads=4,
                    num_head_channels=64,
                    attention_resolutions="16",
                    dropout=0.1,
                )
        vae = None
    elif args.dataset=='celeba' or args.dataset=='afhq':
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        if args.dataset=='celeba':
            dataset_name = "celeba"
            train_set = CelebaHQDataset('../celeba-hq-files/train', transform=transform)
            val_set = CelebaHQDataset('../celeba-hq-files/val', transform=transform)
        else:
            dataset_name = "afhq"
            train_set = CelebaHQDataset('../afhqv2-files/train', transform=transform)
            val_set = CelebaHQDataset('../afhqv2-files/train', transform=transform)
        
        channels = 4
        num_classes = None
        input_shape = (3,256,256)

        latent_shape = (args.batch_size, 4, input_shape[2]//8, input_shape[2]//8)

        class_cond = False

        model = DiT_B_2(learn_sigma=False, 
                  num_classes=num_classes, 
                  class_dropout_prob=0.0,
                  training_type="shortcut")
        
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        vae = vae.eval()
        vae.requires_grad_(False)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=16)
    
    if args.method == 'meanflow':
        from wrappers.meanflow import MACWrapper
    elif args.method == 'flow_matching':
        from wrappers.flow_matching import MACWrapper
    elif args.method == 'shortcut':
        from wrappers.shortcut import MACWrapper
    elif args.method == 'batchot':
        from wrappers.batchot import MACWrapper

    # hyperparams dict
    hparams = {
        'channels':    channels,
        'latent_shape': latent_shape,
        'dataset': dataset_name,
        'num_classes': num_classes,
        'class_cond':  class_cond,
        'temp':     args.temp,
        'lr':          args.lr,

    }
    # build model + MAC wrapper
    
    mac = MACWrapper(model, vae, args.add_weight, model_type=args.model_type)
    mac.model.load_state_dict(torch.load('./saved/%s_%s_%s_mac_ema_model_final_p%.2f_w%.2f_class_cond_%s.pth'%(args.dataset, args.method, args.model_type, args.percent, args.add_weight, class_cond),
               map_location='cuda:0'))
    mac.model.cuda()
    mac.model.eval()
    os.makedirs("contents", exist_ok=True)

    with torch.no_grad():
        if class_cond:
            cond = torch.arange(0, 16).cuda() % 10
            uncond = torch.ones_like(cond) * 10
        else:
            cond = None
            uncond = None

        init_noise = torch.randn(16, channels, 32, 32).cuda()
        images = mac.sample(init_noise, cond, uncond, args.sample_steps)
        # image sequences to gif
        gif = []
        for image in images:
            # unnormalize
            image = image * 0.5 + 0.5
            image = image.clamp(0, 1)
            x_as_image = make_grid(image.float(), nrow=4)
            img = x_as_image.permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype(np.uint8)
            gif.append(Image.fromarray(img))

        gif[0].save(
            f"contents/sample_test_{args.sample_steps}.gif",
            save_all=True,
            append_images=gif[1:],
            duration=100,
            loop=0,
        )

        last_img = gif[-1]
        last_img.save(f"contents/sample_test_{args.sample_steps}_last.png")

    # Calculate FID score
    if args.calculate_fid:
        print(f"Calculating FID score with {args.num_samples} samples...")
        
        # Load test dataset for real images

        
        # Collect real images for FID calculation
        real_images = []
        for i, (images, _) in enumerate(train_loader):
            if i * args.batch_size >= args.num_samples:
                break
            real_images.append(images * 0.5 + 0.5)
        real_images = torch.cat(real_images, dim=0)[:args.num_samples]
        
        # Generate images for FID calculation
        generated_images = []
        total_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
        
        for i in tqdm(range(total_batches), desc="Generating images for FID"):
            curr_batch_size = min(args.batch_size, args.num_samples - i * args.batch_size)
            if curr_batch_size <= 0:
                break
                
            # Random classes for conditional generation
            if class_cond:
                classes = torch.randint(0, 10, (curr_batch_size,)).cuda()
                uncond = torch.ones_like(classes) * 10
            else:
                classes = None
                uncond = None
            
            # Generate images
            z = torch.randn(curr_batch_size, channels, 32, 32).cuda()
            sample_seq = mac.sample(z, classes, uncond, args.sample_steps)
            final_images = sample_seq[-1]
            
            # Unnormalize images to [0, 1] range
            final_images = final_images * 0.5 + 0.5
            final_images = final_images.clamp(0, 1)
            
            generated_images.append(final_images.cpu())
            
            # Save some sample images
            if i == 0:
                sample_grid = make_grid(final_images[:16], nrow=4)
                save_image(sample_grid, f"contents/fid_samples_{args.sample_steps}.png")
        
        generated_images = torch.cat(generated_images, dim=0)[:args.num_samples]
        
        # Make sure both tensors are on CPU for FID calculation
        real_images = real_images.cpu()
        generated_images = generated_images.cpu()
        print(real_images.mean(),generated_images.mean())
        print(real_images.std(), generated_images.std())   
        
        # Calculate FID score
        print(f"Real images shape: {real_images.shape}, Generated images shape: {generated_images.shape}")
        print("Calculating FID score...")
        fid_score = calculate_fid(real_images, generated_images, device='cuda')
        print(f"FID Score: {fid_score:.4f}")
        
        # Save FID score to file
        with open(f"contents/fid_score_{dataset_name}_{args.sample_steps}.txt", "w") as f:
            f.write(f"FID Score ({dataset_name}, {args.sample_steps} steps): {fid_score:.4f}\n")
            f.write(f"Number of samples: {args.num_samples}\n")
