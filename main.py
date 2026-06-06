import argparse
import os
import io
import pandas as pd
import numpy as np
from PIL import Image

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from models.unet.unet import UNetModelWrapper
from models.DiT import DiT_B_2
from diffusers.models import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution


class CelebaLatentDataset(Dataset):
    """
    Read .pth files saved by preprocess.py:
    {
        "moments":       [2*C, h, w]
        "moments_flip":  [2*C, h, w]
        "label":         int
    }

    Returns:
        latent [C, h, w] sampled through DiagonalGaussianDistribution.sample()
        label  int
    """
    def __init__(self,
                 latent_dir: str,
                 flip_prob: float = 0.5):
        self.latent_dir = latent_dir
        self.files = sorted([f for f in os.listdir(latent_dir) if f.endswith(".pth")])
        self.flip_prob = float(flip_prob)

        # Default Stable Diffusion latent scale/bias.
        self.latents_scale = torch.tensor(
            [0.18125, 0.18125, 0.18125, 0.18125]
        ).view(1, 4, 1, 1)

        self.latents_bias = torch.tensor(
            [0.0, 0.0, 0.0, 0.0]
        ).view(1, 4, 1, 1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.latent_dir, self.files[idx])

        try:
            data = torch.load(path, weights_only=True, map_location="cpu")
        except TypeError:
            data = torch.load(path, map_location="cpu")

        moments = data["moments"]         # [2*C, h, w]
        moments_flip = data["moments_flip"]
        label = data["label"]

        # Randomly choose whether to use the horizontally flipped version.
        use_flip = torch.rand(1).item() < self.flip_prob
        moments_to_use = moments_flip if use_flip else moments

        # Build the posterior and sample the latent.
        posterior = DiagonalGaussianDistribution(moments_to_use.unsqueeze(0))  # [1, 2*C, h, w]
        x = posterior.sample()                               # [1, C, h, w]
        x = x * self.latents_scale + self.latents_bias
        x = x.squeeze(0)  # Remove the batch dimension.

        return x, label

def get_current_percentile(step, warmup_steps=100, start_val=1.0, end_val=0.4):
    if step < warmup_steps:
        return start_val - step * (start_val - end_val) / warmup_steps
    else:
        return end_val

class MACModulePL(pl.LightningModule):
    def __init__(self, hparams, mac):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.mac = mac
        self.model = mac.model
        self.ema_model = mac.ema_model
        for p in self.ema_model.parameters():
            p.requires_grad = False


    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)

    def on_train_start(self):
        # move EMA model to same device
        self.mac.ema_model.to(self.device)
        if self.mac.vae is not None:
            self.mac.vae.to(self.device)

    def training_step(self, batch, batch_idx):
        x, c = batch
        percentile = get_current_percentile(self.global_step, warmup_steps=20000, start_val=1.0, end_val=self.hparams['percent'])
        loss = self.mac.forward(x, c, percentile, self.global_step)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        # Update EMA after each optimizer batch update.
        self.mac.update_ema()

    def on_train_epoch_end(self):
        if self.current_epoch % 20 == 0:
            save_dir = f"checkpoints/mf_{self.hparams['dataset']}_class_cond{self.hparams['class_cond']}"
            os.makedirs(save_dir, exist_ok=True)

            model_path = os.path.join(save_dir, f"model_epoch{self.current_epoch:04d}.pth")
            ema_path = os.path.join(save_dir, f"ema_model_epoch{self.current_epoch:04d}.pth")

            torch.save(self.model.state_dict(), model_path)
            torch.save(self.ema_model.state_dict(), ema_path)

            print(f"[Checkpoint] Saved model and EMA at epoch {self.current_epoch}")

    # optional: image sampling
    def sample_images(self, z, cond, null_cond=None, sample_steps=16, cfg=2.0):
        return self.mac.sample(z, cond, null_cond, sample_steps, cfg)
    
    def validation_step(self, batch, batch_idx):
        return None

    def on_validation_epoch_end(self):
        if self.mac.vae is not None:
            print("Moving VAE to device...")
            self.mac.vae.to(self.device)
        epoch = self.current_epoch

        self.mac.model.eval()
        device = self.device
        size = self.hparams['latent_shape'][-1]

        if self.hparams['class_cond']:
            cond   = torch.arange(0, 16, device=device) % self.hparams['num_classes']
            uncond = torch.ones_like(cond, device=device) * self.hparams['num_classes']
        else:
            cond   = torch.arange(0, 16, device=device)
            uncond = None
        init_noise = torch.randn(16, self.hparams['channels'], size, size, device=device)

        with torch.no_grad():
            images = self.mac.sample(init_noise, cond, uncond, sample_steps=1)
        
        save_dir = "contents/%s/%s_%s"%(self.hparams['dataset'], self.hparams['method'], self.hparams['model_type'])
        os.makedirs(save_dir, exist_ok=True)

        gif = []
        for img in images:
            grid = make_grid((img*0.5+0.5).clamp(0,1), nrow=4)
            arr  = (grid.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            gif.append(Image.fromarray(arr))

        gif[0].save(f"{save_dir}/sample_{epoch}.gif",
                    save_all=True, append_images=gif[1:], duration=100, loop=0)
        gif[-1].save(f"{save_dir}/sample_{epoch}_last.png")

        self.mac.model.train()

if __name__ == '__main__':
    seed = 42
    seed_everything(seed, workers=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',      type=str,   default='cifar', choices=['cifar', 'celeba', 'afhq'])
    parser.add_argument('--percent',      type=float, default=0.4)
    parser.add_argument('--add_weight',      type=float, default=0.5)
    parser.add_argument('--batch_size',   type=int,   default=64)
    parser.add_argument('--lr',           type=float, default=5e-4)

    parser.add_argument('--max_epochs',   type=int,   default=200)
    parser.add_argument('--gpus',         type=int,   default=1)
    parser.add_argument('--noncond', action='store_true')
    parser.add_argument('--method', type=str, default='meanflow', choices=['meanflow', 'flow_matching', 'shortcut', 'batchot'])
    parser.add_argument('--model_type', type=str, default='select', choices=['select', 'full'])
    parser.add_argument('--ckpt_path', type=str, default=None)
    args = parser.parse_args()

    if args.method == 'meanflow':
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    # data
    if args.dataset=='cifar':
        dataset_name = "cifar"
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
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
        if args.dataset=='celeba':
            dataset_name = "celeba"
            train_set = CelebaLatentDataset('../celeba-hq-files/train')
            val_set = CelebaLatentDataset('../celeba-hq-files/val')
        else:
            dataset_name = "afhq"
            train_set = CelebaLatentDataset('../afhqv2-files/train')
            val_set = CelebaLatentDataset('../afhqv2-files/train')
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        
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
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=4)

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
        'percent':     args.percent,
        'lr':          args.lr,

        'method':     args.method,
        'model_type': args.model_type,
    }
    # build model + MAC wrapper
    mac = MACWrapper(model, vae, args.add_weight, model_type=args.model_type)
    model_pl = MACModulePL(hparams, mac)

    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',
        filename=f'{args.method}-{args.model_type}-{args.dataset}-{args.percent:.2f}-{args.add_weight:.2f}-{class_cond}-{{epoch:04d}}',
        every_n_epochs=10
    )

    # Build Trainer keyword arguments.
    trainer_kwargs = {
        'max_epochs': args.max_epochs,
        'accelerator': 'gpu',
        'devices': args.gpus,
        'check_val_every_n_epoch': 5,
        'callbacks': [checkpoint_callback],
        'deterministic': True,
    }

    if args.dataset == 'celeba' or args.dataset == 'afhq':
        print('Using gradient clipping for CelebA/AFHQ training')
        trainer_kwargs = {
            'max_epochs': args.max_epochs,
            'accelerator': 'gpu',
            'devices': args.gpus,
            'check_val_every_n_epoch': 5,
            'callbacks': [checkpoint_callback],
            'gradient_clip_val': 1.0,
            'gradient_clip_algorithm': 'norm',
            'deterministic': True,
        }

    # Enable DDP only for multi-GPU runs.
    if args.gpus > 1:
        trainer_kwargs['strategy'] = 'ddp_find_unused_parameters_false'

    # Create the Trainer.
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model_pl, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.ckpt_path)
    
    if not os.path.exists('./saved'):
        os.makedirs('./saved')
    # save final weights
    torch.save(model_pl.model.state_dict(),     f'./saved/{args.dataset}_{args.method}_{args.model_type}_mac_model_final_p{args.percent:.2f}_w{args.add_weight:.2f}_class_cond_{class_cond}.pth')
    torch.save(model_pl.mac.ema_model.state_dict(), f'./saved/{args.dataset}_{args.method}_{args.model_type}_mac_ema_model_final_p{args.percent:.2f}_w{args.add_weight:.2f}_class_cond_{class_cond}.pth')
