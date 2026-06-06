import argparse
import os

import pandas as pd
import torch
from diffusers.models import AutoencoderKL
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


DATASET_CONFIGS = {
    "afhq": {
        "base_dir": "afhqv2-files",
        "splits": ["train"],
    },
    "celeba": {
        "base_dir": "celeba-hq-files",
        "splits": ["train", "val"],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Encode AFHQ/CelebA images into SD-VAE latents.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_CONFIGS),
        required=True,
        help="Dataset to preprocess.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="..",
        help="Root directory containing dataset folders. Default matches main.py paths.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Split names to encode, for example: train val. Defaults depend on the dataset.",
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--vae_name", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for VAE encoding: auto, cuda, cuda:0, or cpu.",
    )
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def selected_splits(config, requested_splits):
    if requested_splits is None:
        return config["splits"]

    unknown = sorted(set(requested_splits) - set(config["splits"]))
    if unknown:
        raise ValueError(f"Unknown split(s) for this dataset: {unknown}")

    return requested_splits


def load_metadata(parquet_dir):
    parquet_paths = sorted(
        os.path.join(parquet_dir, name)
        for name in os.listdir(parquet_dir)
        if name.endswith(".parquet")
    )
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
    return pd.concat([pd.read_parquet(path) for path in parquet_paths], ignore_index=True)


def encode_split(parquet_dir, img_dir, latent_dir, vae, device, size=256):
    os.makedirs(latent_dir, exist_ok=True)
    df = load_metadata(parquet_dir)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    vae.to(device).eval()

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_name = os.path.basename(row["image_path"])
        label = int(row["label"]) if "label" in row else 0
        img_path = os.path.join(img_dir, img_name)

        img = Image.open(img_path).convert("RGB").resize((size, size))
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            posterior = vae.encode(x).latent_dist
            moments = posterior.parameters

            x_flip = torch.flip(x, dims=[3])
            posterior_flip = vae.encode(x_flip).latent_dist
            moments_flip = posterior_flip.parameters

        base, _ = os.path.splitext(img_name)
        save_path = os.path.join(latent_dir, base + ".pth")
        torch.save(
            {
                "moments": moments.squeeze(0).cpu(),
                "moments_flip": moments_flip.squeeze(0).cpu(),
                "label": label,
            },
            save_path,
        )

    print(f"Finished encoding to {latent_dir}")


def main():
    args = parse_args()
    config = DATASET_CONFIGS[args.dataset]
    device = resolve_device(args.device)
    base_dir = os.path.join(args.data_root, config["base_dir"])

    vae = AutoencoderKL.from_pretrained(args.vae_name)

    for split in selected_splits(config, args.splits):
        split_dir = os.path.join(base_dir, split)
        print(f"Encoding {args.dataset} split '{split}' from {split_dir}")
        encode_split(
            parquet_dir=split_dir,
            img_dir=split_dir,
            latent_dir=split_dir,
            vae=vae,
            device=device,
            size=args.size,
        )


if __name__ == "__main__":
    main()
