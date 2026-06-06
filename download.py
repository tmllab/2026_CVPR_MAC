import argparse
import os

import pandas as pd
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


DATASET_CONFIGS = {
    "afhq": {
        "hf_name": "huggan/AFHQv2",
        "base_dir": "afhqv2-files",
        "splits": {
            "train": "train",
        },
    },
    "celeba": {
        "hf_name": "mattymchen/celeba-hq",
        "base_dir": "celeba-hq-files",
        "splits": {
            "train": "train",
            "validation": "val",
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Download image datasets used by this project.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_CONFIGS),
        required=True,
        help="Dataset to download.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="..",
        help="Root directory for dataset folders. Default matches main.py paths.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Local split names to process, for example: train val. Defaults to all configured splits.",
    )
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap per split for quick checks.",
    )
    return parser.parse_args()


def selected_splits(config, requested_splits):
    split_pairs = list(config["splits"].items())
    if requested_splits is None:
        return split_pairs

    local_to_hf = {local: hf for hf, local in split_pairs}
    unknown = sorted(set(requested_splits) - set(local_to_hf))
    if unknown:
        raise ValueError(f"Unknown split(s) for this dataset: {unknown}")

    return [(local_to_hf[local], local) for local in requested_splits]


def save_split(data, save_dir, local_split, chunk_size, max_samples=None):
    os.makedirs(save_dir, exist_ok=True)
    limit = len(data) if max_samples is None else min(len(data), max_samples)
    buffer = []

    print(f"Processing split: {local_split} -> {save_dir}")
    for idx, sample in tqdm(enumerate(data), total=limit):
        if idx >= limit:
            break

        img: Image.Image = sample["image"]
        label = sample["label"] if "label" in sample else 0

        img_name = f"{local_split}_{idx:06d}.png"
        img_path = os.path.join(save_dir, img_name)
        img.convert("RGB").save(img_path, format="PNG")

        buffer.append(
            {
                "image_path": img_path,
                "label": int(label),
            }
        )

        if (idx + 1) % chunk_size == 0 or (idx + 1) == limit:
            df = pd.DataFrame(buffer)
            parquet_path = os.path.join(save_dir, f"{local_split}_{idx // chunk_size:03d}.parquet")
            df.to_parquet(parquet_path, index=False)
            buffer = []
            print(f"Wrote {parquet_path}")


def main():
    args = parse_args()
    config = DATASET_CONFIGS[args.dataset]
    base_dir = os.path.join(args.data_root, config["base_dir"])
    os.makedirs(base_dir, exist_ok=True)

    dataset = load_dataset(config["hf_name"])
    available_splits = set(dataset.keys())
    print(f"Loaded {config['hf_name']} with splits: {sorted(available_splits)}")

    for hf_split, local_split in selected_splits(config, args.splits):
        if hf_split not in available_splits:
            raise ValueError(
                f"Split '{hf_split}' is not available in {config['hf_name']}. "
                f"Available splits: {sorted(available_splits)}"
            )
        save_dir = os.path.join(base_dir, local_split)
        save_split(dataset[hf_split], save_dir, local_split, args.chunk_size, args.max_samples)


if __name__ == "__main__":
    main()
