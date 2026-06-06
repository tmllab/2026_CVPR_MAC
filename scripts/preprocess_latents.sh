#!/usr/bin/env bash
set -euo pipefail

echo "Encoding CelebA-HQ latents"
python preprocess.py \
  --dataset celeba \
  --data_root .. \
  --device cuda \
  --size 256 \
  --vae_name stabilityai/sd-vae-ft-mse

echo "Encoding AFHQ latents"
python preprocess.py \
  --dataset afhq \
  --data_root .. \
  --device cuda \
  --size 256 \
  --vae_name stabilityai/sd-vae-ft-mse
