#!/usr/bin/env bash
set -euo pipefail

echo "Downloading CelebA-HQ"
python download.py \
  --dataset celeba \
  --data_root .. \
  --chunk_size 1000

echo "Downloading AFHQ"
python download.py \
  --dataset afhq \
  --data_root .. \
  --chunk_size 1000
