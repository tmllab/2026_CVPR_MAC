#!/usr/bin/env bash
set -euo pipefail

echo "Training conditional CIFAR-10"
python main.py \
  --dataset cifar \
  --percent 0.5 \
  --add_weight 1.0 \
  --batch_size 128 \
  --method meanflow \
  --gpus 2 \
  --max_epochs 250 \
  --lr 5e-4

echo "Training unconditional CIFAR-10"
python main.py \
  --dataset cifar \
  --noncond \
  --percent 0.5 \
  --add_weight 1.0 \
  --batch_size 128 \
  --method meanflow \
  --gpus 2 \
  --max_epochs 250 \
  --lr 5e-4

echo "Training unconditional CelebA-HQ"
python main.py \
  --dataset celeba \
  --batch_size 64 \
  --method meanflow \
  --gpus 4 \
  --max_epochs 900 \
  --lr 5e-4

echo "Training unconditional AFHQ"
python main.py \
  --dataset afhq \
  --batch_size 64 \
  --method meanflow \
  --gpus 4 \
  --max_epochs 900 \
  --lr 5e-4
