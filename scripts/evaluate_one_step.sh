#!/usr/bin/env bash
set -euo pipefail

echo "Evaluating one-step conditional CIFAR-10"
python eval.py \
  --dataset cifar \
  --percent 0.5 \
  --add_weight 1.0 \
  --sample_steps 1 \
  --calculate_fid \
  --batch_size 128 \
  --method meanflow \
  --model_type select

echo "Evaluating one-step unconditional CIFAR-10"
python eval.py \
  --dataset cifar \
  --noncond \
  --percent 0.5 \
  --add_weight 1.0 \
  --sample_steps 1 \
  --calculate_fid \
  --batch_size 128 \
  --method meanflow \
  --model_type select

echo "Evaluating one-step unconditional CelebA-HQ"
python eval.py \
  --dataset celeba \
  --sample_steps 1 \
  --calculate_fid \
  --batch_size 64 \
  --method meanflow \
  --model_type select

echo "Evaluating one-step unconditional AFHQ"
python eval.py \
  --dataset afhq \
  --sample_steps 1 \
  --calculate_fid \
  --batch_size 64 \
  --method meanflow \
  --model_type select
