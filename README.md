# Beyond Optimal Transport: Model-Aligned Coupling for Flow Matching

This repository contains the official implementation for:

**Beyond Optimal Transport: Model-Aligned Coupling for Flow Matching**  
Yexiong Lin, Yu Yao, Yang Zhou, Tongliang Liu  
CVPR Findings 2026, pp. 3955--3964

Paper page: https://openaccess.thecvf.com/content/CVPR2026F/html/Lin_Beyond_Optimal_Transport_Model-Aligned_Coupling_for_Flow_Matching_CVPRF_2026_paper.html

## Abstract

Flow Matching (FM) is an effective framework for training a model to learn a vector field that transports samples from a source distribution to a target distribution. To train the model, early FM methods use random couplings, which often result in crossing paths and lead the model to learn non-straight trajectories that require many integration steps to generate high-quality samples. To address this, recent methods adopt Optimal Transport (OT) to construct couplings by minimizing geometric distances, which helps reduce path crossings. However, we observe that such geometry-based couplings do not necessarily align with the model's preferred trajectories, making it difficult to learn the vector field induced by these couplings, which prevents the model from learning straight trajectories. Motivated by this, we propose Model-Aligned Coupling (MAC), an effective method that matches training couplings based not only on geometric distance but also on alignment with the model's preferred transport directions based on its prediction error. MAC can be seamlessly integrated into existing frameworks, consistently enhancing their performance in few-step generation. Theoretically, we show that minimizing prediction error bounds trajectory curvature, thus promoting straighter transport paths. Extensive experiments show that MAC can improve the generation quality and efficiency of existing methods in few-step settings.

## Tutorial

We provide a tutorial [`mac_tutorial.ipynb`](mac_tutorial.ipynb) based on a 2D toy example to help readers quickly understand the idea of MAC. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tmllab/2026_CVPR_MAC/blob/main/mac_tutorial.ipynb)

## Repository Structure

- `main.py`: training entry point.
- `eval.py`: sampling and FID evaluation entry point.
- `download.py`: downloads CelebA-HQ and AFHQ from Hugging Face.
- `preprocess.py`: encodes CelebA-HQ and AFHQ images into Stable Diffusion VAE latents.
- `models/`: UNet and DiT model definitions.
- `wrappers/`: MAC training wrappers for each flow-matching baseline.
- `scripts/`: reproducibility scripts for download, preprocessing, and training.

## Environment Setup

Create and activate a Python environment, then install the required packages:

```bash
conda create -n mac python=3.10
conda activate mac
pip install -r requirements.txt
```

The provided `requirements.txt` installs PyTorch 2.3.1 and torchvision 0.18.1 from the CUDA 12.1 PyTorch wheel index:

```txt
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.3.1
torchvision==0.18.1
```

Make sure the server has a compatible NVIDIA driver and CUDA runtime before running training or latent preprocessing.

## Data Preparation

Run scripts from the repository root.

Download CelebA-HQ and AFHQ:

```bash
./scripts/download_datasets.sh
```

Preprocess CelebA-HQ and AFHQ into latent `.pth` files:

```bash
./scripts/preprocess_latents.sh
```

CIFAR-10 is downloaded automatically by `torchvision` during training.

## Training

Run all four paper training jobs:

```bash
./scripts/train_paper_experiments.sh
```

The script runs:

- conditional CIFAR-10 with `--percent 0.5 --add_weight 1.0`
- unconditional CIFAR-10 with `--percent 0.5 --add_weight 1.0`
- unconditional CelebA-HQ with default MAC weighting parameters
- unconditional AFHQ with default MAC weighting parameters

Final checkpoints are saved under `saved/`, and intermediate checkpoints/samples are written to `checkpoints/` and `contents/`.

## Evaluation

After training, generate samples or compute FID with:

```bash
python eval.py --dataset cifar --method meanflow --model_type select
python eval.py --dataset cifar --method meanflow --model_type select --calculate_fid
```

For unconditional CIFAR-10, add `--noncond`. CelebA-HQ and AFHQ are unconditional by default.

## Citation

```bibtex
@InProceedings{Lin_2026_CVPR,
    author    = {Lin, Yexiong and Yao, Yu and Zhou, Yang and Liu, Tongliang},
    title     = {Beyond Optimal Transport: Model-Aligned Coupling for Flow Matching},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
    month     = {June},
    year      = {2026},
    pages     = {3955-3964}
}
```
