# Repository Guidelines

## Project Structure & Module Organization

- `main.py`: primary training entry point. It builds datasets, model variants, wrappers, checkpoints, and validation samples.
- `eval.py`: sampling and metric evaluation entry point, including optional FID calculation.
- `preprocess.py`: offline VAE latent encoding for AFHQ/CelebA-style parquet/image folders.
- `models/`: neural network definitions, including DiT variants and the UNet implementation under `models/unet/`.
- `wrappers/`: training-objective implementations such as `meanflow`, `flow_matching`, `shortcut`, `batchot`, and utilities.

Generated outputs are written to local runtime directories such as `data/`, `checkpoints/`, and `contents/`; keep these out of source control.

## Build, Test, and Development Commands

There is no package manifest. Use a Python environment with `torch`, `torchvision`, `pytorch_lightning`, `diffusers`, `torchmetrics`, `pandas`, `numpy`, `Pillow`, and `tqdm`.

```bash
python main.py --dataset cifar --method meanflow --gpus 1
```

Runs training with CIFAR-10 defaults. Datasets may download into `./data`.

```bash
python eval.py --dataset cifar --method meanflow --sample_steps 64
```

Runs sample generation/evaluation; add `--calculate_fid` when FID is needed.

```bash
python preprocess.py
```

Encodes AFHQ/CelebA-style images to latent `.pth` files. Check the hard-coded paths before running.

## Coding Style & Naming Conventions

Use standard Python style with 4-space indentation. Keep module names lowercase with underscores, matching `wrappers/flow_matching.py`. Use `CamelCase` for model and dataset classes, and descriptive snake_case for functions and variables. Add experiment options through `argparse`.

Comments may be bilingual where they clarify research logic, but keep new comments short and tied to non-obvious behavior.

## Testing Guidelines

No automated tests are currently present. For wrapper or dataset changes, add focused `pytest` tests under a new `tests/` directory when practical. At minimum, run a small smoke test:

```bash
python main.py --dataset cifar --batch_size 4 --max_epochs 1 --gpus 1
```

For evaluation changes, run `eval.py` with a reduced `--num_samples` value.

## Commit & Pull Request Guidelines

Git history is not available in this checkout, so use concise imperative commit messages such as `Add latent preprocessing guard` or `Fix shortcut sampler device handling`.

Pull requests should describe the experiment or bug fix, list commands run, mention dataset/checkpoint assumptions, and include sample images or metric deltas for generation or FID changes.

## Security & Configuration Tips

Do not commit datasets, checkpoints, generated images, or local VAE/model cache files. Avoid embedding machine-specific absolute paths; expose paths as CLI arguments when adding new data pipelines.
