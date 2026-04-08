# basic_UNet

A modular 2D U-Net repository for medical image segmentation, with support for baseline training, structured pruning, retraining, evaluation, and experiment analysis.

The current project is set up around cardiac MRI segmentation experiments, mainly using ACDC-style data and pruning studies on trained U-Net models.

## What This Repo Does

- Train a baseline 2D U-Net for multi-class segmentation
- Evaluate baseline, pruned, and retrained-pruned models
- Prune channels block-wise with multiple pruning methods
- Rebuild smaller pruned U-Nets from pruning masks
- Compare pruning settings such as layer choice, ratio, threshold, and reinitialization mode
- Analyze experiment outputs with notebooks and plotting scripts

## Repository Layout

```text
basic_UNet/
├── README.md
├── requirements.txt
├── src/
│   ├── config.yaml                 # Main experiment config
│   ├── analysis/                   # Plotting utilities for pruning experiments
│   ├── models/                     # U-Net definition
│   ├── notebooks/                  # Analysis and experiment notebooks
│   ├── pipeline/                   # End-to-end training/pruning workflows
│   ├── pruning/                    # Pruning methods, rebuild, reinit, summaries
│   ├── training/                   # Data loading, training, evaluation, metrics
│   └── utils/                      # Paths, config loading, reproducibility, logging
├── results/                        # Local experiment outputs
└── toy/                            # Small toy experiments (ignored in Git)
```

## Setup

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The requirements file now includes the packages used by the current training, pruning, evaluation, and analysis code.

## Configuration

The main config lives at:

[`src/config.yaml`](/mnt/hdd/ttoxopeus/basic_UNet/src/config.yaml)

It defines:

- experiment name, model name, seed, device
- training hyperparameters
- training and evaluation dataset paths
- pruning method, ratios, threshold, and reinitialization mode

Important implementation detail: the config loader currently uses a hard-coded absolute path in [`src/utils/config.py`](/mnt/hdd/ttoxopeus/basic_UNet/src/utils/config.py). If you move this repo to another machine or directory, update that file or make the config path configurable.

## Data Format

Training and evaluation expect image and label folders containing matching `.nii.gz` files. The configured paths point to separate image and label directories, for example:

```text
imagesTr/
├── patient001_frame01.nii.gz
├── patient001_frame12.nii.gz
└── ...

labelsTr/
├── patient001_frame01.nii.gz
├── patient001_frame12.nii.gz
└── ...
```

The dataset loader:

- loads 3D NIfTI volumes
- extracts 2D slices
- resizes slices to a fixed input size
- applies patient-level train/validation splitting
- supports augmentation with Albumentations and TorchIO during training

## Typical Workflows

### 1. Train and evaluate a baseline model

```bash
python -m src.pipeline.baseline
```

This runs:

1. baseline training
2. baseline evaluation
3. config snapshot export into the experiment folder

### 2. Run pruning, evaluate, retrain, and evaluate again

```bash
python -m src.pipeline.pruned
```

This pipeline:

1. loads the trained baseline checkpoint
2. computes pruning masks
3. rebuilds a smaller pruned U-Net
4. evaluates the pruned model
5. retrains the pruned model
6. evaluates the retrained pruned model

### 3. Run scripted pruning sweeps

```bash
python -m src.pipeline.run_full_exp
```

This file is currently used for custom experiment sweeps by editing the script directly. It contains several commented sweep variants and one active sweep implementation.

## Pruning Options

The pruning system supports multiple methods under [`src/pruning/methods`](/mnt/hdd/ttoxopeus/basic_UNet/src/pruning/methods):

- `l1_norm`
- `l2_norm`
- `pearson_correlation`
- `cosine_similarity`
- `random_filters`

The config can specify:

- a global default pruning ratio
- per-block pruning ratios
- an optional threshold for similarity-based methods
- weight handling after pruning:
  - `null`: keep inherited weights
  - `random`: random reinitialization
  - `rewind`: restore early checkpoint weights

## Results Layout

Outputs are written under `results/<model_name>/<experiment_name>/`.

A typical experiment tree looks like:

```text
results/UNet_ACDC/<experiment_name>/
├── baseline/
│   ├── training/
│   └── evaluation/
├── pruned/
│   └── <method_and_ratio_suffix>/
│       ├── pruned_model/
│       ├── pruned_evaluation/
│       ├── retraining_pruned/
│       └── retrained_pruned_evaluation/
└── logs/
```

Saved artifacts include:

- model checkpoints
- evaluation metrics
- training curves
- pruning metadata
- run summaries
- prediction visualizations

## Analysis

Analysis utilities live in [`src/analysis`](/mnt/hdd/ttoxopeus/basic_UNet/src/analysis) and experiment notebooks live in [`src/notebooks`](/mnt/hdd/ttoxopeus/basic_UNet/src/notebooks).

These scripts are mainly for:

- plotting pruning sensitivity by layer
- plotting uniform pruning curves
- visualizing feature maps
- generating experiment figures for reports

## Notes

- `wandb/`, `toy/`, caches, and generated artifacts are now ignored locally and should not be committed.
- The notebooks are kept in the repo because they appear to be part of the analysis workflow, not just disposable scratch files.
- Some scripts still reflect active experimentation and may need small cleanup passes before being used as polished public entry points.

## Next Cleanup Ideas

If you want this repo to be easier for others to run, the highest-value follow-ups would be:

1. remove the hard-coded config path
2. separate stable pipeline entry points from one-off experiment scripts
3. add one small example config for a public dataset layout
