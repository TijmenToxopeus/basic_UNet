# basic_UNet

A modular 2D U-Net repository for medical image segmentation, with support for baseline training, structured pruning, retraining, evaluation, and experiment analysis.

The current project is set up around cardiac MRI segmentation experiments, mainly using ACDC-style data and pruning studies on trained U-Net models.

## Project Background

This repository accompanies a thesis on structured channel pruning for cardiac MRI segmentation with a 2D U-Net. The project focuses on the trade-off between segmentation quality, computational efficiency, and robustness under domain shift.

### Abstract

Cardiac MRI segmentation models have improved substantially in recent years, but these gains have often come at the cost of increasing model size and computational complexity, which can limit their practical deployment. This thesis investigates structured channel pruning for cardiac MRI segmentation with a 2D U-Net, with a particular focus on segmentation performance, computational efficiency, and generalization under domain shift.

A systematic evaluation is conducted for two structured pruning criteria: importance-based pruning using the l1-norm and redundancy-based pruning using Pearson correlation. The pruned models are evaluated on the in-domain ACDC dataset and the out-of-domain M\&M dataset, with an additional ablation study on the STONE dataset. The analysis considers the effect of retraining, pruning ratio, pruning location, and pruning criterion, and compares both segmentation accuracy and practical efficiency measures such as parameter count, FLOPs, inference time, and GPU memory usage.

The results show that post-pruning retraining in the form of fine-tuning is essential to recover performance that was lost by pruning. On the ACDC dataset, l1-norm pruning achieves the best trade-off, reaching up to 96% model compression with only a 0.93% Dice loss. The analysis further reveals that most redundancy is located in the deepest layers of the network, particularly the bottleneck. These in-domain findings also extend to the out-of-domain setting. Prior studies and common assumptions suggest that structured pruning degrades out-of-domain performance. In contrast, this work shows that it not only preserves segmentation performance but can even substantially improve robustness under domain shift. On the M\&M dataset, pruned models outperform the baseline by up to 5% Dice, while maintaining similar performance limits as in-domain. This behavior is further supported by the additional ablation study on the STONE dataset, which shows consistent pruning trends under a different type of domain shift.

Overall, this thesis shows that structured pruning is an effective compression strategy for cardiac MRI segmentation. When combined with retraining, it enables substantially smaller and more efficient models while largely preserving predictive performance and, in some cases, improving robustness under domain shift.

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
│   ├── quantization/               # Preliminary quantization experiments
│   ├── training/                   # Data loading, training, evaluation, metrics
│   └── utils/                      # Paths, config loading, reproducibility, logging
└── results/                        # Local experiment outputs
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

- The notebooks are not explicit parts of the main training and pruning pipelines, but they are used for data exploration, post-processing experiment outputs, and generating plots and figures for analysis and reporting.
- The [`src/quantization`](/mnt/hdd/ttoxopeus/basic_UNet/src/quantization) folder contains preliminary code that is currently only used for a quick experiment, not as a stable part of the main workflow.
