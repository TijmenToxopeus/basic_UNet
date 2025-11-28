# 🧩 basic_UNet

A lightweight and modular **2D U-Net framework** for medical image segmentation, with a full pipeline for **training, evaluation, structured pruning, rewinding, model inspection, and experiment automation**. Developed and tested on the **ACDC cardiac MRI dataset**, but compatible with any 2D segmentation dataset.

---

# 📌 Table of Contents
- [✨ Features](#-features)
- [📁 Project Structure](#-project-structure)
- [📦 Installation](#-installation)
- [📚 Dataset](#-dataset)
- [⚙️ Configuration System](#️-configuration-system)
- [🚀 Baseline Training](#-baseline-training)
- [🎯 Evaluation](#-evaluation)
- [✂️ Structured Pruning Pipeline](#️-structured-pruning-pipeline)
  - [L1 block-wise pruning](#l1-block-wise-pruning)
  - [Rewinding options](#rewinding-options)
  - [Rebuilding a pruned UNet](#rebuilding-a-pruned-unet)
- [🧪 Model Inspection & L1 Analysis](#-model-inspection--l1-analysis)
- [📊 Experiment Logging](#-experiment-logging)
- [🧵 Full Experiment Runner](#-full-experiment-runner)
- [📈 Example Results](#-example-results)
- [🛣️ Roadmap](#️-roadmap)
- [🧠 Author](#-author)

---

# ✨ Features

### 🧠 UNet Architecture
- Clean, modular UNet defined in `src/models/unet.py`
- Easily modifiable architecture (depth, channels, features)

### 🚀 Training & Evaluation
- Full baseline training pipeline (`src/pipeline/baseline.py`)
- Dice, IoU, and loss logging
- Learning rate scheduling
- Automatic checkpointing
- Evaluation pipeline (`src/training/eval.py`)

### ✂️ Structured L1 Pruning (Block-wise)
- L1 filter norm computation
- Block-wise pruning ratios (e.g., `decoders.1: 0.3`)
- Pruning masks stored as JSON
- Rebuild a smaller pruned UNet automatically

### 🔄 Weight Reinitialization Modes
- `none` → keep weights post-pruning
- `random` → reinitialize pruned model from scratch
- `rewind` → restore weights from early checkpoint

### 📉 Model Inspection
- L1 histograms
- Layer statistics
- Channel shapes
- Visualization tools

### ⚙️ Dynamic Configuration System
- YAML config with structured training + pruning configuration
- Runtime overrides (epochs, LR, pruning mode, ratios)
- Automatic path generation via `utils/paths.py`

### 🧪 Experiment Automation
- `run_full_exp.py` runs a full sweep: `baseline → prune → retrain/evaluate → repeat for each mode`

### 📈 Logging
- Local logging (JSON, PNG, checkpoints)
- W&B integration available

---

# 📁 Project Structure

    src/
        models/
            unet.py                # U-Net architecture

        pipeline/
            baseline.py            # Baseline training pipeline
            pruned.py              # Pruning + retraining pipeline
            run_full_exp.py        # Automates full experiment runs

        pruning/
            l1_pruning.py          # L1 mask generation + pruning logic
            model_inspect.py       # Inspect shapes, channels, parameters
            rebuild.py             # Rebuild pruned UNet
            visualize_pruning.py   # Mask visualization tools
            l1_analysis/           # Histograms, stats, notebooks

        training/
            data_loader.py         # ACDC dataset handling
            train.py               # Training loop
            eval.py                # Evaluation loop
            metrics.py             # Dice, IoU, etc.
            loss.py                # Loss functions

        utils/
            config.py              # YAML loader + overrides
            paths.py               # Experiment folder management
            checkpoint.py          # Saving/loading checkpoints
            wandb_utils.py         # Optional logging to W&B

        config.yaml                # Main configuration file
        main.py                    # Optional runner

---

# 📦 Installation

    git clone https://github.com/TijmenToxopeus/basic_UNet.git
    cd basic_UNet
    pip install -r requirements.txt

Requires:
- Python ≥ 3.10  
- PyTorch ≥ 2.0  

---

# 📚 Dataset

The framework expects simple 2D image–mask pairs. For ACDC, structure like:

    data/
        images/
        masks/

Specify paths in `config.yaml`.

---

# ⚙️ Configuration System

All experiment settings are defined in:

    config.yaml

Example:

    model:
      in_channels: 1
      out_channels: 4

    training:
      batch_size: 8
      learning_rate: 1e-3
      num_epochs: 40

    pruning:
      block_ratios:
        encoders.1: 0.1
        decoders.3: 0.3
      reinitialize_weights: rewind

Pipelines may override LR, epochs, pruning ratio, or rewinding mode during sweeps.

---

# 🚀 Baseline Training

Train the full UNet:

    python -m src.pipeline.baseline

Outputs include:
- model checkpoints  
- `metrics.json`  
- `training_curves.png`  
- prediction samples  

---

# 🎯 Evaluation

Evaluate a trained model:

    python -m src.training.eval

Metrics include:
- Dice score  
- IoU  
- Pixel accuracy  
- Precision/recall  

---

# ✂️ Structured Pruning Pipeline

Prune the UNet and evaluate:

    python -m src.pipeline.pruned --mode rewind

Modes:
- `none`  
- `random`  
- `rewind`  

---

## L1 Block-wise Pruning

Block ratios define how many filters to prune in each block.

Example:

    block_ratios:
      encoders.0: 0.0
      encoders.1: 0.1
      decoders.3: 0.4
      decoders.5: 0.2

Process:
1. Compute L1 norm  
2. Rank filters  
3. Drop lowest-norm filters  
4. Save pruning mask  
5. Apply pruning to UNet  

---

## Rewinding Options

| Mode   | Description                            |
|--------|----------------------------------------|
| none   | Keep pruned weights                    |
| random | Reinitialize the pruned model          |
| rewind | Restore weights from an earlier checkpoint |

Example:

    python -m src.pipeline.pruned --reinitialize_weights rewind

---

## Rebuilding a Pruned UNet

    python -m src.pruning.rebuild

This script:
- Reads pruning masks  
- Computes new channel sizes  
- Builds a reduced UNet  
- Loads surviving weights  

---

# 🧪 Model Inspection & L1 Analysis

Inspect L1 statistics:

    python -m src.pruning.model_inspect

Generates:
- Histograms  
- Layer statistics  
- CSV summaries  

Located in:

    results/analysis/

Notebooks:
- `l1_distributions.ipynb`  
- `pruning_notebook.ipynb`  

---

# 📊 Experiment Logging

### Local Logging (default)

    results/<experiment>/<timestamp>/

Includes:
- `metrics.json`  
- training curves  
- sample predictions  
- model `.pt` files  

### Weights & Biases (optional)

Enable via:

    logging:
      use_wandb: true
      project: "basic_unet_pruning"

---

# 🧵 Full Experiment Runner

Run the entire pipeline:

    python -m src.pipeline.run_full_exp

This performs:

    1. Train baseline
    2. Prune (mode=none)
    3. Prune (mode=random)
    4. Prune (mode=rewind)
    5. Evaluate all

Each experiment overrides LR, epochs, and pruning settings automatically.

---

# 📈 Example Results

| Model         | Params | FLOPs     | Dice | Notes            |
|---------------|--------|----------:|------|------------------|
| Baseline UNet | 1.9M   | 55 GFLOPs | 0.88 | —                |
| Pruned 30%    | 1.4M   | 38 GFLOPs | 0.87 | Smaller model    |
| Pruned 50%    | 1.0M   | 28 GFLOPs | 0.85 | More aggressive  |

Training/validation curves saved as:

    training_curves.png

---

# 🛣️ Roadmap

- [ ] Learning rate finder  
- [ ] FLOPs/latency benchmarking  
- [ ] Add Attention UNet / UNet++  
- [ ] 3D support  
- [ ] Combined pruning + quantization  
- [ ] Export models to ONNX/TensorRT  

---

# 🧠 Author

**Tijmen Toxopeus**  
Master’s student in Applied Physics (TU Delft)  
Focus: medical image segmentation, structured pruning, efficient deep learning.
