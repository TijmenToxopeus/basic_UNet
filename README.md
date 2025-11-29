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
            l1_pruning.py          # Pruning pipeline (combines all pruning scripts)
            model_inspect.py       # L1 norm & mask functions + inspect shapes, features, distributions
            rebuild.py             # Rebuild pruned UNet functions
            l1_analysis/           # Histograms of l1 norm distributions between layers

        training/
            data_loader.py         # ACDC dataset handling (preprocessing + Dataset + Dataloader)
            train.py               # Training pipeline
            eval.py                # Evaluation pipeline
            metrics.py             # Dice, IoU, flops, inference time
            loss.py                # Loss functions and combinations

        utils/
            config.py              # YAML loader + overrides
            paths.py               # Experiment folder management
            wandb_utils.py         # W&B logging

        config.yaml                # Main configuration file

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

This framework uses **2D slices extracted from 3D NIfTI volumes (`.nii.gz`)**, such as those provided by the **ACDC cardiac MRI dataset**.  
Each patient folder contains end-diastolic (ED) and end-systolic (ES) frames together with corresponding ground-truth masks.

---

## 📂 Example Patient Folder (ACDC)
    patient001/
        patient001_4d.nii.gz # Full 4D cine MRI: (H, W, slices, time)
        patient001_frame01.nii.gz # ED frame (raw image)
        patient001_frame01_gt.nii.gz # ED segmentation mask
        patient001_frame12.nii.gz # ES frame (raw image)
        patient001_frame12_gt.nii.gz # ES segmentation mask
        MANDATORY_CITATION # Required citation file
        Info # Metadata
  
---

## 📘 Meaning of Each File

| File | Description |
|------|-------------|
| `patient001_4d.nii.gz` | Complete 4D cine stack (not always used directly) |
| `patient001_frame01.nii.gz` | End-diastolic (ED) volume |
| `patient001_frame01_gt.nii.gz` | ED ground-truth mask |
| `patient001_frame12.nii.gz` | End-systolic (ES) volume |
| `patient001_frame12_gt.nii.gz` | ES ground-truth mask |

The masks contain **integer class labels** (not RGB colors):
- 0 → background  
- 1 → RV  
- 2 → myocardium  
- 3 → LV  

---
## 🖼️ Example (ED and ES Slices)

### End-Diastolic (ED)
<table>
<tr>
<td><strong>Image</strong></td>
<td><strong>Overlay</strong></td>
</tr>
<tr>
<td><img src="data_examples/snapshot0001.png" width="300"/></td>
<td><img src="data_examples/snapshot0002.png" width="300"/></td>
</tr>
</table>

### End-Systolic (ES)
<table>
<tr>
<td><strong>Image</strong></td>
<td><strong>Overlay</strong></td>
</tr>
<tr>
<td><img src="data_examples/snapshot0003.png" width="300"/></td>
<td><img src="data_examples/snapshot0004.png" width="300"/></td>
</tr>
</table>


## 🔧 Preprocessing

Because the UNet model is **2D**, while MRI volumes are **3D**, the preprocessing pipeline converts full 3D NIfTI volumes into clean, normalized, augmentable **2D slices** that can be fed into the network.  
The preprocessing steps are as follows:

---

1. **Raw Data Organization**  
   The raw MRI scans and their corresponding labels are stored as full 3D NIfTI volumes (`.nii.gz`) in separate folders for images and labels.

2. **3D → 2D Slice Extraction**  
   Each 3D volume typically has shape `(H, W, Slices)`.  
   Because the model is 2D, individual 2D slices are extracted along a specified axis:
   ```python
   img2d = np.take(img3d, slice_index, axis=slice_axis)
   lbl2d = np.take(lbl3d, slice_index, axis=slice_axis)

3. **Normalization & Resizing**  
Every extracted 2D slice is resized to a fixed `256 × 256` resolution and normalized to produce consistent MRI contrast across patients.

4. **Data Augmentation (Training Only)**  
   During training, augmentations such as `RandomElasticDeformation`, `RandomNoise`, and `RandomGamma` are applied to:
   - increase dataset diversity  
   - improve robustness  
   - enhance generalization to unseen MRI scans  

5. **Train/Validation Split and Batching**  
After preprocessing, the dataset is split into training and validation subsets, which are loaded in batches and fed to the training loop.

# ⚙️ Configuration System

All experiment settings are defined in a single central file:

    config.yaml

This file contains **all model parameters, training hyperparameters, data settings, and pruning options**, allowing the entire system to be controlled from one unified location.

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

Pipelines can override fields such as learning rate, number of epochs, pruning ratios, or the rewinding mode during automated sweeps. This makes experiments fully reproducible while keeping configuration management clean and centralized.

---

# 🧱 Repository Pipelines Overview

This repository consists of **two main pipelines**:

---

# 1️⃣ Baseline Training + Evaluation

Run the full UNet training and a final evaluation:

    python -m src.pipeline.baseline

This pipeline:
- trains the UNet from scratch  
- saves model checkpoints  
- evaluates the final model on the validation set  
- outputs:
  - `metrics.json`
  - training curves (`training_curves.png`)
  - prediction samples

---

# 2️⃣ Pruning + Evaluation + Retraining + Evaluation

Run the structured pruning workflow:

    python -m src.pipeline.pruned --mode <none|random|rewind>

This pipeline performs the full pruning cycle:

---

## **Step 1 — Structured L1 Block-wise Pruning**
- compute L1 norms for each filter  
- rank and prune the lowest-norm filters according to `block_ratios`  
- generate pruning masks  
- apply the masks to remove channels from the UNet  

Example block ratios:
block_ratios:  
  encoders.0: 0.0
  encoders.1: 0.1
  decoders.3: 0.4
  decoders.5: 0.2

---

## **Step 2 — Rebuild the Pruned UNet**
After pruning, the architecture is rebuilt:
- reads pruning masks  
- computes the new (reduced) channel widths  
- constructs a smaller UNet  
- loads the surviving weights correctly  

---

## **Step 3 — Apply Weight Initialization Mode**
The rebuilt model is initialized using the specified rewinding option:

- `none` → keep surviving weights  
- `random` → reinitialize pruned model  
- `rewind` → restore weights from an earlier checkpoint  

---

## **Step 4 — Evaluate the Pruned Model**
Evaluate the pruned model before retraining:

    python -m src.training.eval

---

## **Step 5 — Retrain the Pruned Model**
(Optional depending on pipeline configuration)

Retrains the reduced model to recover performance.

---

## **Step 6 — Final Evaluation**
Evaluate again after retraining to measure pruning impact.

------

# 📊 L1 Analysis & Model Inspection

Inspect L1 statistics and visualize pruning patterns:

    python -m src.pruning.model_inspect

Outputs include:
- L1 histograms  
- layer-wise statistics  
- CSV summaries  

See:

    results/analysis/

Notebooks:
- `l1_distributions.ipynb`
- `pruning_notebook.ipynb`

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

This script executes **both previously described pipelines** (baseline training/evaluation and the full pruning workflow) in a single automated sequence.

It performs:

    1. Train baseline
    2. Prune (mode=none)
    3. Prune (mode=random)
    4. Prune (mode=rewind)
    5. Evaluate all

The runner can include **multiple nested loops** to systematically test different hyperparameters (e.g., learning rates, epochs) and/or compare the effect of various **block ratio configurations**.  
Each experiment automatically overrides the LR, epoch count, pruning ratios, and rewinding mode.

---

# 🧠 Author

**Tijmen Toxopeus**  
Master’s student in Applied Physics (TU Delft)  
Focus: medical image segmentation, structured pruning, efficient deep learning.
