# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# from basic_UNet.model import UNet
# from basic_UNet.pruning_utils import compute_block_l1_norms

# # ---------- Config ----------
# MODEL_PATH = "/media/ttoxopeus/basic_UNet/results/basicUNet_best.pth"
# SAVE_FIG = True
# SAVE_PATH = "/media/ttoxopeus/basic_UNet/results/pruning_l1norms_combined.png"

# # ---------- Load model ----------
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = UNet(in_ch=1, out_ch=4).to(device)
# assert os.path.exists(MODEL_PATH), f"❌ Model not found: {MODEL_PATH}"
# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# model.eval()
# print(f"✅ Loaded model from {MODEL_PATH}")

# # ---------- Compute block-wise L1 norms ----------
# norms = compute_block_l1_norms(model)

# # Collect mean L1 per block and all filter L1 values
# block_names = []
# mean_norms = []
# all_filter_norms = []

# for name, l1 in norms.items():
#     block_names.append(name)
#     mean_norms.append(l1.mean().item())
#     all_filter_norms.extend(l1.cpu().numpy().tolist())

# # Sort by block name for nicer plotting
# block_names, mean_norms = zip(*sorted(zip(block_names, mean_norms)))

# # ---------- Plot ----------
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# # --- (a) Block-wise mean L1 norm ---
# axes[0].barh(block_names, mean_norms, color='skyblue')
# axes[0].set_xlabel("Mean L1 Norm per Filter")
# axes[0].set_title("Block-wise Filter Strengths (L1 Norms)")
# axes[0].grid(axis='x', linestyle='--', alpha=0.6)

# # --- (b) Global histogram ---
# all_filter_norms = np.array(all_filter_norms)
# axes[1].hist(all_filter_norms, bins=40, color='lightcoral', edgecolor='k', alpha=0.8)
# axes[1].set_xlabel("L1 Norm Value (per Filter)")
# axes[1].set_ylabel("Number of Filters")
# axes[1].set_title("Global Distribution of Filter L1 Norms")
# axes[1].grid(axis='y', linestyle='--', alpha=0.6)

# # Optional: draw red line at 30% pruning threshold
# if len(all_filter_norms) > 0:
#     sorted_norms = np.sort(all_filter_norms)
#     thresh_idx = int(0.3 * len(sorted_norms))
#     threshold = sorted_norms[thresh_idx]
#     axes[1].axvline(threshold, color='red', linestyle='--', label=f"30% threshold = {threshold:.2f}")
#     axes[1].legend()

# plt.tight_layout()

# # ---------- Save ----------
# if SAVE_FIG:
#     os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
#     plt.savefig(SAVE_PATH, dpi=150)
#     print(f"📊 Figure saved to {SAVE_PATH}")

# plt.show()


def plot_l1_distribution(norms, save_path=None):
    """Plot histograms of L1 norms per layer."""

def plot_pruning_summary(summary_json_path):
    """Visualize compression ratios over multiple experiments."""
