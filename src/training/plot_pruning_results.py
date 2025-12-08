import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def load_pruning_results(root_dir):
    """
    Loads pruning_summary.json and eval_metrics.json from each pruning experiment.
    Returns a flat list of dicts.
    """
    results = []

    pruned_root = os.path.join(root_dir, "pruned")

    for folder in os.listdir(pruned_root):
        full_path = os.path.join(pruned_root, folder)
        if not os.path.isdir(full_path):
            continue

        summary_path = os.path.join(full_path, "pruned_model", "pruning_summary.json")
        eval_path    = os.path.join(full_path, "retrained_pruned_evaluation", "eval_metrics.json")

        if not (os.path.exists(summary_path) and os.path.exists(eval_path)):
            continue

        with open(summary_path, "r") as f:
            summary = json.load(f)

        block_ratios = summary["block_ratios"]
        layer = None
        ratio = 0.0

        for blk, r in block_ratios.items():
            if r > 0:
                layer = blk
                ratio = float(r)
                break

        if layer is None:
            continue

        with open(eval_path, "r") as f:
            eval_data = json.load(f)

        results.append({
            "layer": layer,
            "ratio": ratio,
            "flops": float(eval_data["flops_g"]),
            "inference_ms": float(eval_data["inference_ms"]),
            "vram": float(eval_data["vram_peak_mb"]),
            "dice_fg": float(eval_data["mean_dice_fg"]),
            "iou_fg": float(eval_data["mean_iou_fg"]),
            "folder": folder
        })

    return results


# def plot_layer_sensitivity(results, title="Layer Sensitivity: Dice vs. Pruning Ratio", save_path=None):
#     """
#     results = { layer_name : { ratio : dice } }

#     If save_path is provided, the plot will be saved to that file.
#     """

#     plt.figure(figsize=(12, 8))

#     for layer, vals in results.items():
#         ratios = sorted(vals.keys())
#         dices  = [vals[r] for r in ratios]

#         plt.plot(
#             ratios, dices,
#             marker="o",
#             linewidth=2,
#             label=layer
#         )

#     plt.xlabel("Pruning Ratio", fontsize=14)
#     plt.ylabel("Dice Score", fontsize=14)
#     plt.ylim(0, 1)
#     plt.title(title, fontsize=16)
#     plt.grid(True, alpha=0.3)
#     plt.legend(title="Layer", fontsize=10)
#     plt.tight_layout()

#     # --- SAVE IF PATH IS PROVIDED ---
#     if save_path is not None:
#         plt.savefig(save_path, dpi=300)
#         print(f"💾 Plot saved to: {save_path}")

#     plt.show()


def plot_layer_sensitivity(
        results,
        title="Layer Sensitivity: Dice vs. Pruning Ratio",
        save_path=None
    ):
    """
    results = { layer_name : { ratio : dice } }
    """

    plt.figure(figsize=(14, 8))

    # Use a clean style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Color map for clear distinction
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (layer, vals), color in zip(results.items(), colors):
        ratios = sorted(vals.keys())
        dices  = [vals[r] for r in ratios]

        plt.plot(
            ratios,
            dices,
            marker="o",
            markersize=8,
            linewidth=2.5,
            label=layer,
            color=color
        )

    # Axis labels
    plt.xlabel("Pruning Ratio", fontsize=16, labelpad=10)
    plt.ylabel("Dice Score", fontsize=16, labelpad=10)

    # Title
    plt.title(title, fontsize=18, pad=20)

    # Axes formatting
    plt.ylim(0, 1)
    plt.xlim(min(min(vals.keys()) for vals in results.values()) - 0.02,
             max(max(vals.keys()) for vals in results.values()) + 0.02)

    # Ticks styling
    plt.xticks(sorted({x for vals in results.values() for x in vals.keys()}),
               fontsize=12)
    plt.yticks(np.linspace(0, 1, 11), fontsize=12)

    # Legend outside plot for readability
    plt.legend(
        title="Layer",
        fontsize=12,
        title_fontsize=13,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0
    )

    plt.tight_layout()

    # Save if needed
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"💾 Plot saved to: {save_path}")

    plt.show()

if __name__ == "__main__":
    ROOT = "/mnt/hdd/ttoxopeus/basic_UNet/results/UNet_ACDC/exp48"
    
    raw_results = load_pruning_results(ROOT)

    # Convert list → dict[layer][ratio] = dice
    results = defaultdict(dict)
    for entry in raw_results:
        results[entry["layer"]][entry["ratio"]] = entry["dice_fg"]

    print("Loaded layers:", results.keys())

    save_path = os.path.join(ROOT, "dice_vs_ratio_plot.png")
    plot_layer_sensitivity(results, save_path=save_path)

