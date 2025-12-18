import os
import json
import numpy as np
import matplotlib.pyplot as plt


def load_global_pruning_results(root_dir):
    """
    Loads pruning experiments where ALL blocks have the same pruning ratio.
    Returns a list of dicts.
    """
    results = []

    pruned_root = os.path.join(root_dir, "pruned")

    for folder in os.listdir(pruned_root):
        full_path = os.path.join(pruned_root, folder)
        if not os.path.isdir(full_path):
            continue

        summary_path = os.path.join(
            full_path, "pruned_model", "pruning_summary.json"
        )
        eval_path = os.path.join(
            full_path, "pruned_evaluation", "eval_metrics.json"
        )

        if not (os.path.exists(summary_path) and os.path.exists(eval_path)):
            continue

        with open(summary_path, "r") as f:
            summary = json.load(f)

        block_ratios = summary["block_ratios"]
        ratios = set(float(r) for r in block_ratios.values())

        # Keep only uniform pruning runs
        if len(ratios) != 1:
            continue

        ratio = ratios.pop()

        with open(eval_path, "r") as f:
            eval_data = json.load(f)

        results.append({
            "ratio": ratio,
            "dice_fg": float(eval_data["mean_dice_fg"]),
            "iou_fg": float(eval_data["mean_iou_fg"]),
            "flops": float(eval_data["flops_g"]),
            "inference_ms": float(eval_data["inference_ms"]),
            "vram": float(eval_data["vram_peak_mb"]),
            "folder": folder,
        })

    return results


def plot_global_pruning_curve(
    results,
    title="Global Pruning: Dice vs. Pruning Ratio",
    save_path=None,
):
    """
    results = list of dicts with keys: ratio, dice_fg
    """

    # Sort by pruning ratio
    results = sorted(results, key=lambda x: x["ratio"])

    ratios = [r["ratio"] for r in results]
    dices = [r["dice_fg"] for r in results]

    plt.figure(figsize=(10, 6))
    plt.style.use("seaborn-v0_8-whitegrid")

    plt.plot(
        ratios,
        dices,
        marker="o",
        markersize=8,
        linewidth=2.5,
    )

    plt.xlabel("Pruning Ratio", fontsize=16)
    plt.ylabel("Dice Score", fontsize=16)
    plt.title(title, fontsize=18, pad=15)

    plt.ylim(0, 1)
    plt.xticks(ratios, fontsize=12)
    plt.yticks(np.linspace(0, 1, 11), fontsize=12)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"💾 Plot saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    ROOT = "/mnt/hdd/ttoxopeus/basic_UNet/results/UNet_ACDC/exp50"

    global_results = load_global_pruning_results(ROOT)

    print(f"Loaded {len(global_results)} global pruning points")

    save_path = os.path.join(ROOT, "global_dice_vs_ratio.png")
    plot_global_pruning_curve(
        global_results,
        save_path=save_path,
    )
