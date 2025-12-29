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
            full_path, "retrained_pruned_evaluation", "eval_metrics.json"
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
    ROOT = "/mnt/hdd/ttoxopeus/basic_UNet/results/UNet_ACDC/exp54_stone"

    global_results = load_global_pruning_results(ROOT)

    print(f"Loaded {len(global_results)} global pruning points")

    save_path = os.path.join(ROOT, "global_dice_vs_ratio.png")
    plot_global_pruning_curve(
        global_results,
        save_path=save_path,
    )



# import os
# import json
# import numpy as np
# import matplotlib.pyplot as plt


# def parse_ratios_from_folder(folder, prefix="l1_norm_"):
#     """
#     Example: l1_norm_0_0_0_0_0_25_25_25_25_25_25
#     Returns list of 11 floats (0.0..1.0) if it matches, else None.
#     """
#     if not folder.startswith(prefix):
#         return None

#     parts = folder[len(prefix):].split("_")
#     if len(parts) != 11:
#         return None

#     try:
#         # your naming seems like percentages (e.g. 25 -> 0.25)
#         vals = [float(x) / 100.0 for x in parts]
#         return vals
#     except ValueError:
#         return None


# def classify_from_ratios(vec, eps=1e-12):
#     """
#     vec has length 11 in this fixed order:
#     [enc0 enc1 enc2 enc3 enc4 bottleneck dec1 dec3 dec5 dec7 dec9]

#     Returns: ("encoder"|"middle"|"decoder", ratio) or (None, None)
#     """
#     if vec is None or len(vec) != 11:
#         return None, None

#     active = [i for i, v in enumerate(vec) if v > eps]
#     if not active:
#         return None, None

#     # must be contiguous active region
#     if max(active) - min(active) + 1 != len(active):
#         return None, None

#     # must be uniform ratio on active part
#     active_vals = [vec[i] for i in active]
#     if len({round(v, 8) for v in active_vals}) != 1:
#         return None, None
#     ratio = active_vals[0]

#     # everything outside active must be ~0
#     for i, v in enumerate(vec):
#         if i not in active and v > eps:
#             return None, None

#     # your middle rule: first and last should both be 0
#     first_zero = vec[0] <= eps
#     last_zero = vec[-1] <= eps

#     start, end = min(active), max(active)

#     # encoder-only: starts at index 0
#     if start == 0:
#         return "encoder", ratio

#     # decoder-only: ends at last index
#     if end == len(vec) - 1:
#         return "decoder", ratio

#     # middle: neither start nor end touches boundaries, and your explicit rule holds
#     if first_zero and last_zero:
#         return "middle", ratio

#     return None, None


# def load_results_by_foldername(root_dir):
#     results = {"encoder": [], "middle": [], "decoder": []}
#     pruned_root = os.path.join(root_dir, "pruned")

#     for folder in os.listdir(pruned_root):
#         full_path = os.path.join(pruned_root, folder)
#         if not os.path.isdir(full_path):
#             continue

#         eval_path = os.path.join(full_path, "retrained_pruned_evaluation", "eval_metrics.json")
#         if not os.path.exists(eval_path):
#             continue

#         vec = parse_ratios_from_folder(folder)
#         part, ratio = classify_from_ratios(vec)

#         if part is None:
#             continue

#         with open(eval_path, "r") as f:
#             eval_data = json.load(f)

#         results[part].append({
#             "ratio": float(ratio),
#             "dice_fg": float(eval_data["mean_dice_fg"]),
#             "iou_fg": float(eval_data["mean_iou_fg"]),
#             "flops": float(eval_data["flops_g"]),
#             "inference_ms": float(eval_data["inference_ms"]),
#             "vram": float(eval_data["vram_peak_mb"]),
#             "folder": folder,
#         })

#     return results


# def plot_part(results_list, title, save_path=None):
#     if not results_list:
#         print(f"⚠️ No results for: {title}")
#         return

#     results_list = sorted(results_list, key=lambda x: x["ratio"])
#     ratios = [r["ratio"] for r in results_list]
#     dices = [r["dice_fg"] for r in results_list]

#     plt.figure(figsize=(10, 6))
#     plt.style.use("seaborn-v0_8-whitegrid")
#     plt.plot(ratios, dices, marker="o", markersize=8, linewidth=2.5)

#     plt.xlabel("Pruning Ratio", fontsize=16)
#     plt.ylabel("Dice Score (FG)", fontsize=16)
#     plt.title(title, fontsize=18, pad=15)

#     plt.ylim(0, 1)
#     plt.xticks(ratios, fontsize=12)
#     plt.yticks(np.linspace(0, 1, 11), fontsize=12)

#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=300)
#         print(f"💾 Plot saved to: {save_path}")

#     plt.show()


# if __name__ == "__main__":
#     ROOT = "/mnt/hdd/ttoxopeus/basic_UNet/results/UNet_ACDC/exp53_stone"

#     selective = load_results_by_foldername(ROOT)

#     plot_part(selective["encoder"],
#               "Encoder-only Pruning: Dice vs Pruning Ratio",
#               save_path=os.path.join(ROOT, "encoder_only_dice_vs_ratio.png"))

#     plot_part(selective["middle"],
#               "Middle-chunk Pruning: Dice vs Pruning Ratio",
#               save_path=os.path.join(ROOT, "middle_only_dice_vs_ratio.png"))

#     plot_part(selective["decoder"],
#               "Decoder-only Pruning: Dice vs Pruning Ratio",
#               save_path=os.path.join(ROOT, "decoder_only_dice_vs_ratio.png"))

