# import os
# import yaml
# import json
# import torch
# from datetime import datetime

# from src.models.unet import UNet
# from src.pruning.utils import (
#     compute_block_l1_norms,
#     get_pruning_masks,
#     load_prune_config,
#     rebuild_pruned_unet,
# )


# def count_parameters(model):
#     """Return total number of trainable parameters in a model."""
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)


# def main():
#     # --- Load configs ---
#     print("🔧 Loading configuration...")
#     cfg = load_prune_config("configs/prune.yaml")
#     with open("configs/eval.yaml", "r") as f:
#         eval_cfg = yaml.safe_load(f)

#     device = torch.device(eval_cfg["training"]["device"] if torch.cuda.is_available() else "cpu")

#     # --- Setup paths ---
#     checkpoint_path = cfg.get("checkpoint_path", eval_cfg["evaluation"]["checkpoint"])
#     base_save_dir = "/media/ttoxopeus/basic_UNet/results/pruned_models"
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     save_dir = os.path.join(base_save_dir, f"{timestamp}_structured_prune")
#     os.makedirs(save_dir, exist_ok=True)

#     # --- Load model ---
#     print("📦 Loading model checkpoint...")
#     model = UNet(**eval_cfg["model"]).to(device)
#     state = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(state)
#     model.eval()
#     params_before = count_parameters(model)

#     # --- Pruning ---
#     print("✂️ Starting structured pruning...")
#     norms = compute_block_l1_norms(model)
#     masks = get_pruning_masks(
#         norms,
#         ratios=cfg.get("ratios", {}),
#         default_ratio=cfg.get("default_ratio", 0.3),
#         global_prune=cfg.get("global_prune", False),
#     )

#     # --- Apply pruning ---
#     pruned_model = rebuild_pruned_unet(
#         model,
#         masks,
#         save_path=os.path.join(save_dir, "model_pruned_structured.pth"),
#     )

#     # --- Analyze structure ---
#     enc_features = [blk.net[0].out_channels for blk in pruned_model.encoders]
#     bottleneck_features = pruned_model.bottleneck.net[0].out_channels
#     dec_features = [blk.net[0].out_channels for blk in pruned_model.decoders[1::2]]

#     print("\n📊 Feature map summary after pruning:")
#     print(f"  Encoder features:   {enc_features}")
#     print(f"  Bottleneck features: {bottleneck_features}")
#     print(f"  Decoder features:   {dec_features}")
#     print("──────────────────────────────────────────────")

#     # --- Stats ---
#     params_after = count_parameters(pruned_model)
#     compression_ratio = params_after / params_before
#     total_pruned_filters = sum((len(m) - m.sum().item()) for m in masks.values())
#     total_filters = sum(len(m) for m in masks.values())
#     total_pruned_pct = 100 * total_pruned_filters / total_filters

#     # --- Save summary in correct fine-tuning format ---
#     summary = {
#         "checkpoint_pruned_from": checkpoint_path,
#         "save_dir": save_dir,
#         "model_architecture": {
#             "in_channels": eval_cfg["model"]["in_ch"],
#             "out_channels": eval_cfg["model"]["out_ch"],
#             "encoder_features": enc_features,
#             "bottleneck_features": bottleneck_features,
#             "decoder_features": dec_features
#         },
#         "compression": {
#             "parameters_before": params_before,
#             "parameters_after": params_after,
#             "compression_ratio": round(compression_ratio, 4),
#             "total_filters": int(total_filters),
#             "total_pruned_filters": int(total_pruned_filters),
#             "total_pruned_percentage": round(total_pruned_pct, 2)
#         },
#         "timestamp": timestamp
#     }

#     summary_path = os.path.join(save_dir, "pruning_summary.json")
#     with open(summary_path, "w") as f:
#         json.dump(summary, f, indent=4)

#     # --- Final concise report ---
#     print("\n✅ Structured pruning complete!")
#     print(f"   • Parameters before: {params_before:,}")
#     print(f"   • Parameters after:  {params_after:,}")
#     print(f"   • Compression ratio: {compression_ratio:.3f}")
#     print(f"   • Filters pruned:    {total_pruned_pct:.1f}%")
#     print(f"   • Saved to:          {save_dir}")
#     print(f"   • Summary JSON:      {summary_path}")


# if __name__ == "__main__":
#     main()


# 1. Load model + checkpoint
# 2. Compute filter importance metrics (e.g. L₁ norm)
# 3. Decide which filters to prune (mask generation)
# 4. Apply pruning → rebuild model
# 5. Analyze structure and parameter count
# 6. Save pruned model + JSON summary + visualizations


# def prune_model(cfg_path: str):
#     """
#     High-level pruning pipeline:
#     1. Load config, model, and checkpoint
#     2. Compute L1 norms per filter
#     3. Generate pruning masks
#     4. Rebuild pruned model
#     5. Save model + summary JSON
#     """
#     pass


import os, json, torch, yaml
from datetime import datetime
from src.models.unet import UNet
from src.pruning.utils import compute_block_l1_norms, get_pruning_masks, count_parameters
from src.pruning.rebuild import rebuild_pruned_unet
from src.pruning.model_inspect import print_model_summary

def prune_model(cfg_path="configs/prune.yaml"):
    # 1. Load config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    checkpoint_path = cfg["checkpoint"]
    prune_ratio = cfg.get("prune_ratio", 0.3)
    save_root = cfg.get("save_root", "results/pruned_models")
    os.makedirs(save_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load model + checkpoint
    model = UNet(**cfg["model"]).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print_model_summary(model)
    params_before = count_parameters(model)

    # 3. Compute filter importance
    norms = compute_block_l1_norms(model)

    # 4. Generate pruning masks
    masks = get_pruning_masks(norms, prune_ratio=prune_ratio)

    # 5. Rebuild pruned model
    pruned_model = rebuild_pruned_unet(model, masks)
    print_model_summary(pruned_model)
    params_after = count_parameters(pruned_model)

    # 6. Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(save_root, f"{timestamp}_structured")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(pruned_model.state_dict(), os.path.join(save_dir, "pruned_model.pth"))

    summary = {
        "checkpoint": checkpoint_path,
        "ratio": prune_ratio,
        "params_before": params_before,
        "params_after": params_after,
        "compression_ratio": round(params_after / params_before, 3),
    }
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("✅ Pruning complete.")
    print(f"Compression: {summary['compression_ratio']*100:.1f}% params kept")

if __name__ == "__main__":
    prune_model()
