import torch
import torch.nn as nn


def compute_block_l1_norms(model: nn.Module, include_transpose=True):
    """
    Compute L1 norms of each filter in all Conv2d (and optionally ConvTranspose2d) layers.
    
    Returns:
        dict: {layer_name: tensor of L1 norms per output channel}
    """
    norms = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or (include_transpose and isinstance(module, nn.ConvTranspose2d)):
            # Flatten each filter and sum absolute values
            weight = module.weight.data.abs()
            l1_per_filter = weight.view(weight.size(0), -1).sum(dim=1)
            norms[name] = l1_per_filter

    return norms


def get_pruning_masks(norms, ratios, default_ratio=0.3, global_prune=False):
    """
    Compute boolean masks for each block based on L1 norms.
    Supports per-block pruning ratios and global thresholding.
    """
    masks = {}

    if global_prune:
        # Combine all norms for a single threshold
        all_l1 = torch.cat(list(norms.values()))
        num_prune = int(default_ratio * len(all_l1))
        global_thresh = torch.topk(all_l1, num_prune, largest=False).values.max()

    for name, l1 in norms.items():
        # Determine block-specific ratio
        ratio = ratios.get(name, default_ratio)
        num_filters = len(l1)
        num_prune = int(ratio * num_filters)

        if global_prune:
            mask = l1 > global_thresh
        else:
            if num_prune == 0:
                mask = torch.ones_like(l1, dtype=torch.bool)
            else:
                threshold = torch.topk(l1, num_prune, largest=False).values.max()
                mask = l1 > threshold

        masks[name] = mask

    return masks

