import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------------
# 1. Collect feature activations
# -------------------------------------------------------------
def get_feature_maps(model, x):
    activations = {}

    def hook(name):
        def fn(m, inp, out):
            activations[name] = out.detach().cpu()
        return fn

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(hook(name)))

    model.eval()
    with torch.no_grad():
        _ = model(x)

    for h in handles:
        h.remove()

    return activations


# -------------------------------------------------------------
# 2. Compute correlation matrix per layer
# -------------------------------------------------------------
def compute_similarity_matrix(fmap):
    fmap = fmap[0]  # [C, H, W]
    C = fmap.size(0)

    sim = np.zeros((C, C))
    flat = fmap.reshape(C, -1).cpu().numpy()

    for i in range(C):
        for j in range(i, C):
            if i == j:
                sim[i, j] = 1.0
            else:
                val = np.corrcoef(flat[i], flat[j])[0, 1]
                sim[i, j] = sim[j, i] = val

    return sim


# -------------------------------------------------------------
# 3. Find redundant channel pairs
# -------------------------------------------------------------
def find_redundant_pairs(sim_matrix, threshold):
    C = sim_matrix.shape[0]
    pairs = []

    for i in range(C):
        for j in range(i + 1, C):
            if sim_matrix[i, j] > threshold:
                pairs.append((i, j))

    return pairs


# -------------------------------------------------------------
# 4. Group channels into clusters (union-find logic)
# -------------------------------------------------------------
def group_channels(redundant_pairs):
    groups = {}
    gid = 0

    for (i, j) in redundant_pairs:
        found = False

        for g, members in groups.items():
            if i in members or j in members:
                members.add(i)
                members.add(j)
                found = True
                break

        if not found:
            groups[gid] = {i, j}
            gid += 1

    return groups


# -------------------------------------------------------------
# 5. Compute pruning mask for one layer
# -------------------------------------------------------------
def compute_layer_mask(C, groups, sim_matrix, prune_ratio):
    """
    Improved pruning logic:
    - For every redundant pair (i, j), compute redundancy score from similarity.
    - Aggregate scores for each channel j.
    - Select top-k channels globally to prune.
    """

    keep_mask = torch.ones(C, dtype=torch.bool)

    # Step 1: collect candidate channels and redundancy scores
    scores = {}  # ch_idx → cumulative similarity score

    for _, chs in groups.items():
        chs = sorted(chs)
        rep = chs[0]            # group leader is kept
        others = chs[1:]        # redundant candidates

        for j in others:
            # Compute similarity to representative
            sim_value = sim_matrix[rep, j]

            # Accumulate score
            scores[j] = scores.get(j, 0) + sim_value

    if len(scores) == 0:
        return keep_mask, []   # Nothing to prune

    # Step 2: sort channels by redundancy score (highest first)
    sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    candidate_idxs = [idx for idx, score in sorted_candidates]

    # Step 3: prune top fraction
    max_prune = int(C * prune_ratio)
    to_prune = candidate_idxs[:max_prune]

    # Step 4: apply pruning
    for idx in to_prune:
        keep_mask[idx] = False

    return keep_mask, to_prune


# -------------------------------------------------------------
# 6. Plot redundant groups
# -------------------------------------------------------------
def plot_redundancy_groups(fmap, groups):
    fmap = fmap[0]

    for gid, chans in groups.items():
        chans = sorted(chans)
        n = len(chans)

        fig, axes = plt.subplots(1, n, figsize=(1.2 * n, 1.4), dpi=130)
        if n == 1:
            axes = [axes]

        for ax, ch in zip(axes, chans):
            img = fmap[ch]
            img = (img - img.min()) / (img.max() - img.min() + 1e-5)
            ax.imshow(img, cmap="gray")
            ax.set_title(f"ch {ch}", fontsize=8)
            ax.axis("off")

        fig.suptitle(f"Group {gid}", fontsize=12, y=1.08)
        plt.tight_layout(pad=0.3)
        plt.show()


def get_ratio_for_layer(layer_name, block_ratios):
    """Return prune ratio by matching prefixes like 'encoders.0'."""
    for block, ratio in block_ratios.items():
        if layer_name.startswith(block):
            return ratio
    return 0.0


# -------------------------------------------------------------
# 7. FULL PIPELINE-INTEGRATED REDUNDANCY MASKING
# -------------------------------------------------------------
def get_redundancy_masks(model, x, block_ratios, threshold=0.9):
    activations = get_feature_maps(model, x)
    masks = {}

    total_channels_before = 0
    total_channels_after = 0

    for layer_name, fmap in activations.items():

        prune_ratio = get_ratio_for_layer(layer_name, block_ratios)

        if prune_ratio == 0.0:
            continue

        C = fmap.shape[1]
        total_channels_before += C

        print(f"\n=== Processing Layer: {layer_name} (ratio={prune_ratio}) ===")

        # 1. Similarity matrix
        sim = compute_similarity_matrix(fmap)

        # 2. Redundant pairs
        pairs = find_redundant_pairs(sim, threshold)

        # 3. Group channels
        groups = group_channels(pairs)

        # 4. Visualization
        if len(groups) > 0:
            plot_redundancy_groups(fmap, groups)
        else:
            print("  No redundancy groups detected.")

        # 5. Build mask + pruned channel indexes
        layer_mask, pruned_ix = compute_layer_mask(C, groups, sim, prune_ratio)

        kept = layer_mask.sum().item()
        pruned = C - kept
        perc_pruned = 100 * pruned / C
        total_channels_after += kept

        # PRINT SUMMARY FOR THIS LAYER
        print(f"Layer {layer_name}:")
        print(f"  - Total channels  : {C}")
        print(f"  - Channels kept   : {kept}")
        print(f"  - Channels pruned : {pruned}")
        print(f"  - Pruned percentage: {perc_pruned:.2f}%")
        print(f"  - Pruned indexes  : {pruned_ix}")

        # 6. Save weight + bias masks
        masks[f"{layer_name}.weight"] = layer_mask.clone()
        masks[f"{layer_name}.bias"] = layer_mask.clone()

    # ------------------------------
    # FINAL SUMMARY
    # ------------------------------
    print("\n================ PRUNING SUMMARY ================")
    print(f"Total channels before pruning: {total_channels_before}")
    print(f"Total channels after pruning:  {total_channels_after}")

    if total_channels_before > 0:
        total_pruned = total_channels_before - total_channels_after
        total_pruned_pct = 100 * total_pruned / total_channels_before
        print(f"Total pruned: {total_pruned} channels")
        print(f"Overall pruning percentage: {total_pruned_pct:.2f}%")

    print("=================================================\n")

    return masks

