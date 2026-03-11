"""Plot helpers for first-layer feature-map pruning visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import numpy as np
import torch


def plot_first_layer_grid(
    feature_maps,
    l1_keep,
    fm_keep,
    l1_norms=None,
    num_cols=8,
    normalize=True,
    layer_name="encoders.0.net.0",
    show_annotations=True,
):
    fmap = feature_maps[0].detach().cpu()  # [C, H, W]
    C_total = fmap.shape[0]

    # Force a true 8x8 grid (64 tiles).
    grid_size = 8
    max_tiles = grid_size * grid_size
    C_show = min(C_total, max_tiles)

    if l1_norms is not None:
        l1_norms = l1_norms.detach().cpu().float()
        if show_annotations and l1_norms.numel() < C_show:
            raise ValueError(f'l1_norms has {l1_norms.numel()} entries but need at least {C_show}')

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2.05, grid_size * 2.05), dpi=140)
    axes = np.atleast_1d(axes).reshape(grid_size, grid_size)

    for i, ax in enumerate(axes.flatten()):
        ax.axis('off')
        if i >= C_show:
            continue

        img = fmap[i]
        if normalize:
            mn = float(img.min())
            mx = float(img.max())
            if mx > mn:
                img = (img - mn) / (mx - mn)

        ax.imshow(img, cmap='copper', interpolation='nearest')

        if show_annotations:
            # Channel index
            ax.text(
                0.02, 0.02, f'ch {i}',
                color='white', fontsize=12, fontweight='bold',
                ha='left', va='bottom', transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.20', facecolor='black', edgecolor='white', linewidth=1.0, alpha=0.80),
                zorder=30,
            )

        if show_annotations and l1_norms is not None:
            ax.text(
                0.97, 0.97, f'L1 {float(l1_norms[i]):.2f}',
                color='yellow', fontsize=12, fontweight='bold',
                ha='right', va='top', transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.14', facecolor='black', edgecolor='yellow', linewidth=0.8, alpha=0.80),
                zorder=31,
            )

        l1_pruned = bool(~l1_keep[i])
        fm_pruned = bool(~fm_keep[i])

        # Add a light tint to make pruning membership easier to read at a glance.
        if l1_pruned and fm_pruned:
            ax.add_patch(
                patches.Rectangle(
                    (0, 0), 1, 1,
                    fill=True, facecolor='magenta', alpha=0.18,
                    linewidth=0, transform=ax.transAxes, zorder=2
                )
            )
        elif l1_pruned:
            ax.add_patch(
                patches.Rectangle(
                    (0, 0), 1, 1,
                    fill=True, facecolor='red', alpha=0.35,
                    linewidth=0, transform=ax.transAxes, zorder=2
                )
            )
        elif fm_pruned:
            ax.add_patch(
                patches.Rectangle(
                    (0, 0), 1, 1,
                    fill=True, facecolor='royalblue', alpha=0.45,
                    linewidth=0, transform=ax.transAxes, zorder=2
                )
            )

        if l1_pruned and fm_pruned:
            ax.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='red', linewidth=6.0, transform=ax.transAxes, zorder=10))
            ax.add_patch(patches.Rectangle((0.060, 0.060), 0.88, 0.88, fill=False, edgecolor='royalblue', linewidth=6.0, transform=ax.transAxes, zorder=11))
        elif l1_pruned:
            ax.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='red', linewidth=6.0, transform=ax.transAxes, zorder=10))
        elif fm_pruned:
            ax.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='royalblue', linewidth=6.0, transform=ax.transAxes, zorder=10))

    legend_handles = [
        patches.Patch(facecolor='none', edgecolor='red', linewidth=6, label='L1'),
        patches.Patch(facecolor='none', edgecolor='royalblue', linewidth=6, label='Pearson'),
    ]

    fig.suptitle(f'First-layer feature maps with pruning overlays ({layer_name})', fontsize=26, y=0.995)
    fig.legend(handles=legend_handles, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.03), prop={'size': 18})
    plt.subplots_adjust(top=0.91, bottom=0.08, wspace=0.05, hspace=0.08)
    plt.show()

    if C_total > max_tiles:
        print(f'Note: showing first {max_tiles} of {C_total} channels.')
