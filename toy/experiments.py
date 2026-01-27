from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import torch

from src.pruning.artifacts import compute_param_stats
from src.pruning.methods import get_method
from src.pruning.rebuild import rebuild_pruned_unet
from src.training.logging import log_epoch
from src.training.loss import get_loss_function
from src.training.metrics import dice_score, iou_score
from src.training.training_loop import train_one_epoch, validate
from src.utils.reproducibility import seed_everything

from toy.datasets import (
    ShapesDatasetConfig,
    MNISTConfig,
    FashionMNISTConfig,
    get_synthetic_loaders,
    get_mnist_loaders,
    get_fashion_mnist_loaders,
)
from toy.models import build_tiny_unet
from toy.utils import make_run_dir, resolve_device, save_json


def _parse_features(s: str) -> list[int]:
    items = [x.strip() for x in s.split(",") if x.strip()]
    return [int(x) for x in items]


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _num_classes(mode: str, fg_classes: int | None = None) -> int:
    return 2 if mode == "binary" else 1 + (fg_classes if fg_classes is not None else 2)


# Foreground-only metrics (exclude background class 0)
def fg_dice_score(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    # Accept logits [N,C,H,W] or label maps [N,H,W]
    if preds.dim() == 4:
        preds = preds.argmax(dim=1)
    assert preds.dim() == 3 and targets.dim() == 3, "Expect [N,H,W] label maps"
    N = preds.shape[0]
    dice_vals = []
    eps = 1e-6
    for c in range(1, num_classes):
        pred_c = (preds == c).float().view(N, -1)
        targ_c = (targets == c).float().view(N, -1)
        inter = (pred_c * targ_c).sum(dim=1)
        denom = pred_c.sum(dim=1) + targ_c.sum(dim=1)
        dice_c = (2.0 * inter + eps) / (denom + eps)
        dice_vals.append(dice_c)
    if not dice_vals:
        return 1.0
    dice_stack = torch.stack(dice_vals, dim=0).mean(dim=0)  # mean over classes, per-sample
    return float(dice_stack.mean().item())                   # mean over batch

def fg_iou_score(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    if preds.dim() == 4:
        preds = preds.argmax(dim=1)
    assert preds.dim() == 3 and targets.dim() == 3, "Expect [N,H,W] label maps"
    N = preds.shape[0]
    iou_vals = []
    eps = 1e-6
    for c in range(1, num_classes):
        pred_c = (preds == c).float().view(N, -1)
        targ_c = (targets == c).float().view(N, -1)
        inter = (pred_c * targ_c).sum(dim=1)
        union = pred_c.sum(dim=1) + targ_c.sum(dim=1) - inter
        iou_c = (inter + eps) / (union + eps)
        iou_vals.append(iou_c)
    if not iou_vals:
        return 1.0
    iou_stack = torch.stack(iou_vals, dim=0).mean(dim=0)  # mean over classes, per-sample
    return float(iou_stack.mean().item())                  # mean over batch


def train_baseline(
    *,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    loss_name: str,
    out_ch: int,
):
    criterion = get_loss_function(loss_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    metrics_log = {
        "epoch": [],
        "train_loss_mean": [],
        "train_loss_std": [],
        "val_dice_mean": [],
        "val_dice_std": [],
        "val_iou_mean": [],
        "val_iou_std": [],
        "lr": [],
        "vram_max": [],
    }

    for epoch in range(epochs):
        train_res = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch + 1,
            epochs=epochs,
            log_interval=50,
            track_vram=True,
        )
        val_res = validate(
            model=model,
            loader=val_loader,
            device=device,
            out_ch=out_ch,
            dice_fn=fg_dice_score,   # foreground-only
            iou_fn=fg_iou_score,     # foreground-only
        )

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss_mean": train_res.loss_mean,
            "train_loss_std": train_res.loss_std,
            "val_dice_mean": val_res.dice_mean,
            "val_dice_std": val_res.dice_std,
            "val_iou_mean": val_res.iou_mean,
            "val_iou_std": val_res.iou_std,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "vram_max": train_res.vram_epoch_max_mb,
        }
        log_epoch(metrics_log, epoch_metrics)

        print(
            f"Epoch {epoch+1:02d} | "
            f"loss={epoch_metrics['train_loss_mean']:.4f} | "
            f"dice={epoch_metrics['val_dice_mean']:.4f} | "
            f"iou={epoch_metrics['val_iou_mean']:.4f}"
        )

    return metrics_log


def eval_model(*, model, loader, device, out_ch) -> dict:
    res = validate(
        model=model,
        loader=loader,
        device=device,
        out_ch=out_ch,
        dice_fn=fg_dice_score,   # foreground-only
        iou_fn=fg_iou_score,     # foreground-only
    )
    return {
        "dice_mean": res.dice_mean,
        "dice_std": res.dice_std,
        "iou_mean": res.iou_mean,
        "iou_std": res.iou_std,
    }


@torch.no_grad()
def recalibrate_bn(model, loader, device, *, num_batches: int = 5) -> int:
    model.train()
    seen = 0
    for imgs, _masks in loader:
        imgs = imgs.to(device)
        _ = model(imgs)
        seen += 1
        if seen >= num_batches:
            break
    model.eval()
    return seen


@torch.no_grad()
def save_eval_examples(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    out_ch: int,
    out_dir: Path,
    max_items: int = 6,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    saved = []
    seen = 0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device, dtype=torch.long)
        logits = model(imgs)
        preds = torch.argmax(logits, dim=1)

        b = imgs.shape[0]
        for i in range(b):
            if seen >= max_items:
                return saved

            img = imgs[i, 0].detach().cpu().numpy()
            gt = masks[i].detach().cpu().numpy()
            pr = preds[i].detach().cpu().numpy()

            img = np.clip(img, 0.0, 1.0)

            fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5))
            axes[0].imshow(img, cmap="gray")
            axes[0].set_title("input")
            axes[1].imshow(gt, vmin=0, vmax=out_ch - 1, cmap="viridis")
            axes[1].set_title("ground_truth")
            axes[2].imshow(pr, vmin=0, vmax=out_ch - 1, cmap="viridis")
            axes[2].set_title("prediction")
            for ax in axes:
                ax.axis("off")

            fname = out_dir / f"sample_{seen:02d}.png"
            fig.tight_layout(pad=0.2)
            fig.savefig(fname, dpi=120)
            plt.close(fig)

            saved.append(str(fname))
            seen += 1

    return saved


def _plot_pruning_curves(pruning_runs, baseline_eval, baseline_params, out_dir: Path):
    if not pruning_runs or baseline_eval is None or baseline_params is None:
        return
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    ratios = [r["ratio"] for r in pruning_runs]
    dice = [r["pruned_eval"]["dice_mean"] for r in pruning_runs]
    size_pct = [100.0 - r["params"]["reduction_percent"] for r in pruning_runs]

    # include baseline
    ratios_with_base = [0.0] + ratios
    dice_with_base = [baseline_eval["dice_mean"]] + dice
    size_with_base = [100.0] + size_pct

    # Pruned curves
    plt.figure()
    plt.plot(size_with_base, dice_with_base, marker="o")
    plt.xlabel("Model size (%)")
    plt.ylabel("Dice")
    plt.title("Model size vs Dice (pruned)")
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "model_size_vs_dice.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(ratios_with_base, dice_with_base, marker="o")
    plt.xlabel("Prune ratio")
    plt.ylabel("Dice")
    plt.title("Prune ratio vs Dice (pruned)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "prune_ratio_vs_dice.png", dpi=150)
    plt.close()

    # Retrained curves (only if available)
    retr_points = [
        (r["ratio"], 100.0 - r["params"]["reduction_percent"], r["retrained_eval"]["dice_mean"])
        for r in pruning_runs
        if r.get("retrained_eval") is not None
    ]
    if retr_points:
        ret_ratios = [p[0] for p in retr_points]
        ret_sizes = [p[1] for p in retr_points]
        ret_dice = [p[2] for p in retr_points]

        # include baseline
        ret_ratios_with_base = [0.0] + ret_ratios
        ret_dice_with_base = [baseline_eval["dice_mean"]] + ret_dice
        ret_sizes_with_base = [100.0] + ret_sizes

        plt.figure()
        plt.plot(ret_sizes_with_base, ret_dice_with_base, marker="o")
        plt.xlabel("Model size (%)")
        plt.ylabel("Dice")
        plt.title("Model size vs Dice (retrained)")
        plt.gca().invert_xaxis()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "model_size_vs_dice_retrained.png", dpi=150)
        plt.close()

        plt.figure()
        plt.plot(ret_ratios_with_base, ret_dice_with_base, marker="o")
        plt.xlabel("Prune ratio")
        plt.ylabel("Dice")
        plt.title("Prune ratio vs Dice (retrained)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "prune_ratio_vs_dice_retrained.png", dpi=150)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy pruning sanity-check experiment")
    parser.add_argument("--dataset-type", default="synthetic", choices=["synthetic", "mnist", "fashion_mnist"])
    parser.add_argument("--dataset", default="binary", choices=["binary", "multiclass"])
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--retrain-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", default="ce_dice")
    parser.add_argument("--features", default="8,16,32")
    parser.add_argument("--prune-ratios", default="0.3", help="Comma-separated ratios, e.g. 0.1,0.3,0.5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="toy/results")
    parser.add_argument("--bn-recalib", choices=["on", "off"], default="on")
    parser.add_argument("--data-root", default="./data", help="Root dir for MNIST download")
    parser.add_argument("--fg-classes", type=int, default=2, help="Foreground classes for synthetic multiclass")
    args = parser.parse_args()

    seed_everything(args.seed, deterministic=False)
    device = resolve_device(args.device)

    # Load dataset
    if args.dataset_type == "synthetic":
        cfg = ShapesDatasetConfig(
            num_samples=args.num_samples,
            image_size=args.image_size,
            mode=args.dataset,
            seed=args.seed,
            fg_classes=args.fg_classes,
        )
        train_loader, val_loader = get_synthetic_loaders(
            cfg=cfg,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            num_workers=0,
            deterministic=False,
        )
        out_ch = _num_classes(args.dataset, args.fg_classes)
        run_name = f"{args.dataset}_tinyunet"
    elif args.dataset_type == "mnist":
        mnist_cfg = MNISTConfig(root=args.data_root, seed=args.seed)
        train_loader, val_loader = get_mnist_loaders(
            cfg=mnist_cfg,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            num_workers=0,
            deterministic=False,
            mask_threshold=0.1,
        )
        out_ch = 2  # binary: background vs digit
        run_name = "mnist_tinyunet"
    else:  # fashion_mnist
        fmnist_cfg = FashionMNISTConfig(root=args.data_root, seed=args.seed)
        train_loader, val_loader = get_fashion_mnist_loaders(
            cfg=fmnist_cfg,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            num_workers=0,
            deterministic=False,
            mask_threshold=0.1,
        )
        out_ch = 2
        run_name = "fashion_mnist_tinyunet"

    features = _parse_features(args.features)
    prune_ratios = _parse_float_list(args.prune_ratios)
    model = build_tiny_unet(in_ch=1, out_ch=out_ch, features=features).to(device)

    run_dir = make_run_dir(args.output_dir, run_name)
    print(f"📁 Run dir: {run_dir}")
    print(f"📊 Dataset: {args.dataset_type}, output channels: {out_ch}")

    # -----------------------
    # Baseline training
    # -----------------------
    metrics_log = train_baseline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        loss_name=args.loss,
        out_ch=out_ch,
    )

    baseline_ckpt = run_dir / "baseline.pth"
    torch.save(model.state_dict(), baseline_ckpt)
    baseline_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    baseline_eval = eval_model(model=model, loader=val_loader, device=device, out_ch=out_ch)
    baseline_examples = save_eval_examples(
        model=model,
        loader=val_loader,
        device=device,
        out_ch=out_ch,
        out_dir=run_dir / "examples" / "baseline",
        max_items=6,
    )

    # -----------------------
    # Pruning loop (L1-norm)
    # -----------------------
    pruning_runs = []
    pruner = get_method("l1_norm")
    baseline_params = None

    for ratio in prune_ratios:
        ratio_slug = str(ratio).replace(".", "_")
        pruning_cfg = {
            "pruning": {
                "method": "l1_norm",
                "ratios": {"default": float(ratio), "block_ratios": {}},
            }
        }

        model_cpu = build_tiny_unet(in_ch=1, out_ch=out_ch, features=features).to("cpu")
        model_cpu.load_state_dict(baseline_state)

        prune_out = pruner.compute_masks(
            model=model_cpu,
            cfg=pruning_cfg,
            seed=args.seed,
            deterministic=False,
            device="cpu",  # model_cpu is on CPU, so use CPU for mask computation
        )

        # Debug: print which channels are actually pruned
        print(f"\n[DEBUG] Pruned channels for ratio {ratio}:")
        for lname, mask in prune_out.masks.items():
            pruned_ch = [i for i, keep in enumerate(mask) if not keep]
            if pruned_ch:
                print(f"  {lname}: pruned {pruned_ch}")

        pruned_ckpt = run_dir / f"pruned_model_r{ratio_slug}.pth"
        pruned_model = rebuild_pruned_unet(
            model_cpu,
            prune_out.masks,
            save_path=None,  # don't let rebuild save anything
            seed=args.seed,
            deterministic=False,
        )

        # Move to device and save FULL model (not state_dict)
        pruned_model = pruned_model.to(device)
        torch.save(pruned_model, pruned_ckpt)

        bn_batches = 0
        if args.bn_recalib == "on":
            bn_batches = recalibrate_bn(pruned_model, val_loader, device, num_batches=5)
            print(f"🧪 BN recalibration done (used {bn_batches} batches) for ratio {ratio}")

        pruned_eval = eval_model(model=pruned_model, loader=val_loader, device=device, out_ch=out_ch)
        pruned_examples = save_eval_examples(
            model=pruned_model,
            loader=val_loader,
            device=device,
            out_ch=out_ch,
            out_dir=run_dir / "examples" / f"pruned_r{ratio_slug}",
            max_items=6,
        )
        pstats = compute_param_stats(model, pruned_model)
        if baseline_params is None:
            baseline_params = pstats.original_params

        retrain_eval = None
        retrain_examples = None
        retrained_ckpt = None
        if args.retrain_epochs > 0:
            _ = train_baseline(
                model=pruned_model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=args.retrain_epochs,
                lr=args.lr,
                loss_name=args.loss,
                out_ch=out_ch,
            )
            retrain_eval = eval_model(model=pruned_model, loader=val_loader, device=device, out_ch=out_ch)
            retrain_examples = save_eval_examples(
                model=pruned_model,
                loader=val_loader,
                device=device,
                out_ch=out_ch,
                out_dir=run_dir / "examples" / f"retrained_r{ratio_slug}",
                max_items=6,
            )
            retrained_ckpt = run_dir / f"retrained_pruned_r{ratio_slug}.pth"
            torch.save(pruned_model, retrained_ckpt)  # full model, not state_dict

        pruning_runs.append({
            "ratio": float(ratio),
            "params": {
                "original": pstats.original_params,
                "pruned": pstats.pruned_params,
                "reduction_percent": pstats.reduction_percent,
            },
            "pruned_eval": pruned_eval,
            "pruned_examples": pruned_examples,
            "retrained_eval": retrain_eval,
            "retrained_examples": retrain_examples,
            "bn_recalibration_batches": bn_batches,
            "artifacts": {
                "pruned_ckpt": str(pruned_ckpt),
                "retrained_ckpt": str(retrained_ckpt) if retrained_ckpt else None,
            },
        })

    # -----------------------
    # Summary
    # -----------------------
    summary = {
        "dataset": {
            "name": args.dataset,
            "num_samples": args.num_samples,
            "image_size": args.image_size,
        },
        "model": {"features": features, "out_ch": out_ch},
        "train": metrics_log,
        "baseline_eval": baseline_eval,
        "baseline_examples": baseline_examples,
        "prune_runs": pruning_runs,
        "artifacts": {"baseline_ckpt": str(baseline_ckpt)},
    }

    save_json(run_dir / "summary.json", summary)
    _plot_pruning_curves(pruning_runs, baseline_eval, baseline_params, run_dir)
    print("✅ Done. Summary saved.")


if __name__ == "__main__":
    main()


### EXAMPLE COMMANDS

# python -m toy.experiments   --dataset-type synthetic --dataset multiclass   --epochs 150   --retrain-epochs 20   --features 2,4,8   --batch-size 64  --lr 0.01   --prune-ratio 0.05,0.
# 1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95  --fg-classes 3