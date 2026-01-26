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

from toy.datasets import ShapesDatasetConfig, get_synthetic_loaders
from toy.models import build_tiny_unet
from toy.utils import make_run_dir, resolve_device, save_json


def _parse_features(s: str) -> list[int]:
    items = [x.strip() for x in s.split(",") if x.strip()]
    return [int(x) for x in items]


def _num_classes(mode: str) -> int:
    return 2 if mode == "binary" else 3


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
            dice_fn=dice_score,
            iou_fn=iou_score,
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
        dice_fn=dice_score,
        iou_fn=iou_score,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy pruning sanity-check experiment")
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
    parser.add_argument("--prune-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="toy/results")
    parser.add_argument("--bn-recalib", choices=["on", "off"], default="on")
    args = parser.parse_args()

    seed_everything(args.seed, deterministic=False)
    device = resolve_device(args.device)

    cfg = ShapesDatasetConfig(
        num_samples=args.num_samples,
        image_size=args.image_size,
        mode=args.dataset,
        seed=args.seed,
    )

    train_loader, val_loader = get_synthetic_loaders(
        cfg=cfg,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=0,
        deterministic=False,
    )

    out_ch = _num_classes(args.dataset)
    features = _parse_features(args.features)
    model = build_tiny_unet(in_ch=1, out_ch=out_ch, features=features).to(device)

    run_dir = make_run_dir(args.output_dir, f"{args.dataset}_tinyunet")
    print(f"📁 Run dir: {run_dir}")

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
    # Pruning (L1-norm)
    # -----------------------
    pruning_cfg = {
        "pruning": {
            "method": "l1_norm",
            "ratios": {"default": float(args.prune_ratio), "block_ratios": {}},
        }
    }

    pruner = get_method("l1_norm")
    model_cpu = model.to("cpu")
    prune_out = pruner.compute_masks(
        model=model_cpu,
        cfg=pruning_cfg,
        seed=args.seed,
        deterministic=False,
        device=device,
    )

    pruned_ckpt = run_dir / "pruned_model.pth"
    pruned_model = rebuild_pruned_unet(
        model_cpu,
        prune_out.masks,
        save_path=pruned_ckpt,
        seed=args.seed,
        deterministic=False,
    ).to(device)

    bn_batches = 0
    if args.bn_recalib == "on":
        bn_batches = recalibrate_bn(pruned_model, val_loader, device, num_batches=5)
        print(f"🧪 BN recalibration done (used {bn_batches} batches)")

    pruned_eval = eval_model(model=pruned_model, loader=val_loader, device=device, out_ch=out_ch)
    pruned_examples = save_eval_examples(
        model=pruned_model,
        loader=val_loader,
        device=device,
        out_ch=out_ch,
        out_dir=run_dir / "examples" / "pruned",
        max_items=6,
    )
    pstats = compute_param_stats(model, pruned_model)

    # -----------------------
    # Optional retraining
    # -----------------------
    retrain_eval = None
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
            out_dir=run_dir / "examples" / "retrained",
            max_items=6,
        )
        retrained_ckpt = run_dir / "retrained_pruned.pth"
        torch.save(pruned_model.state_dict(), retrained_ckpt)
    else:
        retrain_examples = None

    resize_log = run_dir / "pruned_model_resize_log.json"
    summary = {
        "dataset": {
            "name": args.dataset,
            "num_samples": args.num_samples,
            "image_size": args.image_size,
        },
        "model": {
            "features": features,
            "out_ch": out_ch,
        },
        "train": metrics_log,
        "baseline_eval": baseline_eval,
        "baseline_examples": baseline_examples,
        "prune": {
            "ratio": float(args.prune_ratio),
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
        },
        "artifacts": {
            "baseline_ckpt": str(baseline_ckpt),
            "pruned_ckpt": str(pruned_ckpt),
            "resize_log": str(resize_log) if resize_log.exists() else None,
        },
    }

    save_json(run_dir / "summary.json", summary)
    print("✅ Done. Summary saved.")


if __name__ == "__main__":
    main()
