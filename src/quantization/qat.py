# src/quantization/qat.py
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.ao.quantization.quantize_fx import convert_fx, fuse_fx, prepare_qat_fx

from src.training.eval_loop import run_evaluation
from src.training.metrics import dice_score, iou_score
from src.utils.config import load_config
from src.utils.reproducibility import seed_everything
from src.utils.run_summary import base_run_info, write_json

from .common import (
    build_loader,
    build_qconfig_mapping,
    count_state_dict_bytes,
    extract_inputs_targets,
    latency_ms,
    load_target_model,
    make_example_inputs,
    make_unet_fx_traceable,
)


def _freeze_bn_stats(m: nn.Module) -> None:
    if isinstance(m, nn.BatchNorm2d):
        # Stop running_mean/var updates, but keep affine params trainable.
        m.eval()
        if m.weight is not None:
            m.weight.requires_grad_(True)
        if m.bias is not None:
            m.bias.requires_grad_(True)


def _train_qat(
    model: nn.Module,
    train_loader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float | None,
    bn_freeze_after: int,
    log_every: int,
) -> None:
    """
    Minimal, self-contained QAT fine-tune loop.
    Assumes segmentation logits [B,C,H,W] and targets [B,H,W] long.
    """
    model.to(device).train()

    # If you use a different loss in your project (Dice+CE etc.),
    # swap this for your own loss function.
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    global_step = 0
    for ep in range(epochs):
        if bn_freeze_after >= 0 and ep >= bn_freeze_after:
            model.apply(_freeze_bn_stats)

        running = 0.0
        seen = 0

        for batch in train_loader:
            x, y = extract_inputs_targets(batch)
            if y is None:
                raise RuntimeError(
                    "QAT training requires labels from the loader. "
                    "Your loader seems to not return targets."
                )

            x = x.to(device, non_blocking=True)
            y = y.to(device, dtype=torch.long, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))

            opt.step()

            running += float(loss.detach().cpu())
            seen += 1
            global_step += 1

            if log_every > 0 and (global_step % log_every == 0):
                avg = running / max(1, seen)
                print(f"[QAT] epoch {ep+1}/{epochs} step {global_step} loss {avg:.4f}")

        avg = running / max(1, seen)
        print(f"[QAT] epoch {ep+1}/{epochs} done | avg loss {avg:.4f}")


def run_qat(
    cfg: dict,
    *,
    model_phase: str,
    train_source: str,
    backend: str,
    qat_epochs: int,
    qat_lr: float,
    qat_weight_decay: float,
    qat_grad_clip: float | None,
    bn_freeze_after: int,
    train_batch_size: int,
    eval_batch_size: int,
    keep_convtranspose_fp32: bool,
    keep_final_conv_fp32: bool,
    keep_batchnorm_fp32: bool,
    keep_concat_fp32: bool,
    fuse_conv_bn: bool,
    num_threads: int | None,
    bench_warmup: int,
    bench_runs: int,
    num_workers: int,
    qat_device: str,
) -> Path:
    seed = cfg["experiment"].get("seed", 42)
    deterministic = cfg["experiment"].get("deterministic", False)
    seed_everything(seed, deterministic=deterministic)

    if backend not in {"fbgemm", "qnnpack"}:
        raise ValueError("backend must be one of: fbgemm, qnnpack")

    torch.backends.quantized.engine = backend

    model_fp32, model_ckpt = load_target_model(cfg, model_phase=model_phase)

    train_cfg = cfg["train"]
    eval_cfg = cfg["evaluation"]

    # Train split to fine-tune QAT
    if train_source == "train":
        train_img_dir = Path(train_cfg["paths"]["train_dir"])
        train_lbl_dir = Path(train_cfg["paths"]["label_dir"])
        train_num_slices = train_cfg.get("num_slices_per_volume")
    elif train_source == "eval":
        train_img_dir = Path(eval_cfg["paths"]["eval_dir"])
        train_lbl_dir = Path(eval_cfg["paths"]["label_dir"])
        train_num_slices = eval_cfg.get("num_slices_per_volume")
    else:
        raise ValueError("train_source must be 'train' or 'eval'")

    # Eval split for reporting
    eval_img_dir = Path(eval_cfg["paths"]["eval_dir"])
    eval_lbl_dir = Path(eval_cfg["paths"]["label_dir"])
    eval_num_slices = eval_cfg.get("num_slices_per_volume")

    train_loader = build_loader(
        img_dir=train_img_dir,
        lbl_dir=train_lbl_dir,
        batch_size=train_batch_size,
        num_slices_per_volume=train_num_slices,
        num_workers=num_workers,
        pin_memory=(qat_device == "cuda"),
    )
    eval_loader = build_loader(
        img_dir=eval_img_dir,
        lbl_dir=eval_lbl_dir,
        batch_size=eval_batch_size,
        num_slices_per_volume=eval_num_slices,
        num_workers=num_workers,
        pin_memory=False,
    )

    # FX tracing & prepare happens on CPU (stable + matches final int8 backend),
    # then we move the prepared QAT model to GPU/CPU for fine-tuning.
    cpu = torch.device("cpu")
    example_inputs = make_example_inputs(train_loader, device=cpu)

    qconfig_mapping = build_qconfig_mapping(
        backend,
        keep_convtranspose_fp32=keep_convtranspose_fp32,
        keep_final_conv_fp32=keep_final_conv_fp32,
        keep_batchnorm_fp32=keep_batchnorm_fp32,
        keep_concat_fp32=keep_concat_fp32,
    )

    model_for_qat = make_unet_fx_traceable(copy.deepcopy(model_fp32)).train()
    if fuse_conv_bn:
        model_for_qat = fuse_fx(model_for_qat)

    prepared_qat = prepare_qat_fx(model_for_qat, qconfig_mapping, example_inputs)

    # Fine-tune
    if qat_device == "cuda" and not torch.cuda.is_available():
        print("⚠️  qat_device=cuda requested but CUDA not available. Falling back to CPU.")
        qat_device = "cpu"
    train_dev = torch.device(qat_device)

    _train_qat(
        prepared_qat,
        train_loader,
        train_dev,
        epochs=qat_epochs,
        lr=qat_lr,
        weight_decay=qat_weight_decay,
        grad_clip=qat_grad_clip,
        bn_freeze_after=bn_freeze_after,
        log_every=50,
    )

    # Convert to int8 on CPU
    prepared_qat.to(cpu).eval()
    model_int8 = convert_fx(prepared_qat).eval()

    # Evaluate on CPU for apples-to-apples with PTQ
    out_ch = int(train_cfg["model"]["out_channels"])
    eval_fp32, _ = run_evaluation(
        model=model_fp32,
        loader=eval_loader,
        device=cpu,
        num_classes=out_ch,
        dice_fn=dice_score,
        iou_fn=iou_score,
        vram_track=False,
    )
    eval_int8, _ = run_evaluation(
        model=model_int8,
        loader=eval_loader,
        device=cpu,
        num_classes=out_ch,
        dice_fn=dice_score,
        iou_fn=iou_score,
        vram_track=False,
    )

    sample, _ = extract_inputs_targets(next(iter(eval_loader)))
    sample = sample.to(cpu)
    lat_fp32 = latency_ms(model_fp32, sample, warmup=bench_warmup, runs=bench_runs, num_threads=num_threads)
    lat_int8 = latency_ms(model_int8, sample, warmup=bench_warmup, runs=bench_runs, num_threads=num_threads)

    from src.utils.paths import get_paths

    paths = get_paths(cfg)
    q_dir = paths.base_dir / "quantization_qat" / f"{model_phase}_{backend}"
    q_dir.mkdir(parents=True, exist_ok=True)

    # Save artifacts
    fp32_sd_path = q_dir / "fp32_reference_state_dict.pth"
    qat_sd_path = q_dir / "qat_prepared_state_dict.pth"
    int8_sd_path = q_dir / "int8_state_dict.pth"

    torch.save(model_fp32.state_dict(), fp32_sd_path)
    torch.save(prepared_qat.state_dict(), qat_sd_path)
    torch.save(model_int8.state_dict(), int8_sd_path)

    fp32_bytes = count_state_dict_bytes(model_fp32.state_dict())
    int8_bytes = count_state_dict_bytes(model_int8.state_dict())

    summary = base_run_info(cfg, stage="quantization")
    summary["quantization"] = {
        "method": "qat_fx_finetune",
        "model_phase": model_phase,
        "backend": backend,
        "model_checkpoint": str(model_ckpt),
        "qat_training": {
            "train_source": train_source,
            "img_dir": str(train_img_dir),
            "label_dir": str(train_lbl_dir),
            "batch_size": int(train_batch_size),
            "epochs": int(qat_epochs),
            "lr": float(qat_lr),
            "weight_decay": float(qat_weight_decay),
            "grad_clip": float(qat_grad_clip) if qat_grad_clip is not None else None,
            "bn_freeze_after_epoch": int(bn_freeze_after),
            "device": str(train_dev),
        },
        "qconfig": {
            "keep_convtranspose_fp32": bool(keep_convtranspose_fp32),
            "keep_final_conv_fp32": bool(keep_final_conv_fp32),
            "keep_batchnorm_fp32": bool(keep_batchnorm_fp32),
            "keep_concat_fp32": bool(keep_concat_fp32),
            "fuse_conv_bn": bool(fuse_conv_bn),
        },
        "cpu_benchmark": {
            "threads": int(num_threads) if num_threads is not None else None,
            "warmup_runs": int(bench_warmup),
            "timed_runs": int(bench_runs),
            "batch_shape": list(sample.shape),
            "fp32_latency_ms": float(lat_fp32),
            "int8_latency_ms": float(lat_int8),
            "speedup_x": float(lat_fp32 / lat_int8) if lat_int8 > 0 else float("nan"),
        },
        "metrics_eval": {
            "num_classes": out_ch,
            "fp32": {"mean_dice_fg": float(eval_fp32.mean_dice_fg), "mean_iou_fg": float(eval_fp32.mean_iou_fg)},
            "int8": {"mean_dice_fg": float(eval_int8.mean_dice_fg), "mean_iou_fg": float(eval_int8.mean_iou_fg)},
            "delta_int8_minus_fp32": {
                "mean_dice_fg": float(eval_int8.mean_dice_fg - eval_fp32.mean_dice_fg),
                "mean_iou_fg": float(eval_int8.mean_iou_fg - eval_fp32.mean_iou_fg),
            },
        },
        "state_dict_size": {
            "fp32_bytes": int(fp32_bytes),
            "int8_bytes": int(int8_bytes),
            "compression_ratio_fp32_over_int8": float(fp32_bytes / int8_bytes) if int8_bytes > 0 else float("nan"),
        },
        "artifacts": {
            "output_dir": str(q_dir),
            "fp32_state_dict": str(fp32_sd_path),
            "qat_prepared_state_dict": str(qat_sd_path),
            "int8_state_dict": str(int8_sd_path),
        },
    }

    summary_path = write_json(q_dir / "qat_summary.json", summary)
    return summary_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run QAT fine-tuning (FX graph mode) for UNet, then convert to int8.")
    p.add_argument("--config", type=str, default=None, help="Path to config.yaml.")
    p.add_argument(
        "--model-phase",
        type=str,
        default="baseline",
        choices=["baseline", "pruned", "retrained_pruned"],
        help="Which checkpoint family to quantize.",
    )
    p.add_argument("--train-source", type=str, default="train", choices=["train", "eval"])
    p.add_argument("--backend", type=str, default="fbgemm", choices=["fbgemm", "qnnpack"])

    p.add_argument("--qat-epochs", type=int, default=10)
    p.add_argument("--qat-lr", type=float, default=1e-4)
    p.add_argument("--qat-weight-decay", type=float, default=0.0)
    p.add_argument("--qat-grad-clip", type=float, default=0.0)
    p.add_argument(
        "--bn-freeze-after",
        type=int,
        default=1,
        help="Freeze BatchNorm running stats after this epoch index (0-based). Use -1 to never freeze.",
    )

    p.add_argument("--train-batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--qat-device", type=str, default="cuda", choices=["cuda", "cpu"])

    p.add_argument("--keep-convtranspose-fp32", dest="keep_convtranspose_fp32", action="store_true")
    p.add_argument("--quantize-convtranspose", dest="keep_convtranspose_fp32", action="store_false")
    p.add_argument("--keep-final-conv-fp32", action="store_true")

    p.add_argument("--keep-bn-fp32", dest="keep_batchnorm_fp32", action="store_true")
    p.add_argument("--quantize-bn", dest="keep_batchnorm_fp32", action="store_false")

    p.add_argument("--keep-concat-fp32", dest="keep_concat_fp32", action="store_true")
    p.add_argument("--quantize-concat", dest="keep_concat_fp32", action="store_false")

    p.add_argument("--fuse-conv-bn", dest="fuse_conv_bn", action="store_true")
    p.add_argument("--no-fuse-conv-bn", dest="fuse_conv_bn", action="store_false")

    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--bench-warmup", type=int, default=20)
    p.add_argument("--bench-runs", type=int, default=100)

    p.set_defaults(keep_convtranspose_fp32=True)
    p.set_defaults(keep_batchnorm_fp32=False)
    p.set_defaults(keep_concat_fp32=False)
    p.set_defaults(fuse_conv_bn=True)
    return p.parse_args()


def main():
    args = _parse_args()
    cfg = load_config(config_path=args.config) if args.config else load_config()

    grad_clip = None if args.qat_grad_clip is None or args.qat_grad_clip <= 0 else float(args.qat_grad_clip)

    summary_path = run_qat(
        cfg,
        model_phase=args.model_phase,
        train_source=args.train_source,
        backend=args.backend,
        qat_epochs=args.qat_epochs,
        qat_lr=args.qat_lr,
        qat_weight_decay=args.qat_weight_decay,
        qat_grad_clip=grad_clip,
        bn_freeze_after=args.bn_freeze_after,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        keep_convtranspose_fp32=args.keep_convtranspose_fp32,
        keep_final_conv_fp32=args.keep_final_conv_fp32,
        keep_batchnorm_fp32=args.keep_batchnorm_fp32,
        keep_concat_fp32=args.keep_concat_fp32,
        fuse_conv_bn=args.fuse_conv_bn,
        num_threads=args.num_threads,
        bench_warmup=args.bench_warmup,
        bench_runs=args.bench_runs,
        num_workers=args.num_workers,
        qat_device=args.qat_device,
    )
    print(f"✅ QAT finished. Summary: {summary_path}")


if __name__ == "__main__":
    main()
