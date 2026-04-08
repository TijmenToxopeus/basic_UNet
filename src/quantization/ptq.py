# src/quantization/ptq.py
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx, fuse_fx

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
    run_observer_calibration,
)


def run_ptq(
    cfg: dict,
    *,
    model_phase: str,
    calib_source: str,
    backend: str,
    calib_batches: int,
    calib_batch_size: int,
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
) -> Path:
    seed = cfg["experiment"].get("seed", 42)
    deterministic = cfg["experiment"].get("deterministic", False)
    seed_everything(seed, deterministic=deterministic)

    if backend not in {"fbgemm", "qnnpack"}:
        raise ValueError("backend must be one of: fbgemm, qnnpack")

    torch.backends.quantized.engine = backend
    device = torch.device("cpu")

    model_fp32, model_ckpt = load_target_model(cfg, model_phase=model_phase)

    train_cfg = cfg["train"]
    eval_cfg = cfg["evaluation"]

    if calib_source == "train":
        calib_img_dir = Path(train_cfg["paths"]["train_dir"])
        calib_lbl_dir = Path(train_cfg["paths"]["label_dir"])
        calib_num_slices = train_cfg.get("num_slices_per_volume")
    elif calib_source == "eval":
        calib_img_dir = Path(eval_cfg["paths"]["eval_dir"])
        calib_lbl_dir = Path(eval_cfg["paths"]["label_dir"])
        calib_num_slices = eval_cfg.get("num_slices_per_volume")
    else:
        raise ValueError("calib_source must be 'train' or 'eval'")

    eval_img_dir = Path(eval_cfg["paths"]["eval_dir"])
    eval_lbl_dir = Path(eval_cfg["paths"]["label_dir"])
    eval_num_slices = eval_cfg.get("num_slices_per_volume")

    calib_loader = build_loader(
        img_dir=calib_img_dir,
        lbl_dir=calib_lbl_dir,
        batch_size=calib_batch_size,
        num_slices_per_volume=calib_num_slices,
        num_workers=num_workers,
        pin_memory=False,
    )
    eval_loader = build_loader(
        img_dir=eval_img_dir,
        lbl_dir=eval_lbl_dir,
        batch_size=eval_batch_size,
        num_slices_per_volume=eval_num_slices,
        num_workers=num_workers,
        pin_memory=False,
    )

    example_inputs = make_example_inputs(calib_loader, device=device)
    qconfig_mapping = build_qconfig_mapping(
        backend,
        keep_convtranspose_fp32=keep_convtranspose_fp32,
        keep_final_conv_fp32=keep_final_conv_fp32,
        keep_batchnorm_fp32=keep_batchnorm_fp32,
        keep_concat_fp32=keep_concat_fp32,
    )

    model_for_quant = make_unet_fx_traceable(copy.deepcopy(model_fp32)).eval()
    if fuse_conv_bn:
        model_for_quant = fuse_fx(model_for_quant)

    prepared = prepare_fx(model_for_quant, qconfig_mapping, example_inputs)
    used_calib_batches = run_observer_calibration(prepared, calib_loader, device, num_batches=calib_batches)
    model_int8 = convert_fx(prepared).eval()

    out_ch = int(train_cfg["model"]["out_channels"])
    eval_fp32, _ = run_evaluation(
        model=model_fp32,
        loader=eval_loader,
        device=device,
        num_classes=out_ch,
        dice_fn=dice_score,
        iou_fn=iou_score,
        vram_track=False,
    )
    eval_int8, _ = run_evaluation(
        model=model_int8,
        loader=eval_loader,
        device=device,
        num_classes=out_ch,
        dice_fn=dice_score,
        iou_fn=iou_score,
        vram_track=False,
    )

    sample, _ = extract_inputs_targets(next(iter(eval_loader)))
    sample = sample.to(device)

    lat_fp32 = latency_ms(model_fp32, sample, warmup=bench_warmup, runs=bench_runs, num_threads=num_threads)
    lat_int8 = latency_ms(model_int8, sample, warmup=bench_warmup, runs=bench_runs, num_threads=num_threads)

    from src.utils.paths import get_paths

    paths = get_paths(cfg)
    q_dir = paths.base_dir / "quantization" / f"{model_phase}_{backend}"
    q_dir.mkdir(parents=True, exist_ok=True)

    fp32_sd_path = q_dir / "fp32_reference_state_dict.pth"
    int8_sd_path = q_dir / "int8_state_dict.pth"
    torch.save(model_fp32.state_dict(), fp32_sd_path)
    torch.save(model_int8.state_dict(), int8_sd_path)

    fp32_bytes = count_state_dict_bytes(model_fp32.state_dict())
    int8_bytes = count_state_dict_bytes(model_int8.state_dict())

    summary = base_run_info(cfg, stage="quantization")
    summary["quantization"] = {
        "method": "static_ptq_fx",
        "model_phase": model_phase,
        "backend": backend,
        "model_checkpoint": str(model_ckpt),
        "calibration": {
            "source": calib_source,
            "img_dir": str(calib_img_dir),
            "label_dir": str(calib_lbl_dir),
            "batch_size": int(calib_batch_size),
            "requested_batches": int(calib_batches),
            "used_batches": int(used_calib_batches),
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
            "int8_state_dict": str(int8_sd_path),
        },
    }

    summary_path = write_json(q_dir / "ptq_summary.json", summary)
    return summary_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run static post-training quantization (PTQ) for UNet.")
    p.add_argument("--config", type=str, default=None, help="Path to config.yaml.")
    p.add_argument(
        "--model-phase",
        type=str,
        default="baseline",
        choices=["baseline", "pruned", "retrained_pruned"],
        help="Which checkpoint family to quantize.",
    )
    p.add_argument("--calib-source", type=str, default="train", choices=["train", "eval"])
    p.add_argument("--backend", type=str, default="fbgemm", choices=["fbgemm", "qnnpack"])
    p.add_argument("--calib-batches", type=int, default=300)
    p.add_argument("--calib-batch-size", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument("--keep-convtranspose-fp32", dest="keep_convtranspose_fp32", action="store_true")
    p.add_argument("--quantize-convtranspose", dest="keep_convtranspose_fp32", action="store_false")
    p.add_argument("--keep-final-conv-fp32", action="store_true")

    p.add_argument("--keep-bn-fp32", dest="keep_batchnorm_fp32", action="store_true")
    p.add_argument("--quantize-bn", dest="keep_batchnorm_fp32", action="store_false")

    p.add_argument("--keep-concat-fp32", dest="keep_concat_fp32", action="store_true")
    p.add_argument("--quantize-concat", dest="keep_concat_fp32", action="store_false")

    p.add_argument("--fuse-conv-bn", dest="fuse_conv_bn", action="store_true")
    p.add_argument("--no-fuse-conv-bn", dest="fuse_conv_bn", action="store_false")

    p.add_argument("--num-threads", type=int, default=1)
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

    summary_path = run_ptq(
        cfg,
        model_phase=args.model_phase,
        calib_source=args.calib_source,
        backend=args.backend,
        calib_batches=args.calib_batches,
        calib_batch_size=args.calib_batch_size,
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
    )
    print(f"✅ PTQ finished. Summary: {summary_path}")


if __name__ == "__main__":
    main()





# # Example command to run PTQ with custom options:
# # python -m src.quantization.ptq \
# #   --config /mnt/hdd/ttoxopeus/basic_UNet/src/config.yaml \
# #   --model-phase baseline \
# #   --calib-batches 400 \
# #   --num-threads 4 \
# #   --keep-concat-fp32