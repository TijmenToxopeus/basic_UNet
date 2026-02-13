# from __future__ import annotations

# import argparse
# import copy
# import json
# import time
# from pathlib import Path
# from typing import Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.ao.quantization import QConfigMapping, get_default_qconfig
# from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

# from src.models.unet import UNet
# from src.pruning.rebuild import load_full_pruned_model
# from src.training.data_factory import build_eval_loader
# from src.training.eval_loop import run_evaluation
# from src.training.metrics import dice_score, iou_score
# from src.utils.config import load_config
# from src.utils.paths import get_paths
# from src.utils.reproducibility import seed_everything
# from src.utils.run_summary import base_run_info, write_json


# def _extract_inputs(batch):
#     x = batch[0] if isinstance(batch, (list, tuple)) else batch
#     if isinstance(x, (list, tuple)):
#         x = x[0]
#     return x


# def _make_example_inputs(loader, device: torch.device) -> Tuple[torch.Tensor]:
#     first = next(iter(loader))
#     x = _extract_inputs(first).to(device)
#     return (x,)


# @torch.inference_mode()
# def _run_calibration(
#     prepared_model: nn.Module,
#     loader,
#     device: torch.device,
#     *,
#     num_batches: int,
# ) -> int:
#     prepared_model.eval()
#     seen = 0
#     for batch in loader:
#         x = _extract_inputs(batch).to(device)
#         _ = prepared_model(x)
#         seen += 1
#         if seen >= num_batches:
#             break
#     return seen


# def _build_qconfig_mapping(
#     backend: str,
#     *,
#     keep_convtranspose_fp32: bool,
#     keep_final_conv_fp32: bool,
# ) -> QConfigMapping:
#     qconfig = get_default_qconfig(backend)
#     mapping = QConfigMapping().set_global(qconfig)

#     if keep_convtranspose_fp32:
#         mapping = mapping.set_object_type(nn.ConvTranspose2d, None)
#     if keep_final_conv_fp32:
#         # UNet stores this as `final_conv`.
#         mapping = mapping.set_module_name("final_conv", None)
#     return mapping


# def _count_state_dict_bytes(state_dict: dict[str, torch.Tensor]) -> int:
#     total = 0
#     for value in state_dict.values():
#         if isinstance(value, torch.Tensor):
#             total += value.numel() * value.element_size()
#     return total


# def _latency_ms(
#     model: nn.Module,
#     input_tensor: torch.Tensor,
#     *,
#     warmup: int,
#     runs: int,
#     num_threads: int | None,
# ) -> float:
#     if num_threads is not None and num_threads > 0:
#         torch.set_num_threads(int(num_threads))

#     model.eval()
#     with torch.inference_mode():
#         for _ in range(max(0, warmup)):
#             _ = model(input_tensor)

#         t0 = time.perf_counter()
#         for _ in range(max(1, runs)):
#             _ = model(input_tensor)
#         t1 = time.perf_counter()

#     return (t1 - t0) * 1000.0 / max(1, runs)


# def _load_target_model(cfg: dict, model_phase: str) -> tuple[nn.Module, Path]:
#     exp_cfg = cfg["experiment"]
#     train_cfg = cfg["train"]
#     in_ch = int(train_cfg["model"]["in_channels"])
#     out_ch = int(train_cfg["model"]["out_channels"])

#     paths = get_paths(cfg)
#     device = torch.device("cpu")

#     if model_phase == "baseline":
#         enc = train_cfg["model"]["features"]
#         model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc).to(device)
#         ckpt = paths.base_dir / "baseline" / "training" / "final_model.pth"
#         if not ckpt.exists():
#             raise FileNotFoundError(f"Baseline checkpoint not found: {ckpt}")
#         state = torch.load(ckpt, map_location=device)
#         model.load_state_dict(state)
#         return model.eval(), ckpt

#     if model_phase not in {"pruned", "retrained_pruned"}:
#         raise ValueError(f"Unsupported model_phase={model_phase}")

#     meta_path = paths.pruned_model.with_name(paths.pruned_model.stem + "_meta.json")
#     if not meta_path.exists():
#         raise FileNotFoundError(f"Pruned meta file not found: {meta_path}")
#     with meta_path.open("r") as f:
#         meta = json.load(f)

#     if model_phase == "pruned":
#         ckpt = paths.pruned_model
#     else:
#         ckpt = paths.retrain_pruned_dir / "final_model.pth"
#     if not ckpt.exists():
#         raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

#     model = load_full_pruned_model(meta=meta, ckpt_path=ckpt, in_ch=in_ch, out_ch=out_ch, device=device)
#     return model.eval(), ckpt


# def _build_loader(
#     *,
#     img_dir,
#     lbl_dir,
#     batch_size: int,
#     num_slices_per_volume,
#     num_workers: int,
#     pin_memory: bool,
# ):
#     loader, _ = build_eval_loader(
#         img_dir=img_dir,
#         lbl_dir=lbl_dir,
#         batch_size=batch_size,
#         num_slices_per_volume=num_slices_per_volume,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#     )
#     return loader


# def _make_unet_fx_traceable(model: nn.Module) -> nn.Module:
#     """
#     Replace Python control flow in UNet forward with an FX-traceable variant.
#     """
#     if not all(hasattr(model, name) for name in ("encoders", "bottleneck", "decoders", "pool", "final_conv")):
#         return model

#     def _forward_fx(self, x):
#         skip_connections = []
#         for enc in self.encoders:
#             x = enc(x)
#             skip_connections.append(x)
#             x = self.pool(x)

#         x = self.bottleneck(x)
#         skip_connections = skip_connections[::-1]

#         for idx in range(0, len(self.decoders), 2):
#             x = self.decoders[idx](x)
#             skip = skip_connections[idx // 2]
#             # Always interpolate to avoid Proxy bool control flow during FX tracing.
#             x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
#             x = torch.cat((skip, x), dim=1)
#             x = self.decoders[idx + 1](x)

#         return self.final_conv(x)

#     # FX uses class-level forward lookup during tracing, so override at class level
#     # for this instance by swapping to a temporary subclass.
#     traceable_cls = type(
#         f"{model.__class__.__name__}FXTraceable",
#         (model.__class__,),
#         {"forward": _forward_fx},
#     )
#     model.__class__ = traceable_cls
#     return model


# def run_ptq(
#     cfg: dict,
#     *,
#     model_phase: str,
#     calib_source: str,
#     backend: str,
#     calib_batches: int,
#     calib_batch_size: int,
#     eval_batch_size: int,
#     keep_convtranspose_fp32: bool,
#     keep_final_conv_fp32: bool,
#     num_threads: int | None,
#     bench_warmup: int,
#     bench_runs: int,
#     num_workers: int,
# ) -> Path:
#     seed = cfg["experiment"].get("seed", 42)
#     deterministic = cfg["experiment"].get("deterministic", False)
#     seed_everything(seed, deterministic=deterministic)

#     if backend not in {"fbgemm", "qnnpack"}:
#         raise ValueError("backend must be one of: fbgemm, qnnpack")

#     torch.backends.quantized.engine = backend
#     device = torch.device("cpu")

#     model_fp32, model_ckpt = _load_target_model(cfg, model_phase=model_phase)

#     train_cfg = cfg["train"]
#     eval_cfg = cfg["evaluation"]

#     if calib_source == "train":
#         calib_img_dir = Path(train_cfg["paths"]["train_dir"])
#         calib_lbl_dir = Path(train_cfg["paths"]["label_dir"])
#         calib_num_slices = train_cfg.get("num_slices_per_volume")
#     elif calib_source == "eval":
#         calib_img_dir = Path(eval_cfg["paths"]["eval_dir"])
#         calib_lbl_dir = Path(eval_cfg["paths"]["label_dir"])
#         calib_num_slices = eval_cfg.get("num_slices_per_volume")
#     else:
#         raise ValueError("calib_source must be 'train' or 'eval'")

#     eval_img_dir = Path(eval_cfg["paths"]["eval_dir"])
#     eval_lbl_dir = Path(eval_cfg["paths"]["label_dir"])
#     eval_num_slices = eval_cfg.get("num_slices_per_volume")

#     calib_loader = _build_loader(
#         img_dir=calib_img_dir,
#         lbl_dir=calib_lbl_dir,
#         batch_size=calib_batch_size,
#         num_slices_per_volume=calib_num_slices,
#         num_workers=num_workers,
#         pin_memory=False,
#     )
#     eval_loader = _build_loader(
#         img_dir=eval_img_dir,
#         lbl_dir=eval_lbl_dir,
#         batch_size=eval_batch_size,
#         num_slices_per_volume=eval_num_slices,
#         num_workers=num_workers,
#         pin_memory=False,
#     )

#     example_inputs = _make_example_inputs(calib_loader, device=device)
#     qconfig_mapping = _build_qconfig_mapping(
#         backend,
#         keep_convtranspose_fp32=keep_convtranspose_fp32,
#         keep_final_conv_fp32=keep_final_conv_fp32,
#     )

#     model_for_quant = _make_unet_fx_traceable(copy.deepcopy(model_fp32)).eval()
#     prepared = prepare_fx(model_for_quant, qconfig_mapping, example_inputs)
#     used_calib_batches = _run_calibration(prepared, calib_loader, device, num_batches=calib_batches)
#     model_int8 = convert_fx(prepared).eval()

#     out_ch = int(train_cfg["model"]["out_channels"])
#     eval_fp32, _ = run_evaluation(
#         model=model_fp32,
#         loader=eval_loader,
#         device=device,
#         num_classes=out_ch,
#         dice_fn=dice_score,
#         iou_fn=iou_score,
#         vram_track=False,
#     )
#     eval_int8, _ = run_evaluation(
#         model=model_int8,
#         loader=eval_loader,
#         device=device,
#         num_classes=out_ch,
#         dice_fn=dice_score,
#         iou_fn=iou_score,
#         vram_track=False,
#     )

#     sample = _extract_inputs(next(iter(eval_loader))).to(device)
#     lat_fp32 = _latency_ms(
#         model_fp32,
#         sample,
#         warmup=bench_warmup,
#         runs=bench_runs,
#         num_threads=num_threads,
#     )
#     lat_int8 = _latency_ms(
#         model_int8,
#         sample,
#         warmup=bench_warmup,
#         runs=bench_runs,
#         num_threads=num_threads,
#     )

#     paths = get_paths(cfg)
#     q_dir = paths.base_dir / "quantization" / f"{model_phase}_{backend}"
#     q_dir.mkdir(parents=True, exist_ok=True)

#     fp32_sd_path = q_dir / "fp32_reference_state_dict.pth"
#     int8_sd_path = q_dir / "int8_state_dict.pth"
#     torch.save(model_fp32.state_dict(), fp32_sd_path)
#     torch.save(model_int8.state_dict(), int8_sd_path)

#     fp32_bytes = _count_state_dict_bytes(model_fp32.state_dict())
#     int8_bytes = _count_state_dict_bytes(model_int8.state_dict())

#     summary = base_run_info(cfg, stage="quantization")
#     summary["quantization"] = {
#         "method": "static_ptq_fx",
#         "model_phase": model_phase,
#         "backend": backend,
#         "model_checkpoint": str(model_ckpt),
#         "calibration": {
#             "source": calib_source,
#             "img_dir": str(calib_img_dir),
#             "label_dir": str(calib_lbl_dir),
#             "batch_size": int(calib_batch_size),
#             "requested_batches": int(calib_batches),
#             "used_batches": int(used_calib_batches),
#         },
#         "qconfig": {
#             "keep_convtranspose_fp32": bool(keep_convtranspose_fp32),
#             "keep_final_conv_fp32": bool(keep_final_conv_fp32),
#         },
#         "cpu_benchmark": {
#             "threads": int(num_threads) if num_threads is not None else None,
#             "warmup_runs": int(bench_warmup),
#             "timed_runs": int(bench_runs),
#             "batch_shape": list(sample.shape),
#             "fp32_latency_ms": float(lat_fp32),
#             "int8_latency_ms": float(lat_int8),
#             "speedup_x": float(lat_fp32 / lat_int8) if lat_int8 > 0 else float("nan"),
#         },
#         "metrics_eval": {
#             "num_classes": out_ch,
#             "fp32": {
#                 "mean_dice_fg": float(eval_fp32.mean_dice_fg),
#                 "mean_iou_fg": float(eval_fp32.mean_iou_fg),
#             },
#             "int8": {
#                 "mean_dice_fg": float(eval_int8.mean_dice_fg),
#                 "mean_iou_fg": float(eval_int8.mean_iou_fg),
#             },
#             "delta_int8_minus_fp32": {
#                 "mean_dice_fg": float(eval_int8.mean_dice_fg - eval_fp32.mean_dice_fg),
#                 "mean_iou_fg": float(eval_int8.mean_iou_fg - eval_fp32.mean_iou_fg),
#             },
#         },
#         "state_dict_size": {
#             "fp32_bytes": int(fp32_bytes),
#             "int8_bytes": int(int8_bytes),
#             "compression_ratio_fp32_over_int8": float(fp32_bytes / int8_bytes) if int8_bytes > 0 else float("nan"),
#         },
#         "artifacts": {
#             "output_dir": str(q_dir),
#             "fp32_state_dict": str(fp32_sd_path),
#             "int8_state_dict": str(int8_sd_path),
#         },
#     }

#     summary_path = write_json(q_dir / "ptq_summary.json", summary)
#     return summary_path


# def _parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Run static post-training quantization (PTQ) for UNet.")
#     parser.add_argument("--config", type=str, default=None, help="Path to config.yaml.")
#     parser.add_argument(
#         "--model-phase",
#         type=str,
#         default="baseline",
#         choices=["baseline", "pruned", "retrained_pruned"],
#         help="Which checkpoint family to quantize.",
#     )
#     parser.add_argument(
#         "--calib-source",
#         type=str,
#         default="train",
#         choices=["train", "eval"],
#         help="Dataset split used for calibration.",
#     )
#     parser.add_argument("--backend", type=str, default="fbgemm", choices=["fbgemm", "qnnpack"])
#     parser.add_argument("--calib-batches", type=int, default=300)
#     parser.add_argument("--calib-batch-size", type=int, default=1)
#     parser.add_argument("--eval-batch-size", type=int, default=1)
#     parser.add_argument("--num-workers", type=int, default=0)
#     parser.add_argument(
#         "--keep-convtranspose-fp32",
#         dest="keep_convtranspose_fp32",
#         action="store_true",
#         help="Keep ConvTranspose2d layers in float (recommended for UNet).",
#     )
#     parser.add_argument(
#         "--quantize-convtranspose",
#         dest="keep_convtranspose_fp32",
#         action="store_false",
#         help="Allow ConvTranspose2d quantization.",
#     )
#     parser.add_argument(
#         "--keep-final-conv-fp32",
#         action="store_true",
#         help="Keep final logits conv in float for better boundary stability.",
#     )
#     parser.add_argument("--num-threads", type=int, default=1)
#     parser.add_argument("--bench-warmup", type=int, default=20)
#     parser.add_argument("--bench-runs", type=int, default=100)
#     parser.set_defaults(keep_convtranspose_fp32=True)
#     return parser.parse_args()


# def main():
#     args = _parse_args()
#     if args.config:
#         cfg = load_config(config_path=args.config)
#     else:
#         cfg = load_config()

#     summary_path = run_ptq(
#         cfg,
#         model_phase=args.model_phase,
#         calib_source=args.calib_source,
#         backend=args.backend,
#         calib_batches=args.calib_batches,
#         calib_batch_size=args.calib_batch_size,
#         eval_batch_size=args.eval_batch_size,
#         keep_convtranspose_fp32=args.keep_convtranspose_fp32,
#         keep_final_conv_fp32=args.keep_final_conv_fp32,
#         num_threads=args.num_threads,
#         bench_warmup=args.bench_warmup,
#         bench_runs=args.bench_runs,
#         num_workers=args.num_workers,
#     )
#     print(f"✅ PTQ finished. Summary: {summary_path}")


# if __name__ == "__main__":
#     main()



from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import QConfigMapping, get_default_qconfig
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx, fuse_fx

from src.models.unet import UNet
from src.pruning.rebuild import load_full_pruned_model
from src.training.data_factory import build_eval_loader
from src.training.eval_loop import run_evaluation
from src.training.metrics import dice_score, iou_score
from src.utils.config import load_config
from src.utils.paths import get_paths
from src.utils.reproducibility import seed_everything
from src.utils.run_summary import base_run_info, write_json


def _extract_inputs(batch):
    x = batch[0] if isinstance(batch, (list, tuple)) else batch
    if isinstance(x, (list, tuple)):
        x = x[0]
    return x


def _make_example_inputs(loader, device: torch.device) -> Tuple[torch.Tensor]:
    first = next(iter(loader))
    x = _extract_inputs(first).to(device)
    return (x,)


@torch.inference_mode()
def _run_calibration(
    prepared_model: nn.Module,
    loader,
    device: torch.device,
    *,
    num_batches: int,
) -> int:
    prepared_model.eval()
    seen = 0
    for batch in loader:
        x = _extract_inputs(batch).to(device)
        _ = prepared_model(x)
        seen += 1
        if seen >= num_batches:
            break
    return seen


def _build_qconfig_mapping(
    backend: str,
    *,
    keep_convtranspose_fp32: bool,
    keep_final_conv_fp32: bool,
    keep_batchnorm_fp32: bool,
) -> QConfigMapping:
    qconfig = get_default_qconfig(backend)
    mapping = QConfigMapping().set_global(qconfig)

    if keep_convtranspose_fp32:
        mapping = mapping.set_object_type(nn.ConvTranspose2d, None)
    if keep_final_conv_fp32:
        # UNet stores this as `final_conv`.
        mapping = mapping.set_module_name("final_conv", None)
    if keep_batchnorm_fp32:
        mapping = mapping.set_object_type(nn.BatchNorm2d, None)

    return mapping


def _count_state_dict_bytes(state_dict: dict[str, torch.Tensor]) -> int:
    total = 0
    for value in state_dict.values():
        if isinstance(value, torch.Tensor):
            total += value.numel() * value.element_size()
    return total


def _latency_ms(
    model: nn.Module,
    input_tensor: torch.Tensor,
    *,
    warmup: int,
    runs: int,
    num_threads: int | None,
) -> float:
    if num_threads is not None and num_threads > 0:
        torch.set_num_threads(int(num_threads))

    model.eval()
    with torch.inference_mode():
        for _ in range(max(0, warmup)):
            _ = model(input_tensor)

        t0 = time.perf_counter()
        for _ in range(max(1, runs)):
            _ = model(input_tensor)
        t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / max(1, runs)


def _load_target_model(cfg: dict, model_phase: str) -> tuple[nn.Module, Path]:
    train_cfg = cfg["train"]
    in_ch = int(train_cfg["model"]["in_channels"])
    out_ch = int(train_cfg["model"]["out_channels"])

    paths = get_paths(cfg)
    device = torch.device("cpu")

    if model_phase == "baseline":
        enc = train_cfg["model"]["features"]
        model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc).to(device)
        ckpt = paths.base_dir / "baseline" / "training" / "final_model.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"Baseline checkpoint not found: {ckpt}")
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        return model.eval(), ckpt

    if model_phase not in {"pruned", "retrained_pruned"}:
        raise ValueError(f"Unsupported model_phase={model_phase}")

    meta_path = paths.pruned_model.with_name(paths.pruned_model.stem + "_meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Pruned meta file not found: {meta_path}")
    with meta_path.open("r") as f:
        meta = json.load(f)

    if model_phase == "pruned":
        ckpt = paths.pruned_model
    else:
        ckpt = paths.retrain_pruned_dir / "final_model.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model = load_full_pruned_model(meta=meta, ckpt_path=ckpt, in_ch=in_ch, out_ch=out_ch, device=device)
    return model.eval(), ckpt


def _build_loader(
    *,
    img_dir,
    lbl_dir,
    batch_size: int,
    num_slices_per_volume,
    num_workers: int,
    pin_memory: bool,
):
    loader, _ = build_eval_loader(
        img_dir=img_dir,
        lbl_dir=lbl_dir,
        batch_size=batch_size,
        num_slices_per_volume=num_slices_per_volume,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader


def _make_unet_fx_traceable(model: nn.Module) -> nn.Module:
    """
    Replace Python control flow in UNet forward with an FX-traceable variant.
    """
    if not all(hasattr(model, name) for name in ("encoders", "bottleneck", "decoders", "pool", "final_conv")):
        return model

    def _forward_fx(self, x):
        skip_connections = []
        for enc in self.encoders:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.decoders), 2):
            x = self.decoders[idx](x)
            skip = skip_connections[idx // 2]
            # Always interpolate to avoid Proxy bool control flow during FX tracing.
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat((skip, x), dim=1)
            x = self.decoders[idx + 1](x)

        return self.final_conv(x)

    traceable_cls = type(
        f"{model.__class__.__name__}FXTraceable",
        (model.__class__,),
        {"forward": _forward_fx},
    )
    model.__class__ = traceable_cls
    return model


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

    model_fp32, model_ckpt = _load_target_model(cfg, model_phase=model_phase)

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

    calib_loader = _build_loader(
        img_dir=calib_img_dir,
        lbl_dir=calib_lbl_dir,
        batch_size=calib_batch_size,
        num_slices_per_volume=calib_num_slices,
        num_workers=num_workers,
        pin_memory=False,
    )
    eval_loader = _build_loader(
        img_dir=eval_img_dir,
        lbl_dir=eval_lbl_dir,
        batch_size=eval_batch_size,
        num_slices_per_volume=eval_num_slices,
        num_workers=num_workers,
        pin_memory=False,
    )

    example_inputs = _make_example_inputs(calib_loader, device=device)
    qconfig_mapping = _build_qconfig_mapping(
        backend,
        keep_convtranspose_fp32=keep_convtranspose_fp32,
        keep_final_conv_fp32=keep_final_conv_fp32,
        keep_batchnorm_fp32=keep_batchnorm_fp32,
    )

    model_for_quant = _make_unet_fx_traceable(copy.deepcopy(model_fp32)).eval()
    if fuse_conv_bn:
        # Fuses patterns like Conv-BN-(ReLU) into single modules for better PTQ stability.
        model_for_quant = fuse_fx(model_for_quant)

    prepared = prepare_fx(model_for_quant, qconfig_mapping, example_inputs)
    used_calib_batches = _run_calibration(prepared, calib_loader, device, num_batches=calib_batches)
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

    sample = _extract_inputs(next(iter(eval_loader))).to(device)
    lat_fp32 = _latency_ms(
        model_fp32,
        sample,
        warmup=bench_warmup,
        runs=bench_runs,
        num_threads=num_threads,
    )
    lat_int8 = _latency_ms(
        model_int8,
        sample,
        warmup=bench_warmup,
        runs=bench_runs,
        num_threads=num_threads,
    )

    paths = get_paths(cfg)
    q_dir = paths.base_dir / "quantization" / f"{model_phase}_{backend}"
    q_dir.mkdir(parents=True, exist_ok=True)

    fp32_sd_path = q_dir / "fp32_reference_state_dict.pth"
    int8_sd_path = q_dir / "int8_state_dict.pth"
    torch.save(model_fp32.state_dict(), fp32_sd_path)
    torch.save(model_int8.state_dict(), int8_sd_path)

    fp32_bytes = _count_state_dict_bytes(model_fp32.state_dict())
    int8_bytes = _count_state_dict_bytes(model_int8.state_dict())

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
            "fp32": {
                "mean_dice_fg": float(eval_fp32.mean_dice_fg),
                "mean_iou_fg": float(eval_fp32.mean_iou_fg),
            },
            "int8": {
                "mean_dice_fg": float(eval_int8.mean_dice_fg),
                "mean_iou_fg": float(eval_int8.mean_iou_fg),
            },
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
    parser = argparse.ArgumentParser(description="Run static post-training quantization (PTQ) for UNet.")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml.")
    parser.add_argument(
        "--model-phase",
        type=str,
        default="baseline",
        choices=["baseline", "pruned", "retrained_pruned"],
        help="Which checkpoint family to quantize.",
    )
    parser.add_argument(
        "--calib-source",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="Dataset split used for calibration.",
    )
    parser.add_argument("--backend", type=str, default="fbgemm", choices=["fbgemm", "qnnpack"])
    parser.add_argument("--calib-batches", type=int, default=300)
    parser.add_argument("--calib-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--keep-convtranspose-fp32",
        dest="keep_convtranspose_fp32",
        action="store_true",
        help="Keep ConvTranspose2d layers in float (recommended for UNet).",
    )
    parser.add_argument(
        "--quantize-convtranspose",
        dest="keep_convtranspose_fp32",
        action="store_false",
        help="Allow ConvTranspose2d quantization.",
    )
    parser.add_argument(
        "--keep-final-conv-fp32",
        action="store_true",
        help="Keep final logits conv in float for better boundary stability.",
    )
    parser.add_argument(
        "--keep-bn-fp32",
        dest="keep_batchnorm_fp32",
        action="store_true",
        help="Keep BatchNorm2d layers in float (useful if fusion fails or Dice drops).",
    )
    parser.add_argument(
        "--quantize-bn",
        dest="keep_batchnorm_fp32",
        action="store_false",
        help="Allow BatchNorm2d quantization / fusion.",
    )
    parser.add_argument(
        "--fuse-conv-bn",
        dest="fuse_conv_bn",
        action="store_true",
        help="Run FX fusion (Conv-BN-(ReLU)) before PTQ (recommended).",
    )
    parser.add_argument(
        "--no-fuse-conv-bn",
        dest="fuse_conv_bn",
        action="store_false",
        help="Disable FX fusion before PTQ.",
    )
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--bench-warmup", type=int, default=20)
    parser.add_argument("--bench-runs", type=int, default=100)

    # Defaults: keep deconv float; fuse conv-bn; allow BN fusion by default (i.e., do NOT force BN float)
    parser.set_defaults(keep_convtranspose_fp32=True)
    parser.set_defaults(keep_batchnorm_fp32=False)
    parser.set_defaults(fuse_conv_bn=True)

    return parser.parse_args()


def main():
    args = _parse_args()
    if args.config:
        cfg = load_config(config_path=args.config)
    else:
        cfg = load_config()

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
        fuse_conv_bn=args.fuse_conv_bn,
        num_threads=args.num_threads,
        bench_warmup=args.bench_warmup,
        bench_runs=args.bench_runs,
        num_workers=args.num_workers,
    )
    print(f"✅ PTQ finished. Summary: {summary_path}")


if __name__ == "__main__":
    main()
