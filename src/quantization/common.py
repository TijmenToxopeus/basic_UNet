# src/quantization/common.py
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import QConfigMapping, get_default_qconfig

from src.models.unet import UNet
from src.pruning.rebuild import load_full_pruned_model
from src.training.data_factory import build_eval_loader
from src.utils.paths import get_paths


# -----------------------
# Batch / loader helpers
# -----------------------
def extract_inputs_targets(batch: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Supports common dataset returns:
      - (x, y)
      - (x, y, ...)
      - x
      - ((x, ...), y, ...)
    """
    if isinstance(batch, (list, tuple)):
        x = batch[0]
        y = batch[1] if len(batch) > 1 and isinstance(batch[1], torch.Tensor) else None
    else:
        x, y = batch, None

    if isinstance(x, (list, tuple)):
        x = x[0]

    return x, y


def make_example_inputs(loader, device: torch.device) -> Tuple[torch.Tensor]:
    first = next(iter(loader))
    x, _ = extract_inputs_targets(first)
    return (x.to(device),)


@torch.inference_mode()
def run_observer_calibration(
    prepared_model: nn.Module,
    loader,
    device: torch.device,
    *,
    num_batches: int,
) -> int:
    prepared_model.eval()
    seen = 0
    for batch in loader:
        x, _ = extract_inputs_targets(batch)
        _ = prepared_model(x.to(device))
        seen += 1
        if seen >= num_batches:
            break
    return seen


def build_loader(
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


# -----------------------
# Model loading
# -----------------------
def load_target_model(cfg: dict, model_phase: str) -> tuple[nn.Module, Path]:
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


# -----------------------
# FX traceability helper
# -----------------------
def make_unet_fx_traceable(model: nn.Module) -> nn.Module:
    """
    Replace Python control flow in UNet forward with an FX-traceable variant.

    Note: This assumes your UNet has attributes:
      encoders, bottleneck, decoders, pool, final_conv
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


# -----------------------
# Quant mapping / misc
# -----------------------
def build_qconfig_mapping(
    backend: str,
    *,
    keep_convtranspose_fp32: bool,
    keep_final_conv_fp32: bool,
    keep_batchnorm_fp32: bool,
    keep_concat_fp32: bool,
) -> QConfigMapping:
    """
    QConfigMapping for FX quantization.

    NOTE on keep_concat_fp32:
      Operator-level exclusions are a bit backend/version dependent.
      We try a couple of common operator handles. If neither applies, it will
      simply behave as if keep_concat_fp32=False (still safe).
    """
    qconfig = get_default_qconfig(backend)
    mapping = QConfigMapping().set_global(qconfig)

    if keep_convtranspose_fp32:
        mapping = mapping.set_object_type(nn.ConvTranspose2d, None)

    if keep_final_conv_fp32:
        mapping = mapping.set_module_name("final_conv", None)

    if keep_batchnorm_fp32:
        mapping = mapping.set_object_type(nn.BatchNorm2d, None)

    if keep_concat_fp32:
        # Try operator overloads that FX may surface for cat
        try:
            mapping = mapping.set_object_type(torch.ops.aten.cat.default, None)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            mapping = mapping.set_object_type(torch.cat, None)  # may or may not be honored by FX
        except Exception:
            pass

    return mapping


def count_state_dict_bytes(state_dict: dict[str, torch.Tensor]) -> int:
    total = 0
    for v in state_dict.values():
        if isinstance(v, torch.Tensor):
            total += v.numel() * v.element_size()
    return total


def latency_ms(
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


def deepcopy_eval(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model).eval()
