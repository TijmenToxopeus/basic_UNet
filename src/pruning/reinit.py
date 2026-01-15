# src/pruning/reinit.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch


@dataclass
class ReinitStats:
    mean_before: float
    std_before: float
    mean_after: float
    std_after: float


def _flatten_weights(model: torch.nn.Module) -> torch.Tensor:
    params = []
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
            if m.weight is not None:
                params.append(m.weight.detach().cpu().flatten())
    if not params:
        return torch.tensor([])
    return torch.cat(params)


def default_init_fn(m: torch.nn.Module) -> None:
    """
    Reasonable default init for UNet-ish conv nets.
    """
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    elif isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.ones_(m.weight)
        torch.nn.init.zeros_(m.bias)


def random_reinitialize(
    model: torch.nn.Module,
    *,
    init_fn=default_init_fn,
) -> ReinitStats:
    """
    Applies init_fn to the model (in-place) and returns simple global stats.
    """
    before = _flatten_weights(model)
    mean_before = float(before.mean().item()) if before.numel() else float("nan")
    std_before = float(before.std().item()) if before.numel() else float("nan")

    model.apply(init_fn)

    after = _flatten_weights(model)
    mean_after = float(after.mean().item()) if after.numel() else float("nan")
    std_after = float(after.std().item()) if after.numel() else float("nan")

    return ReinitStats(
        mean_before=mean_before,
        std_before=std_before,
        mean_after=mean_after,
        std_after=std_after,
    )


def load_rewind_model(
    *,
    rewind_ckpt: Path,
    model_ctor,
    device: torch.device,
) -> torch.nn.Module:
    """
    Loads a model from a rewind checkpoint.
    - model_ctor: a zero-arg callable that returns the correct model architecture
      (e.g. lambda: UNet(in_ch=..., out_ch=..., enc_features=...))
    """
    print(f"🔄 Rewind checkpoint requested: {rewind_ckpt}")

    if rewind_ckpt is None or not Path(rewind_ckpt).exists():
        raise FileNotFoundError(
            f"❌ Rewind checkpoint not found at: {rewind_ckpt}"
        )

    model = model_ctor().to(device)
    state = torch.load(rewind_ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model