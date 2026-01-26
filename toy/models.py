from __future__ import annotations

from typing import Iterable, List

from src.models.unet import UNet


def build_tiny_unet(
    *,
    in_ch: int,
    out_ch: int,
    features: Iterable[int] = (8, 16, 32),
    bottleneck_out: int | None = None,
) -> UNet:
    feat_list: List[int] = [int(f) for f in features]
    return UNet(
        in_ch=in_ch,
        out_ch=out_ch,
        enc_features=feat_list,
        dec_features=None,
        bottleneck_out=bottleneck_out,
    )
