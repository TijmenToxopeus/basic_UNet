import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) × 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------------------
# Flexible U-Net (supports asymmetric pruning)
# -------------------------------------------------------------
class UNet(nn.Module):
    """
    Flexible U-Net that supports asymmetric encoder/decoder pruning.

    Args:
        in_ch (int): Input channels (e.g., 1 for grayscale, 3 for RGB).
        out_ch (int): Output channels (e.g., number of segmentation classes).
        enc_features (list[int]): Encoder block output channels.
        dec_features (list[int], optional): Decoder block output channels.
            If None, defaults to reversed enc_features.
        bottleneck_out (int, optional): Bottleneck output channels.
            If None, defaults to enc_features[-1] * 2.
    """
    def __init__(self,
                 in_ch=1,
                 out_ch=1,
                 enc_features=(32, 64, 128, 256),
                 dec_features=None,
                 bottleneck_out=None):
        super().__init__()

        # ---------- Encoder ----------
        self.encoders = nn.ModuleList()
        prev_ch = in_ch
        for feat in enc_features:
            self.encoders.append(DoubleConv(prev_ch, feat))
            prev_ch = feat

        # ---------- Bottleneck ----------
        if bottleneck_out is None:
            bottleneck_out = enc_features[-1] * 2
        self.bottleneck = DoubleConv(enc_features[-1], bottleneck_out)

        # ---------- Decoder ----------
        if dec_features is None:
            dec_features = list(reversed(enc_features))

        self.decoders = nn.ModuleList()
        prev_ch = bottleneck_out

        for i, feat in enumerate(dec_features):
            # Skip connection channel count (mirror encoder)
            skip_ch = enc_features[-(i + 1)]

            # Upsampling layer
            self.decoders.append(
                nn.ConvTranspose2d(prev_ch, feat, kernel_size=2, stride=2)
            )

            # After concatenation: skip + upsampled feature map
            in_ch_dec = skip_ch + feat
            self.decoders.append(DoubleConv(in_ch_dec, feat))

            prev_ch = feat  # update for next decoder block

        # ---------- Final convolution ----------
        self.final_conv = nn.Conv2d(dec_features[-1], out_ch, kernel_size=1)
        self.pool = nn.MaxPool2d(2)

        # # ---------- Channel summary (for debugging / verification) ----------
        # print("\n🧩 U-Net Architecture Summary:")
        # print(f"Input channels:  {in_ch}")
        # for i, feat in enumerate(enc_features):
        #     print(f"Encoder {i+1:<2}: in={in_ch if i==0 else enc_features[i-1]:<4} → out={feat}")
        # print(f"Bottleneck : in={enc_features[-1]:<4} → out={bottleneck_out}")
        # for i, feat in enumerate(dec_features):
        #     skip_ch = enc_features[-(i + 1)]
        #     up_ch = bottleneck_out if i == 0 else dec_features[i - 1]
        #     print(f"Decoder {i+1:<2}: skip={skip_ch:<4} + up={up_ch:<4} → out={feat}")
        # print(f"Output conv: {dec_features[-1]} → {out_ch}\n")



    # ---------- Forward ----------
    def forward(self, x):
        skip_connections = []

        # Encoder path
        for enc in self.encoders:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder path
        for idx in range(0, len(self.decoders), 2):
            x = self.decoders[idx](x)  # ConvTranspose2d (upsample)
            skip = skip_connections[idx // 2]

            # Handle small mismatches (due to pooling/upsampling)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

            # Concatenate skip connection
            x = torch.cat((skip, x), dim=1)

            # DoubleConv
            x = self.decoders[idx + 1](x)

        return torch.sigmoid(self.final_conv(x))
