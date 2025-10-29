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


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=(32, 64, 128, 256)):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Encoder
        for feat in features:
            self.encoders.append(DoubleConv(in_ch, feat))
            in_ch = feat

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Decoder
        for feat in reversed(features):
            self.decoders.append(
                nn.ConvTranspose2d(feat*2, feat, kernel_size=2, stride=2)
            )
            self.decoders.append(DoubleConv(feat*2, feat))

        self.final_conv = nn.Conv2d(features[0], out_ch, kernel_size=1)
# class UNet(nn.Module):
#     def __init__(
#         self,
#         in_ch: int,
#         out_ch: int,
#         encoder_features: list,
#         decoder_features: list = None,
#         bottleneck_features: int = None
#     ):
#         super().__init__()

#         if decoder_features is None:
#             decoder_features = encoder_features[::-1]  # fallback for baseline symmetry
#         if bottleneck_features is None:
#             bottleneck_features = encoder_features[-1] * 2  # typical UNet doubling rule

#         self.encoders = nn.ModuleList()
#         self.decoders = nn.ModuleList()

#         # --- Encoder ---
#         for i, out_f in enumerate(encoder_features):
#             in_f = in_ch if i == 0 else encoder_features[i - 1]
#             self.encoders.append(DoubleConv(in_f, out_f))

#         # --- Bottleneck ---
#         self.bottleneck = DoubleConv(encoder_features[-1], bottleneck_features)

#         # --- Decoder ---
#         for i in range(len(decoder_features)):
#             up_in = bottleneck_features if i == 0 else decoder_features[i - 1]
#             self.decoders.append(nn.ConvTranspose2d(up_in, decoder_features[i], kernel_size=2, stride=2))
#             self.decoders.append(DoubleConv(decoder_features[i] * 2, decoder_features[i]))  # skip connection concat

#         self.final_conv = nn.Conv2d(decoder_features[-1], out_ch, kernel_size=1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip_connections = []
        for enc in self.encoders:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.decoders), 2):
            x = self.decoders[idx](x)  # upconv
            skip = skip_connections[idx//2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = self.decoders[idx+1](x)

        return torch.sigmoid(self.final_conv(x))
