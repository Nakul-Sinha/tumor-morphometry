"""ResNet18-UNet with 7-channel stride-2 output head.

Channels: 0-3 class density (x100 scaled), 4 center heatmap, 5 log-area map, 6 elongation map.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UNetR18(nn.Module):
    def __init__(self, out_ch=7, pretrained=True):
        super().__init__()
        import torchvision

        try:
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            enc = torchvision.models.resnet18(weights=weights)
            self.pretrained_ok = pretrained
        except Exception as e:  # network hiccup -> random init
            print(f"[model] pretrained download failed ({e}); using random init", flush=True)
            enc = torchvision.models.resnet18(weights=None)
            self.pretrained_ok = False

        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)  # /2, 64
        self.pool = enc.maxpool                                   # /4
        self.l1 = enc.layer1                                      # /4, 64
        self.l2 = enc.layer2                                      # /8, 128
        self.l3 = enc.layer3                                      # /16, 256
        self.l4 = enc.layer4                                      # /32, 512

        self.d4 = _block(512 + 256, 256)   # /16
        self.d3 = _block(256 + 128, 128)   # /8
        self.d2 = _block(128 + 64, 64)     # /4
        self.d1 = _block(64 + 64, 32)      # /2
        self.head = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        c1 = self.stem(x)          # /2
        c2 = self.l1(self.pool(c1))  # /4
        c3 = self.l2(c2)           # /8
        c4 = self.l3(c3)           # /16
        c5 = self.l4(c4)           # /32

        u = F.interpolate(c5, size=c4.shape[-2:], mode="bilinear", align_corners=False)
        u = self.d4(torch.cat([u, c4], 1))
        u = F.interpolate(u, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        u = self.d3(torch.cat([u, c3], 1))
        u = F.interpolate(u, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        u = self.d2(torch.cat([u, c2], 1))
        u = F.interpolate(u, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        u = self.d1(torch.cat([u, c1], 1))
        return self.head(u)
