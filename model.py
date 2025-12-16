import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Basic layers
# ---------------------------
class Conv3x3(nn.Module):
    def __init__(self, cin, cout, stride=1, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, groups=groups, bias=bias)

    def forward(self, x):
        return self.conv(x)


class DWConv3x3(nn.Module):
    """Depthwise 3x3 + pointwise 1x1"""
    def __init__(self, c, bias=True):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=bias)
        self.pw = nn.Conv2d(c, c, 1, bias=bias)

    def forward(self, x):
        return self.pw(self.dw(x))


class SimpleGate(nn.Module):
    """NAFNet-style gating: split channels -> elementwise multiply"""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


# ---------------------------
# Simplified NAFBlock
# ---------------------------
class NAFBlock(nn.Module):
    """
    A simplified NAFNet block:
      - 1x1 expand
      - depthwise conv
      - SimpleGate
      - channel attention (optional light)
      - 1x1 project
      - residual with learnable scale
    """
    def __init__(self, c, expand=2, drop=0.0):
        super().__init__()
        hidden = c * expand
        self.norm1 = nn.GroupNorm(1, c)  # LayerNorm-like for conv features
        self.pw1 = nn.Conv2d(c, hidden * 2, 1)  # *2 for gate split
        self.dw = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2)
        self.gate = SimpleGate()

        # light channel attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, max(hidden // 8, 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(hidden // 8, 4), hidden, 1),
            nn.Sigmoid()
        )

        self.pw2 = nn.Conv2d(hidden, c, 1)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))

        # second FFN-like part (optional)
        self.norm2 = nn.GroupNorm(1, c)
        self.ffn = nn.Sequential(
            nn.Conv2d(c, c * 2, 1),
            SimpleGate(),
            nn.Conv2d(c, c, 1),
        )
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.norm1(x)
        y = self.pw1(y)
        y = self.dw(y)
        y = self.gate(y)              # (B, hidden, H, W)
        y = y * self.ca(y)
        y = self.pw2(y)
        y = self.drop(y)
        x = x + y * self.beta

        z = self.norm2(x)
        z = self.ffn(z)
        x = x + z * self.gamma
        return x


# ---------------------------
# Blur Uncertainty Estimator (M map)
# ---------------------------
class BlurUncertaintyNet(nn.Module):
    """
    Predict blur/confidence map M in [0,1].
    Lightweight CNN, input: RGB image.
    """
    def __init__(self, cin=3, c=32):
        super().__init__()
        self.net = nn.Sequential(
            Conv3x3(cin, c),
            nn.ReLU(inplace=True),
            Conv3x3(c, c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------
# Encoder / Decoder
# ---------------------------
class Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, target):
        x = F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(x)


class SharedEncoder(nn.Module):
    """
    Multi-scale encoder producing F1 (1x), F2 (1/2x), F3 (1/4x)
    """
    def __init__(self, base=48, blocks=(2, 2, 4)):
        super().__init__()
        self.stem = nn.Conv2d(3, base, 3, padding=1)

        self.stage1 = nn.Sequential(*[NAFBlock(base) for _ in range(blocks[0])])
        self.down1 = Down(base, base * 2)

        self.stage2 = nn.Sequential(*[NAFBlock(base * 2) for _ in range(blocks[1])])
        self.down2 = Down(base * 2, base * 4)

        self.stage3 = nn.Sequential(*[NAFBlock(base * 4) for _ in range(blocks[2])])

    def forward(self, x):
        x = self.stem(x)
        f1 = self.stage1(x)             # (B, base, H, W)
        x = self.down1(f1)
        f2 = self.stage2(x)             # (B, 2base, H/2, W/2)
        x = self.down2(f2)
        f3 = self.stage3(x)             # (B, 4base, H/4, W/4)
        return f1, f2, f3


class DeblurDecoder(nn.Module):
    """
    Deblur branch: uses skip connections, aims for faithful structure restoration.
    """
    def __init__(self, base=48, blocks=(2, 2)):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4

        self.up2 = Up(c3, c2)
        self.fuse2 = nn.Sequential(
            nn.Conv2d(c2 + c2, c2, 1),
            *[NAFBlock(c2) for _ in range(blocks[0])]
        )

        self.up1 = Up(c2, c1)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(c1 + c1, c1, 1),
            *[NAFBlock(c1) for _ in range(blocks[1])]
        )

        self.out = nn.Conv2d(c1, 3, 3, padding=1)

    def forward(self, f1, f2, f3):
        x = self.up2(f3, f2)
        x = self.fuse2(torch.cat([x, f2], dim=1))

        x = self.up1(x, f1)
        x = self.fuse1(torch.cat([x, f1], dim=1))

        return self.out(x)


class RepairDecoder(nn.Module):
    """
    Inpainting / repair branch:
      - guided by M map (blur uncertainty)
      - uses dilated conv + local attention-ish gating
    """
    def __init__(self, base=48, blocks=(2, 2)):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4

        # project M into feature channels
        self.m_proj1 = nn.Conv2d(1, c1, 1)
        self.m_proj2 = nn.Conv2d(1, c2, 1)
        self.m_proj3 = nn.Conv2d(1, c3, 1)

        # context aggregation at low-res
        self.context3 = nn.Sequential(
            nn.Conv2d(c3, c3, 3, padding=2, dilation=2, groups=c3),
            nn.Conv2d(c3, c3, 1),
            *[NAFBlock(c3) for _ in range(blocks[0])]
        )

        self.up2 = Up(c3, c2)
        self.repair2 = nn.Sequential(
            nn.Conv2d(c2 + c2, c2, 1),
            nn.Conv2d(c2, c2, 3, padding=2, dilation=2, groups=c2),
            nn.Conv2d(c2, c2, 1),
            *[NAFBlock(c2) for _ in range(blocks[0])]
        )

        self.up1 = Up(c2, c1)
        self.repair1 = nn.Sequential(
            nn.Conv2d(c1 + c1, c1, 1),
            nn.Conv2d(c1, c1, 3, padding=2, dilation=2, groups=c1),
            nn.Conv2d(c1, c1, 1),
            *[NAFBlock(c1) for _ in range(blocks[1])]
        )

        self.out = nn.Conv2d(c1, 3, 3, padding=1)

    def forward(self, f1, f2, f3, m):
        # resize M to scales
        m1 = F.interpolate(m, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        m2 = F.interpolate(m, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        m3 = F.interpolate(m, size=f3.shape[-2:], mode="bilinear", align_corners=False)

        # blur-guided gating: emphasize uncertain regions
        f1g = f1 * (1.0 + self.m_proj1(m1))
        f2g = f2 * (1.0 + self.m_proj2(m2))
        f3g = f3 * (1.0 + self.m_proj3(m3))

        x = self.context3(f3g)

        x = self.up2(x, f2g)
        x = self.repair2(torch.cat([x, f2g], dim=1))

        x = self.up1(x, f1g)
        x = self.repair1(torch.cat([x, f1g], dim=1))

        return self.out(x)


# ---------------------------
# Full model: Dual-Branch Deblur + Repair + Fusion
# ---------------------------
class DualBranchDeblurRepairNet(nn.Module):
    def __init__(self, base=48):
        super().__init__()
        self.encoder = SharedEncoder(base=base)
        self.mnet = BlurUncertaintyNet(cin=3, c=32)

        self.deblur = DeblurDecoder(base=base)
        self.repair = RepairDecoder(base=base)

    def forward(self, x):
        m = self.mnet(x)                # (B,1,H,W) in [0,1]
        f1, f2, f3 = self.encoder(x)

        id_ = self.deblur(f1, f2, f3)   # I_d
        ir  = self.repair(f1, f2, f3, m) # I_r

        # confidence-guided fusion
        iout = (1.0 - m) * id_ + m * ir
        return {"I_d": id_, "I_r": ir, "M": m, "I_out": iout}


# ---------------------------
# Quick test
# ---------------------------
if __name__ == "__main__":
    net = DualBranchDeblurRepairNet(base=48)
    x = torch.randn(2, 3, 256, 256)
    y = net(x)
    for k, v in y.items():
        print(k, v.shape)

    # 固定输入，看输出是否稳定
    x = torch.ones(1, 3, 256, 256)
    y = net(x)

    print(y["M"].min(), y["M"].max())
