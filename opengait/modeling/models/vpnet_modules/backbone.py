import torch
import torch.nn as nn
from .fts import FTS

class Stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class VPBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, stride=1):
        super().__init__()

        self.conv1 = nn.Conv3d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_ch)

        self.fts = FTS(mid_ch)

        self.conv2 = nn.Conv3d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_ch)

        self.conv3 = nn.Conv3d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_ch)

        self.relu = nn.ReLU(inplace=True)

        if in_ch != out_ch or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.fts(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)

class Backbone(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()

        self.stem = Stem()

        self.stage1 = nn.Sequential(
            VPBlock(64, 64, 64),
            VPBlock(64, 64, 64),
        )

        self.stage2 = nn.Sequential(
            VPBlock(64, 128, 128, stride=2),
            VPBlock(128, 128, 128),
        )

        self.stage3 = nn.Sequential(
            VPBlock(128, 256, 256, stride=2),
            VPBlock(256, 256, 256),
        )

        self.stage4 = nn.Sequential(
            VPBlock(256, 512, 512, stride=2),
            VPBlock(512, 512, 512),
        )

        self.final_conv = nn.Conv3d(512, feat_dim, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.final_conv(x)
        return x
