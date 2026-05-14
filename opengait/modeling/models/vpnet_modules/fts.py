import torch
import torch.nn as nn

class FTS(nn.Module):
    def __init__(self, channels, seg=4):
        super().__init__()
        self.seg = seg
        self.channels = channels

    def forward(self, x):
        N, C, T, H, W = x.shape
        seg_c = C // self.seg

        out = x.clone()
        out[:, :seg_c, 1:] = x[:, :seg_c, :-1]
        out[:, :seg_c, 0]  = x[:, :seg_c, -1]

        return out
