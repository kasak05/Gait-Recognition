import torch
import torch.nn as nn

class HorizontalPooling(nn.Module):
    def __init__(self, num_parts):
        super().__init__()
        self.num_parts = num_parts

    def forward(self, x):
        x = x.mean(dim=4)
        parts = torch.chunk(x, self.num_parts, dim=3)

        part_feats = []
        for p in parts:
            part_feats.append(p.mean(dim=3))

        return torch.stack(part_feats, dim=1)
