import torch
import torch.nn as nn

class DynamicTransformer(nn.Module):
    def __init__(self, feat_dim=256, num_heads=8, num_layers=3, local_window=16, k=6):
        super().__init__()
        self.k = k
        self.local_window = local_window

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(feat_dim, num_heads, batch_first=True),
                "norm1": nn.LayerNorm(feat_dim),
                "norm2": nn.LayerNorm(feat_dim),
                "ffn": nn.Sequential(
                    nn.Linear(feat_dim, feat_dim*4),
                    nn.ReLU(),
                    nn.Linear(feat_dim*4, feat_dim)
                )
            }) for _ in range(num_layers)
        ])

    def forward(self, x):
        mask = self.build_mask(x.shape[1]).to(x.device)

        for layer in self.layers:
            attn_out, _ = layer["attn"](x, x, x, attn_mask=mask)
            x = layer["norm1"](x + attn_out)
            x = layer["norm2"](x + layer["ffn"](x))

        return x

    def build_mask(self, L):
        mask = torch.ones(L, L) * -1e9
        for i in range(L):
            if i < self.k:
                mask[i, :] = 0
            else:
                left = max(self.k, i - self.local_window)
                right = min(L, i + self.local_window)
                mask[i, left:right] = 0
                mask[i, :self.k] = 0
        return mask
