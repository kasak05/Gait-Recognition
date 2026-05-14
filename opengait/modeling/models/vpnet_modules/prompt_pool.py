import torch
import torch.nn as nn
import torch.nn.functional as F

class PromptPool(nn.Module):
    def __init__(self, num_parts=16, num_tokens=256, feat_dim=256, top_k=6):
        super().__init__()
        self.top_k = top_k

        self.keys = nn.Parameter(torch.randn(num_parts, num_tokens, feat_dim))
        self.values = nn.Parameter(torch.randn(num_parts, num_tokens, feat_dim))

    def forward(self, part_feat):
        part_norm = F.normalize(part_feat, dim=-1)
        P = part_feat.shape[1]

        key_norm = F.normalize(self.keys[:P], dim=-1)
        values = self.values[:P]

        sim = torch.einsum('npc,ptc->npt', part_norm, key_norm)
        _, idx = torch.topk(sim, self.top_k, dim=-1)

        N = part_feat.shape[0]
        values = values.unsqueeze(0).expand(N, -1, -1, -1)

        idx = idx.unsqueeze(-1).expand(-1, -1, -1, values.shape[-1])
        selected = torch.gather(values, 2, idx)

        return selected, idx
