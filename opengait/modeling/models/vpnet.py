import torch
import torch.nn as nn
import torch.nn.functional as F

from .vpnet_modules.backbone import Backbone
from .vpnet_modules.hp import HorizontalPooling
from .vpnet_modules.prompt_pool import PromptPool
from .vpnet_modules.dynamic_transformer import DynamicTransformer
from .vpnet_modules.width_motion_encoder import WidthMotionEncoder


class VPNet(nn.Module):
    def __init__(self, num_parts=16, feat_dim=256, num_classes=74):
        super().__init__()

        self.backbone = Backbone(feat_dim)
        self.hp = HorizontalPooling(num_parts)

        self.prompt_pool = PromptPool(
            num_parts=num_parts,
            feat_dim=feat_dim
        )

        self.motion_encoder = WidthMotionEncoder(
            feat_dim=feat_dim
        )

        self.transformer = DynamicTransformer(
            feat_dim=feat_dim,
            k=6
        )

        self.bnneck = nn.BatchNorm1d(feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):

        # backbone features
        feat = self.backbone(x)

        # horizontal pooling
        # (N,P,C,T)
        part_feat = self.hp(feat)

        # visual prompts
        part_mean = part_feat.mean(dim=-1)
        prompts, _ = self.prompt_pool(part_mean)

        # motion prompt
        motion_prompt = self.motion_encoder(part_feat)

        # stabilize motion prompt
        motion_prompt = 0.05 * motion_prompt

        # expand to match parts
        motion_prompt = motion_prompt.unsqueeze(1).unsqueeze(2)

        motion_prompt = motion_prompt.repeat(
            1,
            part_feat.shape[1],
            1,
            1
        )

        # (N,P,C,T) -> (N,P,T,C)
        part_feat = part_feat.permute(0, 1, 3, 2)

        # concatenate prompts + sequence
        seq = torch.cat(
            [
                prompts,
                motion_prompt,
                part_feat
            ],
            dim=2
        )

        N, P, L, C = seq.shape

        seq = seq.view(
            N * P,
            L,
            C
        )

        # transformer
        out = self.transformer(seq)

        # average over sequence
        out = out.mean(dim=1)

        # reshape back
        out = out.view(N, P, C)

        # average over parts
        out = out.mean(dim=1)

        # bnneck
        feat = self.bnneck(out)

        # normalize feature
        feat = F.normalize(
            feat,
            dim=1,
            eps=1e-8
        )

        # classifier
        logits = self.classifier(feat)

        return feat, logits
