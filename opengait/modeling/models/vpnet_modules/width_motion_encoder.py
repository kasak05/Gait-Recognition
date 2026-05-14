import torch
import torch.nn as nn
import torch.nn.functional as F


class WidthMotionEncoder(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()

        self.proj = nn.Linear(
            4,
            feat_dim
        )

    def forward(self, part_feat):
        """
        part_feat: (N,P,C,T)
        """

        # select arms + legs
        selected_parts = torch.cat([
            part_feat[:, 4:8],
            part_feat[:, 12:16]
        ], dim=1)

        # average over feature dimension
        # width signal
        width = selected_parts.mean(
            dim=2
        )   # (N,8,T)

        # average over limbs
        width_signal = width.mean(
            dim=1
        )   # (N,T)

        T = width_signal.shape[1]

        # if sequence too short
        if T < 2:
            motion_descriptor = torch.zeros(
                width_signal.shape[0],
                4,
                device=width_signal.device
            )

            return self.proj(
                motion_descriptor
            )

        # temporal width difference
        width_diff = (
            width_signal[:, 1:]
            -
            width_signal[:, :-1]
        )

        # FEATURE 1: speed
        speed = width_diff.abs().mean(
            dim=1,
            keepdim=True
        )

        # FEATURE 2: rhythm
        if width_diff.shape[1] > 1:
            rhythm = width_diff.std(
                dim=1,
                keepdim=True,
                unbiased=False
            )
        else:
            rhythm = torch.zeros(
                width_diff.shape[0],
                1,
                device=width_diff.device
            )

        # FEATURE 3: variance
        variance = width_signal.var(
            dim=1,
            keepdim=True,
            unbiased=False
        )

        # FEATURE 4: turning
        if width_diff.shape[1] > 1:
            turning = torch.sign(
                width_diff[:,1:]
            ) * torch.sign(
                width_diff[:,:-1]
            )
            
            turning = turning.mean(
                dim=1,
                keepdim=True
            )
        else:
            turning = torch.zeros(
                width_diff.shape[0],
                1,
                device=width_diff.device
            )

        # concatenate descriptors
        motion_descriptor = torch.cat([
            speed,
            rhythm,
            variance,
            turning
        ], dim=1)

        # clamp extreme values
        motion_descriptor = torch.clamp(
            motion_descriptor,
            min=-100.0,
            max=100.0
        )

        # remove NaN/Inf
        motion_descriptor = torch.nan_to_num(
            motion_descriptor,
            nan=0.0,
            posinf=100.0,
            neginf=-100.0
        )

        # project to feature space
        motion_descriptor = F.normalize(
            motion_descriptor,
            dim=1,
            eps=1e-8
        )
        motion_prompt = self.proj(
            motion_descriptor
        )

        return motion_prompt
