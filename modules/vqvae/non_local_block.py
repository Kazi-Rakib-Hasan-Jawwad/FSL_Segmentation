import torch
import torch.nn as nn
import torch.nn.functional as F

class NonLocalBlock(nn.Module):
    def __init__(self, channels, num_groups=32):
        super().__init__()
        self.theta = nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                stride=1,
                padding=0)
        self.phi = nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                stride=1,
                padding=0)
        self.g = nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                stride=1,
                padding=0)
        self.conv_last = nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                stride=1,
                padding=0)
        self.norm = nn.GroupNorm(
                num_groups = num_groups,
                num_channels = channels,
                eps = 1e-6)


    def forward(self, X):
        output = self.norm(X)
        q = self.theta(output)
        k = self.phi(output)
        v = self.g(output)

        B, C, H, W = output.shape
        q = q.flatten(start_dim=2).permute(0,2,1)
        k = k.flatten(start_dim=2)
        w = torch.bmm(q, k) * (C ** (-0.5))
        w = F.softmax(w, dim=2).permute(0,2,1)

        v = v.flatten(start_dim=2)
        output = torch.bmm(v, w).reshape(B, C, H, W)
        output = self.conv_last(output)

        return X + output
