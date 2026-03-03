import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups, dropout):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) if self.in_channels != self.out_channels else nn.Identity()

    def forward(self, X):
        output = X
        output = self.norm1(output)
        output = F.silu(output)
        output = self.conv1(output)

        output = self.norm2(output)
        output = F.silu(output)
        output = self.dropout(output)
        output = self.conv2(output)
        return self.shortcut(X) + output
