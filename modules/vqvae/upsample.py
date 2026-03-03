import torch.nn as nn

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        output = nn.functional.interpolate(X, scale_factor=2.0, mode="nearest")
        output = self.conv(output)
        return output
