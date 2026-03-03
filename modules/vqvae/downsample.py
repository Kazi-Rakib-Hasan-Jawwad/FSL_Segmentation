import torch.nn as nn

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, X):
        output = nn.functional.pad(X, (0, 1, 0, 1), mode = "constant", value = 0)
        output = self.conv(output)
        return output
