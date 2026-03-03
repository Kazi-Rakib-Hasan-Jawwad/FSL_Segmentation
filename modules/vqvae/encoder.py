import torch.nn as nn
from argparse import Namespace
from .resnet_block import ResNetBlock
from .downsample import Downsample
from .non_local_block import NonLocalBlock

class Encoder(nn.Module):
    def __init__(self, config: Namespace):
        super().__init__()
        self.config = config
        self.parse_config()
        self.init_blocks()

    def parse_config(self):
        self.conv1_kwargs = self.config.encoder.get("conv1_kwargs")
        self.resb1_kwargs = self.config.encoder.get("resb1_kwargs")
        self.nlcb1_kwargs = self.config.encoder.get("nlcb1_kwargs")
        self.resb2_kwargs = self.config.encoder.get("resb2_kwargs")
        self.norm_last_kwargs = self.config.encoder.get("norm_last_kwargs")
        self.actv_last_kwargs = self.config.encoder.get("actv_last_kwargs")
        self.conv_last_kwargs = self.config.encoder.get("conv_last_kwargs")
        self.downsample_blocks_config = self.config.encoder.get("downsample_blocks")
        self.num_res_blocks_per_downsample = self.downsample_blocks_config.get("num_res_blocks_per_downsample")

    def init_blocks(self):
        self.conv1 = nn.Conv2d(**self.conv1_kwargs)
        self.resb1 = ResNetBlock(**self.resb1_kwargs)
        self.nlcb1 = NonLocalBlock(**self.nlcb1_kwargs)
        self.resb2 = ResNetBlock(**self.resb2_kwargs)
        self.norm_last = nn.GroupNorm(**self.norm_last_kwargs)
        self.actv_last = nn.SiLU(**self.actv_last_kwargs)
        self.conv_last = nn.Conv2d(**self.conv_last_kwargs)

        self.downsample_blocks = nn.ModuleList()
        self.res_blocks = nn.ModuleList()

        in_channel = self.conv1.out_channels
        for idx, out_channel in enumerate(self.downsample_blocks_config["channels"]):
            for _ in range(self.num_res_blocks_per_downsample):
                self.res_blocks.append(ResNetBlock(
                    in_channels=in_channel, 
                    out_channels=out_channel,
                    num_groups=self.downsample_blocks_config["res_block_norm_num_groups"],
                    dropout = self.downsample_blocks_config["dropout"]))
                in_channel = out_channel
            self.downsample_blocks.append(Downsample(in_channel, in_channel))

    def forward(self, X):
        output = X
        output = self.conv1(output)

        for downsample_idx, downsample_block in enumerate(self.downsample_blocks):
            for res_block_idx in range(self.num_res_blocks_per_downsample):
                output = self.res_blocks[downsample_idx * self.num_res_blocks_per_downsample + res_block_idx](output)
            output = downsample_block(output)

        output = self.resb1(output)
        output = self.nlcb1(output)
        output = self.resb2(output)
        output = self.norm_last(output)
        output = self.actv_last(output)
        output = self.conv_last(output)

        return output
