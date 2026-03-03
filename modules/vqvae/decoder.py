from argparse import Namespace
import torch.nn as nn
from .resnet_block import ResNetBlock
from .non_local_block import NonLocalBlock
from .upsample import Upsample

class Decoder(nn.Module):
    def __init__(self, config: Namespace):
        super().__init__()
        self.config = config
        self.parse_config()
        self.init_blocks()

    def parse_config(self):
        self.conv1_kwargs = self.config.decoder.get("conv1_kwargs")
        self.resb1_kwargs = self.config.decoder.get("resb1_kwargs")
        self.nlcb1_kwargs = self.config.decoder.get("nlcb1_kwargs")
        self.resb2_kwargs = self.config.decoder.get("resb2_kwargs")
        self.norm_last_kwargs = self.config.decoder.get("norm_last_kwargs")
        self.actv_last_kwargs = self.config.decoder.get("actv_last_kwargs")
        self.conv_last_kwargs = self.config.decoder.get("conv_last_kwargs")
        self.upsample_blocks_config = self.config.decoder.get("upsample_blocks")
        self.num_res_blocks_per_upsample = self.upsample_blocks_config["num_res_blocks_per_upsample"]

    def init_blocks(self):
        self.conv1 = nn.Conv2d(**self.conv1_kwargs)
        self.resb1 = ResNetBlock(**self.resb1_kwargs)
        self.nlcb1 = NonLocalBlock(**self.nlcb1_kwargs)
        self.resb2 = ResNetBlock(**self.resb2_kwargs)

        self.norm_last = nn.GroupNorm(**self.norm_last_kwargs)
        self.actv_last = nn.SiLU(**self.actv_last_kwargs)
        self.conv_last = nn.Conv2d(**self.conv_last_kwargs)

        self.tanh = nn.Tanh()

        self.upsample_blocks = nn.ModuleList()
        self.res_blocks = nn.ModuleList()

        in_channel = self.resb2.out_channels
        for idx, out_channel in enumerate(self.upsample_blocks_config["channels"]):
            for _ in range(self.num_res_blocks_per_upsample):
                self.res_blocks.append(ResNetBlock(
                    in_channels=in_channel, 
                    out_channels=out_channel,
                    num_groups=self.upsample_blocks_config["res_block_norm_num_groups"],
                    dropout = self.upsample_blocks_config["dropout"]))
                in_channel = out_channel
            self.upsample_blocks.append(Upsample(in_channel, in_channel))

    def forward(self, X):
        output = X
        output = self.conv1(output)
        output = self.resb1(output) 
        output = self.nlcb1(output)
        output = self.resb2(output)

        for upsample_idx, upsample_block in enumerate(self.upsample_blocks):
            for res_block_idx in range(self.num_res_blocks_per_upsample):
                output = self.res_blocks[upsample_idx * self.num_res_blocks_per_upsample + res_block_idx](output)
            output = upsample_block(output)

        output = self.norm_last(output)
        output = self.actv_last(output)
        output = self.conv_last(output)
        output = self.tanh(output)
        return output
