from collections import OrderedDict

import einops
import torch
from torch.nn import (
    Conv2d,
    Module,
    Sequential,
    Tanh,
)

from geode.models.blocks import ConvBlock, ResidualBlock


class StarGenerator(Module):
    def __init__(
            self,
            n_residual_blocks: int = 3,
            n_channels: int = 3, #3 + one-hot n_classes
            image_size: int = 64,
        ):
        super().__init__()
        downsample_dict = OrderedDict()
        downsample_dict['conv_block_0'] = ConvBlock(
            in_channels=n_channels,
            out_channels=64,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False,
            instance_norm=True,
            act=True,
        )
        downsample_dict['conv_block_1'] = ConvBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
            instance_norm=True,
            act=True,
        )
        downsample_dict['conv_block_2'] = ConvBlock(
            in_channels=128,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
            instance_norm=True,
            act=True,
        )
        self.downsample = Sequential(downsample_dict)

        bottleneck_dict = OrderedDict()
        for i in range(n_residual_blocks):
            bottleneck_dict[f'residual_block_{i}'] = ResidualBlock(
                in_channels=256,
                out_channels=256,
            )
        self.bottleneck = Sequential(bottleneck_dict)

        upsample_dict = OrderedDict()
        upsample_dict['conv_block_0'] = ConvBlock(
            in_channels=256,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            transposed=True,
            instance_norm=True,
            act=True,
        )
        upsample_dict['conv_block_1'] = ConvBlock(
            in_channels=128,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1,
            transposed=True,
            instance_norm=True,
            act=True,
        )
        upsample_dict['last_conv'] = Conv2d(
            in_channels=64,
            out_channels=3,
            kernel_size=7,
            stride=1,
            padding=3,
        )
        upsample_dict['tanh'] = Tanh()
        self.upsample = Sequential(upsample_dict)

    def forward(
            self,
            images,
            labels,
        ):
        labels = einops.repeat(
            tensor=labels,
            pattern='b 1 -> b 1 h w',
            h=image.shape[2],
            w=image.shape[3],
        )
        x = torch.cat(
            [images, labels],
            dim=1,
        )
        x = self.downsample(x)
        x = self.bottleneck(x)
        x = self.upsample(x)

        return x


if __name__ == '__main__':
    model = StarGenerator()
    print(model)
