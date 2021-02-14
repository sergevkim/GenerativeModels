from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    ConvTranspose2d,
    LeakyReLU,
    Module,
    ModuleDict,
    Sequential,
)


class ConvBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        transposed: bool = False,
        activation: str = 'relu',
        negative_slope: float = 0,
    ):
        super().__init__()
        convs = ModuleDict(
            regular=Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size//2,
            ),
            transposed=Conv2d( #TODO
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size//2,
            )
        )
        block_ordered_dict = OrderedDict(
            conv=convs['transposed' if transposed else 'regular'],
            norm=BatchNorm2d(
                num_features=out_channels,
                momentum=None, #? 
            ),
            act=LeakyReLU(
                negative_slope=0.2,
                inplace=True,
            ),
        )
        self.block = Sequential(block_ordered_dict)

    def forward(self, x):
        x = self.block(x)

        return x

