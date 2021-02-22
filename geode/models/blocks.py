from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    ConvTranspose2d,
    MaxPool2d,
    Module,
    ModuleDict,
    ReLU,
    Sequential,
)


class ConvBlock(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            transposed: bool = False,
            norm: bool = False,
            pool: bool = False,
        ):
        super().__init__()
        block_ordered_dict = OrderedDict()
        block_ordered_dict['conv'] = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,
        ) if transposed else ConvTranspose2d( #TODO
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,
        )
        if norm:
            block_ordered_dict['norm'] = BatchNorm2d(
                num_features=out_channels,
                momentum=None, #?
            )
        if pool:
            block_ordered_dict['pool'] = MaxPool2d(
                kernel_size=2,
            )
        block_ordered_dict['act'] = ReLU()
        self.block = Sequential(block_ordered_dict)

    def forward(self, x):
        x = self.block(x)

        return x

