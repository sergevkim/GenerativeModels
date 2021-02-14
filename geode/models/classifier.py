from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    Module,
    Sequential,
)


class Classifier(Module):
    def __init__(
            self,
            n_classes: int = 10,
        ):
        super().__init__()
        body_ordered_dict = OrderedDict()
        self.body = Sequential(body_ordered_dict)

        head_ordered_dict = OrderedDict()
        self.head = Sequential(head_ordered_dict)
        
    def forward(self, x):
        x = self.body(x)
        x = self.head(x)

        return x
    
    def get_activations(self, x):
        x = self.body(x)

        return x


class ConvBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        transposed: bool = False,
    ):
        super().__init__()
        block_ordered_dict = OrederedDict(
            conv=Conv2d(
                in_channels,
                out_channels,
                kernel_size
            ),
            norm=BatchNorm2d(),
            act=LeakyReLU(0.2),
        )
        self.block = Sequential(block_ordered_dict)

    def forward(self, x):
        x = self.block(x)

        return x


class AutoEncoder(Module):
    def __init__(
        self,
        n_blocks: int,
    ):
        super().__init__()
        encoder_ordered_dict = OrderedDict()
        for i in range(n_blocks):
            encoder_ordered_dict[f'block_{i}'] = ConvBlock(
                in_channels=,
                out_channels=,
            )
        self.encoder = Sequential(encoder_ordered_dict)

        decoder_ordered_dict = OrderedDict()
