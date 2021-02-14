from collections import OrderedDict
from typing import Tuple

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
    MSELoss,
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


class SimpleAutoencoder(Module):
    def __init__(
        self,
        hidden_dim: int = 10,
        n_blocks: int = 3,
    ):
        super().__init__()
        encoder_ordered_dict = OrderedDict()
        encoder_ordered_dict['block_0'] = ConvBlock(
            in_channels=3,
            out_channels=hidden_dim,
        )
        for i in range(1, n_blocks):
            encoder_ordered_dict[f'block_{i}'] = ConvBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
            )
        self.encoder = Sequential(encoder_ordered_dict)

        decoder_ordered_dict = OrderedDict()
        for i in range(n_blocks - 1):
            decoder_ordered_dict[f'block_{i}'] = ConvBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
            )
        decoder_ordered_dict[f'block_{n_blocks - 1}'] = ConvBlock(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
        )
        self.decoder = Sequential(decoder_ordered_dict)

        self.criterion = MSELoss()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

    def training_step(
        self,
        batch,
        batch_idx,
    ):
        images, _ = batch
        images.to(device)

        predicts = self.forward(images)
        loss = self.criterion(
            images,
            predicts,
        )

        return loss

    def validation_step(
        self,
        batch,
        batch_idx,
    ):
        return self.training_step(batch, batch_idx)

    def configure_optimizers(
        self,
    ):
        self.optimizer = 
        return
    
    def get_latent_features(self, x):
        x = self.encoder(x)

        return x

