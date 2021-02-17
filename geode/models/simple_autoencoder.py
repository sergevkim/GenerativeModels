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
from torch.optim import Adam

from geode.models.base_module import BaseModule
from geode.models.blocks import ConvBlock


class SimpleAutoencoder(BaseModule):
    def __init__(
            self,
            hidden_dim: int = 10,
            n_blocks: int = 3,
            device: torch.device = torch.device('cpu'),
            learning_rate: float = 3e-4,
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
        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

    def training_step(self, batch, batch_idx):
        images, _ = batch
        images.to(self.device)

        predicts = self.forward(images)
        loss = self.criterion(
            images,
            predicts,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = Adam(
            params=self.parameters(),
            lr=self.learning_rate,
        )

        return [optimizer], []

    def get_latent_features(self, x):
        x = self.encoder(x)

        return x


if __name__ == '__main__':
    model = SimpleAutoencoder()
    print(model)
