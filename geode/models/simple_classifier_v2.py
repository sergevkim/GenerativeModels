from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch.nn import (
    AdaptiveAvgPool2d,
    Linear,
    Module,
    NLLLoss,
    ReLU,
    Sequential,
)
from torch.optim import Adam

from geode.models.base_module import BaseModule
from geode.models.blocks import ConvBlock


class SimpleClassifierV2(BaseModule):
    def __init__(
            self,
            n_blocks: int = 6,
            n_channels: int = 1,
            n_classes: int = 10,
            hidden_dim: int = 100,
            device: torch.device = torch.device('cpu'),
            learning_rate: float = 3e-4,
        ):
        super().__init__()
        body_ordered_dict = OrderedDict()
        body_ordered_dict['block_0'] = ConvBlock(
            in_channels=n_channels,
            out_channels=hidden_dim,
            norm=True,
            pool=True,
        )

        for i in range(1, n_blocks - 1):
            body_ordered_dict[f'block_{i}'] = ConvBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                norm=True,
            )

        body_ordered_dict[f'block_{n_blocks - 1}'] = ConvBlock(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            norm=True,
            pool=True,
            act=False,
        )

        self.body = Sequential(body_ordered_dict)

        self.neck = Sequential(
            AdaptiveAvgPool2d(output_size=(1, 1)),
            Rearrange('b c h w -> b (c h w)'),
        )

        self.head = Sequential(
            Linear(
                in_features=hidden_dim,
                out_features=hidden_dim * 2,
            ),
            ReLU(),
            Linear(
                in_features=hidden_dim * 2,
                out_features=n_classes * 2,
            ),
            ReLU(),
            Linear(
                in_features=n_classes * 2,
                out_features=n_classes,
            ),
        )

        self.criterion = NLLLoss()
        self.device = device
        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.body(x)
        x = self.neck(x)
        x = self.head(x)

        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        outputs = self.forward(x)
        y_hat = F.log_softmax(outputs, dim=1)
        _, predicts = torch.max(y_hat, dim=1)
        loss = self.criterion(y_hat, y)
        accuracy = (predicts == y).float().mean()

        info = {
            'accuracy': accuracy,
            'loss': loss,
        }

        return info

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        outputs = self.forward(x)
        y_hat = F.log_softmax(outputs, dim=1)
        _, predicts = torch.max(y_hat, dim=1)
        loss = self.criterion(y_hat, y)
        accuracy = (predicts == y).float().mean()

        info = {
            'accuracy': accuracy,
            'loss': loss,
        }

        return info

    def configure_optimizers(self):
        optimizer = Adam(
            params=self.parameters(),
            lr=self.learning_rate,
        )

        return [optimizer], []

    def get_activations(self, x):
        x = self.body(x)

        return x


if __name__ == '__main__':
    model = SimpleClassifierV2()
    print(model)
