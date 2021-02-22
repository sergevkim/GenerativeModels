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


class SimpleClassifier(BaseModule):
    def __init__(
            self,
            n_channels: int = 1,
            n_classes: int = 10,
            hidden_dim: int = 100,
            device: torch.device = torch.device('cpu'),
            learning_rate: float = 3e-4,
        ):
        super(SimpleClassifier, self).__init__()
        body_ordered_dict = OrderedDict(
            block_0=ConvBlock(
                in_channels=n_channels,
                out_channels=hidden_dim,
                norm=True,
                pool=True,
            ),
            block_1=ConvBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                norm=True,
                pool=True,
            ),
            block_2=ConvBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                norm=True,
                pool=True,
            ),
        )
        self.body = Sequential(body_ordered_dict)

        neck_ordered_dict = OrderedDict(
            avg_pool=AdaptiveAvgPool2d(output_size=(1, 1)),
            rearrange=Rearrange('b c h w -> b (c h w)'),
        )
        self.neck = Sequential(neck_ordered_dict)

        head_ordered_dict = OrderedDict(
            linear_0=Linear(
                in_features=hidden_dim,
                out_features=hidden_dim,
            ),
            act_0=ReLU(),
            linear_1=Linear(
                in_features=hidden_dim,
                out_features=hidden_dim,
            ),
            act_1=ReLU(),
            linear_2=Linear(
                in_features=hidden_dim,
                out_features=n_classes,
            ),
        )
        self.head = Sequential(head_ordered_dict)

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
        loss = self.criterion(y_hat, y)

        info = {
            'loss': loss,
            'outputs': outputs,
        }

        return info

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, 0)

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
    model = SimpleClassifier()
    print(model)
