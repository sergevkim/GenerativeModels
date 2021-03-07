from collections import OrderedDict

import torch
from torch.nn import (
    Conv2d,
    Module,
    Sequential,
)

from geode.models.blocks import ConvBlock


class StarCritic(Module):
    def __init__(
            self,
            n_hidden_blocks: int = 3,
            n_channels: int = 3,
            n_classes: int = 10, #TODO think about n_classes
            image_size: int = 64,
        ):
        super().__init__()
        body_dict = OrderedDict()
        body_dict['conv_block_0'] = ConvBlock(
            in_channels=n_channels,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1,
            act=True,
        )
        for i in range(n_hidden_blocks):
            body_dict[f'conv_block_{i}'] = ConvBlock(
                in_channels=64 * 2 ** i,
                out_channels=64 * 2 ** (i + 1),
                kernel_size=4,
                stride=2,
                padding=1,
                act=True,
            )
        self.body = Sequential(body_dict)

        self.head_src = Conv2d(
            in_channels=64 * 2 ** n_hidden_blocks,
            out_channels=1,
            kernel_size=3,
            padding=1,
        )
        self.head_cls = Conv2d(
            in_channels=64 * 2 ** n_hidden_blocks,
            out_channels=n_classes,
            kernel_size=image_size // 2 ** (n_hidden_blocks + 1),
        )

    def forward(
            self,
            x,
        ):
        h = self.body(x)
        src_predict = self.head_src(h)
        cls_predict = self.head_cls(h)

        predicts = {
            'src': src_predict,
            'cls': cls_predict,
        }

        return predicts


if __name__ == '__main__':
    model = StarCritic()
    print(model)
