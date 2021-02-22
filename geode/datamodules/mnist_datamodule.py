from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import (
    Compose,
    Normalize,
    Resize,
    ToTensor,
)

from geode.datamodules.base_datamodule import BaseDataModule


class MNISTDataModule(BaseDataModule):
    def setup(
            self,
            val_ratio: float = 0.1,
            download: bool = False,
        ) -> None:
        data_transforms = Compose([
            Resize((64, 64)),
            ToTensor(),
            Normalize([0.5], [0.5])
        ])
        full_dataset = MNIST(
            root=self.data_path,
            train=True,
            transform=data_transforms,
            download=download,
        )
        full_size = len(full_dataset)
        val_size = int(val_ratio * full_size)
        train_size = full_size - val_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset=full_dataset,
            lengths=[train_size, val_size],
        )


if __name__ == '__main__':
    dm = MNISTDataModule(
        data_path='./data',
        batch_size=64,
        num_workers=4,
    )
    dm.setup(download=True)
    dl = dm.train_dataloader()
    print(len(dl))

