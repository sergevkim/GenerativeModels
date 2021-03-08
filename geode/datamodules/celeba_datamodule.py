from typing import Tuple

from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision.transforms import (
    Compose,
    Normalize,
    Resize,
    ToTensor,
)

from geode.datamodules.base_datamodule import BaseDataModule


class CelebADataModule(BaseDataModule):
    def setup(
            self,
            val_ratio,
            new_size: Tuple[int, int] = (256, 256),
            download: bool = False,
        ):
        data_transforms = Compose([
            Resize((256, 256)),
            ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        full_dataset = CelebA(
            root=self.data_path,
            target_type='attr',
            transform=data_transforms,
            download=download,
        )


if __name__ == '__main__':
    dm = CelebADataModule(
        data_path='./data',
        batch_size=64,
        num_workers=4,
    )
    dm.setup(download=True)
    dl = dm.train_dataloader()
    print(len(dl))