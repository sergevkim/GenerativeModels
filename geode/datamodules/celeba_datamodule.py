from typing import Tuple

from torch.utils.data import DataLoader, random_split
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
            val_ratio: float = 0.9,
            new_size: Tuple[int, int] = (256, 256),
            download: bool = False,
        ):
        data_transforms = Compose([
            Resize((64, 64)),
            ToTensor(),
            #Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        full_dataset = CelebA(
            root=self.data_path,
            target_type='identity',
            transform=data_transforms,
            download=download,
        )

        full_size = len(full_dataset)
        val_size = int(val_ratio * full_size)
        train_size = full_size - val_size

        self.train_dataset, self.val_dataset = random_split(
            dataset=full_dataset,
            lengths=[train_size, val_size],
        )


if __name__ == '__main__':
    dm = CelebADataModule(
        data_path='./data/celeba',
        batch_size=64,
        num_workers=4,
    )
    dm.setup(download=True)
    dl = dm.train_dataloader()
    print(len(dl))
