# from https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/datasets/full_omniglot.py

#!/usr/bin/env python3

import os

import sklearn.model_selection as model_selection
import torch
import tqdm
from torch.utils.data import (
    ConcatDataset,
    Dataset,
    Subset,
)
from torchvision.datasets.omniglot import Omniglot
from torchvision.transforms import (
    Compose,
    Normalize,
    Resize,
    ToTensor,
)

from geode.datamodules.base_datamodule import BaseDataModule
from geode.datamodules.utils import InvertImage


class OmniglotDataset(Dataset):
    """

    [[Source]]()

    **Description**

    This class provides an interface to the Omniglot dataset.

    The Omniglot dataset was introduced by Lake et al., 2015.
    Omniglot consists of 1623 character classes from 50 different alphabets, each containing 20 samples.
    While the original dataset is separated in background and evaluation sets,
    this class concatenates both sets and leaves to the user the choice of classes splitting
    as was done in Ravi and Larochelle, 2017.
    The background and evaluation splits are available in the `torchvision` package.

    **References**

    1. Lake et al. 2015. “Human-Level Concept Learning through Probabilistic Program Induction.” Science.
    2. Ravi and Larochelle. 2017. “Optimization as a Model for Few-Shot Learning.” ICLR.

    **Arguments**

    * **root** (str) - Path to download the data.
    * **transform** (Transform, *optional*, default=None) - Input pre-processing.
    * **target_transform** (Transform, *optional*, default=None) - Target pre-processing.
    * **download** (bool, *optional*, default=False) - Whether to download the dataset.

    **Example**
    ~~~python
    omniglot = FullOmniglot(root='./data',
                            transform=transforms.Compose([
                                transforms.Resize(28, interpolation=LANCZOS),
                                transforms.ToTensor(),
                                lambda x: 1.0 - x]),
                            download=True)
    ~~~

    """

    def __init__(self, root, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        # Set up both the background and eval dataset
        omni_background = Omniglot(self.root, background=True, download=download)
        # Eval labels also start from 0.
        # It's important to add 964 to label values in eval so they don't overwrite background dataset.
        omni_evaluation = Omniglot(self.root,
                                   background=False,
                                   download=download,
                                   target_transform=lambda x: x + len(omni_background._characters))

        self.dataset = ConcatDataset((omni_background, omni_evaluation))
        self._bookkeeping_path = os.path.join(self.root, 'omniglot-bookkeeping.pkl')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, character_class = self.dataset[item]
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            character_class = self.target_transform(character_class)

        return image, character_class


class OmniglotDataModule(BaseDataModule):
    def __init__(
            self,
            data_path,
            batch_size: int,
            num_workers: int,
        ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(
            self,
            val_ratio: float = 0.1,
            download: bool = False
        ) -> None:
        data_transforms = Compose([
            Resize((64, 64)),
            ToTensor(),
            InvertImage(),
            Normalize([0.5], [0.5]),
        ])
        full_dataset = OmniglotDataset(
            root=self.data_path,
            transform=data_transforms,
            download=download,
        )
        '''
        full_size = len(full_dataset)
        val_size = int(val_ratio * full_size)
        train_size = full_size - val_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset=full_dataset,
            lengths=[train_size, val_size],
        )
        '''
        labels = []
        for _, label in tqdm.tqdm(full_dataset):
            labels.append(label)

        indices_train, indices_test, _, _ = model_selection.train_test_split(
            list(range(len(labels))),
            labels,
            test_size=val_ratio,
            stratify=labels,
        )

        self.train_dataset = Subset(full_dataset, indices=indices_train)
        self.test_dataset = Subset(full_dataset, indices=indices_test)


if __name__ == '__main__':
    dm = OmniglotDataModule(
        data_path='./data',
        batch_size=64,
        num_workers=4,
    )
    dm.setup(download=True)
    dl = dm.train_dataloader()
    print(len(dl))

