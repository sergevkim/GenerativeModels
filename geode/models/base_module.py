import abc
from abc import ABC
from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


class BaseModule(ABC, Module):
    @abc.abstractmethod
    def __init__(
            self,
        ):
        pass

    @abc.abstractmethod
    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        pass

    @abc.abstractmethod
    def training_step(
            self,
            batch: Tensor,
            batch_idx: int,
        ) -> Tensor:
        pass

    def training_step_end(
            self,
            batch_idx: int,
        ):
        pass

    def training_epoch_end(
            self,
            epoch_idx: int,
        ):
        pass

    @abc.abstractmethod
    def validation_step(
            self,
            batch: Tensor,
            batch_idx: int,
        ) -> Tensor:
        pass

    def validation_step_end(
            self,
            batch_idx: int,
        ):
        pass

    def validation_epoch_end(
            self,
            epoch_idx: int,
        ):
        pass

    @abc.abstractmethod
    def configure_optimizers(
            self,
        ) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        pass
