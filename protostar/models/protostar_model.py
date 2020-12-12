from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import (
    Module,
)
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
from torch.optim.optimizer import Optimizer


class ProtostarModel(Module):
    def __init__(
            self,
            device: torch.device,
            learning_rate: float,
            scheduler_step_size: int,
            scheduler_gamma: float,
            verbose: bool,
        ):
        super().__init__()

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        pass

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

    def configure_optimizers(
            self,
        ) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        optimizer = Adam(
            params=self.parameters(),
            lr=self.learning_rate,
        )
        scheduler = LambdaLR(
            optimizer=optimizer,
            step_size=self.step_size,
            gamma=self.gamma,
            verbose=self.verbose,
        )

        return [optimizer], [scheduler]

