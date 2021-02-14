from typing import List, Tuple

import torch
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler, StepLR
from torch.optim.optimizer import Optimizer
from torchaudio.transforms import MelSpectrogram

from protostar.models import BaseModule


class ProtostarModel(BaseModule):
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
            optimizer_idx: int,
        ) -> Tensor:
        pass

    def validation_step(
            self,
            batch: Tensor,
            batch_idx: int,
        ) -> Tensor:
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
        )

        return [optimizer], [scheduler]

