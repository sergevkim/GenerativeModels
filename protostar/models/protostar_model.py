import torch
from torch import Tensor
from torch.nn import (
    Module,
)
from torch.optim import Adam
from torch.optim.optimizer import Optimizer


class ProtostarModel(Module):
    def __init__(
            self,
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

    def training_step_end(self):
        pass

    def training_epoch_end(self):
        pass

    def validation_step(
            self,
            batch: Tensor,
            batch_idx: int,
        ) -> Tensor:
        pass

    def validation_step_end(self):
        pass

    def validation_epoch_end(self):
        pass

    def configure_optimizers(self) -> Optimizer:
        optimizer = Adam(
            params=self.parameters(),
            lr=self.learning_rate,
        )

        return optimizer

