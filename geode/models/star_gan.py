from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    Module,
    MSELoss,
    Sequential,
)
from torch.optim import Adam

from geode.models.base_module import BaseModule
from geode.models.blocks import ConvBlock
from geode.models.star_critic import StarCritic
from geode.models.star_generator import StarGenerator
from geode.utils import MetricCalculator


class StarGAN(BaseModule):
    def __init__(
            self,
            hidden_dim: int = 10,
            n_channels: int = 1,
            n_blocks: int = 3,
            device: torch.device = torch.device('cpu'),
            learning_rate: float = 3e-3,
        ):
        super().__init__()
        self.G = StarGenerator()
        self.D = StarCritic()

        self.G_rec_criterion = MSELoss()
        self.G_adv_criterion = MSELoss()
        self.G_cls_criterion = MSELoss()
        self.D_adv_criterion = MSELoss()
        self.D_cls_criterion = MSELoss()

    def forward(self, image, label):
        return self.G.forward(image, label)

    def training_step(
            self,
            batch,
            batch_idx,
            optimizer_idx,
        ):
        original_images = batch['images']
        original_labels = batch['original_labels']
        target_labels = batch['target_labels']
        #TODO sampling target labels not in dataset.__getitem__ but here
        images = images.to(self.device)
        original_labels = original_labels.to(self.device)
        target_labels = target_labels.to(self.device)

        if optimizer_idx == 0:  #generator step
            generated_images = self.G.forward(
                images=original_images,
                labels=target_labels,
            )
            reconstructed_images = self.G.forward(
                images=generated_images,
                labels=original_labels,
            )
            predicts = self.D.forward(
                images=generated_images,
            )
            src_predicts = predicts['src']
            cls_predicts = predicts['cls']

            rec_loss = self.G_rec_criterion(
                reconstructed_images,
                original_images,
            )
            adv_loss = self.G_adv_criterion(
                src_predicts,
                torch.ones_like(src_predicts),
            )
            cls_loss = self.G_cls_criterion(
                cls_predicts,
                target_labels,
            )
            loss = rec_loss + adv_loss + cls_loss #TODO coefs
            info = {
                'loss': loss,
                'rec_loss': rec_loss,
                'adv_loss': adv_loss,
                'cls_loss': cls_loss,
                'fid': fid,
            }
        elif optimizer_idx == 1:  # discriminator step
            generated_images = self.G.forward(
                images=original_images,
                labels=target_labels,
            )
            fake_predicts = self.D.forward(
                images=generated_images,
            )
            src_fake_predicts = fake_predicts['src']
            cls_fake_predicts = fake_predicts['cls']
            real_predicts = self.D.forward(
                images=original_images,
            )
            src_real_predicts = real_predicts['src']
            cls_real_predicts = real_predicts['cls']

            adv_fake_loss = self.D_adv_criterion(
                src_fake_predicts,
                torch.zeros_like(src_predicts),
            )
            cls_target_loss = self.D_cls_criterion(
                cls_fake_predicts,
                target_labels,
            )
            adv_real_loss = self.D_adv_criterion(
                src_real_predicts,
                torch.ones_like(src_predicts),
            )
            cls_original_loss = self.D_cls_criterion(
                cls_real_predicts,
                original_labels,
            )
            loss = (
                adv_fake_loss +
                adv_real_loss +
                cls_fake_loss +
                cls_real_loss
            )

            info = {
                'loss': loss,
                'adv_fake_loss': adv_fake_loss,
                'adv_real_loss': adv_real_loss,
                'cls_fake_loss': cls_fake_loss,
                'cls_real_loss': cls_real_loss,
            }

        return info

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, 0)

    def configure_optimizers(self):
        G_optimizer = Adam(
            params=self.G.parameters(),
            lr=self.learning_rate,
        )
        D_optimizer = Adam(
            params=self.D.parameters(),
            lr=self.learning_rate,
        )

        return [G_optimizer, D_optimizer], []

    def generate(self, image, label):
        pass


if __name__ == '__main__':
    model = StarGAN()
    print(model)
