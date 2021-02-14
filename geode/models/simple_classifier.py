from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    Module,
    Sequential,
)


class SimpleClassifier(Module):
    def __init__(
            self,
            n_classes: int = 10,
        ):
        super().__init__()
        body_ordered_dict = OrderedDict()
        self.body = Sequential(body_ordered_dict)

        head_ordered_dict = OrderedDict()
        self.head = Sequential(head_ordered_dict)
        
    def forward(self, x):
        x = self.body(x)
        x = self.head(x)

        return x
    
    def get_activations(self, x):
        x = self.body(x)

        return x

