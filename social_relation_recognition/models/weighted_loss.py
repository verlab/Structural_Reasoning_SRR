"""
    Loss with learnable weights for each classifier.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedLoss(nn.Module):

    def __init__(self, weights_class, type_loss):

        super(WeightedLoss, self).__init__()

        # Attributes
        self.type_loss = type_loss
        self.weights_class = weights_class

        # Parameters
        if(self.type_loss == 'weighted'):
            self.weights_loss = nn.Parameter(torch.tensor(1.))

    def forward(self, outputs_1, outputs_2, targets):
        """Forward step."""

        loss_1 = F.cross_entropy(outputs_1, targets, weight=self.weights_class)
        loss_2 = F.cross_entropy(outputs_2, targets, weight=self.weights_class)

        if(self.type_loss == 'sum'):
            return loss_1 + loss_2

        else:
            return (loss_1 * self.weights_loss) + (loss_2 / self.weights_loss)
