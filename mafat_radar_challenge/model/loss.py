import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def nll_loss(output, target):
    return F.nll_loss(output, target)


def binary_cross_entropy_with_logits(output, target, pos_weight=1.0):
    output_shape = list(output.size())
    weight = torch.tensor(np.ones(output_shape[1]) * pos_weight)
    return F.binary_cross_entropy_with_logits(output, target, pos_weight=weight.cuda())


class WeightedFocalLoss(nn.Module):
    "Weighted version of Focal Loss"

    def __init__(self, alpha=0.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


def focal_loss(output, target, alpha=0.25, gamma=2):
    loss = WeightedFocalLoss(alpha, gamma)
    return loss.forward(output, target)
