"""
Module to implement training loss
"""

import torch 
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional, Union


class RegLoss(nn.Module):

    """
    Regression Loss supporting L1, MSE, and Smooth L1 losses.

    Args:
        cfg (Dict): Configuration dictionary with keys:
            - training.loss (str): Type of loss ('l1', 'mse', 'smooth_l1').
            - model.loss_scale (float, optional): Scaling factor for loss (default: 1.0).
        target_pad (float, optional): Padding value in targets to ignore in loss calculation (default: 0.0).
        reduction (str, optional): Reduction mode ('mean', 'sum', 'none') (default: 'mean').
    """

    def __init__(
        self,
        cfg: Dict,
        target_pad: float = 0.0,
        reduction: str = "mean"
    ):

        super(RegLoss, self).__init__()

        self.loss_type = cfg["training"]["loss"].lower()
        self.target_pad = target_pad
        self.loss_scale = cfg["model"].get("loss scale", 1.0)
        self.reduction = reduction

        # Initialize loss criterion based on loss_type
        if self.loss_type == "l1":
            self.criterion = nn.L1Loss(reduction=reduction)
        elif self.loss_type == "mse":
            self.criterion = nn.MSELoss(reduction=reduction)
        elif self.loss_type == "smooth_l1":
            self.criterion = nn.SmoothL1Loss(reduction=reduction, beta=1.0)
        else:
            print(f"Loss type '{self.loss_type}' not supported. Defaulting to L1 loss.")
            self.criterion = nn.L1Loss(reduction=reduction)

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:

        loss_mask = (targets != self.target_pad)

        # Find the masked predictions and targets using loss mask
        preds_masked = preds * loss_mask
        targets_masked = targets * loss_mask

        # Calculate loss just over the masked predictions
        loss = self.criterion(preds_masked, targets_masked)

        # Multiply loss by the loss scale
        if self.loss_scale != 1.0:
            loss = loss * self.loss_scale

        return loss