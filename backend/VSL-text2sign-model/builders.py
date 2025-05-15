"""
Collection of builder functions for training components.
"""

from typing import Callable, Optional, Dict, Generator, Tuple, Union, Literal
import torch
from torch import nn
from torch.optim import Optimizer, Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, ExponentialLR

from helpers import ConfigurationError


def build_gradient_clipper(config: dict) -> Optional[Callable[[Generator], None]]:
    """
    Build a gradient clipping function based on configuration.

    Supported options:
        - clip_grad_val: Clip gradients by value (torch.nn.utils.clip_grad_value_).
        - clip_grad_norm: Clip gradients by norm (torch.nn.utils.clip_grad_norm_).

    Args:
        config: Dictionary with training configurations, including:
            - clip_grad_val (float, optional): Value to clip gradients.
            - clip_grad_norm (float, optional): Max norm to clip gradients.
            - clip_norm_foreach (bool, optional): Use foreach implementation for norm clipping (default: True).

    Returns:
        Callable or None: Gradient clipping function (in-place) or None if no clipping.

    Raises:
        ConfigurationError: If both clip_grad_val and clip_grad_norm are specified.
    """
    if "clip_grad_val" in config and "clip_grad_norm" in config:
        raise ConfigurationError("Specify either clip_grad_val or clip_grad_norm, not both.")

    clip_grad_fun = None
    if "clip_grad_val" in config:
        clip_value = config["clip_grad_val"]
        clip_grad_fun = lambda params: nn.utils.clip_grad_value_(parameters=params, clip_value=clip_value)
    elif "clip_grad_norm" in config:
        max_norm = config["clip_grad_norm"]
        foreach = config.get("clip_norm_foreach", True)  # Use foreach for performance
        clip_grad_fun = lambda params: nn.utils.clip_grad_norm_(parameters=params, max_norm=max_norm, foreach=foreach)

    return clip_grad_fun


def build_optimizer(config: dict, parameters: Generator) -> Optimizer:
    """
    Build an optimizer (Adam or AdamW) for the given parameters.

    Supported optimizers:
        - adam: Standard Adam optimizer with optional amsgrad and fused implementation.
        - adamw: AdamW optimizer for better weight decay handling.

    Args:
        config: Configuration dictionary with keys:
            - optimizer (str): Optimizer type ('adam' or 'adamw', default: 'adam').
            - learning_rate (float, optional): Initial learning rate (default: 1e-3).
            - weight_decay (float, optional): Weight decay (default: 1e-5).
            - adam_betas (tuple, optional): Betas for Adam/AdamW (default: (0.9, 0.999)).
            - amsgrad (bool, optional): Use AMSGrad variant (default: False).
            - fused (bool, optional): Use fused implementation if available (default: True).

    Returns:
        Optimizer: Configured Adam or AdamW optimizer.

    Raises:
        ConfigurationError: If optimizer type is not 'adam' or 'adamw'.
    """
    optimizer_name = config.get("optimizer", "adam").lower()
    learning_rate = config.get("learning_rate", 1e-3)
    weight_decay = config.get("weight_decay", 1e-5)
    adam_betas = config.get("adam_betas", (0.9, 0.999))
    amsgrad = config.get("amsgrad", False)
    fused = config.get("fused", True)  # Enable fused implementation if supported

    if optimizer_name == "adam":
        optimizer = Adam(
            parameters,
            lr=learning_rate,
            betas=adam_betas,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            fused=fused if hasattr(torch.optim.Adam, "fused") else False
        )
    elif optimizer_name == "adamw":
        optimizer = AdamW(
            parameters,
            lr=learning_rate,
            betas=adam_betas,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            fused=fused if hasattr(torch.optim.AdamW, "fused") else False
        )
    else:
        raise ConfigurationError("Optimizer must be 'adam' or 'adamw'.")

    return optimizer


def build_scheduler(
    config: Dict, optimizer: Optimizer, scheduler_mode: Literal["min", "max"], hidden_size: int = 0
) -> Tuple[Optional[Union["NoamScheduler", ReduceLROnPlateau, StepLR, ExponentialLR]], Optional[str]]:
    """
    Build a learning rate scheduler based on configuration.

    Supported schedulers:
        - plateau: ReduceLROnPlateau based on validation score.
        - decaying: StepLR for periodic decay.
        - exponential: ExponentialLR for exponential decay.
        - noam: NoamScheduler for Transformer-style scheduling.

    Args:
        config: Configuration dictionary with keys:
            - scheduling (str, optional): Scheduler type.
            - decrease_factor (float, optional): Decay factor (default: 0.1 for plateau, 0.99 for exponential).
            - patience (int, optional): Patience for plateau (default: 10).
            - decaying_step_size (int, optional): Step size for decaying (default: 1).
            - learning_rate_factor (float, optional): Factor for Noam (default: 1).
            - learning_rate_warmup (int, optional): Warmup steps for Noam (default: 4000).
        optimizer: Optimizer to schedule.
        scheduler_mode: 'min' or 'max' for plateau scheduler.
        hidden_size: Encoder hidden size for Noam scheduler.

    Returns:
        Tuple[Optional[Scheduler], Optional[str]]: Scheduler and step timing ('validation', 'epoch', 'step').

    Raises:
        ConfigurationError: If scheduler type is invalid.
    """
    scheduler, scheduler_good_at = None, None
    if not config.get("scheduling"):
        return None, None

    scheduler_type = config.get("scheduling").lower()
    if scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode=scheduler_mode,
            factor=config.get("decrease_factor", 0.1),
            patience=config.get("patience", 10),
            threshold=1e-8,
            threshold_mode="abs"
        )
        scheduler_step_at = "validation"

    elif scheduler_type == "decaying":
        scheduler = StepLR(
            optimizer=optimizer,
            step_size=config.get("decaying_step_size", 1),
            gamma=config.get("decrease_factor", 0.1)
        )
        scheduler_step_at = "epoch"

    elif scheduler_type == "exponential":
        scheduler = ExponentialLR(
            optimizer=optimizer,
            gamma=config.get("decrease_factor", 0.99)
        )
        scheduler_step_at = "epoch"

    elif scheduler_type == "noam":
        scheduler = NoamScheduler(
            hidden_size=hidden_size,
            optimizer=optimizer,
            factor=config.get("learning_rate_factor", 1),
            warmup=config.get("learning_rate_warmup", 4000)
        )
        scheduler_step_at = "step"

    else:
        raise ConfigurationError(
            "Invalid scheduler. Valid options: 'plateau', 'decaying', 'exponential', 'noam'."
        )

    return scheduler, scheduler_step_at


class NoamScheduler:
    """
    Noam learning rate scheduler from 'Attention is All You Need'.
    Implements warmup followed by inverse square-root decay.

    Args:
        hidden_size (int): Model hidden size (e.g., Transformer embedding size).
        optimizer (Optimizer): Optimizer to schedule.
        factor (float, optional): Scaling factor (default: 1).
        warmup (int, optional): Number of warmup steps (default: 4000).
        init_lr (float, optional): Initial learning rate (default: 1.0).
    """

    def __init__(
        self,
        hidden_size: int,
        optimizer: Optimizer,
        factor: float = 1,
        warmup: int = 4000,
        init_lr: float = 1.0
    ):
        self.optimizer = optimizer
        self.hidden_size = hidden_size
        self.factor = factor
        self.warmup = warmup
        self.init_lr = init_lr
        self._step_count = 0
        self._update_lr()  # Initialize learning rates

    def step(self):
        """Update the learning rate for the next step."""
        self._step_count += 1
        self._update_lr()

    def _update_lr(self):
        """Compute and apply learning rates for each parameter group."""
        step = max(self._step_count, 1)
        scale = self.factor * (self.hidden_size ** (-0.5) *
                              min(step ** (-0.5), step * self.warmup ** (-1.5)))
        for group in self.optimizer.param_groups:
            group["lr"] = scale * self.init_lr

    def state_dict(self) -> dict:
        """Return scheduler state."""
        return {
            "hidden_size": self.hidden_size,
            "factor": self.factor,
            "warmup": self.warmup,
            "step_count": self._step_count,
            "init_lr": self.init_lr
        }

    def load_state_dict(self, state_dict: dict):
        """Load scheduler state."""
        self.hidden_size = state_dict["hidden_size"]
        self.factor = state_dict["factor"]
        self.warmup = state_dict["warmup"]
        self._step_count = state_dict["step_count"]
        self.init_lr = state_dict.get("init_lr", 1.0)  # Backward compatibility
        self._update_lr()  # Apply loaded state

    def get_lr(self) -> list:
        """Return current learning rates for each parameter group."""
        step = max(self._step_count, 1)
        scale = self.factor * (self.hidden_size ** (-0.5) *
                              min(step ** (-0.5), step * self.warmup ** (-1.5)))
        return [scale * self.init_lr for _ in self.optimizer.param_groups]