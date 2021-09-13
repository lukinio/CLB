from enum import Enum
from typing import Any, Optional, Sequence, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from avalanche.training import BaseStrategy
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from rich import print
from torch import Tensor
from torch.optim import Optimizer

from intervalnet.models.mlp import IntervalMLP


class Mode(Enum):
    VANILLA = 0
    INTERVALS = 1


class IntervalTraining(BaseStrategy):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = 1,
        device: torch.device = torch.device('cpu'),
        plugins: Optional[Sequence[StrategyPlugin]] = None,
        evaluator: Optional[EvaluationPlugin] = None,
        eval_every: int = -1,
        *,
        vanilla_loss_threshold: float,
        robust_loss_threshold: float,
        radius_multiplier: float,
    ):

        self.mb_output_all: dict[str, Tensor]
        """ All model's outputs computed on the current mini-batch (lower, middle, upper bounds) per layer. """

        self.device: torch.device
        self.model: IntervalMLP

        self.mode: Mode = Mode.VANILLA

        self.vanilla_loss: Optional[Tensor] = None
        self.robust_loss: Optional[Tensor] = None
        self.robust_penalty: Optional[Tensor] = None
        self.bounds_penalty: Optional[Tensor] = None
        self.radius_penalty: Optional[Tensor] = None
        self.radius_mean: Optional[Tensor] = None

        assert vanilla_loss_threshold is not None
        assert robust_loss_threshold is not None
        assert radius_multiplier is not None

        self.vanilla_loss_threshold = torch.tensor(vanilla_loss_threshold)
        self.robust_loss_threshold = torch.tensor(robust_loss_threshold)
        self.radius_multiplier = torch.tensor(radius_multiplier)

        criterion = nn.CrossEntropyLoss()

        super().__init__(model, optimizer, criterion=criterion, train_mb_size=train_mb_size,  # type: ignore
                         train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device,
                         plugins=plugins, evaluator=evaluator, eval_every=eval_every)

        self.model.set_radius_multiplier(self.radius_multiplier)

    @property
    def mode_num(self):
        return torch.tensor(self.mode.value).float()

    def after_forward(self, **kwargs: Any):
        self.mb_output_all = self.mb_output  # type: ignore
        self.mb_output = self.mb_output['last'][:, 1, :].rename(None)  # type: ignore  # middle bound

        super().after_forward(**kwargs)  # type: ignore

    def after_eval_forward(self, **kwargs: Any):
        self.mb_output_all = self.mb_output  # type: ignore
        self.mb_output = self.mb_output['last'][:, 1, :].rename(None)  # type: ignore  # middle bound

        super().after_eval_forward(**kwargs)  # type: ignore

    def before_backward(self, **kwargs: Any):
        super().before_backward(**kwargs)  # type: ignore

        self.loss = cast(Tensor, self.loss)  # type: ignore  # Fix Avalanche type-checking

        # Save base loss for reporting
        self.vanilla_loss = self.loss.clone().detach()  # type: ignore
        self.robust_loss = cast(Tensor, self._criterion(self.robust_output(), self.mb_y))  # type: ignore

        self.robust_penalty = torch.tensor(0.0, device=self.device)
        self.bounds_penalty = torch.tensor(0.0, device=self.device)
        self.radius_penalty = torch.tensor(0.0, device=self.device)
        self.radius_mean = torch.cat([
            self.model.radius_transform(param.flatten())
            for name, param in self.model.named_parameters() if 'radius' in name
        ]).mean()

        if self.mode == Mode.INTERVALS:
            # Maintain an acceptable increase in worst-case loss
            self.robust_penalty = F.relu(self.robust_loss - self.robust_loss_threshold) / self.robust_loss_threshold
            # Force quasi-hard constraint
            self.robust_penalty = (self.robust_penalty + 1).pow(2) - 1

            # Control activation bounds in hidden layers
            bounds = torch.cat([params for name, params in self.mb_output_all.items() if name != 'last'])
            bounds_width = bounds[:, 2, :] - bounds[:, 0, :]
            self.bounds_penalty = bounds_width.pow(2).mean() * 100.0

            # Maximize interval size up to radii of 1
            radii: list[Tensor] = []
            for name, param in self.model.named_parameters():
                if 'radius' in name:
                    radii.append(self.model.radius_transform(param.flatten()))

            self.radius_penalty = F.relu(torch.tensor(1.0) - torch.cat(radii)).sqrt().mean()

        self.loss += self.robust_penalty
        self.loss += self.bounds_penalty
        self.loss += self.radius_penalty

    def before_training_exp(self, **kwargs: Any):
        super().before_training_exp(**kwargs)  # type: ignore
        self.switch_mode(Mode.VANILLA)

    def before_training_epoch(self, **kwargs: Any):
        super().before_training_epoch(**kwargs)  # type: ignore

        torch.autograd.set_detect_anomaly(True)

        if self.mode == Mode.VANILLA and self.vanilla_loss is not None \
                and self.vanilla_loss < self.vanilla_loss_threshold:
            self.switch_mode(Mode.INTERVALS)

        # if self.mode == Mode.INTERVALS:
        #     for name, param in self.model.named_parameters():
        #         if 'radius' in name:
        #             with torch.no_grad():
        #                 param += 0.25 / self.radius_multiplier

    def after_training_epoch(self, **kwargs: Any):
        super().after_training_epoch(**kwargs)  # type: ignore

    def robust_output(self):
        output_lower, _, output_higher = self.mb_output_all['last'].unbind('bounds')
        y_oh = F.one_hot(self.mb_y)  # type: ignore
        return torch.where(y_oh.bool(), output_lower.rename(None), output_higher.rename(None))  # type: ignore

    def switch_mode(self, mode: Mode):
        if mode == Mode.VANILLA:
            self.mode = Mode.VANILLA
            print(f'\n[bold cyan]» :green_circle: Switching to vanilla training phase...')

            for name, param in self.model.named_parameters():
                if 'weight' in name:
                    param.requires_grad = True

        elif mode == Mode.INTERVALS:
            self.mode = Mode.INTERVALS
            print(f'\n[bold cyan]» :heavy_large_circle: Switching to interval expansion phase...')

            for name, param in self.model.named_parameters():
                if 'weight' in name:
                    param.requires_grad = False
