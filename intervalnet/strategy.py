from typing import Any, Optional, Sequence, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from avalanche.training import BaseStrategy
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from torch import Tensor
from torch.optim import Optimizer


class IntervalTraining(BaseStrategy):
    def __init__(self, model: nn.Module, optimizer: Optimizer, train_mb_size: int = 1,
                 train_epochs: int = 1, eval_mb_size: int = 1,
                 device: torch.device = torch.device('cpu'),
                 plugins: Optional[Sequence[StrategyPlugin]] = None,
                 evaluator: Optional[EvaluationPlugin] = None, eval_every: int = -1):

        self.mb_output_all: Tensor
        """ All model's outputs computed on the current mini-batch (lower, middle, upper bounds). """

        self.vanilla_loss: Optional[Tensor] = None
        self.robust_loss: Optional[Tensor] = None
        self.radius_loss: Optional[Tensor] = None

        criterion = nn.CrossEntropyLoss()

        super().__init__(model, optimizer, criterion=criterion, train_mb_size=train_mb_size,  # type: ignore
                         train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device,
                         plugins=plugins, evaluator=evaluator, eval_every=eval_every)

    def after_forward(self, **kwargs: Any):
        self.mb_output_all = self.mb_output  # type: ignore
        self.mb_output = self.mb_output[:, 1, :].rename(None)  # type: ignore  # middle bound

        super().after_forward(**kwargs)  # type: ignore

    def after_eval_forward(self, **kwargs: Any):
        self.mb_output_all = self.mb_output  # type: ignore
        self.mb_output = self.mb_output[:, 1, :].rename(None)  # type: ignore  # middle bound

        super().after_eval_forward(**kwargs)  # type: ignore

    def before_backward(self, **kwargs: Any):
        super().before_backward(**kwargs)  # type: ignore

        # Save base loss for reporting
        self.vanilla_loss = self.loss.clone().detach()  # type: ignore

        # Maximize interval size up to radii of 1
        radius_penalties: list[Tensor] = []
        for name, param in self.model.named_parameters():
            if 'radius' in name:
                param = cast(Tensor, param)
                penalty: Tensor = (1 - param).clamp(min=0).mean()  # type: ignore
                radius_penalties.append(penalty)

        self.radius_loss = torch.stack(radius_penalties).mean()
        self.radius_loss *= 10.0

        # Maintain an acceptable worst-case loss
        self.robust_loss = self._criterion(self.robust_output(), self.mb_y)  # type: ignore

        self.loss += self.radius_loss  # type: ignore

    def robust_output(self):
        output_lower, _, output_higher = self.mb_output_all.unbind('bounds')
        y_oh = F.one_hot(self.mb_y)  # type: ignore
        return torch.where(y_oh.bool(), output_lower.rename(None), output_higher.rename(None))  # type: ignore
