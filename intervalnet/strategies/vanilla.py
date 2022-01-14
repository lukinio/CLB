from typing import Optional, Sequence

import torch
import torch.linalg
import torch.nn as nn
from avalanche.training import BaseStrategy
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from rich import print  # type: ignore # noqa
from torch import Tensor
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import Optimizer

from intervalnet.cfg import Settings


class VanillaTraining(BaseStrategy):
    """Benchmark CL training."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = 1,
        device: torch.device = torch.device("cpu"),
        plugins: Optional[Sequence[StrategyPlugin]] = None,
        evaluator: Optional[EvaluationPlugin] = None,
        eval_every: int = -1,
        *,
        cfg: Settings,
    ):
        super().__init__(  # type: ignore
            model,
            optimizer,
            criterion=nn.CrossEntropyLoss(),
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
        )

        # Avalanche typing specifications
        self.mb_it: int  # type: ignore
        self.mb_output: Tensor  # type: ignore
        self.loss: Tensor  # type: ignore
        self.training_exp_counter: int  # type: ignore
        self.optimizer: Optimizer
        self._criterion: CrossEntropyLoss
        self.device: torch.device

        # Config values
        self.cfg = cfg

        self.valid_classes = 0

    @property
    def mb_y(self) -> Tensor:
        """Current mini-batch target."""
        return super().mb_y  # type: ignore

    # def criterion(self):
    #     if self.is_training:
    #         # Use class masking for incremental class training in the same way as Continual Learning Benchmark
    #         preds = self.mb_output[:, : self.valid_classes]
    #     else:
    #         preds = self.mb_output

    #     return self._criterion(preds, self.mb_y)
