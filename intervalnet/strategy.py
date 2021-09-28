from typing import Any, Optional, Sequence, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import visdom
from avalanche.training import BaseStrategy
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from rich import print  # type: ignore
from torch import Tensor
from torch.optim import Optimizer

from intervalnet.models.interval import IntervalMLP, Mode


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
        enable_visdom: bool = False,
        *,
        vanilla_loss_threshold: float,
        robust_loss_threshold: float,
        radius_multiplier: float,
    ):

        self.mb_output_all: dict[str, Tensor]
        """ All model's outputs computed on the current mini-batch (lower, middle, upper bounds) per layer. """

        self.mb_it: int
        self.training_exp_counter: int

        self.device: torch.device
        self.model: IntervalMLP

        self.vanilla_loss: Optional[Tensor] = None
        self.robust_loss: Optional[Tensor] = None
        self.robust_penalty: Optional[Tensor] = None
        self.bounds_penalty: Optional[Tensor] = None
        self.radius_penalty: Optional[Tensor] = None
        self.radius_mean: Optional[Tensor] = None

        self.radius_mean_per_layer: dict[str, Optional[Tensor]] = {}
        self.bounds_width_per_layer: dict[str, Optional[Tensor]] = {}

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

        self.viz = visdom.Visdom() if enable_visdom else None
        self.windows: dict[str, str] = {}

    @property
    def mode(self):
        return self.model.mode

    @property
    def mode_numeric(self) -> Tensor:
        return torch.tensor(self.mode.value).float()

    def after_forward(self, **kwargs: Any):
        self.mb_output_all = self.mb_output  # type: ignore
        self.mb_output = self.mb_output['last'][:, 1, :].rename(None)  # type: ignore  # middle bound

        super().after_forward(**kwargs)  # type: ignore

    def after_eval_forward(self, **kwargs: Any):
        self.mb_output_all = self.mb_output  # type: ignore
        self.mb_output = self.mb_output['last'][:, 1, :].rename(None)  # type: ignore  # middle bound

        super().after_eval_forward(**kwargs)  # type: ignore

    def bounds_width(self, layer_name: str):
        bounds: Tensor = self.mb_output_all[layer_name].rename(None)  # type: ignore
        return bounds[:, 2, :] - bounds[:, 0, :]

    def before_backward(self, **kwargs: Any):
        super().before_backward(**kwargs)  # type: ignore

        # Save base loss for reporting
        self.loss = cast(Tensor, self.loss)  # type: ignore  # Fix Avalanche type-checking
        self.vanilla_loss = self.loss.clone().detach()  # type: ignore

        self.robust_loss = cast(Tensor, self._criterion(self.robust_output(), self.mb_y))  # type: ignore

        # Additional penalties
        self.robust_penalty = torch.tensor(0.0, device=self.device)
        self.radius_penalty = torch.tensor(0.0, device=self.device)
        self.bounds_penalty = torch.tensor(0.0, device=self.device)

        if self.mode == Mode.EXPANSION:
            # ---------------------------------------------------------------------------------------------------------
            # Expansion phase
            # ---------------------------------------------------------------------------------------------------------
            # === Robust loss ===
            # Maintain an acceptable increase in worst-case loss
            robust_overflow = F.relu(self.robust_loss - self.robust_loss_threshold) / self.robust_loss_threshold
            # Force quasi-hard constraint
            self.robust_penalty = ((robust_overflow + 1).pow(2) - 1).sqrt()

            # === Radius penalty ===
            # Maximize interval size up to radii of 1
            radii: list[Tensor] = []
            for name, module in self.model.named_interval_children():
                radii.append(module.radius.flatten())

            # INFO: sqrt -> more sparse, push to saturate
            # INFO: pow(2) -> more regular, push to smooth distribution
            # :: Flat version:
            # self.radius_penalty = F.relu(torch.tensor(1.0) - torch.cat(radii)).pow(2).mean()
            # :: Per layer version:
            self.radius_penalty = torch.stack([F.relu(torch.tensor(1.0) - r).pow(2).mean() for r in radii]).mean()

            # === Bounds penalty ===
            bounds = [self.bounds_width(name).flatten() for name, _ in self.model.named_children()]
            self.bounds_penalty = torch.cat(bounds).pow(2).mean().sqrt()

        elif self.mode == Mode.CONTRACTION:
            # ---------------------------------------------------------------------------------------------------------
            # Contraction phase
            # ---------------------------------------------------------------------------------------------------------
            # === Robust loss ===
            #
            pass

            # === Radius (contraction) penalty ===
            pass

        self.loss += self.robust_penalty
        self.loss += self.radius_penalty
        # self.loss += self.bounds_penalty

        # ---------------------------------------------------------------------------------------------------------
        # Diagnostics
        # ---------------------------------------------------------------------------------------------------------
        radii = []

        for name, module in self.model.named_interval_children():
            radii.append(module.radius.detach().cpu().flatten())
            self.radius_mean_per_layer[name] = radii[-1].mean()
            self.bounds_width_per_layer[name] = self.bounds_width(name).mean()

            if self.viz and self.mb_it == len(self.dataloader) - 1:  # type: ignore
                win = self.windows.get(name)
                title = f'{name}.radius --> epoch {(self.epoch or 0) + 1}'
                self.windows[name] = self.viz.heatmap(module.radius, win=win, opts={'title': title})

        self.radius_mean = torch.cat(radii).mean()

    def after_update(self, **kwargs: Any):
        super().after_update(**kwargs)  # type: ignore

        self.model.clamp_radii()

    def before_training_exp(self, **kwargs: Any):
        super().before_training_exp(**kwargs)  # type: ignore

        if self.training_exp_counter == 1:
            self.model.switch_mode(Mode.CONTRACTION)

    def before_training_epoch(self, **kwargs: Any):
        super().before_training_epoch(**kwargs)  # type: ignore

        if self.mode == Mode.VANILLA and self.vanilla_loss is not None \
                and self.vanilla_loss < self.vanilla_loss_threshold:
            self.model.switch_mode(Mode.EXPANSION)

    def robust_output(self):
        output_lower, _, output_higher = self.mb_output_all['last'].unbind('bounds')
        y_oh = F.one_hot(self.mb_y)  # type: ignore
        return torch.where(y_oh.bool(), output_lower.rename(None), output_higher.rename(None))  # type: ignore
