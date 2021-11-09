from collections import deque
from dataclasses import InitVar, dataclass, field, fields
from typing import Any, Optional, Sequence, Union, cast

import numpy as np
import torch
import torch.linalg
import torch.nn as nn
import torch.nn.functional as F
import visdom
from avalanche.training import BaseStrategy
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from rich import print  # type: ignore
from torch import Tensor
from torch.optim import Optimizer
from torchmetrics.functional.classification.accuracy import accuracy

from intervalnet.models.interval import IntervalMLP, Mode


class IntervalTraining(BaseStrategy):
    """Main interval training strategy."""

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
        visdom_reset_every_epoch: bool = False,
        *,
        vanilla_loss_threshold: float,
        robust_loss_threshold: float,
        radius_multiplier: float,
        l1_lambda: float,
        metric_lookback: int,
    ):

        self.mb_output_all: dict[str, Tensor]
        """All model's outputs computed on the current mini-batch (lower, middle, upper bounds), per layer."""

        # Avalanche typing specifications
        self.mb_it: int
        self.mb_output: Union[dict[str, Tensor], Tensor]
        self.loss: Tensor
        self.training_exp_counter: int

        self.device: torch.device
        self.model: IntervalMLP

        self._criterion = nn.CrossEntropyLoss()

        # Config values
        assert vanilla_loss_threshold is not None
        assert robust_loss_threshold is not None
        assert radius_multiplier is not None
        assert l1_lambda is not None
        assert metric_lookback is not None

        self.vanilla_loss_threshold = torch.tensor(vanilla_loss_threshold)
        self.robust_loss_threshold = torch.tensor(robust_loss_threshold)
        self.radius_multiplier = torch.tensor(radius_multiplier)
        self.l1_lambda = torch.tensor(l1_lambda)
        self.metric_lookback = metric_lookback

        # Training metrics for the current mini-batch
        self.losses: Optional[IntervalTraining.Losses] = None  # Reported as 'Loss/*' metrics
        self.status: Optional[IntervalTraining.Status] = None  # Reported as 'Status/*' metrics

        # Running metrics
        self._accuracy: deque[Tensor] = deque(maxlen=metric_lookback)  # latest readings from the left
        self._robust_accuracy: deque[Tensor] = deque(maxlen=metric_lookback)  # latest readings from the left

        super().__init__(model, optimizer, criterion=self._criterion, train_mb_size=train_mb_size,  # type: ignore
                         train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device,
                         plugins=plugins, evaluator=evaluator, eval_every=eval_every)

        self.model.set_radius_multiplier(self.radius_multiplier)

        self.viz = visdom.Visdom() if enable_visdom else None
        self.viz_debug = visdom.Visdom(env='debug') if enable_visdom else None
        self.viz_reset_every_epoch = visdom_reset_every_epoch
        self.windows: dict[str, str] = {}

    @property
    def mb_y(self) -> Tensor:
        """Current mini-batch target."""
        return super().mb_y  # type: ignore

    @property
    def mode(self):
        """Current phase of model training.

        Returns
        -------
        Mode
            VANILLA, EXPANSION or CONTRACTION.

        """
        return self.model.mode

    @property
    def mode_numeric(self) -> Tensor:
        """Current phase of model training converted to a float.

        Returns
        -------
        Tensor
            0 - VANILLA
            1 - EXPANSION
            2 - CONTRACTION

        """
        return torch.tensor(self.mode.value).float()

    def accuracy(self, n_last: int = 1) -> Tensor:
        """Moving average of the batch accuracy."""
        assert n_last <= self.metric_lookback
        if not self._accuracy:
            return torch.tensor(0.0)
        return torch.stack(list(self._accuracy)[:n_last]).mean().detach().cpu()

    def robust_accuracy(self, n_last: int = 1) -> Tensor:
        """Moving average of the batch robust accuracy."""
        assert n_last <= self.metric_lookback
        if not self._robust_accuracy:
            return torch.tensor(0.0)
        return torch.stack(list(self._robust_accuracy)[:n_last]).mean().detach().cpu()

    # ----------------------------------------------------------------------------------------------
    # Training hooks
    # ----------------------------------------------------------------------------------------------
    def after_forward(self, **kwargs: Any):
        """Rebind the model's default output to the middle bound."""
        assert isinstance(self.mb_output, dict)
        self.mb_output_all = self.mb_output
        self.mb_output = self.mb_output['last'][:, 1, :].rename(None)  # type: ignore  # middle bound

        super().after_forward(**kwargs)  # type: ignore

    def after_eval_forward(self, **kwargs: Any):
        """Rebind the model's default output to the middle bound."""
        assert isinstance(self.mb_output, dict)
        self.mb_output_all = self.mb_output
        self.mb_output = self.mb_output['last'][:, 1, :].rename(None)  # type: ignore  # middle bound

        super().after_eval_forward(**kwargs)  # type: ignore

    @dataclass
    class Losses():
        """Model losses reported as 'Loss/*'."""
        device: InitVar[torch.device] = torch.device('cpu')

        total: Tensor = torch.tensor(0.0)
        vanilla: Tensor = torch.tensor(0.0)
        robust: Tensor = torch.tensor(0.0)

        robust_penalty: Tensor = torch.tensor(0.0)
        bounds_penalty: Tensor = torch.tensor(0.0)
        radius_penalty: Tensor = torch.tensor(0.0)

        def __post__init__(self, device: torch.device):
            for field in fields(self):
                if field.type == Tensor:
                    # Move to device & clone, because we use immutable defaults (no default_factory)
                    setattr(self, field.name, getattr(self, field.name).clone().to(device))

    def before_backward(self, **kwargs: Any):
        """Compute interval training losses."""
        super().before_backward(**kwargs)  # type: ignore

        self.losses = IntervalTraining.Losses(self.device)
        self.losses.vanilla = self.loss.clone().detach()
        self.losses.robust = cast(Tensor, self._criterion(self.robust_output(), self.mb_y))

        assert isinstance(self.mb_output, Tensor)
        self._accuracy.appendleft(accuracy(self.mb_output, self.mb_y))
        self._robust_accuracy.appendleft(accuracy(self.robust_output(), self.mb_y))

        if self.mode == Mode.VANILLA:
            self.losses.total = self.loss
        elif self.mode == Mode.EXPANSION:
            # === Robust penalty ===
            # Maintain an acceptable increase in worst-case loss
            # self.robust_penalty = self.robust_loss * F.relu(acc - robust_acc - 0.10) * 100 / 2

            robust_overflow = F.relu(self.losses.robust - self.robust_loss_threshold) / self.robust_loss_threshold
            # Force quasi-hard constraint
            self.losses.robust_penalty = ((robust_overflow + 1).pow(2) - 1).sqrt()

            # === Radius penalty ===
            # Maximize interval size up to radii of 1
            radii: list[Tensor] = []
            for module in self.model.interval_children():
                radii.append(module.radius.flatten())

            self.losses.radius_penalty = torch.stack([F.relu(torch.tensor(1.0) - r).pow(2).mean()  # mean per layer
                                                      for r in radii]).mean()

            # === Bounds penalty ===
            # bounds = [self.bounds_width(name).flatten() for name, _ in self.model.named_children()]
            # self.bounds_penalty = torch.cat(bounds).pow(2).mean().sqrt()
            self.losses.total = self.losses.robust_penalty + self.losses.radius_penalty
        elif self.mode == Mode.CONTRACTION:
            # ---------------------------------------------------------------------------------------------------------
            # Contraction phase
            # ---------------------------------------------------------------------------------------------------------
            # === Robust loss ===
            #
            pass

            # === Radius (contraction) penalty ===
            pass

            self.losses.total = self.loss

        # weights = torch.cat([m.weight.flatten() for m in self.model.interval_children()])
        # l1: Tensor = torch.linalg.vector_norm(weights, ord=1) / weights.shape[0]  # type: ignore
        # self.l1_penalty += self.l1_lambda * l1

        self.loss = self.losses.total  # Rebind as Avalanche loss
        self.diagnostics()

    def after_update(self, **kwargs: Any):
        """Cleanup after each step."""
        super().after_update(**kwargs)  # type: ignore

        self.model.clamp_radii()

    def before_training_exp(self, **kwargs: Any):
        """Switch mode or freeze on each consecutive experience."""
        super().before_training_exp(**kwargs)  # type: ignore

        if self.training_exp_counter == 1:
            self.model.switch_mode(Mode.CONTRACTION)
        elif self.training_exp_counter > 1:
            self.model.freeze_task()

    def before_training_epoch(self, **kwargs: Any):
        """Switch to expansion phase when ready."""
        super().before_training_epoch(**kwargs)  # type: ignore

        if self.mode == Mode.VANILLA and self.losses and self.losses.vanilla < self.vanilla_loss_threshold:
            self.model.switch_mode(Mode.EXPANSION)

        if self.viz_debug:
            self.reset_viz_debug()

    # ----------------------------------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------------------------------
    def robust_output(self):
        """Get the robust version of the current output.

        Returns
        -------
        Tensor
            Robust output logits (lower bound for correct class, upper bounds for incorrect classes).

        """
        output_lower, _, output_higher = self.mb_output_all['last'].unbind('bounds')
        y_oh = F.one_hot(self.mb_y)  # type: ignore
        return torch.where(y_oh.bool(), output_lower.rename(None), output_higher.rename(None))  # type: ignore

    def bounds_width(self, layer_name: str):
        """Compute the width of the activation bounds.

        Parameters
        ----------
        layer_name : str
            Name of the layer.

        Returns
        -------
        Tensor
            Difference between the upper and lower bounds of activations for a given layer.

        """
        bounds: Tensor = self.mb_output_all[layer_name].rename(None)  # type: ignore
        return bounds[:, 2, :] - bounds[:, 0, :]

    @dataclass
    class Status():
        """Diagnostic values reported as 'Status/*'."""
        mode: Tensor = torch.tensor(0.0)
        radius_multiplier: Tensor = torch.tensor(0.0)
        radius_mean: Tensor = torch.tensor(0.0)

        radius_mean_: dict[str, Tensor] = field(default_factory=lambda: {})
        bounds_width_: dict[str, Tensor] = field(default_factory=lambda: {})

    def diagnostics(self):
        """Save training diagnostics before each update."""
        self.status = IntervalTraining.Status()
        self.status.mode = self.mode_numeric
        self.status.radius_multiplier = self.radius_multiplier

        radii: list[Tensor] = []

        for name, module in self.model.named_interval_children():
            radii.append(module.radius.detach().cpu().flatten())
            self.status.radius_mean_[name] = radii[-1].mean()
            self.status.bounds_width_[name] = self.bounds_width(name).mean()

            if self.viz and self.mb_it == len(self.dataloader) - 1:  # type: ignore
                self.windows[f'{name}.radius'] = self.viz.heatmap(
                    module.radius,
                    win=self.windows.get(f'{name}.radius'),
                    opts={'title': f'{name}.radius --> epoch {(self.epoch or 0) + 1}'}
                )

                self.windows[f'{name}.weight'] = self.viz.heatmap(
                    module.weight.abs().clamp(max=module.weight.abs().quantile(0.99)),
                    win=self.windows.get(f'{name}.weight'),
                    opts={'title': f'{name}.weight.abs() (w/o outliers) --> epoch {(self.epoch or 0) + 1}'}
                )

        self.status.radius_mean = torch.cat(radii).mean()

        if self.viz_debug:
            for metric, name, window, _, color, dash, yrange in self.get_debug_metrics():
                self.append_viz_debug(metric, name, window, color, dash, yrange)

    def get_debug_metrics(self) -> list[tuple[
            Tensor, str, str, str, tuple[int, int, int], str, tuple[float, float]
    ]]:
        """Return a list of batch debug metrics to visualize with Visdom plots.

        Returns
        -------
        list[tuple[ Tensor, str, str, str, tuple[int, int, int], str, tuple[float, float] ]]
            List of (metric, metric_name, window_name, window_title, linecolor) tuples.

        """

        epoch = f'(epoch: {(self.epoch or 0) + 1})'
        _ = torch.tensor(0.0)

        return [
            (self.robust_accuracy(1), 'robust_accuracy',
                'accuracy', f'Batch accuracy {epoch}', (7, 126, 143), 'solid', (-0.1, 1.1)),
            (self.accuracy(1), 'accuracy',
                'accuracy', f'Batch accuracy {epoch}', (219, 0, 108), 'solid', (-0.1, 1.1)),
            (self.robust_accuracy(10), 'robust_accuracy_ma',
                'accuracy', f'Batch accuracy {epoch}', (7, 126, 143), 'dot', (-0.1, 1.1)),
            (self.accuracy(10), 'accuracy_ma',
                'accuracy', f'Batch accuracy {epoch}', (219, 0, 108), 'dot', (-0.1, 1.1)),

            (self.losses.robust_penalty if self.losses else _, 'robust_penalty',
                'penalties', f'Penalties {epoch}', (7, 126, 143), 'solid', (-0.1, 1.1)),
            (self.losses.radius_penalty if self.losses else _, 'radius_penalty',
                'penalties', f'Penalties {epoch}', (230, 203, 0), 'solid', (-0.1, 1.1)),
            (self.status.radius_mean if self.status else _, 'radius_mean',
                'penalties', f'Penalties {epoch}', (230, 203, 0), 'dot', (-0.1, 1.1)),

            (self.losses.total if self.losses else _, 'total_loss',
                'loss', f'Loss {epoch}', (219, 0, 108), 'solid', (-0.1, float(self.robust_loss_threshold) * 1.25)),
            (self.losses.robust if self.losses else _, 'robust_loss',
                'loss', f'Loss {epoch}', (7, 126, 143), 'solid', (-0.1, float(self.robust_loss_threshold) * 1.25)),
        ]

    def append_viz_debug(self, val: Tensor, name: str, window_name: str,
                         color: tuple[int, int, int], dash: str, yrange: tuple[float, float]):
        """Append single value to a Visdom line plot."""

        assert self.viz_debug

        if self.viz_reset_every_epoch:
            window_name = f'{window_name}_{(self.epoch or 0) + 1}'

        self.viz_debug.line(X=torch.tensor([self.mb_it]), Y=torch.tensor([val]),
                            win=self.windows[window_name], update='append', name=name,
                            opts={'linecolor': np.array([color]),  # type: ignore
                                  'dash': np.array([dash]),  # type: ignore
                                  'layoutopts': {'plotly': {
                                      'ytickmin': yrange[0], 'ytickmax': yrange[1],
                                  }}})

    def reset_viz_debug(self):
        """Recreate Visdom line plots before new epoch."""

        assert self.viz_debug

        for _, name, window_name, title, color, dash, yrange in self.get_debug_metrics():
            if self.viz_reset_every_epoch:
                window_name = f'{window_name}_{(self.epoch or 0) + 1}'

            # Reset plot line or create new plot
            self.windows[window_name] = self.viz_debug.line(
                X=torch.tensor([0]), Y=torch.tensor([0]), win=self.windows.get(window_name, None),
                opts={'title': title, 'linecolor': np.array([color]),  # type: ignore
                      'dash': np.array([dash]),  # type: ignore
                      'layoutopts': {'plotly': {
                          'margin': dict(l=40, r=40, b=80, t=80, pad=5),
                          'font': {'color': 'rgb(0, 0, 0)'},
                          'legend': {'orientation': 'h'},
                          'showlegend': True,
                          'yaxis': {
                              'autorange': False,
                              'range': yrange
                          }
                      }}}, name=name
            )
