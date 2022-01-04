from collections import deque
from dataclasses import InitVar, dataclass, field, fields
from typing import Any, Optional, Sequence, cast

import numpy as np
import torch
import torch.linalg
import torch.nn as nn
import torch.nn.functional as F
import visdom
from avalanche.training import BaseStrategy
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from rich import print  # type: ignore # noqa
from torch import Tensor
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import Optimizer
from torchmetrics.functional.classification.accuracy import accuracy

from intervalnet.cfg import Settings
from intervalnet.models.interval import IntervalMLP, Mode


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

    def criterion(self):
        if self.is_training:
            # Use class masking for incremental class training in the same way as Continual Learning Benchmark
            preds = self.mb_output[:, : self.valid_classes]
        else:
            preds = self.mb_output

        return self._criterion(preds, self.mb_y)


class IntervalTraining(VanillaTraining):
    """Main interval training strategy."""

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
        enable_visdom: bool = False,
        visdom_reset_every_epoch: bool = False,
        *,
        cfg: Settings,
    ):
        super().__init__(  # type: ignore
            model,
            optimizer,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            cfg=cfg,
        )

        self.mb_output_all: dict[str, Tensor]
        """All model's outputs computed on the current mini-batch (lower, middle, upper bounds), per layer."""

        self.model: IntervalMLP

        self._current_lambda = self.cfg.interval.robust_lambda

        # Training metrics for the current mini-batch
        self.losses: Optional[IntervalTraining.Losses] = None  # Reported as 'Loss/*' metrics
        self.status: Optional[IntervalTraining.Status] = None  # Reported as 'Status/*' metrics

        # Running metrics
        self._accuracy: deque[Tensor] = deque(
            maxlen=self.cfg.interval.metric_lookback
        )  # latest readings from the left
        self._robust_accuracy: deque[Tensor] = deque(
            maxlen=self.cfg.interval.metric_lookback
        )  # latest readings from the left

        self.model.radius_multiplier = self.cfg.interval.radius_multiplier
        self.model.max_radius = self.cfg.interval.max_radius

        self.viz = visdom.Visdom() if enable_visdom else None
        self.viz_debug = visdom.Visdom(env="debug") if enable_visdom else None
        self.viz_reset_every_epoch = visdom_reset_every_epoch
        self.windows: dict[str, str] = {}

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
        assert n_last <= self.cfg.interval.metric_lookback
        if not self._accuracy:
            return torch.tensor(0.0)
        return torch.stack(list(self._accuracy)[:n_last]).mean().detach().cpu()

    def robust_accuracy(self, n_last: int = 1) -> Tensor:
        """Moving average of the batch robust accuracy."""
        assert n_last <= self.cfg.interval.metric_lookback
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
        self.mb_output = self.mb_output["last"][:, 1, :].rename(None)  # type: ignore  # middle bound

        super().after_forward(**kwargs)  # type: ignore

    def after_eval_forward(self, **kwargs: Any):
        """Rebind the model's default output to the middle bound."""
        assert isinstance(self.mb_output, dict)
        self.mb_output_all = self.mb_output
        self.mb_output = self.mb_output["last"][:, 1, :].rename(None)  # type: ignore  # middle bound

        super().after_eval_forward(**kwargs)  # type: ignore

    @dataclass
    class Losses:
        """Model losses reported as 'Loss/*'."""

        device: InitVar[torch.device] = torch.device("cpu")

        total: Tensor = torch.tensor(0.0)
        vanilla: Tensor = torch.tensor(0.0)
        robust: Tensor = torch.tensor(0.0)

        radius_penalty: Tensor = torch.tensor(0.0)
        robust_penalty: Tensor = torch.tensor(0.0)
        bounds_penalty: Tensor = torch.tensor(0.0)

        def __post_init__(self, device: torch.device):
            for field in fields(self):  # noqa
                if field.type == Tensor:
                    # Move to device & clone, because we use immutable defaults (no default_factory)
                    setattr(self, field.name, getattr(self, field.name).clone().to(device))

    # def criterion(self, ):
    #     if self.is_training:
    #         # Use class masking for incremental class training in the same way as Continual Learning Benchmark
    #         preds = self.mb_output[:, : self.valid_classes]
    #     else:
    #         preds = self.mb_output

    #     return self._criterion(preds, self.mb_y)

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
            # === Radius penalty ===
            # Maximize interval size up to `max_radius`
            radii: list[Tensor] = []
            for module in self.model.interval_children():
                radii.append(module.radius.flatten())

            self.losses.radius_penalty = torch.stack(
                [
                    F.relu(torch.tensor(1.0) - r / self.cfg.interval.max_radius)
                    .pow(self.cfg.interval.radius_exponent)
                    .mean()
                    for r in radii
                ]  # mean per layer
            ).mean()

            # === Robust penalty ===
            # Maintain an acceptable increase in worst-case loss
            if self.robust_accuracy(self.cfg.interval.metric_lookback) < self.cfg.interval.robust_accuracy_threshold:
                self.losses.robust_penalty = self.losses.robust * self._current_lambda

            #     if self._lambda is None:
            #         self._lambda = start_lambda
            #     self.losses.robust_penalty = self.losses.robust * self._lambda
            #     self._lambda *= 1.1
            # else:
            #     self._lambda = start_lambda

            # === Bounds penalty ===
            # bounds = [self.bounds_width(name).flatten() for name, _ in self.model.named_children()]
            # self.bounds_penalty = torch.cat(bounds).pow(2).mean().sqrt()
            self.losses.total = self.losses.radius_penalty + self.losses.robust_penalty
        elif self.mode == Mode.CONTRACTION:
            # ---------------------------------------------------------------------------------------------------------
            # Contraction phase
            # ---------------------------------------------------------------------------------------------------------
            # === Robust penalty ===
            if self.robust_accuracy(self.cfg.interval.metric_lookback) < self.cfg.interval.robust_accuracy_threshold:
                self.losses.robust_penalty = self.losses.robust * self._current_lambda

            # === Radius (contraction) penalty ===
            pass

            self.losses.total = self.loss + self.losses.robust_penalty

        # weights = torch.cat([m.weight.flatten() for m in self.model.interval_children()])
        # l1: Tensor = torch.linalg.vector_norm(weights, ord=1) / weights.shape[0]  # type: ignore
        # self.l1_penalty += self.l1_lambda * l1

        self.loss = self.losses.total  # Rebind as Avalanche loss
        self.diagnostics()

    def after_backward(self, **kwargs: Any):
        super().after_backward(**kwargs)  # type: ignore

        pass  # Debugging breakpoint

    def after_update(self, **kwargs: Any):
        """Cleanup after each step."""
        super().after_update(**kwargs)  # type: ignore

        self.model.clamp_radii()

    def before_training_exp(self, **kwargs: Any):
        """Switch mode or freeze on each consecutive experience."""
        super().before_training_exp(**kwargs)  # type: ignore

        if self.training_exp_counter == 1:
            self.model.switch_mode(Mode.CONTRACTION)
            self.make_optimizer()
        elif self.training_exp_counter > 1:
            self.model.freeze_task()
            self.make_optimizer()

        self._accuracy.clear()
        self._robust_accuracy.clear()

    def before_training_epoch(self, **kwargs: Any):
        """Switch to expansion phase when ready."""
        super().before_training_epoch(**kwargs)  # type: ignore

        if (
            self.mode == Mode.VANILLA
            and self.losses
            and self.losses.vanilla < self.cfg.interval.vanilla_loss_threshold
        ):
            self.model.switch_mode(Mode.EXPANSION)
            self.make_optimizer()
            self.optimizer.param_groups[0]["lr"] = self.cfg.interval.expansion_learning_rate  # type: ignore

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
        output_lower, _, output_higher = self.mb_output_all["last"].unbind("bounds")
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
    class Status:
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
        self.status.radius_multiplier = torch.tensor(self.cfg.interval.radius_multiplier)

        radii: list[Tensor] = []

        for name, module in self.model.named_interval_children():
            radii.append(module.radius.detach().cpu().flatten())
            self.status.radius_mean_[name] = radii[-1].mean()
            self.status.bounds_width_[name] = self.bounds_width(name).mean()

            if self.viz and self.mb_it == len(self.dataloader) - 1:  # type: ignore
                self.windows[f"{name}.radius"] = self.viz.heatmap(
                    module.radius,
                    win=self.windows.get(f"{name}.radius"),
                    opts={"title": f"{name}.radius --> epoch {(self.epoch or 0) + 1}"},
                )

                self.windows[f"{name}.weight"] = self.viz.heatmap(
                    module.weight.abs().clamp(max=module.weight.abs().quantile(0.99)),
                    win=self.windows.get(f"{name}.weight"),
                    opts={"title": f"{name}.weight.abs() (w/o outliers) --> epoch {(self.epoch or 0) + 1}"},
                )

        self.status.radius_mean = torch.cat(radii).mean()

        if self.viz_debug:
            for (
                metric,
                name,
                window,
                _,
                color,
                dash,
                yrange,
            ) in self.get_debug_metrics():
                self.append_viz_debug(metric, name, window, color, dash, yrange)

    def get_debug_metrics(self):
        """Return a list of batch debug metrics to visualize with Visdom plots.

        Returns
        -------
        list[tuple[ Tensor, str, str, str, tuple[int, int, int], str, tuple[float, float] ]]
            List of (metric, metric_name, window_name, window_title, linecolor) tuples.

        """

        epoch = f"(epoch: {(self.epoch or 0) + 1})"
        _ = torch.tensor(0.0)

        output_type = list[tuple[Tensor, str, str, str, tuple[int, int, int], str, tuple[float, float]]]

        metrics = cast(
            output_type,
            [
                (
                    self.robust_accuracy(1),
                    "robust_accuracy",
                    "accuracy",
                    f"Batch accuracy {epoch}",
                    (7, 126, 143),
                    "solid",
                    (-0.1, 1.1),
                ),
                (
                    self.accuracy(1),
                    "accuracy",
                    "accuracy",
                    f"Batch accuracy {epoch}",
                    (219, 0, 108),
                    "solid",
                    (-0.1, 1.1),
                ),
                (
                    self.robust_accuracy(self.cfg.interval.metric_lookback),
                    f"robust_accuracy_ma{self.cfg.interval.metric_lookback}",
                    "accuracy",
                    f"Batch accuracy {epoch}",
                    (7, 126, 143),
                    "dot",
                    (-0.1, 1.1),
                ),
                (
                    self.accuracy(self.cfg.interval.metric_lookback),
                    f"accuracy_ma{self.cfg.interval.metric_lookback}",
                    "accuracy",
                    f"Batch accuracy {epoch}",
                    (219, 0, 108),
                    "dot",
                    (-0.1, 1.1),
                ),
                (
                    self.losses.robust_penalty if self.losses else _,
                    "robust_penalty",
                    "penalties",
                    f"Penalties {epoch}",
                    (7, 126, 143),
                    "solid",
                    (-0.1, self.cfg.interval.max_radius + 0.1),
                ),
                (
                    self.losses.radius_penalty if self.losses else _,
                    "radius_penalty",
                    "penalties",
                    f"Penalties {epoch}",
                    (230, 203, 0),
                    "solid",
                    (-0.1, self.cfg.interval.max_radius + 0.1),
                ),
                (
                    self.status.radius_mean if self.status else _,
                    "radius_mean",
                    "penalties",
                    f"Penalties {epoch}",
                    (230, 203, 0),
                    "dot",
                    (-0.1, self.cfg.interval.max_radius + 0.1),
                ),
                (
                    self.losses.total if self.losses else _,
                    "total_loss",
                    "loss",
                    f"Loss {epoch}",
                    (219, 0, 108),
                    "solid",
                    (-0.1, 35.0),
                ),
                (
                    self.losses.robust if self.losses else _,
                    "robust_loss",
                    "loss",
                    f"Loss {epoch}",
                    (7, 126, 143),
                    "solid",
                    (-0.1, 35.0),
                ),
            ],
        )

        # for layer, __ in self.model.named_interval_children():  # type: ignore
        #     metrics.append(
        #         (self.status.radius_mean_[layer] if self.status else _, f'radius_mean_{layer}',
        #             'penalties', f'Penalties {epoch}', (203, 203, 203), 'dash', (-0.1, 1.1))
        #     )

        return metrics

    def append_viz_debug(
        self,
        val: Tensor,
        name: str,
        window_name: str,
        color: tuple[int, int, int],
        dash: str,
        yrange: tuple[float, float],
    ):
        """Append single value to a Visdom line plot."""

        assert self.viz_debug

        if self.viz_reset_every_epoch:
            window_name = f"{window_name}_{(self.epoch or 0) + 1}"

        self.viz_debug.line(
            X=torch.tensor([self.mb_it]),
            Y=torch.tensor([val]),
            win=self.windows[window_name],
            update="append",
            name=name,
            opts={
                "linecolor": np.array([color]),  # type: ignore
                "dash": np.array([dash]),  # type: ignore
                "layoutopts": {
                    "plotly": {
                        "ytickmin": yrange[0],
                        "ytickmax": yrange[1],
                    }
                },
            },
        )

    def reset_viz_debug(self):
        """Recreate Visdom line plots before new epoch."""

        assert self.viz_debug

        for (
            _,
            name,
            window_name,
            title,
            color,
            dash,
            yrange,
        ) in self.get_debug_metrics():
            if self.viz_reset_every_epoch:
                window_name = f"{window_name}_{(self.epoch or 0) + 1}"

            # Reset plot line or create new plot
            self.windows[window_name] = self.viz_debug.line(
                X=torch.tensor([0]),
                Y=torch.tensor([0]),
                win=self.windows.get(window_name, None),
                opts={
                    "title": title,
                    "linecolor": np.array([color]),  # type: ignore
                    "dash": np.array([dash]),  # type: ignore
                    "layoutopts": {
                        "plotly": {
                            "margin": dict(l=40, r=40, b=80, t=80, pad=5),
                            "font": {"color": "rgb(0, 0, 0)"},
                            "legend": {"orientation": "h"},
                            "showlegend": True,
                            "yaxis": {"autorange": False, "range": yrange},
                        }
                    },
                },
                name=name,
            )
