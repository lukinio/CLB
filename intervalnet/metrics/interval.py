from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import wandb
import wandb.viz
from avalanche.evaluation.metric_definitions import GenericPluginMetric
from avalanche.evaluation.metrics.accuracy import AccuracyPluginMetric
from avalanche.evaluation.metrics.loss import LossPluginMetric
from avalanche.training.strategies.base_strategy import BaseStrategy
from intervalnet.strategy import IntervalTraining
from torch import Tensor

from .generic import MetricNamingMixin


class RobustAccuracy(MetricNamingMixin[float], AccuracyPluginMetric):
    def __init__(self):
        super().__init__(reset_at='experience', emit_at='experience', mode='eval')  # type: ignore

    def __str__(self):
        return "RobustAccuracy"

    def update(self, strategy: IntervalTraining):
        task_labels: list[Any] = strategy.experience.task_labels  # type: ignore
        if len(task_labels) > 1:
            task_label: Any = strategy.mb_task_id  # type: ignore
        else:
            task_label: Any = task_labels[0]

        with torch.no_grad():
            robust_output = strategy.robust_output()

        self._accuracy.update(robust_output, strategy.mb_y, task_label)  # type: ignore


class LayerDiagnostics(MetricNamingMixin[Tensor], GenericPluginMetric[Tensor]):
    def __init__(self, layer_name: str, start: float = -1, stop: float = 1, n_bins: int = 100,
                 reset_at: str = 'epoch', emit_at: str = 'epoch', mode: str = 'train'):
        self.layer_name = layer_name
        self.start = start
        self.stop = stop
        self.n_bins = n_bins

        assert self.n_bins <= 512, 'W&B does not support that many bins for visualization.'

        self.data: Optional[Tensor] = None
        super().__init__(self.data, reset_at=reset_at, emit_at=emit_at, mode=mode)  # type: ignore

    def update(self, strategy: BaseStrategy) -> None:
        values: Tensor = strategy.model.state_dict()[self.layer_name].detach().cpu()
        self.data = values

    def get_histogram(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if self.data is None:
            return None

        bins = np.linspace(self.start, self.stop, num=self.n_bins)  # type: ignore
        return np.histogram(self.data.view(-1).numpy(), bins=bins)

    def result(self, strategy: BaseStrategy) -> Optional[wandb.Histogram]:
        return wandb.Histogram(np_histogram=self.get_histogram())

    def reset(self, strategy: BaseStrategy) -> None:
        self.data = None

    def __str__(self):
        return f'Diagnostics/{self.layer_name}'


class LayerDiagnosticsHist(LayerDiagnostics):
    def __init__(self, layer_name: str, start: float = -1, stop: float = 1, n_bins: int = 10):
        super().__init__(layer_name, start=start, stop=stop, n_bins=n_bins)

    def __str__(self):
        return f'DiagnosticsHist/{self.layer_name}'

    def _get_metric_name(self, strategy: BaseStrategy, add_experience: bool = True, add_task: Any = True):
        return super()._get_metric_name(strategy, add_experience=True, add_task=add_task)

    def result(self, strategy: BaseStrategy) -> Optional[wandb.viz.CustomChart]:  # type: ignore
        hist = self.get_histogram()
        if hist is None:
            return None

        data: list[list[Any]] = []
        for i in range(len(hist[0])):
            data.append([hist[0][i], f'{i}: [{hist[1][i]:+.2f}, {hist[1][i+1]:+.2f}]'])

        table = wandb.Table(data=data, columns=['count', 'bin'])
        title = self._get_metric_name(strategy, add_experience=True, add_task=False)
        return wandb.plot.bar(table, 'bin', 'count', title=title)  # type: ignore


def radius_diagnostics(model: nn.Module):
    return [
        LayerDiagnostics(layer) for layer in model.state_dict().keys() if 'radius' in layer
    ] + [
        LayerDiagnosticsHist(layer) for layer in model.state_dict().keys() if 'radius' in layer
    ]


class LossReporter(MetricNamingMixin[Tensor], LossPluginMetric):
    def __init__(self, metric_name: str, strategy_attribute: str, reset_at: str = 'epoch',
                 emit_at: str = 'epoch', mode: str = 'train'):
        self.metric_name = metric_name
        self.strategy_attribute = strategy_attribute

        super().__init__(reset_at=reset_at, emit_at=emit_at, mode=mode)  # type: ignore

    def update(self, strategy: BaseStrategy) -> None:
        task_labels: list[Any] = strategy.experience.task_labels  # type: ignore
        if len(task_labels) > 1:
            task_label = 0
        else:
            task_label = task_labels[0]
        loss = getattr(strategy, self.strategy_attribute)
        self._loss.update(loss, patterns=len(strategy.mb_y), task_label=task_label)  # type: ignore

    def __str__(self):
        return f'{self.metric_name}'


def interval_losses():
    return [
        LossReporter('Loss/vanilla', 'vanilla_loss'),
        LossReporter('Loss/robust', 'robust_loss'),
        LossReporter('Loss/radius', 'radius_loss'),
    ]
