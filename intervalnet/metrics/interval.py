from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import wandb.viz
from avalanche.evaluation.metric_definitions import GenericPluginMetric
from avalanche.evaluation.metrics.accuracy import AccuracyPluginMetric
from avalanche.training.strategies.base_strategy import BaseStrategy
from intervalnet.strategy import IntervalTraining
from torch import Tensor

from .generic import MetricNamingMixin


class LayerDiagnostics(MetricNamingMixin[Tensor], GenericPluginMetric[Tensor]):
    def __init__(self, layer_name: str, start: float = -1, stop: float = 1, n_bins: int = 100,
                 reset_at: str = 'epoch', emit_at: str = 'epoch', mode: str = 'train'):
        self.layer_name = layer_name
        self.start = start
        self.stop = stop
        self.n_bins = n_bins

        assert self.n_bins <= 512, 'W&B does not support that many bins for visualization.'

        self.data: Optional[Tensor] = None
        super().__init__(self.data, reset_at=reset_at, emit_at=emit_at, mode=mode)

    def update(self, strategy: BaseStrategy) -> None:
        values: Tensor = strategy.model.state_dict()[self.layer_name].detach().cpu()
        self.data = values

    def get_histogram(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if self.data is None:
            return None

        bins = np.linspace(self.start, self.stop, num=self.n_bins)
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

    def result(self, strategy: BaseStrategy) -> Optional[wandb.viz.CustomChart]:
        hist = self.get_histogram()
        if hist is None:
            return None

        data = []
        for i in range(len(hist[0])):
            data.append([hist[0][i], f'{i}: [{hist[1][i]:+.2f}, {hist[1][i+1]:+.2f}]'])

        table = wandb.Table(data=data, columns=['count', 'bin'])
        title = self._get_metric_name(strategy, add_experience=True, add_task=False)
        return wandb.plot.bar(table, 'bin', 'count', title=title)


def radius_diagnostics(model: nn.Module):
    return [
        LayerDiagnostics(layer) for layer in model.state_dict().keys() if 'radius' in layer
    ] + [
        LayerDiagnosticsHist(layer) for layer in model.state_dict().keys() if 'radius' in layer
    ]


class RobustAccuracy(MetricNamingMixin[float], AccuracyPluginMetric):
    def __init__(self):
        super().__init__(reset_at='experience', emit_at='experience', mode='eval')  # type: ignore

    def __str__(self):
        return "RobustAccuracy"

    def update(self, strategy: IntervalTraining):
        task_labels = strategy.experience.task_labels
        if len(task_labels) > 1:
            task_labels = strategy.mb_task_id
        else:
            task_labels = task_labels[0]

        with torch.no_grad():
            output_lower, _, output_higher = strategy.mb_output_all.unbind('bounds')
            y_oh = F.one_hot(strategy.mb_y)
            output_robust = torch.where(y_oh.bool(), output_lower.rename(None), output_higher.rename(None))

        self._accuracy.update(output_robust, strategy.mb_y, task_labels)
