from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from avalanche.evaluation.metric_definitions import GenericPluginMetric
from avalanche.evaluation.metrics.accuracy import AccuracyPluginMetric
from avalanche.training.strategies.base_strategy import BaseStrategy
from intervalnet.strategy import IntervalTraining
from torch import Tensor

from .generic import MetricNamingMixin


class LayerDiagnostics(MetricNamingMixin[Tensor], GenericPluginMetric[Tensor]):
    def __init__(self, layer_name: str):
        self.layer_name = layer_name
        self.data: Optional[Tensor] = None
        super().__init__(self.data, reset_at='epoch', emit_at='epoch', mode='train')

    def update(self, strategy: BaseStrategy) -> None:
        values: Tensor = strategy.model.state_dict()[self.layer_name].detach().cpu()
        self.data = values

    def result(self, strategy: BaseStrategy) -> Optional[Tensor]:
        return self.data

    def reset(self, strategy: BaseStrategy) -> None:
        self.data = None

    def __str__(self):
        return f'Diagnostics/{self.layer_name}'


def radius_diagnostics(model: nn.Module):
    return [
        LayerDiagnostics(layer) for layer in model.state_dict().keys() if 'radius' in layer
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
