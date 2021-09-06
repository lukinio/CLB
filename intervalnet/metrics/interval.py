from typing import Optional

import torch.nn as nn
from avalanche.evaluation.metric_definitions import GenericPluginMetric
from avalanche.training.strategies.base_strategy import BaseStrategy
from torch import Tensor

from .generic import MetricNamingMixin


class RadiusDiagnostics(MetricNamingMixin[Tensor], GenericPluginMetric[Tensor]):
    def __init__(self, layer_name: str):
        self.layer_name = layer_name
        self.data: Optional[Tensor] = None
        super().__init__(self.data, reset_at='epoch', emit_at='epoch', mode='train')

    def update(self, strategy: BaseStrategy):
        values: Tensor = strategy.model.state_dict()[self.layer_name].detach().cpu()
        self.data = values

    def result(self, strategy: BaseStrategy) -> Optional[Tensor]:
        return self.data

    def reset(self, strategy: BaseStrategy) -> None:
        self.data = None

    def __str__(self):
        return f'Diagnostics/radius/{self.layer_name}'


def radius_diagnostics(model: nn.Module):
    return [
        RadiusDiagnostics(layer) for layer in model.state_dict().keys() if 'radius' in layer
    ]
